import argparse
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from one.api import ONE

from iblnm.config import (
    EXCLUDE_SUBJECTS, SESSIONS_FPATH, SESSIONS_QC_FPATH, SESSIONS_LOG_FPATH, MIN_NTRIALS
)
from iblnm.io import (
    get_subject_info, get_datasets, unpack_session_dict, get_target_regions,
    get_extended_qc
)
from iblnm.util import (
    protocol2type, df2pqt, clean_sessions, add_dataset_flags, get_session_length,
    resolve_session_status, drop_junk_duplicates
)


def ensure_list(x):
    """Convert value to list. Returns empty list for NaN, wraps scalars."""
    if isinstance(x, list):
        return x
    return [] if pd.isna(x) else [x]


# Parse command line arguments
parser = argparse.ArgumentParser(description='Query IBL fibrephotometry database')
parser.add_argument('--redownload', action='store_true',
                    help='Re-download all data, ignoring existing sessions file')
parser.add_argument('--extended-qc', action='store_true',
                    help='Fetch extended QC data (saved to separate file)')
args = parser.parse_args()

one = ONE()

# Initialize exceptions log
exlog = []

# Get list of session dicts
print("Querying database...")
sessions = one.alyx.rest('sessions', 'list', project='ibl_fibrephotometry')

# Rename id column to eid
df_sessions = pd.DataFrame(sessions).rename(columns={'id': 'eid'})
# Add useful label for session types
df_sessions['session_type'] = df_sessions['task_protocol'].map(protocol2type)


# Load existing sessions if not redownloading
df_existing = None
if not args.redownload and SESSIONS_FPATH.exists():
    print(f"Loading existing sessions from {SESSIONS_FPATH}...")
    df_existing = pd.read_parquet(SESSIONS_FPATH)
    existing_eids = set(df_existing['eid'].values)
    n_existing = len(df_existing)

    # Filter to only new sessions
    df_sessions = df_sessions[~df_sessions['eid'].isin(existing_eids)].copy()
    n_new = len(df_sessions)

    print(f"Found {n_existing} existing sessions, {n_new} new sessions to process")

    if n_new == 0:
        print("No new sessions to process. Exiting.")
        exit(0)
else:
    if args.redownload:
        print("Re-downloading all data...")
    else:
        print("No existing sessions file found, downloading all data...")

# Add genotype info and infer NM cell type
print("Adding subject info...")
df_sessions = df_sessions.progress_apply(
    get_subject_info, axis='columns', exlog=exlog
).copy()

print("Getting datasets...")
df_sessions = df_sessions.progress_apply(
    get_datasets, axis='columns', exlog=exlog
).copy()

print("Unpacking session dicts...")
df_sessions = df_sessions.progress_apply(
    unpack_session_dict, axis='columns', exlog=exlog
).copy()

# Get brain region targets
print("Getting photometry targets...")
df_sessions = df_sessions.progress_apply(
    get_target_regions, axis='columns', exlog=exlog
).copy()

# Normalize list columns
print("Normalizing list columns...")
for col in ['target', 'roi', 'datasets']:
    if col in df_sessions.columns:
        df_sessions[col] = df_sessions[col].apply(ensure_list)

# Merge with existing data if applicable
if df_existing is not None:
    print(f"Merging {len(df_sessions)} new sessions with {len(df_existing)} existing sessions...")
    df_sessions = pd.concat([df_existing, df_sessions], ignore_index=True)

# Fetch extended QC if requested
df_qc = None
if args.extended_qc:
    print("Fetching extended QC...")
    df_qc = df_sessions[['eid']].copy()
    df_qc = df_qc.progress_apply(
        get_extended_qc, axis='columns', exlog=exlog
    ).copy()

# Count exceptions by unique eid
n_total = len(df_sessions)
eids_with_exceptions = set(ex.get('eid') for ex in exlog if 'eid' in ex)
n_exceptions = len(eids_with_exceptions)

print(
    f"Finished processing sessions:"
    f"\n  {n_total} total sessions"
    f"\n  {n_exceptions} sessions with exceptions"
)


# Add convenience columns
df_sessions['date'] = pd.to_datetime(df_sessions['start_time'], format='ISO8601').dt.date
df_sessions['day_n'] = df_sessions.groupby('subject')['date'].transform(
    lambda x: [(date - x.min()).days for date in x]
)
df_sessions['session_n'] = df_sessions.groupby('subject')['date'].rank(method='dense')

# Get session length
df_sessions['session_length'] = df_sessions.apply(get_session_length, axis='columns')

# Add dataset flags
df_sessions = add_dataset_flags(df_sessions)
# Add convenience flags
# ~df_sessions['has_extracted_behavior'] = df_sessions['has_extracted_task'] & df_sessions['has_extracted_wheel']
df_sessions['has_extracted_photometry'] = df_sessions['has_extracted_photometry_signal'] & df_sessions['has_extracted_photometry_locations']

# Define completion criteria (requires both task & photometry)
df_sessions['data_complete'] = (
    df_sessions['has_extracted_task'] &
    df_sessions['has_extracted_photometry']
)
df_sessions['trials_complete'] = df_sessions['n_trials'] > MIN_NTRIALS

# Resolve session status (good/junk/conflict)
session_groups = df_sessions.groupby(['subject', 'date', 'session_type'], group_keys=False)
df_sessions['session_status'] = session_groups.apply(
    resolve_session_status,
    columns=['data_complete', 'trials_complete'],
    include_groups=False
)


# Save sessions
df2pqt(df_sessions, SESSIONS_FPATH)

# Save QC if fetched
if df_qc is not None:
    df2pqt(df_qc, SESSIONS_QC_FPATH)

# Save exceptions
if exlog:
    df_exceptions = pd.DataFrame(exlog)
    df2pqt(df_exceptions, SESSIONS_LOG_FPATH)
