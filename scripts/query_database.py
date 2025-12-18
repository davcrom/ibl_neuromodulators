import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
tqdm.pandas()

from one.api import ONE

from iblnm.config import *
from iblnm.io import get_subject_info, check_datasets, unpack_session_dict, get_target_regions
from iblnm.util import protocol2type, df2pqt

# Parse command line arguments
parser = argparse.ArgumentParser(description='Query IBL fibrephotometry database')
parser.add_argument('--redownload', action='store_true',
                    help='Re-download all data, ignoring existing sessions file')
parser.add_argument('--recheck-missing', action='store_true',
                    help='Re-check sessions that previously had missing datasets')
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

# Remove mice that shouldn't be part of the analysis
df_sessions = df_sessions.query('subject not in @EXCLUDE_SUBJECTS')

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

    if n_new == 0 and not args.recheck_missing:
        print("No new sessions to process. Exiting.")
        exit(0)
    elif n_new == 0:
        print("No new sessions to process, but will re-check existing sessions for missing datasets.")
else:
    if args.redownload:
        print("Re-downloading all data...")
    else:
        print("No existing sessions file found, downloading all data...")

# Add genotype info and tries to infer the GCaMP-expressing NM cell type
print("Adding subject info...")
df_sessions = df_sessions.progress_apply(
    get_subject_info, axis='columns', exlog=exlog
).copy()

print("Checking datasets...")
df_sessions = df_sessions.progress_apply(
    check_datasets, axis='columns', exlog=exlog
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

# Ensure target and roi columns have uniform dtype (always lists)
print("Normalizing target and roi columns...")
for col in ['target', 'roi']:
    if col in df_sessions.columns:
        df_sessions[col] = df_sessions[col].apply(
            lambda x: x if isinstance(x, list) else ([] if pd.isna(x) else [x])
        )

# Re-check datasets for existing sessions if requested
if df_existing is not None and args.recheck_missing:
    print(f"Re-checking datasets for {len(df_existing)} existing sessions...")
    df_existing = df_existing.progress_apply(
        check_datasets, axis='columns', exlog=exlog
    ).copy()

# Merge with existing data if applicable
if df_existing is not None:
    if len(df_sessions) > 0:
        print(f"Merging {len(df_sessions)} new sessions with {len(df_existing)} existing sessions...")
    else:
        print(f"Using {len(df_existing)} existing sessions...")
    df_sessions = pd.concat([df_existing, df_sessions], ignore_index=True)

# Save sessions and exceptions
n_total = len(df_sessions)
n_exceptions = len(exlog)
n_successful = n_total - n_exceptions

print(
    f"Finished processing sessions:"
    f"\n{n_total} total sessions in database"
    f"\n{n_successful} successful\n{n_exceptions} exceptions"
    )

# Save sessions
df2pqt(df_sessions, SESSIONS_FPATH)

# Save exceptions
if exlog:
    df_exceptions = pd.DataFrame(exlog)
    exception_fpath = PROJECT_ROOT / 'metadata/query_database_log.pqt'
    df2pqt(df_exceptions, exception_fpath)
