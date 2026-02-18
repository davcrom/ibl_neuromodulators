import argparse
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from one.api import ONE

from iblnm.config import (
    SESSION_SCHEMA, SESSIONS_FPATH, SESSIONS_QC_FPATH, QUERY_DATABASE_LOG_FPATH, VALID_TARGETS,
)
from iblnm.io import (
    get_subject_info, get_session_info, get_datasets, get_extended_qc,
)
from iblnm.util import (
    enforce_schema, validate_subject, validate_strain, validate_line,
    validate_neuromodulator, validate_target, validate_hemisphere,
    validate_datasets, get_session_type, get_targetNM, get_session_length,
    df2pqt,
)


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

# Initialize df to pre-defined schema
df_sessions = enforce_schema(df_sessions, SESSION_SCHEMA)


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


print("Adding subject info...")
df_sessions = df_sessions.progress_apply(
    get_subject_info, axis='columns', exlog=exlog
).copy()
df_sessions.apply(validate_subject, axis='columns', exlog=exlog)
df_sessions.apply(validate_strain, axis='columns', exlog=exlog)
df_sessions.apply(validate_line, axis='columns', exlog=exlog)
df_sessions.apply(validate_neuromodulator, axis='columns', exlog=exlog)


print("Getting experiment descriptions...")
df_sessions = df_sessions.progress_apply(
    get_session_info, axis='columns', exlog=exlog
).copy()
df_sessions.apply(validate_target, axis='columns', exlog=exlog)
df_sessions.apply(validate_hemisphere, axis='columns', exlog=exlog)


print("Getting datasets...")
df_sessions = df_sessions.progress_apply(
    get_datasets, axis='columns', exlog=exlog
).copy()
df_sessions.apply(validate_datasets, axis='columns', exlog=exlog)


# Add convenience columns
df_sessions = df_sessions.apply(get_session_type, axis='columns', exlog=exlog)
df_sessions = df_sessions.apply(get_targetNM, axis='columns', exlog=exlog)
df_sessions = df_sessions.apply(get_session_length, axis='columns', exlog=exlog)
df_sessions['date'] = pd.to_datetime(df_sessions['start_time'], format='ISO8601').dt.date
df_sessions['day_n'] = df_sessions.groupby('subject')['date'].transform(
    lambda x: [(date - x.min()).days for date in x]
)
df_sessions['session_n'] = df_sessions.groupby('subject')['date'].rank(method='dense')
df_sessions = df_sessions.drop(columns='date').copy()


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


# Save sessions
df2pqt(df_sessions, SESSIONS_FPATH)

# Save QC if fetched
if df_qc is not None:
    df2pqt(df_qc, SESSIONS_QC_FPATH)

# Save error log
if exlog:
    df_exceptions = pd.DataFrame(exlog)
    df2pqt(df_exceptions, QUERY_DATABASE_LOG_FPATH)
