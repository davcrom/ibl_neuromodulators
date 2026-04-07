"""Query the IBL fibrephotometry database and build per-session H5 files.

For each session, queries Alyx for metadata (subject info, brain regions,
datasets), validates, and saves to {eid}.h5. At the end, collects all H5
metadata into a catalog parquet and an aggregated error log.

Usage:
    python scripts/query_database.py                  # incremental (skip existing)
    python scripts/query_database.py --redownload     # re-query all sessions
    python scripts/query_database.py --workers 4      # parallel queries
    python scripts/query_database.py --extended-qc    # also fetch extended QC
"""
import argparse

import pandas as pd

from one.api import ONE

from iblnm.config import (
    SESSION_SCHEMA, SESSIONS_FPATH, SESSIONS_QC_FPATH, SESSIONS_H5_DIR,
    QUERY_DATABASE_LOG_FPATH,
)
from iblnm.data import PhotometrySessionGroup
from iblnm.io import get_extended_qc
from iblnm.util import (
    collect_catalog, collect_errors,
    enforce_schema, fill_empty_lists_from_group, fill_brain_region_from_fibers,
    fix_brain_regions, derive_target_nm, df2pqt,
)


def query_session(ps):
    """Query Alyx metadata for a single session and save to H5."""
    ps.from_alyx()
    ps.save_h5(groups=['metadata', 'errors'])
    return ps.eid


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Query IBL fibrephotometry database')
    parser.add_argument('--redownload', action='store_true',
                        help='Re-query all sessions, ignoring existing H5 files')
    parser.add_argument('--workers', '-w', type=int, default=1,
                        help='Number of parallel workers')
    parser.add_argument('--extended-qc', action='store_true',
                        help='Fetch extended QC data (saved to separate file)')
    args = parser.parse_args()

    one = ONE()

    # Get list of session dicts from Alyx
    print("Querying database...")
    sessions = one.alyx.rest('sessions', 'list', project='ibl_fibrephotometry')
    print(f"Found {len(sessions)} sessions on Alyx")

    # Build DataFrame from Alyx dicts
    df = pd.DataFrame(sessions).rename(columns={'id': 'eid'})

    # Skip sessions with existing H5 metadata (unless --redownload)
    if not args.redownload:
        existing = {p.stem for p in SESSIONS_H5_DIR.glob('*.h5')}
        before = len(df)
        df = df[~df['eid'].isin(existing)]
        print(f"Skipping {before - len(df)} with existing H5, "
              f"{len(df)} to process")
        if len(df) == 0:
            print("No new sessions. Collecting catalog from existing H5 files...")
    else:
        print("Re-downloading all data...")

    # Process sessions
    if len(df) > 0:
        group = PhotometrySessionGroup(df, one=one)
        results = group.process(query_session, workers=args.workers)

        n_ok = sum(1 for r in results if r is not None)
        n_failed = sum(1 for r in results if r is None)
        print(f"\nQueried: {n_ok} succeeded, {n_failed} failed")

    # Collect catalog from all H5 files
    print("Collecting catalog from H5 metadata...")
    df_sessions = collect_catalog(SESSIONS_H5_DIR)

    # -----------------------------------------------------------------
    # TEMPFIX: cross-session fill/fix operations
    # These compensate for incomplete Alyx metadata and should be removed
    # once the upstream data is corrected.
    # -----------------------------------------------------------------
    df_sessions = fill_empty_lists_from_group(df_sessions, 'brain_region')
    df_sessions = fill_empty_lists_from_group(df_sessions, 'hemisphere')
    df_sessions = fill_brain_region_from_fibers(df_sessions)
    df_sessions = fix_brain_regions(df_sessions)
    df_sessions = derive_target_nm(df_sessions)
    # -----------------------------------------------------------------

    # Derive convenience columns (display/query aids on the catalog)
    df_sessions['date'] = pd.to_datetime(
        df_sessions['start_time'], format='ISO8601'
    ).dt.date
    df_sessions['day_n'] = df_sessions.groupby('subject')['date'].transform(
        lambda x: [(date - x.min()).days for date in x]
    )
    df_sessions['session_n'] = df_sessions.groupby('subject')['date'].rank(
        method='dense'
    )
    df_sessions = df_sessions.drop(columns='date').copy()

    # Normalize schema
    df_sessions = enforce_schema(df_sessions, SESSION_SCHEMA)

    # Extended QC (optional)
    error_log = []
    df_qc = None
    if args.extended_qc:
        from tqdm import tqdm
        print("Fetching extended QC...")
        tqdm.pandas()
        df_qc = df_sessions[['eid']].copy()
        df_qc = df_qc.progress_apply(
            get_extended_qc, axis='columns', exlog=error_log
        ).copy()

    # Save derived catalog
    df_sessions.to_parquet(SESSIONS_FPATH, index=False)
    print(f"Saved {len(df_sessions)} sessions to {SESSIONS_FPATH}")

    # Save QC
    if df_qc is not None:
        df2pqt(df_qc, SESSIONS_QC_FPATH)

    # Save aggregated error log
    df_errors = collect_errors(SESSIONS_H5_DIR)
    if error_log:
        extra = pd.DataFrame(error_log)
        df_errors = pd.concat([df_errors, extra], ignore_index=True)
    if len(df_errors) > 0:
        df_errors.to_parquet(QUERY_DATABASE_LOG_FPATH, index=False)
        print(f"Saved {len(df_errors)} error entries to {QUERY_DATABASE_LOG_FPATH}")
        print(f"Error types:\n{df_errors['error_type'].value_counts().to_string()}")
    else:
        print("No errors logged.")
