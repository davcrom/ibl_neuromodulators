"""
Collect Trials Data

Loops over proficient sessions and collects all trials data from H5 files
into a single parquet file with eid, subject, and session_n metadata.

Output: data/trials.pqt

Usage:
    python scripts/collect_trials.py
"""
import argparse

import pandas as pd
from tqdm import tqdm

from iblnm.config import (
    PROJECT_ROOT, SESSIONS_FPATH, SESSIONS_H5_DIR,
    ANALYSIS_QC_BLOCKERS, SESSION_TYPES_TO_ANALYZE, TARGETNMS_TO_ANALYZE,
)
from iblnm.data import PhotometrySessionGroup
from iblnm.io import _get_default_connection

OUTPUT_FPATH = PROJECT_ROOT / 'data' / 'trials.pqt'


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args(argv)


if __name__ == '__main__':
    parse_args()

    # Load sessions and create group (same pattern as task_encoding.py)
    print(f"Loading sessions from {SESSIONS_FPATH}")
    df = pd.read_parquet(SESSIONS_FPATH)

    one = _get_default_connection()
    group = PhotometrySessionGroup.from_catalog(df, one=one, h5_dir=SESSIONS_H5_DIR)
    group.filter_sessions(
        session_types=SESSION_TYPES_TO_ANALYZE,
        qc_blockers=ANALYSIS_QC_BLOCKERS,
        targetnms=TARGETNMS_TO_ANALYZE,
    )
    # Deduplicate to unique sessions (filter gives one row per recording)
    df_sessions = group.sessions.drop_duplicates(subset='eid')
    print(f"  {len(df_sessions)} sessions after filtering")

    # Collect trials from the store, now that every session in scope holds them.
    all_trials = []
    n_missing = 0
    for _, row in tqdm(df_sessions.iterrows(), total=len(df_sessions), desc='Loading trials'):
        ps = group._get_session(row)
        if not ps.filepath.exists():
            n_missing += 1
            continue
        ps.load_h5(groups=['trials'])
        if not hasattr(ps, 'trials'):
            n_missing += 1
            continue
        trials = ps.trials.copy()
        trials['eid'] = row['eid']
        trials['subject'] = row['subject']
        trials['session_n'] = row['session_n']
        trials['session_type'] = row['session_type']
        all_trials.append(trials)

    if n_missing:
        print(f"  Skipped {n_missing} sessions (no H5 or no trials group)")

    df_trials = pd.concat(all_trials, ignore_index=True)
    print(f"  {len(df_trials)} total trials from {len(all_trials)} sessions")

    df_trials.to_parquet(OUTPUT_FPATH)
    print(f"Saved to {OUTPUT_FPATH}")
