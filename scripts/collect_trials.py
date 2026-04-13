"""
Collect Trials Data

Loops over proficient sessions and collects all trials data from H5 files
into a single parquet file with eid, subject, and session_n metadata.

Output: data/trials.pqt

Usage:
    python scripts/collect_trials.py
"""
import pandas as pd
from tqdm import tqdm

from iblnm.config import (
    PROJECT_ROOT, SESSIONS_FPATH, SESSIONS_H5_DIR, PERFORMANCE_FPATH,
    QUERY_DATABASE_LOG_FPATH, PHOTOMETRY_LOG_FPATH, TASK_LOG_FPATH,
    ANALYSIS_QC_BLOCKERS, SESSION_TYPES_TO_ANALYZE, TARGETNMS_TO_ANALYZE,
)
from iblnm.data import PhotometrySession, PhotometrySessionGroup
from iblnm.io import _get_default_connection
from iblnm.util import collect_session_errors

OUTPUT_FPATH = PROJECT_ROOT / 'data' / 'trials.pqt'


if __name__ == '__main__':
    # Load sessions and create group (same pattern as task_encoding.py)
    print(f"Loading sessions from {SESSIONS_FPATH}")
    df = pd.read_parquet(SESSIONS_FPATH)
    df = collect_session_errors(
        df, [QUERY_DATABASE_LOG_FPATH, PHOTOMETRY_LOG_FPATH, TASK_LOG_FPATH])
    if PERFORMANCE_FPATH.exists():
        perf = pd.read_parquet(
            PERFORMANCE_FPATH, columns=['eid', 'fraction_correct', 'contrasts'])
        df = df.merge(perf, on='eid', how='left')

    group = PhotometrySessionGroup.from_catalog(df, one=None)
    group.filter_sessions(
        session_types=SESSION_TYPES_TO_ANALYZE,
        qc_blockers=ANALYSIS_QC_BLOCKERS,
        targetnms=TARGETNMS_TO_ANALYZE,
    )
    # Deduplicate to unique sessions (filter gives one row per recording)
    df_sessions = group.sessions.drop_duplicates(subset='eid')
    print(f"  {len(df_sessions)} sessions after filtering")

    # Collect trials from H5 files
    one = _get_default_connection()
    all_trials = []
    n_missing = 0
    for _, row in tqdm(df_sessions.iterrows(), total=len(df_sessions), desc='Loading trials'):
        h5_path = SESSIONS_H5_DIR / f"{row['eid']}.h5"
        if not h5_path.exists():
            n_missing += 1
            continue
        ps = PhotometrySession(row, one=one)
        ps.load_h5(h5_path, groups=['trials'])
        if ps.trials is None:
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
