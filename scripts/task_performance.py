"""
Task Performance Analysis

Computes for each session:
- Sessions to reach each training stage (training > biased > ephys)
- Overall performance (fraction correct, excluding no-go trials)
- Performance on easy trials (>= 50% contrast)
- Fraction of no-go trials
- Psychometric function parameters per block type (bias, threshold, lapse_left, lapse_right, r_squared)
- Block structure validation (flag sessions with rapidly flipping blocks)
- Bias shift (difference between 80% and 20% blocks)

Output: metadata/performance.pqt
"""
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from iblnm.config import SESSIONS_FPATH, PERFORMANCE_FPATH
from iblnm.io import _get_default_connection
from iblnm.util import clean_sessions, drop_junk_duplicates, merge_session_metadata
from iblnm.data import PhotometrySession


def compute_all_session_performance(df_sessions, one=None, verbose=True):
    if one is None:
        one = _get_default_connection()

    results = []

    for _, session_series in tqdm(df_sessions.iterrows(), total=len(df_sessions),
                                   desc="Computing task performance", disable=not verbose):
        eid = session_series['eid']
        result = {'eid': eid}

        ps = PhotometrySession(session_series, one=one)
        ps.load_trials()

        if not ps.has_trials:
            results.append(result)
            continue

        try:
            perf = ps.task_performance()
            result.update(perf)
        except Exception as e:
            if verbose:
                print(f"Error computing performance for {eid}: {e}")

        results.append(result)

    return pd.DataFrame(results)


if __name__ == '__main__':
    # Load sessions
    print(f"Loading sessions from {SESSIONS_FPATH}")
    df_sessions = pd.read_parquet(SESSIONS_FPATH)
    df_sessions = clean_sessions(df_sessions)
    group_cols = ['subject', 'day_n']
    df_sessions = drop_junk_duplicates(df_sessions, group_cols)
    print(f"Loaded {len(df_sessions)} sessions")

    one = _get_default_connection()

    # Compute performance metrics
    print("\nComputing performance metrics for each session...")
    df_performance = compute_all_session_performance(df_sessions, one=one)

    # Merge session metadata
    df_performance = merge_session_metadata(df_performance)

    # Save performance data
    output_path = Path(PERFORMANCE_FPATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_performance.to_parquet(PERFORMANCE_FPATH)
    print(f"\nSaved performance data to {PERFORMANCE_FPATH}")
