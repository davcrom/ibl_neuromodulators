"""
Task Performance Analysis

Computes for each session:
- Overall performance (fraction correct, excluding no-go trials)
- Performance on easy trials (>= 50% contrast)
- Fraction of no-go trials
- Psychometric function parameters per block type
- Block structure validation
- Bias shift (difference between 80% and 20% blocks)
- Event completeness flags

No write-back to sessions.pqt — produces standalone outputs.

Output: data/performance.pqt, data/performance_log.pqt
"""
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from iblnm.config import (
    SESSIONS_FPATH, PERFORMANCE_FPATH, TASK_LOG_FPATH, QUERY_DATABASE_LOG_FPATH,
    SESSION_TYPES_TO_ANALYZE,
)
from iblnm.io import _get_default_connection
from iblnm.util import make_log_entry, LOG_COLUMNS, collect_session_errors
from iblnm.data import PhotometrySession


def compute_all_session_performance(df_sessions, one=None, verbose=True):
    """
    Compute task performance metrics for all sessions.

    Expects a pre-filtered df_sessions (session_type and upstream errors already
    screened). Use collect_session_errors() + filters before calling.

    Returns
    -------
    df_performance : pd.DataFrame
        Performance metrics per session; basic metrics always present,
        block metrics present when session_type is biased/ephys and block
        structure is valid.
    df_log : pd.DataFrame
        Error log with schema (eid, error_type, error_message, traceback).
    """
    if one is None:
        one = _get_default_connection()

    results = []
    error_log = []

    for _, session_series in tqdm(df_sessions.iterrows(), total=len(df_sessions),
                                   desc="Computing task performance", disable=not verbose):
        eid = session_series['eid']

        # Block 1: Load (fatal)
        try:
            ps = PhotometrySession(session_series, one=one)
            ps.load_trials()
        except Exception as e:
            error_log.append(make_log_entry(eid, error=e))
            continue

        # Block 2: n_trials (fatal)
        n_trial_errors = ps.validate_n_trials()
        error_log.extend(n_trial_errors)
        if n_trial_errors:
            continue

        # Block 3: event completeness (non-blocking — logged only)
        error_log.extend(ps.validate_event_completeness())

        # Block 4: block structure (blocks block_performance for biased/ephys)
        block_errors = ps.validate_block_structure()
        error_log.extend(block_errors)

        # Block 5: basic performance (fatal)
        try:
            result = {'eid': eid, 'n_trials': len(ps.trials)}
            result.update(ps.basic_performance())
        except Exception as e:
            error_log.append(make_log_entry(eid, error=e))
            continue

        # Block 6: block performance (skipped if block_errors)
        if not block_errors:
            try:
                result.update(ps.block_performance())
            except Exception as e:
                error_log.append(make_log_entry(eid, error=e))

        results.append(result)

    df_performance = pd.DataFrame(results)
    df_log = pd.DataFrame(error_log) if error_log else pd.DataFrame(columns=LOG_COLUMNS)
    return df_performance, df_log


if __name__ == '__main__':
    print(f"Loading sessions from {SESSIONS_FPATH}")
    df_sessions = pd.read_parquet(SESSIONS_FPATH)
    print(f"Loaded {len(df_sessions)} sessions")

    # Merge error flags from upstream pipeline logs and drop sessions with fatal upstream errors
    df_sessions = collect_session_errors(df_sessions, [QUERY_DATABASE_LOG_FPATH])
    fatal_errors = {'InvalidNeuromodulator', 'InvalidTarget', 'InvalidTargetNM'}
    df_sessions = df_sessions[
        df_sessions['logged_errors'].apply(lambda errs: not any(e in fatal_errors for e in errs))
    ]

    # Filter to sessions valid for task analysis
    df_sessions = df_sessions[df_sessions['session_type'].isin(SESSION_TYPES_TO_ANALYZE)].copy()
    print(f"Processing {len(df_sessions)} sessions after filtering")

    one = _get_default_connection()
    df_performance, df_log = compute_all_session_performance(df_sessions, one=one)

    Path(PERFORMANCE_FPATH).parent.mkdir(parents=True, exist_ok=True)
    df_performance.to_parquet(PERFORMANCE_FPATH)
    df_log.to_parquet(TASK_LOG_FPATH)
    print(f"\nSaved performance data to {PERFORMANCE_FPATH}")
    print(f"Saved error log to {TASK_LOG_FPATH}")
    if len(df_log) > 0:
        print(f"  {len(df_log)} sessions with errors/warnings")
        print(f"  Error types:\n{df_log['error_type'].value_counts().to_string()}")
