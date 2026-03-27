"""
Parallel task performance computation for good sessions.

Applies the same error-log QC filters as the photometry script, skips
sessions already in performance.pqt, and processes the rest.

Output: data/performance.pqt, data/trial_timing.pqt
        (does NOT overwrite error logs)

Usage:
    python scripts/test_parallel_task.py -n 2           # test 2 sessions
    python scripts/test_parallel_task.py -n 0 -w 4      # all sessions, 4 workers
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from iblnm.config import (
    SESSIONS_FPATH, SESSIONS_H5_DIR, PERFORMANCE_FPATH, TRIAL_TIMING_FPATH,
    QUERY_DATABASE_LOG_FPATH, PHOTOMETRY_LOG_FPATH, TASK_LOG_FPATH,
    SESSION_TYPES_TO_ANALYZE, SUBJECTS_TO_EXCLUDE,
    RESPONSE_EVENTS,
)
from iblnm.io import _get_default_connection
from iblnm.util import collect_session_errors, LOG_COLUMNS
from iblnm.validation import make_log_entry
from iblnm.data import PhotometrySession


def _process_one_session(session_series):
    """Compute task performance for a single session.

    Returns (result_dict_or_None, timing_df_or_None, list_of_error_dicts).
    """
    one = _get_default_connection()
    eid = session_series['eid']
    errors = []

    # Load trials (fatal)
    try:
        ps = PhotometrySession(session_series, one=one)
        ps.load_trials()
    except Exception as e:
        return None, None, [make_log_entry(eid, error=e)]

    # n_trials (fatal)
    try:
        ps.validate_n_trials()
    except Exception as e:
        return None, None, [make_log_entry(eid, error=e)]

    # Event completeness (non-blocking)
    try:
        ps.validate_event_completeness()
    except Exception as e:
        errors.append(make_log_entry(eid, error=e))

    # Basic performance (fatal)
    try:
        result = {'eid': eid, 'n_trials': len(ps.trials)}
        result['contrasts'] = sorted(ps.trials['contrast'].unique().tolist())
        result.update(ps.basic_performance())
    except Exception as e:
        errors.append(make_log_entry(eid, error=e))
        return None, None, errors

    # Block performance (non-fatal)
    try:
        ps.validate_block_structure()
        result.update(ps.block_performance())
    except Exception as e:
        errors.append(make_log_entry(eid, error=e))

    timing = ps.get_trial_timings()
    return result, timing, errors


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--workers', type=int, default=1,
                        help='Number of parallel workers (default: 1 = sequential)')
    parser.add_argument('-n', '--n-sessions', type=int, default=2,
                        help='Number of sessions to process (0 = all)')
    args = parser.parse_args()

    # Filter to good sessions (same filters as parallel photometry)
    df = pd.read_parquet(SESSIONS_FPATH)
    df = df[
        df['session_type'].isin(SESSION_TYPES_TO_ANALYZE) &
        ~df['subject'].isin(SUBJECTS_TO_EXCLUDE)
    ]
    df = collect_session_errors(
        df, [QUERY_DATABASE_LOG_FPATH, PHOTOMETRY_LOG_FPATH, TASK_LOG_FPATH],
    )
    _qc_blockers = {
        'MissingRawData', 'MissingExtractedData', 'InsufficientTrials',
        'TrialsNotInPhotometryTime', 'QCValidationError', 'FewUniqueSamples',
    }
    df = df[df['logged_errors'].apply(
        lambda e: not any(err in _qc_blockers for err in e)
    )]
    print(f"  Good sessions after filters: {len(df)}")

    # Skip sessions already in performance.pqt
    if PERFORMANCE_FPATH.exists():
        existing_eids = set(pd.read_parquet(PERFORMANCE_FPATH, columns=['eid'])['eid'])
        n_before = len(df)
        df = df[~df['eid'].isin(existing_eids)]
        print(f"  Skipping {n_before - len(df)} already in performance.pqt")

    if args.n_sessions > 0:
        df = df.head(args.n_sessions)
    print(f"Processing {len(df)} sessions, {args.workers} worker(s)")

    rows = [row for _, row in df.iterrows()]
    results = []
    timing_frames = []
    error_log = []

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_process_one_session, row): row['eid']
                       for row in rows}
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc="Task performance"):
                result, timing, errors = future.result()
                if result is not None:
                    results.append(result)
                if timing is not None:
                    timing_frames.append(timing)
                error_log.extend(errors)
    else:
        for row in tqdm(rows, desc="Task performance"):
            result, timing, errors = _process_one_session(row)
            if result is not None:
                results.append(result)
            if timing is not None:
                timing_frames.append(timing)
            error_log.extend(errors)

    # Merge with existing performance.pqt if present
    df_new = pd.DataFrame(results)
    if PERFORMANCE_FPATH.exists() and len(df_new) > 0:
        df_existing = pd.read_parquet(PERFORMANCE_FPATH)
        df_performance = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_performance = df_new

    df_trial_timing_new = (pd.concat(timing_frames, ignore_index=True)
                           if timing_frames else pd.DataFrame(
                               columns=['eid', 'trial', 'reaction_time',
                                        'movement_time', 'response_time']))
    if TRIAL_TIMING_FPATH.exists() and len(df_trial_timing_new) > 0:
        df_existing_timing = pd.read_parquet(TRIAL_TIMING_FPATH)
        df_trial_timing = pd.concat([df_existing_timing, df_trial_timing_new],
                                    ignore_index=True)
    else:
        df_trial_timing = df_trial_timing_new

    Path(PERFORMANCE_FPATH).parent.mkdir(parents=True, exist_ok=True)
    df_performance.to_parquet(PERFORMANCE_FPATH)
    df_trial_timing.to_parquet(TRIAL_TIMING_FPATH, index=False)
    print(f"\nSaved {len(df_performance)} sessions to {PERFORMANCE_FPATH}")
    print(f"Saved {len(df_trial_timing)} trial timings to {TRIAL_TIMING_FPATH}")

    if error_log:
        df_log = pd.DataFrame(error_log)
        print(f"{len(df_log)} errors:")
        print(df_log['error_type'].value_counts().to_string())
