"""
Parallel photometry processing for good biased+ephys sessions.

Applies the same error-log QC filters as responses.py, skips sessions
with existing H5 files, and processes the rest.

Usage:
    python scripts/test_parallel_photometry.py -n 2           # test 2 sessions
    python scripts/test_parallel_photometry.py -n 0 -w 4      # all sessions, 4 workers
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from iblnm.config import (
    SESSIONS_FPATH, SESSIONS_H5_DIR, PERFORMANCE_FPATH,
    QUERY_DATABASE_LOG_FPATH, PHOTOMETRY_LOG_FPATH, TASK_LOG_FPATH,
    SUBJECTS_TO_EXCLUDE,
    RESPONSE_EVENTS, MIN_TRAINING_PERFORMANCE,
)
from iblnm.io import _get_default_connection
from iblnm.util import collect_session_errors, resolve_duplicate_group, LOG_COLUMNS
from iblnm.validation import make_log_entry
from iblnm.data import PhotometrySession


def _process_one_session(session_series):
    """Process a single session: QC, preprocess, extract responses.

    Returns (qc_dict_or_None, list_of_error_dicts).
    Creates its own ONE connection (one per process).
    """
    one = _get_default_connection()
    eid = session_series['eid']
    errors = []

    # Block 1: Load data (fatal)
    try:
        ps = PhotometrySession(session_series, one=one)
        ps.load_trials()
        ps.load_photometry()
    except Exception as e:
        return None, [make_log_entry(eid, error=e)]

    # Trials in photometry time (fatal)
    try:
        ps.validate_trials_in_photometry_time()
    except Exception as e:
        return None, [make_log_entry(eid, error=e)]

    # Block 2: Raw QC (fatal)
    try:
        ps.run_raw_qc()
    except Exception as e:
        return None, [make_log_entry(eid, error=e)]

    try:
        ps.validate_qc()
    except Exception as e:
        return None, [make_log_entry(eid, error=e)]

    # Sliding QC (fatal)
    try:
        ps.run_sliding_qc()
        qc = ps.qc
    except Exception as e:
        return None, [make_log_entry(eid, error=e)]

    try:
        ps.validate_few_unique_samples()       # non-fatal: logged only
    except Exception as e:
        errors.append(make_log_entry(eid, error=e))

    # Block 3: Preprocess + save signal (fatal for response extraction)
    try:
        ps.preprocess()
        ps.save_h5(groups=['signal'])
    except Exception as e:
        errors.append(make_log_entry(eid, error=e))
        return qc, errors

    # Block 4: Extract responses
    try:
        ps.validate_n_trials()
    except Exception as e:
        errors.append(make_log_entry(eid, error=e))
        return qc, errors

    events_to_extract = list(RESPONSE_EVENTS)
    try:
        ps.validate_event_completeness()
    except Exception as e:
        errors.append(make_log_entry(eid, error=e))
        events_to_extract = [
            ev for ev in RESPONSE_EVENTS if ev not in getattr(e, 'missing_events', [])
        ]

    if events_to_extract:
        try:
            ps.extract_responses(events=events_to_extract)
            ps.save_h5(groups=['trials', 'responses'], mode='a')
        except Exception as e:
            errors.append(make_log_entry(eid, error=e))

    return qc, errors


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--workers', type=int, default=1,
                        help='Number of parallel workers (default: 1 = sequential)')
    parser.add_argument('-n', '--n-sessions', type=int, default=2,
                        help='Number of sessions to test (default: 2)')
    args = parser.parse_args()

    # Filter to good biased + ephys + high-performance training sessions
    df = pd.read_parquet(SESSIONS_FPATH)
    df = df[
        df['session_type'].isin(('biased', 'ephys', 'training')) &
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

    # Training sessions: require min performance from performance table
    if PERFORMANCE_FPATH.exists():
        df_perf = pd.read_parquet(PERFORMANCE_FPATH, columns=['eid', 'fraction_correct'])
        df = df.merge(df_perf, on='eid', how='left')
        is_training = df['session_type'] == 'training'
        meets_perf = df['fraction_correct'] >= MIN_TRAINING_PERFORMANCE
        df = df[~is_training | meets_perf]
        df = df.drop(columns='fraction_correct')

    print(f"  Good sessions after filters: {len(df)}")

    # Skip sessions that already have H5 files
    from pathlib import Path
    existing_h5 = {p.stem for p in Path(SESSIONS_H5_DIR).glob('*.h5')}
    n_before = len(df)
    df = df[~df['eid'].isin(existing_h5)]
    print(f"  Skipping {n_before - len(df)} with existing H5")

    if args.n_sessions > 0:
        df = df.head(args.n_sessions)
    print(f"Processing {len(df)} sessions, {args.workers} worker(s)")

    rows = [row for _, row in df.iterrows()]
    qc_results = []
    error_log = []

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_process_one_session, row): row['eid']
                       for row in rows}
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc="Photometry"):
                qc, errors = future.result()
                if qc is not None:
                    qc_results.append(qc)
                error_log.extend(errors)
    else:
        for row in tqdm(rows, desc="Photometry"):
            qc, errors = _process_one_session(row)
            if qc is not None:
                qc_results.append(qc)
            error_log.extend(errors)

    print(f"\nResults: {len(qc_results)} succeeded, {len(error_log)} errors")
    if error_log:
        df_log = pd.DataFrame(error_log)
        print(f"Error types:\n{df_log['error_type'].value_counts().to_string()}")

    # Verify H5 files were created
    for _, row in df.iterrows():
        h5 = Path(SESSIONS_H5_DIR) / f"{row['eid']}.h5"
        status = "OK" if h5.exists() else "MISSING"
        print(f"  {row['eid']}: {status}")
