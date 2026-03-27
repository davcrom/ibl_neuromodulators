"""
Photometry Processing Pipeline

For each session:
1. Load trials + photometry from ONE
2. Validate trials in photometry time (fatal)
3. Run QC metrics; validate band inversions + early samples (fatal), few unique samples (logged)
4. Preprocess (bleach correct → isosbestic correct → zscore via pipeline; resample separately)
5. Save preprocessed signal to HDF5
6. Validate n_trials (fatal for response extraction)
7. Extract peri-event responses per complete event → save to HDF5

Input:  metadata/sessions.pqt
Output: data/sessions/{eid}.h5, data/qc_photometry.pqt, metadata/photometry_log.pqt
"""
import pandas as pd
from tqdm import tqdm

from iblnm.config import (
    SESSIONS_FPATH, SESSIONS_H5_DIR, QCPHOTOMETRY_FPATH, PHOTOMETRY_LOG_FPATH,
    QUERY_DATABASE_LOG_FPATH, TASK_LOG_FPATH, PERFORMANCE_FPATH,
    SESSION_TYPES_TO_ANALYZE, SUBJECTS_TO_EXCLUDE,
    RESPONSE_EVENTS, MIN_TRAINING_PERFORMANCE, REQUIRED_CONTRASTS,
)
from iblnm.io import _get_default_connection
from iblnm.util import collect_session_errors, resolve_duplicate_group, LOG_COLUMNS
from iblnm.validation import make_log_entry
from iblnm.data import PhotometrySession


def run_photometry_pipeline(df_sessions, one=None, verbose=True):
    """Run QC, preprocessing, and response extraction on all sessions.

    Returns
    -------
    df_qc : pd.DataFrame
        QC metrics, one row per (eid, brain_region, band).
    df_log : pd.DataFrame
        Error log with schema (eid, error_type, error_message, traceback).
    """
    if one is None:
        one = _get_default_connection()

    qc_results = []
    error_log = []

    for _, session_series in tqdm(df_sessions.iterrows(), total=len(df_sessions),
                                  disable=not verbose, desc="Photometry"):
        eid = session_series['eid']

        # Block 1: Load data (fatal — nothing else can proceed)
        try:
            ps = PhotometrySession(session_series, one=one)
            ps.load_trials()
            ps.load_photometry()
        except Exception as e:
            error_log.append(make_log_entry(eid, error=e))
            continue

        # Trials in photometry time (fatal)
        try:
            ps.validate_trials_in_photometry_time()
        except Exception as e:
            error_log.append(make_log_entry(eid, error=e))
            continue

        # Block 2: Raw QC (fatal — band inversions / early samples)
        try:
            ps.run_raw_qc()
        except Exception as e:
            error_log.append(make_log_entry(eid, error=e))
            continue

        try:
            ps.validate_qc()
        except Exception as e:
            error_log.append(make_log_entry(eid, error=e))
            continue

        # Sliding QC (fatal)
        try:
            ps.run_sliding_qc()
            qc_results.append(ps.qc)
        except Exception as e:
            error_log.append(make_log_entry(eid, error=e))
            continue

        try:
            ps.validate_few_unique_samples()       # non-fatal: logged only
        except Exception as e:
            error_log.append(make_log_entry(eid, error=e))

        # Block 3: Preprocess + save signal (fatal for response extraction)
        try:
            ps.preprocess()
            ps.save_h5(groups=['signal'])
        except Exception as e:
            error_log.append(make_log_entry(eid, error=e))
            continue

        # Block 4: Extract responses
        # Too few trials → fatal (skip extraction entirely)
        try:
            ps.validate_n_trials()
        except Exception as e:
            error_log.append(make_log_entry(eid, error=e))
            continue

        # Incomplete event times → per-event (skip that event, process the rest)
        events_to_extract = list(RESPONSE_EVENTS)
        try:
            ps.validate_event_completeness()
        except Exception as e:
            error_log.append(make_log_entry(eid, error=e))
            events_to_extract = [  # missing_events carries the list for logic
                ev for ev in RESPONSE_EVENTS if ev not in getattr(e, 'missing_events', [])
            ]

        if events_to_extract:
            try:
                ps.extract_responses(events=events_to_extract)
                ps.save_h5(groups=['trials', 'responses'], mode='a')
            except Exception as e:
                error_log.append(make_log_entry(eid, error=e))

    df_qc = pd.concat(qc_results, ignore_index=True) if qc_results else pd.DataFrame()
    df_log = (pd.DataFrame(error_log) if error_log
              else pd.DataFrame(columns=LOG_COLUMNS))
    return df_qc, df_log


if __name__ == '__main__':
    print(f"Loading sessions from {SESSIONS_FPATH}")
    df_sessions = pd.read_parquet(SESSIONS_FPATH)
    print(f"Loaded {len(df_sessions)} sessions")

    # Filter to sessions that responses.py will analyze:
    # biased + ephys + qualifying training (performance + contrast filters)
    df_sessions = df_sessions[
        df_sessions['session_type'].isin(SESSION_TYPES_TO_ANALYZE) &
        ~df_sessions['subject'].isin(SUBJECTS_TO_EXCLUDE)
    ]
    print(f"  After session type filter: {len(df_sessions)}")

    # Attach upstream error logs (needed for dedup and QC filter)
    df_sessions = collect_session_errors(
        df_sessions,
        [QUERY_DATABASE_LOG_FPATH, PHOTOMETRY_LOG_FPATH, TASK_LOG_FPATH],
    )

    # Deduplicate: one session per (subject, day_n)
    dup_log = []
    df_sessions = (
        df_sessions.groupby(['subject', 'day_n'], group_keys=False)
        .apply(resolve_duplicate_group, exlog=dup_log, include_groups=True)
        .reset_index(drop=True)
    )
    print(f"  After deduplication: {len(df_sessions)}")

    # Apply basic QC filter
    _qc_blockers = {
        'MissingRawData', 'MissingExtractedData', 'InsufficientTrials',
        'TrialsNotInPhotometryTime', 'QCValidationError', 'FewUniqueSamples',
    }
    df_sessions = df_sessions[
        df_sessions['logged_errors'].apply(
            lambda e: not any(err in _qc_blockers for err in e)
        )
    ]
    print(f"  After basic QC filter: {len(df_sessions)}")

    # Performance + contrast filter from performance table
    if PERFORMANCE_FPATH.exists():
        import numpy as np
        df_perf = pd.read_parquet(
            PERFORMANCE_FPATH, columns=['eid', 'fraction_correct', 'contrasts'],
        )
        df_sessions = df_sessions.merge(df_perf, on='eid', how='left')

        # Performance: training must meet threshold; biased/ephys pass through
        is_training = df_sessions['session_type'] == 'training'
        meets_perf = df_sessions['fraction_correct'] >= MIN_TRAINING_PERFORMANCE
        df_sessions = df_sessions[~is_training | meets_perf]

        # Contrasts: exact match required for all session types
        required_set = set(REQUIRED_CONTRASTS)
        df_sessions = df_sessions[df_sessions['contrasts'].apply(
            lambda c: set(c) == required_set
            if isinstance(c, (list, np.ndarray)) else False
        )]
        df_sessions = df_sessions.drop(columns=['fraction_correct', 'contrasts'])
        print(f"  After performance + contrast filter: {len(df_sessions)}")
    else:
        print(f"  WARNING: {PERFORMANCE_FPATH} not found, skipping performance filter")

    # Skip sessions that already have H5 files
    from pathlib import Path
    existing_h5 = {p.stem for p in Path(SESSIONS_H5_DIR).glob('*.h5')}
    n_before = len(df_sessions)
    df_sessions = df_sessions[~df_sessions['eid'].isin(existing_h5)]
    print(f"  Skipping {n_before - len(df_sessions)} sessions with existing H5 files")
    print(f"Processing {len(df_sessions)} sessions")

    one = _get_default_connection()
    df_qc, df_log = run_photometry_pipeline(df_sessions, one=one)

    QCPHOTOMETRY_FPATH.parent.mkdir(parents=True, exist_ok=True)
    df_qc.to_parquet(QCPHOTOMETRY_FPATH)
    print(f"\nSaved QC results to {QCPHOTOMETRY_FPATH}")
    # NOTE: not saving error log to avoid overwriting the complete log with a
    # partial subset (see issue #2). Print errors for inspection instead.
    if len(df_log) > 0:
        print(f"  {len(df_log)} sessions with errors")
        print(f"  Error types:\n{df_log['error_type'].value_counts().to_string()}")
