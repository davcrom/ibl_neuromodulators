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
    SESSIONS_FPATH, QCPHOTOMETRY_FPATH, PHOTOMETRY_LOG_FPATH,
    QUERY_DATABASE_LOG_FPATH,
    SESSION_TYPES_TO_ANALYZE, VALID_NEUROMODULATORS, VALID_TARGETS, VALID_TARGETNMS,
    RESPONSE_EVENTS,
)
from iblnm.io import _get_default_connection
from iblnm.util import make_log_entry, LOG_COLUMNS, collect_session_errors
from iblnm.data import PhotometrySession


def run_photometry_pipeline(df_sessions, one=None, verbose=True):
    """Run QC, preprocessing, and response extraction on all sessions.

    Expects a pre-filtered df_sessions (session_type, NM, brain_region, target_NM
    already validated). Use collect_session_errors() + manual filter before calling.

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
        tipt_errors = ps.validate_trials_in_photometry_time()
        error_log.extend(tipt_errors)
        if tipt_errors:
            continue

        # Block 2: QC (run_qc non-fatal; band inversions + early samples fatal)
        try:
            ps.run_qc()
            qc_results.append(ps.qc)
        except Exception as e:
            error_log.append(make_log_entry(eid, error=e))
            continue

        qc_errors = ps.validate_qc()          # band inversions, early samples → fatal
        error_log.extend(qc_errors)
        if qc_errors:
            continue

        error_log.extend(ps.validate_few_unique_samples())  # logged only

        # Block 3: Preprocess + save signal (fatal for response extraction)
        try:
            ps.preprocess()
            ps.save_h5(mode='w')
        except Exception as e:
            error_log.append(make_log_entry(eid, error=e))
            continue

        # Block 4: Extract responses
        # Too few trials → fatal (skip extraction entirely)
        n_trial_errors = ps.validate_n_trials()
        error_log.extend(n_trial_errors)
        if n_trial_errors:
            continue

        # Incomplete event times → per-event (skip that event, process the rest)
        event_errors = ps.validate_event_completeness()
        error_log.extend(event_errors)
        failed_events = {e['error_message'].split(': ')[-1] for e in event_errors}
        events_to_extract = [e for e in RESPONSE_EVENTS if e not in failed_events]

        if events_to_extract:
            try:
                ps.extract_responses(events=events_to_extract)
                ps.save_h5(mode='a')
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

    # Merge error flags from upstream pipeline logs and drop sessions with fatal upstream errors
    df_sessions = collect_session_errors(df_sessions, [QUERY_DATABASE_LOG_FPATH])
    fatal_errors = {'InvalidNeuromodulator', 'InvalidTarget', 'InvalidTargetNM'}
    df_sessions = df_sessions[
        df_sessions['logged_errors'].apply(lambda errs: not any(e in fatal_errors for e in errs))
    ]

    # Filter to sessions that are valid for photometry analysis
    df_sessions = df_sessions[df_sessions['session_type'].isin(SESSION_TYPES_TO_ANALYZE)]

    print(f"Processing {len(df_sessions)} sessions after filtering")

    one = _get_default_connection()
    df_qc, df_log = run_photometry_pipeline(df_sessions, one=one)

    QCPHOTOMETRY_FPATH.parent.mkdir(parents=True, exist_ok=True)
    df_qc.to_parquet(QCPHOTOMETRY_FPATH)
    df_log.to_parquet(PHOTOMETRY_LOG_FPATH)
    print(f"\nSaved QC results to {QCPHOTOMETRY_FPATH}")
    print(f"Saved error log to {PHOTOMETRY_LOG_FPATH}")
    if len(df_log) > 0:
        print(f"  {len(df_log)} sessions with errors")
        print(f"  Error types:\n{df_log['error_type'].value_counts().to_string()}")
