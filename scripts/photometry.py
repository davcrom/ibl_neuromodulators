"""
Photometry Processing Pipeline

For each session:
1. Load trials + photometry from ONE
2. Validate trials in photometry time (fatal)
3. Run QC metrics; validate band inversions + early samples (fatal), few unique samples (logged)
4. Preprocess (bleach correct -> isosbestic correct -> zscore via pipeline; resample separately)
5. Save preprocessed signal + QC to HDF5
6. Validate n_trials (fatal for response extraction)
7. Extract peri-event responses per complete event -> save to HDF5

Input:  metadata/sessions.pqt (via PhotometrySessionGroup.from_catalog)
Output: data/sessions/{eid}.h5, data/qc_photometry.pqt

Usage:
    python scripts/photometry.py                  # incremental (skip processed)
    python scripts/photometry.py --reprocess      # re-process all sessions
    python scripts/photometry.py --workers 4      # parallel processing
"""
import argparse

import pandas as pd

from iblnm.config import (
    SESSIONS_FPATH, SESSIONS_H5_DIR, QCPHOTOMETRY_FPATH,
    RESPONSE_EVENTS, VALID_TARGETNMS,
)
from iblnm.data import PhotometrySessionGroup
from iblnm.io import _get_default_connection
from iblnm.util import collect_qc, collect_errors


def process_session(ps, reprocess=False):
    """Run the photometry pipeline on a single PhotometrySession.

    Fatal errors are raised (caught by group.process()).
    Non-fatal errors are logged via ps.log_error().
    """
    import h5py

    # Skip if already processed (signal group exists in H5)
    if not reprocess:
        h5_path = SESSIONS_H5_DIR / f'{ps.eid}.h5'
        if h5_path.exists():
            with h5py.File(h5_path, 'r') as f:
                if 'preprocessed' in f:
                    return 'skipped'

    # Block 1: Load data (fatal)
    ps.load_trials()
    ps.load_photometry()

    # Trials in photometry time (fatal)
    ps.validate_trials_in_photometry_time()

    # Block 2: Raw QC (fatal)
    ps.run_raw_qc()
    ps.validate_qc()

    # Sliding QC (fatal)
    ps.run_sliding_qc()

    # Few unique samples (non-fatal)
    try:
        ps.validate_few_unique_samples()
    except Exception as e:
        ps.log_error(e)

    # Block 3: Preprocess + save signal and QC (fatal)
    ps.preprocess()
    ps.save_h5(groups=['signal', 'photometry_qc_metrics'])

    # Block 4: Extract responses
    # Too few trials (fatal)
    ps.validate_n_trials()

    # Incomplete event times (non-fatal: skip missing events, process the rest)
    events_to_extract = list(RESPONSE_EVENTS)
    try:
        ps.validate_event_completeness()
    except Exception as e:
        ps.log_error(e)
        events_to_extract = [
            ev for ev in RESPONSE_EVENTS
            if ev not in getattr(e, 'missing_events', [])
        ]

    if events_to_extract:
        ps.extract_responses(events=events_to_extract)
        ps.save_h5(groups=['trials', 'responses'])

    return 'processed'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Photometry processing pipeline')
    parser.add_argument('--reprocess', action='store_true',
                        help='Re-process all sessions, ignoring existing signal data')
    parser.add_argument('--workers', '-w', type=int, default=1,
                        help='Number of parallel workers')
    args = parser.parse_args()

    one = _get_default_connection()

    print(f"Loading sessions from {SESSIONS_FPATH}")
    group = PhotometrySessionGroup.from_catalog(pd.read_parquet(SESSIONS_FPATH), one=one)
    group.filter_sessions(
        session_types=False, qc_blockers=set(),
        targetnms=VALID_TARGETNMS, min_performance=False,
        required_contrasts=False,
    )
    print(f"  {len(group.sessions)} sessions after filtering")

    results = group.process(process_session, workers=args.workers,
                            reprocess=args.reprocess)

    n_processed = sum(1 for r in results if r == 'processed')
    n_skipped = sum(1 for r in results if r == 'skipped')
    n_failed = sum(1 for r in results if r is None)
    print(f"\nResults: {n_processed} processed, {n_skipped} skipped, {n_failed} failed")

    # Collect QC from H5 files
    print("Collecting QC metrics from H5 files...")
    df_qc = collect_qc(SESSIONS_H5_DIR)
    if len(df_qc) > 0:
        QCPHOTOMETRY_FPATH.parent.mkdir(parents=True, exist_ok=True)
        df_qc.to_parquet(QCPHOTOMETRY_FPATH, index=False)
        print(f"Saved {len(df_qc)} QC rows to {QCPHOTOMETRY_FPATH}")
    else:
        print("No QC data to save.")

    # Print error summary from H5 files
    df_errors = collect_errors(SESSIONS_H5_DIR)
    if len(df_errors) > 0:
        print(f"\nError summary ({len(df_errors)} entries):")
        print(df_errors['error_type'].value_counts().to_string())
