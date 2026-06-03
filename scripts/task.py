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
import argparse
from pathlib import Path

import pandas as pd

from iblnm.config import (
    SESSIONS_FPATH, PERFORMANCE_FPATH, TASK_LOG_FPATH,
    SESSIONS_H5_DIR,
)
from iblnm.data import PhotometrySessionGroup
from iblnm.io import _get_default_connection
from iblnm.util import collect_errors
from iblnm.validation import BlockStructureBug


def process_task(ps, reprocess=False):
    """Compute task performance metrics for a single session."""
    import h5py

    # Skip if already processed (trials group exists in H5)
    if not reprocess:
        h5_path = SESSIONS_H5_DIR / f'{ps.eid}.h5'
        if h5_path.exists():
            with h5py.File(h5_path, 'r') as f:
                if 'trials' in f:
                    return 'skipped'

    ps.load_trials()
    ps.validate_n_trials()

    try:
        ps.validate_event_completeness()
    except Exception as e:
        ps.log_error(e)

    result = {'eid': ps.eid, 'n_trials': len(ps.trials)}
    result['contrasts'] = sorted(ps.trials['contrast'].unique().tolist())
    result.update(ps.basic_performance())

    try:
        ps.validate_block_structure()
    except BlockStructureBug as e:
        ps.log_error(e)
        ps.fix_block_structure()

    if 'probabilityLeft' in ps.trials.columns:
        result.update(ps.block_performance())

    ps.save_h5(groups=['trials'])
    return {'performance': result}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Task performance analysis')
    parser.add_argument('--reprocess', action='store_true',
                        help='Re-process all sessions, ignoring existing trials data')
    parser.add_argument('--workers', '-w', type=int, default=1,
                        help='Number of parallel workers')
    args = parser.parse_args()

    one = _get_default_connection()

    print(f"Loading sessions from {SESSIONS_FPATH}")
    group = PhotometrySessionGroup.from_catalog(pd.read_parquet(SESSIONS_FPATH), one=one)
    group.filter_sessions(
        session_types=False, qc_blockers=set(),
        targetnms=False, min_performance=False,
        required_contrasts=False,
    )
    print(f"  {len(group.sessions)} sessions after filtering")

    results = group.process(process_task, workers=args.workers,
                             reprocess=args.reprocess)

    # Aggregate results
    processed = [r for r in results if isinstance(r, dict)]
    perf_rows = [r['performance'] for r in processed]

    df_performance = pd.DataFrame(perf_rows) if perf_rows else pd.DataFrame()

    # Merge with existing parquet for skipped sessions (unless --reprocess)
    if not args.reprocess and Path(PERFORMANCE_FPATH).exists():
        df_prev = pd.read_parquet(PERFORMANCE_FPATH)
        keep = df_prev[~df_prev['eid'].isin(df_performance['eid'])]
        df_performance = pd.concat([keep, df_performance], ignore_index=True)

    n_processed = len(processed)
    n_skipped = sum(1 for r in results if r == 'skipped')
    n_failed = sum(1 for r in results if r is None)
    print(f"\nResults: {n_processed} processed, {n_skipped} skipped, {n_failed} failed")

    # Save outputs
    Path(PERFORMANCE_FPATH).parent.mkdir(parents=True, exist_ok=True)
    df_performance.to_parquet(PERFORMANCE_FPATH)
    print(f"Saved {len(df_performance)} sessions to {PERFORMANCE_FPATH}")

    # Collect errors from H5 files
    df_errors = collect_errors(SESSIONS_H5_DIR)
    if len(df_errors) > 0:
        df_errors.to_parquet(TASK_LOG_FPATH)
        print(f"Saved {len(df_errors)} error entries to {TASK_LOG_FPATH}")
        print(f"Error types:\n{df_errors['error_type'].value_counts().to_string()}")
    else:
        print("No errors logged.")
