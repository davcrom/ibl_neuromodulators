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

import pandas as pd

from iblnm.config import (
    SESSIONS_FPATH, PERFORMANCE_FPATH, TASK_LOG_FPATH, TRIAL_TIMING_FPATH,
    SESSIONS_H5_DIR, VALID_TARGETNMS,
)
from iblnm.data import PhotometrySessionGroup
from iblnm.io import _get_default_connection
from iblnm.util import collect_errors
from iblnm.validation import BlockStructureBug


def process_task(ps):
    """Compute task performance metrics for a single session."""
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
        if ps.fix_block_structure():
            ps.save_h5(groups=['trials'])

    if 'probabilityLeft' in ps.trials.columns:
        result.update(ps.block_performance())

    timing = ps.get_trial_timings()
    return {'performance': result, 'timing': timing}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Task performance analysis')
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

    results = group.process(process_task, workers=args.workers)

    # Aggregate results
    perf_rows = [r['performance'] for r in results if r is not None]
    timing_frames = [r['timing'] for r in results if r is not None]

    df_performance = pd.DataFrame(perf_rows) if perf_rows else pd.DataFrame()
    df_trial_timing = (
        pd.concat(timing_frames, ignore_index=True) if timing_frames
        else pd.DataFrame(
            columns=['eid', 'trial', 'reaction_time', 'movement_time',
                     'response_time']
        )
    )

    n_processed = sum(1 for r in results if r is not None)
    n_failed = sum(1 for r in results if r is None)
    print(f"\nResults: {n_processed} processed, {n_failed} failed")

    # Save outputs
    Path(PERFORMANCE_FPATH).parent.mkdir(parents=True, exist_ok=True)
    df_performance.to_parquet(PERFORMANCE_FPATH)
    df_trial_timing.to_parquet(TRIAL_TIMING_FPATH, index=False)
    print(f"Saved {len(df_performance)} sessions to {PERFORMANCE_FPATH}")
    print(f"Saved trial timing to {TRIAL_TIMING_FPATH}")

    # Collect errors from H5 files
    df_errors = collect_errors(SESSIONS_H5_DIR)
    if len(df_errors) > 0:
        df_errors.to_parquet(TASK_LOG_FPATH)
        print(f"Saved {len(df_errors)} error entries to {TASK_LOG_FPATH}")
        print(f"Error types:\n{df_errors['error_type'].value_counts().to_string()}")
    else:
        print("No errors logged.")
