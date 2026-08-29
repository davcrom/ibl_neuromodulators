"""
Wheel Velocity Pipeline

For each session that has an existing HDF5 file:
1. Load trials (stimOn_times, feedback_times) from the HDF5 file
2. Build the wheel products: raw encoder position from ONE, velocity
   differentiated from it, and the per-trial stimOn → feedback matrix
3. Append the wheel/ group to the HDF5 file

Input:  metadata/sessions.pqt, data/sessions/{eid}.h5 (created by photometry.py)
Output: data/sessions/{eid}.h5 (wheel/ group appended)
"""
import argparse

import pandas as pd

from iblnm.config import SESSIONS_FPATH, SESSION_TYPES
from iblnm.data import PhotometrySessionGroup
from iblnm.io import _get_default_connection


def process_wheel(ps, reprocess=False):
    """Extract and save per-trial wheel velocity for a single session.

    Fatal errors are raised (caught by group.process()).
    """
    if not reprocess and ps.product_status('wheel/responses') == 'current':
        return 'skipped'

    ps.load_trials()
    ps.load_responses('wheel')

    return 'processed'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Wheel velocity pipeline')
    parser.add_argument('--reprocess', action='store_true',
                        help='Re-process all sessions, ignoring existing wheel data')
    parser.add_argument('--workers', '-w', type=int, default=1,
                        help='Number of parallel workers')
    parser.add_argument('--session-type', nargs='+', choices=SESSION_TYPES,
                        default=None,
                        help='Restrict processing to these session types '
                             '(default: all types)')
    args = parser.parse_args()

    one = _get_default_connection()

    print(f"Loading sessions from {SESSIONS_FPATH}")
    group = PhotometrySessionGroup.from_catalog(pd.read_parquet(SESSIONS_FPATH), one=one)
    group.filter_sessions(
        session_types=args.session_type or False, qc_blockers=set(),
        targetnms=False, min_performance=False,
        required_contrasts=False,
    )
    print(f"  {len(group.sessions)} sessions after filtering")

    results = group.process(process_wheel, workers=args.workers,
                            reprocess=args.reprocess)

    n_processed = sum(1 for r in results if r == 'processed')
    n_skipped = sum(1 for r in results if r == 'skipped')
    n_failed = sum(1 for r in results if r is None)
    print(f"\nResults: {n_processed} processed, {n_skipped} skipped, {n_failed} failed")

    # Collect errors from H5 files
    df_errors = group.collect_errors()
    if len(df_errors) > 0:
        print(f"\nError summary ({len(df_errors)} entries):")
        print(df_errors['error_type'].value_counts().to_string())
