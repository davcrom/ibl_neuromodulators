"""
Wheel Velocity Pipeline

For each session that has an existing HDF5 file:
1. Load trials (stimOn_times, feedback_times) from the HDF5 file
2. Download wheel position + timestamps from ONE and compute velocity
3. Extract per-trial wheel velocity (stimOn → feedback), NaN-padded to longest trial
4. Append wheel/velocity to the HDF5 file

Input:  metadata/sessions.pqt, data/sessions/{eid}.h5 (created by photometry.py)
Output: data/sessions/{eid}.h5 (wheel/ group appended)
"""
import argparse

import pandas as pd

from iblnm.config import SESSIONS_FPATH, SESSIONS_H5_DIR
from iblnm.data import PhotometrySessionGroup
from iblnm.io import _get_default_connection
from iblnm.util import collect_errors


def process_wheel(ps, reprocess=False):
    """Extract and save per-trial wheel velocity for a single session.

    Fatal errors are raised (caught by group.process()).
    """
    import h5py

    # Skip if already processed (wheel group exists in H5)
    if not reprocess:
        h5_path = SESSIONS_H5_DIR / f'{ps.eid}.h5'
        if h5_path.exists():
            with h5py.File(h5_path, 'r') as f:
                if 'wheel/responses' in f:
                    return 'skipped'

    ps.load_trials()
    ps.load_wheel()
    ps.extract_wheel_velocity()
    ps.save_h5(groups=['wheel'])

    return 'processed'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Wheel velocity pipeline')
    parser.add_argument('--reprocess', action='store_true',
                        help='Re-process all sessions, ignoring existing wheel data')
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

    results = group.process(process_wheel, workers=args.workers,
                            reprocess=args.reprocess)

    n_processed = sum(1 for r in results if r == 'processed')
    n_skipped = sum(1 for r in results if r == 'skipped')
    n_failed = sum(1 for r in results if r is None)
    print(f"\nResults: {n_processed} processed, {n_skipped} skipped, {n_failed} failed")

    # Collect errors from H5 files
    df_errors = collect_errors(SESSIONS_H5_DIR)
    if len(df_errors) > 0:
        print(f"\nError summary ({len(df_errors)} entries):")
        print(df_errors['error_type'].value_counts().to_string())
