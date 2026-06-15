"""
LightningPose Pose Extraction Pipeline

For each session with LightningPose output available:
1. Load trials and wheel from ONE
2. Load LP pose + leftCamera.times (the availability test)
3. Extract per-trial peri-event movement traces per bodypart
4. Compute the paw-wheel cross-correlation timing diagnostic
5. Write the video/ group to the session HDF5 file

Sessions without LP pose raise MissingLP, which is logged (non-fatal) and the
session is skipped; no video/ group is written.

Input:  metadata/sessions.pqt, ONE (LP pose, camera times, wheel)
Output: data/sessions/{eid}.h5 (video/ group appended)
"""
import argparse

import h5py
import pandas as pd

from iblnm.config import SESSIONS_FPATH, SESSIONS_H5_DIR
from iblnm.data import PhotometrySessionGroup
from iblnm.io import _get_default_connection
from iblnm.util import collect_errors
from iblnm.validation import MissingLP


def process_pose(ps, reprocess=False):
    """Extract and save pose movement traces + timing cross-correlation.

    Skips sessions whose H5 already holds a ``video/`` group unless
    ``reprocess`` is set. A ``MissingLP`` from ``load_pose`` is logged
    (non-fatal) and the session is skipped without writing a group.
    Fatal errors are raised (caught by group.process()).
    """
    if not reprocess:
        h5_path = SESSIONS_H5_DIR / f'{ps.eid}.h5'
        if h5_path.exists():
            with h5py.File(h5_path, 'r') as f:
                if 'video' in f:
                    return 'skipped'

    ps.load_trials()
    ps.load_wheel()
    try:
        ps.load_pose()
    except MissingLP as e:
        ps.log_error(e)
        return 'skipped'
    ps.extract_movement_traces()
    ps.extract_paw_wheel_xcorr()
    ps.save_h5(groups=['video'])

    return 'processed'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LightningPose pose extraction pipeline')
    parser.add_argument('--reprocess', action='store_true',
                        help='Re-process all sessions, ignoring existing video '
                             'data (the spec --overwrite flag maps to this)')
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

    results = group.process(process_pose, workers=args.workers,
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
