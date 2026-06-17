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
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from iblnm.analysis import movement_delta
from iblnm.config import (
    BASELINE_WINDOW,
    MOVEMENT_RESPONSE_WINDOW,
    POSE_FPATH,
    SESSIONS_FPATH,
    SESSIONS_H5_DIR,
)
from iblnm.data import (
    LP_QC_NOT_SET,
    PhotometrySessionGroup,
    _load_pose_traces,
    _load_pose_xcorr,
    _read_video_qc,
)
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


def collect_pose(h5_dir) -> pd.DataFrame:
    """Roll up the per-session ``video`` H5 groups into a flat pose table.

    For each H5 holding a ``video`` group, recompute the four scalar measures
    as post-minus-pre deltas: the response mean over ``MOVEMENT_RESPONSE_WINDOW``
    on the event-locked trace minus the baseline mean over ``BASELINE_WINDOW``
    on the stimOn-locked baseline trace. Also read the cross-correlation drift,
    the three per-third peak lags, and the two manual QC labels. Recomputing the
    scalars here keeps both windows adjustable without re-extracting traces.

    Parameters
    ----------
    h5_dir : Path or str
        Directory containing ``{eid}.h5`` files.

    Returns
    -------
    pd.DataFrame
        One row per eid with a ``video`` group: ``eid``, one column per
        bodypart scalar, ``drift``, ``peak_lag_early/mid/late``, ``qc_lp``,
        ``qc_movement``.
    """
    rows = []
    for fpath in sorted(Path(h5_dir).glob('*.h5')):
        with h5py.File(fpath, 'r') as f:
            if 'video' not in f:
                continue
            traces = _load_pose_traces(f['video'])
            if traces is None:
                continue
            baseline_traces = _load_pose_traces(f['video'], name='baseline_traces')
            xcorr = _load_pose_xcorr(f['video'])
            qc = _read_video_qc(f)
            mean_rt = _read_mean_rt(f)

        tpts = traces.coords['time'].values
        row = {'eid': fpath.stem}
        for bodypart in traces.coords['bodypart'].values:
            row[bodypart] = movement_delta(
                traces.sel(bodypart=bodypart).values,
                baseline_traces.sel(bodypart=bodypart).values, tpts,
                MOVEMENT_RESPONSE_WINDOW, BASELINE_WINDOW,
            )
        early, mid, late = xcorr['peak_lags']
        row['drift'] = xcorr['drift']
        row['peak_lag_early'] = early
        row['peak_lag_mid'] = mid
        row['peak_lag_late'] = late
        for label in ('qc_lp', 'qc_movement'):
            value = qc.get(label, LP_QC_NOT_SET)
            row[label] = value.decode() if isinstance(value, bytes) else value
        row['mean_rt'] = mean_rt
        rows.append(row)
    return pd.DataFrame(rows)


def _read_mean_rt(f: h5py.File) -> float:
    """Mean reaction time from the H5 ``trials`` group, NaN when unavailable.

    Reaction time is ``feedback_times - stimOn_times`` per trial, averaged with
    ``nanmean``. Returns NaN if the ``trials`` group or either dataset is absent.
    """
    if 'trials' not in f:
        return np.nan
    trials = f['trials']
    if 'stimOn_times' not in trials or 'feedback_times' not in trials:
        return np.nan
    return np.nanmean(trials['feedback_times'][:] - trials['stimOn_times'][:])


def read_eids(path) -> list[str]:
    """Read one eid per line from a text/CSV file, ignoring blank lines."""
    return [line.strip() for line in Path(path).read_text().splitlines()
            if line.strip()]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LightningPose pose extraction pipeline')
    parser.add_argument('--reprocess', action='store_true',
                        help='Re-process all sessions, ignoring existing video '
                             'data (the spec --overwrite flag maps to this)')
    parser.add_argument('--workers', '-w', type=int, default=1,
                        help='Number of parallel workers')
    parser.add_argument('--collect', action='store_true',
                        help='Skip extraction; roll up existing video H5 groups '
                             f'into {POSE_FPATH} and exit')
    parser.add_argument('--eids', nargs='+', default=None,
                        help='Restrict processing to these session eids (testing)')
    parser.add_argument('--eids-file', default=None,
                        help='Restrict processing to eids listed one-per-line in '
                             'this file (avoids shell expansion under IPython)')
    args = parser.parse_args()

    if args.collect:
        df_pose = collect_pose(SESSIONS_H5_DIR)
        df_pose.to_parquet(POSE_FPATH)
        print(f"Wrote {len(df_pose)} session rows to {POSE_FPATH}")
        raise SystemExit

    one = _get_default_connection()

    print(f"Loading sessions from {SESSIONS_FPATH}")
    catalog = pd.read_parquet(SESSIONS_FPATH)
    eids = read_eids(args.eids_file) if args.eids_file else args.eids
    if eids:
        catalog = catalog[catalog['eid'].isin(eids)]
        if catalog.empty:
            raise SystemExit(f"None of the {len(eids)} requested eids are in "
                             f"{SESSIONS_FPATH}")
    group = PhotometrySessionGroup.from_catalog(catalog, one=one)
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

    # Roll up the video H5 groups into the pose table
    df_pose = collect_pose(SESSIONS_H5_DIR)
    df_pose.to_parquet(POSE_FPATH)
    print(f"\nWrote {len(df_pose)} session rows to {POSE_FPATH}")
