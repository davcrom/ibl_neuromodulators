"""
LightningPose Pose Extraction Pipeline

For each session with LightningPose output available:
1. Load leftCamera.times and compute basic-video measures (blocks the session
   if absent)
2. Load trials and wheel from ONE
3. Load LP pose (the availability test)
4. Extract per-trial peri-event movement traces per bodypart
5. Compute the paw-wheel cross-correlation timing diagnostic
6. Write the video/ group to the session HDF5 file

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
    LP_QC_LABELS,
    MOVEMENT_RESPONSE_WINDOW,
    PERFORMANCE_FPATH,
    POSE_FPATH,
    POSE_MEASURES,
    QCVAL2NUM,
    SESSION_TYPES,
    SESSIONS_FPATH,
    SESSIONS_H5_DIR,
    VIDEO_QC_COLS,
    VIDEO_QC_QUALITY_COLS,
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
from iblnm.validation import (
    MissingLP, MissingMotionEnergy, MissingVideoTimestamps,
    VideoLengthError, VideoTimestampsQCError,
    VideoDroppedFramesQCError, VideoPinStateQCError,
    validate_video_length, validate_video_timestamps_qc,
    validate_video_dropped_frames_qc, validate_video_pin_state_qc,
)

# leftCamera QC validations run in process_pose; failures are logged
# non-blocking so extraction always proceeds regardless of the verdict.
VIDEO_QC_VALIDATORS = (
    validate_video_length,
    validate_video_timestamps_qc,
    validate_video_dropped_frames_qc,
    validate_video_pin_state_qc,
)
VIDEO_QC_ERRORS = (
    VideoLengthError, VideoTimestampsQCError,
    VideoDroppedFramesQCError, VideoPinStateQCError,
)
# Error types that disqualify a session in the rollup: video_qc_score forced
# to -1 (missing timestamps plus the four leftCamera QC failures).
VIDEO_QC_DISQUALIFYING_ERRORS = frozenset(
    e.__name__ for e in (MissingVideoTimestamps, *VIDEO_QC_ERRORS))
# Trace-derived scalar columns guaranteed in the rollup, NaN when their source
# is absent: the LP keypoint measures plus the motion-energy channel.
POSE_TRACE_COLUMNS = [*POSE_MEASURES, 'motion_energy']


def run_video_validations(ps):
    """Run the four leftCamera QC checks, logging any failure non-blocking.

    Builds a row-like dict from ``ps.length_discrepancy`` and the eight
    ``ps.video_qc`` labels and runs each ``validate_video_*`` check. Any raised
    QC error is routed to ``ps.log_error`` so extraction continues regardless
    of the verdict.
    """
    qc_row = {'length_discrepancy': ps.length_discrepancy, **ps.video_qc}
    for validate in VIDEO_QC_VALIDATORS:
        try:
            validate(qc_row)
        except VIDEO_QC_ERRORS as e:
            ps.log_error(e)


def process_pose(ps, reprocess=False):
    """Extract and save pose movement traces + timing cross-correlation.

    Checks basic video first: load ``leftCamera.times`` and compute the
    basic-video measures before LP. A ``MissingVideoTimestamps`` blocks the
    whole session (logged, no group). Skips sessions whose H5 already holds a
    ``video/`` group unless ``reprocess`` is set. LP and motion energy load
    independently; ``MissingLP`` and ``MissingMotionEnergy`` are logged
    (non-fatal). Movement traces are extracted from whichever sources are
    present (paw-wheel xcorr only when LP is), and the basic-video group is
    written whenever timestamps existed. Fatal errors are raised (caught by
    group.process()).
    """
    if not reprocess:
        h5_path = SESSIONS_H5_DIR / f'{ps.eid}.h5'
        if h5_path.exists():
            with h5py.File(h5_path, 'r') as f:
                if 'video' in f:
                    return 'skipped'

    try:
        ps.load_camera_times()
    except MissingVideoTimestamps as e:
        ps.log_error(e)
        return 'skipped'
    ps.compute_video_measures()
    ps.fetch_video_qc()
    run_video_validations(ps)
    ps.load_trials()
    ps.load_wheel()
    try:
        ps.load_pose()
    except MissingLP as e:
        ps.log_error(e)
    try:
        ps.load_motion_energy()
    except MissingMotionEnergy as e:
        ps.log_error(e)
    if ps.pose is not None or ps.motion_energy is not None:
        ps.extract_movement_traces()
    if ps.pose is not None:
        ps.extract_paw_wheel_xcorr()
    ps.save_h5(groups=['video'])

    return 'processed'


def _decode(value):
    """Decode an HDF5 bytes value to ``str``; pass non-bytes through unchanged."""
    return value.decode() if isinstance(value, bytes) else value


def _read_session_type(f: h5py.File) -> str | None:
    """Session type from the H5 ``metadata`` group; None when the group is absent."""
    if 'metadata' not in f:
        return None
    return _decode(f['metadata'].attrs.get('session_type'))


def _collect_error_types(h5_dir) -> dict[str, set[str]]:
    """Map each eid to the set of error types in its H5 ``errors`` group."""
    df_errors = collect_errors(h5_dir)
    if df_errors.empty:
        return {}
    return df_errors.groupby('eid')['error_type'].agg(set).to_dict()


def _has_lp_traces(traces) -> bool:
    """True when ``traces`` hold an LP keypoint channel (not just motion energy)."""
    if traces is None:
        return False
    return any(bodypart != 'motion_energy'
               for bodypart in traces.coords['bodypart'].values)


def _score_video_qc(attrs, error_types: set[str]) -> float:
    """Video QC score in [0, 1], or ``-1`` when a disqualifying error is logged.

    The five ``VIDEO_QC_QUALITY_COLS`` labels in the ``video`` group ``attrs``
    are mapped through ``config.QCVAL2NUM`` and averaged with ``nanmean``. Any
    error type in ``VIDEO_QC_DISQUALIFYING_ERRORS`` forces the score to ``-1``;
    a group with no quality labels scores NaN.
    """
    if error_types & VIDEO_QC_DISQUALIFYING_ERRORS:
        return -1.0
    quality = [QCVAL2NUM.get(_decode(attrs[col]), np.nan)
               for col in VIDEO_QC_QUALITY_COLS if col in attrs]
    return float(np.nanmean(quality)) if quality else np.nan


def _add_trace_deltas(row: dict, traces, baseline_traces) -> None:
    """Add one post-minus-pre movement delta per bodypart; no-op when absent.

    The delta is the response mean over ``MOVEMENT_RESPONSE_WINDOW`` minus the
    stimOn-locked baseline mean over ``BASELINE_WINDOW``. The ``motion_energy``
    channel flows through this loop like any other bodypart.
    """
    if traces is None:
        return
    tpts = traces.coords['time'].values
    for bodypart in traces.coords['bodypart'].values:
        row[bodypart] = movement_delta(
            traces.sel(bodypart=bodypart).values,
            baseline_traces.sel(bodypart=bodypart).values, tpts,
            MOVEMENT_RESPONSE_WINDOW, BASELINE_WINDOW,
        )


def _add_xcorr_scalars(row: dict, xcorr) -> None:
    """Add drift and per-third peak lag/value; all NaN when no cross-correlation."""
    if xcorr is None:
        for col in ('drift', 'peak_lag_early', 'peak_lag_mid', 'peak_lag_late',
                    'peak_val_early', 'peak_val_mid', 'peak_val_late'):
            row[col] = np.nan
        return
    row['drift'] = xcorr['drift']
    row['peak_lag_early'], row['peak_lag_mid'], row['peak_lag_late'] = \
        xcorr['peak_lags']
    # Peak value of each third's cross-correlation function (alignment strength)
    row['peak_val_early'], row['peak_val_mid'], row['peak_val_late'] = \
        np.nanmax(xcorr['functions'], axis=1)


def collect_pose(h5_dir, performance_fpath=PERFORMANCE_FPATH) -> pd.DataFrame:
    """Roll up the per-session ``video`` H5 groups into the unified pose table.

    Emits one row per session with a ``video`` group (LP-absent sessions
    included, their trace-derived columns NaN), plus a bare row for each session
    whose ``errors`` group holds ``MissingVideoTimestamps`` but has no ``video``
    group. For each ``video`` group: recompute the per-bodypart movement deltas
    (post-minus-pre, see ``_add_trace_deltas``), read the cross-correlation
    scalars, the manual QC labels (``LP_QC_LABELS``), the eight
    ``VIDEO_QC_COLS`` and the basic-video measures (``length_discrepancy``,
    ``framerate_from_tpts``) from the group attrs, and derive ``lp_exists`` and
    ``video_qc_score`` (``_score_video_qc``). Recomputing the deltas here keeps
    both windows adjustable without re-extracting traces. ``fraction_correct`` is
    left-joined from ``performance.pqt`` by ``eid``; rows are not sorted.

    Parameters
    ----------
    h5_dir : Path or str
        Directory containing ``{eid}.h5`` files.
    performance_fpath : Path or str
        Per-eid performance table (``performance.pqt``) holding
        ``fraction_correct``.

    Returns
    -------
    pd.DataFrame
        One row per session: ``eid``, ``session_type`` (read from the
        ``metadata`` group for both video-group and bare error-stub rows, None
        only when that group is absent), ``lp_exists``, one column per bodypart
        scalar, ``drift``, ``peak_lag_early/mid/late``,
        ``peak_val_early/mid/late``, the ``LP_QC_LABELS`` manual QC labels,
        ``mean_rt``, the
        8 ``VIDEO_QC_COLS``, ``length_discrepancy``, ``framerate_from_tpts``,
        ``video_qc_score``, and ``fraction_correct``.
    """
    errors_by_eid = _collect_error_types(h5_dir)
    rows = []
    eids_with_video = set()
    for fpath in sorted(Path(h5_dir).glob('*.h5')):
        eid = fpath.stem
        error_types = errors_by_eid.get(eid, set())
        with h5py.File(fpath, 'r') as f:
            if 'video' not in f:
                continue
            eids_with_video.add(eid)
            video = f['video']
            traces = _load_pose_traces(video)
            baseline_traces = _load_pose_traces(video, name='baseline_traces')
            xcorr = _load_pose_xcorr(video)
            qc = _read_video_qc(f)
            row = {
                'eid': eid,
                'session_type': _read_session_type(f),
                'lp_exists': _has_lp_traces(traces),
                'mean_rt': _read_mean_rt(f),
                'length_discrepancy': video.attrs.get('length_discrepancy', np.nan),
                'framerate_from_tpts': video.attrs.get('framerate_from_tpts', np.nan),
                'video_qc_score': _score_video_qc(video.attrs, error_types),
            }
            row.update({col: _decode(video.attrs[col])
                        for col in VIDEO_QC_COLS if col in video.attrs})
            _add_trace_deltas(row, traces, baseline_traces)
            _add_xcorr_scalars(row, xcorr)
        row.update({label: _decode(qc.get(label, LP_QC_NOT_SET))
                    for label in LP_QC_LABELS})
        rows.append(row)

    for eid, error_types in errors_by_eid.items():
        if eid in eids_with_video or 'MissingVideoTimestamps' not in error_types:
            continue
        with h5py.File(Path(h5_dir) / f'{eid}.h5', 'r') as f:
            session_type = _read_session_type(f)
        rows.append({'eid': eid, 'session_type': session_type,
                     'lp_exists': False, 'video_qc_score': -1.0})

    df_pose = pd.DataFrame(rows)
    for col in POSE_TRACE_COLUMNS:
        if col not in df_pose.columns:
            df_pose[col] = np.nan
    df_perf = pd.read_parquet(performance_fpath)[['eid', 'fraction_correct']]
    return df_pose.merge(df_perf, on='eid', how='left')


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
    parser.add_argument('--session-type', nargs='+', choices=SESSION_TYPES,
                        default=None,
                        help='Restrict processing to these session types '
                             '(default: all types)')
    args = parser.parse_args()

    if not args.collect:

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
            session_types=args.session_type or False, qc_blockers=set(),
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


    # Roll up the video H5 groups into the pose table
    df_pose = collect_pose(SESSIONS_H5_DIR)
    df_pose.to_parquet(POSE_FPATH)
    print(f"\nWrote {len(df_pose)} session rows to {POSE_FPATH}")

    # Post-hoc analysis-ready CSV: drop rows whose eid is excluded by the
    # cohort filters (subject/eid exclusions, QC blockers, target NMs). Session
    # type, performance and contrast filters are skipped so every type is kept.
    group = PhotometrySessionGroup.from_catalog(catalog, one=one, h5_dir=SESSIONS_H5_DIR)
    group.filter_sessions(
        session_types=False,
        min_performance=False,
        required_contrasts=False
        )
    _ = group.deduplicate()
    df_csv = df_pose[df_pose['eid'].isin(group.sessions['eid'])]
    type2val = {
            'ephys': 3,
            'biased': 2,
            'training': 1,
            'habituation': 0,
        }
    df_csv['session_type_val'] = df_csv['session_type'].apply(
        lambda x: type2val.get(x, -1)
        )
    df_csv = df_csv.sort_values(
        ['lp_exists', 'session_type_val', 'video_qc_score'],
        ascending=[False, False, False]
        )
    csv_fpath = 'metadata/LightningPoseSessions.csv'
    df_csv.to_csv(csv_fpath, index=False)
    print(f"Wrote {len(df_csv)} analysis-ready session to {csv_fpath}")
