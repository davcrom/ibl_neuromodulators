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
import pandas as pd
from tqdm import tqdm

from iblnm.config import (
    MOVEMENT_EVENTS,
    POSE_FPATH,
    SESSION_TYPES,
    SESSIONS_FPATH,
    SESSIONS_H5_DIR,
)
from iblnm.data import VIDEO_QC_ERRORS, PhotometrySessionGroup
from iblnm.io import _get_default_connection, get_video_qc
from iblnm.validation import (
    MissingExtractedData, MissingLP, MissingMotionEnergy, MissingRawData,
    MissingVideoTimestamps,
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


def run_video_validations(ps):
    """Run the four leftCamera QC checks, logging any failure non-blocking.

    Builds a row-like dict from ``ps.video_times_qc`` and the eight
    ``ps.video_qc`` labels and runs each ``validate_video_*`` check. Any raised
    QC error is routed to ``ps.log_error`` so extraction continues regardless
    of the verdict.
    """
    qc_row = {**ps.video_times_qc, **ps.video_qc}
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
        ps.load_video_times_qc()
    except MissingVideoTimestamps as e:
        ps.log_error(e, product='video/times')
        return 'skipped'
    ps.fetch_video_qc()
    run_video_validations(ps)
    ps.load_trials()
    ps.load_wheel()
    try:
        ps.load_pose()
    except MissingLP as e:
        ps.log_error(e, product='video/pose')
    try:
        ps.load_motion_energy()
    except MissingMotionEnergy as e:
        ps.log_error(e, product='video/motion_energy')
    if ps.pose is not None or ps.motion_energy is not None:
        ps.movement_responses = ps.extract_responses(
            ps.resample_movement_signals(), events=MOVEMENT_EVENTS)
    if ps.pose is not None:
        # Cross-modal: the wheel is an input, so its absence is a pose-QC
        # failure and is logged against that product, not against the pose.
        try:
            ps.load_pose_qc()
        except (MissingRawData, MissingExtractedData) as e:
            ps.log_error(e, product='video/pose/qc')
    ps.save_h5(groups=['video'])

    return 'processed'


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

    if not args.collect:
        results = group.process(process_pose, workers=args.workers,
                                reprocess=args.reprocess)

        n_processed = sum(1 for r in results if r == 'processed')
        n_skipped = sum(1 for r in results if r == 'skipped')
        n_failed = sum(1 for r in results if r is None)
        print(f"\nResults: {n_processed} processed, {n_skipped} skipped, {n_failed} failed")


    # Roll up the video H5 groups into the pose table. The Alyx QC labels are
    # not in those files, so they are fetched here — one REST call per session.
    video_qc = {eid: get_video_qc(eid, one=one)
                for eid in tqdm(group.sessions['eid'], desc='Fetching video QC')}
    df_pose = group.collect_pose(video_qc=video_qc)
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
    df_csv = df_csv.rename(columns={'lp_exists': 'LP status'})
    df_csv['LP status'] = df_csv['LP status'].map({True: 'COMPLETE', False: 'NONE'})
    csv_columns = [
        'eid', 'LP status', 'session_type', 'fraction_correct', 'mean_rt',
        'framerate_from_tpts', 'length_discrepancy',
        'qc_videoLeft_timestamps', 'qc_videoLeft_dropped_frames',
        'qc_videoLeft_pin_state', 'qc_videoLeft_focus', 'qc_videoLeft_position',
        'qc_videoLeft_brightness', 'qc_videoLeft_resolution',
        'qc_videoLeft_wheel_alignment', 'video_qc_score', 'drift',
        'peak_lag_early', 'peak_lag_mid', 'peak_lag_late',
        'peak_val_early', 'peak_val_mid', 'peak_val_late',
        'motion_energy', 'nose', 'paw', 'tongue_likelihood', 'tongue_speed',
        'qc_lp', 'qc_movement', 'qc_timing',
    ]
    df_csv = df_csv[csv_columns]
    csv_fpath = 'metadata/LightningPoseSessions.csv'
    df_csv.to_csv(csv_fpath, index=False)
    print(f"Wrote {len(df_csv)} analysis-ready session to {csv_fpath}")
