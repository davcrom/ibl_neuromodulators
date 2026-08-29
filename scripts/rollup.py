"""Regenerate the rollup files from the session store.

Six files, read by no analysis: they exist for inspection and for the two
viewers, and every one is rebuilt from `data/sessions/*.h5` through
`PhotometrySessionGroup`, so what lands in a rollup is what the group's filters
admit. Nothing here builds a product — `scripts/download.py` does that, and a
rollup over a store it has not filled reports what is missing rather than
fetching it.

Usage:
    python scripts/rollup.py               # all six files
    python scripts/rollup.py --skip-pose   # the four that need no Alyx call
"""
import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from iblnm.config import (
    ERRORS_FPATH, LP_SESSIONS_FPATH, PERFORMANCE_FPATH, POSE_FPATH,
    QCPHOTOMETRY_FPATH, SESSIONS_FPATH, SESSIONS_H5_DIR,
)
from iblnm.data import PhotometrySessionGroup
from iblnm.io import _get_default_connection, get_video_qc
from iblnm.util import build_catalog

# Columns of `metadata/LightningPoseSessions.csv`, in the order the label sheet
# expects them.
LP_SESSIONS_COLUMNS = [
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

# Session types ranked by how much of the task they carry, so the export puts
# the sessions most worth labelling first. A type outside the map sorts last.
SESSION_TYPE_RANK = {'ephys': 3, 'biased': 2, 'training': 1, 'habituation': 0}


def rollup_catalog(h5_dir: Path = SESSIONS_H5_DIR, one=None,
                   path: Path = SESSIONS_FPATH) -> pd.DataFrame:
    """Rebuild the session catalog from the store's `metadata` groups.

    The one rollup taking no group, because it is what a group is built from.
    `scripts/download.py` writes the same table from the same source after its
    Alyx pass; regenerating it here needs no connection, so `one` is only
    carried onto the group it constructs.
    """
    catalog = build_catalog(
        PhotometrySessionGroup.from_h5_dir(h5_dir, one=one).sessions)
    catalog.to_parquet(path, index=False)
    return catalog


def rollup_qc(group, path: Path = QCPHOTOMETRY_FPATH) -> pd.DataFrame:
    """Write the raw photometry QC of every surviving recording to `path`.

    One row per (eid, brain_region), each metric named with the band it scored
    suffixed on (`n_unique_samples_GCaMP`), as `photometry/{region}/raw/qc`
    stores it. `collect_qc` walks the whole catalog, because the QC threshold
    filter it feeds does not exist while it runs, so the result is cut back to
    the recordings still in scope — a session keeps the regions that pass and
    loses the rest.
    """
    keys = ['eid', 'brain_region']
    qc = group.collect_qc().merge(group.recordings[keys], on=keys, how='inner')
    qc.to_parquet(path, index=False)
    return qc


def rollup_performance(group, path: Path = PERFORMANCE_FPATH) -> pd.DataFrame:
    """Write the sessions' `trials/performance` products to `path`.

    `load_performance` likewise reads every catalogued session, since the
    performance and contrast filters it feeds are computed from what it
    returns; the rollup reports the cohort those filters admit, so its result
    is cut back to `group.sessions` here.
    """
    performance = group.load_performance()
    performance = performance[performance['eid'].isin(group.sessions['eid'])]
    performance.to_parquet(path, index=False)
    return performance


def rollup_errors(group, path: Path = ERRORS_FPATH) -> pd.DataFrame:
    """Write the filtered sessions' logged errors to `path`.

    One row per entry in a session's `errors/` tree, carrying the `product` it
    was logged against, so a failure can be traced to the build that raised it.
    """
    errors = group.collect_errors()
    errors.to_parquet(path, index=False)
    return errors


def fetch_video_qc(group) -> dict[str, dict[str, str]]:
    """The eight leftCamera QC labels per filtered session, fetched from Alyx.

    One REST call each, and the only part of any rollup needing a connection:
    the labels are in no H5, because IBL re-runs its QC without anything here
    being able to tell that a stored copy went stale.
    """
    return {eid: get_video_qc(eid, one=group.one)
            for eid in tqdm(group.sessions['eid'], desc='Fetching video QC')}


def rollup_pose(group, video_qc: dict, path: Path = POSE_FPATH) -> pd.DataFrame:
    """Write the filtered sessions' `video` groups to `path`, one row each.

    `video_qc` maps eid -> the eight Alyx labels, as `fetch_video_qc` returns
    them; a session missing from it scores NaN rather than blocking the rollup.
    """
    pose = group.collect_pose(video_qc=video_qc)
    pose.to_parquet(path, index=False)
    return pose


def cohort_eids(catalog: pd.DataFrame, one=None,
                h5_dir: Path = SESSIONS_H5_DIR) -> set[str]:
    """Eids the analysis cohort admits, with the task-side filters skipped.

    The pose export is about which videos are worth labelling, so session type,
    performance and contrast set say nothing — a habituation session is as
    labellable as an ephys one. Everything else applies: the subject and eid
    exclusions, the blocking error types and the target neuromodulators.
    """
    group = PhotometrySessionGroup.from_catalog(catalog, one=one, h5_dir=h5_dir)
    group.filter_sessions(session_types=False, min_performance=False,
                          required_contrasts=False)
    group.deduplicate()
    return set(group.sessions['eid'])


def pose_export(pose: pd.DataFrame, eids,
                path: Path = LP_SESSIONS_FPATH) -> pd.DataFrame:
    """Write the analysis-ready CSV read by whoever labels the pose sessions.

    The rows of `pose` whose eid is in `eids`, ordered so the sessions most
    worth labelling come first: LightningPose output present, then richest
    session type, then video QC score, each descending. `lp_exists` becomes
    `LP status`, relabelled COMPLETE/NONE — the vocabulary of the label sheet.

    Parameters
    ----------
    pose : pd.DataFrame
        The pose rollup, as `rollup_pose` writes it.
    eids : set of str
        The cohort to keep, from `cohort_eids`.
    path : Path
        Destination CSV.
    """
    export = pose[pose['eid'].isin(eids)].assign(
        session_type_rank=lambda df: df['session_type'].map(
            SESSION_TYPE_RANK).fillna(-1))
    export = export.sort_values(
        ['lp_exists', 'session_type_rank', 'video_qc_score'], ascending=False)
    export = export.rename(columns={'lp_exists': 'LP status'})
    export['LP status'] = export['LP status'].map({True: 'COMPLETE',
                                                   False: 'NONE'})
    export = export[LP_SESSIONS_COLUMNS]
    export.to_csv(path, index=False)
    return export


def store_group(catalog: pd.DataFrame, one=None) -> PhotometrySessionGroup:
    """The whole store as one group: every filter off, nothing deduplicated.

    A rollup describes what was built, so it reports every catalogued session
    rather than the analysis cohort. The one place that cohort is applied is
    the pose export, through `cohort_eids`.
    """
    group = PhotometrySessionGroup.from_catalog(
        catalog, one=one, h5_dir=SESSIONS_H5_DIR, scan_h5_errors=False)
    group.filter_sessions(session_types=False, qc_blockers=set(),
                          targetnms=False, photometry_qc=False,
                          min_performance=False, required_contrasts=False)
    return group


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Regenerate the rollup files from the session store')
    parser.add_argument('--skip-pose', action='store_true',
                        help='Skip pose.pqt and the LightningPose CSV, whose '
                             'video QC labels cost one Alyx call per session')
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    # Only the pose rollup fetches, so a run without it needs no connection.
    one = None if args.skip_pose else _get_default_connection()

    catalog = rollup_catalog(one=one)
    print(f'Catalogued {len(catalog)} sessions in {SESSIONS_FPATH}')

    group = store_group(catalog, one=one)
    for rollup, path in ((rollup_qc, QCPHOTOMETRY_FPATH),
                         (rollup_performance, PERFORMANCE_FPATH),
                         (rollup_errors, ERRORS_FPATH)):
        print(f'Wrote {len(rollup(group, path))} rows to {path}')

    if args.skip_pose:
        return
    pose = rollup_pose(group, fetch_video_qc(group))
    print(f'Wrote {len(pose)} session rows to {POSE_FPATH}')
    export = pose_export(pose, cohort_eids(catalog, one=one))
    print(f'Wrote {len(export)} analysis-ready sessions to {LP_SESSIONS_FPATH}')


if __name__ == '__main__':
    main()
