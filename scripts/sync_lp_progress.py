"""
Sync LightningPose Progress

Merge LP-only columns from LightningPoseSessions.csv into video.pqt.
Only adds columns that don't already exist in video.pqt. Saves the
merged result as video.csv — video.pqt is left untouched.

Run after check_video_data.py to produce a CSV with LP progress
alongside video QC data.
"""
import sys

import pandas as pd

from iblnm.config import VIDEO_FPATH, LP_SESSIONS_FPATH, POSE_FPATH


def merge_pose(df_video: pd.DataFrame, df_pose: pd.DataFrame) -> pd.DataFrame:
    """Left-join the pose roll-up onto the video table, keyed by ``eid``.

    Adds every non-``eid`` column from ``df_pose``; video eids with no pose
    row get NaN in those columns. Returns one row per ``df_video`` eid.
    """
    pose_cols = [c for c in df_pose.columns if c != 'eid']
    return df_video.merge(df_pose[['eid'] + pose_cols], on='eid', how='left')


if __name__ == "__main__":

    # Load video.pqt (must exist)
    print(f"Loading video data from {VIDEO_FPATH}")
    df_video = pd.read_parquet(VIDEO_FPATH)
    print(f"  {len(df_video)} sessions")

    # Load LP csv
    if not LP_SESSIONS_FPATH.exists():
        print(f"No LightningPose file at {LP_SESSIONS_FPATH}, nothing to sync.")
        sys.exit(0)

    print(f"Loading LP data from {LP_SESSIONS_FPATH}")
    df_lp = pd.read_csv(LP_SESSIONS_FPATH)
    print(f"  {len(df_lp)} sessions")

    # Identify LP-only columns (not already in video.pqt)
    lp_only_cols = [c for c in df_lp.columns if c not in df_video.columns and c != 'eid']
    if not lp_only_cols:
        print("No new columns to add from LP csv.")
        df_out = df_video.copy()
    else:
        print(f"  Adding {len(lp_only_cols)} LP columns: {lp_only_cols}")
        df_out = df_video.merge(df_lp[['eid'] + lp_only_cols], on='eid', how='left')

    # Merge the pose roll-up, if it exists
    if POSE_FPATH.exists():
        print(f"Loading pose roll-up from {POSE_FPATH}")
        df_pose = pd.read_parquet(POSE_FPATH)
        print(f"  {len(df_pose)} sessions, columns: {[c for c in df_pose.columns if c != 'eid']}")
        df_out = merge_pose(df_out, df_pose)
    else:
        print(f"No pose roll-up at {POSE_FPATH}, skipping pose merge.")

    # Save as CSV (video.pqt is left untouched)
    output_fpath = VIDEO_FPATH.with_suffix('.csv')
    df_out.to_csv(output_fpath, index=False)
    print(f"Saved: {output_fpath}")

    # Integrity check: every LP eid must be in video.pqt
    video_eids = set(df_out['eid'])
    lp_eids = set(df_lp['eid'])
    missing = lp_eids - video_eids
    if missing:
        print(f"\nWARNING: {len(missing)} LP eids not found in video.pqt:")
        for eid in sorted(missing):
            print(f"  {eid}")
    else:
        print(f"\nIntegrity check passed: all {len(lp_eids)} LP eids present in video.pqt.")
