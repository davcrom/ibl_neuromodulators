"""
Check Video Data Quality

Assess video data quality across sessions using pre-computed QC from
sessions_qc.pqt and session metadata from sessions.pqt.

Pipeline:
1. Filter to analyzable sessions with required datasets
2. Download leftCamera.times per session, compare with session_length
3. Split into df_problems / df_good based on QC flags
4. Score df_good on 5 QC metrics, sort, batch into ~100-200 session chunks

Outputs: data/video_problems.pqt, data/video_batches.pqt
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm

from iblnm.config import (
    PROJECT_ROOT, SESSIONS_FPATH, SESSIONS_QC_FPATH,
    SESSION_TYPES_TO_ANALYZE, SUBJECTS_TO_EXCLUDE,
    QCVAL2NUM, DATASET_CATEGORIES
)
from iblnm.io import _get_default_connection
from iblnm.util import has_dataset_category

# --- Parameters ---
VIDEO_PROBLEMS_FPATH = PROJECT_ROOT / 'metadata/iblnm_video_problems.csv'
VIDEO_BATCHES_FPATH = PROJECT_ROOT / 'metadata/iblnm_video_lp_batches.csv'

DATA_CATEGORIES = [
    'raw_video',
    'extracted_task',
    'extracted_photometry_signal'
# TODO: add video timestamps
]

VIDEO_QC_QUALITY_COLS = [
    'qc_videoLeft_focus',
    'qc_videoLeft_position',
    'qc_videoLeft_brightness',
    'qc_videoLeft_resolution',
    'qc_videoLeft_wheel_alignment',
]

VIDEO_QC_PROBLEM_COLS = [
    'qc_videoLeft_timestamps',
    'qc_videoLeft_dropped_frames',
    'qc_videoLeft_pin_state',
]

LENGTH_MISMATCH_THRESHOLD = 120  # seconds

QCVAL2NUM = {
    np.nan: np.nan,
    # ~'nan': np.nan,  # string 'nan' from parquet files
    # ~'NOT SET': 0.01,
    'NOT_SET': np.nan,
    'PASS': 1.,
    'WARNING': 0.66,
    'CRITICAL': 0.33,
    'FAIL': 0.
}


if __name__ == "__main__":

    # =========================================================================
    # Step 1: Load and filter
    # =========================================================================

    print(f"Loading sessions from {SESSIONS_FPATH}")
    df_sessions = pd.read_parquet(SESSIONS_FPATH)
    n_total = len(df_sessions)
    print(f"  Total sessions: {n_total}")

    # Keep analyzable session types only
    valid_subject = ~df_sessions['subject'].isin(SUBJECTS_TO_EXCLUDE)
    valid_session_type = df_sessions['session_type'].isin(SESSION_TYPES_TO_ANALYZE)
    valid_data = df_sessions.apply(
        lambda x: all([has_dataset_category(x, c) for c in DATA_CATEGORIES]),
        axis='columns'
    )

    df = df_sessions[valid_subject & valid_session_type & valid_data]
    print(f"  Valid sessions: {len(df)}")

    # Merge with QC data
    print(f"\nLoading QC from {SESSIONS_QC_FPATH}")
    df_qc = pd.read_parquet(SESSIONS_QC_FPATH)
    qc_cols = ['eid'] + VIDEO_QC_QUALITY_COLS + VIDEO_QC_PROBLEM_COLS
    df = df.merge(df_qc[qc_cols], on='eid', how='left')
    print(f"  Merged QC data ({len(qc_cols) - 1} QC columns)")

    # Save dataframe indicating missing qc flags
    df_qc[qc_cols].isnull().drop(columns='eid').set_index(df_qc['eid']).to_csv('iblnm_video_missing_qc_flags.csv')

    # =========================================================================
    # Step 2: Download camera timestamps and compare with session_length
    # =========================================================================

    print(f"\nDownloading leftCamera.times for {len(df)} sessions...")
    one = _get_default_connection()

    df['video_t0'] = np.nan
    df['video_t1'] = np.nan
    df['framerate_from_tpts'] = np.nan
    df['n_tpts'] = np.nan
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            times = one.load_dataset(row['eid'], '*leftCamera.times*')
            df.at[idx, 'video_t0'] = times[0]
            df.at[idx, 'video_t1'] = times[-1]
            df.at[idx, 'framerate_from_tpts'] = np.median(np.diff(times))
            df.at[idx, 'n_tpts'] = len(times)
        except Exception:
            pass

    df['video_length'] = df['video_t1'] - df['video_t0']
    df['length_discrepancy'] = df['video_length'] - df['session_length']

    n_valid_video = df['video_length'].notna().sum()
    n_download_errors = df['video_length'].isna().sum()
    print(f"  Downloaded: {n_valid_video} successful, {n_download_errors} errors")

    # =========================================================================
    # Step 3: Flag problem sessions
    # =========================================================================

    print("\nFlagging problem sessions...")

    valid_qc = (df[VIDEO_QC_PROBLEM_COLS] == 'PASS').all(axis=1)
    valid_length = df['length_discrepancy'] < LENGTH_MISMATCH_THRESHOLD

    df_problems = df[~(valid_qc & valid_length)].copy()
    df_good = df[valid_qc & valid_length].copy()

    # Problem breakdown
    n_qc_fail = (~valid_qc).sum()
    n_length_fail = (~valid_length).sum()
    n_both_fail = (~valid_qc & ~valid_length).sum()
    print(f"  Problem sessions: {len(df_problems)}")
    print(f"    QC fail: {n_qc_fail}")
    for col in VIDEO_QC_PROBLEM_COLS:
        n_fail = (df[col] != 'PASS').sum()
        n_missing = df[col].isna().sum()
        print(f"      {col}: {n_fail} fail ({n_missing} missing)")
    print(f"    Length mismatch (>{LENGTH_MISMATCH_THRESHOLD}s): {n_length_fail}")
    print(f"    Both: {n_both_fail}")
    print(f"  Good sessions: {len(df_good)}")

    # =========================================================================
    # Step 4: Score and batch good sessions
    # =========================================================================

    print("\nScoring and batching good sessions...")

    for col in VIDEO_QC_QUALITY_COLS:
        df_good[col + '_num'] = df_good[col].map(QCVAL2NUM)

    num_cols = [col + '_num' for col in VIDEO_QC_QUALITY_COLS]
    df_good['video_qc_score'] = np.nanmean(df_good[num_cols], axis=1)

    # Sort by score descending (best first)
    df_good = df_good.sort_values('video_qc_score', ascending=False).reset_index(drop=True)

    # Cluster by video_qc_score, then order batches best-first
    n_good = len(df_good)
    if n_good > 0:
        k = max(1, n_good // 250)
        km = KMeans(n_clusters=k, n_init='auto', random_state=42)
        labels = km.fit_predict(df_good[['video_qc_score']])
        # Rank clusters by centroid score descending (batch 0 = best)
        centroids = km.cluster_centers_.flatten()
        rank = np.argsort(-centroids)
        label_to_batch = {label: batch for batch, label in enumerate(rank)}
        df_good['batch'] = pd.Series(labels, index=df_good.index).map(label_to_batch)
    else:
        df_good['batch'] = pd.Series(dtype=int)

    batch_cols = (
        ['eid', 'subject', 'session_type', 'batch', 'video_qc_score']
        + VIDEO_QC_QUALITY_COLS
    )
    df_batches = df_good[batch_cols].copy()

    n_batches = df_batches['batch'].nunique() if len(df_batches) > 0 else 0
    print(f"  Good sessions: {n_good}, batches: {n_batches}")
    if n_batches > 0:
        for batch_id, grp in df_batches.groupby('batch'):
            print(f"    Batch {batch_id}: {len(grp)} sessions, "
                  f"score {grp['video_qc_score'].min():.2f}-{grp['video_qc_score'].max():.2f}")

    # =========================================================================
    # Step 5: Save and summarize
    # =========================================================================

    VIDEO_PROBLEMS_FPATH.parent.mkdir(parents=True, exist_ok=True)
    df_problems.to_csv(VIDEO_PROBLEMS_FPATH)
    df_batches.to_csv(VIDEO_BATCHES_FPATH)

    # Final summary
    n_excluded_subjects = (~valid_subject).sum()
    n_excluded_type = (~valid_session_type).sum()
    n_excluded_data = (~valid_data).sum()

    print(f"\n{'='*50}")
    print("Summary")
    print(f"{'='*50}")
    print(f"Total sessions: {n_total}")
    print(f"Excluded:")
    print(f"  Subjects: {n_excluded_subjects}")
    print(f"  Session type: {n_excluded_type}")
    print(f"  Missing datasets: {n_excluded_data}")
    print(f"Valid sessions: {len(df)}")
    print(f"  Video times downloaded: {n_valid_video}, failed: {n_download_errors}")
    print(f"  Problems: {len(df_problems)}")
    print(f"  Good: {n_good} ({n_batches} batches)")
    print(f"\nSaved: {VIDEO_PROBLEMS_FPATH}")
    print(f"Saved: {VIDEO_BATCHES_FPATH}")
