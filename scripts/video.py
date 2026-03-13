"""
Check Video Data Quality

Assess video data quality across sessions using pre-computed QC from
sessions_qc.pqt and session metadata from sessions.pqt.

Pipeline:
1. Filter to analyzable sessions (type, subject, dedup, upstream QC)
2. Merge video QC columns from sessions_qc.pqt
3. Download leftCamera.times per session, compare with session_length
4. Validate video QC problem columns and length discrepancy
5. Score all sessions, sort unified output

Outputs:
- metadata/video.pqt — all sessions, sorted by video_qc_score
- metadata/video_log.pqt — error log
"""
import numpy as np
import pandas as pd
# from sklearn.cluster import KMeans
from tqdm import tqdm

from iblnm.config import (
    SESSIONS_FPATH, SESSIONS_QC_FPATH,
    SESSION_TYPES_TO_ANALYZE, SUBJECTS_TO_EXCLUDE,
    QUERY_DATABASE_LOG_FPATH, PHOTOMETRY_LOG_FPATH, TASK_LOG_FPATH,
    VIDEO_LOG_FPATH, VIDEO_FPATH,
)
from iblnm.io import _get_default_connection
from iblnm.util import (
    collect_session_errors, resolve_duplicate_group, LOG_COLUMNS,
)
from iblnm.validation import (
    make_log_entry,
    validate_video_length,
    validate_video_timestamps_qc,
    validate_video_dropped_frames_qc,
    validate_video_pin_state_qc,
)

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

QCVAL2NUM = {
    np.nan: np.nan,
    'NOT_SET': np.nan,
    'PASS': 1.,
    'WARNING': 0.66,
    'CRITICAL': 0.33,
    'FAIL': 0.
}

SESSION_TYPE_ORDER = {'ephys': 0, 'biased': 1, 'training': 2}


if __name__ == "__main__":

    error_log = []

    # =========================================================================
    # Step 1: Load and filter
    # =========================================================================

    print(f"Loading sessions from {SESSIONS_FPATH}")
    df_sessions = pd.read_parquet(SESSIONS_FPATH)
    n_total = len(df_sessions)
    print(f"  Total sessions: {n_total}")

    # Attach upstream error logs
    df_sessions = collect_session_errors(
        df_sessions,
        [QUERY_DATABASE_LOG_FPATH, PHOTOMETRY_LOG_FPATH, TASK_LOG_FPATH],
    )

    # Filter by session type and subject
    df_sessions = df_sessions[
        df_sessions['session_type'].isin(SESSION_TYPES_TO_ANALYZE) &
        ~df_sessions['subject'].isin(SUBJECTS_TO_EXCLUDE)
    ]
    n_after_type = len(df_sessions)

    # Deduplicate: one session per (subject, day_n)
    dup_log = []
    df_sessions = (
        df_sessions.groupby(['subject', 'day_n'], group_keys=False)
        .apply(resolve_duplicate_group, exlog=dup_log, include_groups=True)
        .reset_index(drop=True)
    )
    n_after_dedup = len(df_sessions)

    print(f"  After session type / subject filter: {n_after_type} (-{n_total - n_after_type})")
    print(f"  After deduplication: {n_after_dedup} (-{n_after_type - n_after_dedup})")

    _errs = df_sessions['logged_errors']
    _qc_blockers = {
        'MissingRawData', 'MissingExtractedData', 'InsufficientTrials',
        'TrialsNotInPhotometryTime', 'QCValidationError'
    }
    df_sessions['passes_qc'] = _errs.apply(
        lambda errs: not any(err in _qc_blockers for err in errs)
    )
    df = df_sessions[df_sessions['passes_qc']].copy()
    n_after_errs = len(df)
    print(f"  After QC: {n_after_errs} (-{n_after_dedup - n_after_errs})")

    # =========================================================================
    # Step 2: Merge video QC columns
    # =========================================================================

    print(f"\nLoading QC from {SESSIONS_QC_FPATH}")
    df_qc = pd.read_parquet(SESSIONS_QC_FPATH)
    qc_cols = ['eid'] + VIDEO_QC_QUALITY_COLS + VIDEO_QC_PROBLEM_COLS
    df = df.merge(df_qc[qc_cols], on='eid', how='left')
    print(f"  Merged QC data ({len(qc_cols) - 1} QC columns)")

    # =========================================================================
    # Step 3: Download camera timestamps
    # =========================================================================

    print(f"\nDownloading leftCamera.times for {len(df)} sessions...")
    one = _get_default_connection()

    df['video_t0'] = np.nan
    df['video_t1'] = np.nan
    df['framerate_from_tpts'] = np.nan
    df['n_tpts'] = np.nan
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        times = None
        try:
            times = one.load_dataset(row['eid'], '*leftCamera.times*')
        except Exception as e:
            error_log.append(make_log_entry(
                row['eid'],
                error_type='MissingVideoTimestamps',
                error_message=str(e),
            ))
        if times is not None:
            df.at[idx, 'video_t0'] = times[0]
            df.at[idx, 'video_t1'] = times[-1]
            df.at[idx, 'framerate_from_tpts'] = np.median(np.diff(times))
            df.at[idx, 'n_tpts'] = len(times)


    df['video_length'] = df['video_t1'] - df['video_t0']
    df['length_discrepancy'] = df['video_length'] - df['session_length']

    n_valid_video = df['video_length'].notna().sum()
    n_download_errors = df['video_length'].isna().sum()
    print(f"  Downloaded: {n_valid_video} successful, {n_download_errors} errors")

    # =========================================================================
    # Step 4: Validate video QC and length
    # =========================================================================

    print("\nValidating video data...")
    df.apply(validate_video_length, axis='columns', exlog=error_log)
    df.apply(validate_video_timestamps_qc, axis='columns', exlog=error_log)
    df.apply(validate_video_dropped_frames_qc, axis='columns', exlog=error_log)
    df.apply(validate_video_pin_state_qc, axis='columns', exlog=error_log)

    # Collect eids with errors
    error_eids = {entry['eid'] for entry in error_log}
    n_errors = len(error_eids)
    print(f"  Sessions with errors: {n_errors}")

    # Print error breakdown
    from collections import Counter
    error_counts = Counter(entry['error_type'] for entry in error_log)
    for err_type, count in error_counts.most_common():
        print(f"    {err_type}: {count}")

    # =========================================================================
    # Step 5: Score and batch
    # =========================================================================

    print("\nScoring and batching sessions...")

    # Compute video_qc_score for all sessions
    for col in VIDEO_QC_QUALITY_COLS:
        df[col + '_num'] = df[col].map(QCVAL2NUM)

    num_cols = [col + '_num' for col in VIDEO_QC_QUALITY_COLS]
    df['video_qc_score'] = np.nanmean(df[num_cols], axis=1)

    # Sessions with errors get score = -1
    df.loc[df['eid'].isin(error_eids), 'video_qc_score'] = -1

    n_good = (df['video_qc_score'] > 0).sum()
    print(f"  Good sessions (score > 0): {n_good}")

    # # Batch good sessions (score > 0) with KMeans
    # good_mask = df['video_qc_score'] > 0
    # df['batch'] = np.nan
    # df_good = df[good_mask]
    #
    # if n_good > 0:
    #     k = max(1, n_good // 250)
    #     km = KMeans(n_clusters=k, n_init='auto', random_state=42)
    #     labels = km.fit_predict(df_good[['video_qc_score']])
    #     centroids = km.cluster_centers_.flatten()
    #     rank = np.argsort(-centroids)
    #     label_to_batch = {label: batch for batch, label in enumerate(rank)}
    #     df.loc[good_mask, 'batch'] = pd.Series(
    #         labels, index=df_good.index,
    #     ).map(label_to_batch)

    # Sort: video_qc_score descending, then session_type order
    df['_sort_type'] = df['session_type'].map(SESSION_TYPE_ORDER)
    df = df.sort_values(
        ['video_qc_score', '_sort_type'], ascending=[False, True],
    ).reset_index(drop=True)
    df = df.drop(columns='_sort_type')

    # Drop intermediate numeric columns
    df = df.drop(columns=num_cols)

    # =========================================================================
    # Step 6: Save and summarize
    # =========================================================================

    output_cols = (
        ['eid', 'subject', 'session_type', 'video_qc_score']
        + VIDEO_QC_QUALITY_COLS
        + VIDEO_QC_PROBLEM_COLS
        + ['session_length', 'video_length', 'length_discrepancy',
           'framerate_from_tpts']
    )

    df_out = df[output_cols].copy()

    VIDEO_FPATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(VIDEO_FPATH, index=False)

    # Save error log
    df_log = (pd.DataFrame(error_log) if error_log
              else pd.DataFrame(columns=LOG_COLUMNS))
    df_log.to_parquet(VIDEO_LOG_FPATH, index=False)

    # Final summary
    print(f"\n{'='*50}")
    print("Summary")
    print(f"{'='*50}")
    print(f"Total sessions: {n_total}")
    print(f"  After type/subject filter: {n_after_type}")
    print(f"  After deduplication: {n_after_dedup}")
    print(f"  After upstream QC: {n_after_errs}")
    print(f"  Video times downloaded: {n_valid_video}, failed: {n_download_errors}")
    print(f"  Sessions with video errors: {n_errors}")
    print(f"  Good (score > 0): {n_good}")
    print(f"  Total output: {len(df_out)} sessions")
    print(f"\nSaved: {VIDEO_FPATH}")
    print(f"Saved: {VIDEO_LOG_FPATH}")
