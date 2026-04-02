"""
Dataset Overview

Produces session overview matrices and target barplots from the session catalog.

Plots 1–4: session matrices (all, raw data, complete, QC-passing)
Plot 5: video-ready matrix (if VIDEO_FPATH exists)
Plots 6–7: target/mouse barplots (QC-passing, VALID_TARGETNMS)
Plots 8–9: target/mouse barplots (QC-passing, TARGETNMS_TO_ANALYZE)
Recording capacity projection

Usage:
    python scripts/dataset_overview.py [--session_split {session_type,proficient}]
                                       [--session_x {session_n,day_n}]
                                       [--horizontal]
"""
import argparse
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from iblnm.config import (
    PROJECT_ROOT, SESSIONS_FPATH, FIGURE_DPI,
    QUERY_DATABASE_LOG_FPATH, PHOTOMETRY_LOG_FPATH, TASK_LOG_FPATH,
    PERFORMANCE_FPATH, ERRORS_FPATH,
    SESSION_TYPES_TO_ANALYZE, VALID_TARGETNMS,
    TARGETNMS_TO_ANALYZE, VIDEO_FPATH,
    SESSIONTYPE2FLOAT, SESSIONTYPE2COLOR,
    MIN_TRAINING_PERFORMANCE, REQUIRED_CONTRASTS,
)
from iblnm.data import PhotometrySessionGroup
from iblnm.util import (
    concat_logs, deduplicate_log, collect_session_errors,
)
from iblnm.vis import (
    session_overview_matrix, target_overview_barplot, mouse_overview_barplot, set_plotsize,
)

# ---- Color maps for --session_split options ----

SESSION_TYPE_FLOAT_MAP = SESSIONTYPE2FLOAT
SESSION_TYPE_COLOR_MAP = SESSIONTYPE2COLOR

PROFICIENT_FLOAT_MAP = {
    'not_proficient': 0.33,
    'proficient': 0.80,
}
PROFICIENT_COLOR_MAP = {
    'not_proficient': 'cornflowerblue',
    'proficient': 'hotpink',
}

# ---- Error filter sets ----

RAW_DATA_BLOCKERS = {'MissingRawData'}
COMPLETE_DATA_BLOCKERS = RAW_DATA_BLOCKERS | {
    'MissingExtractedData', 'InsufficientTrials', 'TrialsNotInPhotometryTime',
}
QC_BLOCKERS = COMPLETE_DATA_BLOCKERS | {'QCValidationError', 'FewUniqueSamples'}
VIDEO_QC_COLS = [
    'qc_videoLeft_timestamps', 'qc_videoLeft_dropped_frames', 'qc_videoLeft_pin_state',
]
VIDEO_QC_BLOCKERS = QC_BLOCKERS | {'VideoQCFail'}

PROJECTION_DEADLINE = date(2026, 7, 31)
PROJECTION_CAPACITY_PER_DAY = 16
PROJECTION_TARGET_N = 3  # proficient sessions per mouse

LOG_FPATHS = [QUERY_DATABASE_LOG_FPATH, PHOTOMETRY_LOG_FPATH, TASK_LOG_FPATH]


# ---- Helper functions ----

def _compute_proficient_label(df):
    """Binary proficiency label. Requires fraction_correct and contrasts columns.

    Training sessions meeting the performance threshold and full contrast set are
    'proficient'. Biased and ephys sessions are always 'proficient'. All other
    training sessions are 'not_proficient'.
    """
    required = set(REQUIRED_CONTRASTS)
    is_training = df['session_type'] == 'training'
    has_full_contrasts = df['contrasts'].apply(
        lambda c: set(c) == required if isinstance(c, (list, np.ndarray)) else False
    )
    meets_perf = df['fraction_correct'].fillna(0) >= MIN_TRAINING_PERFORMANCE
    is_proficient = (~is_training) | (is_training & has_full_contrasts & meets_perf)
    return is_proficient.map({True: 'proficient', False: 'not_proficient'})


def _count_weekdays(start, end):
    """Count business days between start and end (exclusive of end)."""
    total = (end - start).days
    if total <= 0:
        return 0
    weeks, rem = divmod(total, 7)
    wd = weeks * 5
    cur = start + timedelta(days=weeks * 7)
    for _ in range(rem):
        if cur.weekday() < 5:
            wd += 1
        cur += timedelta(days=1)
    return wd


# ---- argparse ----

parser = argparse.ArgumentParser()
parser.add_argument('--session_split', choices=['session_type', 'proficient'],
                    default='session_type')
parser.add_argument('--session_x', choices=['session_n', 'day_n'],
                    default='session_n')
parser.add_argument('--horizontal', action='store_true',
                    help='Draw barplots with horizontal bars')
args = parser.parse_args()

if args.session_split == 'proficient':
    if not PERFORMANCE_FPATH.exists():
        print(f"Performance file not found: {PERFORMANCE_FPATH}")
        sys.exit(1)
    split_col = 'proficient_label'
    float_map = PROFICIENT_FLOAT_MAP
    color_map = PROFICIENT_COLOR_MAP
else:
    split_col = 'session_type'
    float_map = SESSION_TYPE_FLOAT_MAP
    color_map = SESSION_TYPE_COLOR_MAP

# ---- Output directory ----

figures_dir = PROJECT_ROOT / 'figures/dataset_overview'
figures_dir.mkdir(parents=True, exist_ok=True)

plt.ion()

# ---- Load and enrich ----

if not SESSIONS_FPATH.exists():
    print(f"Sessions file not found: {SESSIONS_FPATH}")
    sys.exit(1)

df = pd.read_parquet(SESSIONS_FPATH)
df = collect_session_errors(df, LOG_FPATHS)
if PERFORMANCE_FPATH.exists():
    perf = pd.read_parquet(PERFORMANCE_FPATH, columns=['eid', 'fraction_correct', 'contrasts'])
    df = df.merge(perf, on='eid', how='left')

if 'fraction_correct' in df.columns and 'contrasts' in df.columns:
    df['proficient_label'] = _compute_proficient_label(df)

# Merge video QC flags if available
if VIDEO_FPATH.exists():
    df_video = pd.read_parquet(VIDEO_FPATH, columns=['eid'] + VIDEO_QC_COLS)
    df = df.merge(df_video, on='eid', how='left')
    df['_passes_video_qc'] = (
        df['qc_videoLeft_timestamps'].eq('PASS')
        & df['qc_videoLeft_dropped_frames'].eq('PASS')
        & df['qc_videoLeft_pin_state'].isin(['PASS', 'WARNING'])
    )
    df['_video_qc_blocker'] = df['_passes_video_qc'].apply(
        lambda p: [] if p else ['VideoQCFail']
    )
    df['logged_errors'] = df.apply(
        lambda row: row['logged_errors'] + row['_video_qc_blocker']
        if isinstance(row['logged_errors'], list) else row['_video_qc_blocker'],
        axis=1,
    )
    df = df.drop(columns=['_passes_video_qc', '_video_qc_blocker'])

group = PhotometrySessionGroup.from_catalog(df, one=None)
dedup_errors = group.deduplicate()

# ---- Base filter kwargs ----

_base = dict(
    session_types=SESSION_TYPES_TO_ANALYZE,
    min_performance=False,
    required_contrasts=False,
)
xcol = args.session_x

# =============================================================================
# Session Overview Matrices
# =============================================================================

print("\nGenerating session matrices...")

# The catalog for each matrix is derived from structural filters only (no QC).
# QC filters are then applied to determine which sessions are shown at 100% opacity.
print("\n[Matrix universe: structural filters, no QC]")
group.filter_sessions(**_base, qc_blockers=set(), targetnms=VALID_TARGETNMS)
df_all = group.sessions.copy()
grp = PhotometrySessionGroup.from_catalog(df_all, one=None)

print("\n[Plot 1: All sessions]")
grp.filter_sessions(session_types=False, qc_blockers=set(), targetnms=False,
                    min_performance=False, required_contrasts=False)
ax = session_overview_matrix(grp, columns=xcol, color_by=split_col,
                             split_float_map=float_map, split_color_map=color_map)
ax.set_title(f'All sessions (n={len(grp.sessions)})')
set_plotsize(w=48, h=32, ax=ax)
ax.get_figure().savefig(figures_dir / '1_all_sessions.svg', dpi=FIGURE_DPI, bbox_inches='tight')
plt.close('all')

print("\n[Plot 2: Raw data]")
grp.filter_sessions(session_types=False, qc_blockers=RAW_DATA_BLOCKERS, targetnms=False,
                    min_performance=False, required_contrasts=False)
ax = session_overview_matrix(grp, columns=xcol, color_by=split_col,
                             split_float_map=float_map, split_color_map=color_map)
ax.set_title(f'Raw data (n={len(grp.sessions)})')
set_plotsize(w=48, h=32, ax=ax)
ax.get_figure().savefig(figures_dir / '2_raw_data.svg', dpi=FIGURE_DPI, bbox_inches='tight')
plt.close('all')

print("\n[Plot 3: Complete data]")
grp.filter_sessions(session_types=False, qc_blockers=COMPLETE_DATA_BLOCKERS, targetnms=False,
                    min_performance=False, required_contrasts=False)
ax = session_overview_matrix(grp, columns=xcol, color_by=split_col,
                             split_float_map=float_map, split_color_map=color_map)
ax.set_title(f'Complete data (n={len(grp.sessions)})')
set_plotsize(w=48, h=32, ax=ax)
ax.get_figure().savefig(figures_dir / '3_complete_data.svg', dpi=FIGURE_DPI, bbox_inches='tight')
plt.close('all')

print("\n[Plot 4: QC-passing]")
grp.filter_sessions(session_types=False, qc_blockers=QC_BLOCKERS, targetnms=False,
                    min_performance=False, required_contrasts=False)
ax = session_overview_matrix(grp, columns=xcol, color_by=split_col,
                             split_float_map=float_map, split_color_map=color_map)
ax.set_title(f'QC-passing (n={len(grp.sessions)})')
set_plotsize(w=48, h=32, ax=ax)
ax.get_figure().savefig(figures_dir / '4_passes_qc.svg', dpi=FIGURE_DPI, bbox_inches='tight')
plt.close('all')

if VIDEO_FPATH.exists():
    print("\n[Plot 5: Video-ready]")
    grp.filter_sessions(session_types=False, qc_blockers=VIDEO_QC_BLOCKERS, targetnms=False,
                        min_performance=False, required_contrasts=False)
    ax = session_overview_matrix(grp, columns=xcol, color_by=split_col,
                                 split_float_map=float_map, split_color_map=color_map)
    ax.set_title(f'Video-ready (n={len(grp.sessions)})')
    set_plotsize(w=48, h=32, ax=ax)
    ax.get_figure().savefig(figures_dir / '5_video_ready.svg', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close('all')

# =============================================================================
# Target/Mouse Barplots
# =============================================================================

# QC-passing, all valid targets
print("\n[Plots 6-7: Barplots — QC-passing, valid targets]")
grp.filter_sessions(session_types=False, qc_blockers=QC_BLOCKERS, targetnms=VALID_TARGETNMS,
                    min_performance=False, required_contrasts=False)
n_rec = len(grp.recordings)
n_ses = grp.sessions['eid'].nunique()

if n_rec > 0:
    ax = target_overview_barplot(grp.recordings, color_by=split_col, split_color_map=color_map,
                                horizontal=args.horizontal)
    ax.set_title(f'Recordings by target ({n_rec} recordings, {n_ses} sessions)')
    set_plotsize(w=24, h=12, ax=ax)
    ax.get_figure().savefig(figures_dir / '6_target_overview.svg',
                            dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close('all')

    ax = mouse_overview_barplot(grp.recordings, min_sessions=1,
                                color_by=split_col, split_color_map=color_map,
                                horizontal=args.horizontal)
    ax.set_title(f'Mice by target ({n_rec} recordings, {n_ses} sessions)')
    set_plotsize(w=24, h=12, ax=ax)
    ax.get_figure().savefig(figures_dir / '7_mouse_overview.svg',
                            dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close('all')

# QC-passing, analysis-ready targets only
print("\n[Plots 8-9: Barplots — QC-passing, analysis targets]")
grp.filter_sessions(session_types=False, qc_blockers=QC_BLOCKERS, targetnms=TARGETNMS_TO_ANALYZE,
                    min_performance=False, required_contrasts=False)
n_rec_a = len(grp.recordings)
n_ses_a = grp.sessions['eid'].nunique()

if n_rec_a > 0:
    ax = target_overview_barplot(grp.recordings, color_by=split_col, split_color_map=color_map,
                                horizontal=args.horizontal)
    ax.set_title(f'Analysis-ready by target ({n_rec_a} recordings, {n_ses_a} sessions)')
    set_plotsize(w=24, h=12, ax=ax)
    ax.get_figure().savefig(figures_dir / '8_target_overview_analyze.svg',
                            dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close('all')

    ax = mouse_overview_barplot(grp.recordings, min_sessions=1,
                                color_by=split_col, split_color_map=color_map,
                                horizontal=args.horizontal)
    ax.set_title(f'Mice (analysis-ready, n={n_ses_a} sessions)')
    set_plotsize(w=24, h=12, ax=ax)
    ax.get_figure().savefig(figures_dir / '9_mouse_overview_analyze.svg',
                            dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close('all')

# =============================================================================
# Recording Capacity Projection
# =============================================================================

_proj_base = dict(
    lab='mainenlab',
    start_time_min='2024-01-01',
    session_types=SESSION_TYPES_TO_ANALYZE,
    targetnms=VALID_TARGETNMS,
    min_performance=False,
    required_contrasts=False,
)

print("\n[Capacity projection: all sessions]")
group.filter_sessions(**_proj_base, qc_blockers=set())
n_total_mice = group.sessions['subject'].nunique()
total_sessions = len(group.sessions)

print("\n[Capacity projection: QC-passing]")
group.filter_sessions(**_proj_base, qc_blockers=QC_BLOCKERS)
df_qc = group.sessions

if 'proficient_label' in df_qc.columns:
    is_proficient = df_qc['proficient_label'] == 'proficient'
else:
    is_proficient = df_qc['session_type'].isin({'biased', 'ephys'})

df_proj_proficient = df_qc[is_proficient]
n_proficient_per_subject = df_proj_proficient.groupby('subject')['eid'].nunique()
subjects_reached = n_proficient_per_subject[
    n_proficient_per_subject >= PROJECTION_TARGET_N].index

n_reached = len(subjects_reached)
effective_sessions = total_sessions / n_reached if n_reached > 0 else float('nan')

weekdays = _count_weekdays(date.today(), PROJECTION_DEADLINE)
total_slots = weekdays * PROJECTION_CAPACITY_PER_DAY
n_mice_projected = int(total_slots / effective_sessions) if effective_sessions > 0 else 0

print(f"\n{'='*70}")
print("Recording Capacity Projection")
print(f"{'='*70}")
print(f"Mainenlab mice (started 2024+): {n_total_mice}")
print(f"Reached >={PROJECTION_TARGET_N} proficient sessions: {n_reached}/{n_total_mice}")
print(f"Effective sessions/mouse: {effective_sessions:.1f}")
print(f"Deadline: {PROJECTION_DEADLINE}, capacity: {PROJECTION_CAPACITY_PER_DAY}/day")
print(f"Weekdays available: {weekdays}")
print(f"Total slots: {total_slots}")
print(f"Projected mice reaching target: {n_mice_projected}")


# =============================================================================
# Save unified error log
# =============================================================================

upstream_logs = [pd.read_parquet(p) for p in LOG_FPATHS if p.exists()] + [dedup_errors]
df_errors = deduplicate_log(concat_logs(upstream_logs))
ERRORS_FPATH.parent.mkdir(parents=True, exist_ok=True)
df_errors.to_parquet(ERRORS_FPATH)
print(f"\nSaved {len(df_errors)} error entries to {ERRORS_FPATH}")
print(f"Figures saved to {figures_dir}")
