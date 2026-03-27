"""
Dataset Overview

Produces six session overview matrices:
1. All registered sessions (post-metadata and session-type filtering)
2. Sessions with raw data (task + photometry)
3. Sessions with complete data (extracted + sufficient trials + TIPT)
4. Sessions passing basic photometry QC
5. Video-ready sessions (extracted task data + video QC pass)
6. QC-passing sessions only, exploded by target, sorted by target then first session date

All flags are derived from upstream error logs and video.pqt.
No write-back to any upstream file.
"""
import pandas as pd
from matplotlib import pyplot as plt

from iblnm.config import (
    PROJECT_ROOT, SESSIONS_FPATH,
    QUERY_DATABASE_LOG_FPATH, PHOTOMETRY_LOG_FPATH,
    TASK_LOG_FPATH,
    ERRORS_FPATH, FIGURE_DPI,
    SUBJECTS_TO_EXCLUDE, SESSION_TYPES_TO_ANALYZE, VALID_TARGETNMS,
    TARGETNMS_TO_ANALYZE, VIDEO_FPATH,
)
from iblnm.util import (
    resolve_duplicate_group,
    concat_logs, deduplicate_log,
    collect_session_errors, derive_target_nm, LOG_COLUMNS,
)
from iblnm.vis import (
    session_overview_matrix, target_overview_barplot, mouse_overview_barplot, set_plotsize
)

plt.ion()

# Create output directory
figures_dir = PROJECT_ROOT / 'figures/dataset_overview'
figures_dir.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Load Data
# =============================================================================

print(f"Loading sessions from {SESSIONS_FPATH}")
df_sessions = pd.read_parquet(SESSIONS_FPATH)
n_total = len(df_sessions)

# Attach all upstream error logs (needed for filtering and dedup)
df_sessions = collect_session_errors(
    df_sessions,
    [QUERY_DATABASE_LOG_FPATH, PHOTOMETRY_LOG_FPATH, TASK_LOG_FPATH],
)

# Filter: remove sessions with fatal metadata errors
# ~fatal_errors = {'InvalidNeuromodulator', 'InvalidBrainRegion', 'InvalidTargetNM'}
# ~df_sessions = df_sessions[
    # ~df_sessions['logged_errors'].apply(lambda errs: not any(e in fatal_errors for e in errs))
# ~].copy()
# ~n_after_fatal = len(df_sessions)


# Filter to sessions that are valid for task analysis
df_sessions = df_sessions[
    df_sessions['session_type'].isin(SESSION_TYPES_TO_ANALYZE) &
    ~df_sessions['subject'].isin(SUBJECTS_TO_EXCLUDE)
]
n_after_type = len(df_sessions)

# Deduplicate: one session per (subject, day_n), prefer sessions without disqualifying errors
dup_log = []
df_sessions = (
    df_sessions.groupby(['subject', 'day_n'], group_keys=False)
    .apply(resolve_duplicate_group, exlog=dup_log, include_groups=True)
    .reset_index(drop=True)
)
df_dup_log = pd.DataFrame(dup_log) if dup_log else pd.DataFrame(columns=LOG_COLUMNS)
n_after_dedup = len(df_sessions)

print("\nSession counts:")
print(f"  Registered: {n_total}")
print(f"  After session type filter: {n_after_type} (-{n_total - n_after_type})")
print(f"  After deduplication: {n_after_dedup} (-{n_after_type - n_after_dedup})")


# =============================================================================
# Derive flags from error logs
# =============================================================================

_errs = df_sessions['logged_errors']

# Matrix 2: raw data present (task + photometry)
df_sessions['has_raw_data'] = _errs.apply(
    lambda e: 'MissingRawData' not in e)

# Matrix 3: complete data (raw data present + extracted + sufficient trials + TIPT)
# MissingRawData must be included: photometry.py exits early when raw data is absent,
# so MissingExtractedData / InsufficientTrials / TrialsNotInPhotometryTime are never
# logged for those sessions — omitting it causes complete_data > raw_data (wrong).
_complete_blockers = {'MissingRawData', 'MissingExtractedData', 'InsufficientTrials',
                      'TrialsNotInPhotometryTime'}
df_sessions['has_complete_data'] = _errs.apply(
    lambda e: not any(err in _complete_blockers for err in e))

# Matrix 4: passes basic photometry QC
_qc_blockers = _complete_blockers | {'QCValidationError', 'FewUniqueSamples'}
df_sessions['passes_basic_qc'] = _errs.apply(
    lambda e: not any(err in _qc_blockers for err in e))

# Matrix 5: video-ready (extracted task data + video QC pass)
_extraction_blockers = {'MissingRawData', 'MissingExtractedData'}
df_sessions['has_extracted_task'] = _errs.apply(
    lambda e: not any(err in _extraction_blockers for err in e))

_VIDEO_QC_PROBLEM_COLS = [
    'qc_videoLeft_timestamps',
    'qc_videoLeft_dropped_frames',
    'qc_videoLeft_pin_state',
]
df_video = pd.read_parquet(VIDEO_FPATH, columns=['eid'] + _VIDEO_QC_PROBLEM_COLS)
df_sessions = df_sessions.merge(df_video, on='eid', how='left')
df_sessions['passes_video_qc'] = (
    df_sessions['qc_videoLeft_timestamps'].eq('PASS')
    & df_sessions['qc_videoLeft_dropped_frames'].eq('PASS')
    & df_sessions['qc_videoLeft_pin_state'].isin(['PASS', 'WARNING'])
)
df_sessions['video_ready'] = df_sessions['has_extracted_task'] & df_sessions['passes_video_qc']


# =============================================================================
# Session Overview Matrices
# =============================================================================

print("\nGenerating session matrices...")

n_raw = df_sessions['has_raw_data'].sum()
n_complete = df_sessions['has_complete_data'].sum()
n_qc = df_sessions['passes_basic_qc'].sum()
n_video = df_sessions['video_ready'].sum()


def _save_matrix(df, highlight, title, filename):
    ax = session_overview_matrix(df, highlight=highlight)
    ax.set_title(title)
    set_plotsize(w=48, h=32, ax=ax)
    ax.get_figure().savefig(figures_dir / filename, dpi=FIGURE_DPI, bbox_inches='tight')
    return ax


_save_matrix(df_sessions, 'all',
             f'All registered sessions (n={n_after_dedup})',
             '1_all_sessions.svg')

_save_matrix(df_sessions, lambda df: df['has_raw_data'],
             f'Sessions with raw data (n={n_raw})',
             '2_raw_data.svg')

_save_matrix(df_sessions, lambda df: df['has_complete_data'],
             f'Sessions with complete data (n={n_complete})',
             '3_complete_data.svg')

_save_matrix(df_sessions, lambda df: df['passes_basic_qc'],
             f'Sessions passing basic QC (n={n_qc})',
             '4_passes_qc.svg')

_save_matrix(df_sessions, lambda df: df['video_ready'],
             f'Video-ready sessions (n={n_video})',
             '5_video_ready.svg')

# Matrix 6: QC-passing sessions only, exploded by target
df_sessions = derive_target_nm(df_sessions)
df_qc = (
    df_sessions[df_sessions['passes_basic_qc']]
    .explode(['target_NM', 'brain_region', 'hemisphere'])
    .loc[lambda df: df['target_NM'].isin(TARGETNMS_TO_ANALYZE)]
    .copy()
)
# One row per subject-NM-session (collapse bilateral fibers for the same NM)
df_qc['NM'] = df_qc['target_NM'].str.split('-').str[-1]
df_qc = df_qc.drop_duplicates(subset=['subject', 'NM', 'session_n'])
df_qc['subject'] = df_qc['subject'] + ' (' + df_qc['NM'] + ')'

# Dense sequential session index per subject (no gaps)
df_qc = df_qc.sort_values('session_n')
df_qc['_dense_session_n'] = df_qc.groupby('subject').cumcount()

# Sort by NM (preserving TARGETNMS_TO_ANALYZE order), then by first session start_time
nm_order = list(dict.fromkeys(t.split('-')[-1] for t in TARGETNMS_TO_ANALYZE))
nm_rank = {nm: i for i, nm in enumerate(nm_order)}
first_start = df_qc.groupby('subject')['start_time'].min()
subject_order = (
    df_qc[['subject', 'NM']].drop_duplicates()
    .assign(
        _nm_rank=lambda df: df['NM'].map(nm_rank),
        _first_start=lambda df: df['subject'].map(first_start),
    )
    .sort_values(['_nm_rank', '_first_start'])
    ['subject'].tolist()
)

n_qc_sessions = df_qc['eid'].nunique()
n_qc_mice = df_qc['subject'].str.extract(r'^(.+?) \(')[0].nunique()
ax = session_overview_matrix(df_qc, columns='_dense_session_n', highlight='all',
                             subject_order=subject_order)
ax.set_title(f'QC-passing sessions by target (n={n_qc_sessions})', fontsize=28)
ax.set_xlabel('Sessions', fontsize=28)
ax.set_ylabel('Mice', fontsize=28, labelpad=60)
ax.tick_params(axis='x', labelsize=28)
cbar = ax.images[0].colorbar
if cbar is not None:
    cbar.ax.tick_params(labelsize=28)
ax.text(1.0, -0.02, f'{n_qc_sessions} sessions, {n_qc_mice} mice',
        transform=ax.transAxes, ha='right', va='top', fontsize=28)

# Replace subject labels with one NM label per group
ax.set_yticks([])
nm_subjects = {nm: [] for nm in nm_order}
for i, subj in enumerate(subject_order):
    nm = subj.rsplit('(', 1)[-1].rstrip(')')
    nm_subjects[nm].append(i)
for nm, rows in nm_subjects.items():
    if rows:
        mid = (rows[0] + rows[-1]) / 2
        ax.text(-0.01, mid, nm, transform=ax.get_yaxis_transform(),
                ha='right', va='center', fontsize=28)
        if rows[0] > 0:
            ax.axhline(rows[0] - 0.5, color='black', linewidth=1.5)

set_plotsize(w=48, h=32, ax=ax)
ax.get_figure().savefig(figures_dir / '6_qc_sessions_by_target.svg',
                        dpi=FIGURE_DPI, bbox_inches='tight')


# =============================================================================
# Dataset Summary
# =============================================================================

print(f"\n{'='*50}")
print("Dataset Summary")
print(f"{'='*50}")
print(f"Total sessions (after filtering): {n_after_dedup}")
print(f"  Has raw data: {n_raw}")
print(f"  Has complete data: {n_complete}")
print(f"  Passes basic QC: {n_qc}")
print(f"  Video-ready: {n_video}")


# =============================================================================
# Target-NM Barplots (per recording, QC-passing sessions only)
# =============================================================================

# Explode to one row per recording (brain_region / target_NM are parallel lists)
df_recordings = (
    df_sessions[df_sessions['passes_basic_qc']]
    .explode(['target_NM', 'brain_region', 'hemisphere'])
    .loc[lambda df: df['target_NM'].isin(VALID_TARGETNMS)]
    .copy()
)

if len(df_recordings) > 0:
    n_rec = len(df_recordings)
    n_ses = df_recordings['eid'].nunique()

    ax = target_overview_barplot(df_recordings)
    ax.set_title(f'Complete recordings by target ({n_rec} recordings, {n_ses} sessions)')
    set_plotsize(w=24, h=12, ax=ax)
    ax.get_figure().savefig(figures_dir / '7_target_overview.svg',
                            dpi=FIGURE_DPI, bbox_inches='tight')

    ax = mouse_overview_barplot(df_recordings)
    ax.set_title(f'Mice by target ({n_rec} recordings, {n_ses} sessions)')
    set_plotsize(w=24, h=12, ax=ax)
    ax.get_figure().savefig(figures_dir / '8_mouse_overview.svg',
                            dpi=FIGURE_DPI, bbox_inches='tight')

    df_rec_analyze = df_recordings[df_recordings['target_NM'].isin(TARGETNMS_TO_ANALYZE)]
    n_rec_a = len(df_rec_analyze)
    n_ses_a = df_rec_analyze['eid'].nunique()
    ax = mouse_overview_barplot(df_rec_analyze, min_biased_ephys=1, min_ephys=1)
    ax.set_title(f'Mice by target, ≥1 session ({n_rec_a} recordings, {n_ses_a} sessions)',
                 fontsize=28)
    ax.set_xlabel(ax.get_xlabel(), fontsize=28)
    ax.set_ylabel(ax.get_ylabel(), fontsize=28)
    ax.tick_params(axis='both', labelsize=28)
    ax.legend(fontsize=28)
    for t in ax.texts:
        t.set_fontsize(14)
    set_plotsize(w=24, h=12, ax=ax)
    ax.get_figure().savefig(figures_dir / '9_mouse_overview_min1.svg',
                            dpi=FIGURE_DPI, bbox_inches='tight')

    ax = target_overview_barplot(df_rec_analyze)
    ax.set_title(ax.get_title(), fontsize=28)
    ax.set_xlabel(ax.get_xlabel(), fontsize=28)
    ax.set_ylabel(ax.get_ylabel(), fontsize=28)
    ax.tick_params(axis='both', labelsize=28)
    ax.legend(fontsize=28)
    for t in ax.texts:
        t.set_fontsize(14)
    set_plotsize(w=24, h=12, ax=ax)
    ax.get_figure().savefig(figures_dir / '10_target_overview_analyze.svg',
                            dpi=FIGURE_DPI, bbox_inches='tight')
else:
    print("No complete recordings to plot.")


# =============================================================================
# Save unified error log
# =============================================================================

upstream_logs = [pd.read_parquet(p) for p in
                 [QUERY_DATABASE_LOG_FPATH, PHOTOMETRY_LOG_FPATH, TASK_LOG_FPATH]
                 if p.exists()]
df_errors = deduplicate_log(concat_logs(upstream_logs + [df_dup_log]))

ERRORS_FPATH.parent.mkdir(parents=True, exist_ok=True)
df_errors.to_parquet(ERRORS_FPATH)
print(f"\nSaved {len(df_errors)} error entries to {ERRORS_FPATH}")
print(f"Figures saved to {figures_dir}")
