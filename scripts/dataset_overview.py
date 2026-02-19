"""
Dataset Overview

Produces session overview matrices at each stage of data processing:
1. All registered sessions
2. Sessions with raw data (task | photometry)
3. Sessions with extracted data (task | photometry)
4. Sessions passing QC (basic photometry QC | trials in photometry time)

Barplots evaluate each brain region independently against QC criteria,
while session-level checks (extracted task, trials_in_photometry_time) are applied uniformly.

Joins sessions.pqt + qc_photometry.pqt + performance.pqt + photometry_log.pqt at read time.
No write-back to any upstream file.
"""
import pandas as pd
from matplotlib import pyplot as plt

from iblnm.config import (
    PROJECT_ROOT, SESSIONS_FPATH,
    QCPHOTOMETRY_FPATH,
    PERFORMANCE_FPATH,
    QUERY_DATABASE_LOG_FPATH, PHOTOMETRY_LOG_FPATH,
    TASK_LOG_FPATH,
    ERRORS_FPATH, FIGURE_DPI, VALID_TARGETS,
)
from iblnm.util import (
    clean_sessions, drop_junk_duplicates,
    aggregate_qc_per_session, concat_logs, deduplicate_log, make_log_entry,
    collect_session_errors, LOG_COLUMNS,
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

# Clean sessions
df_sessions = clean_sessions(df_sessions)
n_after_clean = len(df_sessions)

# Recompute session_n so all animals start at session 1 after filtering
df_sessions['session_n'] = df_sessions.groupby('subject')['start_time'].rank(method='dense')

# Drop duplicates
df_sessions = drop_junk_duplicates(df_sessions, ['subject', 'day_n'])
n_after_dedup = len(df_sessions)

print(f"\nSession counts:")
print(f"  Registered: {n_total}")
print(f"  After cleaning: {n_after_clean} (-{n_total - n_after_clean})")
print(f"  After deduplication: {n_after_dedup} (-{n_after_clean - n_after_dedup})")

# Load QC results
df_qc = None
qc_agg = None
if QCPHOTOMETRY_FPATH.exists():
    print(f"\nLoading QC results from {QCPHOTOMETRY_FPATH}")
    df_qc = pd.read_parquet(QCPHOTOMETRY_FPATH)
    qc_agg = aggregate_qc_per_session(df_qc, require_all=True)

# Load performance data
df_perf = None
if PERFORMANCE_FPATH.exists():
    print(f"Loading performance from {PERFORMANCE_FPATH}")
    perf_cols = ['eid', 'n_trials', 'has_block_bug',
                 'has_goCue_times', 'has_firstMovement_times', 'has_feedback_times']
    df_perf_full = pd.read_parquet(PERFORMANCE_FPATH)
    available_perf_cols = [c for c in perf_cols if c in df_perf_full.columns]
    df_perf = df_perf_full[available_perf_cols]

# Load upstream error logs and attach to df_sessions
upstream_logs = []
for name, path in [('query_database', QUERY_DATABASE_LOG_FPATH),
                    ('photometry', PHOTOMETRY_LOG_FPATH),
                    ('task', TASK_LOG_FPATH)]:
    if path.exists():
        df_log = pd.read_parquet(path)
        upstream_logs.append(df_log)
        print(f"Loaded {len(df_log)} errors from {name}")

df_sessions = collect_session_errors(df_sessions, upstream_logs)

# Determine trials_in_photometry_time from logged_errors
# Sessions with QC results that don't have a TrialsNotInPhotometryTime error pass
eids_with_photometry = set(df_qc['eid'].unique()) if df_qc is not None else set()
eids_tipt_failed = set(
    df_sessions[df_sessions['logged_errors'].apply(
        lambda errs: 'TrialsNotInPhotometryTime' in errs
    )]['eid']
)
eids_tipt_ok = eids_with_photometry - eids_tipt_failed

# Build session-level filter DataFrame
df_filters = df_sessions[['eid', 'subject', 'has_raw_task', 'has_raw_photometry',
                           'has_extracted_task', 'has_extracted_photometry']].copy()

# Basic photometry QC (session-level, aggregated)
if qc_agg is not None and len(qc_agg) > 0:
    df_filters = df_filters.merge(qc_agg[['eid', 'passes_basic_qc']], on='eid', how='left')
    df_filters['passes_basic_qc'] = df_filters['passes_basic_qc'].fillna(False)
else:
    df_filters['passes_basic_qc'] = False

# Trials in photometry time
df_filters['trials_in_photometry_time'] = df_filters['eid'].isin(eids_tipt_ok)

if df_perf is not None:
    df_filters = df_filters.merge(df_perf, on='eid', how='left')


# =============================================================================
# Session Overview Matrices
# =============================================================================

def plot_paired_matrices(df, left_highlight, right_highlight, left_title, right_title, filename):
    """Plot two session matrices side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(32, 16))

    session_overview_matrix(df, highlight=left_highlight, ax=axes[0])
    axes[0].set_title(left_title)

    session_overview_matrix(df, highlight=right_highlight, ax=axes[1])
    axes[1].set_title(right_title)

    plt.tight_layout()
    fig.savefig(figures_dir / filename, dpi=FIGURE_DPI, bbox_inches='tight')
    return fig


# 1. Full dataset (all registered sessions)
print("\nGenerating session matrices...")
ax = session_overview_matrix(df_sessions, highlight='all')
ax.set_title(f'All registered sessions (n={len(df_sessions)})')
set_plotsize(w=48, h=32, ax=ax)
ax.get_figure().savefig(figures_dir / '1_all_sessions.svg', dpi=FIGURE_DPI, bbox_inches='tight')

# 2. Raw data: task | photometry
n_raw_task = df_filters['has_raw_task'].sum()
n_raw_phot = df_filters['has_raw_photometry'].sum()
plot_paired_matrices(
    df_sessions,
    left_highlight=lambda df: df['has_raw_task'],
    right_highlight=lambda df: df['has_raw_photometry'],
    left_title=f'Raw task data (n={n_raw_task})',
    right_title=f'Raw photometry data (n={n_raw_phot})',
    filename='2_raw_data.svg'
)

# 3. Extracted data: task | photometry
n_task = df_filters['has_extracted_task'].sum()
n_phot = df_filters['has_extracted_photometry'].sum()
plot_paired_matrices(
    df_sessions,
    left_highlight=lambda df: df['eid'].isin(df_filters[df_filters['has_extracted_task']]['eid']),
    right_highlight=lambda df: df['eid'].isin(df_filters[df_filters['has_extracted_photometry']]['eid']),
    left_title=f'Extracted task data (n={n_task})',
    right_title=f'Extracted photometry data (n={n_phot})',
    filename='3_extracted_data.svg'
)

# 4. QC criteria: basic photometry QC | trials_in_photometry_time
n_qc = df_filters['passes_basic_qc'].sum()
n_tipt = df_filters['trials_in_photometry_time'].sum()
plot_paired_matrices(
    df_sessions,
    left_highlight=lambda df: df['eid'].isin(df_filters[df_filters['passes_basic_qc']]['eid']),
    right_highlight=lambda df: df['eid'].isin(df_filters[df_filters['trials_in_photometry_time']]['eid']),
    left_title=f'Passes basic QC (n={n_qc})',
    right_title=f'Trials in photometry time (n={n_tipt})',
    filename='4_qc_criteria.svg'
)


# =============================================================================
# Generate Error Log Entries from Flags and QC
# =============================================================================

flag_errors = []

# Missing data flags → log entries
flag_map = {
    'has_raw_task': 'MissingRawTask',
    'has_raw_photometry': 'MissingRawPhotometry',
    'has_extracted_task': 'MissingExtractedTask',
    'has_extracted_photometry': 'MissingExtractedPhotometry',
}
for col, error_type in flag_map.items():
    missing_eids = df_filters.loc[~df_filters[col], 'eid']
    for eid in missing_eids:
        flag_errors.append(make_log_entry(eid, error_type=error_type, error_message=f'{col}=False'))

# QC metric flags → log entries (per session, not per region)
if df_qc is not None and len(df_qc) > 0:
    # BandInversion: any region with n_band_inversions > 0
    inversions = df_qc[df_qc['n_band_inversions'] > 0].groupby('eid').first()
    for eid in inversions.index:
        flag_errors.append(make_log_entry(
            eid, error_type='BandInversion',
            error_message="n_band_inversions > 0"
        ))
    # EarlySamples: any region with n_early_samples > 0
    if 'n_early_samples' in df_qc.columns:
        early = df_qc[df_qc['n_early_samples'] > 0].groupby('eid').first()
        for eid in early.index:
            flag_errors.append(make_log_entry(
                eid, error_type='EarlySamples',
                error_message="n_early_samples > 0"
            ))

df_flag_errors = pd.DataFrame(flag_errors) if flag_errors else pd.DataFrame(columns=LOG_COLUMNS)

# Concatenate all logs → unified errors.pqt (deduplicated by eid+type+message)
df_errors = deduplicate_log(concat_logs(upstream_logs + [df_flag_errors]))


# =============================================================================
# Final Dataset Summary
# =============================================================================

# Sessions passing all criteria (session-level)
complete_mask = (
    df_filters['has_extracted_task'] &
    df_filters['has_extracted_photometry'] &
    df_filters['passes_basic_qc'] &
    df_filters['trials_in_photometry_time']
)
n_complete_sessions = complete_mask.sum()

print(f"\n{'='*50}")
print("Dataset Summary")
print(f"{'='*50}")
print(f"Total sessions: {len(df_sessions)}")
print(f"  Raw task: {n_raw_task}")
print(f"  Raw photometry: {n_raw_phot}")
print(f"  Extracted task: {n_task}")
print(f"  Extracted photometry: {n_phot}")
print(f"  Passes basic QC: {n_qc}")
print(f"  Trials in photometry time: {n_tipt}")
print(f"  Complete (all criteria): {n_complete_sessions}")


# =============================================================================
# Target and Mouse Overview (per brain region, independent QC evaluation)
# =============================================================================

if df_qc is not None and len(df_qc) > 0:
    # Per-brain-region QC: use GCaMP band only
    df_qc_gcamp = df_qc[df_qc['band'] == 'GCaMP'].copy()
    df_qc_gcamp['passes_basic_qc'] = (
        (df_qc_gcamp['n_unique_samples'] > 0.1) &
        (df_qc_gcamp['n_band_inversions'] == 0)
    )

    # Join with session-level flags
    session_flags = df_filters[['eid', 'has_extracted_task', 'trials_in_photometry_time']].copy()
    df_recordings = df_qc_gcamp[['eid', 'brain_region', 'passes_basic_qc']].merge(
        session_flags, on='eid', how='left'
    )

    # A recording is complete if it passes all criteria
    df_recordings['is_complete'] = (
        df_recordings['passes_basic_qc'] &
        df_recordings['has_extracted_task'].fillna(False) &
        df_recordings['trials_in_photometry_time'].fillna(False)
    )

    # Merge session metadata for plotting (subject, session_type, NM)
    session_meta = df_sessions[['eid', 'subject', 'session_type', 'NM']].copy()
    df_recordings = df_recordings.merge(session_meta, on='eid', how='left')

    # Filter to complete recordings with valid target_NM
    df_complete_recordings = df_recordings[df_recordings['is_complete']].copy()
    if len(df_complete_recordings) > 0:
        df_complete_recordings['target_NM'] = (
            df_complete_recordings['brain_region'].str.split('-').str[0] + '-' + df_complete_recordings['NM']
        )
        df_complete_recordings = df_complete_recordings[
            df_complete_recordings['target_NM'].isin(VALID_TARGETS)
        ].copy()

        if len(df_complete_recordings) > 0:
            n_complete_recordings = len(df_complete_recordings)
            n_complete_sessions_bar = df_complete_recordings['eid'].nunique()

            # Target overview
            ax = target_overview_barplot(df_complete_recordings)
            ax.set_title(f'Complete recordings by target ({n_complete_recordings} recordings, '
                         f'{n_complete_sessions_bar} sessions)')
            set_plotsize(w=24, h=12, ax=ax)
            ax.get_figure().savefig(figures_dir / '5_target_overview.svg',
                                     dpi=FIGURE_DPI, bbox_inches='tight')

            # Mouse overview
            ax = mouse_overview_barplot(df_complete_recordings)
            ax.set_title(f'Mice by target ({n_complete_recordings} recordings, '
                         f'{n_complete_sessions_bar} sessions)')
            set_plotsize(w=24, h=12, ax=ax)
            ax.get_figure().savefig(figures_dir / '6_mouse_overview.svg',
                                     dpi=FIGURE_DPI, bbox_inches='tight')

# Save unified error log
ERRORS_FPATH.parent.mkdir(parents=True, exist_ok=True)
df_errors.to_parquet(ERRORS_FPATH)
print(f"\nSaved {len(df_errors)} error entries to {ERRORS_FPATH}")

print(f"\nFigures saved to {figures_dir}")
