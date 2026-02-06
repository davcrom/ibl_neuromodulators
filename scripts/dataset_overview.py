"""
Dataset Overview

Produces session overview matrices at each stage of data processing:
1. All registered sessions
2. Sessions with raw data (task | photometry)
3. Sessions with extracted data (task | photometry) - verified accessible via QC flags
4. Sessions passing basic QC (n_unique_samples, n_band_inversions | trials_in_photometry_time)

Also tracks sessions failing each criterion and merges with error logs.
"""
import pandas as pd
from matplotlib import pyplot as plt

from iblnm.config import (
    PROJECT_ROOT, SESSIONS_FPATH, SESSIONS_LOG_FPATH,
    QCPHOTOMETRY_FPATH, QCPHOTOMETRY_LOG_FPATH, FIBERS_FPATH,
    VALID_TARGETS, FIGURE_DPI, TARGET2NM, REGION_NORMALIZE,
    STANDARD_LINES, STANDARD_STRAINS, STRAIN2NM, LINE2NM, MIN_NTRIALS,
)
from iblnm.util import (
    clean_sessions, drop_junk_duplicates, process_regions,
    aggregate_qc_per_session, build_filter_status, merge_failure_logs,
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

# Re-infer NM from strain/line using current mappings
def infer_nm(row):
    s_nm = STRAIN2NM.get(row['strain'], 'none')
    l_nm = LINE2NM.get(row['line'], 'none')
    if s_nm != 'none':
        return s_nm
    return l_nm

df_sessions['NM'] = df_sessions.apply(infer_nm, axis=1)

# Recompute session_n so all animals start at session 1 after filtering
df_sessions['session_n'] = df_sessions.groupby('subject')['date'].rank(method='dense')

# Drop duplicates
df_sessions = drop_junk_duplicates(df_sessions, ['subject', 'day_n'])
n_after_dedup = len(df_sessions)

print(f"\nSession counts:")
print(f"  Registered: {n_total}")
print(f"  After cleaning: {n_after_clean} (-{n_total - n_after_clean})")
print(f"  After deduplication: {n_after_dedup} (-{n_after_clean - n_after_dedup})")

# Load QC results if available
df_qc = None
qc_agg = None
if QCPHOTOMETRY_FPATH.exists():
    print(f"\nLoading QC results from {QCPHOTOMETRY_FPATH}")
    df_qc = pd.read_parquet(QCPHOTOMETRY_FPATH)
    qc_agg = aggregate_qc_per_session(df_qc, require_all=True)

# Check if QC flags exist in sessions
has_qc_flags = all(col in df_sessions.columns for col in ['has_trials', 'has_photometry', 'trials_in_photometry_time'])
if not has_qc_flags:
    print("\nWarning: QC flags not found in sessions. Using extracted data flags instead.")
    df_sessions['has_trials'] = df_sessions.get('has_extracted_task', False)
    df_sessions['has_photometry'] = df_sessions.get('has_extracted_photometry', False)
    df_sessions['trials_in_photometry_time'] = False

# Build filter status
df_filters = build_filter_status(df_sessions, qc_agg)
df_filters['has_valid_nm'] = (df_sessions['NM'] != 'none').values

# Check target-NM combo validity (e.g., LC-DA is invalid)
def valid_combo(row):
    targets, nm = row['target'], row['NM']
    if nm == 'none' or len(targets) == 0:
        return True
    return any(TARGET2NM.get(REGION_NORMALIZE.get(t.split('-')[0], t.split('-')[0])) == nm for t in targets)

df_filters['has_valid_target_nm'] = df_sessions.apply(valid_combo, axis=1).values

# Check for non-standard target names (e.g., DRN instead of DR)
def has_nonstandard_target(targets):
    if len(targets) == 0:
        return False
    return any(t.split('-')[0] in REGION_NORMALIZE for t in targets)

df_filters['has_nonstandard_target'] = df_sessions['target'].apply(has_nonstandard_target).values

# Check for non-standard line/strain names (NM inferred but naming needs fixing)
df_filters['has_nonstandard_line'] = ~df_sessions['line'].isin(STANDARD_LINES | {None}).values
df_filters['has_nonstandard_strain'] = ~df_sessions['strain'].isin(STANDARD_STRAINS | {None}).values

# Check session status (junk sessions have too few trials, etc.)
df_filters['is_good_session'] = (df_sessions['session_status'] == 'good').values

# Check hemisphere mismatches using fiber coordinates
df_fibers = pd.read_csv(FIBERS_FPATH) if FIBERS_FPATH.exists() else None
if df_fibers is not None:
    df_fibers['hemi'] = (df_fibers['X-ml_um'] > 0).map({True: 'L', False: 'R'})
    fiber_hemi = df_fibers.groupby(['subject', 'targeted_region'])['hemi'].first().to_dict()

    def has_mismatch(row):
        targets, subject = row['target'], row['subject']
        if len(targets) == 0:
            return False
        for t in targets:
            parts = t.split('-')
            if len(parts) > 1 and parts[-1].lower() in ('l', 'r'):
                hemi_fiber = fiber_hemi.get((subject, parts[0]))
                if hemi_fiber and parts[-1].upper() != hemi_fiber:
                    return True
        return False

    df_filters['has_hemisphere_mismatch'] = df_sessions.apply(has_mismatch, axis=1).values
else:
    df_filters['has_hemisphere_mismatch'] = False

# Load error logs
error_logs = []
if SESSIONS_LOG_FPATH.exists():
    df_log_sessions = pd.read_parquet(SESSIONS_LOG_FPATH)
    error_logs.append(('query_database', df_log_sessions))
    print(f"Loaded {len(df_log_sessions)} errors from query_database")

if QCPHOTOMETRY_LOG_FPATH.exists():
    df_log_qc = pd.read_parquet(QCPHOTOMETRY_LOG_FPATH)
    error_logs.append(('photometry_qc', df_log_qc))
    print(f"Loaded {len(df_log_qc)} errors from photometry_qc")


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

# 3. Extracted data: task | photometry (verified accessible)
n_trials = df_filters['has_trials'].sum()
n_phot = df_filters['has_photometry'].sum()
plot_paired_matrices(
    df_sessions,
    left_highlight=lambda df: df['eid'].isin(df_filters[df_filters['has_trials']]['eid']),
    right_highlight=lambda df: df['eid'].isin(df_filters[df_filters['has_photometry']]['eid']),
    left_title=f'Accessible trials data (n={n_trials})',
    right_title=f'Accessible photometry data (n={n_phot})',
    filename='3_extracted_data.svg'
)

# 4. QC criteria: basic photometry QC | trials_in_photometry_time
# NOTE: is_good_session check commented out pending investigation of CQ mice (n_trials=0 since Aug 2025)
# passes_qc_and_trials = df_filters['passes_basic_qc'] & df_filters['is_good_session']
n_qc = df_filters['passes_basic_qc'].sum()
n_sync = df_filters['trials_in_photometry_time'].sum()
plot_paired_matrices(
    df_sessions,
    left_highlight=lambda df: df['eid'].isin(df_filters[df_filters['passes_basic_qc']]['eid']),
    right_highlight=lambda df: df['eid'].isin(df_filters[df_filters['trials_in_photometry_time']]['eid']),
    left_title=f'Passes basic QC (n={n_qc})',
    right_title=f'Trials in photometry time (n={n_sync})',
    filename='4_qc_criteria.svg'
)


# =============================================================================
# Track Failures
# =============================================================================

# Sessions that fail any criterion
df_failures = df_filters[
    ~df_filters['has_raw_task'] |
    ~df_filters['has_raw_photometry'] |
    ~df_filters['has_trials'] |
    ~df_filters['has_photometry'] |
    ~df_filters['passes_basic_qc'] |
    ~df_filters['trials_in_photometry_time'] |
    ~df_filters['has_valid_nm'] |
    ~df_filters['has_valid_target_nm'] |
    df_filters['has_nonstandard_target'] |
    df_filters['has_nonstandard_line'] |
    df_filters['has_nonstandard_strain'] |
    df_filters['has_hemisphere_mismatch']
].copy()

# Merge with error logs
df_failures = merge_failure_logs(df_failures, error_logs)


# =============================================================================
# Final Dataset Summary
# =============================================================================

# Sessions passing all criteria (NM='none' sessions included with inferred values)
complete_mask = (
    df_filters['has_trials'] &
    df_filters['has_photometry'] &
    df_filters['passes_basic_qc'] &
    df_filters['trials_in_photometry_time']
)
df_complete = df_sessions[df_sessions['eid'].isin(df_filters[complete_mask]['eid'])].copy()

print(f"\n{'='*50}")
print("Dataset Summary")
print(f"{'='*50}")
print(f"Total sessions: {len(df_sessions)}")
print(f"  Raw task: {n_raw_task}")
print(f"  Raw photometry: {n_raw_phot}")
print(f"  Accessible trials: {n_trials}")
print(f"  Accessible photometry: {n_phot}")
print(f"  Passes basic QC: {n_qc}")
print(f"  Trials in photometry time: {n_sync}")
print(f"  Complete (all criteria): {len(df_complete)}")


# =============================================================================
# Target and Mouse Overview (for complete sessions)
# =============================================================================

if len(df_complete) > 0:
    # Explode to one row per target and process regions
    df_targets = df_complete.explode(column='target').dropna(subset='target')
    df_targets = process_regions(df_targets, region_col='target')

    if len(df_targets) > 0:
        # Target overview
        ax = target_overview_barplot(df_targets)
        ax.set_title(f'Complete sessions by target (n={len(df_complete)} sessions)')
        set_plotsize(w=24, h=12, ax=ax)
        ax.get_figure().savefig(figures_dir / '5_target_overview.svg', dpi=FIGURE_DPI, bbox_inches='tight')

        # Mouse overview with hemisphere
        ax = mouse_overview_barplot(df_targets)
        ax.set_title(f'Mice by target (n={len(df_complete)} sessions)')
        set_plotsize(w=24, h=12, ax=ax)
        ax.get_figure().savefig(figures_dir / '6_mouse_overview.svg', dpi=FIGURE_DPI, bbox_inches='tight')

# Save failures
failures_fpath = PROJECT_ROOT / 'data/dataset_failures.pqt'
failures_fpath.parent.mkdir(parents=True, exist_ok=True)
df_failures.to_parquet(failures_fpath)
print(f"\nSaved {len(df_failures)} session failures to {failures_fpath}")

print(f"\nFigures saved to {figures_dir}")
