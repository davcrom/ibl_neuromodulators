"""
QC Overview Visualizations

Produces:
1. Session overview matrices highlighting data availability flags
2. Target overview barplots for sessions passing QC criteria
"""
import pandas as pd
from matplotlib import pyplot as plt

from iblnm.config import (
    PROJECT_ROOT, SESSIONS_CLEAN_FPATH, QCPHOTOMETRY_FPATH, FIGURE_DPI,
    VALID_TARGETS,  # noqa: F401 (used in pandas query with @VALID_TARGETS)
)
from iblnm.vis import (
    session_overview_matrix, target_overview_barplot, mouse_overview_barplot, set_plotsize
)

plt.ion()

# Create output directory
figures_dir = PROJECT_ROOT / 'figures/qc_overview'
figures_dir.mkdir(parents=True, exist_ok=True)

# Load cleaned sessions and QC results
df_sessions = pd.read_parquet(SESSIONS_CLEAN_FPATH)
df_qc = pd.read_parquet(QCPHOTOMETRY_FPATH)

# --- Session Overview Matrices ---
columns = 'session_n'

highlight_configs = [
    ('trials data', lambda df: df['has_trials']),
    ('photometry data', lambda df: df['has_photometry']),
    ('synchronized trials and photometry', lambda df: df['trials_in_photometry_time']),
]

for name, highlight_func in highlight_configs:
    ax = session_overview_matrix(df_sessions, columns=columns, highlight=highlight_func)
    ax.set_title(f'Sessions with {name}')
    set_plotsize(w=48, h=32, ax=ax)
    fpath = figures_dir / f'session_overview_{name.replace(" ", "_")}.svg'
    ax.get_figure().savefig(fpath, dpi=FIGURE_DPI, bbox_inches='tight')

# --- Target Overview Barplots ---

# Filter sessions with all data available
df_complete = df_sessions[
    df_sessions['has_trials'] &
    df_sessions['has_photometry'] &
    df_sessions['trials_in_photometry_time']
].copy()

# Explode to one row per target and add target_NM
df_targets = df_complete.explode(column='target').dropna(subset='target')
df_targets['target_NM'] = df_targets['target'].str.split('-').str[0] + '-' + df_targets['NM']
df_targets = df_targets.query('target_NM in @VALID_TARGETS').copy()

# Plot target overview for complete sessions
ax = target_overview_barplot(df_targets)
ax.set_title(f'Sessions with synchronized trials and photometry\n{ax.get_title()}')
fpath = figures_dir / 'target_overview_complete.svg'
ax.get_figure().savefig(fpath, dpi=FIGURE_DPI, bbox_inches='tight')

# --- Target Overview with QC criteria ---

# Merge QC results with sessions
# QC has columns: band, brain_region, n_early_samples, n_band_inversions, ..., eid
# We want sessions where n_early_samples == 0 and n_band_inversions == 0 for all recordings

# Aggregate QC per session - check if any recording has issues
qc_per_session = df_qc.groupby('eid').agg({
    'n_early_samples': 'max',  # if any recording has early samples, max > 0
    'n_band_inversions': 'max',
    'n_unique_samples': 'min',  # if any recording has low unique samples, min < threshold
}).reset_index()

qc_per_session['passes_qc'] = (
    (qc_per_session['n_early_samples'] == 0) &
    (qc_per_session['n_band_inversions'] == 0)
)

# Baseline QC: no early samples, no band inversions, >10% unique samples
qc_per_session['passes_baseline_qc'] = (
    (qc_per_session['n_early_samples'] == 0) &
    (qc_per_session['n_band_inversions'] == 0) &
    (qc_per_session['n_unique_samples'] > 0.1)
)

# Merge with complete sessions
df_qc_good = df_complete.merge(
    qc_per_session[['eid', 'passes_qc']],
    on='eid',
    how='left'
)
df_qc_good = df_qc_good[df_qc_good['passes_qc'].fillna(False)].copy()

# Explode to targets
df_targets_qc = df_qc_good.explode(column='target').dropna(subset='target')
df_targets_qc['target_NM'] = df_targets_qc['target'].str.split('-').str[0] + '-' + df_targets_qc['NM']
df_targets_qc = df_targets_qc.query('target_NM in @VALID_TARGETS').copy()

# Plot
ax = target_overview_barplot(df_targets_qc)
ax.set_title(f'Sessions with no early samples & no band inversions\n{ax.get_title()}')
fpath = figures_dir / 'target_overview_qc.svg'
ax.get_figure().savefig(fpath, dpi=FIGURE_DPI, bbox_inches='tight')

# --- Baseline QC: no early samples, no band inversions, >10% unique samples ---

# Merge baseline QC with complete sessions
df_baseline = df_complete.merge(
    qc_per_session[['eid', 'passes_baseline_qc']],
    on='eid',
    how='left'
)
df_baseline = df_baseline[df_baseline['passes_baseline_qc'].fillna(False)].copy()

# Session overview matrix for baseline QC
ax = session_overview_matrix(
    df_sessions, columns=columns,
    highlight=lambda df: df['eid'].isin(df_baseline['eid'])
)
ax.set_title('Sessions passing baseline QC')
set_plotsize(w=48, h=32, ax=ax)
fpath = figures_dir / 'session_overview_baseline_qc.svg'
ax.get_figure().savefig(fpath, dpi=FIGURE_DPI, bbox_inches='tight')

# Explode to targets for barplots
df_targets_baseline = df_baseline.explode(column='target').dropna(subset='target')
df_targets_baseline['target_NM'] = df_targets_baseline['target'].str.split('-').str[0] + '-' + df_targets_baseline['NM']
df_targets_baseline = df_targets_baseline.query('target_NM in @VALID_TARGETS').copy()

# Target overview barplot
ax = target_overview_barplot(df_targets_baseline)
ax.set_title(f'Sessions passing baseline QC\n{ax.get_title()}')
fpath = figures_dir / 'target_overview_baseline_qc.svg'
ax.get_figure().savefig(fpath, dpi=FIGURE_DPI, bbox_inches='tight')

# Mouse overview barplot
ax = mouse_overview_barplot(df_targets_baseline)
ax.set_title(f'Mice passing baseline QC\n{ax.get_title()}')
fpath = figures_dir / 'mouse_overview_baseline_qc.svg'
ax.get_figure().savefig(fpath, dpi=FIGURE_DPI, bbox_inches='tight')

# Summary
print("\n=== Summary ===")
print(f"Total sessions: {len(df_sessions)}")
print(f"  has_trials: {df_sessions['has_trials'].sum()}")
print(f"  has_photometry: {df_sessions['has_photometry'].sum()}")
print(f"  trials_in_photometry_time: {df_sessions['trials_in_photometry_time'].sum()}")
print(f"  all complete: {len(df_complete)}")
print(f"  passes QC: {len(df_qc_good)}")
print(f"  passes baseline QC: {len(df_baseline)}")
