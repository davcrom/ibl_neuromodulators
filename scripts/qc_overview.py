"""
QC Overview Visualizations

Produces:
1. Session overview matrices highlighting data availability flags
2. Target overview barplots for sessions passing QC criteria
"""
import pandas as pd
from matplotlib import pyplot as plt

from iblnm.config import (
    PROJECT_ROOT, SESSIONS_CLEAN_FPATH, QCPHOTOMETRY_FPATH,
    VALID_TARGETS,  # noqa: F401 (used in pandas query with @VALID_TARGETS)
)
from iblnm.vis import session_overview_matrix, target_overview_barplot, set_plotsize

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
    ('has_trials', lambda df: df['has_trials']),
    ('has_photometry', lambda df: df['has_photometry']),
    ('trials_in_photometry_time', lambda df: df['trials_in_photometry_time']),
]

for name, highlight_func in highlight_configs:
    ax = session_overview_matrix(df_sessions, columns=columns, highlight=highlight_func)
    ax.set_title(f'Sessions with {name}')
    set_plotsize(w=48, h=32, ax=ax)
    ax.get_figure().savefig(figures_dir / f'session_overview_{name}.svg', bbox_inches='tight')

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
ax.set_title(f'Sessions with trials + photometry + aligned times\n{ax.get_title()}')
# ~ set_plotsize(w=24, h=12, ax=ax)

# --- Target Overview with QC criteria ---

# Merge QC results with sessions
# QC has columns: band, brain_region, n_early_samples, n_band_inversions, ..., eid
# We want sessions where n_early_samples == 0 and n_band_inversions == 0 for all recordings

# Aggregate QC per session - check if any recording has issues
qc_per_session = df_qc.groupby('eid').agg({
    'n_early_samples': 'max',  # if any recording has early samples, max > 0
    'n_band_inversions': 'max',
}).reset_index()

qc_per_session['passes_qc'] = (
    (qc_per_session['n_early_samples'] == 0) &
    (qc_per_session['n_band_inversions'] == 0)
)

# Merge with complete sessions
df_qc_good = df_complete.merge(
    qc_per_session[['eid', 'passes_qc']],
    on='eid',
    how='left'
)
df_qc_good = df_qc_good[df_qc_good['passes_qc']].copy()

# Explode to targets
df_targets_qc = df_qc_good.explode(column='target').dropna(subset='target')
df_targets_qc['target_NM'] = df_targets_qc['target'].str.split('-').str[0] + '-' + df_targets_qc['NM']
df_targets_qc = df_targets_qc.query('target_NM in @VALID_TARGETS').copy()

# Plot
ax = target_overview_barplot(df_targets_qc)
ax.set_title(f'Sessions with no early samples & no band inversions\n{ax.get_title()}')
# ~ set_plotsize(w=24, h=12, ax=ax)

# Summary
print("\n=== Summary ===")
print(f"Total sessions: {len(df_sessions)}")
print(f"  has_trials: {df_sessions['has_trials'].sum()}")
print(f"  has_photometry: {df_sessions['has_photometry'].sum()}")
print(f"  trials_in_photometry_time: {df_sessions['trials_in_photometry_time'].sum()}")
print(f"  all complete: {len(df_complete)}")
print(f"  passes QC: {len(df_qc_good)}")
