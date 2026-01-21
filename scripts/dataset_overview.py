import pandas as pd
from matplotlib import pyplot as plt
plt.ion()  # Enable interactive plotting

from iblnm.config import (
    PROJECT_ROOT, SESSIONS_FPATH, SESSIONS_CLEAN_FPATH, MIN_NTRIALS,
    VALID_TARGETS,  # noqa: F401 (used in pandas query)
)
from iblnm.util import (
    clean_sessions, get_session_length, resolve_session_status,
    add_dataset_flags, drop_junk_duplicates, add_hemisphere
)
from iblnm.vis import session_overview_matrix, target_overview_barplot, mouse_overview_barplot, set_plotsize

# Create output directory for figures
figures_dir = PROJECT_ROOT / 'figures/dataset_overview'
figures_dir.mkdir(parents=True, exist_ok=True)

# Load and clean sessions
df_sessions = pd.read_parquet(SESSIONS_FPATH)
df_sessions = clean_sessions(df_sessions)

# Summary
print(f"\nTotal sessions: {len(df_sessions)}")
print(df_sessions['session_status'].value_counts())

# Drop junk duplicates and conflicts before plotting
group_cols = ['subject', 'day_n']
df_plot = drop_junk_duplicates(df_sessions, group_cols)

# Save cleaned sessions
df_plot.to_parquet(SESSIONS_CLEAN_FPATH)

# Plot overview matrices with different highlight criteria
highlight_configs = [
    ('raw task', lambda df: df['has_raw_task']),
    ('raw photometry', lambda df: df['has_raw_photometry']),
    ('extracted behavior', lambda df: df['has_extracted_task']),
    ('extracted photometry', lambda df: df['has_extracted_photometry']),
    ('extracted task & photometry', lambda df: df['session_status'] == 'good'),
]

for name, highlight_func in highlight_configs:
    ax = session_overview_matrix(df_plot, highlight=highlight_func)
    ax.set_title(f'Sessions with {name}')
    set_plotsize(w=48, h=32, ax=ax)
    ax.get_figure().savefig(figures_dir / f'session_overview_{name}.svg', bbox_inches='tight')

# Filter for good sessions only
df_good = df_sessions.query('session_status == "good"').copy()

# Explode multi-fiber sessions (one row per target)
df_photometry = df_good.explode(column='target').dropna(subset='target')

# Add target_NM column (e.g., 'VTA-DA')
df_photometry['target_NM'] = df_photometry['target'].str.split('-').str[0] + '-' + df_photometry['NM']
df_photometry = df_photometry.query('target_NM in @VALID_TARGETS').copy()

# Plot target overview
ax = target_overview_barplot(df_photometry)
set_plotsize(w=24, h=12, ax=ax)
ax.get_figure().savefig(figures_dir / 'target_overview_barplot.svg', bbox_inches='tight')

# Plot mouse overview (with hemisphere info)
df_photometry = add_hemisphere(df_photometry)
ax = mouse_overview_barplot(df_photometry)
set_plotsize(w=24, h=12, ax=ax)
ax.get_figure().savefig(figures_dir / 'mouse_overview_barplot.svg', bbox_inches='tight')
