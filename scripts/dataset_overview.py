import pandas as pd
from datetime import datetime
from tqdm import tqdm
tqdm.pandas()

from iblnm.config import *
from iblnm.io import get_target_regions
from iblnm.util import (get_session_length, check_extracted_data,
                        resolve_session_status)
from iblnm.vis import (session_overview_matrix, target_overview_barplot,
                       mouse_overview_barplot, set_plotsize)

# Create output directory for figures
figures_dir = PROJECT_ROOT / 'figures/dataset_overview'
figures_dir.mkdir(parents=True, exist_ok=True)

# Load sessions from previous query
df_sessions = pd.read_parquet(SESSIONS_FPATH)  # raw output of io.fetch_sessions

# Remove session types we would not analyze
df_sessions = df_sessions.query('session_type not in @EXCLUDE_SESSION_TYPES')

# Add some convenience columns
df_sessions['date'] = df_sessions['start_time'].apply(datetime.fromisoformat).dt.date
df_sessions['day_n'] = df_sessions.groupby('subject')['date'].transform(lambda x: [(date - x.min()).days for date in x])
df_sessions['session_n'] = df_sessions.groupby('subject')['date'].rank(method='dense')

# Get session length each session
df_sessions['session_length'] = df_sessions.apply(get_session_length, axis='columns')

# Check data completion
df_sessions['data_complete'] = df_sessions.apply(check_extracted_data, axis='columns')
df_sessions['trials_complete'] = df_sessions['n_trials'] > MIN_NTRIALS

# Resolve session status
session_groups = df_sessions.groupby(['subject', 'date', 'session_type'], group_keys=False)
df_sessions['session_status'] = session_groups.apply(
    resolve_session_status, columns=['data_complete', 'trials_complete'],
    include_groups=False
    )
# Confirm there are no duplicates among good sessions
good_sessions = df_sessions.query('session_status == "good"')
assert all(
    good_sessions.groupby(['subject', 'date', 'session_type']).size( ) == 1
    )

# Print the number of sessions
print(f"Total sessions: {len(df_sessions)}")
# Display duplicate session resolutions
print(df_sessions['session_status'].value_counts())

# Display all sessions as a matrix with good sessions highlighted
# columns = 'day_n'  # to see the actual timeline
columns = 'session_n'  # to see a condensed representation of the sessions
ax = session_overview_matrix(df_sessions, columns=columns)
set_plotsize(w=48, h=32, ax=ax)
ax.get_figure().savefig(figures_dir / 'session_overview_matrix.svg', bbox_inches='tight')

# Filter for good sessions only
df_sessions = df_sessions.query('session_status == "good"').copy()

# Get brain region targets
if 'target' not in df_sessions.columns:
    df_sessions = df_sessions.progress_apply(get_target_regions, axis='columns')

# Explode multi-fiber sessions so that each fiber gets a row
df_photometry_sessions = df_sessions.explode(column='target').dropna(subset='target')

# Clean up the target column
df_photometry_sessions['target'] = df_photometry_sessions['target']
df_photometry_sessions['target_NM'] = df_photometry_sessions.apply(
    lambda x: '-'.join([x['target'].split('-')[0], x['NM']]),
    axis='columns'
)
df_photometry_sessions = df_photometry_sessions.query('target_NM in @VALID_TARGETS').copy()

# Number of sessions of each type per target
ax = target_overview_barplot(df_photometry_sessions)
set_plotsize(w=24, h=12, ax=ax)
ax.get_figure().savefig(figures_dir / 'target_overview_barplot.svg', bbox_inches='tight')

# Number of mice with sufficient BCW and ECW sessions
ax = mouse_overview_barplot(df_photometry_sessions)
set_plotsize(w=24, h=12, ax=ax)
ax.get_figure().savefig(figures_dir / 'mouse_overview_barplot.svg', bbox_inches='tight')
