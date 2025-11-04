import argparse
import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
from matplotlib import pyplot as plt
from matplotlib import colors
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
tqdm.pandas()

from one.api import ONE

from iblnm import util, io, vis
from iblnm.config import *

def _resolve_session_status(session_group):
    """
    Resolve duplicate sessions by flagging them as 'good', 'junk', 'conflict', or 'missing'
    """
    # Create a copy to avoid modifying original data
    group = session_group.copy()

    # Initialize all sessions as 'junk'
    group['session_status'] = 'junk'

    # Find sessions that have behavior data AND meet quality criteria (trials OR length)
    good_sessions_mask = (
        group['trialsTable_exists'] &
        (
            (group['n_trials'] > MIN_NTRIALS) |
            (group['session_length'] > MIN_SESSIONLENGTH)
        )
    )
    good_sessions = group[good_sessions_mask]

    if len(good_sessions) == 1:
        # One session has behavior and quality - mark as 'good', others remain 'junk'
        group.loc[good_sessions.index, 'session_status'] = 'good'

    elif len(good_sessions) > 1:
            group.loc[good_sessions.index, 'session_status'] = 'conflict'

    # No sessions in group have raw task data & sufficient trials or length
    else:
        # Check for rare cases where there is no raw task data in Alyx,
        # but the session dictionary says there should be
        missing_behavior_mask = (
            ~group['trialsTable_exists'] &
            (group['n_trials'] > MIN_NTRIALS)
        )
        missing_sessions = group[missing_behavior_mask]

        if len(missing_sessions) == 1:
            # Single session with missing behavior - mark as 'missing'
            group.loc[missing_sessions.index, 'session_status'] = 'missing'
        elif len(missing_sessions) > 1:
            # Multiple sessions with missing behavior - mark as 'missing_conflict'
            group.loc[missing_sessions.index, 'session_status'] = 'missing_conflict'

    return group['session_status']


# TODO: make session file name an arg
df_sessions = pd.read_parquet('metadata/sessions_2025-10-31-20h08.pqt')

# Convert start_time to datetime and date columns
df_sessions['start_time_dt'] = df_sessions['start_time'].apply(datetime.fromisoformat)
df_sessions['date'] = df_sessions['start_time_dt'].dt.date
df_sessions['day_n'] = df_sessions.groupby('subject')['date'].transform(  # days since first session
    lambda x: [(date - x.min()).days for date in x]
)
df_sessions['session_n'] = df_sessions.groupby('subject')['date'].rank(method='dense')  # session number

# Add a flag to potential duplicates
# Duplicates: multiple sessions on the same day with the same task protocol for a single animal
df_sessions['duplicate_session'] = df_sessions.duplicated(
    subset=['subject', 'date', 'session_type'], keep=False
)

# Make a couple new columns that are easier to work with
# ~df_sessions['raw_taskData_exists'] = df_sessions['_iblrig_taskData.raw.jsonable'].copy()
df_sessions['trialsTable_exists'] = df_sessions['alf/task_00/_ibl_trials.table.pqt'].copy()
df_sessions['n_trials_sessionDict'] = df_sessions['n_trials'].copy()  # number of trials listed in the session dictionary

# Get n_trials from the raw task data for each session
# Note: this takes some time so best not to re-run unless a new query is made
# ~if 'n_trials_taskData' not in df_sessions.columns:
    # ~print("Getting N trials from raw task data...")
    # ~df_sessions['n_trials_taskData'] = df_sessions.progress_apply(
        # ~io._get_ntrials_from_raw_taskData, one=ONE(), axis='columns'
    # ~)

# Get session length each session
df_sessions['session_length'] = df_sessions.apply(util.get_session_length, axis='columns')

# Apply the function to resolve duplicates and assign an overall status to each session
df_sessions['session_status'] = df_sessions.groupby(['subject', 'date', 'session_type'], group_keys=False).apply(
    _resolve_session_status, include_groups=False
)

# Check that there are no remaining duplciates among the good sessions
assert all(df_sessions.query('session_status in ["good", "missing"]').groupby(['subject', 'date', 'session_type']).apply(len, include_groups=False) == 1)

# Save df_sessions with a timestamp
util.save_timestamped_pqt(df_sessions, SESSIONS_FPATH)


# Restrict rows and columns of df_sessions to get df_qc
filter_list = [  # list of filtering operations to apply (joined by AND)
    '(session_type not in @EXCLUDE_SESSION_TYPES)',
    '(subject not in @EXCLUDE_SUBJECTS)',
]
# filter_list = []  # empty list to take all sessions
if filter_list:
    df_qc = df_sessions.query(' and '.join(filter_list)).copy()
else:
    df_qc = df_sessions.copy()

# Clean up the columns
columns_to_include = [
    'eid', 'session_status', 'subject', 'NM',
    'day_n', 'session_n', 'start_time', 'end_time', 'session_length',
    'task_protocol', 'session_type', 'duplicate_session',
    'trialsTable_exists', 'n_trials',
    # ~'n_trials_taskData', 'target'
]
df_qc = df_qc[columns_to_include].sort_values(['subject', 'start_time']).copy()

# Print the number of sessions
print(f"Total sessions: {len(df_qc)}")

# Display duplicate session resolutions
print(df_qc['session_status'].value_counts())

# Make a folder and save the results
session_qc_path = Path("session_qc")
session_qc_path.mkdir(exist_ok=True)
# Full table
df_qc.to_parquet(session_qc_path / f'session_qc_{timestamp}.pqt')
# Duplicate sessions that could not be resolved
df_qc.query('session_status == "conflict"').to_parquet(session_qc_path / f'conflict_{timestamp}.pqt')
# Non-duplicate sessions that are missing raw task data
df_qc.query('not duplicate_session & not raw_taskData_exists').to_parquet(session_qc_path / f'missing_taskData_{timestamp}.pqt')
# Session where the number of trials in the session dict and raw task data don't match (potential extraction errors)
df_qc.query('raw_taskData_exists & (n_trials_sessionDict != n_trials_taskData)').to_parquet(session_qc_path / f'check_taskData_extraction_{timestamp}.pqt')

# Make a session overview plot
timestamp = datetime.now().strftime("%Y-%m-%d-%Hh%M")
ax = session_overview_matrix(df_qc, columns='day_n')
ax.get_figure().savefig(session_qc_path / f'IBL-NM_timeline_overview_{timestamp}.pdf')
ax = session_overview_matrix(df_qc, columns='session_n')
ax.get_figure().savefig(session_qc_path / f'IBL-NM_session_overview_{timestamp}.pdf')
