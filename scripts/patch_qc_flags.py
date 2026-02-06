"""
Patch: add has_trials, has_photometry, trials_in_photometry_time to sessions.pqt.

Verifies data actually loads for sessions with QC results.
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
from one.api import ONE

from iblnm.config import SESSIONS_FPATH, QCPHOTOMETRY_FPATH

one = ONE()

# Load metadata
df_sessions = pd.read_parquet(SESSIONS_FPATH)
df_qc = pd.read_parquet(QCPHOTOMETRY_FPATH)

# Initialize flags
df_sessions['has_trials'] = False
df_sessions['has_photometry'] = False
df_sessions['trials_in_photometry_time'] = False

# Check sessions with QC results
eids_to_check = df_qc['eid'].unique().tolist()
print(f"Checking {len(eids_to_check)} sessions...")

for eid in tqdm(eids_to_check):
    idx = df_sessions[df_sessions['eid'] == eid].index[0]
    signal = None
    trials = None

    # Load photometry
    try:
        signal = one.load_dataset(eid, 'photometry.signal.pqt', collection='alf/photometry')
        df_sessions.loc[idx, 'has_photometry'] = True
    except Exception:
        pass

    # Load trials (independently of photometry)
    try:
        try:
            trials = one.load_dataset(eid, '_ibl_trials.table.pqt', collection='alf/task_00')
        except Exception:
            trials = one.load_dataset(eid, '_ibl_trials.table.pqt', collection='alf')
        df_sessions.loc[idx, 'has_trials'] = True
    except Exception:
        pass

    # Check time overlap (only if both loaded)
    if signal is not None and trials is not None:
        time_cols = [c for c in trials.columns if c.endswith('_times')]
        t_trials = trials[time_cols].values.flatten()
        t_trials = t_trials[~np.isnan(t_trials)]

        in_range = (t_trials.min() >= signal.index.min() - 1) and (t_trials.max() <= signal.index.max() + 1)
        df_sessions.loc[idx, 'trials_in_photometry_time'] = in_range

# Save
df_sessions.to_parquet(SESSIONS_FPATH)
print(f"\nhas_photometry: {df_sessions['has_photometry'].sum()}")
print(f"has_trials: {df_sessions['has_trials'].sum()}")
print(f"trials_in_photometry_time: {df_sessions['trials_in_photometry_time'].sum()}")
print(f"Saved to {SESSIONS_FPATH}")
