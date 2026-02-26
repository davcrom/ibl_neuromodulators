"""
Demo: Loading and exploring a photometry session
=================================================
This script shows how to:
  1. Load the sessions table and filter it to find sessions of interest
  2. Load trial, photometry, and wheel data for a single session
  3. Plot the photometry response around feedback, split by outcome

Requirements: numpy, pandas, matplotlib, h5py, one, iblnm
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from one.api import ONE

from iblnm.config import SESSIONS_FPATH, SESSIONS_H5_DIR
from iblnm.data import PhotometrySession


# ===========================================================================
# 1. Load the sessions table
# ===========================================================================
# sessions.pqt is a parquet file — a compact column-oriented table.
# Each row is one session. Key columns:
#   subject         — mouse name
#   NM              — neuromodulator (DA, 5HT, NE, ACh)
#   brain_region    — list of recording targets, e.g. ['VTA']
#   session_type    — 'training', 'biased', or 'ephys'
#   eid             — unique session identifier (used to find the HDF5 file)

df_sessions = pd.read_parquet(SESSIONS_FPATH)
print(f"Sessions in table: {len(df_sessions)}")


# ===========================================================================
# 2. Filter to find sessions of interest
# ===========================================================================
# Standard pandas boolean indexing. brain_region is a list column, so we
# use .apply() to check whether a region appears in each row's list.

mask = (
    # ~(df_sessions['NM'] == 'DA') &
    (df_sessions['session_type'] == 'biased') &
    (df_sessions['brain_region'].apply(lambda regions: 'VTA' in regions))
)
df_vta_biased = df_sessions[mask]
print(f"VTA dopamine biased sessions: {len(df_vta_biased)}")


# ===========================================================================
# 3. Pick one session and load its data
# ===========================================================================
# Each session's data is stored as an HDF5 file at data/sessions/{eid}.h5.
# We find the first session that already has a file on disk.

session_row = next(
    row for _, row in df_vta_biased.iterrows()
    if (SESSIONS_H5_DIR / f"{row['eid']}.h5").exists()
)

one = ONE()
ps = PhotometrySession(session_row, one=one)
fpath = 'DA_sessions' / f'{ps.eid}.h5'

print(f"\nLoading: {ps.subject}  {ps.date}  ({ps.session_type})")

# load_h5 with no arguments loads everything in the file:
#   ps.trials          — DataFrame of trial events and outcomes
#   ps.photometry      — dict; ['GCaMP_preprocessed'] is a DataFrame (time × region)
#   ps.responses       — xarray DataArray (region × event × trial × time)
#   ps.wheel_velocity  — numpy array (n_trials × n_samples)
ps.load_h5(fpath)


# ===========================================================================
# 4. Explore the trials DataFrame
# ===========================================================================
# Each row is one trial. Key columns:
#   stimOn_times        — stimulus appeared (seconds, session clock)
#   firstMovement_times — first wheel movement
#   feedback_times      — reward or punishment delivered
#   choice              — -1 left, 1 right, 0 no-go
#   feedbackType        — 1 reward, -1 punishment
#   signed_contrast     — stimulus contrast (negative = left, positive = right)

print(f"\nTrials: {len(ps.trials)}")
print(f"Fraction correct: {(ps.trials['feedbackType'] == 1).mean():.2f}")

reaction_time = ps.trials['firstMovement_times'] - ps.trials['stimOn_times']
print(f"Median reaction time: {reaction_time.median() * 1000:.0f} ms")


# ===========================================================================
# 5. Explore the photometry signal
# ===========================================================================
# ps.photometry['GCaMP_preprocessed'] is a DataFrame:
#   index  — time in seconds (sampled at 30 Hz)
#   columns — one per brain region (z-score units)

signal = ps.photometry['GCaMP_preprocessed']
print(f"\nPhotometry: {signal.shape[0]} samples, regions: {signal.columns.tolist()}")
print(f"Duration: {signal.index[-1] - signal.index[0]:.0f} s")


# ===========================================================================
# 6. Explore the per-trial responses
# ===========================================================================
# ps.responses is an xarray DataArray with dimensions:
#   region — brain region
#   event  — trial event the signal is aligned to
#   trial  — trial index
#   time   — seconds relative to the event

print(f"\nResponses shape: {dict(zip(ps.responses.dims, ps.responses.shape))}")
print(f"Events: {ps.responses.coords['event'].values.tolist()}")
print(f"Time window: {float(ps.responses.time[0]):.2f} to {float(ps.responses.time[-1]):.2f} s")


# ===========================================================================
# 7. Explore the wheel velocity
# ===========================================================================
# ps.wheel_velocity is a (n_trials × n_samples) numpy array.
# Each row is one trial: velocity from stimOn to feedback (cm/s).
# Rows shorter than the longest trial are NaN-padded on the right.

print(f"\nWheel velocity: {ps.wheel_velocity.shape}  (trials × samples at {ps.wheel_fs} Hz)")

# Build a time axis for the velocity matrix (time 0 = stimulus onset)
n_samples = ps.wheel_velocity.shape[1]
wheel_times = np.arange(n_samples) / ps.wheel_fs


# ===========================================================================
# 8. Plot: photometry at feedback, reward vs punishment
# ===========================================================================

region = ps.responses.coords['region'].values[0]
tpts = ps.responses.coords['time'].values

reward_mask = ps.trials['feedbackType'].values == 1
punish_mask = ps.trials['feedbackType'].values == -1

# Clip each trial at the time of the next event, then subtract the pre-event baseline.
# mask_subsequent_events: replaces timepoints after the next event onset with NaN,
#   so responses to e.g. stimOn don't bleed into the movement period.
# subtract_baseline: subtracts the mean of the pre-event window (default: full pre-event period).
responses = ps.mask_subsequent_events(ps.responses)
responses = ps.subtract_baseline(responses)

resp = responses.sel(region=region, event='stimOn_times').values  # (n_trials, n_time)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# --- Photometry ---
ax = axes[0]
for mask, color, label in [
    (reward_mask, 'green', 'Correct'),
    (punish_mask, 'red',   'Incorrect'),
]:
    mean = np.nanmean(resp[mask], axis=0)
    sem  = np.nanstd(resp[mask],  axis=0) / np.sqrt(mask.sum())
    ax.fill_between(tpts, mean - sem, mean + sem, alpha=0.3, color=color)
    ax.plot(tpts, mean, color=color, label=f'{label} (n={mask.sum()})')

ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
ax.axhline(0, color='gray', linestyle='-',  linewidth=0.5)
ax.set_xlabel('Time from feedback (s)')
ax.set_ylabel('Photometry signal (z-score)')
ax.set_title(f'{region}')
ax.legend()

# --- Wheel velocity ---
# Trim to the first 1 s after stimulus onset
n_plot = int(ps.wheel_fs)                      # number of samples in 1 s
t_plot = wheel_times[:n_plot]
vel_plot = ps.wheel_velocity[:, :n_plot]       # (n_trials, n_plot)

ax = axes[1]
for mask, color in [(reward_mask, 'green'), (punish_mask, 'red')]:
    ax.plot(t_plot, vel_plot[mask].T, color=color, linewidth=0.3, alpha=0.4)

# Add a dummy line per colour for the legend
ax.plot([], [], color='green', label=f'Correct (n={reward_mask.sum()})')
ax.plot([], [], color='red',   label=f'Incorrect (n={punish_mask.sum()})')

ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
ax.set_xlabel('Time from stimulus onset (s)')
ax.set_ylabel('Wheel velocity (cm/s)')
ax.set_title('Wheel velocity')
ax.legend()

fig.suptitle(f'{ps.subject}  —  {ps.date}  ({ps.session_type})')
fig.tight_layout()
plt.show()
