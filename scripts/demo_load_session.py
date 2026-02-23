"""
Demo: Loading and exploring a photometry session
=================================================
This script shows you how to:
  1. Load the sessions table and filter it to find sessions of interest
  2. Open an HDF5 file and load trials, photometry, and wheel data
  3. Do some basic exploration with numpy and pandas

Prerequisites: numpy, pandas, matplotlib, h5py
    pip install numpy pandas matplotlib h5py
"""

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — adjust these to match your local setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
SESSIONS_FPATH = PROJECT_ROOT / 'metadata' / 'sessions.pqt'
H5_DIR = PROJECT_ROOT / 'data' / 'sessions'


# ===========================================================================
# 1. Load the sessions table
# ===========================================================================
# sessions.pqt is a Parquet file — a compact, column-oriented table format.
# pd.read_parquet() loads it into a standard pandas DataFrame.

df_sessions = pd.read_parquet(SESSIONS_FPATH)

print(f"Total sessions: {len(df_sessions)}")
print(f"\nColumns:\n{df_sessions.columns.tolist()}")
print(f"\nFirst row:\n{df_sessions.iloc[0]}")


# ===========================================================================
# 2. Filtering sessions
# ===========================================================================
# df_sessions is just a pandas DataFrame, so you can filter it using
# standard boolean indexing.

# --- Filter by neuromodulator ---
# The 'NM' column records which neuromodulator is measured (DA, 5HT, NE, ACh).
da_sessions = df_sessions[df_sessions['NM'] == 'DA']
print(f"\nDopamine sessions: {len(da_sessions)}")

# --- Filter by brain region ---
# 'brain_region' is a list column — each cell is a list of brain region names.
# We use .apply() to check whether a given region appears in that list.
vta_sessions = df_sessions[
    df_sessions['brain_region'].apply(lambda regions: 'VTA' in regions)
]
print(f"VTA sessions: {len(vta_sessions)}")

# --- Filter by session type ---
# 'session_type' can be 'training', 'biased', or 'ephys'.
biased_sessions = df_sessions[df_sessions['session_type'] == 'biased']
print(f"Biased sessions: {len(biased_sessions)}")

# --- Combine filters ---
# Use & (and) or | (or) to combine boolean masks.
vta_biased = df_sessions[
    (df_sessions['session_type'] == 'biased') &
    (df_sessions['brain_region'].apply(lambda r: 'VTA' in r))
]
print(f"VTA biased sessions: {len(vta_biased)}")

# --- Filter by subject ---
subjects = df_sessions['subject'].unique()
print(f"\nSubjects: {subjects}")
# one_subject = df_sessions[df_sessions['subject'] == subjects[0]]


# ===========================================================================
# 3. Pick a session and find its HDF5 file
# ===========================================================================
# Each session has a unique identifier called an 'eid' (experiment ID).
# After running the pipeline, each session has a corresponding HDF5 file
# at data/sessions/{eid}.h5 .

# Pick the first VTA biased session with an existing HDF5 file.
session = None
for _, row in vta_biased.iterrows():
    fpath = H5_DIR / f"{row['eid']}.h5"
    if fpath.exists():
        session = row
        break

if session is None:
    raise FileNotFoundError(
        "No HDF5 file found for any VTA biased session. "
        "Run the photometry pipeline first."
    )

eid = session['eid']
fpath = H5_DIR / f"{eid}.h5"
print(f"\nLoading session: {eid}")
print(f"Subject: {session['subject']}, type: {session['session_type']}")


# ===========================================================================
# 4. Explore the HDF5 file structure
# ===========================================================================
# HDF5 files are organised like a file system, with groups (folders) and
# datasets (arrays). h5py lets you navigate them using dict-like syntax.

with h5py.File(fpath, 'r') as f:
    print(f"\nTop-level groups in the HDF5 file: {list(f.keys())}")
    print(f"File metadata: {dict(f.attrs)}")

    if 'trials' in f:
        print(f"\nTrial columns: {list(f['trials'].keys())}")
    if 'preprocessed' in f:
        print(f"Photometry regions: {list(f['preprocessed'].keys())}")
    if 'wheel' in f:
        print(f"Wheel metadata: {dict(f['wheel'].attrs)}")


# ===========================================================================
# 5. Load trials
# ===========================================================================
# Trials are stored as individual 1D arrays (one per column) inside the
# 'trials' group. We load them into a pandas DataFrame for easy analysis.

with h5py.File(fpath, 'r') as f:
    trials = pd.DataFrame({
        col: f[f'trials/{col}'][:]   # [:] reads the whole array into numpy
        for col in f['trials']
    })

print(f"\nTrials: {len(trials)} rows")
print(trials.head())

# Key trial columns:
#   stimOn_times      — when the visual stimulus appeared (seconds, session clock)
#   feedback_times    — when reward or punishment was delivered
#   firstMovement_times — when the mouse first moved the wheel
#   choice            — which way the mouse turned: -1 = left, 1 = right, 0 = no-go
#   feedbackType      — 1 = reward, -1 = punishment
#   signed_contrast   — stimulus contrast, negative = left side, positive = right
#   probabilityLeft   — prior probability of stimulus being on the left (0.2 / 0.5 / 0.8)

fraction_correct = (trials['feedbackType'] == 1).mean()
print(f"\nFraction correct: {fraction_correct:.2f}")

# Reaction time = first movement minus stimulus onset
trials['reaction_time'] = trials['firstMovement_times'] - trials['stimOn_times']
print(f"Median reaction time: {trials['reaction_time'].median() * 1000:.0f} ms")


# ===========================================================================
# 6. Load photometry signal
# ===========================================================================
# The preprocessed signal is stored in the 'preprocessed' group.
# Each column is a brain region. The shared time axis is in 'times'.
# The signal is in z-score units (mean 0, std 1).

with h5py.File(fpath, 'r') as f:
    times = f['times'][:]                          # 1D array of timestamps (seconds)
    photometry = pd.DataFrame({
        region: f[f'preprocessed/{region}'][:]
        for region in f['preprocessed']
    }, index=times)

print(f"\nPhotometry signal: {photometry.shape[0]} samples x {photometry.shape[1]} regions")
print(f"Sampling rate: {1 / np.diff(times).mean():.1f} Hz")
print(f"Duration: {times[-1] - times[0]:.0f} s")
print(f"Regions: {photometry.columns.tolist()}")


# ===========================================================================
# 7. Load pre-computed per-trial responses
# ===========================================================================
# The 'responses' group contains peri-event response matrices — snippets of
# photometry signal aligned to each trial event.
#
# Layout: responses/{region}/{event}  →  array of shape (n_trials, n_timepoints)
#         responses/time              →  1D array of timepoints (relative to event)

with h5py.File(fpath, 'r') as f:
    resp_grp = f['responses']
    tpts = resp_grp['time'][:]                       # time axis, e.g. -1 to +1 s
    regions = [k for k in resp_grp.keys() if k != 'time']
    events  = list(resp_grp[regions[0]].keys())

    # Load as a dict: responses[region][event] = (n_trials, n_timepoints) array
    responses = {
        region: {
            event: resp_grp[f'{region}/{event}'][:]
            for event in events
        }
        for region in regions
    }

print(f"\nResponse window: {tpts[0]:.2f} to {tpts[-1]:.2f} s")
print(f"Regions: {regions}")
print(f"Events: {events}")
print(f"Shape per region/event: {responses[regions[0]][events[0]].shape}")

# Example: average response across all trials, for the first region at feedback
region = regions[0]
resp_at_feedback = responses[region]['feedback_times']       # (n_trials, n_timepoints)
mean_response = np.nanmean(resp_at_feedback, axis=0)         # average over trials
print(f"\nMean peak response at feedback ({region}): {mean_response.max():.3f} z")


# ===========================================================================
# 8. Load wheel velocity
# ===========================================================================
# wheel_velocity is a (n_trials, n_samples) matrix.
# Each row is one trial: wheel velocity from stimOn to feedback, in cm/s.
# Trials shorter than the longest trial are NaN-padded on the right.

with h5py.File(fpath, 'r') as f:
    if 'wheel' not in f:
        print("\nNo wheel data in this file — run wheel.py first.")
        wheel_velocity = None
    else:
        wheel_velocity = f['wheel/velocity'][:]      # (n_trials, n_samples)
        wheel_fs = f['wheel'].attrs['fs']            # sampling rate in Hz
        t0_event = f['wheel'].attrs['t0_event']      # 'stimOn_times'
        t1_event = f['wheel'].attrs['t1_event']      # 'feedback_times'

if wheel_velocity is not None:
    # Build a time axis for a single trial (starts at t0_event = 0)
    n_samples = wheel_velocity.shape[1]
    wheel_times = np.arange(n_samples) / wheel_fs    # seconds from stimOn

    print(f"\nWheel velocity: {wheel_velocity.shape} (trials x samples)")
    print(f"Sampling rate: {wheel_fs} Hz")
    print(f"Aligned from '{t0_event}' to '{t1_event}'")

    # How many trials have any movement (max |velocity| > threshold)?
    max_velocity = np.nanmax(np.abs(wheel_velocity), axis=1)
    n_moving = (max_velocity > 10).sum()
    print(f"Trials with movement (>10 cm/s): {n_moving}/{len(trials)}")


# ===========================================================================
# 9. Plot: photometry response at feedback, split by reward vs punishment
# ===========================================================================

region = regions[0]
resp = responses[region]['feedback_times']     # (n_trials, n_timepoints)

reward_mask  = trials['feedbackType'].values == 1
punish_mask  = trials['feedbackType'].values == -1

reward_mean  = np.nanmean(resp[reward_mask],  axis=0)
punish_mean  = np.nanmean(resp[punish_mask],  axis=0)
reward_sem   = np.nanstd(resp[reward_mask],   axis=0) / np.sqrt(reward_mask.sum())
punish_sem   = np.nanstd(resp[punish_mask],   axis=0) / np.sqrt(punish_mask.sum())

fig, ax = plt.subplots(figsize=(6, 4))
ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)   # event time
ax.axhline(0, color='gray', linestyle='-',  linewidth=0.5)

ax.fill_between(tpts, reward_mean - reward_sem, reward_mean + reward_sem,
                alpha=0.3, color='green')
ax.fill_between(tpts, punish_mean - punish_sem, punish_mean + punish_sem,
                alpha=0.3, color='red')
ax.plot(tpts, reward_mean, color='green', label=f'Reward (n={reward_mask.sum()})')
ax.plot(tpts, punish_mean, color='red',   label=f'Punishment (n={punish_mask.sum()})')

ax.set_xlabel('Time from feedback (s)')
ax.set_ylabel('Photometry signal (z-score)')
ax.set_title(f'{region}  —  {session["subject"]}  ({session["session_type"]})')
ax.legend()
fig.tight_layout()
plt.show()


# ===========================================================================
# 10. Plot: wheel velocity on reward vs punishment trials
# ===========================================================================

if wheel_velocity is not None:
    reward_vel = wheel_velocity[reward_mask]
    punish_vel = wheel_velocity[punish_mask]

    reward_vel_mean = np.nanmean(reward_vel, axis=0)
    punish_vel_mean = np.nanmean(punish_vel, axis=0)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    ax.plot(wheel_times, reward_vel_mean, color='green', label='Reward')
    ax.plot(wheel_times, punish_vel_mean, color='red',   label='Punishment')
    ax.set_xlabel('Time from stimulus onset (s)')
    ax.set_ylabel('Wheel velocity (cm/s)')
    ax.set_title(f'Wheel velocity  —  {session["subject"]}')
    ax.legend()
    fig.tight_layout()
    plt.show()
