"""
Simple script to fit a PsyTrack model to one example IBL session.

This script loads one session from metadata/sessions.pqt, fetches the trial data
using the ONE API, and fits a PsyTrack GLM model to predict the evolution of
weights for different predictors (bias, left contrast, right contrast).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psytrack as psy
from one.api import ONE

from iblnm.config import SESSIONS_FPATH
from iblnm.data import PhotometrySession

# Initialize ONE connection
print("Connecting to database...")
one = ONE()

# Load sessions metadata
print("Loading sessions metadata...")
df_sessions = pd.read_parquet(SESSIONS_FPATH)

# Pick the first available session
example_session = df_sessions.iloc[0]
eid = example_session['eid']
subject = example_session['subject']

print(f"\nSelected session:")
print(f"  EID: {eid}")
print(f"  Subject: {subject}")
print(f"  Date: {example_session.get('start_time', 'N/A')}")

# Load trial data for this session using PhotometrySession
print("\nLoading trial data...")
try:
    # Create PhotometrySession instance (without loading photometry data)
    session = PhotometrySession(example_session, one=one, load_data=False)

    # Load trials
    session.load_trials()

    # Get trials dataframe
    df_trials = session.trials.copy()

    # Remove NaN trials
    df_trials = df_trials.dropna(subset=['choice']).reset_index(drop=True)

    print(f"Loaded {len(df_trials)} trials")
    print(f"\nFirst few trials:")
    print(df_trials.head())

except Exception as e:
    print(f"Error loading trial data: {e}")
    print("Please check that the session has valid trial data.")
    exit(1)

# Prepare data for PsyTrack
print("\nPreparing data for PsyTrack...")

# Convert choice: 1 = left, 2 = right (PsyTrack convention from notebooks)
y = np.array([2 if c < 0 else 1 for c in df_trials['choice']])

# Prepare inputs (take absolute values of contrasts)
inputs = {
    's1': np.abs(df_trials['contrastRight'].fillna(0)).values.reshape(-1, 1),
    's2': np.abs(df_trials['contrastLeft'].fillna(0)).values.reshape(-1, 1)
}

# Set dayLength (all trials in one session)
dayLength = len(df_trials)

# Create input dictionary
input_dict = {
    'y': y,
    'inputs': inputs,
    'dayLength': dayLength
}

# Define weights
weights = {
    'bias': 1,  # intercept
    's1': 1,    # right contrast weight
    's2': 1     # left contrast weight
}
K = sum(weights.values())  # total number of weights

# Define hyperparameters
hyper = {
    'sigInit': 2**4,          # initial sigma
    'sigma': [2**-6] * K,     # how weights can evolve over trials
    'sigDay': dayLength       # allows weight jumps at day boundaries
}

# Specify which hyperparameters to optimize
optList = ['sigma', 'sigDay']

# Fit PsyTrack model
print("\nFitting PsyTrack model...")
print("This may take a minute...")
hyp, evd, wMode, hess_info = psy.hyperOpt(input_dict, hyper, weights, optList)

print("\nModel fitting complete!")
print(f"Log-evidence: {evd:.2f}")
print(f"Optimized sigma: {hyp['sigma']}")

# Create weights dataframe for visualization
df_weights = pd.DataFrame(wMode).T.rename(columns={0: 'bias', 1: 'right', 2: 'left'})

print(f"\nWeight statistics:")
print(df_weights.describe())

# Plot weight evolution
print("\nGenerating plot...")
fig, ax = plt.subplots(figsize=(12, 5))
ax.axhline(y=0, color='grey', linestyle='--', linewidth=1)

ax.plot(df_weights.index, df_weights['right'], label='right contrast', color='#C86464', linewidth=2)
ax.plot(df_weights.index, df_weights['left'], label='left contrast', color='#5D94D4', linewidth=2)
ax.plot(df_weights.index, df_weights['bias'], label='bias', color='#FBB03B', linewidth=2)

ax.set_xlabel('Trial number', fontsize=12)
ax.set_ylabel('Weight', fontsize=12)
ax.set_title(f'PsyTrack weight evolution - {subject} - Session {eid[:8]}...', fontsize=14)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()

# Save figure
output_path = 'psytrack_example_fit.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Plot saved to: {output_path}")

plt.show()

print("\nDone!")
