"""
Download trial data for all sessions from a specific target and neuromodulator.

Usage:
    python scripts/download_trials.py --target LC --nm NE
    python scripts/download_trials.py --target LC --nm NE --session_type biased
    python scripts/download_trials.py --target LC --nm NE --session_type biased training
"""

import argparse
import sys
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from one.api import ONE

from iblnm.config import *
from iblnm.data import PhotometrySession
from iblnm.io import exception_logger
from iblnm.util import df2pqt

# Initialize ONE
one = ONE()

# Parse command line arguments
parser = argparse.ArgumentParser(description='Download trial data for sessions from a specific target and neuromodulator')
parser.add_argument('--target', type=str, required=True,
                    help='Target to download (e.g., VTA, SNc, DR, LC, NBM)')
parser.add_argument('--nm', type=str, required=True,
                    help='Neuromodulator (e.g., DA, 5HT, NE, ACh)')
parser.add_argument('--session_type', type=str, nargs='+', required=False, default=None,
                    help='Session type(s) to filter (e.g., training, biased, ephys). Can specify multiple types. If not specified, all session types are included.')
args = parser.parse_args()

target = args.target
nm = args.nm
session_types = args.session_type

if session_types:
    print(f"Downloading trial data for target: {target}, NM: {nm}, session_types: {session_types}")
else:
    print(f"Downloading trial data for target: {target}, NM: {nm}")

# Load sessions
print("\nLoading sessions...")
df_sessions = pd.read_parquet(SESSIONS_FPATH)

# Filter for the specified target and NM
mask = df_sessions.apply(
    lambda x: (target in x['target']) and (x['NM'] == nm) and (x['session_type'] in session_types),
    axis='columns'
)
df_filtered = df_sessions[mask].copy()
print(f"Found {len(df_filtered)} sessions for target={target}, NM={nm}")

if len(df_filtered) == 0:
    print(f"\nNo sessions found for target={target}, NM={nm}")

    # Show available combinations
    print("\nAvailable target-NM combinations:")
    unique_combinations = set()
    for idx, row in df_sessions.iterrows():
        if pd.notna(row['target']) and pd.notna(row['NM']):
            targets = row['target'] if isinstance(row['target'], list) else [row['target']]
            for t in targets:
                unique_combinations.add((t, row['NM']))
    for t, n in sorted(unique_combinations):
        print(f"  --target {t} --nm {n}")


@exception_logger
def download_trials(session_series):
    """Download trial data for a session using PhotometrySession class."""
    # Create PhotometrySession instance (without loading photometry data)
    session = PhotometrySession(session_series, one=one, load_data=False)

    # Load trials using the PhotometrySession method
    session.load_trials()

    # Add eid and subject columns to trials
    df_trials = session.trials.copy()
    df_trials['eid'] = session.eid
    df_trials['subject'] = session.subject

    return df_trials

# Download trials for each session
print("\nDownloading trial data...")
trials_list = []
exlog = []  # initialize exception log
for idx, session_series in tqdm(df_filtered.iterrows(), total=len(df_filtered)):
    df_trials = download_trials(session_series, exlog=exlog)
    if df_trials is not None:
        # Add trial_number column before appending
        df_trials['trial_number'] = df_trials.index
        trials_list.append(df_trials)

# Stack all trials into a single dataframe
if trials_list:
    df_trials_all = pd.concat(trials_list, ignore_index=True)
else:
    df_trials_all = pd.DataFrame()

# Count successful downloads
n_sessions_with_trials = len(trials_list)
n_exceptions = len(exlog)
n_total = len(df_filtered)

print(
    f"\nFinished processing {n_total} sessions:"
    f"\n{n_sessions_with_trials} successful"
    f"\n{n_exceptions} exceptions"
    f"\nTotal trials: {len(df_trials_all)}"
)

# Save trial data
output_path = PROJECT_ROOT / f'data/trials_{target}-{nm}.pqt'
df2pqt(df_trials_all, output_path)
print(f"\nTrial data saved to: {output_path}")

# Save exceptions
if exlog:
    df_exceptions = pd.DataFrame(exlog)
    exception_path = PROJECT_ROOT / f'data/trials_{target}-{nm}_log.pqt'
    df2pqt(df_exceptions, exception_path)
    print(f"Exceptions saved to: {exception_path}")
