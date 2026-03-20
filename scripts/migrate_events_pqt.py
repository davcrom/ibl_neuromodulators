"""Split the old data/events.pqt into three separate parquets.

Reads the monolithic events.pqt (which contains response magnitudes, trial
timing, and peak velocity in one DataFrame) and writes:
    data/responses.pqt     — response magnitudes only
    data/trial_timing.pqt  — reaction_time, movement_time per trial
    data/peak_velocity.pqt — peak_velocity per trial

Safe to run multiple times (overwrites output files).
"""
import pandas as pd
from iblnm.config import (
    PROJECT_ROOT, RESPONSES_FPATH, TRIAL_TIMING_FPATH, PEAK_VELOCITY_FPATH,
)

old_fpath = PROJECT_ROOT / 'data/events.pqt'
if not old_fpath.exists():
    raise SystemExit(f"Error: {old_fpath} not found.")

df = pd.read_parquet(old_fpath)
print(f"Loaded {len(df)} rows from {old_fpath}")

# 1. Trial timing — keyed by (eid, trial), no event dimension
timing_cols = ['eid', 'trial', 'reaction_time', 'movement_time']
have = [c for c in timing_cols if c in df.columns]
trial_timing = df[have].drop_duplicates(subset=['eid', 'trial']).copy()
trial_timing.to_parquet(TRIAL_TIMING_FPATH, index=False)
print(f"Saved {len(trial_timing)} rows to {TRIAL_TIMING_FPATH}")

# 2. Peak velocity (deduplicate across events)
if 'peak_velocity' in df.columns:
    peak_velocity = df[['eid', 'trial', 'peak_velocity']].drop_duplicates().copy()
    peak_velocity.to_parquet(PEAK_VELOCITY_FPATH, index=False)
    print(f"Saved {len(peak_velocity)} rows to {PEAK_VELOCITY_FPATH}")
else:
    print("No peak_velocity column found — skipping peak_velocity.pqt")

# 3. Response magnitudes (drop timing/velocity columns)
drop_cols = ['reaction_time', 'movement_time', 'peak_velocity']
responses = df.drop(columns=[c for c in drop_cols if c in df.columns])
responses.to_parquet(RESPONSES_FPATH, index=False)
print(f"Saved {len(responses)} rows to {RESPONSES_FPATH}")

print("\nDone. You can now run: python scripts/responses.py --plot")
