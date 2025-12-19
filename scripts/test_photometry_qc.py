"""
Test script to verify PhotometrySession.run_qc() works correctly
"""
import pandas as pd
from one.api import ONE
from iblnm.config import SESSIONS_FPATH
from iblnm.data import PhotometrySession

# Initialize ONE
one = ONE()

# Load sessions
df_sessions = pd.read_parquet(SESSIONS_FPATH)

print(f"Total sessions: {len(df_sessions)}")

# Get a session that has photometry data available
photometry_col = 'alf/photometry/photometry.signal.pqt'
print(f"Photometry column values: {df_sessions[photometry_col].value_counts()}")

# Filter to sessions where photometry is not False/NaN
sessions_with_photometry = df_sessions[df_sessions[photometry_col].notna() & (df_sessions[photometry_col] != False)]
print(f"Sessions with photometry: {len(sessions_with_photometry)}")

if len(sessions_with_photometry) == 0:
    print("No sessions with photometry found! Using first session anyway...")
    test_session = df_sessions.iloc[0]
else:
    # Try multiple sessions in case first doesn't work
    for idx in range(min(20, len(sessions_with_photometry))):
        test_session = sessions_with_photometry.iloc[idx]
        print(f"\nAttempting session {idx}: {test_session['eid']}, Subject: {test_session['subject']}")
        try:
            session = PhotometrySession(test_session, one=one, load_data=False)
            # Only load photometry
            session.load_photometry()
            break  # Success!
        except Exception as e:
            print(f"  Failed: {type(e).__name__}")
            if idx == 19:
                print("Could not load any sessions!")
                exit(1)
            continue

print(f"\nSuccessfully loaded session: {test_session['eid']}")
print(f"Subject: {test_session['subject']}")

print("\nPhotometry data structure:")
print(f"Type: {type(session.photometry)}")
if isinstance(session.photometry, dict):
    print(f"Keys (bands): {session.photometry.keys()}")
    for band, df in session.photometry.items():
        print(f"  {band}: {type(df)}, shape={df.shape}, columns={list(df.columns)}")

print("\nRunning QC...")
try:
    qc_results = session.run_qc()
    print("\nQC completed successfully!")
    print(f"Results shape: {qc_results.shape}")
    print(f"\nResults columns: {qc_results.columns.tolist()}")
    print(f"\nUnique metrics: {qc_results['metric'].unique()}")
    print(f"\nUnique brain regions: {qc_results['brain_region'].unique()}")
    print("\nFirst few rows:")
    print(qc_results.head(10))
except Exception as e:
    print(f"\nError occurred: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
