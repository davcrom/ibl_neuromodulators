"""
Wheel Velocity Pipeline

For each session that has an existing HDF5 file:
1. Load trials (stimOn_times, feedback_times) from the HDF5 file
2. Download wheel position + timestamps from ONE and compute velocity
3. Extract per-trial wheel velocity (stimOn → feedback), NaN-padded to longest trial
4. Append wheel/velocity to the HDF5 file

Input:  metadata/sessions.pqt, data/sessions/{eid}.h5 (created by photometry.py)
Output: data/sessions/{eid}.h5 (wheel/ group appended), metadata/wheel_log.pqt
"""
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from iblnm.config import (
    SESSIONS_FPATH, SESSIONS_H5_DIR, WHEEL_LOG_FPATH,
    SESSION_TYPES_TO_ANALYZE, SUBJECTS_TO_EXCLUDE
)
from iblnm.io import _get_default_connection
from iblnm.util import make_log_entry, LOG_COLUMNS, collect_session_errors
from iblnm.data import PhotometrySession


def run_wheel_pipeline(df_sessions, one=None, h5_dir=None, verbose=True):
    """Extract and save per-trial wheel velocity for all sessions.

    Parameters
    ----------
    df_sessions : pd.DataFrame
        Session metadata rows to process.
    one : ONE, optional
        Live ONE connection. Created from default settings if None.
    h5_dir : Path, optional
        Directory containing session HDF5 files. Defaults to SESSIONS_H5_DIR.
    verbose : bool
        Show tqdm progress bar.

    Returns
    -------
    pd.DataFrame
        Error log (schema: eid, error_type, error_message, traceback).
    """
    if one is None:
        one = _get_default_connection()
    if h5_dir is None:
        h5_dir = SESSIONS_H5_DIR

    h5_dir = Path(h5_dir)
    error_log = []

    for _, session_series in tqdm(df_sessions.iterrows(), total=len(df_sessions),
                                  disable=not verbose, desc="Wheel"):
        eid = session_series['eid']
        fpath = h5_dir / f'{eid}.h5'

        if not fpath.exists():
            error_log.append(make_log_entry(
                eid, error_type='MissingH5',
                error_message=f'{fpath} not found — run photometry.py first',
            ))
            continue

        try:
            ps = PhotometrySession(session_series, one=one)
            ps.load_h5(fpath, groups=['trials'])
            ps.load_wheel()
            ps.extract_wheel_velocity()
            ps.save_h5(fpath, groups=['wheel'], mode='a')
        except Exception as e:
            error_log.append(make_log_entry(eid, error=e))

    return (pd.DataFrame(error_log) if error_log
            else pd.DataFrame(columns=LOG_COLUMNS))


if __name__ == '__main__':
    print(f"Loading sessions from {SESSIONS_FPATH}")
    df_sessions = pd.read_parquet(SESSIONS_FPATH)
    print(f"Loaded {len(df_sessions)} sessions")

    # Filter to sessions that are valid for photometry analysis
    df_sessions = df_sessions[
        df_sessions['session_type'].isin(SESSION_TYPES_TO_ANALYZE) &
        ~df_sessions['subject'].isin(SUBJECTS_TO_EXCLUDE)
    ]

    one = _get_default_connection()
    df_log = run_wheel_pipeline(df_sessions, one=one)

    WHEEL_LOG_FPATH.parent.mkdir(parents=True, exist_ok=True)
    df_log.to_parquet(WHEEL_LOG_FPATH)
    print(f"\nSaved error log to {WHEEL_LOG_FPATH}")
    if len(df_log) > 0:
        print(f"  {len(df_log)} sessions with errors")
        print(f"  Error types:\n{df_log['error_type'].value_counts().to_string()}")
