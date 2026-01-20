"""
Photometry QC Pipeline

Runs quality control metrics on photometry sessions with:
- Session filtering (excluded subjects/types)
- Validate-as-you-go data loading via PhotometrySession
- Data availability flags (has_trials, has_photometry, trials_in_photometry_time)
- Error logging
- Results caching
"""
import traceback
import pandas as pd
from tqdm import tqdm

from iblnm.config import SESSIONS_CLEAN_FPATH, QCPHOTOMETRY_FPATH, QCPHOTOMETRY_LOG_FPATH
from iblnm.io import _get_default_connection
from iblnm.data import PhotometrySession


def run_qc_pipeline(df_sessions, one=None, verbose=True):
    """
    Run QC on all sessions with validate-as-you-go approach.

    Parameters
    ----------
    df_sessions : pd.DataFrame
        Sessions to process (should already be cleaned)
    one : ONE, optional
        ONE API instance
    verbose : bool
        Print progress

    Returns
    -------
    df_sessions : pd.DataFrame
        Sessions with data availability flags added
    df_qc : pd.DataFrame
        QC results from all sessions
    df_log : pd.DataFrame
        Error log for QC failures
    """
    if one is None:
        one = _get_default_connection()

    df_sessions = df_sessions.copy()
    qc_results = []
    exlog = []

    # Initialize flag columns
    df_sessions['has_trials'] = False
    df_sessions['has_photometry'] = False
    df_sessions['trials_in_photometry_time'] = False

    for idx, session_series in tqdm(df_sessions.iterrows(), total=len(df_sessions),
                                     disable=not verbose, desc="QC"):
        eid = session_series['eid']
        subject = session_series['subject']

        # Load session data (sets has_trials, has_photometry, trials_in_photometry_time)
        ps = PhotometrySession(session_series, one=one)

        # Update flags in df_sessions
        flags = ps.get_data_flags()
        for flag_name, flag_value in flags.items():
            df_sessions.loc[idx, flag_name] = flag_value

        # Only run QC if both trials and photometry loaded
        if not ps.has_photometry:
            continue

        try:
            qc_results.append(ps.run_qc())
        except Exception as e:
            exlog.append({
                'eid': eid,
                'subject': subject,
                'exception_type': type(e).__name__,
                'exception_message': str(e),
                'traceback': traceback.format_exc()
            })

    # Combine QC results
    if qc_results:
        df_qc = pd.concat(qc_results, ignore_index=True)
    else:
        df_qc = pd.DataFrame()

    # Convert error log to DataFrame
    df_log = pd.DataFrame(exlog) if exlog else pd.DataFrame(
        columns=['eid', 'subject', 'exception_type', 'exception_message', 'traceback']
    )

    return df_sessions, df_qc, df_log


if __name__ == '__main__':
    # Load cleaned sessions (from dataset_overview.py)
    df_sessions = pd.read_parquet(SESSIONS_CLEAN_FPATH)

    # Filter to good sessions only (have both task and photometry data)
    df_sessions = df_sessions.query('session_status == "good"').copy()

    # Run QC pipeline
    one = _get_default_connection()
    df_sessions, df_qc, df_log = run_qc_pipeline(df_sessions, one=one)

    # Save results
    QCPHOTOMETRY_FPATH.parent.mkdir(parents=True, exist_ok=True)
    df_qc.to_parquet(QCPHOTOMETRY_FPATH)
    df_log.to_parquet(QCPHOTOMETRY_LOG_FPATH)
    df_sessions.to_parquet(SESSIONS_CLEAN_FPATH)
