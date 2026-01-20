"""
Task Performance Analysis

Computes for each session:
- Sessions to reach each training stage (training > biased > ephys)
- Overall performance (fraction correct)
- Performance on easy trials (>= 50% contrast)
- Psychometric function parameters (bias, threshold, lapse_high, lapse_low)
"""
import pandas as pd
import numpy as np
from tqdm import tqdm

from brainbox.behavior.training import compute_psychometric, compute_performance_easy

from iblnm.config import SESSIONS_CLEAN_FPATH
from iblnm.io import _get_default_connection


def compute_sessions_to_stage(df_sessions):
    """
    Compute number of sessions to reach each training stage per subject.

    Parameters
    ----------
    df_sessions : pd.DataFrame
        Sessions dataframe with 'subject', 'session_type', 'date' columns

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: subject, sessions_to_biased, sessions_to_ephys
    """
    results = []

    for subject, group in df_sessions.groupby('subject'):
        # Sort by date
        group = group.sort_values('date')

        # Find first session of each type
        training_sessions = group[group['session_type'] == 'training']
        biased_sessions = group[group['session_type'] == 'biased']
        ephys_sessions = group[group['session_type'] == 'ephys']

        n_training = len(training_sessions)
        n_to_biased = n_training if len(biased_sessions) > 0 else np.nan
        n_to_ephys = n_training + len(biased_sessions) if len(ephys_sessions) > 0 else np.nan

        results.append({
            'subject': subject,
            'n_training': n_training,
            'n_biased': len(biased_sessions),
            'n_ephys': len(ephys_sessions),
            'sessions_to_biased': n_to_biased,
            'sessions_to_ephys': n_to_ephys,
        })

    return pd.DataFrame(results)


def compute_session_performance(df_sessions, one=None, verbose=True):
    """
    Compute performance metrics for each session.

    Parameters
    ----------
    df_sessions : pd.DataFrame
        Sessions dataframe
    one : ONE, optional
        ONE API instance
    verbose : bool
        Print progress

    Returns
    -------
    pd.DataFrame
        Sessions dataframe with added columns:
        - performance: overall fraction correct
        - performance_easy: fraction correct on easy trials
        - psych_bias: psychometric bias parameter
        - psych_threshold: psychometric threshold parameter
        - psych_lapse_high: psychometric lapse high parameter
        - psych_lapse_low: psychometric lapse low parameter
    """
    if one is None:
        one = _get_default_connection()

    df_sessions = df_sessions.copy()

    # Initialize columns
    df_sessions['performance'] = np.nan
    df_sessions['performance_easy'] = np.nan
    df_sessions['psych_bias'] = np.nan
    df_sessions['psych_threshold'] = np.nan
    df_sessions['psych_lapse_high'] = np.nan
    df_sessions['psych_lapse_low'] = np.nan

    for idx, row in tqdm(df_sessions.iterrows(), total=len(df_sessions),
                         disable=not verbose, desc="Computing performance"):
        eid = row['eid']

        try:
            # Load trials
            trials = one.load_object(eid, 'trials', collection='alf')
        except Exception:
            try:
                trials = one.load_object(eid, 'trials', collection='alf/task_00')
            except Exception:
                continue

        # Check required columns exist
        required = ['contrastLeft', 'contrastRight', 'feedbackType', 'choice', 'probabilityLeft']
        if not all(hasattr(trials, col) for col in required):
            continue

        try:
            # Overall performance
            n_correct = np.sum(trials.feedbackType == 1)
            n_total = len(trials.feedbackType)
            df_sessions.loc[idx, 'performance'] = n_correct / n_total if n_total > 0 else np.nan

            # Performance on easy trials
            df_sessions.loc[idx, 'performance_easy'] = compute_performance_easy(trials)

            # Psychometric fit
            psych = compute_psychometric(trials, plotting=True)
            df_sessions.loc[idx, 'psych_bias'] = psych[0]
            df_sessions.loc[idx, 'psych_threshold'] = psych[1]
            df_sessions.loc[idx, 'psych_lapse_high'] = psych[2]
            df_sessions.loc[idx, 'psych_lapse_low'] = psych[3]

        except Exception:
            continue

    return df_sessions


if __name__ == '__main__':
    # Load cleaned sessions
    df_sessions = pd.read_parquet(SESSIONS_CLEAN_FPATH)

    # Compute sessions to reach each stage
    df_stages = compute_sessions_to_stage(df_sessions)
    print("\n=== Sessions to Training Stage ===")
    print(df_stages.describe())

    # Compute performance metrics
    one = _get_default_connection()
    df_sessions = compute_session_performance(df_sessions, one=one)

    # Save updated sessions
    df_sessions.to_parquet(SESSIONS_CLEAN_FPATH)

    # Summary
    print("\n=== Performance Summary ===")
    print(f"Sessions with performance data: {df_sessions['performance'].notna().sum()}")
    print(f"Mean performance: {df_sessions['performance'].mean():.3f}")
    print(f"Mean performance (easy): {df_sessions['performance_easy'].mean():.3f}")
    print(f"Mean psychometric bias: {df_sessions['psych_bias'].mean():.2f}")
    print(f"Mean psychometric threshold: {df_sessions['psych_threshold'].mean():.2f}")
