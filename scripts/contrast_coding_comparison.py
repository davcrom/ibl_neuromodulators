"""Compare log, linear, and rank contrast coding for LMM fits.

Fits the same LMM (response ~ contrast * side * reward | subject) under each
contrast transform and reports marginal R², AIC, and BIC per target_NM × event.

Usage:
    python scripts/contrast_coding_comparison.py
"""
import numpy as np
import pandas as pd

from iblnm.config import (
    SESSIONS_FPATH, RESPONSES_FPATH, RESPONSE_EVENTS, ANALYSIS_CONTRASTS,
)
from iblnm.data import PhotometrySessionGroup
from iblnm.io import _get_default_connection
from iblnm.task import add_relative_contrast
from iblnm.util import derive_target_nm
from iblnm.analysis import fit_response_lmm

CODINGS = ['log', 'linear', 'rank']
RESPONSE_COL = 'response_early'


def prepare_data(group):
    """Filter and prepare trial-level data for LMM fitting."""
    df = add_relative_contrast(group.response_magnitudes.copy())
    df = df.query('probabilityLeft == 0.5').dropna(subset=[RESPONSE_COL])
    df = df.query('choice != 0')
    return df


def compare_contrast_codings(df, min_subjects=2):
    """Fit LMMs under each contrast coding and collect fit metrics.

    Returns
    -------
    pd.DataFrame
        One row per (target_NM, event, coding) with R²_marginal, AIC, BIC.
    """
    rows = []
    for (target_nm, event), df_group in df.groupby(['target_NM', 'event']):
        if df_group['subject'].nunique() < min_subjects:
            continue
        event_label = event.replace('_times', '')
        for coding in CODINGS:
            lmm = fit_response_lmm(df_group, RESPONSE_COL,
                                    contrast_coding=coding)
            if lmm is None:
                continue
            rows.append({
                'target_NM': target_nm,
                'event': event_label,
                'coding': coding,
                'R2_marginal': lmm.variance_explained['marginal'],
                'AIC': lmm.result.aic,
                'BIC': lmm.result.bic,
            })
    return pd.DataFrame(rows)


def print_tables(df_metrics, metric='AIC'):
    """Print one pivot table per event with target_NM as rows, codings as columns."""
    for event in sorted(df_metrics['event'].unique()):
        subset = df_metrics[df_metrics['event'] == event]
        table = subset.pivot(index='target_NM', columns='coding', values=metric)
        table = table[CODINGS]  # enforce column order

        # Mark best (lowest AIC/BIC, highest R²) per row
        if metric == 'R2_marginal':
            best = table.idxmax(axis=1)
        else:
            best = table.idxmin(axis=1)

        # Format and annotate
        formatted = table.copy()
        for col in CODINGS:
            if metric == 'R2_marginal':
                formatted[col] = table[col].map(
                    lambda x: f'{x:.4f}' if pd.notna(x) else '—')
            else:
                formatted[col] = table[col].map(
                    lambda x: f'{x:.1f}' if pd.notna(x) else '—')
        for idx in formatted.index:
            b = best[idx]
            if pd.notna(b):
                formatted.at[idx, b] = formatted.at[idx, b] + ' *'

        print(f'\n{metric} — {event}')
        print(formatted.to_string())


if __name__ == '__main__':
    # Load sessions
    print(f'Loading sessions from {SESSIONS_FPATH}')
    df_sessions = pd.read_parquet(SESSIONS_FPATH)
    df_sessions = derive_target_nm(df_sessions)

    # Drop mismatched parallel-list sessions
    _parallel_cols = ['target_NM', 'brain_region', 'hemisphere']
    _lengths_match = df_sessions[_parallel_cols].apply(
        lambda row: len(set(
            len(v) if isinstance(v, (list, np.ndarray)) else 1
            for v in row
        )) == 1,
        axis=1,
    )
    df_sessions = df_sessions[_lengths_match].copy()

    # Explode to recordings
    df_recordings = df_sessions.explode(_parallel_cols).copy()
    df_recordings['fiber_idx'] = df_recordings.groupby('eid').cumcount()

    # Filter
    one = _get_default_connection()
    group = PhotometrySessionGroup(df_recordings, one=one)
    group.filter_recordings(session_types=('biased', 'ephys'))

    # Load pre-extracted responses
    print(f'Loading responses from {RESPONSES_FPATH}')
    df_responses = pd.read_parquet(RESPONSES_FPATH)
    df_responses = df_responses[df_responses['eid'].isin(group.recordings['eid'])]
    df_responses = df_responses[
        df_responses['contrast'].isin(ANALYSIS_CONTRASTS)
    ].copy()
    group.response_magnitudes = df_responses

    # Compare
    print('Fitting LMMs for each contrast coding...')
    df_trial = prepare_data(group)
    df_metrics = compare_contrast_codings(df_trial)

    # Print results
    for metric in ['AIC', 'BIC', 'R2_marginal']:
        print_tables(df_metrics, metric=metric)
