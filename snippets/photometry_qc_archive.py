"""
Archived code from photometry_qc.py - kept for reference.

Contains:
- Response similarity matrix visualization
- Old eval_metric and qc_series implementations
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr
from numpy.lib.stride_tricks import as_strided
import matplotlib.pyplot as plt


#### RESPONSE SIMILARITY MATRIX ##############################################

def plot_response_similarity_matrix(df, session_types=['biased', 'ephys']):
    """
    Plot correlation matrix of response vectors across sessions.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: subject, target, AP, ML, DV, session_type, response_vector
    session_types : list
        Session types to include
    """
    df_filtered = df[df['session_type'].isin(session_types)].copy()

    if len(df_filtered) == 0:
        print("No sessions found matching criteria")
        return

    # Take absolute value of ML for sorting
    df_filtered['ML_abs'] = df_filtered['ML'].abs()

    # Sort by coordinates first, then by subject within each coordinate group
    df_filtered = df_filtered.sort_values(['AP', 'ML_abs', 'DV', 'subject']).reset_index(drop=True)

    print(f"Total rows = {len(df_filtered)}")
    print(f"Unique subjects: {df_filtered['subject'].nunique()}")

    if len(df_filtered) < 2:
        print("Not enough sessions to create correlation matrix")
        return

    n = len(df_filtered)

    # Compute correlation matrix
    corr_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                vec_i = df_filtered.iloc[i]['response_vector']
                vec_j = df_filtered.iloc[j]['response_vector']

                # Find indices where both vectors are not NaN
                valid_mask = ~(np.isnan(vec_i) | np.isnan(vec_j))

                if valid_mask.sum() < 2:
                    corr_matrix[i, j] = np.nan
                else:
                    corr_matrix[i, j] = pearsonr(
                        vec_i[valid_mask],
                        vec_j[valid_mask]
                    )[0]

    # Plot
    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    # Add labels
    labels = [
        f"{row['subject']} AP:{row['AP']:.0f} |ML|:{row['ML_abs']:.0f} DV:{row['DV']:.0f}"
        for _, row in df_filtered.iterrows()
    ]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, ha='right', fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)

    # Add separator lines
    prev_coords = None
    prev_subject = None
    for i in range(n):
        curr_coords = (df_filtered.iloc[i]['AP'], df_filtered.iloc[i]['ML_abs'])
        curr_subject = df_filtered.iloc[i]['subject']

        if prev_coords is not None:
            if curr_coords != prev_coords:
                ax.axhline(y=i-0.5, color='black', linestyle='--', linewidth=2)
                ax.axvline(x=i-0.5, color='black', linestyle='--', linewidth=2)
            elif curr_subject != prev_subject:
                ax.axhline(y=i-0.5, color='gray', linestyle='--', linewidth=2)
                ax.axvline(x=i-0.5, color='gray', linestyle='--', linewidth=2)

        prev_coords = curr_coords
        prev_subject = curr_subject

    plt.colorbar(im, ax=ax, label='Correlation')
    plt.title(f'Response Vector Correlation Matrix\n(n={n})', fontsize=12)
    plt.tight_layout()

    return fig, corr_matrix


#### OLD SLIDING METRIC IMPLEMENTATION #######################################

def eval_metric(
    F: pd.Series,
    metric: callable,
    metric_kwargs: dict = {},
    sliding_kwargs: dict = {},
    detrend: bool = True
) -> dict:
    """
    Evaluate a metric on a time series, optionally with sliding window analysis.

    Parameters
    ----------
    F : pd.Series
        Input time series data
    metric : Callable
        Metric function to evaluate
    metric_kwargs : dict, optional
        Arguments to pass to the metric function
    sliding_kwargs : dict, optional
        Sliding window parameters. Expected keys: 'w_len' (window length)
    detrend : bool
        Whether to detrend within each window

    Returns
    -------
    dict : Results dictionary with keys:
        - 'value': metric evaluated on full signal
        - 'sliding_values': metric values for each window
        - 'sliding_timepoints': timepoints for each window
        - 'r': correlation coefficient of sliding values vs time
        - 'p': p-value for the correlation
    """
    results_vals = ['value', 'sliding_values', 'sliding_timepoints', 'r', 'p']
    result = {k: np.nan for k in results_vals}

    # Always calculate the full signal metric
    result['value'] = metric(F.values, **metric_kwargs)

    # Determine windowing parameters
    if sliding_kwargs and 'w_len' in sliding_kwargs:
        dt = np.median(np.diff(F.index))
        w_len = sliding_kwargs['w_len']
        w_size = int(w_len // dt)
        step_size = int(w_size // 2)
        n_windows = int((len(F) - w_size) // step_size + 1)

        if n_windows <= 2:
            return result

        S_times = F.index.values[
            np.linspace(step_size, n_windows * step_size, n_windows).astype(int)
        ]

        a = F.values
        windows = as_strided(
            a,
            shape=(n_windows, w_size),
            strides=(step_size * a.strides[0], a.strides[0])
        )

        if detrend:
            def _metric(w, **metric_kwargs):
                x = np.arange(len(w))
                slope, intercept = stats.linregress(x, w)[:2]
                w_detrended = w - (slope * x + intercept)
                return metric(w_detrended, **metric_kwargs)
        else:
            _metric = metric

        S_values = np.apply_along_axis(
            lambda w: _metric(w, **metric_kwargs), axis=1, arr=windows
        )

        result['sliding_values'] = S_values
        result['sliding_timepoints'] = S_times

        if n_windows > 1:
            result['r'], result['p'] = stats.linregress(S_times, S_values)[2:4]

    return result


def qc_series(
    F: pd.Series,
    metrics: dict,
    sliding_kwargs=None,
    trials=None,
    eid: str = None,
    brain_region: str = None,
    detrend: bool = False,
) -> dict:
    """
    Run multiple QC metrics on a single time series.

    Parameters
    ----------
    F : pd.Series
        Input time series
    metrics : dict
        Dict mapping metric functions to their kwargs
    sliding_kwargs : dict, optional
        Sliding window parameters
    trials : pd.DataFrame, optional
        Trial data to pass to behavior-related metrics
    eid : str, optional
        Session ID for logging
    brain_region : str, optional
        Brain region for logging
    detrend : bool
        Whether to detrend within windows

    Returns
    -------
    dict : QC results for each metric
    """
    if isinstance(F, pd.DataFrame):
        raise TypeError('F cannot be a dataframe')

    qc_results = {}
    for metric, params in metrics.items():
        if trials is not None:
            params['trials'] = trials
        qc_results[metric.__name__] = eval_metric(
            F, metric, params, sliding_kwargs, detrend=detrend
        )
    return qc_results
