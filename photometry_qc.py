import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from matplotlib import pyplot as plt

from one.api import ONE
from brainbox.io.one import PhotometrySessionLoader
from one.alf.exceptions import ALFObjectNotFound
from iblphotometry import fpio, metrics
from iblphotometry.qc import run_qc

from iblnm.config import *
from iblnm.data import PhotometrySession


SESSIONS_FNAME = 'sessions_2025-11-07-12h07.pqt'

# TODO: make session file name an arg
df_sessions = pd.read_parquet(f'metadata/{SESSIONS_FNAME}')

# Apply some filtering to sessions
# ~df_sessions = df_sessions.query('subject.str.contains("CQ")').copy()
# ~df_sessions['CQn'] = df_sessions['subject'].apply(lambda x: int(x.lstrip('CQ')))
# ~df_sessions = df_sessions.query('CQn >= 14')
# ~df_sessions = df_sessions.query('NM == "ACh"')
# ~print(df_sessions.head())

raw_metrics = [
    metrics.n_early_samples,
    metrics.n_edges
]

sliding_metrics = [
    metrics.median_absolute_deviance,
    metrics.percentile_distance,
    metrics.percentile_asymmetry,
    metrics.n_outliers,
    metrics.n_expmax_violations,
    metrics.expmax_violation,
    # metrics.bleaching_tau,  # cant be applied sliding, needs to be series
    # ~metrics.spectral_entropy,
    metrics.ar_score
]

# some custom settings for some metrics
metrics_kwargs = {'percentile_asymmetry': {'pc_comp': 75}}

sliding_kwargs = {
    'n_windows': 10,
    'w_len': 120,
    'detrend': True
}

one = ONE()
# ~df_sessions = df_sessions.query('NM == "ACh"')
eids = df_sessions['eid']
print(len(eids))

bad_eids = []
for eid in tqdm(eids):
    try:
        psl = PhotometrySessionLoader(eid=eid, one=one)
        psl.load_photometry()
    except:
        bad_eids.append(eid)
eids = set(eids) - set(bad_eids)
print(len(eids))

qc_raw = run_qc(
    eids,
    one,
    signal_band='GCaMP',
    metrics=raw_metrics,
    n_jobs=-2,
)

# ~qc_tmp = []
# ~for eid in tqdm(eids):
    # ~qc_tmp.append(qc_eid(
        # ~list(eids)[0],
        # ~one,
        # ~signal_band='GCaMP',
        # ~metrics=raw_metrics,
    # ~))

qc_sliding = run_qc(
    eids,
    one,
    signal_band='GCaMP',
    metrics=sliding_metrics,
    metrics_kwargs=metrics_kwargs,
    sliding_kwargs=sliding_kwargs,
    n_jobs=-2,
)

"""
df_coords = pd.read_csv('metadata/NBM_coordinates.csv')

# Get photometry location data
df_coords = df_coords.set_index(['subject', 'brain_region'])
photometry_sessions = []
no_photometry_sessions = []
for _, session in tqdm(df_sessions.iterrows(), total=len(df_sessions)):
    try:
        # Get fiber locations and ROIs
        df_locations = one.load_dataset(id=session['eid'], dataset='photometryROI.locations.pqt')
    except ALFObjectNotFound:
        no_photometry_sessions.append(session)
        continue
    for roi, location in df_locations.iterrows():
        photometry_session = pd.concat([session, df_coords.loc[session['subject'], location['brain_region']]])
        photometry_session['roi'] = roi
        photometry_sessions.append(photometry_session)

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import AnovaRM

qc_cols = [col for col in df_qc.columns if 'qc' in col]
xpos = {'original': 0, 'new_1': 1, 'new_2': 2}
colors = {'CQ014': 'C0', 'CQ015': 'C1', 'CQ016': 'C2'}
fig, axs = plt.subplots(1, len(qc_cols), figsize=(len(qc_cols) * 8, 8))
fig.suptitle('NBM Raw Signal Quality Metrics')
for ax, col in zip(axs, qc_cols):
    print('\n', f'========== {col} ==========')

    print('\n', 'ANOVA Results')
    # print('=======================================================================')
    # Simple ANOVA
    model = smf.ols(f'{col} ~ C(label, Treatment("original"))', data=df_qc)
    result = model.fit()
    # Perform ANOVA
    anova_table = sm.stats.anova_lm(result, typ=2)
    print(anova_table)

    # Linear Mixed-effects Model
    model = smf.mixedlm(f'{col} ~ C(label, Treatment("original"))',  # main effects
                        data=df_qc,
                        groups=df_qc['subject'],  # group by mouse
                        # groups=df_qc['eid'],  # group by session
                        re_formula='~C(label, Treatment("original"))'  # random slopes
                       )
    result = model.fit()
    print(result.summary())

    for (coord, subject), data in df_qc.groupby(['label', 'subject']):
        ax.scatter(xpos[coord] + np.random.uniform(-0.15, 0.15, len(data)), data[col], fc='none', ec=colors[subject], lw=2)
    ax.set_xticks(list(xpos.values()))
    ax.set_xticklabels(list(xpos.keys()))
    ax.set_xlabel('Coordinate')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
    ax.set_ylabel(col)
"""


#### Saving the old way of sliding metric application #########################

def eval_metric(
    F: pd.Series,
    metric: callable,
    metric_kwargs: dict = {},
    sliding_kwargs: dict = {},
    detrend: bool = True
) -> dict:
    """
    Evaluate a metric on a time series, optionally with sliding window analysis.

    Parameters:
    -----------
    F : pd.Series
        Input time series data
    metric : Callable
        Metric function to evaluate
    metric_kwargs : dict, optional
        Arguments to pass to the metric function
    sliding_kwargs : dict, optional
        Sliding window parameters. If None or empty, evaluates on full signal only.
        Expected keys: 'w_len' (window length)
    full_output : bool
        Whether to include sliding values and timepoints in output

    Returns:
    --------
    dict : Results dictionary with keys:
        - 'value': metric evaluated on full signal
        - 'sliding_values': metric values for each window (if full_output=True)
        - 'sliding_timepoints': timepoints for each window (if full_output=True)
        - 'r': correlation coefficient of sliding values vs time
        - 'p': p-value for the correlation
    """

    ## FIXME: do we want this for consistent output, or prefer missing keys?
    results_vals = ['value', 'sliding_values', 'sliding_timepoints', 'r', 'p']
    result = {k: np.nan for k in results_vals}

    # Always calculate the full signal metric
    result['value'] = metric(F.values, **metric_kwargs)

    # Determine windowing parameters
    if sliding_kwargs and 'w_len' in sliding_kwargs:
        # Sliding window case
        dt = np.median(np.diff(F.index))
        w_len = sliding_kwargs['w_len']
        w_size = int(w_len // dt)
        step_size = int(w_size // 2)
        n_windows = int((len(F) - w_size) // step_size + 1)

        # Check if we have enough data for meaningful sliding analysis
        if n_windows <= 2:
            return result

        # Create time indices for sliding windows
        S_times = F.index.values[
            np.linspace(step_size, n_windows * step_size, n_windows).astype(int)
        ]

        # Create windowed view into array
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
        # Apply metric to each window
        S_values = np.apply_along_axis(
            lambda w: _metric(w, **metric_kwargs), axis=1, arr=windows
        )

        result['sliding_values'] = S_values
        result['sliding_timepoints'] = S_times

        # Calculate trend statistics
        if n_windows > 1:
            result['r'], result['p'] = stats.linregress(S_times, S_values)[2:4]

    return result


def qc_series(
    F: pd.Series,
    metrics: dict,
    sliding_kwargs=None,  # if present, calculate everything in a sliding manner
    trials=None,  # if present, put trials into params
    eid: str = None,  # FIXME but left as is for now just to keep the logger happy
    brain_region: str = None,  # FIXME but left as is for now just to keep the logger happy
    detrend: bool = False,
) -> dict:
    if isinstance(F, pd.DataFrame):
        raise TypeError('F can not be a dataframe')

    qc_results = {}
    for metric, params in metrics.items():
        # try:
        if trials is not None:  # if trials are passed
            params['trials'] = trials
        qc_results[metric.__name__] = eval_metric(
            F, metric, params, sliding_kwargs, detrend=detrend
            )
        # except Exception as e:
            # continue
            # logger.warning(
                # f'{eid}, {brain_region}: metric {metric.__name__} failure: {type(e).__name__}:{e}'
            # )
    return qc_results


qc = []
for idx, session in tqdm(df_sessions.iterrows(), total=len(df_sessions)):

    if not 'alf/photometry/photometry.signal.pqt' in one.list_datasets(session['eid']):
        continue
    try:
        ps = PhotometrySession(session, one=one)
    except ALFObjectNotFound:
        continue

    for target in ps.targets['gcamp']:
        session_dict = session.to_dict()
        session_dict['target'] = target

        # Sampling regularity
        # Need raw df to correct this, but metric will tell if error is present
        session_dict['qc_n_early_samples'] = metrics.n_early_samples(ps.gcamp[target])
        # Number of unique samples in the signal
        session_dict['qc_n_unique_samples'] = metrics.n_unique_samples(ps.gcamp[target])
        # Outliers
        session_dict['qc_n_edges'] = metrics.n_edges(ps.gcamp[target])

        # Sliding metrics
        metrics_params = {metric:{} for metric in metrics_to_apply}
        sliding_kwargs = {'w_len': 120}
        session_dict.update(
            metrics.qc_series(ps.gcamp[target], metrics_params, sliding_kwargs)
        )
        qc.append(session_dict)
df_qc = pd.DataFrame(qc)

metric_names = [metric.__name__ for metric in metrics_to_apply]
for metric in metric_names:
    df_qc[f'{metric}_mean'] = df_qc[metric].apply(
        lambda x: np.nanmean(x['sliding_values'])
        )



df_coords = pd.read_csv('metadata/NBM_coordinates.csv').set_index(['subject', 'brain_region'])
df_session_coords = df_qc.apply(
    lambda x: df_coords.loc[x['subject'], x['target']],
    axis='columns'
)
df_qc = pd.concat([df_qc, df_session_coords], axis=1)


import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import AnovaRM

qc_cols = [
    'median_absolute_deviance_mean',
    'percentile_asymmetry_mean',
    'ar_score_mean'
    ]
xpos = {'original': 0, 'new_1': 1, 'new_2': 2}
colors = {'CQ014': 'C0', 'CQ015': 'C1', 'CQ016': 'C2'}
fig, axs = plt.subplots(1, len(qc_cols), figsize=(len(qc_cols) * 8, 8))
fig.suptitle('NBM Raw Signal Quality Metrics')
for ax, col in zip(axs, qc_cols):
    # ~print('\n', f'========== {col} ==========')

    # ~print('\n', 'ANOVA Results')
    # ~# print('=======================================================================')
    # ~# Simple ANOVA
    # ~model = smf.ols(f'{col} ~ C(label, Treatment("original"))', data=df_qc)
    # ~result = model.fit()
    # ~# Perform ANOVA
    # ~anova_table = sm.stats.anova_lm(result, typ=2)
    # ~print(anova_table)

    for (coord, subject), data in df_qc.groupby(['label', 'subject']):
        ax.scatter(xpos[coord] + np.random.uniform(-0.15, 0.15, len(data)), data[col], fc='none', ec=colors[subject], lw=2)
    ax.set_xticks(list(xpos.values()))
    ax.set_xticklabels(list(xpos.keys()))
    ax.set_xlabel('Coordinate')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
    ax.set_ylabel(col)

        ## TODO: response based qc
        # rename trials kwargs in metrics module to events


        # Signal amplitude
        # qc[target]['qc_deviance'] = metrics.median_absolute_deviance(session.gcamp[target])  # median amplitude
        # qc[target]['qc_percentile_distance'] = metrics.percentile_distance(
            # session.gcamp[target],
            # pc=(50, 95)
            # )  # amplitude of positive transients
        # qc[target]['qc_percentile_asymmetry'] = metrics.percentile_asymmetry(
            # session.gcamp[target],
            # pc_comp=95
            # )  # amplitude of positive versus negative transients
