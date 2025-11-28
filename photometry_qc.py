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


# TODO: make session file name an arg
SESSIONS_FNAME = 'sessions_2025-11-24-13h41.pqt'
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
    # ~'n_windows': 10,
    'w_len': 120,
    'step_len': 60,
    'detrend': True
}

one = ONE()
df_sessions = df_sessions.query('NM == "ACh"')
eids = df_sessions['eid']
print(len(eids))

## TODO: replace with df_sessions query
# ~bad_eids = []
# ~for eid in tqdm(eids):
    # ~try:
        # ~psl = PhotometrySessionLoader(eid=eid, one=one)
        # ~psl.load_photometry()
    # ~except:
        # ~bad_eids.append(eid)
# ~eids = set(eids) - set(bad_eids)
# ~print(len(eids))

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

# Save as parquet with a timestamp
df_qc = pd.concat([qc_raw, qc_sliding])
save_timestamped_pqt(df_qc, QCPHOTOMETRY_FPATH)

# Load a qc dataframe
df_qc = pd.read_parquet('results/qc_photometry_ach_2025-11-24-18h24.pqt')

df_qc = df_qc.dropna(subset='metric')

# Get photometry location data
df_insertions = pd.read_csv('metadata/insertions.csv')
df_insertions['targeted_region'] = df_insertions['targeted_region'] + df_insertions['X-ml_um'].apply(lambda x: '-l' if x > 0 else '-r')
df_insertions = df_insertions.set_index(['subject', 'targeted_region'])

# Get unique eid, brain_region combinations
unique_sessions = df_qc[['eid', 'brain_region']].drop_duplicates()

# Loop over unique session, region combinations and get anatomical coordinate
one = ONE()
sessions_data = []
unresolvable = []
for _, row in tqdm(unique_sessions.iterrows(), total=len(unique_sessions)):
    eid = row['eid']
    brain_region = row['brain_region']

    try:
        session_details = one.get_details(eid)
        subject = session_details['subject']
        df_locations = one.load_dataset(id=eid, dataset='photometryROI.locations.pqt')
    except ALFObjectNotFound:
        continue

    # Check if this brain_region is in the locations
    if brain_region not in df_locations['brain_region'].values:
        unresolvable.append({'eid': eid, 'brain_region': brain_region})
        continue

    try:
        subject_insertions = df_insertions.loc[subject]
    except KeyError:
        unresolvable.append({'eid': eid, 'brain_region': brain_region})
        continue

    in_insertions = [brain_region in region for region in subject_insertions.index]

    if sum(in_insertions) != 1:
        unresolvable.append({'eid': eid, 'brain_region': brain_region})
        continue

    # Get the matching insertion
    insertion = subject_insertions[in_insertions].iloc[0]
    sessions_data.append({
        'subject': subject, 'eid': eid, 'brain_region': brain_region, **insertion.to_dict()
        })

# Convert to DataFrame and merge back to df_qc
df_sessions = pd.DataFrame(sessions_data)
df = df_qc.merge(df_sessions, on=['eid', 'brain_region'], how='inner')

df = df.dropna(subset=['X-ml_um', 'Y-ap_um', 'Z-dv_um'])
df = df[df['brain_region'].apply(lambda x: x not in ['PPT', 'SI'])]

# Create a coordinate label
df['coord'] = df.apply(
    lambda x: f'({abs(x["X-ml_um"])}, {x["Y-ap_um"]}, {x["Z-dv_um"]})',
    axis='columns'
)
df['hemisphere'] = df['X-ml_um'].apply(lambda x: 'l' if x > 0 else 'r')

df.to_parquet('photometry_qc_nbm.pqt')

metrics_to_plot = [
    'median_absolute_deviance',
    'percentile_distance',
    'percentile_asymmetry',
    'ar_score'
    ]

# Set font sizes (big for poster)
fontsize = 16
plt.rcParams.update({
    'font.size': fontsize,
    'axes.labelsize': fontsize,
    'axes.titlesize': fontsize,
    'xtick.labelsize': fontsize,
    'ytick.labelsize': fontsize,
    'legend.fontsize': fontsize
})
from pymer4.models import Lmer

for metric in metrics_to_plot:
    print(f'\n===== {metric} =====\n')
    df_metric = df[df['metric'] == metric]
    df_metric = df_metric.groupby(['subject', 'eid', 'coord']).apply(lambda x: np.nanmean(x['value']), include_groups=False).reset_index()
    df_metric = df_metric.rename(columns={0: 'value'})
    df_metric = df_metric[~df_metric['value'].isna() & ~np.isinf(df_metric['value'])]
    df_metric = df_metric.merge(df[['coord', 'X-ml_um', 'Y-ap_um', 'Z-dv_um']].drop_duplicates(), on='coord')

    fig, ax = plt.subplots()
    ax.set_title(metric)
    coords = df_metric.drop_duplicates('coord').sort_values(['Z-dv_um', 'Y-ap_um', 'X-ml_um'])['coord'].values
    subjects = df_metric['subject'].unique()
    colors = plt.cm.Set3(np.arange(len(subjects)))
    for i, coord in enumerate(coords):
        for j, subj in enumerate(subjects):
            vals = df_metric[(df_metric['coord'] == coord) & (df_metric['subject'] == subj)]['value'].values
            x = np.random.normal(i - 0.5 + j / len(subjects), 0.05, len(vals))
            ax.scatter(x, vals, fc='none', ec=colors[j], label=subj if i == 0 else '')
            if len(vals) > 0:
                ax.plot(i - 0.5 + j / len(subjects), np.median(vals), marker='_', markersize=10, markeredgewidth=2, color=colors[j])
    ax.set_xticks(np.arange(len(coords)))
    ax.set_xticklabels(coords)
    ax.tick_params(axis='x', rotation=90)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
    fig.savefig(f'figures/photometry_qc_nbm_{metric}.svg')

    df_metric['coord'] = df_metric['coord'].astype('category')
    df_metric['coord'] = df_metric['coord'].cat.reorder_categories(
        ['(1750, -700, -4150.0)'] +
        [c for c in df_metric['coord'].cat.categories if c != '(1750, -700, -4150.0)']
        )
    model = Lmer('value ~ coord + (1 | subject)', data=df_metric)
    model.fit()
    print(model.warnings)
    print(model.summary())


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
