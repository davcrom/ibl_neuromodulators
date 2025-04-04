import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from matplotlib import pyplot as plt
from matplotlib import colors

from one.api import ONE
from one.alf.spec import QC
from brainbox.io.one import SessionLoader

import sys
sys.path.append('/home/crombie/code/ibl_photometry/src')
# import iblphotometry.loaders as loaders
import iblphotometry.io as io

QCVAL2NUM = {  
    np.nan: 0.,
    'NOT SET': 0.01,
    'NOT_SET': 0.01,
    'PASS': 1.,
    'WARNING': 0.66,
    'CRITICAL': 0.33,
    'FAIL': 0.1
}

# Create colormap for QC grid plots
QCCMAP = colors.LinearSegmentedColormap.from_list(
    'qc_cmap',
    [(0., 'white'), (0.01, 'gray'), (0.1, 'palevioletred'), (0.33, 'violet'), (0.66, 'orange'), (1., 'limegreen')],
    N=256
)

EXTRACTED_PHOTOMETRY_DATASETS = [
    'alf/photometry/photometry.signal.pqt',
    'alf/photometry/photometryROI.locations.pqt'
]

def fetch_sessions(one, save=True, check_local=True):
    """
    Query Alyx for sessions tagged in the psychedelics project and add session
    info to a dataframe. Sessions are restricted to those with the 
    passiveChoiceWorld task protocol, quality control metadata is unpacked, and
    a list of key datasets is checked. Sessions are sorted and labelled
    (session_n) by their order.

    Parameters
    ----------
    one : one.api.OneAlyx
        Alyx database connection instance
    save : bool
        If true, the dataframe will be saved in ./metadata/sessions.csv

    Returns
    -------
    df_sessions : pandas.core.frame.DataFrame
        Dataframe containing detailed info for each session returned by the
        query
    """
    # Query for all sessions in the project with the specified task
    print("Querying database...")
    sessions = one.alyx.rest('sessions', 'list', project='ibl_fibrephotometry')
    df_sessions = pd.DataFrame(sessions).rename(columns={'id': 'eid'})
    df_sessions.drop(columns='projects')
    # Unpack the extended qc from the session dict into dataframe columns
    print("Unpacking extended qc data...")
    # Note: .copy() is applied to de-fragment the dataframe after repeated column additions
    df_sessions = df_sessions.progress_apply(_unpack_session_dict, one=one, axis='columns').copy()
    # Check if important datasets are present for the session
    print("Checking datasets...")
    df_sessions = df_sessions.progress_apply(_check_datasets, one=one, axis='columns').copy()
    if check_local:
        # Specify path to local cache & instantiate database connection
        local_cache = '/home/crombie/mnt/ccu-iblserver/kb/data/one'
        df_sessions = df_sessions.progress_apply(
            _check_local_datasets, 
            one=ONE(cache_dir=local_cache), 
            axis='columns'
            ).copy()
    # Map all columns with non-uniform data types to strings
    df_sessions = df_sessions.apply(lambda col: col if col.map(type).nunique() == 1 else col.astype(str))
    # Label and sort by session number for each subject
    df_sessions['session_n'] = df_sessions.groupby('subject')['start_time'].rank(method='dense').astype(int)
    df_sessions = df_sessions.sort_values(by=['start_time', 'subject']).reset_index(drop=True)
    # Save as csv
    if save:
        df_sessions.to_parquet('metadata/sessions.pqt', index=False)
    return df_sessions


def _check_datasets(series, one=None):
    """
    Create a boolean entry for each important dataset for the given eid.
    """
    assert one is not None
    photometry_datasets = [
        'alf/photometry/photometry.signal.pqt',
        'alf/photometry/photometryROI.locations.pqt',
        'raw_photometry_data/_neurophotometrics_fpData.channels.csv',
        'raw_photometry_data/_neurophotometrics_fpData.raw.pqt',
    ]
    # Fetch list of datasets listed under the given eid
    datasets = one.list_datasets(series['eid'])
    for dataset in photometry_datasets:
        series[dataset] = dataset in datasets
    return series


def _check_local_datasets(series, one=None, local_cache=None):
    if one is None:
        assert local_cache is not None
        # Instantiate database connection
        one = ONE(cache_dir=local_cache)
    session_path = one.eid2path(series['eid'])
    if session_path is None:
        series['local_photometry'] = False
        return series
    ## TODO: change this to looking at the raw data files, not extracted columns
    pnames = [reg.name for reg in session_path.joinpath('alf').glob('Region*')]
    photometry_pqt_paths = []
    for pname in pnames:
        photometry_pqt_paths.append(session_path / 'alf' / pname / 'raw_photometry.pqt')
    if not photometry_pqt_paths:
        series['local_photometry'] = False
        return series
    series['local_photometry'] = all([os.path.isfile(pqt_path) for pqt_path in photometry_pqt_paths])
    return series
    

def _unpack_session_dict(series, one=None):
    """
    Unpack the extended QC from the session dict for a given eid.
    """
    assert one is not None
    # Fetch full session dict
    session_dict = one.alyx.rest('sessions', 'read', id=series['eid'])
    series['session_qc'] = session_dict['qc']  # aggregate session QC value
    # Skip if there is no extended QC present
    if session_dict['extended_qc'] is None:
        return series
    # Add QC vals to series
    for key, val in session_dict['extended_qc'].items():
        # Add _qc flag to any keys that don't have it 
        if not key.endswith('_qc'): key += '_qc'
        if type(val) == list:  
            series[key.rstrip('_qc')] = val[1:]  # store underlying values
            val = val[0]  # lists have QC outcome as first entry
        if type(val) == int:
            try:
                series[key] = QC(val).name  # convert 0-100 values to string
            except ValueError:
                series[key] = val
        elif type(val) == bool:
            series[key] = 'PASS' if val else 'FAIL'  # convert T/F to pass/fail
        elif (type(val) == float) | (type(val) == str):
            series[key] = val  # directly store strings & floats
        elif val is None:
            series[key] = 'NOT_SET'
        else:
            raise ValueError
    return series

def load_photometry_data(session, one=None, one_local=None):
    if one is None:
        one = ONE()
    photometry_data = {}
    if session['remote_photometry']:
        photometry = one.load_dataset(id=session['eid'], dataset='photometry.signal')
        locations = one.load_dataset(id=session['eid'], dataset='photometryROI.locations.pqt').reset_index()
        rois = locations['ROI'].to_list()
    elif session['local_photometry']:  
        assert one_local is not None
        assert session['ROI']
        raw_data_path = one_local.eid2path(session['eid']) / 'raw_photometry_data' / 'raw_photometry.csv'
        photometry = io.from_raw_neurophotometrics_file_to_ibl_df(raw_data_path, version='old')
        photometry = photometry.drop(columns='index')
        rois = session['ROI']
    return photometry[rois + ['name']].set_index(photometry['times']).dropna()


def restrict_photometry_to_task(eid, photometry, one=None, buffer=2):
    assert eid is not None
    if one is None:
        one = ONE()
    loader = SessionLoader(one, eid=eid)
    try:
        loader.load_trials()
    except:
        return
    timings = [col for col in loader.trials.columns if col.endswith('_times')]
    t0 = loader.trials[timings].min().min()
    t1 = loader.trials[timings].max().max()
    i0 = photometry.index.searchsorted(t0 - buffer)
    i1 = photometry.index.searchsorted(t1 + buffer)
    return photometry.iloc[i0:i1].copy()
    

def _load_event_times(series, one=None):
    """
    Extracts reward_times, cue_times, and movement_times for a single row.

    Parameters
    ----------
    row : pd.Series
        A single row of the dataframe containing 'eid'.
    one : object
        The object used to load datasets like trials.

    Returns
    -------
    list
        A list containing reward_times, cue_times, and movement_times.
    """
    assert one is not None
    try:
        trials = one.load_dataset(series['eid'], '*trials.table')
        series['cue_times'] = trials['goCue_times'].values
        series['movement_times'] = trials['firstMovement_times'].values
        series['reward_times'] = trials.query('feedbackType == 1')['feedback_times'].values
        series['omission_times'] = trials.query('feedbackType == -1')['feedback_times'].values
    except:
        print(f"WARNING: no trial data found for {series['eid']}")
    return series


def get_responses(A, events, window=(0, 1)):
    # signal = A.values.squeeze()
    signal = A.values if isinstance(A, pd.Series) else A
    assert signal.ndim == 1
    tpts = A.index.values
    dt = np.median(np.diff(tpts))
    events = events[events + window[1] < tpts.max()]
    event_inds = tpts.searchsorted(events)
    i0s = event_inds - int(window[0] / dt)
    i1s = event_inds + int(window[1] / dt)
    responses = np.vstack([signal[i0:i1] for i0, i1 in zip(i0s, i1s)])
    responses = (responses.T - signal[event_inds]).T
    tpts = np.arange(window[0], window[1] - dt, dt)
    return responses, tpts


def sample_recordings(df, metric, percentile_range):
    t0, t1 = np.nanpercentile(df[metric], percentile_range)
    samples = df[(df[metric] >= t0) & (df[metric] <= t1)]
    sample = samples.sample().squeeze()
    return sample


def qc_grid(df, qc_columns=None, qcval2num=None, ax=None, xticklabels=None,
           legend=True):
    if qcval2num is None:
        qcval2num = QCVAL2NUM
    if qc_columns is None:
        qc_columns = df.columns
    df_qc = df[qc_columns].replace(qcval2num)
    if ax is None:
        fig, ax = plt.subplots()
    qcmat = df_qc.values.T.astype(float)
    ax.matshow(qcmat, cmap=QCCMAP, vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(np.arange(len(df_qc)))
    if type(xticklabels) == str:
        ax.set_xticklabels(df[xticklabels])
        ax.tick_params(axis='x', rotation=90)
    elif type(xticklabels) == list:
        xticklabels = df.apply(lambda x: '_'.join(x[xticklabels].astype(str)), axis='columns')
        ax.set_xticklabels(xticklabels)
        ax.tick_params(axis='x', rotation=90)
    ax.set_yticks(np.arange(len(df_qc.columns)))
    ax.set_yticklabels(qc_columns)
    for xtick in ax.get_xticks():
        ax.axvline(xtick - 0.5, color='white')
    for ytick in ax.get_yticks():
        ax.axhline(ytick - 0.5, color='white')
    if legend:
        for key, val in qcval2num.items():
            ax.scatter(-1, -1, color=QCCMAP(val), label=key)
        ax.set_xlim(left=-0.5)
        ax.set_ylim(top=-0.5)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    return ax
    

def session_plot(series, pipeline=[], t0=60, t1=120):
    fig = plt.figure(figsize=(12, 6))
    fig.suptitle(f"{series['subject']} - session {series['session_n']} - {series['task_protocol'].split('_')[-1]}, fiber target: {series['brain_region']}")

    psth_axes = [
        fig.add_axes([0.03, 0., 0.3, 0.3]),
        fig.add_axes([0.36, 0., 0.3, 0.3]),
        fig.add_axes([0.69, 0., 0.3, 0.3])
    ]
    signal_axes = [
        fig.add_axes([0.03, 0.7, 0.96, 0.25]),
        fig.add_axes([0.03, 0.4, 0.86, 0.25])
    ]
    
    tpts = series['GCaMP'].index.values
    signal_raw = series['GCaMP'][series['ROI']].values
    processed = pipe.run_pipeline(pipeline, series['GCaMP'])
    signal_processed = processed[series['ROI']].values

    
    signal_axes[0].plot(tpts, proc.z(signal_raw), alpha=0.5, color='gray', label='Raw')
    signal_axes[0].plot(tpts, proc.z(signal_processed), alpha=0.5, color='black', label='Processed')
    signal_axes[0].set_xlim([tpts.min(), tpts.max()])
    signal_axes[0].set_xlabel('Time (s)')
    y_min = min(proc.z(signal_raw).min(), proc.z(signal_processed).min())
    y_max = max(proc.z(signal_raw).max(), proc.z(signal_processed).max())
    signal_axes[0].set_ylim([y_min, y_max])
    signal_axes[0].set_ylabel('Signal (z-score)')
    signal_axes[0].legend(loc='upper left', bbox_to_anchor=[.9, -0.2])
    
    i0, i1 = tpts.searchsorted([t0, t1])
    signal_axes[1].plot(tpts[i0:i1], proc.z(signal_processed)[i0:i1], color='black', label='Processed')
    signal_axes[1].set_xlim([t0, t1])
    signal_axes[1].set_xlabel('Time (s)')
    y_min = proc.z(signal_processed)[i0:i1].min()
    y_max = proc.z(signal_processed)[i0:i1].max()
    signal_axes[1].set_ylim([y_min, y_max])
    signal_axes[1].set_ylabel('Signal (z-score)')    

    events_dict = {'cue': psth_axes[0], 'movement': psth_axes[1], 'reward': psth_axes[2], 'omission': psth_axes[2]}
    colors = ['blue', 'orange', 'green', 'red']
    for event, color in zip(['cue', 'reward', 'omission'], ['blue', 'green', 'red']):
        for t in series[f'{event}_times']:
            if (t < t0) or (t > t1):
                continue
            signal_axes[1].axvline(t, color=color)
    y_max = []
    for (event, ax), color in zip(events_dict.items(), colors):
        responses, tpts = psth(series['GCaMP'], series[f'{event}_times'])
        y_max.append(np.abs(responses.mean(axis=0)).max())
        ax.plot(tpts, responses.mean(axis=0), color=color, label=event)
        ax.plot(tpts, responses.mean(axis=0) - stats.sem(responses, axis=0), ls='--', color=color)
        ax.plot(tpts, responses.mean(axis=0) + stats.sem(responses, axis=0), ls='--', color=color)
        ax.axhline(0, ls='--', color='gray', alpha=0.5)
        ax.set_xlabel('Time (s)')
        ax.legend(loc='upper right')
    for ax in psth_axes:
        ax.set_ylim([-1 * max(y_max), max(y_max)])
        ax.set_yticks([])
    psth_axes[0].set_yticks([-1 * max(y_max), 0, max(y_max)])
    psth_axes[0].ticklabel_format(axis='y', style='sci', scilimits=[-2, 2])
    psth_axes[0].set_ylabel('Response (a.u.)')
    
    return fig


def load_kb_recinfo():
    df = pd.read_csv('metadata/website.csv')
    # Convert acronym strings into lists of strings
    df['region'] = df['_acronyms'].apply(eval)
    # Add additional metadata
    df_insertions = pd.read_csv('metadata/insertions.csv')
    def _merge_metadata(row, df=df_insertions):
        subj = df_insertions[df_insertions['subject'] ==  row['subject']]
        for col in [v for v in subj.columns if v != 'subject']:
            row[col] = subj[col].values
        return row
    df = df.apply(_merge_metadata, df=df_insertions, axis='columns')   
    return df
