import numpy as np
import pandas as pd
from tqdm import tqdm
from pandera.errors import SchemaError

from one.api import ONE
from one.alf.exceptions import ALFObjectNotFound

import sys
sys.path.append('/home/crombie/code/ibl_photometry/src')
import iblphotometry.io as io
import iblphotometry.processing as proc
import iblphotometry.pipelines as pipelines
import iblphotometry.metrics as metrics

from util import load_photometry_data, restrict_photometry_to_task


one_remote = ONE()
one_local = ONE(cache_dir='/home/crombie/mnt/ccu-iblserver')

df_sessions = pd.read_parquet('metadata/sessions.pqt')
df_sessions['local_photometry'] = df_sessions['local_photometry'].replace({'nan': False}).astype(bool)
alf_datasets = ['alf/photometry/photometry.signal.pqt', 'alf/photometry/photometryROI.locations.pqt']
df_sessions['remote_photometry'] = df_sessions.apply(lambda x: all([x[dset] for dset in alf_datasets]), axis='columns')

def _get_regions(series, df_regions=None):
    assert df_regions is not None
    eid = series['eid']
    region = df_regions.query('eid == @eid')
    series['ROI'] = region['ROI'].to_list()
    return series
df_regions = pd.read_csv('metadata/regions.csv')
df_sessions = df_sessions.apply(_get_regions, df_regions=df_regions, axis='columns')  # .explode('ROI')

# df = df_sessions.query('local_photometry == True or remote_photometry == True')
df = df_sessions.query('local_photometry == True')

raw_metrics = [
    'n_unique_samples',
    'n_spikes_dt',
    'n_spikes_dy',
    'n_outliers',
    'expmax_violation',
    'deviance',
    'bleaching_amp'
]

processed_metrics = [
    'percentile_dist',
    'signal_asymmetry',
    'low_freq_power_ratio',
    'ar_score'
]

behavior_metrics = []

pipeline = [
    (proc.remove_spikes, dict()),
    (proc.exponential_bleachcorrect, dict(correction_method='subtract'))
]

metrics_dicts = []
for _, session in tqdm(df.iterrows(), total=len(df)):
    metrics_dict = session[['eid', 'subject', 'lab', 'start_time', 'strain', 'NM', 'session_type']].to_dict()
    try:
        raw_data = load_photometry_data(session, one_remote, one_local)
    except ValueError:
        print(f"Unable to load photometry data for {session['eid']}")
        continue
    metrics_dict['dt_violations'] = metrics.dt_violations(raw_data)
    metrics_dict['interleaved_acquisition'] = metrics.interleaved_acquisition(raw_data)
    if session['remote_photometry']:
        try:
            raw_data = restrict_photometry_to_task(session['eid'], raw_data, one_remote)
        except ALFObjectNotFound:
            print(f"No trials found for {session['eid']}")
            continue
    else:
        t0 = raw_data.index.min()
        t1 = raw_data.index.max()
        i0 = raw_data.index.searchsorted(t0 - 10 * 60)
        i1 = raw_data.index.searchsorted(t1 + 10 * 60)
        raw_data = raw_data.iloc[i0:i1].copy()
    photometry = {
        ch: raw_data.query('name == @ch').drop(columns='name') for ch in ['GCaMP', 'Isosbestic']
    }
    for roi in photometry['GCaMP'].columns:
        metrics_dict['roi'] = roi
        for metric in raw_metrics:
            metric_func = getattr(metrics, metric)
            metrics_dict[metric] = metric_func(photometry['GCaMP'][roi])
        proc = pipelines.run_pipeline(pipeline, photometry['GCaMP'][[roi]])
        for metric in processed_metrics:
            metric_func = getattr(metrics, metric)
            metrics_dict[metric] = metric_func(photometry['GCaMP'][roi])
        metrics_dicts.append(metrics_dict)