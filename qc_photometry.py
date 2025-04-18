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
from iblphotometry.qc import qc_series

from util import load_photometry_data, restrict_photometry_to_task

## TODO:
# Fix spike detection and interpolation
# Add example outliers to fig 1
# Try 2D deviance, asymmetry plot
# Separately compute low_freq_power_ratio or give dt kwarg

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
df_sessions = df_sessions.apply(_get_regions, df_regions=df_regions, axis='columns')

df = df_sessions.query('local_photometry == True or remote_photometry == True')

one_remote = ONE()
one_local = ONE(cache_dir='/home/crombie/mnt/ccu-iblserver')

csv_metrics = [
    'n_early_samples',
    'n_repeated_samples'
]

raw_metrics = [
    'n_unique_samples',
    'f_unique_samples',
    # 'bleaching_tau',
    # 'bleaching_amp',
]

sliding_metrics = [
    'deviance',
    # 'n_spikes_dy',
    'n_expmax_violations',
    'expmax_violation',
    'percentile_dist',
    'signal_asymmetry',
]

processed_metrics = [
    'low_freq_power_ratio',
    'ar_score'
]

behavior_metrics = []

sliding_kwargs = {'w_len': 60}
sliding_zscore_pars = {'w_len': 60}

schema_errors = []
qc_dicts = []
for _, session in tqdm(df.iterrows(), total=len(df)):
    info = session[['eid', 'subject', 'lab', 'start_time', 'strain', 'NM', 'session_type']].to_dict()
    try:
        if session['local_photometry']:
            raw_data = load_photometry_data(session, one_local, extracted=False)
        elif session['remote_photometry']:
            raw_data = load_photometry_data(session, one_remote, extracted=True)
        else:
            print(f"Unclear how to load photometry data for {session['eid']}")
            continue
    except ValueError:
        print(f"No ROIs for {session['eid']}")
        continue
    except SchemaError:
        print(f"Incorrectly formatted photometry data for {session['eid']}")
        schema_errors.append(session['eid'])
        continue
    csv_qc = {metric:getattr(metrics, metric)(raw_data) for metric in csv_metrics}
    for roi in io.infer_data_columns(raw_data):
        raw_data = proc.fix_repeated_sampling(raw_data, roi=roi)
        assert metrics.n_repeated_samples(raw_data) == 0
        if session['remote_photometry']:
            try:
                raw_data = restrict_photometry_to_task(session['eid'], raw_data, one_remote)
            except ALFObjectNotFound:
                print(f"No trials found for {session['eid']}")
                pass
        gcamp = raw_data.query('name == "GCaMP"')
        F = gcamp[[roi]]
        info['ROI'] = [roi]
        raw_qc = qc_series(
            F[roi],
            {metric:{} for metric in raw_metrics},
            sliding_kwargs={metric:{} for metric in raw_metrics}
        )
        sliding_qc = qc_series(
            F[roi],
            {metric:{} for metric in sliding_metrics},
            sliding_kwargs={metric:sliding_kwargs for metric in sliding_metrics}
        )
        # F_proc = pipelines.run_pipeline([(proc.sliding_robust_zscore, sliding_zscore_pars)], F).dropna()
        F_proc = proc.sliding_robust_zscore_rolling(F[roi], **sliding_zscore_pars)
        if len(F_proc) == 0:
            print(f"Signal has no variance for {session['eid']}")
        else:
            metric_kwargs = {metric:{} for metric in processed_metrics}
            metric_kwargs['low_freq_power_ratio'] = {'dt': np.median(np.diff(F_proc.index))}
            processed_qc = qc_series(
                F_proc,
                metric_kwargs,
                sliding_kwargs={metric:{} for metric in processed_metrics}
            )
        qc_dicts.append({**info, **csv_qc, **raw_qc, **sliding_qc, **processed_qc})

df_qc = pd.DataFrame(qc_dicts)
df_qc.to_parquet('qc_photometry_60.pqt')
