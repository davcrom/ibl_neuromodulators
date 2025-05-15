import numpy as np
import pandas as pd
from tqdm import tqdm
from pandera.errors import SchemaError
from datetime import datetime

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
# Note: what about stationarity? Noise should be stationary

df_sessions = pd.read_parquet('metadata/sessions.pqt')
df = df_sessions.query('local_photometry == True or remote_photometry == True').copy()
df['new_recording'] = df['start_time'].apply(lambda x: datetime.fromisoformat(x) >  datetime(2024, 4, 1))
df['target'] = df['target'].apply(lambda x: [x[0]] if len(np.unique(x)) == 1 else x)
df['single_fiber'] = df.apply(lambda x: (len(x['roi']) == 1) and (len(x['target']) == 1), axis='columns') 
df = df.query('(new_recording == True) or (single_fiber == True)')


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
    info = session[['eid', 'subject', 'lab', 'start_time', 'NM', 'session_type']].to_dict()
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
    for roi, target in zip(session['roi'], session['target']):
        raw_data = proc.fix_repeated_sampling(raw_data, roi=roi)
        assert metrics.n_repeated_samples(raw_data) == 0
        info['roi'] = roi
        info['target'] = target
        if session['remote_photometry']:
            try:
                raw_data = restrict_photometry_to_task(session['eid'], raw_data, one_remote)
            except ALFObjectNotFound:
                print(f"No trials found for {session['eid']}")
                pass
        gcamp = raw_data.query('name == "GCaMP"')
        F = gcamp[[roi]]
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
        F_proc = proc.sliding_robust_zscore_rolling(F[roi], **sliding_zscore_pars)
        if len(F_proc) == 0:
            print(f"Signal has no variance for {session['eid']}")
        else:
            fs = 1 / np.median(np.diff(F_proc.index))
            metric_kwargs = {metric:{} for metric in processed_metrics}
            metric_kwargs['low_freq_power_ratio'] = {'dt': 1 / fs} 
            processed_qc = qc_series(
                F_proc,
                metric_kwargs,
                sliding_kwargs={metric:{} for metric in processed_metrics}
            )
        qc_dicts.append({**info, **csv_qc, **raw_qc, **sliding_qc, **processed_qc})

df_qc = pd.DataFrame(qc_dicts)
df_qc.to_parquet('qc_photometry.pqt')
