# %%
import numpy as np
import pandas as pd
from scipy import signal, interpolate
import pynapple as nap
from sklearn import linear_model

from iblphotometry.loaders import KceniaLoader
from iblphotometry import processing as fpp
from iblphotometry import pipelines

import matplotlib.pyplot as plt
import seaborn as sns

from one.api import ONE
from brainbox.behavior import wheel as wheel_methods

one = ONE(mode='remote', cache_dir='/mnt/h0/kb/data/one')

# %% the old eids
eids = [
    '13319506-3b45-4e94-85c6-b1080dc7b10a',  # 5HT ZFM-05236	2023-03-24
    'a1ccc8ed-9829-4af8-91fd-cc1c83b74b98',  # DA ZFM-04022	2022-12-30
    '974a8a32-2156-4250-b9db-4298fc8daa65',  # NE ZFM-06275	2024-01-19
]

eid = eids[2]
one.list_datasets(eid)

# %%
eids = pd.read_csv('/home/georg/Downloads/eids_map_kcenia.csv')
esig = eids[eids.columns[:2]]
esig.columns = ['eid', 'sig']
earea = eids[eids.columns[2:]]
earea.columns = ['eid', 'area']
eids = pd.merge(esig, earea, on='eid')
eids = eids.set_index('eid')

for eid in eids.index:
    dsets = one.list_datasets(eid)
    for dset in dsets:
        if 'lightningPose' in dset:
            eids.loc[eid, 'lp'] = 1

# %%
# eids[eids['lp'] == 1]

# %% select examples here
# eid, brain_area = '89b8ef70-e620-49c2-a0f7-09890ba9fc0e', 'SNc'
eid, brain_area = '974a8a32-2156-4250-b9db-4298fc8daa65', 'LC'
# eid, brain_area = 'ec02dc0b-7a44-46d9-a140-8400cc5092a7', 'DR'
# eid, brain_area = '56ed83ac-c196-4817-bc37-62c02ba89d47', 'SI'

# %%
"""
##     ##  #######  ########  ######## ##           ######  ######## ######## ##     ## ########
###   ### ##     ## ##     ## ##       ##          ##    ## ##          ##    ##     ## ##     ##
#### #### ##     ## ##     ## ##       ##          ##       ##          ##    ##     ## ##     ##
## ### ## ##     ## ##     ## ######   ##           ######  ######      ##    ##     ## ########
##     ## ##     ## ##     ## ##       ##                ## ##          ##    ##     ## ##
##     ## ##     ## ##     ## ##       ##          ##    ## ##          ##    ##     ## ##
##     ##  #######  ########  ######## ########     ######  ########    ##     #######  ##
"""
# first thing, load trial data
trials = one.load_dataset(eid, '_ibl_trials.table.pqt')


def expand_column(df: pd.DataFrame, col: str, dtype='int64'):
    for value, group in df.groupby(col):
        new_col = f'{col}={value}'
        df.loc[group.index, new_col] = 1
        df.loc[pd.isna(df[new_col]), new_col] = 0
        df[new_col] = df[new_col].astype(dtype)
    return df


for expand_by in ['feedbackType', 'choice']:
    trials = expand_column(trials, expand_by)

# to define the global time base of the model
t_start = trials.iloc[0]['intervals_0'] - 5
t_stop = trials.iloc[-1]['intervals_1'] + 5

dt = 0.1
tvec = np.arange(t_start, t_stop, dt)

# the individual model components are
# lagged regressors
lag_regs = {}
# continuous regressors
cont_regs = {}
# trial_regressors
trial_regs = {}

# %%
"""
########  ##     ##  #######  ########    ########     ###    ########    ###
##     ## ##     ## ##     ##    ##       ##     ##   ## ##      ##      ## ##
##     ## ##     ## ##     ##    ##       ##     ##  ##   ##     ##     ##   ##
########  ######### ##     ##    ##       ##     ## ##     ##    ##    ##     ##
##        ##     ## ##     ##    ##       ##     ## #########    ##    #########
##        ##     ## ##     ##    ##       ##     ## ##     ##    ##    ##     ##
##        ##     ##  #######     ##       ########  ##     ##    ##    ##     ##
"""

# load data
loader = KceniaLoader(one=one)
raw_dfs = loader.load_photometry_data(eid)

# process with a simple pipeline
from iblphotometry.pipelines import sliding_mad_pipeline as pipeline

fphot_df = pipelines.run_pipeline(pipeline, raw_dfs['raw_calcium'])
fphot = nap.TsdFrame(t=fphot_df.index, d=fphot_df.values)

# restrict data to session time
fphot = fphot.restrict(nap.IntervalSet(t_start, t_stop))


# %%
"""
 ######   #######  ##    ## ########    ########  ########  ######    ######
##    ## ##     ## ###   ##    ##       ##     ## ##       ##    ##  ##    ##
##       ##     ## ####  ##    ##       ##     ## ##       ##        ##
##       ##     ## ## ## ##    ##       ########  ######   ##   ####  ######
##       ##     ## ##  ####    ##       ##   ##   ##       ##    ##        ##
##    ## ##     ## ##   ###    ##       ##    ##  ##       ##    ##  ##    ##
 ######   #######  ##    ##    ##       ##     ## ########  ######    ######
"""

# %% get pose data

lp = one.load_object(id=eid, obj='leftCamera', collection='alf')
df_lp = pd.DataFrame(lp['lightningPose'])

# keeping this one for splitting and addition of individual dataset based time support
cols = [col for col in df_lp.columns if not col.endswith('likelihood')]
df = df_lp[cols]
df.index = lp['times']
pose = nap.TsdFrame(df)

# store
cont_regs['pose'] = pose

# %%  wheel data
wheel_data = one.load_object(eid, obj='*wheel', collection='alf')


interpolation_frequency = 1000.0  # Hz
interpolated_position, interpolated_timestamps = wheel_methods.interpolate_position(
    re_ts=wheel_data['timestamps'],
    re_pos=wheel_data['position'],
    freq=interpolation_frequency,
)
velocity, acceleration = wheel_methods.velocity_filtered(
    pos=interpolated_position, fs=interpolation_frequency
)
# W = nap.Tsd(t=wheel_data['timestamps'], d=wheel_data['position'])

wheel_df = pd.DataFrame(
    dict(position=interpolated_position, velocity=velocity, acceleration=acceleration)
)
wheel_df.index = interpolated_timestamps
wheel = nap.TsdFrame(wheel_df)

# store
cont_regs['wheel'] = wheel

# %%
"""
######## ########  ####    ###    ##          ########  ########  ######    ######
   ##    ##     ##  ##    ## ##   ##          ##     ## ##       ##    ##  ##    ##
   ##    ##     ##  ##   ##   ##  ##          ##     ## ##       ##        ##
   ##    ########   ##  ##     ## ##          ########  ######   ##   ####  ######
   ##    ##   ##    ##  ######### ##          ##   ##   ##       ##    ##        ##
   ##    ##    ##   ##  ##     ## ##          ##    ##  ##       ##    ##  ##    ##
   ##    ##     ## #### ##     ## ########    ##     ## ########  ######    ######
"""
# %%
# adding side invariant contrast
trials['contrast'] = trials['contrastLeft']
ix = pd.isna(trials['contrastLeft'])
trials.loc[ix, 'contrast'] = trials.loc[ix, 'contrastRight']

# event types to go into the model
EVENT_TYPES = [
    'feedbackType=1',
    'feedbackType=-1',
    # 'feedbackType',
    # 'choice',
    'choice=1',
    'choice=-1',
    'rewardVolume',
    'probabilityLeft',
    'contrast',
]

for event in EVENT_TYPES:
    vvec = np.zeros(tvec.shape[0])
    for i, row in trials.iterrows():
        t_start = row['intervals_0']
        t_stop = row['intervals_1']
        ix_start, ix_stop = np.digitize((t_start, t_stop), tvec)
        vvec[ix_start:ix_stop] = row[event]
    trial_regs[event] = nap.Tsd(t=tvec, d=vvec)


# %%
"""
##          ###     ######    ######   ######## ########     ########  ########  ######    ######
##         ## ##   ##    ##  ##    ##  ##       ##     ##    ##     ## ##       ##    ##  ##    ##
##        ##   ##  ##        ##        ##       ##     ##    ##     ## ##       ##        ##
##       ##     ## ##   #### ##   #### ######   ##     ##    ########  ######   ##   ####  ######
##       ######### ##    ##  ##    ##  ##       ##     ##    ##   ##   ##       ##    ##        ##
##       ##     ## ##    ##  ##    ##  ##       ##     ##    ##    ##  ##       ##    ##  ##    ##
######## ##     ##  ######    ######   ######## ########     ##     ## ########  ######    ######
"""


BEHAV_EVENTS = [
    'stimOn_times',
    'goCue_times',
    # 'response_times',
    # 'feedback_times',
    'feedback_times:feedbackType=1',
    'feedback_times:feedbackType=-1',
    'firstMovement_times',
    'intervals_0',
    'intervals_1',
]
# prep data
for event in BEHAV_EVENTS:
    if ':' in event:
        event_, cond = event.split(':')
        name, value = cond.split('=')
        df = trials.groupby(name).get_group(int(value))
    else:
        df = trials
        event_ = event
    lag_regs[event] = nap.Ts(t=df[event_].values)

# %%
"""
########  ########  ######  ####  ######   ##    ##    ##     ##    ###    ######## ########  #### ##     ##
##     ## ##       ##    ##  ##  ##    ##  ###   ##    ###   ###   ## ##      ##    ##     ##  ##   ##   ##
##     ## ##       ##        ##  ##        ####  ##    #### ####  ##   ##     ##    ##     ##  ##    ## ##
##     ## ######    ######   ##  ##   #### ## ## ##    ## ### ## ##     ##    ##    ########   ##     ###
##     ## ##             ##  ##  ##    ##  ##  ####    ##     ## #########    ##    ##   ##    ##    ## ##
##     ## ##       ##    ##  ##  ##    ##  ##   ###    ##     ## ##     ##    ##    ##    ##   ##   ##   ##
########  ########  ######  ####  ######   ##    ##    ##     ## ##     ##    ##    ##     ## #### ##     ##
"""


# %% continuous regressors
def interp_Tsd(Tsd, tvec):
    # interpolation wrapper
    fcn_interp = interpolate.interp1d(
        Tsd.times(),
        Tsd.values,
        fill_value=np.NaN,
        bounds_error=False,
        kind='quadratic',
        axis=0,
    )
    return fcn_interp(tvec)


all_regs = []
all_regs_names = []

for name, reg in cont_regs.items():
    # interpolate
    reg_intp = interp_Tsd(reg, tvec)

    # z score all continuous regressors
    mus = np.nanmean(reg_intp, axis=0)
    sigs = np.nanstd(reg_intp, axis=0)
    all_regs.append((reg_intp - mus) / sigs)

    all_regs_names.append(reg.columns)

# %% trial regressors
# don't need interpolation
for name, reg in trial_regs.items():
    reg_values = reg.values[:, np.newaxis]
    all_regs.append(reg_values)
    all_regs_names.append(name)

# %% lagged regressors
n_lags = 50
lags = np.linspace(-n_lags / 2, n_lags / 2 - 1, n_lags).astype('int32')


def make_reg(Ts, tvec):
    times = Ts.times()
    times = times[
        np.logical_and(times > tvec[0], times < tvec[-1])
    ]  # restrict temporal support
    bvec = np.zeros(tvec.shape[0])
    bvec[np.digitize(times, tvec)] = 1
    return nap.Tsd(t=tvec, d=bvec)


def lag_reg(Tsd, lags):
    reg = Tsd.values
    rolls = []
    for lag in lags:
        rolls.append(np.roll(reg, lag + 1))  # the +1 is from the LM(L,X) deduced
    reg_ex = np.stack(rolls).T
    return nap.TsdFrame(d=reg_ex, t=Tsd.times())


for name, reg in lag_regs.items():
    digitized_reg = make_reg(reg, tvec)
    lagged_reg = lag_reg(digitized_reg, lags)

    all_regs.append(lagged_reg.values)
    all_regs_names.append([f'{event}{i}' for i in lagged_reg.columns])

# %%
"""
##     ##  #######  ########  ######## ##
###   ### ##     ## ##     ## ##       ##
#### #### ##     ## ##     ## ##       ##
## ### ## ##     ## ##     ## ######   ##
##     ## ##     ## ##     ## ##       ##
##     ## ##     ## ##     ## ##       ##
##     ##  #######  ########  ######## ########
"""

X = np.concatenate(all_regs, axis=1)

if fphot.shape[1] > 1:
    Y = interp_Tsd(fphot, tvec)[:, 1]  # column select here
    Y = Y[:, np.newaxis]
else:
    Y = interp_Tsd(fphot, tvec)

# drop NaN rows
bad_rows = np.any(pd.isna(X), axis=1)
X = X[~bad_rows, :]
Y = Y[~bad_rows, :]

# %% fit
reg = linear_model.Ridge(alpha=1.0)
reg.fit(X, Y)

B_hat = reg.coef_.T
Y_hat = X @ B_hat + reg.intercept_


# %% eval
def Tss(Y):
    return np.sum((Y - np.average(Y)) ** 2)


def Rss(Y, Y_hat):
    return np.sum((Y - Y_hat) ** 2)


def Rsq(Y, Y_hat):
    return 1 - (Rss(Y, Y_hat) / Tss(Y))


# freq matched Rsq
f = 3
fs = 1 / np.average(np.diff(fphot.times()))
sos = signal.butter(5, f, 'low', fs=fs, output='sos')
Y_filt = signal.sosfilt(sos, Y.flatten())
Y_hat_filt = signal.sosfilt(sos, Y_hat.flatten())

rsq = Rsq(Y, Y_hat)
rsqa = Rsq(Y_filt, Y_hat_filt)
print(f'model Rsq:{rsq:.3f}, freq adjusted: f={f}, Rsq={rsqa:.3f}')


# %% plot some time traces
fig, axes = plt.subplots(figsize=[10, 5])
axes.plot(tvec[~bad_rows], Y, 'k', label='data')
axes.plot(tvec[~bad_rows], Y_hat, 'r', label='model')
axes.set_xlabel('time (s)')
axes.set_ylabel('signal (au)')
axes.legend()
axes.set_title(f'eid:{eid}, area={brain_area}, Rsq={rsqa:.3f}')
sns.despine(fig)
times = trials['feedback_times'].values
for t in times:
    axes.axvline(t, alpha=0.5, lw=1, color='k', zorder=-1)
axes.set_xlim(60, 120)

# %% plot kernels
off = B_hat.shape[0] - (len(lag_regs.keys()) * n_lags)
kernels = np.split(B_hat[off:].flatten(), len(BEHAV_EVENTS))
fig, axes = plt.subplots(ncols=len(BEHAV_EVENTS), sharey=True, figsize=[11, 4])
for i, (event, kernel) in enumerate(zip(BEHAV_EVENTS, kernels)):
    lags_t = lags * dt
    axes[i].plot(lags_t, kernel)
    axes[i].set_title(event, fontsize='small')
    axes[i].axhline(0, linestyle=':', color='k', lw=1)
    axes[i].axvline(0, linestyle=':', color='k', lw=1)
    axes[i].set_xlabel('time (s)')
axes[0].set_ylabel('beta')


sns.despine(fig)

# %% plot the other regs
off = B_hat.shape[0] - (len(lag_regs.keys()) * n_lags)

B_hat[:off]
all_regs_names = np.concatenate([np.array(n).flatten() for n in all_regs_names])

fig, axes = plt.subplots(figsize=[8, 5])
axes.bar(np.arange(off), B_hat[:off].flatten())
axes.set_xticks(np.arange(off))
axes.set_xticklabels(all_regs_names[:off], ha='right')
axes.tick_params(axis='x', labelrotation=45)
axes.axhline(0, color='k', lw=1, linestyle=':')

for i in np.arange(off):
    axes.axvline(i, lw=0.5, color='k', zorder=-1, alpha=0.2)
sns.despine(fig)
axes.set_ylabel('beta')
axes.set_title(f'eid:{eid}, {brain_area}')
fig.tight_layout()


# %%
# fig, axes = plt.subplots()
# axes.matshow(X, vmin=-2, vmax=2)
# axes.set_aspect('auto')
