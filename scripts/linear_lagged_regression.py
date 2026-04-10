# %% 
import sys
import Path
sys.path.append(Path(__file__).parent / 'RRRlib') # to be removed
import RRRlib as rrr

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from iblphotometry.fpio import PhotometrySessionLoader

import iblphotometry.preprocessing
from iblphotometry.pipelines import sliding_mad_pipeline, run_pipeline

import pynapple as nap
import scipy.interpolate

from one.api import ONE

one = ONE()

# %%
"""
 
  ######  ########  ######   ######  ####  #######  ##    ## 
 ##    ## ##       ##    ## ##    ##  ##  ##     ## ###   ## 
 ##       ##       ##       ##        ##  ##     ## ####  ## 
  ######  ######    ######   ######   ##  ##     ## ## ## ## 
       ## ##             ##       ##  ##  ##     ## ##  #### 
 ##    ## ##       ##    ## ##    ##  ##  ##     ## ##   ### 
  ######  ########  ######   ######  ####  #######  ##    ## 
 
"""

django = [
    "users__username,laura.silva",
    "lab__name,mainenlab",
    "projects__name__icontains,ibl_fibrephotometry",
    "data_dataset_session_related__name__icontains,lightning",
]

sessions = one.alyx.rest("sessions", "list", django=django)
# print(len(sessions))

# %% show the genotype per animal
subjects = list({session["subject"] for session in sessions})
for subject in subjects:
    n = len([s for s in sessions if s["subject"] == subject])
    print(subject, one.alyx.rest("subjects", "read", subject)["line"], n)

# %% select a subject
subject = "ZFM-09343"
sessions_ = [s for s in sessions if s["subject"] == subject]
eids = [s["id"] for s in sessions_]

# select a single session for now
eid = eids[-1]

psl = PhotometrySessionLoader(one=one, eid=eid)

# %%
"""
 
 ########  ##     ##  #######  ########  #######  ##     ## ######## ######## ########  ##    ## 
 ##     ## ##     ## ##     ##    ##    ##     ## ###   ### ##          ##    ##     ##  ##  ##  
 ##     ## ##     ## ##     ##    ##    ##     ## #### #### ##          ##    ##     ##   ####   
 ########  ######### ##     ##    ##    ##     ## ## ### ## ######      ##    ########     ##    
 ##        ##     ## ##     ##    ##    ##     ## ##     ## ##          ##    ##   ##      ##    
 ##        ##     ## ##     ##    ##    ##     ## ##     ## ##          ##    ##    ##     ##    
 ##        ##     ##  #######     ##     #######  ##     ## ########    ##    ##     ##    ##    
 
"""

# load and process photometry data
# the modelled quantity

brain_region = "SNc-l" # TODO this is dataset specific
psl.load_photometry()
raw_photometry = psl.photometry
fluorescence = run_pipeline(sliding_mad_pipeline, raw_photometry["GCaMP"][brain_region])
fluorescence = nap.Tsd(t=fluorescence.index, d=fluorescence.values)

# %% Model prep

continuous_regressors = {} # continuous regressors
timestamp_regressors = {} # time stamp regressors for lagged regression

t_start = fluorescence.times()[0]
t_stop = fluorescence.times()[-1]

dt = 0.1 # temporal resolution of the model
# this is the global tvec for all modeling purposes
# reinterpolate all data onto this
tvec = np.arange(t_start, t_stop, dt)

# %%
"""
 
 ########  ######## ##     ##    ###    ##     ## ####  #######  ########  
 ##     ## ##       ##     ##   ## ##   ##     ##  ##  ##     ## ##     ## 
 ##     ## ##       ##     ##  ##   ##  ##     ##  ##  ##     ## ##     ## 
 ########  ######   ######### ##     ## ##     ##  ##  ##     ## ########  
 ##     ## ##       ##     ## #########  ##   ##   ##  ##     ## ##   ##   
 ##     ## ##       ##     ## ##     ##   ## ##    ##  ##     ## ##    ##  
 ########  ######## ##     ## ##     ##    ###    ####  #######  ##     ## 
 
"""

psl.load_trials()
trials = psl.trials

# side invariant contrast
trials["contrast"] = trials["contrastLeft"]
ix = pd.isna(trials["contrastLeft"])
trials.loc[ix, "contrast"] = trials.loc[ix, "contrastRight"]


# %% Lagged regressors for behavioral events
BEHAV_EVENTS = [
    "stimOn_times",
    "goCue_times",
    "response_times",
    "firstMovement_times",
    "intervals_0",
    "intervals_1",
]
# prep data
Events = {}
for event in BEHAV_EVENTS:
    Events[event] = nap.Ts(t=trials[event].values)

# special case: feedback split into outcomes
for outcome, group in trials.groupby("feedbackType"):
    Events[f"feedbackType:{outcome}"] = nap.Ts(t=group["feedback_times"].values)

# make regressors
n_lags = 50
lags = np.linspace(-n_lags / 2, n_lags / 2 - 1, n_lags).astype("int32")


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


lagged_regs = []
# establish order for adding (and later retrieval of kernels)
event_names = sorted(list(Events.keys()))
for name in event_names:
    reg = make_reg(Events[name], tvec)
    lagged_reg = lag_reg(reg, lags)
    lagged_regs.append(lagged_reg.values)

lagged_regs = np.concatenate(lagged_regs, axis=1)

# %% these are trial constant
# and likely a good target for better feature engineering
# for example split choice by pre-post decision
# contrast only for after stimOn etc
EVENT_TYPES = ["choice", "probabilityLeft", "contrast"]

EventsC = {}
for event in EVENT_TYPES:
    trials[event]
    vvec = np.zeros(tvec.shape[0])
    for i, row in trials.iterrows():
        t_start = row["intervals_0"]
        t_stop = row["intervals_1"]
        ix_start, ix_stop = np.digitize((t_start, t_stop), tvec)
        vvec[ix_start:ix_stop] = row[event]
    EventsC[event] = nap.Tsd(t=tvec, d=vvec)

# %%
"""
 
 ########   #######   ######  ######## 
 ##     ## ##     ## ##    ## ##       
 ##     ## ##     ## ##       ##       
 ########  ##     ##  ######  ######   
 ##        ##     ##       ## ##       
 ##        ##     ## ##    ## ##       
 ##         #######   ######  ######## 
 
"""
lp = one.load_object(id=eid, obj="leftCamera", collection="alf")
df_lp = pd.DataFrame(lp["lightningPose"])

# keeping this one for splitting and addition of individual dataset based time support
# np.unique([col.split('_')[0] for col in df_lp.columns])
cols = [col for col in df_lp.columns if not col.endswith("likelihood")]
df = df_lp[cols]
df.index = lp["times"]

pose = nap.TsdFrame(df)

continuous_regressors["pose"] = pose

# %%
"""
 
 ##      ## ##     ## ######## ######## ##       
 ##  ##  ## ##     ## ##       ##       ##       
 ##  ##  ## ##     ## ##       ##       ##       
 ##  ##  ## ######### ######   ######   ##       
 ##  ##  ## ##     ## ##       ##       ##       
 ##  ##  ## ##     ## ##       ##       ##       
  ###  ###  ##     ## ######## ######## ######## 
 
"""
wheel_data = one.load_object(eid, obj="*wheel", collection="alf")
from brainbox.behavior import wheel as wheel_methods

interpolation_frequency = 1000.0  # Hz
interpolated_position, interpolated_timestamps = wheel_methods.interpolate_position(
    re_ts=wheel_data["timestamps"],
    re_pos=wheel_data["position"],
    freq=interpolation_frequency,
)
velocity, acceleration = wheel_methods.velocity_filtered(
    pos=interpolated_position, fs=interpolation_frequency
)

wheel_df = pd.DataFrame(
    dict(
        position=interpolated_position,
        velocity=velocity,
        acceleration=acceleration,
    )
)
wheel_df.index = interpolated_timestamps

wheel = nap.TsdFrame(wheel_df)

# and store
continuous_regressors["wheel"] = wheel

# %%
continuous_regressors.keys()

def interpolate_Tsd(Tsd, tvec):
    fcn_interp = scipy.interpolate.interp1d(
        Tsd.times(),
        Tsd.values,
        fill_value=np.NaN,
        bounds_error=False,
        kind="quadratic",
        axis=0,
    )
    return fcn_interp(tvec)


C = []
for k, v in continuous_regressors.items():
    C.append(interpolate_Tsd(v, tvec))

for k, v in EventsC.items():
    C.append(v.values[:, np.newaxis])

C = np.concatenate(C, axis=1)

# %%
# combine for linear modelling
X = np.concatenate([C, lagged_regs], axis=1)
Y = interpolate_Tsd(fluorescence, tvec)[:, np.newaxis]

# drop NaN rows
bad_rows = np.any(pd.isna(X), axis=1)
X = X[~bad_rows, :]
Y = Y[~bad_rows, :]

# %% fit model
# much room for improvement here
from sklearn import linear_model

reg = linear_model.Ridge(alpha=1.0)
reg.fit(X, Y)

B_hat = reg.coef_.T

Y_hat = X @ B_hat + reg.intercept_
Y_hat = Y_hat[:, np.newaxis]


def Tss(Y):
    return np.sum((Y - np.average(Y)) ** 2)


def Rss(Y, Y_hat):
    return np.sum((Y - Y_hat) ** 2)


def Rsq(Y, Y_hat):
    return 1 - (Rss(Y, Y_hat) / Tss(Y))


Rsq(Y, Y_hat)


# %% plotting - visual inspection of Y_hat and Y
import seaborn as sns
fig, axes = plt.subplots()
axes.plot(tvec[~bad_rows], Y, label='data')
axes.plot(tvec[~bad_rows], Y_hat, "r", label='model')
axes.legend()

axes.set_xlabel('time (s)')
sns.despine(fig)
fig.suptitle(f"{subject}:{eid}")
fig.tight_layout()

# %%
# plot kernels
off = B_hat.shape[0] - lagged_regs.shape[1]
kernels = np.split(B_hat[off:].flatten(), len(event_names))
fig, axes = plt.subplots(ncols=len(event_names), sharey=True, figsize=[15,3])
for i, (event, kernel) in enumerate(zip(event_names, kernels)):
    axes[i].plot(lags, kernel)
    axes[i].set_title(event, fontsize="small")
    axes[i].axhline(0, linestyle=":", color="k", lw=1)
    axes[i].axvline(0, linestyle=":", color="k", lw=1)
    axes[i].set_xlabel('time (s)')


# %% psth for visual inspection
from iblphotometry.plotters import plot_psths_from_trace
plot_psths_from_trace(pd.Series(fluorescence.d, index=fluorescence.t), trials)

# # %% spectrally matched rsq
# import scipy.signal as signal
# fs = 1 / np.average(np.diff(fluorescence.times()))
# freqs, Pxx = signal.periodogram(Y.flatten(), fs)
# plt.semilogx(freqs, Pxx)

# # %%
# fig, axes = plt.subplots()

# fcs = np.linspace(1, 13, 20)
# ords = np.arange(1, 15)
# for o in ords:
#     rsqs = []
#     for fc in fcs:
#         sos = signal.butter(o, fc, "low", fs=fs, output="sos")
#         Y_filt = signal.sosfilt(sos, Y.flatten())
#         Y_hat_filt = signal.sosfilt(sos, Y_hat.flatten())
#         rsqs.append(Rsq(Y_filt, Y_hat_filt))
#     axes.plot(fcs, rsqs, label=o)


# # %%
# sos = signal.butter(5, 5, "low", fs=fs, output="sos")
# Y_filt = signal.sosfilt(sos, Y.flatten())
# Y_hat_filt = signal.sosfilt(sos, Y_hat.flatten())

# fig, axes = plt.subplots()
# axes.plot(tvec[~bad_rows], Y_filt)
# axes.plot(tvec[~bad_rows], Y_hat_filt, "r")
# axes.set_title(f"Rsq={Rsq(Y_filt, Y_hat_filt)}")

# # %%
