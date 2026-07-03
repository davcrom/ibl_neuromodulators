# %%
import pandas as pd
from one.api import ONE
from iblphotometry.plotters import plot_psths_from_trace
import seaborn as sns
import matplotlib as mpl
from pathlib import Path
from tqdm import tqdm
import pickle
import numpy as np
import matplotlib.pyplot as plt

# dpi luv
mpl.rcParams["figure.dpi"] = 284  # screen dpi adjustment - laptop
mpl.rcParams["figure.dpi"] = 221  # screen dpi adjustment - CCU

from data_loaders import (
    load_session_data,
    get_brain_regions,
)
from encoding_model import (
    split_pose,
    make_time_grid,
    make_lags,
    events_from_trials,
    interpolate_to_grid,
    get_kernel,
)

one = ONE()

SAVE_PLOTS: bool = False

PLOT_FOLDER = Path(__file__).parent / "plots"
PLOT_FOLDER.mkdir(parents=True, exist_ok=True)

# %% subject selection
# subject = "ZFM-09365"  # 5HT
subject = "ZFM-09343"  # DA
# subject = "ZFM-09439"  # 5HT-2
# subject = "ZFM-08871"  # DBh

eids = (Path(__file__).parent / f"{subject}-sessions.txt").read_text().splitlines()

# %%
# subject = one.eid2ref(eid)['subject']
genotype = one.alyx.rest("subjects", "read", subject)["line"]

# model config
DT = 0.1
N_LAGS = 30

# %% fit every session of the subject
EVENTS = {
    "stimOn_times": "signed_contrast",
    "response_times": "choice",
    "firstMovement_times": "choice",
    "feedback_times": "feedbackType",
}


# %% reloads the fits
fits_file = Path(__file__).parent / f"{subject}-fits.pkl"
with open(fits_file, "rb") as fH:
    fits = pickle.load(fH)

fits.keys()

# %%
eid = "6931684c-a721-4db8-9698-e3101d0e4a1b"  # first session
label = "early"

# %%
eid = "a9456015-4533-4bfa-bf6f-451d10fbcc53"
# eid = "5e57fcd0-8743-41c8-8360-d846a4e0469d"
label = "late"

# %% different
# eid = list(fits.keys())[0]
# label = "late"

# %% load data
fit = fits[eid]
brain_region = get_brain_regions(eid=eid, one=one)[0]
fluorescence, trials, continuous = load_session_data(one, eid, brain_region)
pose = continuous.pop("pose")
continuous.update(split_pose(pose))
events = events_from_trials(trials, event_splits=EVENTS)

# %% plot traces
fig, axes = plt.subplots(figsize=[4, 2])
# plot model and data
times = fit.tvec[fit.valid]
axes.plot(times, fit.target, label="data", lw=1)
axes.plot(times, fit.prediction, "r", label="model", lw=1)
# add reward times
reward_times = trials.groupby("feedbackType").get_group(1.0)["feedback_times"].values
for t in reward_times:
    axes.axvline(t, lw=0.25, alpha=1, zorder=-1, color="k")
# deco
axes.set_xlabel("time (s)", size="smaller")
axes.set_title(f"{subject}:{genotype}, R^2 = {fit.r2:.3f}", size="smaller")
axes.set_ylabel("fluorescence (mad-scored)", size="smaller")
axes.set_xlim(500, 560)
axes.set_xticklabels(axes.get_xticklabels(), size="smaller")
axes.set_ylim(-2, 4)
axes.set_yticklabels(axes.get_yticklabels(), size="smaller")
axes.legend(
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),  # outside to the right
    borderaxespad=0.0,
    fontsize="smaller",
)
sns.despine(axes.figure)
fig.tight_layout()
# store
if SAVE_PLOTS:
    fig.savefig(PLOT_FOLDER / f"{subject}_poster_traces.svg")


# %% build names map
names_map = dict()
# events.pop("firstMovement_times:choice=0.0")
for event in events.keys():
    if "signed_contrast" in event:
        l = event.split("signed_contrast")
        l.insert(1, "contrast")
        names_map[event] = "".join(l)
        continue
    if "choice" in event:
        parts = event.split("=")
        if parts[1] == "-1.0":
            parts[1] = "CCW"
        if parts[1] == "0.0":
            parts[1] = "missed"
        if parts[1] == "1.0":
            parts[1] = "CW"
        names_map[event] = "=".join(parts)
        continue
    if "feedbackType" in event:
        parts = event.split("=")
        if parts[1] == "-1.0":
            parts[1] = "no reward"
        if parts[1] == "1.0":
            parts[1] = "reward"
        names_map[event] = "=".join(parts)
        continue

    names_map[event] = event


lags = make_lags(N_LAGS)
fontsize = "smaller"
names = list(events.keys())

# convert sample lags to seconds, shared by both layouts
dt = fit.tvec[1] - fit.tvec[0]
lag_seconds = lags * dt

kernels = np.stack([get_kernel(fit, name) for name in names])
fig, axes = plt.subplots(figsize=[8, 0.2 * len(names) + 1.5])
# symmetric diverging scale centred on zero
limit = np.abs(kernels).max() * 0.8
# extent maps columns to seconds and rows to 0..n-1 (top row first)
extent = [
    lag_seconds[0] - dt / 2,
    lag_seconds[-1] + dt / 2,
    len(names) - 0.5,
    -0.5,
]
image = axes.matshow(
    kernels,
    aspect="auto",
    cmap="RdBu_r",
    vmin=-limit,
    vmax=limit,
    extent=extent,
)
# move the time axis to the bottom (matshow defaults it to the top)
axes.xaxis.set_ticks_position("bottom")
axes.xaxis.set_label_position("bottom")
axes.axvline(0, linestyle=":", color="k", lw=1)
if names_map is not None:
    names = [names_map[name] for name in names]
axes.set_yticks(range(len(names)))
axes.set_yticklabels(names, fontsize=fontsize)
axes.set_xlabel("time (s)", fontsize=fontsize)
axes.tick_params(labelsize=fontsize)
colorbar = fig.colorbar(image, ax=axes)
colorbar.set_label("coefficient", fontsize=fontsize)
colorbar.ax.tick_params(labelsize=fontsize)
axes.set_title(f"{subject}:{genotype} - {label} in training")
fig.tight_layout()
sns.despine(fig)
if SAVE_PLOTS:
    fig.savefig(PLOT_FOLDER / f"{subject}_poster_kernels_{label}.svg")

# %% compare to PSTHs
import pynapple as nap

tvec = make_time_grid(fluorescence.times()[0], fluorescence.times()[-1], DT)
psths = {}
bad_names = []
for event, times in events.items():
    signal = interpolate_to_grid(fluorescence, tvec)
    try:
        psths[event] = nap.compute_perievent(
            signal, events=times, window=(lags * dt)[[0, -1]]
        )
    except ValueError:
        bad_names.append(event)
        continue


# %%
names = list(events.keys())
psths_arr = np.stack([psths[name].d.mean(axis=1) for name in names])
fig, axes = plt.subplots(figsize=[8, 0.2 * len(names) + 1.5])
# symmetric diverging scale centred on zero
limit = np.abs(psths_arr).max()
# extent maps columns to seconds and rows to 0..n-1 (top row first)
extent = [
    lag_seconds[0] - dt / 2,
    lag_seconds[-1] + dt / 2,
    len(names) - 0.5,
    -0.5,
]
image = axes.matshow(
    psths_arr,
    aspect="auto",
    cmap="RdBu_r",
    vmin=-limit,
    vmax=limit,
    extent=extent,
)
# move the time axis to the bottom (matshow defaults it to the top)
axes.xaxis.set_ticks_position("bottom")
axes.xaxis.set_label_position("bottom")
axes.axvline(0, linestyle=":", color="k", lw=1)
if names_map is not None:
    names = [names_map[name] for name in names]
axes.set_yticks(range(len(names)))
axes.set_yticklabels(names, fontsize=fontsize)
axes.set_xlabel("time (s)", fontsize=fontsize)
axes.tick_params(labelsize=fontsize)
colorbar = fig.colorbar(image, ax=axes)
colorbar.set_label("coefficient", fontsize=fontsize)
colorbar.ax.tick_params(labelsize=fontsize)
axes.set_title(f"{subject}:{genotype} - {label} in training")
fig.tight_layout()
sns.despine(fig)
if SAVE_PLOTS:
    fig.savefig(PLOT_FOLDER / f"{subject}_poster_PSTHs_{label}.svg")


# %% per session events stacking
from iblphotometry.fpio import PhotometrySessionLoader
from data_loaders import load_trials
from datetime import datetime

kernels = {}
current_max = 0.0
for eid, fit in tqdm(fits.items()):
    psl = PhotometrySessionLoader(one=one, eid=eid)
    trials = load_trials(psl)
    date = datetime.strftime(one.eid2ref(eid)["date"], "%Y-%m-%d")
    events = events_from_trials(trials, event_splits=EVENTS)
    for event in events:
        kernel = get_kernel(fit, event)
        if event not in kernels:
            kernels[event] = {}
        kernels[event][date] = kernel
        # compute global max for scale
        limit = np.max([current_max, np.abs(kernel).max()])


# %%
import matplotlib.pyplot as plt

subject_folder = PLOT_FOLDER / subject
subject_folder.mkdir(exist_ok=True, parents=True)

for event, _kernels in kernels.items():
    fig, axes = plt.subplots()
    dates = list(_kernels.keys())
    kernels_mat = np.stack(list(_kernels.values()))
    # print(dates[0])
    # kernels_mat[0,:]=1 # for verification
    limit = np.abs(kernels_mat).max()
    # limit = 1.5
    lags = make_lags(N_LAGS)
    extent = [lags[0] * DT, lags[-1] * DT, 0.0, float(len(dates))]
    axes.matshow(
        kernels_mat,
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        extent=extent,
        origin="lower",
    )
    axes.axvline(0, linestyle=":", color="k", lw=1)
    axes.set_yticks(np.arange(len(dates)) + 0.5)
    axes.set_yticklabels(dates, size="smaller")
    axes.set_title(event)
    axes.set_aspect(0.1)
    fig.savefig(subject_folder / (event + ".png"))


# %% per-regressor contribution (leave-one-regressor-out)
# for fit in tqdm(fits):
#     deltas = delta_r_squared(fit, cv=None)  # in-sample; pass cv=5 for cross-validated
# deltas_file = Path(__file__).parent / f"{subject}-delta_rsq.pkl"

# # and write
# with open(deltas_file, "wb") as fH:
#     pickle.dump(deltas, fH)
