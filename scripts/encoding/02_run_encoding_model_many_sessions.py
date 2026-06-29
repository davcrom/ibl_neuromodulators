# %% [markdown]
# # Photometry encoding model
# Run the encoding model for one subject's sessions.
#
# Notebook-style script: run cell-by-cell. The `modelling/` directory must be
# importable (run from here, or add it to the path).

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

mpl.rcParams["figure.dpi"] = 284  # screen dpi adjustment

from data_loaders import load_session_data, get_brain_regions
from encoding_model import (
    split_pose,
    make_time_grid,
    make_lags,
    events_from_trials,
    design_lagged,
    design_cosine,
    continuous_blocks,
    trial_constant_blocks,
    build_design_matrix,
    interpolate_to_grid,
    fit_encoding_model,
    delta_r_squared,
    get_kernel,
)
from plotters import (
    plot_prediction,
    plot_kernels,
    plot_cosine_basis,
    plot_delta_r_squared,
)

one = ONE()

PLOT_FOLDER = Path(__file__).parent / "plots"
PLOT_FOLDER.mkdir(parents=True, exist_ok=True)

# %% subject selection
# subject = "ZFM-09365"  # 5HT
# subject = "ZFM-09343"  # DA
# subject = "ZFM-09439"  # 5HT-2
subject = "ZFM-08871"  # DBh

eids = (Path(__file__).parent / f"{subject}-sessions.txt").read_text().splitlines()

# %%
# subject = one.eid2ref(eid)['subject']
genotype = one.alyx.rest("subjects", "read", subject)["line"]

# model config
DT = 0.1
N_LAGS = 50

# %% fit every session of the subject
EVENTS = {
    "stimOn_times": "signed_contrast",
    "response_times": "choice",
    "firstMovement_times": "choice",
    "feedback_times": "feedbackType",
}

# %% compute fits
fits = {}
for eid in tqdm(eids):
    # this defaults to the first brain region in the animal
    # TODO generalize for the future
    brain_region = get_brain_regions(eid=eid, one=one)[0]
    # splitting pose should be a kwarg of load_session_data
    fluorescence, trials, continuous = load_session_data(one, eid, brain_region)
    pose = continuous.pop("pose")
    continuous.update(split_pose(pose))
    # the model
    tvec = make_time_grid(fluorescence.times()[0], fluorescence.times()[-1], DT)
    events = events_from_trials(trials, event_splits=EVENTS)
    blocks = {
        **continuous_blocks(continuous, tvec),
        # **trial_constant_blocks(trials, tvec),
        **design_lagged(events, tvec, n_lags=N_LAGS),
        # **design_cosine(events, tvec, n_basis=10),
    }
    design, slices = build_design_matrix(blocks)
    target = interpolate_to_grid(fluorescence, tvec)
    fits[eid] = fit_encoding_model(
        design, target, slices, label=f"{subject}:{eid}", alpha=50
    )
    #  drop design matrix for lower memory profile
    fits[eid].design = None
    # TODO also calc delta rsq

# store
fits_file = Path(__file__).parent / f"{subject}-fits.pkl"
with open(fits_file, "wb") as fH:
    pickle.dump(fits, fH)

# %% reloads the fits
fits_file = Path(__file__).parent / f"{subject}-fits.pkl"
with open(fits_file, "rb") as fH:
    fits = pickle.load(fH)

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
    # limit = np.abs(kernels_mat).max()
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
