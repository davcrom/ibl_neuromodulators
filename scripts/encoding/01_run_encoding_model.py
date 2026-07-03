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

mpl.rcParams["figure.dpi"] = 284  # screen dpi adjustment

from data_loaders import load_session_data
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

NM = "5HT"
WHEN = "late"

# %% session selection
if NM == "DA":
    if WHEN == "early":
        eid = "6931684c-a721-4db8-9698-e3101d0e4a1b"  # first session
    if WHEN == "late":
        eid = "5e57fcd0-8743-41c8-8360-d846a4e0469d"  # last session
    brain_region = "SNc-l"  # TODO dataset-specific
    subject = one.eid2ref(eid)["subject"]

if NM == "5HT":
    if WHEN == "early":
        eid = "5c5a5e99-d353-496c-9c84-7aa657d81e44"
        brain_region = "DRN"  # TODO dataset-specific
    if WHEN == "late":
        eid = "9880ac8a-3efe-485f-bea4-bc3450c64e81"
        brain_region = "DR"  # TODO dataset-specific
        ...

# %% common
subject = one.eid2ref(eid)["subject"]
genotype = one.alyx.rest("subjects", "read", subject)["line"]
label = WHEN

# %% model config
DT = 0.1
N_LAGS = 30

# %% load and fit a single session
fluorescence, trials, continuous = load_session_data(one, eid, brain_region)
pose = continuous.pop("pose")
continuous.update(split_pose(pose))

tvec = make_time_grid(fluorescence.times()[0], fluorescence.times()[-1], DT)
EVENTS = {
    "stimOn_times": [None, "signed_contrast"],
    "response_times": [None, "choice"],
    "firstMovement_times": [None, "choice", "feedbackType"],
    "feedback_times": [None, "feedbackType"],
}

# EVENTS = {
#     "stimOn_times": "signed_contrast",
#     "response_times": "choice",
#     "firstMovement_times": "choice",
#     "feedback_times": "feedbackType",
# }

events = events_from_trials(trials, event_splits=EVENTS)
blocks = {
    **continuous_blocks(continuous, tvec),
    # **trial_constant_blocks(trials, tvec),
    **design_lagged(events, tvec, n_lags=N_LAGS),
    # **design_cosine(events, tvec, n_basis=10),
}
design, slices = build_design_matrix(blocks)
target = interpolate_to_grid(fluorescence, tvec)
fit = fit_encoding_model(design, target, slices, label=f"{subject}:{eid}", alpha=200)
print(f"R^2 = {fit.r2:.3f}")

# %% inspect the fit
axes = plot_prediction(fit)
axes.set_title(f"{subject}:{genotype}, R^2 = {fit.r2:.3f}")
sns.despine(axes.figure)
axes.set_xlim(500, 600)
axes.set_ylabel("fluorescence (mad-scored)")
axes.figure.savefig(
    PLOT_FOLDER / f"{subject}-{label}_fit_model_trace_comparison.pdf", dpi=300
)


# %% alpha brute force
from tqdm import tqdm
from encoding_model import _cv_r_squared
import numpy as np
import matplotlib.pyplot as plt

alphas = np.logspace(-2, 3, 20)
rsqs = []
for alpha in tqdm(alphas):
    # drop rows with NaNs from interpolation edges or missing support
    y = target.values[:, None]
    valid = ~np.isnan(design).any(axis=1) & ~np.isnan(y).any(axis=1)
    rsqs.append(_cv_r_squared(design[valid], target[valid], alpha, 5))


fig, axes = plt.subplots()
axes.plot(alphas, rsqs)

# %% plot kernels
axes = plot_kernels(
    fit, list(events), make_lags(N_LAGS), how="matshow", fontsize="large"
)


# %%
import matplotlib.pyplot as plt
import numpy as np
from encoding_model import get_kernel
import seaborn as sns

event_groups = ["stim", "Movement", "feedback"]
lags = make_lags(N_LAGS)
for event_group in event_groups:
    fig, axes = plt.subplots()
    for event_name in events.keys():
        if event_group in event_name.split(":")[0]:
            axes.plot(lags, get_kernel(fit, event_name), label=event_name, lw=1)
    kwargs = dict(lw=1, linestyle=":", alpha=0.5, color="k")
    axes.axhline(0, **kwargs)
    axes.axvline(0, **kwargs)
    axes.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    sns.despine(fig)

# %%
# fig, axes = plt.subplots(ncols=len(names), sharey=True, figsize=[3 * len(names), 3])
# for ax, name in zip(np.atleast_1d(axes), names):
#     ax.plot(lag_seconds, get_kernel(fit, name))
#     ax.set_title(name, fontsize=fontsize)
#     ax.axhline(0, linestyle=":", color="k", lw=1)
#     ax.axvline(0, linestyle=":", color="k", lw=1)
#     ax.set_xlabel("time (s)", fontsize=fontsize)
#     ax.tick_params(labelsize=fontsize)
# fig.tight_layout()


# %% per-regressor contribution (leave-one-regressor-out)
deltas = delta_r_squared(fit, cv=None)  # in-sample; pass cv=5 for cross-validated
print(deltas)

# %% plot
deltas = deltas.loc[list(blocks.keys())[::-1]]
axes = plot_delta_r_squared(deltas, order_by_magnitude=False)
axes.figure.suptitle(f"{subject}:{genotype}")
sns.despine(axes.figure)
axes.figure.tight_layout()
axes.figure.savefig(PLOT_FOLDER / f"{subject}-{label}_rsq_drops.pdf", dpi=300)

# %% PSTH of the signal for visual inspection
# plot_psths_from_trace(pd.Series(fluorescence.d, index=fluorescence.t), trials)
