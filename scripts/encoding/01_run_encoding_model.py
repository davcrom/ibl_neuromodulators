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

from data_loaders import load_session_data
from encoding_model import (
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
    plot_prediction,
    plot_kernels,
    plot_cosine_basis,
    delta_r_squared,
    plot_delta_r_squared,
)

one = ONE(cache_rest=None)

# %% find the photometry sessions and show subjects
django = [
    "users__username,laura.silva",
    "lab__name,mainenlab",
    "projects__name__icontains,ibl_fibrephotometry",
    "data_dataset_session_related__name__icontains,lightning",
]
sessions = one.alyx.rest("sessions", "list", django=django)

# %%
# genotype and session count per subject
# for subject in sorted({session["subject"] for session in sessions}):
#     n_sessions = len(eids_for_subject(sessions, subject))
#     line = one.alyx.rest("subjects", "read", subject)["line"]
#     print(subject, line, n_sessions)

# %% pick a subject
subject = "ZFM-09343"
brain_region = "SNc-l"  # TODO dataset-specific
eids = [session["id"] for session in sessions if session["subject"] == subject]

# %%
eid = "6931684c-a721-4db8-9698-e3101d0e4a1b"
brain_region = "SNc-l"  # TODO dataset-specific

# model config
DT = 0.1
N_LAGS = 50

# %% preview the cosine-bump basis (to choose design_cosine parameters)
# plot_cosine_basis(n_basis=10, rcos_duration=2.5, rcos_nloffset=0.2, dt=DT)

# %% load and fit a single session
# To use raised-cosine kernels instead, `from encoding_model import design_cosine`
# and swap `design_lagged(events, tvec, n_lags=N_LAGS)` for
# `design_cosine(events, tvec, n_basis=10, rcos_duration=2.5, rcos_nloffset=0.2)`.
# eid = eids[-1]
fluorescence, trials, continuous = load_session_data(one, eid, brain_region)

tvec = make_time_grid(fluorescence.times()[0], fluorescence.times()[-1], DT)
EVENTS = {
    "stimOn_times": 'signed_contrast',
    # "goCue_times": None,
    "response_times": 'choice',
    "firstMovement_times": 'choice',
    # "intervals_0": None,
    # "intervals_1": None,
    "feedback_times": "feedbackType",
}
events = events_from_trials(trials, event_splits=EVENTS)
blocks = {
    **continuous_blocks(continuous, tvec),
    # **trial_constant_blocks(trials, tvec),
    **design_lagged(events, tvec, n_lags=N_LAGS),
    # **design_cosine(events, tvec, n_basis=10),
}
design, slices = build_design_matrix(blocks)
target = interpolate_to_grid(fluorescence, tvec)
fit = fit_encoding_model(design, target, slices, label=f"{subject}:{eid}")
print(f"R^2 = {fit.r2:.3f}")

# %% inspect the fit
plot_prediction(fit)
# plot_kernels(fit, list(events), make_lags(N_LAGS))

# %% per-regressor contribution (leave-one-regressor-out)
deltas = delta_r_squared(fit)  # in-sample; pass cv=5 for cross-validated
print(deltas)
plot_delta_r_squared(deltas)

# %% PSTH of the signal for visual inspection
plot_psths_from_trace(pd.Series(fluorescence.d, index=fluorescence.t), trials)

# %% fit every session of the subject
# fits = {}
# for eid in eids:
#     fluorescence, trials, continuous = load_session_data(one, eid, brain_region)
#     tvec = make_time_grid(fluorescence.times()[0], fluorescence.times()[-1], DT)
#     blocks = {
#         **continuous_blocks(continuous, tvec),
#         **trial_constant_blocks(trials, tvec),
#         **design_lagged(events_from_trials(trials), tvec, n_lags=N_LAGS),
#     }
#     design, slices = build_design_matrix(blocks)
#     target = interpolate_to_grid(fluorescence, tvec)
#     fits[eid] = fit_encoding_model(design, target, slices, label=f"{subject}:{eid}")
#     print(eid, f"R^2 = {fits[eid].r2:.3f}")
