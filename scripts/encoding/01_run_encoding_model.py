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
mpl.rcParams['figure.dpi'] = 284 # screen dpi adjustment

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

PLOT_FOLDER = Path(__file__).parent / 'plots'
PLOT_FOLDER.mkdir(parents=True,exist_ok=True)

# %% DA
eid = "6931684c-a721-4db8-9698-e3101d0e4a1b" # first session
label = 'early'

eid = "5e57fcd0-8743-41c8-8360-d846a4e0469d" # last session
label = 'late'
brain_region = "SNc-l"  # TODO dataset-specific

# %% 5-HT
eid = '5c5a5e99-d353-496c-9c84-7aa657d81e44'
label = 'early'
brain_region = "DRN"  # TODO dataset-specific
print(one.eid2ref(eid)['date'])

# %%
eid = 'a3a3c3f1-78c3-4dda-ad25-59184989ed1f' # has tracking
# eid = "234b622f-12ac-4c49-a89a-077d01df9ce3"

label = 'late'
brain_region = "DR"  # TODO dataset-specific
print(one.eid2ref(eid)['date'])


# %%
from iblphotometry.fpio import PhotometrySessionLoader
psl = PhotometrySessionLoader(one=one, eid=eid)
psl.load_photometry()
print(psl.photometry['GCaMP'].columns)
from iblphotometry.plotters import plot_photometry_traces_from_eid
plot_photometry_traces_from_eid(eid=eid, one=one)
brain_region = "DR"

# %%
subject = one.eid2ref(eid)['subject']
genotype = one.alyx.rest('subjects','read', subject)['line']

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
pose = continuous.pop('pose')
continuous.update(split_pose(pose))

tvec = make_time_grid(fluorescence.times()[0], fluorescence.times()[-1], DT)
EVENTS = {
    "stimOn_times": 'signed_contrast',
    "response_times": 'choice',
    "firstMovement_times": 'choice',
    "intervals_0": None,
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
axes = plot_prediction(fit)
axes.set_title(f"{subject}:{genotype}, R^2 = {fit.r2:.3f}")
sns.despine(axes.figure)
axes.set_xlim(500,600)
axes.set_ylabel('fluorescence (mad-scored)')
axes.figure.savefig(PLOT_FOLDER / f'{subject}-{label}_fit_model_trace_comparison.pdf', dpi=300)

# %%
axes = plot_kernels(fit, list(events), make_lags(N_LAGS), how='matshow', fontsize='large')
axes.figure.savefig(PLOT_FOLDER / f'{subject}-{label}_kernels.pdf', dpi=300)

# %% per-regressor contribution (leave-one-regressor-out)
deltas = delta_r_squared(fit, cv=None)  # in-sample; pass cv=5 for cross-validated
print(deltas)

# %% plot
deltas = deltas.loc[list(blocks.keys())[::-1]]
axes = plot_delta_r_squared(deltas, order_by_magnitude=False)
axes.figure.suptitle(f"{subject}:{genotype}")
sns.despine(axes.figure)
axes.figure.tight_layout()
axes.figure.savefig(PLOT_FOLDER / f'{subject}-{label}_rsq_drops.pdf', dpi=300)

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
