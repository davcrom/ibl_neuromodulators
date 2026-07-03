# %% [markdown]
# # Lagged-regression schematic
# Explanatory figure for the photometry encoding model: depicts the model as the
# matrix multiplication it actually is,
#
#     y  =  X  @  beta
#
# with the measured fluorescence `y` as a column vector on the left, the design
# matrix `X` in the middle and the fitted weights `beta` as a column vector on
# the right.
#
# The design matrix is built with the same structure as `encoding_model.py`
# (continuous regressors, then one lagged/FIR block per event), but on synthetic
# data so the figure needs no ONE/Alyx access. `beta` carries hand-shaped kernels
# and `y = X @ beta` so the three panels are a genuine instance of the equation.

# %%
from pathlib import Path

import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

PLOT_FOLDER = Path(__file__).parent / "plots"
PLOT_FOLDER.mkdir(parents=True, exist_ok=True)

# --- colours (customise here) -----------------------------------------------
# Maps each event regressor to its colour in the figure. Neutral / muted by
# default so the schematic reads cleanly when embedded in a poster. Edit the
# hex values to recolour; the order also sets the block order in the matrix.
EVENT_COLORS: dict[str, str] = {
    "stimOn": "#128336",  # muted blue
    "firstMove": "#1c46ce",  # muted green
    "response": "#E18C25",  # muted mauve
    "feedback": "#b61313",  # muted tan
}
CONTINUOUS_COLOR = "#8c8c8c"  # one neutral grey for all continuous regressors

# --- config -----------------------------------------------------------------

N_TIME = 160  # rows of the design matrix (time samples shown)
N_LAGS = 12  # lag columns per event block
N_EVENTS_PER_TYPE = 6  # event occurrences drawn per event type

# continuous regressor columns, grouped under one block label
CONTINUOUS_REGRESSORS = ["wheel", "pose x", "pose y"]
CONTINUOUS_LABEL = "continuous"

rng = np.random.default_rng(0)


# --- design-matrix construction (mirrors encoding_model.py, numpy-only) ------


def make_lags(n_lags: int) -> np.ndarray:
    """Integer sample lags centred on zero (lag 0 == event onset)."""
    return np.arange(n_lags) - n_lags // 2


def shift(array: np.ndarray, n: int) -> np.ndarray:
    """Shift `array` by `n` samples, zero-filling vacated entries (no wrap)."""
    out = np.zeros_like(array)
    if n > 0:
        out[n:] = array[:-n]
    elif n < 0:
        out[:n] = array[-n:]
    else:
        out[:] = array
    return out


def lag_expand(impulse: np.ndarray, lags: np.ndarray) -> np.ndarray:
    """Lagged copies of an impulse train, one column per lag (the FIR block)."""
    return np.stack([shift(impulse, int(lag)) for lag in lags], axis=1)


def make_impulse_train(n_time: int, n_events: int) -> np.ndarray:
    """Binary impulse train with `n_events` ones, spaced so lags don't overlap.

    Events are placed one per equal-width bin (jittered within the bin's central
    60%), which guarantees a minimum spacing wider than the lag window so the
    per-event staircases stay visually separate.
    """
    margin = N_LAGS  # keep events clear of the edges so staircases stay on-grid
    edges = np.linspace(margin, n_time - margin, n_events + 1)
    train = np.zeros(n_time)
    for i in range(n_events):
        width = edges[i + 1] - edges[i]
        position = int(rng.uniform(edges[i] + 0.2 * width, edges[i] + 0.8 * width))
        train[position] = 1.0
    return train


def make_continuous(n_time: int) -> np.ndarray:
    """Smooth continuous regressor on [0, 1] (low-pass-filtered random walk)."""
    walk = np.cumsum(rng.standard_normal(n_time))
    kernel = np.hanning(21)
    smooth = np.convolve(walk, kernel / kernel.sum(), mode="same")
    smooth -= smooth.min()
    return smooth / smooth.max()


def make_kernel(n_lags: int, shape: str) -> np.ndarray:
    """Hand-shaped fitted kernel over lags (for the synthetic `beta`)."""
    lags = make_lags(n_lags)
    if shape == "bump":  # positive post-event response
        return np.exp(-0.5 * ((lags - 2) / 2.0) ** 2)
    if shape == "biphasic":  # positive then negative
        return np.exp(-0.5 * ((lags - 1) / 1.5) ** 2) - 0.7 * np.exp(
            -0.5 * ((lags - 5) / 2.0) ** 2
        )
    if shape == "decay":  # small causal decay
        kernel = np.where(lags >= 0, np.exp(-lags / 3.0), 0.0)
        return 0.6 * kernel
    if shape == "big_bump":  # large positive response
        return 1.4 * np.exp(-0.5 * ((lags - 3) / 2.5) ** 2)
    raise ValueError(shape)


# build the blocks in the model's order: continuous first, then event FIR blocks
lags = make_lags(N_LAGS)
blocks: dict[str, np.ndarray] = {}
for name in CONTINUOUS_REGRESSORS:
    blocks[name] = make_continuous(N_TIME)[:, None]
for name in EVENT_COLORS:
    impulse = make_impulse_train(N_TIME, N_EVENTS_PER_TYPE)
    blocks[name] = lag_expand(impulse, lags)

# assemble the design matrix and record each block's column span
design_columns, slices, start = [], {}, 0
for name, block in blocks.items():
    slices[name] = slice(start, start + block.shape[1])
    design_columns.append(block)
    start += block.shape[1]
design = np.concatenate(design_columns, axis=1)
n_features = design.shape[1]

# synthetic weights: small continuous weights + shaped event kernels
weights = np.zeros(n_features)
weights[slices["wheel"]] = 0.4
weights[slices["pose x"]] = -0.3
weights[slices["pose y"]] = 0.2
weights[slices["stimOn"]] = make_kernel(N_LAGS, "bump")
weights[slices["firstMove"]] = make_kernel(N_LAGS, "biphasic")
weights[slices["response"]] = make_kernel(N_LAGS, "decay")
weights[slices["feedback"]] = make_kernel(N_LAGS, "big_bump")

# the signal is literally the matrix product (+ a little observation noise)
target = design @ weights + 0.05 * rng.standard_normal(N_TIME)

# label-groups for the strips/labels: continuous columns collapse to one group,
# each event is its own group. (label, start, stop, colour)
n_continuous = len(CONTINUOUS_REGRESSORS)
groups: list[tuple[str, int, int, str]] = [
    (CONTINUOUS_LABEL, 0, n_continuous, CONTINUOUS_COLOR)
]
for name in EVENT_COLORS:
    span = slices[name]
    groups.append((name, span.start, span.stop, EVENT_COLORS[name]))


# --- figure ------------------------------------------------------------------

fig = plt.figure(figsize=(6.8, 7.2))
# columns: y | "=" | X | "·" | beta  (X kept narrow for the poster)
grid = GridSpec(
    1, 5, width_ratios=[0.55, 0.45, 2.9, 0.5, 0.55], wspace=0.18, figure=fig
)
ax_y = fig.add_subplot(grid[0, 0])
ax_eq = fig.add_subplot(grid[0, 1])
ax_x = fig.add_subplot(grid[0, 2])
ax_dot = fig.add_subplot(grid[0, 3])
ax_b = fig.add_subplot(grid[0, 4])

signal_limit = np.abs(target).max()
weight_limit = np.abs(weights).max()

# y: the measured fluorescence as a column vector (time increases downward)
ax_y.imshow(
    target[:, None],
    aspect="auto",
    cmap="RdBu_r",
    vmin=-signal_limit,
    vmax=signal_limit,
    interpolation="nearest",
)
ax_y.set_xticks([])
ax_y.set_yticks([0, N_TIME - 1])
ax_y.set_yticklabels(["0", "T"])
ax_y.set_ylabel("time  →", fontsize=11)

# X: the design matrix
ax_x.imshow(
    design,
    aspect="auto",
    cmap="Greys",
    vmin=0,
    vmax=1,
    interpolation="nearest",
)
ax_x.set_yticks([])
ax_x.set_xticks([])
ax_x.set_xlabel("regressors  →", fontsize=11)

# beta: the fitted weights as a column vector
ax_b.imshow(
    weights[:, None],
    aspect="auto",
    cmap="RdBu_r",
    vmin=-weight_limit,
    vmax=weight_limit,
    interpolation="nearest",
)
ax_b.set_xticks([])
ax_b.set_yticks([0, n_features - 1])
ax_b.set_yticklabels(["0", "P"])

# block dividers + coloured membership strip above X and beside beta, by group
strip_height = 0.025 * N_TIME
strip_gap = 0.012 * N_TIME
label_offset = 0.055 * N_TIME
for label, col_start, col_stop, color in groups:
    # faint divider between groups on the design matrix (skip the leftmost edge)
    if col_start > 0:
        ax_x.axvline(col_start - 0.5, color="w", lw=1.2)
    # colour strip just above the matrix marking this group's columns
    ax_x.add_patch(
        plt.Rectangle(
            (col_start - 0.5, -0.5 - strip_gap - strip_height),
            col_stop - col_start,
            strip_height,
            color=color,
            clip_on=False,
        )
    )
    # rotated group label above the strip
    ax_x.annotate(
        label,
        xy=(
            (col_start + col_stop - 1) / 2,
            -0.5 - strip_gap - strip_height - label_offset,
        ),
        ha="center",
        va="bottom",
        rotation=40,
        rotation_mode="anchor",
        fontsize=9,
        color=color,
        annotation_clip=False,
    )
    # matching colour strip flush to beta's left edge, and divider between groups
    ax_b.add_patch(
        plt.Rectangle(
            (-0.5 - 0.55, col_start - 0.5),
            0.45,
            col_stop - col_start,
            color=color,
            clip_on=False,
        )
    )
    if col_start > 0:
        ax_b.axhline(col_start - 0.5, color="w", lw=1.2)

# panel labels, aligned above everything (transAxes y matched across panels)
label_y = 1.16
for axes, text in ((ax_y, "fluorescence"), (ax_x, "design matrix"), (ax_b, "weights")):
    axes.text(
        0.5,
        label_y,
        text,
        transform=axes.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
    )

# explanation of the lag structure, placed clear of the matrix
ax_x.text(
    0.5,
    -0.13,
    f"each event block = impulse train copied across {N_LAGS} lags  (lag 0 = event onset)",
    transform=ax_x.transAxes,
    ha="center",
    va="top",
    fontsize=8.5,
    color="0.35",
)

# the operators between the panels
for axes, symbol in ((ax_eq, "="), (ax_dot, r"$\cdot$")):
    axes.axis("off")
    axes.text(0.5, 0.5, symbol, ha="center", va="center", fontsize=28)

fig.suptitle(
    "Lagged-regression encoding model:   $y = X\\,\\beta$", fontsize=14, y=0.98
)
fig.subplots_adjust(top=0.74, bottom=0.13, left=0.09, right=0.93)

out_pdf = PLOT_FOLDER / "lagged_regression_schematic.pdf"
out_png = PLOT_FOLDER / "lagged_regression_schematic.png"
out_svg = PLOT_FOLDER / "lagged_regression_schematic.svg"
fig.savefig(out_pdf, dpi=300)
fig.savefig(out_png, dpi=150)
fig.savefig(out_svg)
print(f"saved {out_pdf}")
print(f"saved {out_png}")
print(f"saved {out_svg}")

# %%
