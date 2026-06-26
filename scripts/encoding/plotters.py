"""Plotting helpers for the lagged-regression encoding model.

Visualisation of fitted models (`EncodingFit`) and their building blocks:
prediction-vs-data traces, fitted kernels, the raised-cosine basis and
per-regressor ΔR² bars. All computation lives in `encoding_model`; these
functions only render its results.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from encoding_model import EncodingFit, get_kernel, raised_cosine_basis


def plot_prediction(fit: EncodingFit, axes: plt.Axes = None) -> plt.Axes:
    """Plot the measured signal and the model prediction over time."""
    if axes is None:
        _, axes = plt.subplots()
    times = fit.tvec[fit.valid]
    axes.plot(times, fit.target, label="data")
    axes.plot(times, fit.prediction, "r", label="model")
    axes.set_xlabel("time (s)")
    axes.set_title(f"{fit.label}  (R$^2$ = {fit.r2:.3f})")
    axes.legend()
    return axes


def plot_kernels(
    fit: EncodingFit,
    names: list[str],
    lags: np.ndarray,
    how: str = "line",
    fontsize: float | str = "small",
) -> plt.Figure:
    """Plot the fitted lagged kernel for each named event block.

    Assumes the lagged basis: a block's coefficients are the kernel itself.

    Args:
        fit (EncodingFit): a fitted encoding model.
        names (list[str]): event block names to plot (keys in `fit.slices`).
        lags (np.ndarray): sample lags used to build the kernels.
        how ({"line", "matshow"}): "line" draws one line panel per kernel;
            "matshow" stacks the kernels into a single heatmap (one row per
            name). Defaults to "line".
        fontsize (float | str): font size for labels and ticks (any matplotlib
            size, e.g. 8 or "small"). Defaults to "small".

    Returns:
        plt.Figure: the kernel figure.
    """
    # convert sample lags to seconds, shared by both layouts
    dt = fit.tvec[1] - fit.tvec[0]
    lag_seconds = lags * dt

    if how == "matshow":
        kernels = np.stack([get_kernel(fit, name) for name in names])
        fig, axes = plt.subplots(figsize=[8, 0.2 * len(names) + 1.5])
        # symmetric diverging scale centred on zero
        limit = np.abs(kernels).max()
        # extent maps columns to seconds and rows to 0..n-1 (top row first)
        extent = [
            lag_seconds[0] - dt / 2,
            lag_seconds[-1] + dt / 2,
            len(names) - 0.5,
            -0.5,
        ]
        image = axes.matshow(
            kernels, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit, extent=extent
        )
        # move the time axis to the bottom (matshow defaults it to the top)
        axes.xaxis.set_ticks_position("bottom")
        axes.xaxis.set_label_position("bottom")
        axes.axvline(0, linestyle=":", color="k", lw=1)
        axes.set_yticks(range(len(names)))
        axes.set_yticklabels(names, fontsize=fontsize)
        axes.set_xlabel("time (s)", fontsize=fontsize)
        axes.tick_params(labelsize=fontsize)
        colorbar = fig.colorbar(image, ax=axes)
        colorbar.set_label("coefficient", fontsize=fontsize)
        colorbar.ax.tick_params(labelsize=fontsize)
        fig.tight_layout()
        return fig

    fig, axes = plt.subplots(ncols=len(names), sharey=True, figsize=[3 * len(names), 3])
    for ax, name in zip(np.atleast_1d(axes), names):
        ax.plot(lag_seconds, get_kernel(fit, name))
        ax.set_title(name, fontsize=fontsize)
        ax.axhline(0, linestyle=":", color="k", lw=1)
        ax.axvline(0, linestyle=":", color="k", lw=1)
        ax.set_xlabel("time (s)", fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
    fig.tight_layout()
    return fig


def plot_cosine_basis(
    n_basis: int = 10,
    rcos_duration: float = 2.5,
    rcos_nloffset: float = 0.2,
    dt: float = 0.1,
    axes: plt.Axes = None,
) -> plt.Axes:
    """Plot the log-raised-cosine bump basis for the given parameters.

    Args:
        n_basis (int): number of bumps.
        rcos_duration (float): kernel window in seconds.
        rcos_nloffset (float): log-warp offset in seconds.
        dt (float): time-grid resolution in seconds.
        axes (plt.Axes): axes to draw on; created if None.

    Returns:
        plt.Axes: the axes, with one line per bump.
    """
    if axes is None:
        _, axes = plt.subplots()
    basis = raised_cosine_basis(n_basis, rcos_duration, rcos_nloffset, dt)
    times = np.arange(basis.shape[0]) * dt
    axes.plot(times, basis)
    axes.set_xlabel("time after event (s)")
    axes.set_ylabel("basis weight")
    axes.set_title(
        f"raised-cosine basis (n_basis={n_basis}, dur={rcos_duration}s, "
        f"offset={rcos_nloffset}s)"
    )
    return axes


def plot_delta_r_squared(
    deltas: pd.Series, order_by_magnitude: bool = True, axes: plt.Axes = None
) -> plt.Axes:
    """Bar chart of per-regressor ΔR².

    Args:
        deltas (pd.Series): ΔR² indexed by block name (from `delta_r_squared`).
        order ({"magnitude", "name"}): bar ordering. "magnitude" puts the
            largest drop at the top; "name" sorts alphabetically (A at top).
            Defaults to "magnitude".
        axes (plt.Axes): axes to draw on; created if None.

    Returns:
        plt.Axes: the bar-chart axes.
    """
    # barh draws the first row at the bottom, so order so the intended row
    # lands at the top last
    if order_by_magnitude:
        deltas = deltas.sort_values(ascending=True)
    if axes is None:
        _, axes = plt.subplots()
    axes.barh(deltas.index, deltas.values)
    axes.axvline(0, linestyle=":", color="k", lw=1)
    axes.set_xlabel("ΔR² (drop when left out)")
    axes.figure.tight_layout()
    return axes
