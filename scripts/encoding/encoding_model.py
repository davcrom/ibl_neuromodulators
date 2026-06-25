"""Lagged-regression (kernel) encoding model for fiber photometry.

Predicts a preprocessed photometry signal from behavioural events (expanded
into time-lagged or raised-cosine kernels), trial-constant variables and
continuous regressors (pose, wheel) via ridge regression.

Pipeline (all steps are separate, composable functions):
    1. prepare events as a name -> nap.Ts mapping (`events_from_trials`);
    2. expand them into kernel blocks with ONE of the two architectures —
       `design_lagged` (FIR / one column per lag) or `design_cosine`
       (log-raised-cosine bumps); each has its own parameters;
    3. build the shared, non-diverging blocks (`continuous_blocks`,
       `trial_constant_blocks`), merge all blocks and assemble the numpy
       design matrix (`build_design_matrix`, which records each block's
       column span as a `slices` map);
    4. fit ridge on the assembled design (`fit_encoding_model`).

Convention: anything living on the model grid is a pynapple object carrying its
own time axis (`nap.Tsd` / `nap.TsdFrame`). A function only takes a separate
`tvec` argument when its input is NOT already on the grid (event times, trial
intervals, a native-time-base series).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate

import pynapple as nap
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict

# behavioural events -> the trial column to split each by (None = unsplit)
DEFAULT_EVENTS = {
    "stimOn_times": None,
    "goCue_times": None,
    "response_times": None,
    "firstMovement_times": None,
    "intervals_0": None,
    "intervals_1": None,
    "feedback_times": "feedbackType",
}
# trial-constant columns held flat across each trial interval
DEFAULT_TRIAL_CONSTANTS = ["choice", "probabilityLeft", "contrast"]


@dataclass
class EncodingFit:
    """Result of fitting the encoding model to one session.

    Attributes:
        tvec (np.ndarray): full model time grid (the target's time axis).
        valid (np.ndarray): boolean mask of grid samples kept (no NaNs).
        design (np.ndarray): design matrix over valid samples.
        target (np.ndarray): measured signal over valid samples, shape (n, 1).
        prediction (np.ndarray): model prediction over valid samples.
        coefficients (np.ndarray): fitted coefficients, shape (n_features, 1).
        intercept (np.ndarray): fitted intercept.
        slices (dict[str, slice]): block name -> column span in the design.
        r2 (float): in-sample coefficient of determination.
        alpha (float): ridge regularisation strength used for the fit.
        label (str): session label for plots.
    """

    tvec: np.ndarray
    valid: np.ndarray
    design: np.ndarray
    target: np.ndarray
    prediction: np.ndarray
    coefficients: np.ndarray
    intercept: np.ndarray
    slices: dict[str, slice]
    r2: float
    alpha: float
    label: str = ""


# --- time grid helpers ---


def make_time_grid(t_start: float, t_stop: float, dt: float) -> np.ndarray:
    """Build a uniform model time grid spanning [t_start, t_stop)."""
    return np.arange(t_start, t_stop, dt)


def make_lags(n_lags: int) -> np.ndarray:
    """Integer sample lags centred on zero (lag 0 == event onset)."""
    return np.arange(n_lags) - n_lags // 2


def times_to_indices(
    times: np.ndarray, tvec: np.ndarray, clip: bool = False
) -> np.ndarray:
    """Map times to the nearest sample indices on the uniform grid `tvec`.

    Args:
        times (np.ndarray): event/query times in seconds.
        tvec (np.ndarray): uniform model time grid.
        clip (bool): if True, clamp indices to [0, len(tvec)] for use as slice
            bounds; if False, return raw indices (caller masks out-of-range).

    Returns:
        np.ndarray: integer indices, one per input time.
    """
    # robust grid spacing (averages out arange float accumulation)
    dt = (tvec[-1] - tvec[0]) / (tvec.size - 1)
    indices = np.round((times - tvec[0]) / dt).astype(int)
    if clip:
        indices = np.clip(indices, 0, tvec.size)
    return indices


# --- regressor constructors (return pynapple objects on the `tvec` grid) ---


def make_event_regressor(events: nap.Ts, tvec: np.ndarray) -> nap.Tsd:
    """Binary regressor: 1.0 at the nearest grid sample of each event time.

    Args:
        events (nap.Ts): event timestamps.
        tvec (np.ndarray): uniform model time grid.

    Returns:
        nap.Tsd: 1-D binary regressor on `tvec`.
    """
    indices = times_to_indices(events.times(), tvec)
    # keep only events falling inside the grid
    valid = (indices >= 0) & (indices < tvec.size)
    regressor = np.zeros(tvec.size)
    regressor[indices[valid]] = 1.0
    return nap.Tsd(t=tvec, d=regressor)


def lag_expand(regressor: nap.Tsd, lags: np.ndarray) -> nap.TsdFrame:
    """Build lagged copies of a regressor, one column per lag.

    Convention: column j is `regressor` shifted by `lags[j]` samples
    (zero-padded, no wrap), so a positive lag shifts the event's contribution
    later in time. A fitted positive-lag coefficient is therefore the signal's
    post-event response (lag 0 == event onset).

    Args:
        regressor (nap.Tsd): 1-D regressor on the model grid.
        lags (np.ndarray): integer sample lags, e.g. arange(-25, 25).

    Returns:
        nap.TsdFrame: matrix of shape (len(regressor), len(lags)) on the grid.
    """
    values = regressor.values

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

    columns = np.stack([shift(values, int(lag)) for lag in lags], axis=1)
    return nap.TsdFrame(t=regressor.times(), d=columns)


def raised_cosine_basis(
    n_basis: int, rcos_duration: float, rcos_nloffset: float, dt: float
) -> np.ndarray:
    """Build a causal log-raised-cosine "bump" basis.

    Reproduces the brain-wide-map basis (neurencoding.utils.nonlinear_rcos):
    time is log-warped so bumps are dense just after the event and sparse later.
    The basis spans [0, rcos_duration] after the event.

    Args:
        n_basis (int): number of bumps.
        rcos_duration (float): kernel window in seconds.
        rcos_nloffset (float): log-warp offset in seconds (must be > 0); smaller
            packs more bumps near the event.
        dt (float): time-grid resolution in seconds.

    Returns:
        np.ndarray: (n_kernel, n_basis) basis, n_kernel = ceil(duration / dt).
    """
    if rcos_nloffset <= 0:
        raise ValueError("rcos_nloffset must be positive and nonzero")

    def n_bins(seconds: float) -> int:
        """Number of grid bins spanning `seconds` (BWM binfun)."""
        return int(np.ceil(seconds / dt))

    def log_warp(x: np.ndarray) -> np.ndarray:
        """Log time-warp (small epsilon keeps log(0) finite)."""
        return np.log(x + 1e-20)

    def bump(x: np.ndarray, center: np.ndarray, spacing: float) -> np.ndarray:
        """Raised-cosine bump, clamped to one period and scaled to [0, 1]."""
        inner = np.clip(np.pi * (x - center) / (2 * spacing), -np.pi, np.pi)
        return (np.cos(inner) + 1) / 2

    # bump centres in log-warped time (after Pillow; neurencoding.nonlinear_rcos)
    n_kernel = n_bins(rcos_duration)
    offset_bins = n_bins(rcos_nloffset)
    y_range = log_warp(np.array([0, n_kernel]) + offset_bins)
    spacing = (y_range[1] - y_range[0]) / (n_basis - 1)
    centers = y_range[0] + spacing * np.arange(n_basis)
    sample = log_warp(np.arange(n_kernel) + offset_bins)
    return bump(sample[:, None], centers[None, :], spacing)


def raised_cosine_expand(
    regressor: nap.Tsd, n_basis: int, rcos_duration: float, rcos_nloffset: float
) -> nap.TsdFrame:
    """Expand an event impulse train onto a log-raised-cosine kernel basis.

    Returns one column per bump (the impulse train convolved with each basis
    function). Fitted coefficients weight the bumps, and the kernel is their
    weighted sum. See `raised_cosine_basis` for the basis itself.

    Args:
        regressor (nap.Tsd): 1-D binary event regressor on the grid.
        n_basis (int): number of raised-cosine bumps.
        rcos_duration (float): post-event kernel window in seconds.
        rcos_nloffset (float): log-warp offset in seconds.

    Returns:
        nap.TsdFrame: (len(regressor), n_basis) block on the grid.
    """
    tvec = regressor.times()
    dt = (tvec[-1] - tvec[0]) / (tvec.size - 1)
    basis = raised_cosine_basis(n_basis, rcos_duration, rcos_nloffset, dt)

    # convolve the impulse train with each bump, truncate to the grid length
    impulse = regressor.values
    block = np.zeros((tvec.size, n_basis))
    for j in range(n_basis):
        block[:, j] = np.convolve(impulse, basis[:, j])[: tvec.size]
    return nap.TsdFrame(t=tvec, d=block)


def make_trial_constant(
    trials: pd.DataFrame, column: str, tvec: np.ndarray
) -> nap.Tsd:
    """Step regressor: `column` held constant across each trial interval.

    Args:
        trials (pd.DataFrame): trials table with interval columns.
        column (str): trial column whose value fills each interval.
        tvec (np.ndarray): uniform model time grid.

    Returns:
        nap.Tsd: 1-D step regressor on `tvec`.
    """
    values = np.zeros(tvec.size)
    for _, row in trials.iterrows():
        start, stop = times_to_indices(
            np.array([row["intervals_0"], row["intervals_1"]]), tvec, clip=True
        )
        values[start:stop] = row[column]
    return nap.Tsd(t=tvec, d=values)


def interpolate_to_grid(
    tsd: nap.Tsd, tvec: np.ndarray, kind: str = "quadratic"
) -> nap.Tsd | nap.TsdFrame:
    """Resample a pynapple series or frame onto the model grid `tvec`.

    Args:
        tsd (nap.Tsd): source series or frame on its native time base.
        tvec (np.ndarray): uniform model time grid.
        kind (str): scipy interpolation kind. Defaults to "quadratic".

    Returns:
        nap.Tsd | nap.TsdFrame: values on `tvec`; out-of-support samples are NaN.
    """
    interpolator = scipy.interpolate.interp1d(
        tsd.times(),
        tsd.values,
        kind=kind,
        axis=0,
        bounds_error=False,
        fill_value=np.nan,
    )
    values = interpolator(tvec)
    if values.ndim == 1:
        return nap.Tsd(t=tvec, d=values)
    return nap.TsdFrame(t=tvec, d=values)


# --- design matrix: events -> kernel blocks (the two architectures) ---


def split_event(
    trials: pd.DataFrame, event_name: str, split_by: str
) -> dict[str, nap.Ts]:
    """Split one event's times into separate regressors by a trial column.

    Groups `trials` by `split_by` and returns the `event_name` times of each
    group, keyed ``f"{event_name}:{split_by}={value}"``.

    Args:
        trials (pd.DataFrame): trials table.
        event_name (str): column of event times to split (e.g. "stimOn_times").
        split_by (str): column to group trials on (e.g. "choice", "feedbackType").

    Returns:
        dict[str, nap.Ts]: per-group event timestamps (NaNs dropped).
    """
    events = {}
    for value, group in trials.groupby(split_by):
        times = group[event_name].values
        events[f"{event_name}:{split_by}={value}"] = nap.Ts(t=times[~np.isnan(times)])
    return events


def events_from_trials(
    trials: pd.DataFrame, event_splits: dict[str, str | None] | None = None
) -> dict[str, nap.Ts]:
    """Collect behavioural event times as a name -> nap.Ts mapping.

    Args:
        trials (pd.DataFrame): trials table.
        event_splits (dict[str, str | None]): maps each event-time column to the
            trial column to split it by (via `split_event`), or None to keep the
            event unsplit (all trials treated equal). Defaults to DEFAULT_EVENTS.

    Returns:
        dict[str, nap.Ts]: event name -> event timestamps (NaNs dropped). Split
        events are keyed ``f"{event}:{split_by}={value}"``.
    """
    event_splits = DEFAULT_EVENTS if event_splits is None else event_splits
    events = {}
    for name, split_by in event_splits.items():
        if split_by is None:
            times = trials[name].values
            events[name] = nap.Ts(t=times[~np.isnan(times)])
        else:
            events.update(split_event(trials, name, split_by))
    return events


def design_lagged(
    events: dict[str, nap.Ts], tvec: np.ndarray, n_lags: int = 50
) -> dict[str, nap.TsdFrame]:
    """Event-kernel blocks using lagged (FIR) regressors.

    Args:
        events (dict[str, nap.Ts]): event name -> event timestamps.
        tvec (np.ndarray): uniform model time grid.
        n_lags (int): number of lag samples per kernel.

    Returns:
        dict[str, nap.TsdFrame]: event name -> (len(tvec), n_lags) block.
    """
    lags = make_lags(n_lags)
    return {
        name: lag_expand(make_event_regressor(ts, tvec), lags)
        for name, ts in events.items()
    }


def design_cosine(
    events: dict[str, nap.Ts],
    tvec: np.ndarray,
    n_basis: int = 10,
    rcos_duration: float = 2.5,
    rcos_nloffset: float = 0.2,
) -> dict[str, nap.TsdFrame]:
    """Event-kernel blocks using log-raised-cosine bump regressors.

    Args:
        events (dict[str, nap.Ts]): event name -> event timestamps.
        tvec (np.ndarray): uniform model time grid.
        n_basis (int): number of raised-cosine bumps per kernel.
        rcos_duration (float): post-event kernel window in seconds.
        rcos_nloffset (float): log-warp offset in seconds.

    Returns:
        dict[str, nap.TsdFrame]: event name -> (len(tvec), n_basis) block.
    """
    return {
        name: raised_cosine_expand(
            make_event_regressor(ts, tvec), n_basis, rcos_duration, rcos_nloffset
        )
        for name, ts in events.items()
    }


# --- design matrix: shared (non-diverging) blocks + assembly ---


def continuous_blocks(
    continuous: dict[str, nap.TsdFrame], tvec: np.ndarray
) -> dict[str, nap.Tsd | nap.TsdFrame]:
    """Resample each continuous regressor onto the model grid.

    Args:
        continuous (dict[str, nap.TsdFrame]): regressor name -> native series.
        tvec (np.ndarray): uniform model time grid.

    Returns:
        dict[str, nap.Tsd | nap.TsdFrame]: regressor name -> grid-aligned block.
    """
    return {name: interpolate_to_grid(reg, tvec) for name, reg in continuous.items()}


def trial_constant_blocks(
    trials: pd.DataFrame, tvec: np.ndarray, columns: list[str] | None = None
) -> dict[str, nap.Tsd]:
    """Build a step regressor block for each trial-constant column.

    Args:
        trials (pd.DataFrame): trials table.
        tvec (np.ndarray): uniform model time grid.
        columns (list[str]): trial-constant columns; defaults to
            DEFAULT_TRIAL_CONSTANTS.

    Returns:
        dict[str, nap.Tsd]: column name -> step regressor on the grid.
    """
    columns = DEFAULT_TRIAL_CONSTANTS if columns is None else columns
    return {col: make_trial_constant(trials, col, tvec) for col in columns}


def build_design_matrix(
    blocks: dict[str, nap.Tsd | nap.TsdFrame],
) -> tuple[np.ndarray, dict[str, slice]]:
    """Concatenate named blocks; return the matrix and a name -> slice map.

    Block insertion order sets column order; the returned slices let callers
    retrieve any block's coefficients by name.

    Args:
        blocks (dict[str, nap.Tsd | nap.TsdFrame]): name -> regressor block.

    Returns:
        tuple[np.ndarray, dict[str, slice]]: design matrix and column spans.
    """
    matrix, slices, start = [], {}, 0
    for name, block in blocks.items():
        values = block.values
        values = values[:, None] if values.ndim == 1 else values
        slices[name] = slice(start, start + values.shape[1])
        matrix.append(values)
        start += values.shape[1]
    return np.concatenate(matrix, axis=1), slices


# --- fit ---


def fit_encoding_model(
    design: np.ndarray,
    target: nap.Tsd,
    slices: dict[str, slice],
    alpha: float = 1.0,
    label: str = "",
) -> EncodingFit:
    """Fit ridge regression of `target` on a prebuilt `design` matrix.

    Builds nothing — see `design_lagged` / `design_cosine` + the block helpers
    + `build_design_matrix` for assembling `design` and `slices`. Rows with
    NaNs (interpolation edges / missing support) are dropped before fitting.

    Args:
        design (np.ndarray): design matrix, rows aligned to `target`'s grid.
        target (nap.Tsd): measured signal on the model grid (sets the time axis).
        slices (dict[str, slice]): block name -> column span in `design`.
        alpha (float): ridge regularisation strength.
        label (str): session label for plots.

    Returns:
        EncodingFit: fitted model, prediction, R^2 and per-name coefficients.
    """
    tvec = target.times()
    y = target.values[:, None]

    # drop rows with NaNs from interpolation edges or missing support
    valid = ~np.isnan(design).any(axis=1) & ~np.isnan(y).any(axis=1)
    model = Ridge(alpha=alpha).fit(design[valid], y[valid])

    coefficients = model.coef_.T
    prediction = design[valid] @ coefficients + model.intercept_
    return EncodingFit(
        tvec=tvec,
        valid=valid,
        design=design[valid],
        target=y[valid],
        prediction=prediction,
        coefficients=coefficients,
        intercept=model.intercept_,
        slices=slices,
        r2=r2_score(y[valid], prediction),
        alpha=alpha,
        label=label,
    )


# --- inspection ---


def get_kernel(fit: EncodingFit, name: str) -> np.ndarray:
    """Return the fitted coefficients for one block, by name.

    For the lagged basis these coefficients are the kernel itself (one value
    per lag); for other bases they are basis weights, not the kernel.
    """
    return fit.coefficients[fit.slices[name]].flatten()


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


def plot_kernels(fit: EncodingFit, names: list[str], lags: np.ndarray) -> plt.Figure:
    """Plot the fitted lagged kernel for each named event block.

    Assumes the lagged basis: a block's coefficients are the kernel itself.

    Args:
        fit (EncodingFit): a fitted encoding model.
        names (list[str]): event block names to plot (keys in `fit.slices`).
        lags (np.ndarray): sample lags used to build the kernels.

    Returns:
        plt.Figure: the kernel figure.
    """
    fig, axes = plt.subplots(ncols=len(names), sharey=True, figsize=[3 * len(names), 3])
    # convert sample lags to seconds for the x axis
    lag_seconds = lags * (fit.tvec[1] - fit.tvec[0])
    for ax, name in zip(np.atleast_1d(axes), names):
        ax.plot(lag_seconds, get_kernel(fit, name))
        ax.set_title(name, fontsize="small")
        ax.axhline(0, linestyle=":", color="k", lw=1)
        ax.axvline(0, linestyle=":", color="k", lw=1)
        ax.set_xlabel("time (s)")
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


# --- per-regressor contribution (leave-one-regressor-out) ---


def _cv_r_squared(
    design: np.ndarray, target: np.ndarray, alpha: float, cv: int
) -> float:
    """Pooled out-of-fold R² of a ridge fit (KFold, no shuffle).

    Args:
        design (np.ndarray): design matrix.
        target (np.ndarray): target signal, shape (n, 1).
        alpha (float): ridge regularisation strength.
        cv (int): number of KFold splits.

    Returns:
        float: R² of the pooled out-of-fold predictions.
    """
    predictions = cross_val_predict(Ridge(alpha=alpha), design, target, cv=KFold(cv))
    return r2_score(target, predictions)


def delta_r_squared(fit: EncodingFit, cv: int | None = None) -> pd.Series:
    """Leave-one-regressor-out drop in R² for each named block.

    For each block in `fit.slices`, all of its columns are removed together, the
    reduced ridge model is refit (same alpha) and ΔR² = full R² − reduced R².

    Args:
        fit (EncodingFit): a fitted encoding model.
        cv (int): if None, score in-sample (matches `fit.r2`); if an int, score
            the pooled out-of-fold R² over that many KFold splits.

    Returns:
        pd.Series: ΔR² indexed by block name, sorted descending.
    """
    if cv is None:
        full_r2 = fit.r2
    else:
        full_r2 = _cv_r_squared(fit.design, fit.target, fit.alpha, cv)
    deltas = {}
    for name, span in fit.slices.items():
        # drop the whole block (all its columns), keep the rest
        keep = np.ones(fit.design.shape[1], dtype=bool)
        keep[span] = False
        reduced = fit.design[:, keep]
        if cv is None:
            model = Ridge(alpha=fit.alpha).fit(reduced, fit.target)
            prediction = reduced @ model.coef_.T + model.intercept_
            reduced_r2 = r2_score(fit.target, prediction)
        else:
            reduced_r2 = _cv_r_squared(reduced, fit.target, fit.alpha, cv)
        deltas[name] = full_r2 - reduced_r2
    return pd.Series(deltas, name="delta_r2").sort_values(ascending=False)


def plot_delta_r_squared(deltas: pd.Series, axes: plt.Axes = None) -> plt.Axes:
    """Bar chart of per-regressor ΔR² (largest contribution at top)."""
    if axes is None:
        _, axes = plt.subplots()
    # reverse so the largest drop sits at the top of the horizontal bars
    axes.barh(deltas.index[::-1], deltas.values[::-1])
    axes.axvline(0, linestyle=":", color="k", lw=1)
    axes.set_xlabel("ΔR² (drop when left out)")
    axes.figure.tight_layout()
    return axes
