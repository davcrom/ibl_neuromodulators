import re
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from tqdm import tqdm

from iblnm.config import (
    TARGET_FS,
    POSE_FS,
    LIKELIHOOD_THRESHOLD,
    MOVEMENT_RESPONSE_WINDOW,
    BASELINE_WINDOW,
    CROSSCORR_FS,
    CROSSCORR_LAG_WINDOW,
    MOVEMENT_PREDICTORS,
)
from iblnm.util import get_contrast_coding


def get_responses(photometry, events, t0=-1.0, t1=1.0):
    """Extract peri-event responses from a photometry signal.

    Parameters
    ----------
    photometry : pd.Series
        Signal with time index.
    events : 1D array
        Alignment event times.
    t0 : float
        Window start relative to event (seconds).
    t1 : float or 1D array
        Window end. If float, fixed for all trials.
        If array, per-trial endpoint as absolute times.
        NaN values in array → no masking (full window).

    Returns
    -------
    responses : 2D array, shape (n_trials, n_samples)
    tpts : 1D array, shape (n_samples,)
    """
    times = photometry.index.to_numpy()
    values = photometry.values
    fs = round(1 / np.median(np.diff(times)))

    # Determine effective t1 for window size
    variable_t1 = isinstance(t1, np.ndarray)
    if variable_t1:
        t1_relative = t1 - events
        if np.all(np.isnan(t1_relative)):
            t1_max = abs(t0)
        else:
            t1_max = np.nanmax(t1_relative)
    else:
        t1_max = t1

    n_trials = len(events)
    samples = np.arange(round(t0 * fs), round(t1_max * fs))
    tpts = samples / fs
    responses = np.full((n_trials, len(samples)), np.nan)

    valid_events = (events + t0 >= times.min()) & (events + t1_max <= times.max())
    valid_event_times = events[valid_events]
    event_idx = np.searchsorted(times, valid_event_times)

    response_idx = event_idx[:, None] + samples[None, :]
    response_idx = np.clip(response_idx, 0, len(values) - 1)
    responses[valid_events] = values[response_idx]

    # Variable endpoint masking
    if variable_t1:
        valid_t1 = valid_events & ~np.isnan(t1_relative)
        if valid_t1.any():
            mask = tpts[None, :] > t1_relative[valid_t1, None]
            responses[valid_t1] = np.where(mask, np.nan, responses[valid_t1])

    return responses, tpts


def compute_bleaching_tau(signal: pd.Series) -> float:
    """Fit exponential decay to a photometry signal and return the time constant τ.

    Note: iblphotometry.metrics.bleaching_tau has a bug (swapped argument order
    in the fit call). This function calls Regression.fit(t, y) correctly.
    """
    from iblphotometry.processing import Regression, ExponDecay
    reg = Regression(model=ExponDecay())
    reg.fit(signal.index.values, signal.values)
    return float(reg.popt[1])


def compute_iso_correlation(signal: pd.Series, reference: pd.Series,
                             regression_method: str = 'mse') -> float:
    """Compute R² of linear regression of reference onto signal.

    Parameters
    ----------
    signal : pd.Series
        Bleach-corrected GCaMP signal.
    reference : pd.Series
        Bleach-corrected isosbestic reference.
    regression_method : str
        Passed to iblphotometry Regression (default 'mse').
    """
    from iblphotometry.processing import Regression, LinearModel
    reg = Regression(model=LinearModel(), method=regression_method)
    reg.fit(reference.values, signal.values)
    predicted = reg.model.eq(reference.values, *reg.popt)
    ss_res = np.sum((signal.values - predicted) ** 2)
    ss_tot = np.sum((signal.values - np.mean(signal.values)) ** 2)
    return float(1 - ss_res / ss_tot)


def resample_signal(signal, target_fs=TARGET_FS):
    """Resample a photometry signal to a uniform grid using PCHIP interpolation."""
    from scipy.interpolate import PchipInterpolator
    times = signal.index.values
    t_uniform = np.arange(times[0], times[-1], 1 / target_fs)
    interp = PchipInterpolator(times, signal.values)
    return pd.Series(interp(t_uniform), index=t_uniform)


def resample_pose(pose: pd.DataFrame, times: np.ndarray,
                  fs: float = POSE_FS) -> tuple[pd.DataFrame, np.ndarray]:
    """Resample LightningPose columns onto a common uniform 1/fs grid.

    Each ``{part}_x``/``_y``/``_likelihood`` column is finite (LP always emits a
    position and a likelihood), so PCHIP interpolation is clean — resample the
    raw columns here, before any likelihood gating, so speeds computed downstream
    share a session-independent time step. Returns the resampled DataFrame and
    its uniform time axis.
    """
    resampled = {col: resample_signal(pd.Series(pose[col].values, index=times), fs)
                 for col in pose.columns}
    df = pd.DataFrame(resampled)
    return df, df.index.values


def compute_response_magnitude(response, tpts, window):
    """Mean response within a time window.

    Parameters
    ----------
    response : np.ndarray
        Shape (n_samples,) or (n_trials, n_samples).
    tpts : np.ndarray
        Shape (n_samples,). Time points corresponding to the last axis.
    window : tuple of float
        (start, end) in seconds.

    Returns
    -------
    float or np.ndarray
        Scalar for 1D input, shape (n_trials,) for 2D input.
    """
    i0 = np.searchsorted(tpts, window[0])
    i1 = np.searchsorted(tpts, window[1])
    return np.nanmean(response[..., i0:i1], axis=-1)


def keypoint_speed(x, y, likelihood, threshold=LIKELIHOOD_THRESHOLD):
    """Frame-to-frame keypoint speed, gated by tracking likelihood.

    Parameters
    ----------
    x, y : 1D array
        Keypoint pixel coordinates per frame.
    likelihood : 1D array
        Per-frame tracking confidence, same length as x and y.
    threshold : float
        Frames with likelihood below this are set to NaN.

    Returns
    -------
    speed : 1D array
        Directionless frame-to-frame displacement magnitude, same length as the
        input. The first frame is NaN (no preceding frame to difference).
    """
    speed = np.concatenate(([np.nan], np.hypot(np.diff(x), np.diff(y))))
    speed[likelihood < threshold] = np.nan
    return speed


def _guarded_window_mean(trace, tpts, window, min_valid):
    """NaN-aware mean over a time window, or NaN if too few valid samples."""
    i0 = np.searchsorted(tpts, window[0])
    i1 = np.searchsorted(tpts, window[1])
    block = trace[..., i0:i1]
    if np.isfinite(block).sum() < min_valid:
        return np.nan
    return np.nanmean(block)


def movement_delta(response_trace, baseline_trace, tpts,
                   response_window=MOVEMENT_RESPONSE_WINDOW,
                   baseline_window=BASELINE_WINDOW, min_valid=1):
    """Post-minus-pre movement scalar from two separately event-locked traces.

    Unlike :func:`event_locked_scalar`, the response and baseline are drawn from
    different trace matrices that share a time axis: ``response_trace`` is locked
    to the measure's own event, ``baseline_trace`` to stimulus onset. Returns
    ``mean(response_trace over response_window) - mean(baseline_trace over
    baseline_window)``, NaN-aware; NaN if either window has fewer than
    ``min_valid`` valid samples.
    """
    response = _guarded_window_mean(response_trace, tpts, response_window, min_valid)
    baseline = _guarded_window_mean(baseline_trace, tpts, baseline_window, min_valid)
    return response - baseline


def event_locked_scalar(trace, tpts, response_window=MOVEMENT_RESPONSE_WINDOW,
                        baseline_window=BASELINE_WINDOW, min_valid=1):
    """Trial-averaged post-minus-pre scalar from an event-locked trace matrix.

    Parameters
    ----------
    trace : 2D array, shape (n_trials, n_samples)
        Event-locked trace as returned by ``get_responses``.
    tpts : 1D array, shape (n_samples,)
        Time axis matching the last dimension of ``trace``.
    response_window, baseline_window : tuple of float
        (start, end) seconds for the post-event and pre-event windows.
    min_valid : int
        Minimum non-NaN samples a window must contain across all trials; a
        window with fewer is NaN, which propagates to the scalar.

    Returns
    -------
    float
        Mean over ``response_window`` minus mean over ``baseline_window``,
        pooled over trials and time, NaN-aware. NaN if either window has fewer
        than ``min_valid`` valid samples.
    """
    response = _guarded_window_mean(trace, tpts, response_window, min_valid)
    baseline = _guarded_window_mean(trace, tpts, baseline_window, min_valid)
    return response - baseline


def movement_trace(pose, keypoints, reduction, threshold=LIKELIHOOD_THRESHOLD):
    """Per-frame movement trace for one bodypart from LightningPose columns.

    Parameters
    ----------
    pose : pd.DataFrame
        LightningPose output with columns ``{part}_x``, ``{part}_y``,
        ``{part}_likelihood`` for each keypoint.
    keypoints : list of str
        Keypoint name(s) to reduce into one trace.
    reduction : {'speed', 'sum_speed', 'max_likelihood'}
        - ``speed``/``sum_speed``: NaN-aware sum of per-keypoint
          likelihood-gated speeds (a frame is NaN only where every keypoint is
          NaN).
        - ``max_likelihood``: per-frame max of the keypoints' tracking
          likelihoods, ungated.
    threshold : float
        Likelihood gate for the speed reductions (ignored for ``max_likelihood``).

    Returns
    -------
    1D array
        Per-frame trace, length matching ``pose``.
    """
    if reduction == 'max_likelihood':
        likelihoods = np.column_stack([pose[f'{k}_likelihood'].values
                                       for k in keypoints])
        return np.nanmax(likelihoods, axis=1)
    speeds = np.column_stack([
        keypoint_speed(pose[f'{k}_x'].values, pose[f'{k}_y'].values,
                       pose[f'{k}_likelihood'].values, threshold)
        for k in keypoints])
    trace = np.nansum(speeds, axis=1)
    trace[np.isnan(speeds).all(axis=1)] = np.nan
    return trace


def normalized_crosscorr(a, b, fs, lag_window=CROSSCORR_LAG_WINDOW):
    """Normalized cross-correlation of two equal-rate signals over a lag window.

    Both signals are z-scored before correlating and the result is divided by
    the sample count, so the peak is unit-scaled (≈ 1 for identical signals)
    and comparable across sessions.

    Parameters
    ----------
    a, b : 1D array
        Signals sampled on a common rate ``fs``, same length.
    fs : float
        Common sampling rate (Hz).
    lag_window : float
        Half-width of the returned lag range (seconds).

    Returns
    -------
    cc : 1D array
        Normalized cross-correlation over lags in ``[-lag_window, +lag_window]``.
    lags : 1D array
        Lag axis (seconds), same shape as ``cc``.
    peak_lag : float
        Lag of the maximum. A positive lag means ``a`` leads ``b`` (``a``'s
        features occur earlier in time than the matching features of ``b``).
    """
    from scipy.signal import correlate, correlation_lags
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(a)
    a_z = (a - a.mean()) / a.std()
    b_z = (b - b.mean()) / b.std()
    cc = correlate(b_z, a_z, mode='full') / n
    lags = correlation_lags(len(b), n, mode='full') / fs
    keep = np.abs(lags) <= lag_window
    cc, lags = cc[keep], lags[keep]
    return cc, lags, lags[np.argmax(cc)]


def per_third_crosscorr(paw_speed, paw_times, wheel_speed, wheel_times,
                        fs=CROSSCORR_FS, lag_window=CROSSCORR_LAG_WINDOW):
    """Per-session-third paw–wheel cross-correlation and timing drift.

    Resamples both traces onto a shared grid at ``fs``, splits the session
    timeline into thirds (early/mid/late by time), and computes the normalized
    cross-correlation of paw vs wheel speed over ±``lag_window`` seconds within
    each third. The cross-correlation is continuous: still periods self-gate
    (wheel speed ~0 contributes ~0), so a third with little movement simply
    yields a flat curve — itself a diagnostic — rather than being masked out.

    Parameters
    ----------
    paw_speed, paw_times : 1D array
        Paw-sum speed and its frame times (camera clock).
    wheel_speed, wheel_times : 1D array
        Wheel speed and its sample times (session clock).
    fs : float
        Common resample rate (Hz).
    lag_window : float
        Half-width of the lag range (seconds).

    Returns
    -------
    functions : 2D array, shape (3, n_lags)
        Normalized cross-correlation per third.
    lags : 1D array, shape (n_lags,)
        Lag axis (seconds). Positive lag means paw leads wheel.
    peak_lags : 1D array, shape (3,)
        Peak lag per third (seconds).
    drift : float
        ``peak_lags[-1] - peak_lags[0]`` (late minus early).
    """
    grid = np.arange(max(paw_times[0], wheel_times[0]),
                     min(paw_times[-1], wheel_times[-1]), 1 / fs)
    paw = np.interp(grid, paw_times, paw_speed)
    wheel = np.interp(grid, wheel_times, wheel_speed)

    edges = np.linspace(grid[0], grid[-1], 4)
    lag_samples = round(lag_window * fs)
    lags = np.arange(-lag_samples, lag_samples + 1) / fs
    functions = np.full((3, lags.size), np.nan)
    peak_lags = np.full(3, np.nan)

    for third, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        in_third = (grid >= lo) & (grid < hi)
        functions[third], _, peak_lags[third] = normalized_crosscorr(
            paw[in_third], wheel[in_third], fs, lag_window)

    drift = peak_lags[-1] - peak_lags[0]
    return functions, lags, peak_lags, drift


def normalize_responses(responses, tpts, bwin=(-0.1, 0), divide=True):
    i0, i1 = tpts.searchsorted(bwin)
    bvals = responses[:, i0:i1].mean(axis=1, keepdims=True)
    resp_norm = responses - bvals
    if divide:
        resp_norm = resp_norm / bvals
    return resp_norm


# =============================================================================
# Response Vector Analysis
# =============================================================================


def split_features_by_event(response_matrix):
    """Split a response feature matrix into per-event sub-matrices.

    Column names follow the pattern ``{event_stem}_c{contrast}_{side}_{fb}``.
    The event stem is everything before the first ``_c`` token.

    Parameters
    ----------
    response_matrix : pd.DataFrame
        Recordings × features, as returned by ``get_response_features``.

    Returns
    -------
    dict[str, pd.DataFrame]
        Event stem → sub-matrix with only that event's columns.
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for col in response_matrix.columns:
        stem = col.split('_c')[0]
        groups[stem].append(col)
    return {stem: response_matrix[cols] for stem, cols in groups.items()}


def cosine_similarity_matrix(response_matrix):
    """Pairwise cosine similarity, dropping rows with any NaN.

    Parameters
    ----------
    response_matrix : pd.DataFrame
        Rows = recordings, columns = features.

    Returns
    -------
    pd.DataFrame
        Symmetric similarity matrix.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    clean = response_matrix.dropna()
    sim = cosine_similarity(clean.values)
    return pd.DataFrame(sim, index=clean.index, columns=clean.index)


def within_between_similarity(sim_matrix, labels):
    """Partition pairwise similarities into within-group and between-group.

    Parameters
    ----------
    sim_matrix : pd.DataFrame
        Symmetric similarity matrix from ``cosine_similarity_matrix``.
    labels : pd.Series
        Group label per recording, aligned to sim_matrix index.

    Returns
    -------
    pd.DataFrame
        Columns: group1, group2, similarity, comparison.
    """
    idx = sim_matrix.index
    rows = []
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            g1, g2 = labels[idx[i]], labels[idx[j]]
            comp = 'within' if g1 == g2 else 'between'
            rows.append({
                'group1': g1, 'group2': g2,
                'similarity': sim_matrix.iloc[i, j],
                'comparison': comp,
            })
    return pd.DataFrame(rows)


def mean_similarity_by_target(sim_matrix, labels, subjects=None):
    """Mean pairwise cosine similarity between targets.

    Parameters
    ----------
    sim_matrix : pd.DataFrame
        Symmetric pairwise similarity matrix.
    labels : pd.Series
        Target-NM label per recording, aligned to sim_matrix index.
    subjects : pd.Series, optional
        Subject label per recording. When provided, pairs from the same
        subject are excluded (leave-one-subject-out structure).

    Returns
    -------
    pd.DataFrame
        Square target × target matrix of mean pairwise similarities.
        Diagonal entries are within-target means (NaN if fewer than 2
        recordings for a target, or no cross-subject pairs when
        subjects is provided).
    """
    targets = sorted(labels.unique())
    result = pd.DataFrame(np.nan, index=targets, columns=targets)
    for i, t1 in enumerate(targets):
        mask1 = labels == t1
        idx1 = labels.index[mask1]
        for j, t2 in enumerate(targets):
            mask2 = labels == t2
            idx2 = labels.index[mask2]
            sub = sim_matrix.loc[idx1, idx2]
            if t1 == t2:
                # Within-target: upper triangle (exclude self-similarity)
                rows, cols = np.triu_indices(len(sub), k=1)
            else:
                rows, cols = np.indices(sub.shape)
                rows, cols = rows.ravel(), cols.ravel()
            # Exclude same-subject pairs when subjects provided
            if subjects is not None:
                subj1 = subjects.loc[idx1].values
                subj2 = subjects.loc[idx2].values
                cross = subj1[rows] != subj2[cols]
                rows, cols = rows[cross], cols[cross]
            vals = sub.values[rows, cols]
            result.iloc[i, j] = np.nanmean(vals) if len(vals) > 0 else np.nan
    return result


def decode_target_nm(response_matrix, labels, subjects, normalize=True,
                     C=None, Cs=None):
    """Decode target-NM from response vectors using L1 logistic regression.

    Parameters
    ----------
    response_matrix : pd.DataFrame
        Rows = recordings, columns = features. NaN rows are dropped.
    labels : pd.Series
        Target-NM label per recording.
    subjects : pd.Series
        Subject identifier per recording, used as group for
        leave-one-subject-out CV.
    normalize : bool
        If True (default), z-score features within each CV fold (fit on
        train, transform both train and test).
    C : float, optional
        Fixed regularization strength. Skips tuning when provided.
    Cs : list of float, optional
        Grid of C values to search via inner CV. Defaults to
        ``np.logspace(-3, 3, 10)``.

    Returns
    -------
    dict
        Keys: 'accuracy', 'confusion', 'coefficients', 'predictions',
        'best_C', 'n_valid'.
    """
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    if Cs is None:
        Cs = np.logspace(-3, 3, 10).tolist()

    clean = response_matrix.dropna()
    y = labels.loc[clean.index]
    subj = subjects.loc[clean.index]
    groups = subj

    if y.nunique() < 2:
        raise ValueError("Need at least 2 target-NM classes for decoding")

    X = clean.values
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    logo = LeaveOneGroupOut()
    all_preds = np.full(len(y_enc), -1, dtype=int)
    fold_Cs = []

    fixed_C = C is not None

    for train_idx, test_idx in logo.split(X, y_enc, groups):
        # Skip folds where train set doesn't contain all classes
        if len(np.unique(y_enc[train_idx])) < len(le.classes_):
            all_preds[test_idx] = -1
            continue

        X_train, X_test = X[train_idx], X[test_idx]
        if normalize:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        if fixed_C:
            clf = LogisticRegression(
                penalty='l1', solver='saga', max_iter=5000,
                class_weight='balanced', C=C,
            )
        else:
            # Inner CV folds limited by smallest class in training set
            min_class_n = min(np.bincount(y_enc[train_idx]))
            inner_cv = max(2, min(3, min_class_n))
            clf = LogisticRegressionCV(
                penalty='l1', solver='saga', max_iter=5000,
                class_weight='balanced', Cs=Cs, cv=inner_cv,
            )
        clf.fit(X_train, y_enc[train_idx])
        all_preds[test_idx] = clf.predict(X_test)

        if fixed_C:
            fold_Cs.append(C)
        else:
            fold_Cs.append(float(clf.C_[0]))

    # Exclude unpredicted samples from accuracy
    valid = all_preds >= 0
    acc = accuracy_score(y_enc[valid], all_preds[valid]) if valid.any() else 0.0
    bal_acc = balanced_accuracy_score(y_enc[valid], all_preds[valid]) if valid.any() else 0.0

    class_names = le.classes_
    cm = pd.DataFrame(
        confusion_matrix(y_enc[valid], all_preds[valid], labels=range(len(class_names))),
        index=class_names, columns=class_names,
    )

    # Per-class accuracy (recall): diagonal / row sum
    per_class = pd.Series(
        {name: cm.loc[name, name] / cm.loc[name].sum() if cm.loc[name].sum() > 0 else 0.0
         for name in class_names},
        name='recall',
    )

    # Refit on all data with best C for interpretable coefficients
    best_C = float(np.median(fold_Cs)) if fold_Cs else 1.0
    if normalize:
        scaler = StandardScaler()
        X_full = scaler.fit_transform(X)
    else:
        X_full = X
    clf_full = LogisticRegression(
        penalty='l1', solver='saga', max_iter=5000,
        class_weight='balanced', C=best_C,
    )
    clf_full.fit(X_full, y_enc)

    # Binary classification: sklearn returns (1, n_features); expand to (2, n_features)
    full_coef = clf_full.coef_
    if full_coef.shape[0] == 1 and len(class_names) == 2:
        full_coef = np.vstack([full_coef, -full_coef])

    coefs = pd.DataFrame(
        full_coef, index=class_names, columns=clean.columns,
    )

    preds_dict = {
        name: clean.index.get_level_values(name)
        for name in clean.index.names
    }
    preds_dict['true'] = y.values
    preds_dict['predicted'] = le.inverse_transform(
        np.clip(all_preds, 0, len(class_names) - 1),
    )
    preds_df = pd.DataFrame(preds_dict)

    return {
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'per_class_accuracy': per_class,
        'confusion': cm,
        'coefficients': coefs,
        'predictions': preds_df,
        'best_C': best_C,
        'n_valid': int(valid.sum()),
    }


class TargetNMDecoder:
    """Decode target-NM identity from response feature vectors.

    Wraps L1 logistic regression with leave-one-subject-out CV,
    plus drop-one-feature importance analysis.

    Parameters
    ----------
    response_matrix : pd.DataFrame
        Rows = recordings, columns = features.
    labels : pd.Series
        Target-NM label per recording.
    subjects : pd.Series
        Subject identifier per recording.
    normalize : bool
        Z-score features within each CV fold.
    """

    def __init__(self, response_matrix, labels, subjects, normalize=True):
        self.response_matrix = response_matrix
        self.labels = labels
        self.subjects = subjects
        self.normalize = normalize

    def fit(self):
        """Fit decoder via leave-one-subject-out CV with C tuning.

        Stores results as attributes: accuracy, balanced_accuracy,
        per_class_accuracy, confusion, coefficients, predictions, best_C_.
        """
        result = decode_target_nm(
            self.response_matrix, self.labels, self.subjects,
            normalize=self.normalize,
        )
        self.accuracy = result['accuracy']
        self.balanced_accuracy = result['balanced_accuracy']
        self.per_class_accuracy = result['per_class_accuracy']
        self.confusion = result['confusion']
        self.coefficients = result['coefficients']
        self.predictions = result['predictions']
        self.best_C_ = result['best_C']
        return self

    def unique_contribution(self):
        """Compute each feature's unique contribution to decoding accuracy.

        Uses the optimal C from ``fit()`` so reduced models are comparable.

        Returns
        -------
        pd.DataFrame
            Columns: feature, full_accuracy, reduced_accuracy, delta.
        """
        self.contributions = feature_unique_contribution(
            self.response_matrix, self.labels, self.subjects,
            normalize=self.normalize, C=self.best_C_,
        )
        return self.contributions


def _fit_full_data(X, y_enc, C, normalize):
    """Fit L1 logistic regression on all data and return training accuracy."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import balanced_accuracy_score

    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    clf = LogisticRegression(
        penalty='l1', solver='saga', max_iter=5000,
        class_weight='balanced', C=C,
    )
    clf.fit(X, y_enc)
    preds = clf.predict(X)
    return balanced_accuracy_score(y_enc, preds)


def permutation_pvalue(observed: float, null, alternative: str = 'two-sided') -> float:
    """Add-one-corrected permutation p-value for a scalar against a null.

    Parameters
    ----------
    observed : float
        Observed test statistic.
    null : 1-D array-like
        Null distribution of the statistic, length ``n_iter``.
    alternative : {'two-sided', 'greater', 'less'}
        Tail of the test. ``'greater'`` tests ``observed`` in the upper tail,
        ``'less'`` in the lower tail, ``'two-sided'`` doubles the smaller tail.

    Returns
    -------
    float
        p-value in ``(0, 1]``, floored at ``1 / (n_iter + 1)`` by the add-one
        correction (matches the ``fit_cca`` convention).
    """
    null = np.asarray(null)
    n = len(null)
    p_greater = (np.sum(null >= observed) + 1) / (n + 1)
    p_less = (np.sum(null <= observed) + 1) / (n + 1)
    if alternative == 'greater':
        return p_greater
    if alternative == 'less':
        return p_less
    if alternative == 'two-sided':
        return min(1.0, 2 * min(p_greater, p_less))
    raise ValueError(f"Unrecognized alternative: {alternative!r}")


# =============================================================================
# Canonical Correlation Analysis
# =============================================================================


class CCAResult:
    """Container for a CCA fit result.

    Attributes
    ----------
    x_weights : pd.DataFrame
        (K, n_components) neural feature weights.
    y_weights : pd.DataFrame
        (P, n_components) behavioral parameter weights.
    x_scores : np.ndarray
        (n, n_components) neural canonical variates.
    y_scores : np.ndarray
        (n, n_components) behavioral canonical variates.
    correlations : np.ndarray
        (n_components,) canonical correlations.
    p_values : np.ndarray or None
        (n_components,) from permutation test, None if no permutation.
    n_recordings : int
    n_permutations : int
    alpha : float or None
        Selected regularization strength (sparse CCA only).
    l1_ratio : float or None
        L1/L2 mixing ratio (sparse CCA only). 0 = ridge, 1 = lasso.
    x_variance_explained : np.ndarray or None
        (n_components,) variance extracted from the neural block per variate
        (mean squared loading). None if not computed.
    y_variance_explained : np.ndarray or None
        (n_components,) variance extracted from the behavioral block per variate.
    """

    def __init__(self, x_weights, y_weights, x_scores, y_scores,
                 correlations, p_values, n_recordings, n_permutations,
                 alpha=None, l1_ratio=None,
                 x_variance_explained=None, y_variance_explained=None):
        self.x_weights = x_weights
        self.y_weights = y_weights
        self.x_scores = x_scores
        self.y_scores = y_scores
        self.correlations = correlations
        self.p_values = p_values
        self.n_recordings = n_recordings
        self.n_permutations = n_permutations
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.x_variance_explained = x_variance_explained
        self.y_variance_explained = y_variance_explained


def cca_variance_extracted(data, scores):
    """Variance extracted (adequacy) per canonical variate.

    For each variate, the mean over features of the squared loading — the
    squared Pearson correlation between each original feature and the variate.
    This is the fraction of variance in ``data``'s own feature block that the
    variate captures.

    Parameters
    ----------
    data : np.ndarray
        (n, K) feature block (neural or behavioral).
    scores : np.ndarray
        (n, n_components) canonical variates for that block.

    Returns
    -------
    np.ndarray
        (n_components,) variance extracted per variate, in [0, 1]. Constant
        features (zero variance) contribute a loading of 0.
    """
    data = np.asarray(data, dtype=float)
    scores = np.asarray(scores, dtype=float)
    n = data.shape[0]
    dc = data - data.mean(axis=0, keepdims=True)
    sc = scores - scores.mean(axis=0, keepdims=True)
    cov = (dc.T @ sc) / n                          # (K, n_components)
    denom = np.outer(dc.std(axis=0), sc.std(axis=0))
    with np.errstate(divide='ignore', invalid='ignore'):
        loadings = np.where(denom > 0, cov / denom, 0.0)
    return (loadings ** 2).mean(axis=0)


def fit_cca(X, Y, n_components=None, n_permutations=0, session_labels=None,
            seed=None, scale=True):
    """Fit canonical correlation analysis between neural and behavioral features.

    Parameters
    ----------
    X : pd.DataFrame
        (n_recordings, K) neural features. NaN rows dropped.
    Y : pd.DataFrame
        (n_recordings, P) behavioral features, aligned to X.
    n_components : int or None
        Number of canonical variates. Default: min(K, P, n).
    n_permutations : int
        If > 0, run permutation test.
    session_labels : pd.Series or None
        Session identifier per row. When provided, permutations shuffle
        at the session level (all rows of a session move together).
    seed : int or None
        RNG seed for reproducibility.
    scale : bool
        If True (default), standardize X and Y before fitting. Set to False
        when passing pre-standardized data.

    Returns
    -------
    CCAResult
    """
    from sklearn.cross_decomposition import CCA
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import pearsonr

    # Drop rows with NaN in either X or Y
    valid = X.notna().all(axis=1) & Y.notna().all(axis=1)
    X_clean = X.loc[valid].copy()
    Y_clean = Y.loc[valid].copy()

    n = len(X_clean)
    if n < 3:
        raise ValueError(f"Need at least 3 recordings, got {n}")

    # Drop constant Y columns
    y_std = Y_clean.std()
    varying = y_std[y_std > 0].index
    if len(varying) == 0:
        raise ValueError("Y has no variance — all columns are constant")
    Y_clean = Y_clean[varying]

    k = X_clean.shape[1]
    p = Y_clean.shape[1]
    max_components = min(k, p, n)
    if n_components is None:
        n_components = max_components
    else:
        n_components = min(n_components, max_components)

    # Standardize
    if scale:
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        X_z = x_scaler.fit_transform(X_clean.values)
        Y_z = y_scaler.fit_transform(Y_clean.values)
    else:
        X_z = X_clean.values
        Y_z = Y_clean.values

    # Fit CCA
    cca = CCA(n_components=n_components, max_iter=1000)
    x_scores, y_scores = cca.fit_transform(X_z, Y_z)

    # Canonical correlations
    correlations = np.array([
        pearsonr(x_scores[:, i], y_scores[:, i])[0]
        for i in range(n_components)
    ])

    # Weights as DataFrames preserving feature names
    comp_names = [f'CC{i+1}' for i in range(n_components)]
    x_weights = pd.DataFrame(cca.x_weights_, index=X_clean.columns,
                              columns=comp_names)
    y_weights = pd.DataFrame(cca.y_weights_, index=Y_clean.columns,
                              columns=comp_names)

    # Permutation test
    p_values = None
    if n_permutations > 0:
        rng = np.random.default_rng(seed)
        perm_corrs = np.zeros((n_permutations, n_components))

        if session_labels is not None:
            session_labels_clean = session_labels.loc[valid].values
            unique_sessions = np.unique(session_labels_clean)
        else:
            session_labels_clean = None

        for perm_i in tqdm(range(n_permutations), desc='CCA permutations'):
            if session_labels_clean is not None:
                # Shuffle at session level
                shuffled_sessions = rng.permutation(unique_sessions)
                session_map = dict(zip(unique_sessions, shuffled_sessions))
                new_labels = np.array([session_map[s] for s in session_labels_clean])
                # Reorder Y rows according to shuffled session mapping
                sort_idx = np.argsort(session_labels_clean)
                new_sort_idx = np.argsort(new_labels)
                Y_perm = np.empty_like(Y_z)
                Y_perm[sort_idx] = Y_z[new_sort_idx]
            else:
                perm_idx = rng.permutation(n)
                Y_perm = Y_z[perm_idx]

            cca_perm = CCA(n_components=n_components, max_iter=1000)
            try:
                xs_perm, ys_perm = cca_perm.fit_transform(X_z, Y_perm)
                for j in range(n_components):
                    perm_corrs[perm_i, j] = pearsonr(xs_perm[:, j], ys_perm[:, j])[0]
            except Exception:
                perm_corrs[perm_i, :] = 0.0

        p_values = np.array([
            (np.sum(perm_corrs[:, j] >= correlations[j]) + 1) / (n_permutations + 1)
            for j in range(n_components)
        ])

    return CCAResult(
        x_weights=x_weights,
        y_weights=y_weights,
        x_scores=x_scores,
        y_scores=y_scores,
        correlations=correlations,
        p_values=p_values,
        n_recordings=n,
        n_permutations=n_permutations,
        x_variance_explained=cca_variance_extracted(X_z, x_scores),
        y_variance_explained=cca_variance_extracted(Y_z, y_scores),
    )


def fit_sparse_cca(X, Y, n_components=None, n_permutations=0,
                   session_labels=None, seed=None, scale=True,
                   alpha=0.01, l1_ratio=0.0, unit_norm=True):
    """Fit sparse CCA between neural and behavioral features.

    Uses cca-zoo's ElasticCCA with elastic net regularization. The ``alpha``
    parameter controls regularization strength (higher = more shrinkage).
    The ``l1_ratio`` controls the L1/L2 mix (0 = pure ridge, 1 = pure lasso).

    When ``alpha`` or ``l1_ratio`` is a list, performs grid search over all
    combinations on the full data and selects the pair with the highest
    canonical correlation.

    Parameters
    ----------
    X : pd.DataFrame
        (n_recordings, K) neural features. NaN rows dropped.
    Y : pd.DataFrame
        (n_recordings, P) behavioral features, aligned to X.
    n_components : int or None
        Number of canonical variates. Default: min(K, P, n).
    n_permutations : int
        If > 0, run permutation test.
    session_labels : pd.Series or None
        Session identifier per row. Used for session-level permutation
        shuffling.
    seed : int or None
        RNG seed for reproducibility.
    scale : bool
        If True (default), standardize X and Y before fitting.
    alpha : float or list[float]
        Regularization strength. When a list, grid-searched.
    l1_ratio : float or list[float]
        L1/L2 mixing ratio. 0 = ridge, 1 = lasso. When a list,
        grid-searched.
    unit_norm : bool
        If True (default), rescale weight vectors to unit L2 norm per
        component, matching sklearn CCA's convention.

    Returns
    -------
    CCAResult
    """
    try:
        from cca_zoo.linear import ElasticCCA
    except ImportError:
        raise ImportError(
            "cca-zoo is required for sparse CCA. "
            "Install with: uv pip install cca-zoo"
        )
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import pearsonr

    # Drop rows with NaN in either X or Y
    valid = X.notna().all(axis=1) & Y.notna().all(axis=1)
    X_clean = X.loc[valid].copy()
    Y_clean = Y.loc[valid].copy()

    n = len(X_clean)
    if n < 3:
        raise ValueError(f"Need at least 3 recordings, got {n}")

    # Drop constant Y columns
    y_std = Y_clean.std()
    varying = y_std[y_std > 0].index
    if len(varying) == 0:
        raise ValueError("Y has no variance — all columns are constant")
    Y_clean = Y_clean[varying]

    k = X_clean.shape[1]
    p = Y_clean.shape[1]
    max_components = min(k, p, n)
    if n_components is None:
        n_components = max_components
    else:
        n_components = min(n_components, max_components)

    # Standardize
    if scale:
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        X_z = x_scaler.fit_transform(X_clean.values)
        Y_z = y_scaler.fit_transform(Y_clean.values)
    else:
        X_z = X_clean.values
        Y_z = Y_clean.values

    if session_labels is not None:
        session_labels_clean = session_labels.loc[valid].values

    # Grid search over alpha × l1_ratio if lists provided
    alpha_grid = alpha if isinstance(alpha, list) else [alpha]
    l1_grid = l1_ratio if isinstance(l1_ratio, list) else [l1_ratio]

    if len(alpha_grid) * len(l1_grid) > 1:
        import itertools
        best_r, best_alpha, best_l1 = -np.inf, alpha_grid[0], l1_grid[0]
        best_ecca = None
        for a, l1 in tqdm(list(itertools.product(alpha_grid, l1_grid)),
                          desc='Grid search (alpha × l1_ratio)'):
            ecca_candidate = ElasticCCA(
                latent_dimensions=n_components,
                alpha=[a, a],
                l1_ratio=[l1, l1],
                center=False,
                random_state=seed,
            )
            try:
                xs, ys = ecca_candidate.fit_transform([X_z, Y_z])
                r = pearsonr(xs[:, 0], ys[:, 0])[0]
            except Exception:
                r = -np.inf
            if np.isfinite(r) and r > best_r:
                best_r = r
                best_alpha = a
                best_l1 = l1
                best_ecca = ecca_candidate
        alpha = best_alpha
        l1_ratio = best_l1
        ecca = best_ecca
        x_scores, y_scores = ecca.transform([X_z, Y_z])
    else:
        alpha = alpha_grid[0]
        l1_ratio = l1_grid[0]
        ecca = ElasticCCA(
            latent_dimensions=n_components,
            alpha=[alpha, alpha],
            l1_ratio=[l1_ratio, l1_ratio],
            center=False,
            random_state=seed,
        )
        x_scores, y_scores = ecca.fit_transform([X_z, Y_z])

    # Canonical correlations
    correlations = np.array([
        pearsonr(x_scores[:, i], y_scores[:, i])[0]
        for i in range(n_components)
    ])

    # Weights as DataFrames preserving feature names
    comp_names = [f'CC{i+1}' for i in range(n_components)]
    x_raw = ecca.weights_[0].copy()
    y_raw = ecca.weights_[1].copy()
    if unit_norm:
        for i in range(n_components):
            x_n = np.linalg.norm(x_raw[:, i])
            y_n = np.linalg.norm(y_raw[:, i])
            if x_n > 0:
                x_raw[:, i] /= x_n
            if y_n > 0:
                y_raw[:, i] /= y_n
    x_weights = pd.DataFrame(x_raw, index=X_clean.columns,
                              columns=comp_names)
    y_weights = pd.DataFrame(y_raw, index=Y_clean.columns,
                              columns=comp_names)

    # Permutation test
    p_values = None
    if n_permutations > 0:
        rng = np.random.default_rng(seed)
        perm_corrs = np.zeros((n_permutations, n_components))

        if session_labels is not None:
            unique_sessions = np.unique(session_labels_clean)
        else:
            session_labels_clean = None

        for perm_i in tqdm(range(n_permutations),
                           desc='Sparse CCA permutations'):
            if session_labels_clean is not None:
                shuffled_sessions = rng.permutation(unique_sessions)
                session_map = dict(zip(unique_sessions, shuffled_sessions))
                new_labels = np.array([session_map[s]
                                       for s in session_labels_clean])
                sort_idx = np.argsort(session_labels_clean)
                new_sort_idx = np.argsort(new_labels)
                Y_perm = np.empty_like(Y_z)
                Y_perm[sort_idx] = Y_z[new_sort_idx]
            else:
                perm_idx = rng.permutation(n)
                Y_perm = Y_z[perm_idx]

            ecca_perm = ElasticCCA(
                latent_dimensions=n_components,
                alpha=[alpha, alpha],
                l1_ratio=[l1_ratio, l1_ratio],
                center=False,
                random_state=seed,
            )
            try:
                xs_perm, ys_perm = ecca_perm.fit_transform([X_z, Y_perm])
                for j in range(n_components):
                    perm_corrs[perm_i, j] = pearsonr(
                        xs_perm[:, j], ys_perm[:, j])[0]
            except Exception:
                perm_corrs[perm_i, :] = 0.0

        p_values = np.array([
            (np.sum(perm_corrs[:, j] >= correlations[j]) + 1)
            / (n_permutations + 1)
            for j in range(n_components)
        ])

    return CCAResult(
        x_weights=x_weights,
        y_weights=y_weights,
        x_scores=x_scores,
        y_scores=y_scores,
        correlations=correlations,
        p_values=p_values,
        n_recordings=n,
        n_permutations=n_permutations,
        alpha=alpha,
        l1_ratio=l1_ratio,
        x_variance_explained=cca_variance_extracted(X_z, x_scores),
        y_variance_explained=cca_variance_extracted(Y_z, y_scores),
    )


def select_block_terms(columns, mains) -> list:
    """Select model-term columns belonging to one category block.

    A coefficient column name is a statsmodels term: a single factor (a main
    effect, e.g. ``'contrast'``) or factors joined by ``':'`` (an interaction,
    e.g. ``'contrast:side'``). A column belongs to the block when every one of
    its ``':'``-split factors is in ``mains``. This keeps the block's main
    effects and its within-block interactions while excluding the intercept and
    any interaction that crosses into another category. Variable-agnostic: the
    caller supplies the block's main effects.

    Parameters
    ----------
    columns : iterable of str
        Coefficient column (model term) names.
    mains : iterable of str
        The block's main-effect factor names.

    Returns
    -------
    list of str
        Block columns, in input order.
    """
    mains = set(mains)
    return [col for col in columns if set(col.split(':')) <= mains]


def cross_project_cca(X_z, Y_z, target_result):
    """Cross-project data through another cohort's CCA weights.

    Parameters
    ----------
    X_z : np.ndarray
        (n, K) standardized neural features for the source cohort.
    Y_z : np.ndarray
        (n, P) standardized behavioral features for the source cohort.
    target_result : CCAResult
        CCA fit from the target cohort whose weights are used.

    Returns
    -------
    float
        Pearson r between projected neural and behavioral CC1 scores.
    """
    from scipy.stats import pearsonr

    x_proj = X_z @ target_result.x_weights['CC1'].values
    y_proj = Y_z @ target_result.y_weights['CC1'].values
    r, _ = pearsonr(x_proj, y_proj)
    return r


def align_cca_signs(results, reference=None):
    """Align CCA weight signs across cohorts to a reference.

    CCA has sign indeterminacy: ``(w_x, w_y)`` and ``(-w_x, -w_y)`` yield
    the same canonical correlation. This function flips both weight vectors
    (and scores) for each cohort so that the neural CC1 weights point in the
    same direction as the reference cohort.

    Parameters
    ----------
    results : dict[str, CCAResult]
        Per-cohort CCA fits.
    reference : str, optional
        Key of the reference cohort. Default: first in sorted order.

    Returns
    -------
    dict[str, CCAResult]
        New dict with sign-aligned copies. Originals are not mutated.
    """
    if reference is None:
        reference = sorted(results.keys())[0]

    ref_w = results[reference].x_weights['CC1'].values
    aligned = {}

    for key, res in results.items():
        w = res.x_weights['CC1'].values
        cos = np.dot(ref_w, w) / (np.linalg.norm(ref_w) * np.linalg.norm(w))
        if cos < 0:
            aligned[key] = CCAResult(
                x_weights=-res.x_weights,
                y_weights=-res.y_weights,
                x_scores=-res.x_scores,
                y_scores=-res.y_scores,
                correlations=res.correlations,
                p_values=res.p_values,
                n_recordings=res.n_recordings,
                n_permutations=res.n_permutations,
                alpha=res.alpha,
                l1_ratio=res.l1_ratio,
                x_variance_explained=res.x_variance_explained,
                y_variance_explained=res.y_variance_explained,
            )
        else:
            aligned[key] = res

    return aligned


def compare_cca_weights(result_a, result_b):
    """Cosine similarity between two CCA fits' CC1 weight vectors.

    Parameters
    ----------
    result_a, result_b : CCAResult

    Returns
    -------
    dict
        Keys: ``neural_cosine``, ``behavioral_cosine``.
    """
    def _cosine(a, b):
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        return dot / norm if norm > 0 else 0.0

    w_a_x = result_a.x_weights['CC1'].values
    w_b_x = result_b.x_weights['CC1'].values
    w_a_y = result_a.y_weights['CC1'].values
    w_b_y = result_b.y_weights['CC1'].values

    return {
        'neural_cosine': _cosine(w_a_x, w_b_x),
        'behavioral_cosine': _cosine(w_a_y, w_b_y),
    }


# =============================================================================
# Linear Mixed-Effects Models
# =============================================================================



class LMMResult:
    """Container for a single LMM fit result.

    Attributes
    ----------
    model : statsmodels MixedLM
    result : statsmodels MixedLMResults
    summary_df : pd.DataFrame
        Fixed-effects coefficient table.
    variance_explained : dict
        Keys: 'marginal', 'conditional'.
    random_effects : dict
        Subject → pd.Series of BLUPs.
    predictions : pd.DataFrame or None
        Model predictions on a design grid (set by callers, not by fit_lmm).
    emm_reward : pd.DataFrame or None
        Estimated marginal means for reward factor.
    emm_side : pd.DataFrame or None
        Estimated marginal means for side factor.
    emm_contrast : pd.DataFrame or None
        Estimated marginal means for contrast factor.
    contrast_slopes : pd.DataFrame or None
        Population and subject-level contrast slopes per reward condition.
    """

    def __init__(self, model, result, summary_df, variance_explained,
                 random_effects, contrast_coding='log',
                 contrast_center=0.0):
        self.model = model
        self.result = result
        self.summary_df = summary_df
        self.variance_explained = variance_explained
        self.random_effects = random_effects
        self.contrast_coding = contrast_coding
        self.contrast_center = contrast_center
        self.predictions = None
        self.emm_reward = None
        self.emm_side = None
        self.emm_contrast = None
        self.contrast_slopes = None
        self.interaction_contrast_reward = None
        self.interaction_contrast_side = None
        self.interaction_reward_side = None

    @property
    def contrast_col(self):
        """Column name for the contrast predictor in the fitted model."""
        return 'contrast'


def _variance_explained(result, df, response_col):
    """Partition the variance of the observed response explained by a MixedLM.

    A data-based variance partition, not Nakagawa & Schielzeth R². The
    denominator is the empirical ``var(observed y)``, fixed across the nested
    models compared in a drop-one analysis. Holding the denominator constant is
    what makes the difference of two marginal values a clean unique (semipartial)
    R²: the variance the dropped predictor explains over and above the rest.

    The ratios are returned unclipped. With a shared empirical denominator the
    fitted-value variance can exceed ``var(y)`` (sampling noise, misfit), so
    values may fall outside [0, 1]; a value outside that range signals model
    misfit rather than an error to be hidden.

    Parameters
    ----------
    result : statsmodels MixedLMResults
    df : pd.DataFrame
        The data used for fitting.
    response_col : str
        Name of the dependent variable column.

    Returns
    -------
    dict
        Keys: 'marginal' (fixed effects only) and 'conditional' (fixed +
        random), both float and unclipped. ``conditional >= marginal`` always
        holds because the random-effect variance is non-negative.
    """
    y = df[response_col].values
    var_y = np.var(y)
    if var_y == 0:
        return {'marginal': 0.0, 'conditional': 0.0}

    fe_params = result.fe_params.values
    exog = result.model.exog
    y_pred_fe = exog @ fe_params
    y_pred_full = result.fittedvalues.values

    var_fixed = np.var(y_pred_fe)
    var_random = np.var(y_pred_full - y_pred_fe)

    return {
        'marginal': float(var_fixed / var_y),
        'conditional': float((var_fixed + var_random) / var_y),
    }


def _warn_dropped_fit(formula: str, groups: pd.Series, exc: Exception) -> None:
    """Warn that a singular/degenerate LMM fit was dropped, so the loss of a
    fit from a result set is never silent."""
    warnings.warn(
        f"fit_lmm dropped a singular/degenerate fit "
        f"(formula={formula!r}, groups={groups.name!r}): {exc}"
    )


def _event_diff(trials, end_col, start_col):
    """Per-trial `end_col - start_col`, or all-NaN if either column is absent."""
    if end_col in trials.columns and start_col in trials.columns:
        return trials[end_col].values - trials[start_col].values
    return np.full(len(trials), np.nan)


def _peak_velocity(wheel_vel, n_trials):
    """Per-trial max |velocity| over finite samples; NaN where unavailable."""
    if wheel_vel is None:
        return np.full(n_trials, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # all-NaN rows
        return np.nanmax(np.abs(wheel_vel), axis=1)


def build_trial_regressors(
    trials: pd.DataFrame, wheel_velocity: np.ndarray | None
) -> pd.DataFrame:
    """Assemble one session's one-row-per-trial regressor frame.

    Copies the categorical/behavioral trial columns, derives the three
    event-timing differences, and reduces the wheel velocity matrix to a
    per-trial peak. Does not set ``eid`` — the caller tags it.

    Parameters
    ----------
    trials : pd.DataFrame
        One session's trials table. Must carry ``signed_contrast, contrast,
        stim_side, choice, feedbackType, probabilityLeft``; the event-time
        columns (``stimOn_times, firstMovement_times, feedback_times``) are
        optional and yield NaN timing columns when absent.
    wheel_velocity : np.ndarray or None
        ``(n_trials, n_samples)`` wheel velocity, or ``None`` when the wheel
        group is missing. ``None`` yields all-NaN ``peak_velocity``.

    Returns
    -------
    pd.DataFrame
        Columns: ``trial, signed_contrast, contrast, stim_side, choice,
        feedbackType, probabilityLeft, reaction_time, movement_time,
        response_time, peak_velocity``. ``reaction_time`` is
        ``firstMovement_times - stimOn_times``, ``movement_time`` is
        ``feedback_times - firstMovement_times``, ``response_time`` is
        ``feedback_times - stimOn_times`` (seconds).
    """
    copy_cols = ['signed_contrast', 'contrast', 'stim_side', 'choice',
                 'feedbackType', 'probabilityLeft']
    n_trials = len(trials)
    df = pd.DataFrame({'trial': range(n_trials)})
    for col in copy_cols:
        df[col] = trials[col].values
    df['reaction_time'] = _event_diff(
        trials, 'firstMovement_times', 'stimOn_times')
    df['movement_time'] = _event_diff(
        trials, 'feedback_times', 'firstMovement_times')
    df['response_time'] = _event_diff(
        trials, 'feedback_times', 'stimOn_times')
    df['peak_velocity'] = _peak_velocity(wheel_velocity, n_trials)
    return df


def select_modeling_trials(
    df: pd.DataFrame, response_col: str = 'response',
    probability_left: float | None = None,
) -> pd.DataFrame:
    """Keep the go trials usable for response modeling.

    Drops no-go trials (``choice == 0``), false starts
    (``response_time <= 0.05``), and trials with a null ``response_col``. Adds a
    ``log_<var>`` column (base-10, NaN where the value is ≤ 0) for each
    ``config.MOVEMENT_PREDICTORS`` entry coded as ``log_<var>``, so movement
    models can reference them; the NaN rows are dropped per family at fit time.

    Parameters
    ----------
    df : pd.DataFrame
        Merged trial frame carrying ``response_col``, ``probabilityLeft``,
        ``choice``, ``response_time``, and the log-transformed movement columns.
    response_col : str
        Column name for the response magnitude whose NaNs are dropped.
    probability_left : float or None
        When set, keep only trials with this ``probabilityLeft`` (e.g. ``0.5``
        for the unbiased block). ``None`` (default) keeps all blocks.

    Returns
    -------
    pd.DataFrame
        The retained trials, with added ``log_<var>`` columns. A copy; the
        input is not mutated.
    """
    if probability_left is not None:
        df = df[df['probabilityLeft'] == probability_left]
    df = df.dropna(subset=[response_col])
    df = df.query('choice != 0 and response_time > 0.05').copy()
    for var, pred in MOVEMENT_PREDICTORS.items():
        if pred == f'log_{var}' and var in df.columns:
            df[pred] = np.where(df[var] > 0, np.log10(df[var]), np.nan)
    return df


def code_predictors(
    df: pd.DataFrame, contrast_coding: str = 'log2'
) -> pd.DataFrame:
    """Code the trial frame for model fitting; do not mutate the input.

    Returns a copy with ``contrast`` transformed (``contrast_coding``) and
    mean-centered, and ``side`` / ``choice_side`` / ``reward`` deviation-coded
    to ±0.5 (``side`` and ``choice_side``: contra = +0.5, ipsi = −0.5;
    ``reward``: ``feedbackType`` 1 = +0.5, −1 = −0.5). ``log_<timing>`` columns
    are left untouched. Coding a column a given formula does not use, or one
    absent from ``df``, is harmless.

    Parameters
    ----------
    df : pd.DataFrame
        Trial-level frame with columns ``contrast``, ``side``, and
        ``feedbackType``; optionally ``choice_side``.
    contrast_coding : str
        Coding passed to :func:`iblnm.util.get_contrast_coding`.

    Returns
    -------
    pd.DataFrame
        A coded copy; the input is not mutated.
    """
    transform, _ = get_contrast_coding(contrast_coding)
    df = df.copy()
    coded = transform(df['contrast'])
    df['contrast'] = coded - float(np.mean(coded))
    df['side'] = np.where(df['side'] == 'contra', 0.5, -0.5)
    df['reward'] = np.where(df['feedbackType'] == 1, 0.5, -0.5)
    if 'choice_side' in df.columns:
        df['choice_side'] = np.where(df['choice_side'] == 'contra', 0.5, -0.5)
    return df


def formula_columns(formula: str, columns) -> list:
    """Return the data columns a Wilkinson formula references.

    Selects the members of ``columns`` that appear as whole words in
    ``formula``, so wrapping (``C(contrast)``), interactions
    (``contrast * side``), and the response on the left-hand side are all
    handled without parsing the formula grammar. Word boundaries keep
    ``contrast`` from matching inside ``relative_contrast`` and ``reaction_time``
    from matching inside ``log_reaction_time``, so only columns actually named
    are returned.

    Parameters
    ----------
    formula : str
        A Wilkinson formula, e.g. ``'response ~ contrast + log_reaction_time'``.
    columns : iterable of str
        Candidate column names (typically ``df.columns``).

    Returns
    -------
    list of str
        The subset of ``columns`` named in ``formula``, in the order of
        ``columns``.
    """
    return [c for c in columns
            if re.search(rf'\b{re.escape(c)}\b', formula)]


def formula_union_columns(formulas, columns) -> list:
    """Columns referenced by any formula in a family, for complete-case fitting.

    The union of :func:`formula_columns` over ``formulas``, so a set of models
    meant to be compared can be reduced to the rows valid for every member.

    Parameters
    ----------
    formulas : iterable of str
        Wilkinson formulas (one comparison family).
    columns : iterable of str
        Candidate column names (typically ``df.columns``).

    Returns
    -------
    list of str
        Every column named by at least one formula, in the order of ``columns``.
    """
    referenced = set().union(
        *(formula_columns(f, columns) for f in formulas))
    return [c for c in columns if c in referenced]


def fit_lmm(formula, df, groups, re_formula='1', reml=True,
             contrast_coding='log', contrast_center=0.0):
    """Fit a linear mixed-effects model and return an LMMResult.

    Generic fitting function: builds the model, checks for convergence,
    computes coefficient summary and variance explained. Does not compute
    prediction grids or marginal means — callers add those.

    Parameters
    ----------
    formula : str
        Wilkinson formula for fixed effects.
    df : pd.DataFrame
        Trial-level data.
    groups : pd.Series
        Grouping variable for random effects.
    re_formula : str
        Random-effects formula (default ``'1'``).
    reml : bool
        Use REML (True) or ML (False) estimation.

    Returns
    -------
    LMMResult or None
        None if the model fails to converge or data is degenerate.
    """
    import statsmodels.formula.api as smf
    from statsmodels.tools.sm_exceptions import ConvergenceWarning

    # Complete-case rows: statsmodels' formula path drops NaN rows from the
    # design but not from the separately-passed ``groups``, misaligning them.
    # Drop here so ``df``, ``groups``, and the fit all share the same rows.
    keep = df[formula_columns(formula, df.columns)].notna().all(axis=1)
    df, groups = df[keep], groups[keep]

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model = smf.mixedlm(
                formula, df, groups=groups,
                re_formula=re_formula,
            )
            result = model.fit(reml=reml, method='powell', maxiter=500)
        fatal = any(
            issubclass(w.category, ConvergenceWarning)
            and 'failed to converge' in str(w.message).lower()
            for w in caught
        )
        if fatal:
            return None
    except (np.linalg.LinAlgError, ValueError) as exc:
        _warn_dropped_fit(formula, groups, exc)
        return None

    # Determine response column from formula
    response_col = formula.split('~')[0].strip()

    # Summary table
    fe_names = list(result.fe_params.index)
    summary_df = pd.DataFrame({
        'Coef.': result.fe_params,
        'Std.Err.': result.bse_fe,
        'z': result.tvalues[fe_names],
        'P>|z|': result.pvalues[fe_names],
    })

    # Variance explained and the random-effects dict depend on lazily-evaluated
    # BLUPs (result.fittedvalues, result.random_effects), which raise on a
    # singular random-effects covariance (e.g. a random slope whose variance
    # collapsed to 0). Treat that degenerate fit as a failure, like the fit
    # itself failing.
    try:
        ve = _variance_explained(result, df, response_col)
        re_dict = {
            subj: pd.Series(effects)
            for subj, effects in result.random_effects.items()
        }
    except (np.linalg.LinAlgError, ValueError) as exc:
        _warn_dropped_fit(formula, groups, exc)
        return None

    if ve['marginal'] == 0.0 and ve['conditional'] == 0.0 and np.var(df[response_col].values) == 0:
        return None

    return LMMResult(
        model=model,
        result=result,
        summary_df=summary_df,
        variance_explained=ve,
        random_effects=re_dict,
        contrast_coding=contrast_coding,
        contrast_center=contrast_center,
    )


def fit_ols(formula: str, df: pd.DataFrame):
    """Fit one ordinary-least-squares model from a Wilkinson formula.

    Formula-agnostic single-model fit: no predictor coding or column logic
    happens here, so the caller owns the design. Returns ``None`` on a
    degenerate design rather than raising, mirroring ``fit_lmm``'s
    None-on-failure contract so a comparison loop can skip the unit cleanly.
    statsmodels' OLS solves rank-deficient designs through a pseudo-inverse
    instead of raising, so a singular design is caught by comparing the fitted
    design's rank to its column count, not by an exception.

    Parameters
    ----------
    formula : str
        Wilkinson formula, e.g. ``'response ~ contrast + side'``.
    df : pd.DataFrame
        Trial-level data carrying every column the formula references.

    Returns
    -------
    statsmodels RegressionResults or None
        The fitted results (exposes ``.rsquared``, ``.params``), or ``None`` if
        the design is empty or rank-deficient.
    """
    import statsmodels.formula.api as smf

    try:
        result = smf.ols(formula, df).fit()
    except (np.linalg.LinAlgError, ValueError):
        return None
    if result.model.rank < result.model.exog.shape[1]:
        return None
    return result


def dropone_delta_r2(r2_by_name, reference: str = 'full') -> pd.DataFrame:
    """Drop-one ΔR² for one unit, differencing each reduced model off a baseline.

    Pure, in-sample, single-unit counterpart to ``crossval_lmm``'s differencing:
    ``r2_by_name`` maps model name → R² for one recording×event, where one key is
    the full ``reference`` model and each other key is a reduced model with one
    regressor dropped. Each reduced model's ΔR² is the R² the reference gains over
    it — the dropped regressor's unique in-sample contribution.

    Parameters
    ----------
    r2_by_name : dict or pd.Series
        Mapping of model name → R² for one unit. Must contain ``reference``.
    reference : str
        Key naming the full model each reduced model's ΔR² is measured against.

    Returns
    -------
    pd.DataFrame
        One row per non-``reference`` name, columns ``predictor, r2, delta_r2``
        where ``r2`` is the reference R² (same on every row) and ``delta_r2`` is
        ``r2[reference] − r2[name]``. Empty frame with those columns if only the
        reference is present.
    """
    r2_ref = r2_by_name[reference]
    rows = [
        {'predictor': name, 'r2': r2_ref, 'delta_r2': r2_ref - r2}
        for name, r2 in r2_by_name.items()
        if name != reference
    ]
    return pd.DataFrame(rows, columns=['predictor', 'r2', 'delta_r2'])


def compute_feature_dispersion(
    df: pd.DataFrame,
    unit_cols: list[str],
    session_col: str,
    feature_col: str,
    value_col: str,
    standardize_by: str | None = None,
) -> pd.DataFrame:
    """Per-unit normalized dispersion of feature values across sessions.

    Variable-agnostic: operates on long-form ``(unit, session, feature, value)``
    rows. Each feature is first z-scored to a common scale so differently-scaled
    features contribute equally; the dispersion is then the root-mean-square,
    over features, of each unit's population variance across its sessions.

    Parameters
    ----------
    df : pd.DataFrame
        Long-form rows; one value per ``(unit, session, feature)``.
    unit_cols : list[str]
        Columns identifying a unit (the grouping the dispersion is computed for).
    session_col : str
        Column whose distinct values are the sessions a unit varies over.
    feature_col : str
        Column naming the feature each value belongs to.
    value_col : str
        Column holding the numeric values.
    standardize_by : str or None
        If given, z-score within each ``(standardize_by, feature_col)`` group so
        scaling is per-cohort; if None, z-score per feature over the whole frame.
        A feature constant within its standardization group has zero spread and
        is mapped to z = 0 (contributing nothing), not NaN.

    Returns
    -------
    pd.DataFrame
        One row per unit, columns ``unit_cols + ['dispersion', 'n_sessions']``.
        ``dispersion`` is ``sqrt(mean over features of the population variance
        (ddof=0) of the z-scored value across the unit's sessions)``;
        ``n_sessions`` is the unit's distinct ``session_col`` count. Z-scoring
        uses the full frame passed in, before any downstream session-count
        filtering.
    """
    group_keys = [feature_col] if standardize_by is None \
        else [standardize_by, feature_col]
    grouped = df.groupby(group_keys)[value_col]
    std = grouped.transform('std', ddof=0)
    z = (df[value_col] - grouped.transform('mean')) / std
    z = z.where(std != 0, 0.0)

    feature_var = z.groupby([df[c] for c in unit_cols + [feature_col]]).var(ddof=0)
    dispersion = feature_var.groupby(level=unit_cols).mean().pow(0.5)
    n_sessions = df.groupby(unit_cols)[session_col].nunique()
    out = pd.DataFrame({'dispersion': dispersion, 'n_sessions': n_sessions})
    return out.reset_index()[unit_cols + ['dispersion', 'n_sessions']]


def compute_marginal_means(lmm_result, factors):
    """Estimated marginal means for a set of factors from an LMM fit.

    For each combination of the listed factors' levels — taken from the values
    present in the fitted design — predicts the fixed-effects mean response
    (± 95% CI), holding every other predictor at its sample mean (0 under the
    deviation/centered coding). One factor yields main-effect EMMs; two yield the
    interaction grid whose non-parallel pattern is the interaction. Builds its
    own design row via the model's patsy design info and predicts, subsuming the
    old ``compute_predictions`` grid. Names no variable: the caller passes which
    factors, and the output carries each factor's coded level (label mapping is
    the plotting layer's job).

    Parameters
    ----------
    lmm_result : LMMResult
    factors : sequence of str
        Predictor columns (as named in the fitted formula) to cross.

    Returns
    -------
    pd.DataFrame
        One row per factor-level combination: a column per factor (its coded
        level value), plus ``predicted``, ``ci_lower``, ``ci_upper``.
    """
    from itertools import product
    from patsy import dmatrix

    result = lmm_result.result
    fe_names = list(result.fe_params.index)
    fe_params = result.fe_params.values
    fe_cov = result.cov_params().loc[fe_names, fe_names].values
    design_info = result.model.data.orig_exog.design_info
    frame = result.model.data.frame

    predictors = [fi.name() for fi in design_info.factor_infos]
    means = {p: float(frame[p].mean()) for p in predictors}
    grids = {f: sorted(frame[f].unique()) for f in factors}

    rows = []
    for combo in product(*(grids[f] for f in factors)):
        row_vals = dict(means)
        row_vals.update(dict(zip(factors, combo)))
        X = np.asarray(dmatrix(design_info, pd.DataFrame([row_vals])))
        pred = float((X @ fe_params)[0])
        se = float(np.sqrt(max((X @ fe_cov @ X.T)[0, 0], 0.0)))
        row = dict(zip(factors, combo))
        row['predicted'] = pred
        row['ci_lower'] = pred - 1.96 * se
        row['ci_upper'] = pred + 1.96 * se
        rows.append(row)

    return pd.DataFrame(rows)


def feature_unique_contribution(response_matrix, labels, subjects,
                                normalize=True, C=None):
    """Measure each feature's unique contribution to decoding accuracy.

    For each feature, refit the full-data model without it and measure the
    drop in balanced accuracy. Uses a fixed C for comparability.

    Parameters
    ----------
    response_matrix : pd.DataFrame
    labels : pd.Series
    subjects : pd.Series
        Not used directly, kept for API consistency with ``decode_target_nm``.
    normalize : bool
        Z-score features before fitting.
    C : float, optional
        Regularization strength. Should be the ``best_C`` from CV tuning.

    Returns
    -------
    pd.DataFrame
        Columns: feature, full_accuracy, reduced_accuracy, delta.
    """
    from sklearn.preprocessing import LabelEncoder

    if C is None:
        # Fall back to CV to determine C
        result = decode_target_nm(response_matrix, labels, subjects,
                                   normalize=normalize)
        C = result['best_C']

    clean = response_matrix.dropna()
    y = labels.loc[clean.index]
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    full_acc = _fit_full_data(clean.values, y_enc, C, normalize)

    rows = []
    for col in clean.columns:
        reduced_X = clean.drop(columns=[col]).values
        reduced_acc = _fit_full_data(reduced_X, y_enc, C, normalize)
        delta = full_acc - reduced_acc
        rows.append({
            'feature': col,
            'full_accuracy': full_acc,
            'reduced_accuracy': reduced_acc,
            'delta': delta,
        })

    return pd.DataFrame(rows)


# =============================================================================
# Movement Encoding LMM
# =============================================================================


def _centered_r2(lmm_result, df_held, y_centered, ss_tot):
    """Out-of-sample R² for a held-out subject after removing its intercept.

    Builds the held-out design from the fitted model's ``design_info``, predicts
    with the fixed effects only, then centers both the prediction and the
    observed response on the held-out subject's mean — removing the subject's
    own random intercept, which a population fit cannot know. ``y_centered`` is
    the already-mean-centered response and ``ss_tot`` its total sum of squares.
    Shared by the movement and task-model LOSO-CV routines.
    """
    from patsy import dmatrix

    design_info = lmm_result.result.model.data.orig_exog.design_info
    pred = np.asarray(dmatrix(design_info, df_held)) @ lmm_result.result.fe_params.values
    return float(1 - np.sum((y_centered - (pred - np.mean(pred))) ** 2) / ss_tot)


def crossval_lmm(df, formulas, response_col, reference='full',
                 fold_col='subject', min_subjects=3, min_test=5):
    """Out-of-sample ΔR² by leave-one-fold-out cross-validation.

    Formula-agnostic resampling: leave out one ``fold_col`` value at a time, fit
    each model in ``formulas`` on the N−1 training folds, and score the held-out
    fold out of sample with ``_centered_r2`` (which removes the held-out fold's
    own random intercept). The ``reference`` key names the baseline model; every
    other key's ΔR² is the held-out R² the reference gains over that reduced
    model. Subsumes the bespoke ``loso_cv_*`` routines; the caller codes ``df``.

    Parameters
    ----------
    df : pd.DataFrame
        Trial-level data, already coded, with ``response_col`` and ``fold_col``.
    formulas : dict[str, str]
        Name → Wilkinson formula. Must contain the ``reference`` key.
    response_col : str
        Column name for the response magnitude.
    reference : str
        Key in ``formulas`` naming the baseline model each other model's ΔR² is
        measured against.
    fold_col : str
        Column whose unique values define the leave-one-out folds.
    min_subjects : int
        Minimum number of folds required (≥ 3 so training keeps ≥ 2 folds).
    min_test : int
        Minimum held-out trials for a fold to be scored.

    Returns
    -------
    pd.DataFrame
        One row per (``fold``, ``predictor``) for each non-``reference`` formula,
        with columns ``fold, predictor, n_trials, r2, delta_r2``. ``r2`` is the
        held-out reference R²; ``delta_r2`` is ``r2_<reference> − r2_<predictor>``.
        A ``fold == 'aggregate'`` row per predictor holds the across-fold mean
        ``r2``/``delta_r2`` and summed ``n_trials``. Empty frame with those
        columns if fewer than ``min_subjects`` folds or no fold is scorable.
    """
    cols = ['fold', 'predictor', 'n_trials', 'r2', 'delta_r2']
    folds = df[fold_col].unique()
    if len(folds) < min_subjects:
        return pd.DataFrame(columns=cols)

    rows = []
    for fold in folds:
        df_train = df[df[fold_col] != fold]
        df_test = df[df[fold_col] == fold]
        if len(df_test) < min_test or df_train[fold_col].nunique() < 2:
            continue

        fits = {name: fit_lmm(formula, df_train, groups=df_train[fold_col],
                              re_formula='1', reml=False)
                for name, formula in formulas.items()}
        if any(fit is None for fit in fits.values()):
            continue

        y_centered = df_test[response_col].values - df_test[response_col].mean()
        ss_tot = np.sum(y_centered ** 2)
        if ss_tot == 0:
            continue

        r2 = {name: _centered_r2(fit, df_test, y_centered, ss_tot)
              for name, fit in fits.items()}
        rows.extend(
            {'fold': fold, 'predictor': name, 'n_trials': len(df_test),
             'r2': r2[reference], 'delta_r2': r2[reference] - r2[name]}
            for name in formulas if name != reference
        )

    return _aggregate_fold_rows(rows, cols)


def jackknife_lmm(df, formulas, response_col, reference='full',
                  fold_col='subject', min_subjects=3):
    """In-sample-influence ΔR² by leave-one-fold-out jackknife.

    Same fold loop as ``crossval_lmm``, but each fold scores the *training* set
    in sample rather than a held-out fold: leave out one ``fold_col`` value, fit
    each model in ``formulas`` on the N−1 training folds, and read each fit's
    in-sample marginal R² (``variance_explained['marginal']``). The spread of
    the per-fold ΔR² shows whether any single fold drives the reference model's
    advantage. The ``reference`` key names the reference model; every other
    key's ΔR² is the marginal R² the reference gains over that reduced model.

    Parameters
    ----------
    df : pd.DataFrame
        Trial-level data, already coded, with ``response_col`` and ``fold_col``.
    formulas : dict[str, str]
        Name → Wilkinson formula. Must contain the ``reference`` key.
    response_col : str
        Column name for the response magnitude.
    reference : str
        Key in ``formulas`` naming the model each other model's ΔR² is measured
        against.
    fold_col : str
        Column whose unique values define the leave-one-out folds.
    min_subjects : int
        Minimum number of folds required (≥ 3 so training keeps ≥ 2 folds).

    Returns
    -------
    pd.DataFrame
        One row per (``fold``, ``predictor``) for each non-``reference``
        formula, with columns ``fold, predictor, n_trials, r2, delta_r2``.
        ``r2`` is the training-set reference marginal R²; ``delta_r2`` is
        ``r2_<reference> − r2_<predictor>``; ``n_trials`` is the training-set
        size. A ``fold == 'aggregate'`` row per predictor holds the across-fold
        mean ``r2``/``delta_r2`` and summed ``n_trials``. Empty frame with those
        columns if fewer than ``min_subjects`` folds or no fold is scorable.
    """
    cols = ['fold', 'predictor', 'n_trials', 'r2', 'delta_r2']
    folds = df[fold_col].unique()
    if len(folds) < min_subjects:
        return pd.DataFrame(columns=cols)

    rows = []
    for fold in folds:
        df_train = df[df[fold_col] != fold]
        if df_train[fold_col].nunique() < 2:
            continue

        fits = {name: fit_lmm(formula, df_train, groups=df_train[fold_col],
                              re_formula='1', reml=False)
                for name, formula in formulas.items()}
        if any(fit is None for fit in fits.values()):
            continue

        r2 = {name: fit.variance_explained['marginal']
              for name, fit in fits.items()}
        rows.extend(
            {'fold': fold, 'predictor': name, 'n_trials': len(df_train),
             'r2': r2[reference], 'delta_r2': r2[reference] - r2[name]}
            for name in formulas if name != reference
        )

    return _aggregate_fold_rows(rows, cols)


def _aggregate_fold_rows(rows: list, cols: list) -> pd.DataFrame:
    """Append a per-predictor ``aggregate`` row to per-fold resampling rows.

    Shared tail of ``crossval_lmm`` and ``jackknife_lmm``: the aggregate row
    sums ``n_trials`` and averages ``r2``/``delta_r2`` across folds. ``rows`` is
    the list of per-fold dicts; ``cols`` the standard resampling columns. Empty
    frame with ``cols`` if ``rows`` is empty.
    """
    if not rows:
        return pd.DataFrame(columns=cols)

    fold_rows = pd.DataFrame(rows, columns=cols)
    aggregate = (fold_rows.groupby('predictor', sort=False)
                 .agg(n_trials=('n_trials', 'sum'),
                      r2=('r2', 'mean'),
                      delta_r2=('delta_r2', 'mean'))
                 .reset_index())
    aggregate.insert(0, 'fold', 'aggregate')
    return pd.concat([fold_rows, aggregate], ignore_index=True)


def compute_recording_projection(n_analysis_ready, n_total, target_n,
                                 deadline, capacity_per_day, today=None):
    """Compute recording capacity projection per target.

    Parameters
    ----------
    n_analysis_ready : dict
        {target_NM: count} of sessions currently passing all filters.
    n_total : dict
        {target_NM: count} of total sessions recorded per target.
    target_n : int
        Goal number of analysis-ready sessions per target.
    deadline : datetime.date
        Recording deadline.
    capacity_per_day : int
        Total sessions that can be recorded per day across all targets.
    today : datetime.date, optional
        Reference date (defaults to date.today()).

    Returns
    -------
    pd.DataFrame
        One row per target_NM with projection columns.
    """
    from datetime import date as _date
    if today is None:
        today = _date.today()

    days_available = (deadline - today).days

    rows = []
    for target in n_analysis_ready:
        ready = n_analysis_ready[target]
        total = n_total[target]

        if total == 0:
            yield_rate = np.nan
        else:
            yield_rate = ready / total

        shortfall = max(0, target_n - ready)

        if shortfall == 0:
            effective = 0
            rec_days = 0.0
        elif yield_rate == 0:
            effective = np.inf
            rec_days = np.inf
        elif np.isnan(yield_rate):
            effective = np.nan
            rec_days = np.nan
        else:
            effective = int(np.ceil(shortfall / yield_rate))
            rec_days = 0.0  # computed below from totals

        rows.append({
            'target_NM': target,
            'n_analysis_ready': ready,
            'n_total': total,
            'yield_rate': yield_rate,
            'shortfall': shortfall,
            'effective_sessions_needed': effective,
            'recording_days_needed': rec_days,
            'days_available': days_available,
        })

    df = pd.DataFrame(rows)

    # Recording days: total effective sessions across all targets / capacity
    total_effective = df['effective_sessions_needed'].replace([np.inf], np.nan).sum()
    if np.isfinite(total_effective) and total_effective > 0:
        # Distribute days proportionally to each target's share
        finite_mask = np.isfinite(df['effective_sessions_needed'])
        finite_sum = df.loc[finite_mask, 'effective_sessions_needed'].sum()
        if finite_sum > 0:
            df.loc[finite_mask, 'recording_days_needed'] = (
                df.loc[finite_mask, 'effective_sessions_needed'] / capacity_per_day
            ).apply(np.ceil)

    # Restore inf for zero-yield targets
    df.loc[df['yield_rate'] == 0, 'recording_days_needed'] = np.inf

    return df


def anova_rm(df, depvar, subject, within):
    """Repeated-measures ANOVA on a pre-aggregated DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        One row per subject x condition cell. Must contain ``depvar``,
        ``subject``, and every column listed in ``within``.
    depvar : str
        Column name of the dependent variable.
    subject : str
        Column name identifying subjects.
    within : list of str
        Within-subject factor column names. All treated as categorical.

    Returns
    -------
    pd.DataFrame
        ANOVA table with columns Source, F, Num DF, Den DF, Pr(>F).
        Includes an extra column ``method`` ('rm' or 'ols') indicating
        which method was used.
    """
    from itertools import combinations
    from statsmodels.stats.anova import AnovaRM
    import statsmodels.api as sm
    from statsmodels.formula.api import ols

    df = df.copy()
    for col in within:
        df[col] = df[col].astype(str)

    # Check balance: every subject must appear in every cell exactly once
    cells = df.groupby(within).ngroups
    subject_cell_counts = df.groupby(subject)[within[0]].count()
    balanced = (subject_cell_counts == cells).all() and len(subject_cell_counts) >= 2

    if balanced:
        aov = AnovaRM(df, depvar, subject, within=within).fit()
        result = aov.anova_table.reset_index()
        result = result.rename(columns={
            result.columns[0]: 'Source',
            'F Value': 'F',
            'Pr > F': 'Pr(>F)',
        })
        result['method'] = 'rm'
    else:
        warnings.warn(
            "Unbalanced repeated-measures design: not all subjects have data "
            "in every condition cell. Falling back to between-subjects OLS "
            "ANOVA (Type III).",
            UserWarning,
            stacklevel=2,
        )
        # Build formula with all main effects and interactions
        # C() wraps each factor as categorical
        terms = [f'C({w})' for w in within]
        # All interactions: 2-way, 3-way, ...
        interaction_terms = []
        for r in range(2, len(within) + 1):
            for combo in combinations(terms, r):
                interaction_terms.append(':'.join(combo))
        formula = f'{depvar} ~ {" + ".join(terms + interaction_terms)}'
        model = ols(formula, data=df).fit()
        aov_table = sm.stats.anova_lm(model, typ=3)
        # Drop Intercept and Residual rows
        aov_table = aov_table.drop(
            index=[idx for idx in aov_table.index
                   if idx in ('Intercept', 'Residual')],
        )
        result = aov_table.reset_index()
        result.columns = ['Source', 'SS', 'Num DF', 'F', 'Pr(>F)']
        # Clean source names: strip C() wrapping
        result['Source'] = (
            result['Source']
            .str.replace(r'C\(([^)]+)\)', r'\1', regex=True)
        )
        # Compute Den DF from residual
        den_df = model.df_resid
        result['Den DF'] = den_df
        result = result[['Source', 'F', 'Num DF', 'Den DF', 'Pr(>F)']]
        result['method'] = 'ols'

    return result


def kruskal_wallis_groups(df, group_col, value_col):
    """Kruskal-Wallis H-test across groups defined by a categorical column.

    Parameters
    ----------
    df : pd.DataFrame
        Data with at least ``group_col`` and ``value_col``.
    group_col : str
        Column defining groups.
    value_col : str
        Column with the numeric values to compare.

    Returns
    -------
    H : float
        Kruskal-Wallis H statistic (NaN if fewer than 2 non-empty groups).
    p : float
        p-value (NaN if fewer than 2 non-empty groups).
    groups : dict[str, np.ndarray]
        Mapping of group name to array of non-NaN values.
    """
    from scipy.stats import kruskal

    groups = {}
    for name, sub in df.groupby(group_col):
        vals = sub[value_col].dropna().values
        if len(vals) > 0:
            groups[name] = vals

    if len(groups) < 2:
        return np.nan, np.nan, groups

    H, p = kruskal(*groups.values())
    return H, p, groups


def pairwise_mannwhitney(groups, correction='bonferroni'):
    """Pairwise Mann-Whitney U tests with multiple-comparison correction.

    Parameters
    ----------
    groups : dict[str, np.ndarray]
        Mapping of group name to array of values (from kruskal_wallis_groups).
    correction : str
        Correction method. Only 'bonferroni' is supported.

    Returns
    -------
    list of (group_a, group_b, U_statistic, p_corrected)
    """
    from itertools import combinations
    from scipy.stats import mannwhitneyu

    names = list(groups.keys())
    pairs = list(combinations(names, 2))
    n_comparisons = len(pairs)

    results = []
    for a, b in pairs:
        U, p_raw = mannwhitneyu(groups[a], groups[b], alternative='two-sided')
        p_corr = min(p_raw * n_comparisons, 1.0)
        results.append((a, b, U, p_corr))

    return results


# --- FIR encoding-model design primitives ---


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

    Parameters
    ----------
    times : np.ndarray
        Event/query times in seconds.
    tvec : np.ndarray
        Uniform model time grid.
    clip : bool
        If True, clamp indices to [0, len(tvec)] for use as slice bounds; if
        False, return raw indices (caller masks out-of-range).

    Returns
    -------
    np.ndarray
        Integer indices, one per input time.
    """
    # robust grid spacing (averages out arange float accumulation)
    dt = (tvec[-1] - tvec[0]) / (tvec.size - 1)
    indices = np.round((times - tvec[0]) / dt).astype(int)
    if clip:
        indices = np.clip(indices, 0, tvec.size)
    return indices


def make_event_regressor(
    event_times: np.ndarray, tvec: np.ndarray, heights: np.ndarray | None = None
) -> np.ndarray:
    """Event regressor: per-event height at the nearest grid sample.

    Parameters
    ----------
    event_times : np.ndarray
        Event timestamps in seconds.
    tvec : np.ndarray
        Uniform model time grid.
    heights : np.ndarray, optional
        Per-event values placed at each event sample (parametric modulator). Must
        align with `event_times`. Defaults to unit height (binary train).

    Returns
    -------
    np.ndarray
        1-D regressor on `tvec`; events outside the grid are dropped.
    """
    indices = times_to_indices(event_times, tvec)
    # keep only events falling inside the grid
    valid = (indices >= 0) & (indices < tvec.size)
    regressor = np.zeros(tvec.size)
    regressor[indices[valid]] = 1.0 if heights is None else np.asarray(heights)[valid]
    return regressor


def lag_expand(regressor: np.ndarray, lags: np.ndarray) -> np.ndarray:
    """Build lagged copies of a regressor, one column per lag.

    Convention: column ``j`` is `regressor` shifted by ``lags[j]`` samples
    (zero-padded, no wrap), so a positive lag shifts the event's contribution
    later in time. A fitted positive-lag coefficient is therefore the signal's
    post-event response (lag 0 == event onset).

    Parameters
    ----------
    regressor : np.ndarray
        1-D regressor on the model grid.
    lags : np.ndarray
        Integer sample lags, e.g. ``arange(-25, 25)``.

    Returns
    -------
    np.ndarray
        Matrix of shape ``(len(regressor), len(lags))``.
    """
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

    return np.stack([shift(regressor, int(lag)) for lag in lags], axis=1)


def raised_cosine_basis(
    n_basis: int, rcos_duration: float, rcos_nloffset: float, dt: float
) -> np.ndarray:
    """Build a causal log-raised-cosine "bump" basis.

    Reproduces the brain-wide-map basis (``neurencoding.utils.nonlinear_rcos``):
    time is log-warped so bumps are dense just after the event and sparse later.
    The basis spans ``[0, rcos_duration]`` after the event.

    Parameters
    ----------
    n_basis : int
        Number of bumps (basis columns).
    rcos_duration : float
        Kernel window in seconds.
    rcos_nloffset : float
        Log-warp offset in seconds (must be > 0); smaller packs more bumps near
        the event.
    dt : float
        Time-grid resolution in seconds.

    Returns
    -------
    np.ndarray
        ``(n_kernel, n_basis)`` basis, ``n_kernel = ceil(rcos_duration / dt)``.
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
    regressor: np.ndarray,
    tvec: np.ndarray,
    n_basis: int,
    rcos_duration: float,
    rcos_nloffset: float,
) -> np.ndarray:
    """Expand an event impulse train onto a log-raised-cosine kernel basis.

    Returns one column per bump (the impulse train convolved with each basis
    function). Fitted coefficients weight the bumps, and the kernel is their
    weighted sum. See `raised_cosine_basis` for the basis itself.

    Parameters
    ----------
    regressor : np.ndarray
        1-D binary event regressor on the grid (length ``len(tvec)``).
    tvec : np.ndarray
        Uniform model time grid; supplies grid resolution ``dt`` and length.
    n_basis : int
        Number of raised-cosine bumps.
    rcos_duration : float
        Post-event kernel window in seconds.
    rcos_nloffset : float
        Log-warp offset in seconds.

    Returns
    -------
    np.ndarray
        ``(len(tvec), n_basis)`` block on the grid.
    """
    dt = (tvec[-1] - tvec[0]) / (tvec.size - 1)
    basis = raised_cosine_basis(n_basis, rcos_duration, rcos_nloffset, dt)
    # convolve the impulse train with each bump, truncate to the grid length
    return np.stack(
        [np.convolve(regressor, basis[:, j])[: tvec.size] for j in range(n_basis)],
        axis=1,
    )


def make_trial_constant(
    trials: pd.DataFrame, column: str, tvec: np.ndarray
) -> np.ndarray:
    """Step (tonic) regressor: `column` held constant across each trial interval.

    Each trial's value of `column` fills the grid samples spanning its
    ``[intervals_0, intervals_1]`` interval; samples outside every interval
    stay zero.

    Parameters
    ----------
    trials : pd.DataFrame
        Trials table with ``intervals_0``/``intervals_1`` interval bounds (s).
    column : str
        Trial column whose value fills each interval.
    tvec : np.ndarray
        Uniform model time grid.

    Returns
    -------
    np.ndarray
        1-D step regressor on `tvec`.
    """
    values = np.zeros(tvec.size)
    for _, row in trials.iterrows():
        start, stop = times_to_indices(
            np.array([row["intervals_0"], row["intervals_1"]]), tvec, clip=True
        )
        values[start:stop] = row[column]
    return values


def build_design_matrix(
    blocks: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, slice]]:
    """Concatenate named regressor blocks; return the matrix and a span map.

    Block insertion order sets column order; the returned slices let callers
    retrieve any block's coefficients by name.

    Parameters
    ----------
    blocks : dict[str, np.ndarray]
        Ordered ``name -> block`` mapping. A 1-D block is promoted to a single
        column; a 2-D block contributes its columns as-is. All blocks must share
        the same number of rows.

    Returns
    -------
    tuple[np.ndarray, dict[str, slice]]
        The column-wise concatenation in insertion order, and a
        ``name -> slice`` map giving each block's column span in the matrix.
    """
    matrix, slices, start = [], {}, 0
    for name, block in blocks.items():
        columns = block[:, None] if block.ndim == 1 else block
        slices[name] = slice(start, start + columns.shape[1])
        matrix.append(columns)
        start += columns.shape[1]
    return np.concatenate(matrix, axis=1), slices


def interpolate_to_grid(
    series: pd.Series | pd.DataFrame, tvec: np.ndarray, kind: str = "quadratic"
) -> np.ndarray:
    """Resample a time-indexed series onto the model grid `tvec`.

    Unlike `resample_signal`, which builds its own PCHIP grid, this places an
    externally supplied series onto a caller-given grid and NaN-pads samples
    outside the source's time support.

    Parameters
    ----------
    series : pd.Series | pd.DataFrame
        Source values indexed by time (s). A frame interpolates each column.
    tvec : np.ndarray
        Uniform model time grid.
    kind : str
        scipy `interp1d` interpolation kind.

    Returns
    -------
    np.ndarray
        Values on `tvec`: 1-D for a Series, ``(len(tvec), n_columns)`` for a
        frame. Samples outside the source support are NaN.
    """
    from scipy.interpolate import interp1d

    interpolator = interp1d(
        series.index.values,
        series.values,
        kind=kind,
        axis=0,
        bounds_error=False,
        fill_value=np.nan,
    )
    return interpolator(tvec)


def build_continuous_block(
    series: pd.Series, tvec: np.ndarray, lags: np.ndarray | None = None
) -> np.ndarray:
    """Resample a continuous session signal onto the model grid, optionally lagged.

    Parameters
    ----------
    series : pd.Series
        Continuous signal indexed by time (s), e.g. wheel velocity or pose speed.
    tvec : np.ndarray
        Uniform model time grid to resample onto.
    lags : np.ndarray | None
        Integer sample lags. ``None`` (default) returns the resampled signal
        unlagged; otherwise the resampled signal is lag-expanded, one column per
        lag (see `lag_expand`).

    Returns
    -------
    np.ndarray
        1-D of length ``len(tvec)`` when ``lags is None``; otherwise
        ``(len(tvec), len(lags))``.
    """
    resampled = interpolate_to_grid(series, tvec)
    if lags is None:
        return resampled
    return lag_expand(resampled, lags)


def deviation_code(labels: np.ndarray, positive: str) -> np.ndarray:
    """Deviation-code a 2-level categorical to ±0.5.

    Returns +0.5 where ``labels == positive`` and −0.5 elsewhere, so a fitted
    coefficient reads as the contrast between the two levels (the script uses
    this for ``side``/``choice`` relative to the recording hemisphere).

    Parameters
    ----------
    labels : np.ndarray
        Per-event categorical labels.
    positive : str
        Label assigned the +0.5 code.

    Returns
    -------
    np.ndarray
        Float array of ±0.5, same length as `labels`.
    """
    return np.where(np.asarray(labels) == positive, 0.5, -0.5)


def build_event_blocks(
    event_times: np.ndarray,
    tvec: np.ndarray,
    expander: Callable[[np.ndarray], np.ndarray],
    modulators: dict[str, np.ndarray] | None = None,
    interactions: list[tuple[str, ...]] | None = None,
    split: pd.Series | None = None,
    name: str = "",
) -> dict[str, np.ndarray]:
    """Build the kernel blocks for one event term.

    Emits a baseline block (unit-height event train) plus one block per
    modulator and one per interaction, each the event train scaled by per-event
    heights and passed through `expander`. With `split`, the whole block set is
    replicated once per categorical level, each group's blocks firing only at
    that group's events. Stays variable-agnostic: heights arrive already coded
    (continuous mean-centered, categorical deviation-coded ±0.5 by the caller).

    Parameters
    ----------
    event_times : np.ndarray
        Event timestamps in seconds.
    tvec : np.ndarray
        Uniform model time grid.
    expander : Callable[[np.ndarray], np.ndarray]
        Basis expansion applied to a 1-D event train, returning a 2-D block
        (e.g. FIR `lag_expand` or `raised_cosine_expand` with bound parameters).
    modulators : dict[str, np.ndarray], optional
        Maps modulator name to its per-event height array (aligned with
        `event_times`).
    interactions : list[tuple[str, ...]], optional
        Each tuple names modulators whose elementwise product forms an
        interaction block, named by joining the modulator names with ``:``.
    split : pd.Series, optional
        Per-event categorical labels (aligned with `event_times`); its `.name`
        labels the split column. When given, the block set repeats per level.
    name : str
        Event-term prefix for block names.

    Returns
    -------
    dict[str, np.ndarray]
        Ordered ``block name -> (len(tvec), n_basis)`` mapping. Names are
        ``f'{name}|baseline'``, ``f'{name}|{mod}'``, ``f'{name}|{a}:{b}'``; split
        levels prefix the level as ``f'{name}|{split.name}={level}|...'``.
    """
    modulators = modulators or {}
    interactions = interactions or []
    event_times = np.asarray(event_times)

    def group_blocks(mask: np.ndarray, prefix: str) -> dict[str, np.ndarray]:
        """Build the baseline + modulator + interaction blocks for one group."""
        times = event_times[mask]
        blocks = {f"{prefix}|baseline": expander(make_event_regressor(times, tvec))}
        for mod_name, heights in modulators.items():
            block = expander(make_event_regressor(times, tvec, np.asarray(heights)[mask]))
            blocks[f"{prefix}|{mod_name}"] = block
        for terms in interactions:
            product = np.prod([np.asarray(modulators[m]) for m in terms], axis=0)
            block = expander(make_event_regressor(times, tvec, product[mask]))
            blocks[f"{prefix}|{':'.join(terms)}"] = block
        return blocks

    if split is None:
        return group_blocks(np.ones(event_times.size, dtype=bool), name)

    labels = np.asarray(split)
    blocks = {}
    for level in np.unique(labels):
        blocks.update(group_blocks(labels == level, f"{name}|{split.name}={level}"))
    return blocks


@dataclass
class EncodingFit:
    """Result of fitting the ridge encoding model to one session.

    Attributes
    ----------
    tvec : np.ndarray
        Full model time grid (the target's time axis), length n_grid.
    valid : np.ndarray
        Boolean mask over `tvec` of grid samples kept (no NaN in design/target).
    design : np.ndarray
        Z-scored design matrix over the valid rows, shape (n_valid, n_features).
    target : np.ndarray
        Measured signal over the valid rows, shape (n_valid, 1).
    prediction : np.ndarray
        Model prediction over the valid rows, shape (n_valid, 1).
    coefficients : np.ndarray
        Fitted ridge coefficients in z-scored design space, shape
        (n_features, 1). Divide by `scaler.scale_` to back-transform to raw
        signal units (see `get_kernel` / `kernels_to_frame`).
    intercept : np.ndarray
        Fitted ridge intercept.
    slices : dict[str, slice]
        Block name -> column span in `design`.
    scaler : StandardScaler
        Fitted scaler that z-scored the design; its per-column `scale_`
        back-transforms coefficients to raw signal units.
    r2 : float
        In-sample coefficient of determination.
    alpha : float
        Ridge regularisation strength selected by K-fold tuning.
    label : str
        Session label for plots.
    """

    tvec: np.ndarray
    valid: np.ndarray
    design: np.ndarray
    target: np.ndarray
    prediction: np.ndarray
    coefficients: np.ndarray
    intercept: np.ndarray
    slices: dict[str, slice]
    scaler: "StandardScaler"  # noqa: F821 (sklearn imported lazily)
    r2: float
    alpha: float
    label: str = ""

    def get_kernel(self, name: str) -> np.ndarray:
        """Back-transformed coefficients for one block, in raw signal units.

        For the FIR basis these are the kernel itself (one value per lag); for
        the raised-cosine basis they are basis weights. Dividing by the scaler's
        per-column scale undoes the z-scoring applied before the fit.
        """
        span = self.slices[name]
        return self.coefficients[span].flatten() / self.scaler.scale_[span]

    def kernels_to_frame(self) -> pd.DataFrame:
        """Tidy long table of back-transformed kernel coefficients.

        Event blocks (names containing ``'|'``, built by `build_event_blocks`)
        emit one row per lag; continuous/tonic blocks emit a single scalar-coef
        row with NaN ``level``/``lag``/``time``.

        Returns
        -------
        pd.DataFrame
            Columns ``term, level, modulator, lag, time, coef``. ``term``,
            ``level`` and ``modulator`` are parsed from the block name
            (``term|modulator`` or ``term|split=level|modulator``); ``coef`` is
            back-transformed to raw signal units; ``time = lag * dt`` with ``dt``
            read from `tvec`.
        """
        dt = self.tvec[1] - self.tvec[0]
        rows = []
        for name in self.slices:
            coefs = self.get_kernel(name)
            if "|" in name:
                parts = name.split("|")
                level = parts[1].split("=", 1)[1] if len(parts) == 3 else np.nan
                rows += [
                    {"term": parts[0], "level": level, "modulator": parts[-1],
                     "lag": lag, "time": lag * dt, "coef": coef}
                    for lag, coef in enumerate(coefs)
                ]
            else:
                rows.append(
                    {"term": name, "level": np.nan, "modulator": np.nan,
                     "lag": np.nan, "time": np.nan, "coef": coefs[0]}
                )
        return pd.DataFrame(
            rows, columns=["term", "level", "modulator", "lag", "time", "coef"]
        )


def ridge_r2(
    design: np.ndarray, target: np.ndarray, alpha: float, cv: int = None
) -> float:
    """R² of a ridge fit of `target` on `design`, in-sample or cross-validated.

    With ``cv=None`` the model is fit and scored on the same rows (in-sample
    R²). With an integer ``cv`` the score is the pooled out-of-fold R²: each
    fold's held-out predictions are concatenated and scored jointly (one R²
    over all samples), rather than averaging per-fold R². KFold runs without
    shuffling, so folds are contiguous in time.

    Parameters
    ----------
    design : np.ndarray
        Design matrix, shape (n, n_features).
    target : np.ndarray
        Target signal, shape (n, 1).
    alpha : float
        Ridge regularisation strength.
    cv : int, optional
        Number of contiguous KFold splits; ``None`` (default) scores in sample.

    Returns
    -------
    float
        Coefficient of determination.
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.metrics import r2_score

    if cv is None:
        model = Ridge(alpha=alpha).fit(design, target)
        predictions = design @ model.coef_.T + model.intercept_
    else:
        predictions = cross_val_predict(
            Ridge(alpha=alpha), design, target, cv=KFold(cv))
    return r2_score(target, predictions)


def fit_encoding_model(
    design: np.ndarray,
    target: pd.Series,
    slices: dict[str, slice],
    alphas,
    cv: int,
    label: str = "",
) -> EncodingFit:
    """Fit ridge regression of `target` on a prebuilt `design`, tuning alpha.

    Builds nothing — assemble `design`/`slices` with the block helpers plus
    `build_design_matrix`. Predictors are z-scored before fitting so ridge
    penalises columns of different native scales fairly; the fitted scaler is
    stored so coefficients back-transform to raw units. Rows with NaNs
    (interpolation edges / missing support) in design or target are dropped.
    Alpha is tuned over `alphas` by contiguous K-fold, picking the value with
    the best pooled out-of-fold R².

    Parameters
    ----------
    design : np.ndarray
        Design matrix, rows aligned to `target`'s time grid.
    target : pd.Series
        Measured signal indexed by time (s); its index sets `tvec`.
    slices : dict[str, slice]
        Block name -> column span in `design`.
    alphas : sequence of float
        Ridge alpha grid to search.
    cv : int
        Number of contiguous KFold splits for alpha selection.
    label : str
        Session label for plots.

    Returns
    -------
    EncodingFit
        Fitted model over the valid rows, with selected alpha and in-sample R².
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score

    tvec = target.index.values
    y = target.values[:, None]
    valid = ~np.isnan(design).any(axis=1) & ~np.isnan(y).any(axis=1)
    y_valid = y[valid]

    scaler = StandardScaler().fit(design[valid])
    design_scaled = scaler.transform(design[valid])

    alpha = max(alphas, key=lambda a: ridge_r2(design_scaled, y_valid, a, cv))
    model = Ridge(alpha=alpha).fit(design_scaled, y_valid)

    coefficients = model.coef_.T
    prediction = design_scaled @ coefficients + model.intercept_
    return EncodingFit(
        tvec=tvec,
        valid=valid,
        design=design_scaled,
        target=y_valid,
        prediction=prediction,
        coefficients=coefficients,
        intercept=model.intercept_,
        slices=slices,
        scaler=scaler,
        r2=r2_score(y_valid, prediction),
        alpha=alpha,
        label=label,
    )


def fit_measurement_error_varcomp(
    estimates: np.ndarray,
    ses: np.ndarray,
    mouse_ids: np.ndarray,
    *,
    tau_prior: tuple[str, float] = ("halfnormal", 1.0),
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.9,
    random_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the multilevel measurement-error variance-components model.

    Each per-session estimate is a noisy observation of an unknown true effect,
    decomposed into a between-mouse and a between-session (within-mouse)
    component. Standard errors enter as *known* (fixed) measurement noise so
    estimation uncertainty is not counted as real variability — the standard
    multilevel random-effects meta-analysis ("eight schools") model.

    Estimates are standardized internally (``z = (estimates - mean) / std``,
    ``se_z = ses / std``) so the ``tau`` priors are scale-free; returned
    variances are therefore in standardized units.

    Parameters
    ----------
    estimates : 1D array, shape (n_sessions,)
        Per-session coefficient estimates.
    ses : 1D array, shape (n_sessions,)
        Per-session standard errors of ``estimates`` (treated as known).
    mouse_ids : 1D array, shape (n_sessions,)
        Mouse identifier per session; any hashable labels, mapped to contiguous
        integer indices internally.
    tau_prior : tuple of (str, float)
        Prior family and scale for the ``tau`` standard-deviation parameters.
        Family is one of ``'halfnormal'``, ``'halfcauchy'``, ``'uniform'``
        (``Uniform(0, scale)``).
    draws, tune, chains : int
        MCMC draws kept, tuning steps, and chains.
    target_accept : float
        NUTS target acceptance probability.
    random_seed : int
        Seed for reproducible sampling.

    Returns
    -------
    v_mouse : 1D array, shape (chains * draws,)
        Posterior draws of the between-mouse variance ``tau_mouse**2``.
    v_session : 1D array, shape (chains * draws,)
        Posterior draws of the between-session variance ``tau_session**2``.
    """
    import pymc as pm

    estimates = np.asarray(estimates, dtype=float)
    ses = np.asarray(ses, dtype=float)
    scale = estimates.std()
    z = (estimates - estimates.mean()) / scale
    se_z = ses / scale

    _, mouse_idx = np.unique(mouse_ids, return_inverse=True)
    n_mice = int(mouse_idx.max() + 1)
    n_sessions = int(z.size)

    prior = _tau_prior_factory(tau_prior)
    with pm.Model():
        mu = pm.Normal("mu", 0, 1)
        tau_mouse = prior("tau_mouse")
        tau_session = prior("tau_session")
        b_mouse = pm.Normal("b_mouse", 0, tau_mouse, shape=n_mice)
        c_session = pm.Normal("c_session", 0, tau_session, shape=n_sessions)
        theta = mu + b_mouse[mouse_idx] + c_session
        pm.Normal("obs", theta, se_z, observed=z)
        pm.Deterministic("V_mouse", tau_mouse**2)
        pm.Deterministic("V_session", tau_session**2)
        idata = pm.sample(
            draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=False,
        )

    v_mouse = idata.posterior["V_mouse"].values.ravel()
    v_session = idata.posterior["V_session"].values.ravel()
    return v_mouse, v_session


def _tau_prior_factory(tau_prior: tuple[str, float]) -> Callable:
    """Return a ``name -> pm.Distribution`` builder for the chosen tau prior."""
    import pymc as pm

    family, scale = tau_prior
    builders = {
        "halfnormal": lambda name: pm.HalfNormal(name, scale),
        "halfcauchy": lambda name: pm.HalfCauchy(name, scale),
        "uniform": lambda name: pm.Uniform(name, 0, scale),
    }
    if family not in builders:
        raise ValueError(f"Unknown tau prior family: {family!r}")
    return builders[family]


def summarize_posterior(
    samples: np.ndarray,
    *,
    grid_size: int = 200,
    hdi_prob: float = 0.94,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """Reduce a 1-D posterior sample vector to a summary and a KDE outline.

    Parameters
    ----------
    samples : 1D array
        Posterior draws of one quantity.
    grid_size : int
        Number of points in the KDE evaluation grid.
    hdi_prob : float
        Highest-density-interval mass (e.g. 0.94).

    Returns
    -------
    mean : float
        Posterior mean.
    hdi_low, hdi_high : float
        Bounds of the ``hdi_prob`` highest-density interval.
    x_grid : 1D array, shape (grid_size,)
        Grid from 0 to ``samples.max()`` on which the density is evaluated.
    density : 1D array, shape (grid_size,)
        Gaussian-KDE density (the violin outline) on ``x_grid``.
    """
    import arviz as az

    samples = np.asarray(samples, dtype=float)
    mean = float(samples.mean())
    hdi_low, hdi_high = (float(x) for x in az.hdi(samples, prob=hdi_prob))
    x_grid = np.linspace(0, samples.max(), grid_size)
    density = gaussian_kde(samples)(x_grid)
    return mean, hdi_low, hdi_high, x_grid, density
