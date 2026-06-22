import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

from iblnm.config import (
    TARGET_FS,
    POSE_FS,
    LIKELIHOOD_THRESHOLD,
    MOVEMENT_RESPONSE_WINDOW,
    BASELINE_WINDOW,
    CROSSCORR_FS,
    CROSSCORR_LAG_WINDOW,
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


# =============================================================================
# GLM Response Features
# =============================================================================


class GLMPCAResult:
    """Container for PCA on GLM coefficients.

    Attributes
    ----------
    scores : np.ndarray
        (n_recordings, n_components) PC scores from projecting standardized
        (unweighted) data onto weighted-PCA components.
    components : np.ndarray
        (n_components, n_features) principal component loadings.
    explained_variance_ratio : np.ndarray
        (n_components,) fraction of weighted variance explained per PC.
    feature_names : list[str]
        Coefficient names (columns of the input matrix, minus intercept).
    target_labels : np.ndarray
        (n_recordings,) target_NM for each recording.
    index : pd.MultiIndex
        Original DataFrame index.
    """

    def __init__(self, scores, components, explained_variance_ratio,
                 feature_names, target_labels, index):
        self.scores = scores
        self.components = components
        self.explained_variance_ratio = explained_variance_ratio
        self.feature_names = feature_names
        self.target_labels = target_labels
        self.index = index


def pca_glm_coefficients(coefs, n_components=3, cohort_weighted=False):
    """PCA on per-session GLM coefficient vectors.

    Drops the intercept column if present, standardizes each coefficient
    (z-score across sessions), then fits PCA. When ``cohort_weighted=True``,
    each target_NM contributes equally to the covariance matrix regardless
    of sample size (weight per session = 1/n_k). Scores are always computed
    by projecting the unweighted standardized data onto the PCs.

    Parameters
    ----------
    coefs : pd.DataFrame
        (n_recordings, n_coefficients) indexed by ``(eid, target_NM, fiber_idx)``.
    n_components : int
        Number of principal components to retain.
    cohort_weighted : bool
        If True, weight sessions by 1/n_k so each target_NM contributes
        equally. If False, standard (unweighted) PCA.

    Returns
    -------
    GLMPCAResult
    """
    # Drop intercept if present
    if 'intercept' in coefs.columns:
        coefs = coefs.drop(columns='intercept')

    feature_names = list(coefs.columns)
    target_labels = coefs.index.get_level_values('target_NM').values

    X = coefs.values.astype(float)

    # Standardize each feature
    mu = np.nanmean(X, axis=0)
    sigma = np.nanstd(X, axis=0)
    sigma[sigma == 0] = 1.0
    X_std = (X - mu) / sigma

    if cohort_weighted:
        # Cohort weights: 1/n_k per session in cohort k
        unique_targets, counts = np.unique(target_labels, return_counts=True)
        size_map = dict(zip(unique_targets, counts))
        weights = np.array([1.0 / size_map[t] for t in target_labels])
        # Normalize so weights sum to n_recordings (PCA scale invariance)
        weights = weights * len(weights) / weights.sum()
        sqrt_w = np.sqrt(weights)[:, np.newaxis]
        X_for_svd = X_std * sqrt_w
    else:
        X_for_svd = X_std

    # SVD
    U, S, Vt = np.linalg.svd(X_for_svd, full_matrices=False)
    total_var = np.sum(S ** 2)
    n_components = min(n_components, len(S))

    components = Vt[:n_components]
    explained_variance_ratio = (S[:n_components] ** 2) / total_var

    # Project unweighted standardized data onto PCs
    scores = X_std @ components.T

    return GLMPCAResult(
        scores=scores,
        components=components,
        explained_variance_ratio=explained_variance_ratio,
        feature_names=feature_names,
        target_labels=target_labels,
        index=coefs.index,
    )


def ica_glm_coefficients(coefs, n_components=3, cohort_weighted=False):
    """ICA on per-session GLM coefficient vectors.

    Drops the intercept column if present, standardizes each coefficient
    (z-score across sessions), then fits FastICA. Components are sorted by
    descending post-hoc variance explained (fraction of total variance in the
    standardized data captured by each IC's projection).

    Parameters
    ----------
    coefs : pd.DataFrame
        (n_recordings, n_coefficients) indexed by ``(eid, target_NM, fiber_idx)``.
    n_components : int
        Number of independent components to extract.
    cohort_weighted : bool
        If True, weight sessions by 1/n_k so each target_NM contributes
        equally. Scores are computed by projecting unweighted data.

    Returns
    -------
    GLMPCAResult
        Same container as PCA, with post-hoc variance explained per IC.
    """
    from sklearn.decomposition import FastICA

    if 'intercept' in coefs.columns:
        coefs = coefs.drop(columns='intercept')

    feature_names = list(coefs.columns)
    target_labels = coefs.index.get_level_values('target_NM').values

    X = coefs.values.astype(float)

    # Standardize each feature
    mu = np.nanmean(X, axis=0)
    sigma = np.nanstd(X, axis=0)
    sigma[sigma == 0] = 1.0
    X_std = (X - mu) / sigma

    if cohort_weighted:
        unique_targets, counts = np.unique(target_labels, return_counts=True)
        size_map = dict(zip(unique_targets, counts))
        weights = np.array([1.0 / size_map[t] for t in target_labels])
        weights = weights * len(weights) / weights.sum()
        sqrt_w = np.sqrt(weights)[:, np.newaxis]
        X_for_ica = X_std * sqrt_w
    else:
        X_for_ica = X_std

    n_components = min(n_components, X_std.shape[1])
    ica = FastICA(n_components=n_components, random_state=0, max_iter=1000)
    ica.fit(X_for_ica)
    components = ica.components_

    # Project unweighted standardized data onto ICs
    scores = X_std @ np.linalg.pinv(components)

    # Post-hoc variance explained per IC
    total_var = np.sum(np.var(X_std, axis=0))
    var_per_ic = np.array([np.var(scores[:, i]) * np.sum(components[i] ** 2)
                           for i in range(n_components)])
    explained_variance_ratio = var_per_ic / total_var

    # Sort by descending variance explained
    order = np.argsort(-explained_variance_ratio)
    scores = scores[:, order]
    components = components[order]
    explained_variance_ratio = explained_variance_ratio[order]

    return GLMPCAResult(
        scores=scores,
        components=components,
        explained_variance_ratio=explained_variance_ratio,
        feature_names=feature_names,
        target_labels=target_labels,
        index=coefs.index,
    )


def pca_score_stats(pca_result):
    """Kruskal-Wallis and pairwise Mann-Whitney U tests on PC scores.

    For each PC, runs a Kruskal-Wallis omnibus test across all targets,
    then pairwise Mann-Whitney U tests for every target pair.

    Parameters
    ----------
    pca_result : GLMPCAResult

    Returns
    -------
    pd.DataFrame
        One row per (PC, test). Kruskal-Wallis rows have ``target_a``
        and ``target_b`` as NaN; pairwise rows have both filled.
        Columns: pc, kruskal_h, kruskal_p, target_a, target_b, mwu_u, mwu_p.
    """
    from itertools import combinations
    from scipy.stats import kruskal, mannwhitneyu

    targets = sorted(set(pca_result.target_labels))
    n_pcs = pca_result.scores.shape[1]
    rows = []

    for pc_idx in range(n_pcs):
        scores = pca_result.scores[:, pc_idx]
        groups = {t: scores[pca_result.target_labels == t] for t in targets}

        # Kruskal-Wallis omnibus
        group_arrays = [g for g in groups.values() if len(g) >= 2]
        if len(group_arrays) >= 2:
            h, p = kruskal(*group_arrays)
        else:
            h, p = np.nan, np.nan
        rows.append({
            'pc': pc_idx + 1, 'kruskal_h': h, 'kruskal_p': p,
            'target_a': None, 'target_b': None,
            'mwu_u': np.nan, 'mwu_p': np.nan,
        })

        # Pairwise Mann-Whitney U
        for ta, tb in combinations(targets, 2):
            ga, gb = groups[ta], groups[tb]
            if len(ga) >= 2 and len(gb) >= 2:
                u, mp = mannwhitneyu(ga, gb, alternative='two-sided')
            else:
                u, mp = np.nan, np.nan
            rows.append({
                'pc': pc_idx + 1, 'kruskal_h': h, 'kruskal_p': p,
                'target_a': ta, 'target_b': tb,
                'mwu_u': u, 'mwu_p': mp,
            })

    return pd.DataFrame(rows)


def pca_subject_score_stats(pca_result, recordings):
    """Subject-level KW and pairwise Mann-Whitney U on PC scores.

    Averages scores per subject before testing, so each subject contributes
    one observation regardless of how many recordings it has.

    Parameters
    ----------
    pca_result : GLMPCAResult
    recordings : pd.DataFrame
        Must contain 'eid' and 'subject' columns to map recordings to subjects.

    Returns
    -------
    subject_means : pd.DataFrame
        Columns: subject, target_NM, pc, mean, sem.
    stats : pd.DataFrame
        Same schema as ``pca_score_stats`` output.
    """
    from itertools import combinations
    from scipy.stats import kruskal, mannwhitneyu

    # Map eid → subject
    eid_to_subject = recordings.drop_duplicates('eid').set_index('eid')['subject']
    eids = pca_result.index.get_level_values('eid')
    subjects = eid_to_subject.reindex(eids).values

    n_pcs = pca_result.scores.shape[1]
    targets = sorted(set(pca_result.target_labels))

    # Build per-recording long table, then aggregate
    df = pd.DataFrame({
        'subject': subjects,
        'target_NM': pca_result.target_labels,
    })
    for pc_idx in range(n_pcs):
        df[f'pc{pc_idx + 1}'] = pca_result.scores[:, pc_idx]

    # Subject means and SEM
    mean_rows = []
    for (subj, tnm), g in df.groupby(['subject', 'target_NM']):
        for pc_idx in range(n_pcs):
            col = f'pc{pc_idx + 1}'
            mean_rows.append({
                'subject': subj,
                'target_NM': tnm,
                'pc': pc_idx + 1,
                'mean': g[col].mean(),
                'sem': g[col].sem() if len(g) > 1 else 0.0,
            })
    subject_means = pd.DataFrame(mean_rows)

    # Stats on subject means
    stat_rows = []
    for pc_idx in range(n_pcs):
        pc_num = pc_idx + 1
        sm = subject_means[subject_means['pc'] == pc_num]
        groups = {t: sm.loc[sm['target_NM'] == t, 'mean'].values
                  for t in targets}

        group_arrays = [g for g in groups.values() if len(g) >= 2]
        if len(group_arrays) >= 2:
            h, p = kruskal(*group_arrays)
        else:
            h, p = np.nan, np.nan
        stat_rows.append({
            'pc': pc_num, 'kruskal_h': h, 'kruskal_p': p,
            'target_a': None, 'target_b': None,
            'mwu_u': np.nan, 'mwu_p': np.nan,
        })

        for ta, tb in combinations(targets, 2):
            ga, gb = groups[ta], groups[tb]
            if len(ga) >= 2 and len(gb) >= 2:
                u, mp = mannwhitneyu(ga, gb, alternative='two-sided')
            else:
                u, mp = np.nan, np.nan
            stat_rows.append({
                'pc': pc_num, 'kruskal_h': h, 'kruskal_p': p,
                'target_a': ta, 'target_b': tb,
                'mwu_u': u, 'mwu_p': mp,
            })

    stats = pd.DataFrame(stat_rows)
    return subject_means, stats


def fit_response_glm(events, event_name, min_trials=20, contrast_coding='log',
                     response_col='response'):
    """Fit per-recording OLS models and return coefficients as features.

    For each recording in the events DataFrame, fits:
    ``response ~ 1 + contrast + side + reward
    + contrast:side + contrast:reward + side:reward``

    The contrast predictor is mean-centered per recording. Uses deviation
    coding (±0.5) for side and reward, consistent with the LMM:
        side:   contra = +0.5, ipsi = −0.5
        reward: correct = +0.5, incorrect = −0.5

    Only unbiased-block trials (probabilityLeft == 0.5) are used.

    Parameters
    ----------
    events : pd.DataFrame
        Trial-level data from ``get_response_magnitudes`` with ``add_relative_contrast``
        applied (must have ``side`` column). Required columns: ``eid``,
        ``brain_region``, ``target_NM``, ``event``, ``contrast``, ``side``,
        ``feedbackType``, ``probabilityLeft``, ``response_col``.
    event_name : str
        Event to model (e.g., ``'stimOn_times'``).
    min_trials : int
        Minimum valid trials per recording. Recordings with fewer are skipped.
    response_col : str
        Column holding the response magnitude (default ``'response'``).

    Returns
    -------
    coefs : pd.DataFrame
        (n_recordings, 7) coefficient values.
    ses : pd.DataFrame
        (n_recordings, 7) standard errors.
    """
    from iblnm.task import add_relative_contrast

    _transform, _ = get_contrast_coding(contrast_coding)

    if events is None or len(events) == 0:
        raise ValueError("events DataFrame is empty")

    available_events = events['event'].unique()
    if event_name not in available_events:
        raise ValueError(
            f"Event '{event_name}' not found in events. "
            f"Available: {list(available_events)}"
        )

    # FIXME: these filters need to go in the method that calls this fitting function
    # Filter to requested event and unbiased blocks
    df = events[events['event'] == event_name].copy()
    df = df[df['probabilityLeft'] == 0.5]

    # Add lateralized side column if not present
    if 'side' not in df.columns:
        df = add_relative_contrast(df)

    coef_names = [
        'intercept', 'contrast', 'side', 'reward',
        'contrast:side', 'contrast:reward', 'side:reward',
    ]

    coef_rows = {}
    se_rows = {}

    # FIXME: this loop should be in the group method, not in the function
    for (eid, brain_region), grp in df.groupby(['eid', 'brain_region']):
        # Drop NaN responses
        valid = grp.dropna(subset=[response_col])
        if len(valid) < min_trials:
            continue

        y = valid[response_col].values
        coded_c = _transform(valid['contrast'].values)
        coded_c = coded_c - coded_c.mean()
        # Deviation coding ±0.5, matching fit_response_lmm
        side = np.where(valid['side'] == 'contra', 0.5, -0.5)
        reward = np.where(valid['feedbackType'] == 1, 0.5, -0.5)

        X = np.column_stack([
            np.ones(len(y)),
            coded_c,
            side,
            reward,
            coded_c * side,
            coded_c * reward,
            side * reward,
        ])

        # Check rank
        if np.linalg.matrix_rank(X) < X.shape[1]:
            print(
                f"  Singular design matrix for {eid}/{brain_region}, skipping"
            )
            continue

        # OLS via lstsq
        beta, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)

        # Compute standard errors
        n, p = X.shape
        y_hat = X @ beta
        resid = y - y_hat
        sigma2 = np.sum(resid ** 2) / (n - p)
        XtX_inv = np.linalg.inv(X.T @ X)
        se = np.sqrt(sigma2 * np.diag(XtX_inv))

        # Get target_NM and fiber_idx from this group
        target_nm = grp['target_NM'].iloc[0]
        fiber_idx = int(grp['fiber_idx'].iloc[0]) if 'fiber_idx' in grp.columns else 0

        key = (eid, target_nm, brain_region, fiber_idx)
        coef_rows[key] = dict(zip(coef_names, beta))
        se_rows[key] = dict(zip(coef_names, se))

    if not coef_rows:
        empty = pd.DataFrame(columns=coef_names)
        return empty, empty.copy()

    coefs = pd.DataFrame(coef_rows).T
    coefs.index = pd.MultiIndex.from_tuples(
        coefs.index, names=['eid', 'target_NM', 'brain_region', 'fiber_idx'],
    )
    ses = pd.DataFrame(se_rows).T
    ses.index = coefs.index

    return coefs, ses


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
    """

    def __init__(self, x_weights, y_weights, x_scores, y_scores,
                 correlations, p_values, n_recordings, n_permutations,
                 alpha=None, l1_ratio=None):
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
    )


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


def _movement_formulas(response_col, timing_col):
    """The three nested additive movement models, the single source of the
    model specification shared by the full-data, jackknife, and LOSO-CV
    routines. Side is intentionally excluded (see ``_code_movement_predictors``)."""
    return {
        'full': f'{response_col} ~ contrast + {timing_col} + reward',
        'drop_contrast': f'{response_col} ~ {timing_col} + reward',
        'drop_movement': f'{response_col} ~ contrast + reward',
    }


def _code_movement_predictors(df, timing_col):
    """log2-code and mean-center contrast; deviation-code reward (±0.5).

    Side is omitted on purpose: choice-side bias is idiosyncratic per animal,
    not a consistent population-level confound, whereas reward is (correct
    trials are systematically more vigorous across animals). The timing column
    is assumed already log-transformed.
    """
    transform, _ = get_contrast_coding('log2')
    df = df.copy()
    coded = transform(df['contrast'])
    df['contrast'] = coded - float(np.mean(coded))
    df['reward'] = np.where(df['feedbackType'] == 1, 0.5, -0.5)
    return df


def fit_movement_lmm_r2(df, response_col, timing_col):
    """Fit the three nested additive movement LMMs on ``df``; return in-sample
    marginal (fixed-effects) R² and the drop-one deltas.

    Models share a random intercept ``(1 | subject)``:

    - Full: ``response ~ contrast + timing + reward``
    - Drop-contrast: ``response ~ timing + reward``
    - Drop-movement: ``response ~ contrast + reward``

    ``delta_r2_contrast`` / ``delta_r2_timing`` are the marginal R² lost when
    contrast / the movement variable are removed from the full model. The
    marginal R² uses a fixed empirical denominator (``var(observed y)``), so each
    delta is a unique (semipartial) R²; it can be negative when a predictor does
    not help. This is the general-purpose kernel: ``jackknife_movement_lmm``
    calls it on leave-one-subject-out subsets, and the pipeline calls it on the
    full dataset for the absolute-R² bars.

    Parameters
    ----------
    df : pd.DataFrame
        Trial-level data with columns: contrast, feedbackType, subject,
        ``response_col``, and ``timing_col`` (already log-transformed).
    response_col : str
        Column name for the NM response magnitude.
    timing_col : str
        Column name for the (log-transformed) movement variable.

    Returns
    -------
    dict or None
        Keys: r2_full, r2_drop_contrast, r2_drop_movement, delta_r2_contrast,
        delta_r2_timing. None if fewer than two subjects or any fit fails.
    """
    df = _code_movement_predictors(
        df.dropna(subset=[response_col, timing_col]), timing_col)
    if df['subject'].nunique() < 2:
        return None

    fits = {
        name: fit_lmm(formula, df, groups=df['subject'], re_formula='1',
                       reml=False)
        for name, formula in _movement_formulas(response_col, timing_col).items()
    }
    if any(fit is None for fit in fits.values()):
        return None

    r2 = {name: fit.variance_explained['marginal'] for name, fit in fits.items()}
    return {
        'r2_full': r2['full'],
        'r2_drop_contrast': r2['drop_contrast'],
        'r2_drop_movement': r2['drop_movement'],
        'delta_r2_contrast': r2['full'] - r2['drop_contrast'],
        'delta_r2_timing': r2['full'] - r2['drop_movement'],
    }


def jackknife_movement_lmm(df, response_col, timing_col, min_subjects=3):
    """Leave-one-subject-out jackknife of the movement-model marginal R².

    For each subject, refits the three models (``fit_movement_lmm_r2``) on the
    *remaining* subjects and records the in-sample marginal R² and drop-one
    deltas. The spread across subjects shows whether any single animal drives
    the result.

    Parameters
    ----------
    df : pd.DataFrame
        Trial-level data (see ``fit_movement_lmm_r2``).
    response_col : str
        Column name for the NM response magnitude.
    timing_col : str
        Column name for the (log-transformed) movement variable.
    min_subjects : int
        Minimum subjects required to run the jackknife.

    Returns
    -------
    pd.DataFrame
        One row per left-out subject with columns: subject, r2_full,
        r2_drop_contrast, r2_drop_movement, delta_r2_contrast, delta_r2_timing,
        timing_col. Empty DataFrame if fewer than min_subjects.
    """
    cols = ['subject', 'r2_full', 'r2_drop_contrast', 'r2_drop_movement',
            'delta_r2_contrast', 'delta_r2_timing', 'timing_col']
    df = df.dropna(subset=[response_col, timing_col])
    subjects = df['subject'].unique()
    if len(subjects) < min_subjects:
        return pd.DataFrame(columns=cols)

    rows = []
    for held_out in subjects:
        res = fit_movement_lmm_r2(df[df['subject'] != held_out],
                                  response_col, timing_col)
        if res is None:
            continue
        rows.append({'subject': held_out, **res, 'timing_col': timing_col})

    return pd.DataFrame(rows, columns=cols)


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


def loso_cv_movement_lmm(df, response_col, timing_col, min_subjects=3):
    """Leave-one-subject-out cross-validation of the movement models.

    Same three additive models as ``fit_movement_lmm_r2``, but each fold fits
    on N−1 subjects and scores the *held-out* subject out-of-sample, after
    removing that subject's own intercept (centering both responses and the
    fixed-effects prediction on the subject's mean). This tests whether the
    population fixed-effect slopes generalize to a new animal, rather than how
    well they describe the training set; use ``jackknife_movement_lmm`` for the
    in-sample influence view.

    Parameters
    ----------
    df : pd.DataFrame
        Trial-level data (see ``fit_movement_lmm_r2``).
    response_col : str
        Column name for the NM response magnitude.
    timing_col : str
        Column name for the (log-transformed) movement variable.
    min_subjects : int
        Minimum subjects required (≥ 3 so training sets keep ≥ 2 subjects).

    Returns
    -------
    pd.DataFrame
        One row per held-out subject with columns: subject, n_trials, r2_full,
        r2_drop_contrast, r2_drop_movement, delta_r2_contrast, delta_r2_timing,
        timing_col. Empty DataFrame if fewer than min_subjects.
    """
    cols = ['subject', 'n_trials', 'r2_full', 'r2_drop_contrast',
            'r2_drop_movement', 'delta_r2_contrast', 'delta_r2_timing',
            'timing_col']
    df = _code_movement_predictors(
        df.dropna(subset=[response_col, timing_col]), timing_col)
    subjects = df['subject'].unique()
    if len(subjects) < min_subjects:
        return pd.DataFrame(columns=cols)

    formulas = _movement_formulas(response_col, timing_col)

    rows = []
    for held_out in subjects:
        df_train = df[df['subject'] != held_out]
        df_test = df[df['subject'] == held_out]
        if len(df_test) < 5 or df_train['subject'].nunique() < 2:
            continue

        fits = {name: fit_lmm(formula, df_train, groups=df_train['subject'],
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
        rows.append({
            'subject': held_out,
            'n_trials': len(df_test),
            'r2_full': r2['full'],
            'r2_drop_contrast': r2['drop_contrast'],
            'r2_drop_movement': r2['drop_movement'],
            'delta_r2_contrast': r2['full'] - r2['drop_contrast'],
            'delta_r2_timing': r2['full'] - r2['drop_movement'],
            'timing_col': timing_col,
        })

    return pd.DataFrame(rows, columns=cols)


def _code_task_predictors(df, response_col, contrast_coding):
    """Drop null responses and code task predictors for the LOSO models:
    contrast transformed then mean-centered, side and reward deviation-coded
    (±0.5). Shared by the interaction and main-effect LOSO routines."""
    transform, _ = get_contrast_coding(contrast_coding)
    df = df.dropna(subset=[response_col]).copy()
    coded = transform(df['contrast'])
    df['contrast'] = coded - float(np.mean(coded))
    df['side'] = np.where(df['side'] == 'contra', 0.5, -0.5)
    df['reward'] = np.where(df['feedbackType'] == 1, 0.5, -0.5)
    return df


def loso_cv_task_lmm(df, response_col, contrast_coding='log2', min_subjects=3):
    """Leave-one-subject-out cross-validation of the task model.

    Full model = ``response ~ contrast * side * reward``; reduced =
    ``response ~ contrast + side + reward`` (no interactions). Both share a
    random intercept ``(1 | subject)`` and are fit with ML (so the two
    fixed-effects designs are on a comparable scale). Each fold fits on N−1
    subjects and scores the held-out subject out of sample, centering out that
    subject's own intercept (``_centered_r2``). Comparing full vs reduced
    out-of-sample R² tests whether the interaction structure generalizes to a
    new animal — the substitute for random interaction slopes, which the
    subject count cannot support.

    Predictors are coded as in ``fit_response_lmm``: contrast via
    ``contrast_coding`` then mean-centered, side and reward deviation-coded
    (±0.5). Contrast is centered once on the full data before splitting, so
    every fold shares the same coding.

    At 6–11 subjects (one fold each) the CV estimate is noisy and should be
    read qualitatively, not as a precise generalization score.

    Parameters
    ----------
    df : pd.DataFrame
        Trial-level data with columns ``contrast``, ``side``, ``feedbackType``,
        ``subject``, and ``response_col``.
    response_col : str
        Column name for the response magnitude.
    contrast_coding : str
        Continuous contrast coding passed to ``get_contrast_coding``.
    min_subjects : int
        Minimum subjects required (≥ 3 so training sets keep ≥ 2 subjects).

    Returns
    -------
    pd.DataFrame
        One row per held-out subject (``subject``, ``n_trials``, ``r2_full``,
        ``r2_reduced``, ``delta_r2``) plus a final ``subject == 'aggregate'``
        row holding the across-fold means and total trial count. Empty
        DataFrame if fewer than ``min_subjects``.
    """
    cols = ['subject', 'n_trials', 'r2_full', 'r2_reduced', 'delta_r2']
    df = _code_task_predictors(df, response_col, contrast_coding)

    subjects = df['subject'].unique()
    if len(subjects) < min_subjects:
        return pd.DataFrame(columns=cols)

    formulas = {
        'full': f'{response_col} ~ contrast * side * reward',
        'reduced': f'{response_col} ~ contrast + side + reward',
    }

    rows = []
    for held_out in subjects:
        df_train = df[df['subject'] != held_out]
        df_test = df[df['subject'] == held_out]
        if len(df_test) < 5 or df_train['subject'].nunique() < 2:
            continue

        fits = {name: fit_lmm(formula, df_train, groups=df_train['subject'],
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
        rows.append({
            'subject': held_out,
            'n_trials': len(df_test),
            'r2_full': r2['full'],
            'r2_reduced': r2['reduced'],
            'delta_r2': r2['full'] - r2['reduced'],
        })

    if not rows:
        return pd.DataFrame(columns=cols)

    rows.append({
        'subject': 'aggregate',
        'n_trials': sum(r['n_trials'] for r in rows),
        'r2_full': float(np.mean([r['r2_full'] for r in rows])),
        'r2_reduced': float(np.mean([r['r2_reduced'] for r in rows])),
        'delta_r2': float(np.mean([r['delta_r2'] for r in rows])),
    })
    return pd.DataFrame(rows, columns=cols)


def loso_cv_main_effects_lmm(df, response_col, contrast_coding='log2',
                             min_subjects=3):
    """Leave-one-subject-out reliability of each additive main effect.

    Companion to ``loso_cv_task_lmm``, applying the same out-of-sample logic to
    the main effects instead of the interactions. The baseline is the additive
    model ``response ~ contrast + side + reward``; for each main effect, a
    reduced model drops just that term. Each fold fits both on N−1 subjects and
    scores the held-out subject out of sample (``_centered_r2``); the per-term
    ΔR² = R²(additive) − R²(additive without that term) is the out-of-sample
    variance that main effect uniquely carries to a new animal. This is the
    generalization counterpart to the random-slope coefficient estimates, on the
    same axis as the interaction LOSO.

    Predictors are coded as in ``loso_cv_task_lmm`` (contrast transformed then
    mean-centered, side and reward deviation-coded ±0.5), all share a random
    intercept ``(1 | subject)``, and are fit with ML.

    Parameters
    ----------
    df : pd.DataFrame
        Trial-level data with columns ``contrast``, ``side``, ``feedbackType``,
        ``subject``, and ``response_col``.
    response_col : str
        Column name for the response magnitude.
    contrast_coding : str
        Continuous contrast coding passed to ``get_contrast_coding``.
    min_subjects : int
        Minimum subjects required (≥ 3 so training sets keep ≥ 2 subjects).

    Returns
    -------
    pd.DataFrame
        One row per (held-out subject × main effect) with columns ``subject``,
        ``predictor``, ``n_trials``, ``r2_additive``, ``r2_drop``, ``delta_r2``,
        plus a ``subject == 'aggregate'`` row per predictor holding across-fold
        means. Empty DataFrame if fewer than ``min_subjects``.
    """
    cols = ['subject', 'predictor', 'n_trials', 'r2_additive', 'r2_drop',
            'delta_r2']
    df = _code_task_predictors(df, response_col, contrast_coding)

    subjects = df['subject'].unique()
    if len(subjects) < min_subjects:
        return pd.DataFrame(columns=cols)

    predictors = ['contrast', 'side', 'reward']
    additive = f'{response_col} ~ contrast + side + reward'
    drop_formulas = {
        p: f'{response_col} ~ ' + ' + '.join(q for q in predictors if q != p)
        for p in predictors
    }

    rows = []
    for held_out in subjects:
        df_train = df[df['subject'] != held_out]
        df_test = df[df['subject'] == held_out]
        if len(df_test) < 5 or df_train['subject'].nunique() < 2:
            continue

        formulas = {'additive': additive, **drop_formulas}
        fits = {name: fit_lmm(formula, df_train, groups=df_train['subject'],
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
        for p in predictors:
            rows.append({
                'subject': held_out,
                'predictor': p,
                'n_trials': len(df_test),
                'r2_additive': r2['additive'],
                'r2_drop': r2[p],
                'delta_r2': r2['additive'] - r2[p],
            })

    if not rows:
        return pd.DataFrame(columns=cols)

    fold_rows = pd.DataFrame(rows, columns=cols)
    aggregate = (fold_rows.groupby('predictor', sort=False)
                 .agg(n_trials=('n_trials', 'sum'),
                      r2_additive=('r2_additive', 'mean'),
                      r2_drop=('r2_drop', 'mean'),
                      delta_r2=('delta_r2', 'mean'))
                 .reset_index())
    aggregate.insert(0, 'subject', 'aggregate')
    return pd.concat([fold_rows, aggregate], ignore_index=True)


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


_TIDY_LMM_COLS = ['term', 'coef', 'se', 'z', 'p', 'ci_low', 'ci_high',
                  'marginal_r2', 'n_trials', 'n_subjects', 'timing_col']


def _empty_tidy_lmm() -> pd.DataFrame:
    """Empty tidy frame with the standard movement-LMM result columns, returned
    when a group has too few subjects or the fit fails."""
    return pd.DataFrame(columns=_TIDY_LMM_COLS)


def _tidy_lmm_row(lmm: 'LMMResult', term: str, df: pd.DataFrame,
                  timing_col: str) -> pd.DataFrame:
    """One-row tidy frame for the fixed-effect ``term`` of a fitted ``lmm``.

    ``term`` is the coefficient reported (the contrast or timing slope);
    ``timing_col`` is the timing-variable identifier carried on every row.
    """
    res = lmm.result
    ci = res.conf_int().loc[term]
    return pd.DataFrame([{
        'term': term,
        'coef': float(res.fe_params[term]),
        'se': float(res.bse_fe[term]),
        'z': float(res.tvalues[term]),
        'p': float(res.pvalues[term]),
        'ci_low': float(ci.iloc[0]),
        'ci_high': float(ci.iloc[1]),
        'marginal_r2': lmm.variance_explained['marginal'],
        'n_trials': len(df),
        'n_subjects': df['subject'].nunique(),
        'timing_col': timing_col,
    }], columns=_TIDY_LMM_COLS)


def fit_movement_vs_contrast(df, timing_col, min_subjects=2):
    """Movement-vs-contrast claim: does the timing variable track contrast?

    Marginal model ``{timing_col} ~ contrast`` with a by-subject random slope
    ``(1 + contrast | subject)``; no side/reward nuisance terms. Contrast is
    log2-floored and mean-centered (``_code_movement_predictors``). Reports the
    contrast slope. ``df`` is one ``(target_NM, timing_var)`` subset.

    Returns a one-row tidy frame (``_TIDY_LMM_COLS``), empty if fewer than
    ``min_subjects`` subjects or the fit fails.
    """
    df = _code_movement_predictors(df.dropna(subset=[timing_col]), timing_col)
    if df['subject'].nunique() < min_subjects:
        return _empty_tidy_lmm()
    lmm = fit_lmm(f'{timing_col} ~ contrast', df, groups=df['subject'],
                   re_formula='1 + contrast', reml=True)
    if lmm is None:
        return _empty_tidy_lmm()
    return _tidy_lmm_row(lmm, 'contrast', df, timing_col)


def fit_movement_predicts_response(df, response_col, timing_col, min_subjects=2):
    """Unadjusted movement-predicts-response claim (deliberately
    contrast-confounded).

    Model ``{response_col} ~ {timing_col}`` with a by-subject random slope
    ``(1 + timing | subject)``. No contrast control — establishes the gross
    phenomenon before the within-contrast model. ``df`` is one
    ``(target_NM, event, timing_var)`` subset. Reports the timing slope.

    Returns a one-row tidy frame (``_TIDY_LMM_COLS``), empty if fewer than
    ``min_subjects`` subjects or the fit fails.
    """
    df = df.dropna(subset=[response_col, timing_col]).copy()
    if df['subject'].nunique() < min_subjects:
        return _empty_tidy_lmm()
    lmm = fit_lmm(f'{response_col} ~ {timing_col}', df, groups=df['subject'],
                   re_formula=f'1 + {timing_col}', reml=True)
    if lmm is None:
        return _empty_tidy_lmm()
    return _tidy_lmm_row(lmm, timing_col, df, timing_col)


def fit_movement_within_contrast(df, response_col, timing_col, min_subjects=2):
    """Within-contrast movement-predicts-response claim (collinearity-controlled).

    Model ``{response_col} ~ C(contrast) + {timing_col} + side + reward`` with a
    by-subject random slope ``(1 + timing | subject)``. Categorical
    ``C(contrast)`` absorbs all between-contrast variation, so the timing slope
    is the within-contrast estimator; no ``C(contrast):timing`` interaction.
    Side and reward are deviation-coded (±0.5). ``df`` is one
    ``(target_NM, event, timing_var)`` subset. Reports the timing slope and the
    model's marginal R².

    Returns a one-row tidy frame (``_TIDY_LMM_COLS``), empty if fewer than
    ``min_subjects`` subjects or the fit fails.
    """
    df = df.dropna(subset=[response_col, timing_col]).copy()
    if df['subject'].nunique() < min_subjects:
        return _empty_tidy_lmm()
    df['side'] = np.where(df['side'] == 'contra', 0.5, -0.5)
    df['reward'] = np.where(df['feedbackType'] == 1, 0.5, -0.5)
    formula = f'{response_col} ~ C(contrast) + {timing_col} + side + reward'
    lmm = fit_lmm(formula, df, groups=df['subject'],
                   re_formula=f'1 + {timing_col}', reml=True)
    if lmm is None:
        return _empty_tidy_lmm()
    return _tidy_lmm_row(lmm, timing_col, df, timing_col)


def within_contrast_variation(df, timing_col):
    """Descriptive precondition for the within-contrast model: does the timing
    variable still vary *within* a contrast level?

    Partitions the timing variable's spread into within- and between-contrast
    components: ``var_within`` is the mean of the per-contrast variances,
    ``var_between`` is the variance of the per-contrast means. A within
    component comparable to (or larger than) the between component means there
    is within-contrast movement variation for the model to exploit. No model
    fit. ``df`` is one ``(target_NM, timing_var)`` subset.

    Returns a one-row tidy frame with columns ``timing_col``, ``var_within``,
    ``var_between``, ``n_contrasts``, ``n_trials``.
    """
    df = df.dropna(subset=[timing_col])
    by_contrast = df.groupby('contrast')[timing_col]
    return pd.DataFrame([{
        'timing_col': timing_col,
        'var_within': float(by_contrast.var(ddof=1).mean()),
        'var_between': float(by_contrast.mean().var(ddof=1)),
        'n_contrasts': df['contrast'].nunique(),
        'n_trials': len(df),
    }])


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
