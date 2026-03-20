import numpy as np
import pandas as pd

from iblnm.config import TARGET_FS, contrast_transform


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


def fit_response_glm(events, event_name, min_trials=20):
    """Fit per-recording OLS models and return coefficients as features.

    For each recording in the events DataFrame, fits:
    ``response_early ~ 1 + log_contrast + side + feedback
    + log_contrast:side + log_contrast:feedback + side:feedback``

    Only unbiased-block trials (probabilityLeft == 0.5) are used.

    Parameters
    ----------
    events : pd.DataFrame
        Trial-level data from ``get_response_magnitudes`` with ``add_relative_contrast``
        applied (must have ``side`` column). Required columns: ``eid``,
        ``brain_region``, ``target_NM``, ``event``, ``contrast``, ``side``,
        ``feedbackType``, ``probabilityLeft``, ``response_early``.
    event_name : str
        Event to model (e.g., ``'stimOn_times'``).
    min_trials : int
        Minimum valid trials per recording. Recordings with fewer are skipped.

    Returns
    -------
    coefs : pd.DataFrame
        (n_recordings, 7) coefficient values.
    ses : pd.DataFrame
        (n_recordings, 7) standard errors.
    """
    import logging
    from iblnm.task import add_relative_contrast

    if events is None or len(events) == 0:
        raise ValueError("events DataFrame is empty")

    available_events = events['event'].unique()
    if event_name not in available_events:
        raise ValueError(
            f"Event '{event_name}' not found in events. "
            f"Available: {list(available_events)}"
        )

    # Filter to requested event and unbiased blocks
    df = events[events['event'] == event_name].copy()
    df = df[df['probabilityLeft'] == 0.5]

    # Add lateralized side column if not present
    if 'side' not in df.columns:
        df = add_relative_contrast(df)

    coef_names = [
        'intercept', 'log_contrast', 'side', 'feedback',
        'log_contrast:side', 'log_contrast:feedback', 'side:feedback',
    ]

    coef_rows = {}
    se_rows = {}

    for (eid, brain_region), grp in df.groupby(['eid', 'brain_region']):
        # Drop NaN responses
        valid = grp.dropna(subset=['response_early'])
        if len(valid) < min_trials:
            continue

        y = valid['response_early'].values
        log_c = contrast_transform(valid['contrast'].values)
        side = (valid['side'] == 'contra').astype(float).values
        feedback = valid['feedbackType'].values.astype(float)

        X = np.column_stack([
            np.ones(len(y)),
            log_c,
            side,
            feedback,
            log_c * side,
            log_c * feedback,
            side * feedback,
        ])

        # Check rank
        if np.linalg.matrix_rank(X) < X.shape[1]:
            logging.warning(
                f"Singular design matrix for {eid}/{brain_region}, skipping"
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
    """

    def __init__(self, x_weights, y_weights, x_scores, y_scores,
                 correlations, p_values, n_recordings, n_permutations):
        self.x_weights = x_weights
        self.y_weights = y_weights
        self.x_scores = x_scores
        self.y_scores = y_scores
        self.correlations = correlations
        self.p_values = p_values
        self.n_recordings = n_recordings
        self.n_permutations = n_permutations


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

        for perm_i in range(n_permutations):
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

import warnings


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
    predictions : pd.DataFrame
        Model predictions on a contrast grid.
    random_effects : dict
        Subject → pd.Series of BLUPs.
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
                 predictions, random_effects):
        self.model = model
        self.result = result
        self.summary_df = summary_df
        self.variance_explained = variance_explained
        self.predictions = predictions
        self.random_effects = random_effects
        self.emm_reward = None
        self.emm_side = None
        self.emm_contrast = None
        self.contrast_slopes = None


def fit_events_lmm(df, response_col, formula=None, re_formula='1'):
    """Fit a linear mixed-effects model to trial-level response data.

    Model: ``response ~ log_contrast * C(side) * C(reward)`` with subject
    as random effect.

    Parameters
    ----------
    df : pd.DataFrame
        Trial-level data with columns: contrast, side, feedbackType,
        subject, and ``response_col``.
    response_col : str
        Column name for the response magnitude.
    formula : str, optional
        Wilkinson formula for fixed effects. Default:
        ``'{response_col} ~ log_contrast * C(side) * C(reward)'``.
    re_formula : str
        Random effects formula passed to ``statsmodels.MixedLM``.
        Default ``'1'`` (random intercept only).

    Returns
    -------
    LMMResult or None
        None if the model fails to converge or data is degenerate.
    """
    import statsmodels.formula.api as smf
    from statsmodels.tools.sm_exceptions import ConvergenceWarning

    df = df.copy()
    df['log_contrast'] = contrast_transform(df['contrast'])
    df['reward'] = (df['feedbackType'] == 1).astype(int)

    # Check minimum requirements
    if df['subject'].nunique() < 2:
        return None
    if df['side'].nunique() < 2 or df['reward'].nunique() < 2:
        return None
    if df[response_col].std() == 0:
        return None

    if formula is None:
        formula = f'{response_col} ~ log_contrast * C(side) * C(reward)'

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model = smf.mixedlm(
                formula, df, groups=df['subject'],
                re_formula=re_formula,
            )
            result = model.fit(reml=True)
        fatal = any(
            issubclass(w.category, ConvergenceWarning)
            and 'failed to converge' in str(w.message).lower()
            for w in caught
        )
        if fatal:
            return None
    except (np.linalg.LinAlgError, ValueError):
        return None
    except Exception:
        return None

    # Summary table — use fixed-effects params only
    fe_names = list(result.fe_params.index)
    summary_df = pd.DataFrame({
        'Coef.': result.fe_params,
        'Std.Err.': result.bse_fe,
        'z': result.tvalues[fe_names],
        'P>|z|': result.pvalues[fe_names],
    })

    # Variance explained (Nakagawa & Schielzeth R²)
    y = df[response_col].values
    var_y = np.var(y)
    if var_y == 0:
        return None

    # Fixed-effects-only predictions
    fe_params = result.fe_params.values
    exog = result.model.exog
    y_pred_fe = exog @ fe_params

    # Full predictions (fixed + random)
    y_pred_full = result.fittedvalues.values

    var_fixed = np.var(y_pred_fe)
    var_random = np.var(y_pred_full - y_pred_fe)

    r2_marginal = float(np.clip(var_fixed / var_y, 0, 1))
    r2_conditional = float(np.clip((var_fixed + var_random) / var_y, 0, 1))

    # Predictions on contrast grid (fixed effects only)
    contrasts = sorted(df['contrast'].unique())
    pred_rows = []
    # Build design matrix for the grid using the model's formula
    fe_cov = result.cov_params().loc[fe_names, fe_names].values
    for side in ['contra', 'ipsi']:
        for reward in [0, 1]:
            grid = pd.DataFrame({
                'contrast': contrasts,
                'log_contrast': contrast_transform(contrasts),
                'side': side,
                'reward': reward,
                'subject': df['subject'].iloc[0],  # dummy
                response_col: 0.0,
            })
            # Get design matrix for fixed effects
            from patsy import dmatrix
            design_info = result.model.data.orig_exog.design_info
            X_grid = np.asarray(dmatrix(design_info, grid))
            pred = X_grid @ fe_params
            se_pred = np.sqrt(np.diag(X_grid @ fe_cov @ X_grid.T))
            for i, c in enumerate(contrasts):
                pred_rows.append({
                    'contrast': c,
                    'side': side,
                    'reward': reward,
                    'predicted': float(pred[i]),
                    'ci_lower': float(pred[i] - 1.96 * se_pred[i]),
                    'ci_upper': float(pred[i] + 1.96 * se_pred[i]),
                })
    predictions = pd.DataFrame(pred_rows)

    # Extract random effects (BLUPs) per subject
    re_dict = {
        subj: pd.Series(effects)
        for subj, effects in result.random_effects.items()
    }

    return LMMResult(
        model=model,
        result=result,
        summary_df=summary_df,
        variance_explained={'marginal': r2_marginal, 'conditional': r2_conditional},
        predictions=predictions,
        random_effects=re_dict,
    )


def compute_marginal_means(lmm_result, factor):
    """Compute estimated marginal means for a factor from an LMM fit.

    Averages model predictions over the levels of all other factors in the
    design, yielding the marginal mean response at each level of ``factor``.

    Parameters
    ----------
    lmm_result : LMMResult
        Output from ``fit_events_lmm``.
    factor : str
        Factor to compute EMMs for: 'reward' or 'side'.

    Returns
    -------
    pd.DataFrame
        Columns: level, mean, ci_lower, ci_upper.
    """
    predictions = lmm_result.predictions
    other = 'side' if factor == 'reward' else 'reward'

    rows = []
    for level, df_level in predictions.groupby(factor):
        # Average over levels of the other factor and contrast
        # (equal weighting = balanced EMM)
        mean_pred = df_level['predicted'].mean()

        # CI: average the design-matrix rows, propagate through covariance
        # We need to reconstruct design matrix rows for proper CI
        # For a linear model, mean of predictions = prediction at mean of inputs
        # and SE of the mean prediction = sqrt(x_bar' Cov x_bar)
        result = lmm_result.result
        fe_names = list(result.fe_params.index)
        fe_params = result.fe_params.values
        fe_cov = result.cov_params().loc[fe_names, fe_names].values

        # Build design matrix for all rows of this factor level
        from patsy import dmatrix
        design_info = result.model.data.orig_exog.design_info
        # Need a DataFrame with all the right columns
        grid = df_level.copy()
        grid['log_contrast'] = contrast_transform(grid['contrast'])
        # Add dummy columns needed by the formula
        response_col = result.model.endog_names
        if response_col not in grid.columns:
            grid[response_col] = 0.0
        if 'subject' not in grid.columns:
            grid['subject'] = 'dummy'

        X_grid = np.asarray(dmatrix(design_info, grid))
        # Average the design matrix rows (equal weighting)
        x_bar = X_grid.mean(axis=0)
        mean_pred_exact = float(x_bar @ fe_params)
        se = float(np.sqrt(x_bar @ fe_cov @ x_bar))

        rows.append({
            'level': level,
            'mean': mean_pred_exact,
            'ci_lower': mean_pred_exact - 1.96 * se,
            'ci_upper': mean_pred_exact + 1.96 * se,
        })

    return pd.DataFrame(rows)


def compute_contrast_slopes(lmm_result):
    """Compute contrast slopes per reward condition from an LMM fit.

    The contrast slope for reward=0 (incorrect) is β_log_contrast.
    For reward=1 (correct), it is β_log_contrast + β_log_contrast:C(reward)[T.1].

    When the model includes random slopes for log_contrast, subject-level
    slopes (population + BLUP deviation) are also returned.

    Parameters
    ----------
    lmm_result : LMMResult
        Output from ``fit_events_lmm``.

    Returns
    -------
    pd.DataFrame
        Columns: reward, slope, ci_lower, ci_upper, type, subject.
        type is 'population' or 'subject'.
    """
    result = lmm_result.result
    fe_params = result.fe_params
    fe_names = list(fe_params.index)
    fe_cov = result.cov_params().loc[fe_names, fe_names].values

    # Base contrast slope (reward=0)
    beta_contrast = fe_params['log_contrast']
    idx_contrast = fe_names.index('log_contrast')

    # Interaction term: log_contrast:C(reward)[T.1]
    interaction_name = 'log_contrast:C(reward)[T.1]'
    has_interaction = interaction_name in fe_names
    if has_interaction:
        beta_interaction = fe_params[interaction_name]
        idx_interaction = fe_names.index(interaction_name)
    else:
        beta_interaction = 0.0

    rows = []

    # Population slopes
    for reward in [0, 1]:
        if reward == 0:
            slope = beta_contrast
            se = np.sqrt(fe_cov[idx_contrast, idx_contrast])
        else:
            slope = beta_contrast + beta_interaction
            if has_interaction:
                # Var(β1 + β5) = Var(β1) + Var(β5) + 2*Cov(β1, β5)
                var = (fe_cov[idx_contrast, idx_contrast]
                       + fe_cov[idx_interaction, idx_interaction]
                       + 2 * fe_cov[idx_contrast, idx_interaction])
                se = np.sqrt(max(var, 0))
            else:
                se = np.sqrt(fe_cov[idx_contrast, idx_contrast])

        rows.append({
            'reward': reward,
            'slope': float(slope),
            'ci_lower': float(slope - 1.96 * se),
            'ci_upper': float(slope + 1.96 * se),
            'type': 'population',
            'subject': None,
        })

    # Subject-level slopes (population + BLUP)
    re_dict = lmm_result.random_effects
    has_random_slope = any(
        'log_contrast' in effects.index for effects in re_dict.values()
    )
    if has_random_slope:
        for subj, effects in re_dict.items():
            u_slope = effects.get('log_contrast', 0.0)
            for reward in [0, 1]:
                pop_slope = rows[reward]['slope']  # 0=incorrect, 1=correct
                rows.append({
                    'reward': reward,
                    'slope': float(pop_slope + u_slope),
                    'ci_lower': np.nan,
                    'ci_upper': np.nan,
                    'type': 'subject',
                    'subject': subj,
                })

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
# Wheel Kinematics LMM
# =============================================================================


class WheelLMMResult:
    """Container for a nested LMM comparison (base vs full with NM predictor).

    Attributes
    ----------
    dv : str
        Dependent variable name.
    target_nm : str
        Target-NM group.
    contrast : float
        Contrast level.
    base_r2 : dict
        Nakagawa R² for base model {'marginal', 'conditional'}.
    full_r2 : dict
        Nakagawa R² for full model {'marginal', 'conditional'}.
    delta_r2 : float
        full_r2['marginal'] - base_r2['marginal'].
    lrt_chi2 : float
        Likelihood ratio test statistic.
    lrt_pvalue : float
        LRT p-value (chi2, df=1).
    nm_coefficient : float
        Fixed-effect coefficient for the NM predictor in the full model.
    nm_pvalue : float
        p-value for the NM coefficient.
    n_trials : int
    n_subjects : int
    """

    def __init__(self, dv, target_nm, contrast, base_r2, full_r2,
                 delta_r2, lrt_chi2, lrt_pvalue, nm_coefficient, nm_pvalue,
                 n_trials, n_subjects):
        self.dv = dv
        self.target_nm = target_nm
        self.contrast = contrast
        self.base_r2 = base_r2
        self.full_r2 = full_r2
        self.delta_r2 = delta_r2
        self.lrt_chi2 = lrt_chi2
        self.lrt_pvalue = lrt_pvalue
        self.nm_coefficient = nm_coefficient
        self.nm_pvalue = nm_pvalue
        self.n_trials = n_trials
        self.n_subjects = n_subjects


def _compute_nakagawa_r2(result, df, response_col):
    """Compute Nakagawa & Schielzeth R² for a fitted MixedLM.

    Returns dict with 'marginal' and 'conditional' R².
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

    r2_m = float(np.clip(var_fixed / var_y, 0, 1))
    r2_c = float(np.clip((var_fixed + var_random) / var_y, 0, 1))
    return {'marginal': r2_m, 'conditional': r2_c}


def fit_wheel_kinematics_lmm(df, dv_col, response_col='response_early',
                              target_nm='', contrast=0.0, min_subjects=2):
    """Fit nested LMMs comparing base (task structure) vs full (+ NM predictor).

    Base:  ``dv ~ C(stim_side) * C(choice) + (1 | subject)``
    Full:  ``dv ~ C(stim_side) * C(choice) + response_early + (1 | subject)``

    Parameters
    ----------
    df : pd.DataFrame
        Trial-level data with columns: stim_side, choice, subject,
        ``dv_col``, and ``response_col``.
    dv_col : str
        Dependent variable column name.
    response_col : str
        NM response magnitude column name.
    target_nm : str
        Target-NM label (stored in result, not used for filtering).
    contrast : float
        Contrast level (stored in result, not used for filtering).
    min_subjects : int
        Minimum subjects required to fit.

    Returns
    -------
    WheelLMMResult or None
        None if fewer than min_subjects or model fails to converge.
    """
    import statsmodels.formula.api as smf
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    from scipy.stats import chi2

    df = df.dropna(subset=[dv_col, response_col]).copy()

    if df['subject'].nunique() < min_subjects:
        return None
    if len(df) < 10:
        return None

    # Check we have variation in categorical predictors
    if df['stim_side'].nunique() < 2 or df['choice'].nunique() < 2:
        return None

    base_formula = f'{dv_col} ~ C(stim_side) * C(choice)'
    full_formula = f'{dv_col} ~ C(stim_side) * C(choice) + {response_col}'

    def _fit(formula):
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                model = smf.mixedlm(
                    formula, df, groups=df['subject'], re_formula='1',
                )
                result = model.fit(reml=False)  # ML for LRT comparison
            fatal = any(
                issubclass(w.category, ConvergenceWarning)
                and 'failed to converge' in str(w.message).lower()
                for w in caught
            )
            if fatal:
                return None
            return result
        except Exception:
            return None

    base_result = _fit(base_formula)
    full_result = _fit(full_formula)

    if base_result is None or full_result is None:
        return None

    base_r2 = _compute_nakagawa_r2(base_result, df, dv_col)
    full_r2 = _compute_nakagawa_r2(full_result, df, dv_col)
    delta_r2 = full_r2['marginal'] - base_r2['marginal']

    # Likelihood ratio test
    lrt_stat = 2 * (full_result.llf - base_result.llf)
    lrt_stat = max(lrt_stat, 0.0)  # numerical floor
    lrt_p = float(chi2.sf(lrt_stat, df=1))

    # NM coefficient from full model
    nm_coef = float(full_result.fe_params[response_col])
    nm_p = float(full_result.pvalues[response_col])

    return WheelLMMResult(
        dv=dv_col,
        target_nm=target_nm,
        contrast=contrast,
        base_r2=base_r2,
        full_r2=full_r2,
        delta_r2=delta_r2,
        lrt_chi2=float(lrt_stat),
        lrt_pvalue=lrt_p,
        nm_coefficient=nm_coef,
        nm_pvalue=nm_p,
        n_trials=len(df),
        n_subjects=df['subject'].nunique(),
    )
