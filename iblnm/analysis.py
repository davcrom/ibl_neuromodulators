import numpy as np
import pandas as pd
from tqdm import tqdm

from iblnm.config import TARGET_FS, get_contrast_coding


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


def fit_response_glm(events, event_name, min_trials=20, contrast_coding='log'):
    """Fit per-recording OLS models and return coefficients as features.

    For each recording in the events DataFrame, fits:
    ``response_early ~ 1 + log_contrast + side + reward
    + log_contrast:side + log_contrast:reward + side:reward``

    Uses deviation coding (±0.5) for side and reward, consistent with the LMM:
        side:   contra = +0.5, ipsi = −0.5
        reward: correct = +0.5, incorrect = −0.5

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
        'intercept', 'log_contrast', 'side', 'reward',
        'log_contrast:side', 'log_contrast:reward', 'side:reward',
    ]

    coef_rows = {}
    se_rows = {}

    # FIXME: this loop should be in the group method, not in the function
    for (eid, brain_region), grp in df.groupby(['eid', 'brain_region']):
        # Drop NaN responses
        valid = grp.dropna(subset=['response_early'])
        if len(valid) < min_trials:
            continue

        y = valid['response_early'].values
        log_c = _transform(valid['contrast'].values)
        # Deviation coding ±0.5, matching fit_response_lmm
        side = np.where(valid['side'] == 'contra', 0.5, -0.5)
        reward = np.where(valid['feedbackType'] == 1, 0.5, -0.5)

        X = np.column_stack([
            np.ones(len(y)),
            log_c,
            side,
            reward,
            log_c * side,
            log_c * reward,
            side * reward,
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
    random_effects : dict
        Subject → pd.Series of BLUPs.
    predictions : pd.DataFrame or None
        Model predictions on a design grid (set by callers, not by _fit_lmm).
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
                 random_effects, contrast_coding='log'):
        self.model = model
        self.result = result
        self.summary_df = summary_df
        self.variance_explained = variance_explained
        self.random_effects = random_effects
        self.contrast_coding = contrast_coding
        self.predictions = None
        self.emm_reward = None
        self.emm_side = None
        self.emm_contrast = None
        self.contrast_slopes = None
        self.interaction_contrast_reward = None
        self.interaction_contrast_side = None
        self.interaction_reward_side = None


def _variance_explained(result, df, response_col):
    """Compute Nakagawa & Schielzeth R² for a fitted MixedLM.

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
        Keys: 'marginal', 'conditional' (both float, 0–1).
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


def _fit_lmm(formula, df, groups, re_formula='1', reml=True,
             contrast_coding='log'):
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
    except (np.linalg.LinAlgError, ValueError):
        return None
    except Exception:
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

    ve = _variance_explained(result, df, response_col)
    if ve['marginal'] == 0.0 and ve['conditional'] == 0.0 and np.var(df[response_col].values) == 0:
        return None

    re_dict = {
        subj: pd.Series(effects)
        for subj, effects in result.random_effects.items()
    }

    return LMMResult(
        model=model,
        result=result,
        summary_df=summary_df,
        variance_explained=ve,
        random_effects=re_dict,
        contrast_coding=contrast_coding,
    )


def fit_response_lmm(df, response_col, formula=None, re_formula='1',
                      contrast_coding='log'):
    """Fit a linear mixed-effects model to trial-level response data.

    Uses deviation coding (±0.5) for side and reward so that every
    coefficient is interpretable at the grand mean of all other factors.

    Model: ``response ~ log_contrast * side * reward`` with subject
    as random effect.

    Coding:
        side:   contra = +0.5, ipsi = −0.5
        reward: correct = +0.5, incorrect = −0.5

    Parameters
    ----------
    df : pd.DataFrame
        Trial-level data with columns: contrast, side, feedbackType,
        subject, and ``response_col``.
    response_col : str
        Column name for the response magnitude.
    formula : str, optional
        Wilkinson formula for fixed effects. Default:
        ``'{response_col} ~ log_contrast * side * reward'``.
    re_formula : str
        Random effects formula passed to ``statsmodels.MixedLM``.
        Default ``'1'`` (random intercept only).

    Returns
    -------
    LMMResult or None
        None if the model fails to converge or data is degenerate.
    """
    _transform, _ = get_contrast_coding(contrast_coding)

    df = df.copy()
    df['log_contrast'] = _transform(df['contrast'])
    # Deviation coding: ±0.5
    df['side'] = np.where(df['side'] == 'contra', 0.5, -0.5)
    df['reward'] = np.where(df['feedbackType'] == 1, 0.5, -0.5)

    # Check minimum requirements
    if df['subject'].nunique() < 2:
        return None
    if df['side'].nunique() < 2 or df['reward'].nunique() < 2:
        return None
    if df[response_col].std() == 0:
        return None

    if formula is None:
        formula = f'{response_col} ~ log_contrast * side * reward'

    lmm_result = _fit_lmm(formula, df, groups=df['subject'],
                           re_formula=re_formula, reml=True,
                           contrast_coding=contrast_coding)
    if lmm_result is None:
        return None

    # Predictions on contrast grid (fixed effects only)
    result = lmm_result.result
    fe_params = result.fe_params.values
    fe_names = list(result.fe_params.index)
    fe_cov = result.cov_params().loc[fe_names, fe_names].values

    contrasts = sorted(df['contrast'].unique())
    pred_rows = []
    for side_label, side_val in [('contra', 0.5), ('ipsi', -0.5)]:
        for reward_label, reward_val in [('incorrect', -0.5), ('correct', 0.5)]:
            grid = pd.DataFrame({
                'contrast': contrasts,
                'log_contrast': _transform(contrasts),
                'side': side_val,
                'reward': reward_val,
                'subject': df['subject'].iloc[0],  # dummy
                response_col: 0.0,
            })
            from patsy import dmatrix
            design_info = result.model.data.orig_exog.design_info
            X_grid = np.asarray(dmatrix(design_info, grid))
            pred = X_grid @ fe_params
            se_pred = np.sqrt(np.maximum(np.diag(X_grid @ fe_cov @ X_grid.T), 0))
            for i, c in enumerate(contrasts):
                pred_rows.append({
                    'contrast': c,
                    'side': side_label,
                    'reward': reward_label,
                    'predicted': float(pred[i]),
                    'ci_lower': float(pred[i] - 1.96 * se_pred[i]),
                    'ci_upper': float(pred[i] + 1.96 * se_pred[i]),
                })
    lmm_result.predictions = pd.DataFrame(pred_rows)

    return lmm_result


def compute_marginal_means(lmm_result, factor):
    """Compute estimated marginal means for a factor from an LMM fit.

    Averages model predictions over the levels of all other factors in the
    design, yielding the marginal mean response at each level of ``factor``.
    With deviation coding (±0.5), averaged factors are set to 0.

    Parameters
    ----------
    lmm_result : LMMResult
        Output from ``fit_response_lmm`` (deviation-coded model).
    factor : str
        Factor to compute EMMs for: 'reward', 'side', or 'contrast'.

    Returns
    -------
    pd.DataFrame
        Columns: level, mean, ci_lower, ci_upper.
    """
    result = lmm_result.result
    fe_names = list(result.fe_params.index)
    fe_params = result.fe_params.values
    fe_cov = result.cov_params().loc[fe_names, fe_names].values

    _transform, _ = get_contrast_coding(lmm_result.contrast_coding)

    predictions = lmm_result.predictions
    contrast_levels = sorted(predictions['contrast'].unique())
    lc_values = [_transform(c) for c in contrast_levels]
    mean_lc = np.mean(lc_values)

    # Factor level specs: (label, {column: value})
    if factor == 'reward':
        level_specs = [
            ('incorrect', {'reward': -0.5}),
            ('correct', {'reward': 0.5}),
        ]
    elif factor == 'side':
        level_specs = [
            ('contra', {'side': 0.5}),
            ('ipsi', {'side': -0.5}),
        ]
    elif factor == 'contrast':
        level_specs = [
            (c, {'log_contrast': _transform(c)})
            for c in contrast_levels
        ]
    else:
        raise ValueError(f"factor must be 'reward', 'side', or 'contrast', got {factor}")

    # Build name→index map for the 8 deviation-coded coefficients
    idx = {name: i for i, name in enumerate(fe_names)}

    rows = []
    for label, factor_vals in level_specs:
        # Set factor to its level value, all other factors to 0 (grand mean)
        lc = factor_vals.get('log_contrast', mean_lc)
        side = factor_vals.get('side', 0.0)
        reward = factor_vals.get('reward', 0.0)

        # Construct the design vector manually
        c = np.zeros(len(fe_names))
        c[idx['Intercept']] = 1.0
        if 'log_contrast' in idx:
            c[idx['log_contrast']] = lc
        if 'side' in idx:
            c[idx['side']] = side
        if 'reward' in idx:
            c[idx['reward']] = reward
        if 'log_contrast:side' in idx:
            c[idx['log_contrast:side']] = lc * side
        if 'log_contrast:reward' in idx:
            c[idx['log_contrast:reward']] = lc * reward
        if 'side:reward' in idx:
            c[idx['side:reward']] = side * reward
        if 'log_contrast:side:reward' in idx:
            c[idx['log_contrast:side:reward']] = lc * side * reward

        mean_pred = float(c @ fe_params)
        se = float(np.sqrt(max(c @ fe_cov @ c, 0)))

        rows.append({
            'level': label,
            'mean': mean_pred,
            'ci_lower': mean_pred - 1.96 * se,
            'ci_upper': mean_pred + 1.96 * se,
        })

    return pd.DataFrame(rows)


def compute_contrast_slopes(lmm_result):
    """Compute contrast slopes per reward condition from an LMM fit.

    With deviation coding (±0.5):
        incorrect (reward=-0.5): β_lc - 0.5 × β_lc:reward
        correct   (reward=+0.5): β_lc + 0.5 × β_lc:reward

    When the model includes random slopes for log_contrast, subject-level
    slopes (population + BLUP deviation) are also returned.

    Parameters
    ----------
    lmm_result : LMMResult
        Output from ``fit_response_lmm`` (deviation-coded model).

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

    beta_lc = fe_params['log_contrast']
    idx_lc = fe_names.index('log_contrast')

    interaction_name = 'log_contrast:reward'
    has_interaction = interaction_name in fe_names
    if has_interaction:
        beta_int = fe_params[interaction_name]
        idx_int = fe_names.index(interaction_name)
    else:
        beta_int = 0.0

    rows = []
    for label, reward_val in [('incorrect', -0.5), ('correct', 0.5)]:
        slope = beta_lc + reward_val * beta_int
        # Var(β_lc + r × β_int) = Var(β_lc) + r²Var(β_int) + 2r·Cov
        if has_interaction:
            var = (fe_cov[idx_lc, idx_lc]
                   + reward_val**2 * fe_cov[idx_int, idx_int]
                   + 2 * reward_val * fe_cov[idx_lc, idx_int])
            se = np.sqrt(max(var, 0))
        else:
            se = np.sqrt(fe_cov[idx_lc, idx_lc])

        rows.append({
            'reward': label,
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
            for i, (label, _) in enumerate(
                    [('incorrect', -0.5), ('correct', 0.5)]):
                pop_slope = rows[i]['slope']
                rows.append({
                    'reward': label,
                    'slope': float(pop_slope + u_slope),
                    'ci_lower': np.nan,
                    'ci_upper': np.nan,
                    'type': 'subject',
                    'subject': subj,
                })

    return pd.DataFrame(rows)


def compute_interaction_effects(lmm_result, y_factor, x_factor):
    """Compute the effect of y_factor at each level of x_factor.

    Uses deviation coding (±0.5). For each level of x_factor, computes
    the y_factor effect while the remaining factor is set to 0 (its grand
    mean under deviation coding).

    Parameters
    ----------
    lmm_result : LMMResult
        Output from ``fit_response_lmm`` (deviation-coded model).
    y_factor : str
        Factor whose effect to compute: 'contrast', 'reward', or 'side'.
    x_factor : str
        Factor to condition on: 'reward' or 'side'.

    Returns
    -------
    pd.DataFrame
        Columns: x_level, effect, ci_lower, ci_upper, p_interaction.
        ``p_interaction`` is the p-value of the two-way interaction
        coefficient (same for both rows).
    """
    result = lmm_result.result
    fe_params = result.fe_params
    fe_names = list(fe_params.index)
    fe_cov = result.cov_params().loc[fe_names, fe_names].values

    def _idx(name):
        return fe_names.index(name) if name in fe_names else None

    # Deviation-coded coefficient names
    _coef = {
        'contrast': 'log_contrast',
        'reward': 'reward',
        'side': 'side',
        'contrast:reward': 'log_contrast:reward',
        'contrast:side': 'log_contrast:side',
        'reward:side': 'side:reward',
    }

    # X-factor levels: (label, deviation-coded value)
    if x_factor == 'reward':
        x_levels = [('incorrect', -0.5), ('correct', 0.5)]
    elif x_factor == 'side':
        x_levels = [('contra', 0.5), ('ipsi', -0.5)]
    else:
        raise ValueError(f"x_factor must be 'reward' or 'side', got {x_factor}")

    # Two-way interaction coefficient name
    yx_key = f'{min(y_factor, x_factor)}:{max(y_factor, x_factor)}'
    int_name = _coef.get(yx_key)
    if int_name and int_name in fe_names:
        p_interaction = float(result.pvalues[int_name])
    else:
        p_interaction = 1.0

    rows = []
    for x_label, x_val in x_levels:
        # Build contrast vector for the y_factor effect at this x_level.
        # Third factor is set to 0 (grand mean under deviation coding).
        c = np.zeros(len(fe_names))

        # y_factor main effect (slope for contrast, difference for categorical)
        y_name = _coef[y_factor]
        if _idx(y_name) is not None:
            c[_idx(y_name)] = 1.0

        # y_factor × x_factor interaction, weighted by x_val
        if int_name and _idx(int_name) is not None:
            c[_idx(int_name)] = x_val

        # Third factor = 0, so y×third and y×x×third terms vanish.

        effect = float(c @ fe_params.values)
        se = float(np.sqrt(max(c @ fe_cov @ c, 0)))

        rows.append({
            'x_level': x_label,
            'effect': effect,
            'ci_lower': effect - 1.96 * se,
            'ci_upper': effect + 1.96 * se,
            'p_interaction': p_interaction,
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


def fit_wheel_lmm(df, dv_col, response_col='response_early',
                   target_nm='', contrast=0.0, min_subjects=2):
    """Fit nested LMMs comparing base (task structure) vs full (+ NM predictor).

    Base:  ``dv ~ C(stim_side) * C(choice) + (1 | subject)``
    Full:  ``dv ~ C(stim_side) * C(choice) + response_col + (1 | subject)``

    Uses ML (not REML) estimation so the likelihood ratio test is valid.

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
    dict or None
        None if fewer than min_subjects or model fails to converge.
    """
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

    base_lmm = _fit_lmm(base_formula, df, groups=df['subject'],
                          re_formula='1', reml=False)
    full_lmm = _fit_lmm(full_formula, df, groups=df['subject'],
                          re_formula='1', reml=False)

    if base_lmm is None or full_lmm is None:
        return None

    base_r2 = base_lmm.variance_explained
    full_r2 = full_lmm.variance_explained
    delta_r2 = full_r2['marginal'] - base_r2['marginal']

    # Likelihood ratio test
    lrt_stat = 2 * (full_lmm.result.llf - base_lmm.result.llf)
    lrt_stat = max(lrt_stat, 0.0)  # numerical floor
    lrt_p = float(chi2.sf(lrt_stat, df=1))

    # NM coefficient from full model
    nm_coef = float(full_lmm.result.fe_params[response_col])
    nm_p = float(full_lmm.result.pvalues[response_col])

    return {
        'dv': dv_col,
        'target_nm': target_nm,
        'contrast': contrast,
        'base_r2_marginal': base_r2['marginal'],
        'full_r2_marginal': full_r2['marginal'],
        'delta_r2': delta_r2,
        'lrt_chi2': float(lrt_stat),
        'lrt_pvalue': lrt_p,
        'nm_coefficient': nm_coef,
        'nm_pvalue': nm_p,
        'n_trials': len(df),
        'n_subjects': df['subject'].nunique(),
    }
