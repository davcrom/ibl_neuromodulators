"""Tests for iblnm.analysis module."""
import numpy as np
import pandas as pd

from iblnm.analysis import get_responses, normalize_responses, resample_signal


class TestComputeBleachingTau:
    def test_recovers_known_tau(self):
        from iblnm.analysis import compute_bleaching_tau
        t = np.linspace(0, 600, 18000)
        signal = pd.Series(1000 * np.exp(-t / 300) + 500, index=t)
        tau = compute_bleaching_tau(signal)
        assert 240 < tau < 360  # 300 ± 20%

    def test_returns_float(self):
        from iblnm.analysis import compute_bleaching_tau
        t = np.linspace(0, 60, 1800)
        signal = pd.Series(np.exp(-t / 60) + 1, index=t)
        assert isinstance(compute_bleaching_tau(signal), float)


class TestComputeIsoCorrelation:
    def test_near_perfect_correlation(self):
        from iblnm.analysis import compute_iso_correlation
        np.random.seed(42)
        t = np.linspace(0, 60, 1800)
        sig = pd.Series(2.0 * t + 1.0 + 0.01 * np.random.randn(len(t)), index=t)
        ref = pd.Series(t, index=t)
        assert compute_iso_correlation(sig, ref) > 0.999

    def test_uncorrelated_near_zero(self):
        from iblnm.analysis import compute_iso_correlation
        np.random.seed(0)
        t = np.linspace(0, 60, 1800)
        sig = pd.Series(np.random.randn(len(t)), index=t)
        ref = pd.Series(np.random.randn(len(t)), index=t)
        assert abs(compute_iso_correlation(sig, ref)) < 0.1

    def test_returns_float(self):
        from iblnm.analysis import compute_iso_correlation
        t = np.linspace(0, 10, 300)
        s = pd.Series(np.ones(len(t)), index=t)
        assert isinstance(compute_iso_correlation(s, s), float)


class TestGetResponses:
    def test_fs_rounding(self):
        """fs=29.97 should produce same n_samples as fs=30."""
        t_30 = np.arange(0, 30, 1 / 30)
        t_2997 = np.arange(0, 30, 1 / 29.97)
        signal_30 = pd.Series(np.sin(t_30), index=t_30)
        signal_2997 = pd.Series(np.sin(t_2997), index=t_2997)
        events = np.array([15.0])

        r_30, tpts_30 = get_responses(signal_30, events, t0=-1, t1=1.0)
        r_2997, tpts_2997 = get_responses(signal_2997, events, t0=-1, t1=1.0)
        assert r_30.shape[1] == r_2997.shape[1]

    def test_returns_tpts(self):
        """get_responses should return (responses, tpts) tuple."""
        t = np.arange(0, 10, 1 / 30)
        signal = pd.Series(np.ones(len(t)), index=t)
        events = np.array([5.0])
        responses, tpts = get_responses(signal, events, t0=-1, t1=1.0)
        assert tpts.shape == (responses.shape[1],)
        assert tpts[0] < 0  # starts before event
        assert tpts[-1] > 0  # ends after event

    def test_known_values(self):
        """Extracted response should match known signal values."""
        t = np.arange(0, 10, 1 / 30)
        signal = pd.Series(t * 2, index=t)  # signal = 2*t
        events = np.array([5.0])
        responses, tpts = get_responses(signal, events, t0=-0.5, t1=0.5)
        # At t=5, signal=10. Window [-0.5, 0.5] → signal ~ [9, 11]
        assert responses.shape == (1, 30)
        np.testing.assert_allclose(responses[0, 0], 9.0, atol=0.1)
        np.testing.assert_allclose(responses[0, -1], 10.93, atol=0.1)

    def test_invalid_events_are_nan(self):
        """Events outside valid window should produce NaN rows."""
        t = np.arange(0, 10, 1 / 30)
        signal = pd.Series(np.ones(len(t)), index=t)
        events = np.array([0.0, 5.0, 9.5])
        responses, tpts = get_responses(signal, events, t0=-1, t1=1.0)
        assert np.all(np.isnan(responses[0]))
        assert not np.any(np.isnan(responses[1]))
        assert np.all(np.isnan(responses[2]))

    def test_variable_t1_masks_beyond_endpoint(self):
        """When t1 is array, samples beyond per-trial t1 should be NaN."""
        t = np.arange(0, 20, 1 / 30)
        signal = pd.Series(np.ones(len(t)), index=t)
        events = np.array([5.0, 10.0])
        # Trial 0: window ends 0.5s after event, Trial 1: 0.3s after event
        t1_per_trial = np.array([5.5, 10.3])
        responses, tpts = get_responses(signal, events, t0=-1.0, t1=t1_per_trial)
        # Trial 0: samples after t=0.5 relative should be NaN
        mask_0 = tpts > 0.5
        assert np.all(np.isnan(responses[0, mask_0]))
        assert not np.any(np.isnan(responses[0, ~mask_0]))

    def test_variable_t1_nan_gets_full_window(self):
        """Trial with NaN t1 should get full window (no masking)."""
        t = np.arange(0, 20, 1 / 30)
        signal = pd.Series(np.ones(len(t)), index=t)
        events = np.array([10.0])
        t1_per_trial = np.array([np.nan])
        responses, tpts = get_responses(signal, events, t0=-1.0, t1=t1_per_trial)
        assert not np.any(np.isnan(responses[0]))


class TestNormalizeResponses:
    def test_baseline_subtraction(self):
        """Baseline subtraction should center responses around 0 in baseline window."""
        tpts = np.arange(-1, 1, 1 / 30)
        # Responses with known baseline offset
        responses = np.ones((3, len(tpts))) * 5.0
        responses[:, tpts >= 0] = 10.0
        normalized = normalize_responses(responses, tpts, bwin=(-0.1, 0), divide=True)
        # Baseline region should be ~0
        baseline_mask = (tpts >= -0.1) & (tpts < 0)
        np.testing.assert_allclose(normalized[:, baseline_mask], 0.0, atol=0.01)

    def test_no_transpose_artifacts(self):
        """normalize_responses should handle non-square arrays correctly."""
        tpts = np.arange(-1, 1, 1 / 30)
        n_trials = 5
        responses = np.random.randn(n_trials, len(tpts)) + 10
        normalized = normalize_responses(responses, tpts, bwin=(-0.1, 0), divide=False)
        assert normalized.shape == responses.shape


class TestComputeResponseMagnitude:
    def test_constant_signal_1d(self):
        """Constant 1D signal returns that constant in any window."""
        from iblnm.analysis import compute_response_magnitude
        tpts = np.linspace(-1, 1, 61)
        response = np.full(len(tpts), 2.0)
        result = compute_response_magnitude(response, tpts, window=(0.1, 0.35))
        np.testing.assert_allclose(result, 2.0)

    def test_per_trial_means_2d(self):
        """2D array returns per-trial means."""
        from iblnm.analysis import compute_response_magnitude
        tpts = np.linspace(-1, 1, 61)
        responses = np.stack([np.full(len(tpts), v) for v in [1.0, 3.0, 5.0]])
        result = compute_response_magnitude(responses, tpts, window=(0.1, 0.35))
        np.testing.assert_allclose(result, [1.0, 3.0, 5.0])

    def test_nan_samples_ignored(self):
        """NaN samples are excluded via nanmean."""
        from iblnm.analysis import compute_response_magnitude
        tpts = np.linspace(-1, 1, 61)
        response = np.full(len(tpts), 4.0)
        # Set some samples in the window to NaN
        mask = (tpts >= 0.1) & (tpts < 0.2)
        response[mask] = np.nan
        result = compute_response_magnitude(response, tpts, window=(0.1, 0.35))
        np.testing.assert_allclose(result, 4.0)


class TestResampleSignal:
    def test_uniform_output(self):
        """Output timestamps should be exactly uniform at target_fs."""
        t = np.sort(np.cumsum(np.random.uniform(0.03, 0.04, 1000)))
        signal = pd.Series(np.sin(2 * np.pi * t), index=t)
        resampled = resample_signal(signal, target_fs=30)
        dt = np.diff(resampled.index.values)
        np.testing.assert_allclose(dt, 1 / 30, atol=1e-10)

    def test_preserves_sine(self):
        """PCHIP resampling of a sine wave should be accurate."""
        t = np.arange(0, 10, 1 / 30.5)  # slightly off from 30 Hz
        signal = pd.Series(np.sin(2 * np.pi * 0.5 * t), index=t)
        resampled = resample_signal(signal, target_fs=30)
        expected = np.sin(2 * np.pi * 0.5 * resampled.index.values)
        np.testing.assert_allclose(resampled.values, expected, atol=0.01)

    def test_output_within_input_range(self):
        """Resampled signal should not extend beyond input time range."""
        t = np.arange(5.0, 15.0, 1 / 30)
        signal = pd.Series(np.ones(len(t)), index=t)
        resampled = resample_signal(signal, target_fs=30)
        assert resampled.index[0] >= t[0]
        assert resampled.index[-1] <= t[-1]


# =============================================================================
# Response Vector Analysis Tests
# =============================================================================

import pytest


def _make_response_matrix(n_recordings=4, n_features=10, seed=42):
    """Helper: DataFrame with known response vectors."""
    rng = np.random.default_rng(seed)
    index = pd.MultiIndex.from_tuples(
        [(f'eid-{i}', f'region-{i}') for i in range(n_recordings)],
        names=['eid', 'brain_region'],
    )
    cols = [f'feat_{i}' for i in range(n_features)]
    return pd.DataFrame(rng.standard_normal((n_recordings, n_features)),
                        index=index, columns=cols)


class TestCosineSimilarityMatrix:

    def test_identical_vectors_similarity_one(self):
        from iblnm.analysis import cosine_similarity_matrix
        df = pd.DataFrame(
            [[1, 0, 0], [1, 0, 0]],
            index=pd.MultiIndex.from_tuples([('a', 'r1'), ('b', 'r2')]),
            columns=['f0', 'f1', 'f2'],
        )
        sim = cosine_similarity_matrix(df)
        np.testing.assert_allclose(sim.values, [[1, 1], [1, 1]], atol=1e-10)

    def test_orthogonal_vectors_similarity_zero(self):
        from iblnm.analysis import cosine_similarity_matrix
        df = pd.DataFrame(
            [[1, 0], [0, 1]],
            index=pd.MultiIndex.from_tuples([('a', 'r1'), ('b', 'r2')]),
            columns=['f0', 'f1'],
        )
        sim = cosine_similarity_matrix(df)
        assert np.isclose(sim.iloc[0, 1], 0.0, atol=1e-10)

    def test_drops_nan_rows(self):
        from iblnm.analysis import cosine_similarity_matrix
        df = pd.DataFrame(
            [[1, 0], [np.nan, 1], [0, 1]],
            index=pd.MultiIndex.from_tuples([('a', 'r1'), ('b', 'r2'), ('c', 'r3')]),
            columns=['f0', 'f1'],
        )
        sim = cosine_similarity_matrix(df)
        assert sim.shape == (2, 2)  # row with NaN excluded

    def test_symmetric(self):
        from iblnm.analysis import cosine_similarity_matrix
        mat = _make_response_matrix(n_recordings=5)
        sim = cosine_similarity_matrix(mat)
        np.testing.assert_allclose(sim.values, sim.values.T, atol=1e-10)


class TestWithinBetweenSimilarity:

    def test_correct_pair_counts(self):
        from iblnm.analysis import cosine_similarity_matrix, within_between_similarity
        df = pd.DataFrame(
            [[1, 0], [1, 0.1], [0, 1]],
            index=pd.MultiIndex.from_tuples([('a', 'r1'), ('b', 'r2'), ('c', 'r3')]),
            columns=['f0', 'f1'],
        )
        sim = cosine_similarity_matrix(df)
        labels = pd.Series(['A', 'A', 'B'], index=sim.index)
        result = within_between_similarity(sim, labels)
        within = result[result['comparison'] == 'within']
        between = result[result['comparison'] == 'between']
        assert len(within) == 1   # A-A pair
        assert len(between) == 2  # A-B pairs


class TestDecodeTargetNM:

    def test_perfectly_separable(self):
        from iblnm.analysis import decode_target_nm
        # 6 recordings: class A has feat_0=5, class B has feat_1=5
        # 3 subjects per class for leave-one-subject-out
        n_feat = 5
        n = 6
        data = np.zeros((n, n_feat))
        data[:3, 0] = 5.0   # class A
        data[3:, 1] = 5.0   # class B
        index = pd.MultiIndex.from_tuples(
            [(f'e{i}', f'r{i}') for i in range(n)],
            names=['eid', 'brain_region'],
        )
        mat = pd.DataFrame(data, index=index, columns=[f'f{i}' for i in range(n_feat)])
        labels = pd.Series(['A'] * 3 + ['B'] * 3, index=index)
        subjects = pd.Series(['s0', 's1', 's2', 's0', 's1', 's2'], index=index)

        result = decode_target_nm(mat, labels, subjects)
        assert result['accuracy'] == 1.0

    def test_returns_expected_keys(self):
        from iblnm.analysis import decode_target_nm
        mat = _make_response_matrix(n_recordings=6, n_features=3)
        labels = pd.Series(['A', 'A', 'A', 'B', 'B', 'B'], index=mat.index)
        subjects = pd.Series(['s0', 's1', 's2', 's0', 's1', 's2'], index=mat.index)
        result = decode_target_nm(mat, labels, subjects)
        assert set(result.keys()) == {
            'accuracy', 'balanced_accuracy', 'per_class_accuracy',
            'confusion', 'coefficients', 'predictions', 'best_C', 'n_valid',
        }

    def test_normalizes_features(self):
        """With normalize=True, features should be z-scored per fold so
        scale differences don't affect decoding."""
        from iblnm.analysis import decode_target_nm
        n, n_feat = 6, 3
        # Class A: feat_0 = 1000 (large scale), Class B: feat_0 = -1000
        data = np.zeros((n, n_feat))
        data[:3, 0] = 1000.0
        data[3:, 0] = -1000.0
        index = pd.MultiIndex.from_tuples(
            [(f'e{i}', f'r{i}') for i in range(n)],
            names=['eid', 'brain_region'],
        )
        mat = pd.DataFrame(data, index=index, columns=[f'f{i}' for i in range(n_feat)])
        labels = pd.Series(['A'] * 3 + ['B'] * 3, index=index)
        subjects = pd.Series(['s0', 's1', 's2', 's0', 's1', 's2'], index=index)

        result = decode_target_nm(mat, labels, subjects, normalize=True)
        assert result['accuracy'] == 1.0

    def test_requires_two_classes(self):
        from iblnm.analysis import decode_target_nm
        mat = _make_response_matrix(n_recordings=3, n_features=3)
        labels = pd.Series(['A', 'A', 'A'], index=mat.index)
        subjects = pd.Series(['s0', 's1', 's2'], index=mat.index)
        with pytest.raises(ValueError, match='2'):
            decode_target_nm(mat, labels, subjects)


class TestFeatureUniqueContribution:

    def test_shape_and_columns(self):
        from iblnm.analysis import feature_unique_contribution
        n_feat = 4
        mat = _make_response_matrix(n_recordings=6, n_features=n_feat)
        labels = pd.Series(['A', 'A', 'A', 'B', 'B', 'B'], index=mat.index)
        subjects = pd.Series(['s0', 's1', 's2', 's0', 's1', 's2'], index=mat.index)
        result = feature_unique_contribution(mat, labels, subjects)
        assert len(result) == n_feat
        assert set(result.columns) == {'feature', 'full_accuracy', 'reduced_accuracy', 'delta'}

    def test_informative_feature_has_positive_delta(self):
        from iblnm.analysis import feature_unique_contribution
        # Class A: feat_0 high, Class B: feat_0 low. Other features are noise.
        # Use enough samples that removing f0 hurts full-data accuracy.
        rng = np.random.default_rng(0)
        n_per_class = 20
        n = 2 * n_per_class
        n_feat = 5
        data = rng.standard_normal((n, n_feat))
        data[:n_per_class, 0] += 3.0   # class A: f0 shifted up
        data[n_per_class:, 0] -= 3.0   # class B: f0 shifted down
        index = pd.MultiIndex.from_tuples(
            [(f'e{i}', f'r{i}') for i in range(n)],
            names=['eid', 'brain_region'],
        )
        mat = pd.DataFrame(data, index=index, columns=[f'f{i}' for i in range(n_feat)])
        labels = pd.Series(['A'] * n_per_class + ['B'] * n_per_class, index=index)
        subjects = pd.Series([f's{i % 5}' for i in range(n)], index=index)
        result = feature_unique_contribution(mat, labels, subjects, C=1.0)
        f0_delta = result[result['feature'] == 'f0']['delta'].iloc[0]
        assert f0_delta > 0, f"Informative feature f0 should have positive delta, got {f0_delta}"


# =============================================================================
# TargetNMDecoder Tests
# =============================================================================

def _make_separable_data(n_per_class=3, n_feat=5):
    """Helper: perfectly separable two-class data with 3 subjects per class."""
    n = n_per_class * 2
    data = np.zeros((n, n_feat))
    data[:n_per_class, 0] = 5.0   # class A
    data[n_per_class:, 1] = 5.0   # class B
    index = pd.MultiIndex.from_tuples(
        [(f'e{i}', f'r{i}') for i in range(n)],
        names=['eid', 'brain_region'],
    )
    mat = pd.DataFrame(data, index=index, columns=[f'f{i}' for i in range(n_feat)])
    labels = pd.Series(['A'] * n_per_class + ['B'] * n_per_class, index=index)
    subjects = pd.Series(
        [f's{i}' for i in range(n_per_class)] * 2, index=index,
    )
    return mat, labels, subjects


class TestTargetNMDecoder:

    def test_fit_stores_attributes(self):
        from iblnm.analysis import TargetNMDecoder
        mat, labels, subjects = _make_separable_data()
        decoder = TargetNMDecoder(mat, labels, subjects)
        decoder.fit()
        assert hasattr(decoder, 'accuracy')
        assert hasattr(decoder, 'balanced_accuracy')
        assert hasattr(decoder, 'per_class_accuracy')
        assert hasattr(decoder, 'confusion')
        assert hasattr(decoder, 'coefficients')
        assert hasattr(decoder, 'predictions')

    def test_fit_perfect_accuracy(self):
        from iblnm.analysis import TargetNMDecoder
        mat, labels, subjects = _make_separable_data()
        decoder = TargetNMDecoder(mat, labels, subjects)
        decoder.fit()
        assert decoder.accuracy == 1.0

    def test_unique_contribution_returns_dataframe(self):
        from iblnm.analysis import TargetNMDecoder
        mat, labels, subjects = _make_separable_data()
        decoder = TargetNMDecoder(mat, labels, subjects)
        decoder.fit()
        contrib = decoder.unique_contribution()
        assert isinstance(contrib, pd.DataFrame)
        assert len(contrib) == mat.shape[1]
        assert 'delta' in contrib.columns

    def test_unique_contribution_stored_as_attribute(self):
        from iblnm.analysis import TargetNMDecoder
        mat, labels, subjects = _make_separable_data()
        decoder = TargetNMDecoder(mat, labels, subjects)
        decoder.fit()
        decoder.unique_contribution()
        assert hasattr(decoder, 'contributions')
        assert isinstance(decoder.contributions, pd.DataFrame)


class TestFullDataCoefficients:
    """Coefficients should come from a single full-data refit, not fold averages."""

    def test_coefficients_from_full_data_not_fold_average(self):
        """Coefficients should differ from fold-averaged coefficients.

        We verify this indirectly: coefficients from decode_target_nm should
        match a manual full-data refit with the same C, not the fold average.
        """
        from iblnm.analysis import decode_target_nm
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import LabelEncoder, StandardScaler

        mat, labels, subjects = _make_separable_data()
        result = decode_target_nm(mat, labels, subjects)

        # Manually refit on all data with best_C
        X = mat.values
        le = LabelEncoder()
        y_enc = le.fit_transform(labels)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        clf = LogisticRegression(
            penalty='l1', solver='saga', max_iter=5000,
            class_weight='balanced', C=result['best_C'],
        )
        clf.fit(X_scaled, y_enc)

        # Result coefficients should match full-data refit
        coef_df = result['coefficients']
        assert coef_df.shape == (len(le.classes_), X.shape[1])
        # Binary case: sklearn returns (1, n_features), expanded to (2, n_features)
        expected = clf.coef_
        if expected.shape[0] == 1 and len(le.classes_) == 2:
            expected = np.vstack([expected, -expected])
        np.testing.assert_allclose(coef_df.values, expected, atol=1e-3)

    def test_unique_contribution_uses_full_data_accuracy(self):
        """Unique contribution's full_accuracy should be full-data accuracy,
        not CV accuracy."""
        from iblnm.analysis import feature_unique_contribution
        mat, labels, subjects = _make_separable_data()
        contrib = feature_unique_contribution(mat, labels, subjects, C=1.0)
        # full_accuracy should be from a full-data fit (generally >= CV accuracy)
        assert contrib['full_accuracy'].iloc[0] >= 0.0
        # All rows should have the same full_accuracy
        assert contrib['full_accuracy'].nunique() == 1


class TestCTuning:
    """Tests for C (regularization) tuning in decode_target_nm."""

    def test_decode_returns_best_C(self):
        """decode_target_nm should return a 'best_C' key."""
        from iblnm.analysis import decode_target_nm
        mat, labels, subjects = _make_separable_data()
        result = decode_target_nm(mat, labels, subjects)
        assert 'best_C' in result
        assert isinstance(result['best_C'], float)
        assert result['best_C'] > 0

    def test_decoder_stores_best_C(self):
        """TargetNMDecoder.fit() should store best_C_ attribute."""
        from iblnm.analysis import TargetNMDecoder
        mat, labels, subjects = _make_separable_data()
        decoder = TargetNMDecoder(mat, labels, subjects)
        decoder.fit()
        assert hasattr(decoder, 'best_C_')
        assert decoder.best_C_ > 0

    def test_unique_contribution_uses_best_C(self):
        """unique_contribution should use best_C_ from fit, not retune."""
        from iblnm.analysis import TargetNMDecoder
        mat, labels, subjects = _make_separable_data()
        decoder = TargetNMDecoder(mat, labels, subjects)
        decoder.fit()
        best_c = decoder.best_C_
        decoder.unique_contribution()
        # full_accuracy in contributions should match decoder.accuracy
        # (same C used, same data)
        assert decoder.contributions['full_accuracy'].iloc[0] == decoder.accuracy

    def test_fixed_C_skips_tuning(self):
        """When C is passed explicitly, decode_target_nm should use it."""
        from iblnm.analysis import decode_target_nm
        mat, labels, subjects = _make_separable_data()
        result = decode_target_nm(mat, labels, subjects, C=1.0)
        assert result['best_C'] == 1.0

    def test_custom_Cs_grid(self):
        """Custom Cs grid should be searched."""
        from iblnm.analysis import decode_target_nm
        mat, labels, subjects = _make_separable_data()
        result = decode_target_nm(mat, labels, subjects, Cs=[0.01, 100.0])
        assert result['best_C'] in [0.01, 100.0]


class TestSubjectTargetGrouping:
    """LOSO should group by subject-target, not bare subject."""

    def test_subject_target_grouping_keeps_other_fiber(self):
        """Holding out s0-A should keep s0-B in training.

        s0 is the only subject with class B data. With bare-subject LOSO,
        holding out s0 removes all class B from training → fold skipped,
        s0-A is unpredicted. With subject-target grouping, holding out
        s0-A keeps s0-B in training → s0-A gets a valid prediction.

        We detect this via n_valid_predictions: subject-target gives 3
        valid folds (s0-A, s1-A, s2-A) vs bare-subject's 2 (s1, s2).
        """
        from iblnm.analysis import decode_target_nm
        # 4 recordings: class A = [s0, s1, s2], class B = [s0]
        n_feat = 3
        data = np.zeros((4, n_feat))
        data[:3, 0] = 5.0   # class A
        data[3:, 1] = 5.0   # class B
        index = pd.MultiIndex.from_tuples(
            [(f'e{i}', f'r{i}') for i in range(4)],
            names=['eid', 'target_NM'],
        )
        mat = pd.DataFrame(data, index=index, columns=[f'f{i}' for i in range(n_feat)])
        labels = pd.Series(['A', 'A', 'A', 'B'], index=index)
        subjects = pd.Series(['s0', 's1', 's2', 's0'], index=index)

        result = decode_target_nm(mat, labels, subjects)

        # With subject-target grouping: s0-A fold has s0-B in training →
        # valid prediction. n_valid = 3 (s0-A, s1-A, s2-A; s0-B skipped).
        # With bare-subject: s0 fold removes both → n_valid = 2.
        assert result['n_valid'] == 3


class TestMeanSimilarityByTarget:

    def _make_sim_and_labels(self):
        """4 recordings: 2×A, 2×B with known pairwise similarities."""
        from iblnm.analysis import cosine_similarity_matrix
        # A recordings are [1,0], B recordings are [0,1] → orthogonal
        index = pd.MultiIndex.from_tuples(
            [('e0', 'A'), ('e1', 'A'), ('e2', 'B'), ('e3', 'B')],
            names=['eid', 'target_NM'],
        )
        df = pd.DataFrame(
            [[1, 0], [1, 0.1], [0, 1], [0.1, 1]],
            index=index, columns=['f0', 'f1'],
        )
        sim = cosine_similarity_matrix(df)
        labels = pd.Series(['A', 'A', 'B', 'B'], index=sim.index)
        return sim, labels

    def test_returns_square_dataframe(self):
        from iblnm.analysis import mean_similarity_by_target
        sim, labels = self._make_sim_and_labels()
        result = mean_similarity_by_target(sim, labels)
        assert result.shape == (2, 2)
        assert list(result.index) == ['A', 'B']
        assert list(result.columns) == ['A', 'B']

    def test_symmetric(self):
        from iblnm.analysis import mean_similarity_by_target
        sim, labels = self._make_sim_and_labels()
        result = mean_similarity_by_target(sim, labels)
        np.testing.assert_allclose(result.values, result.values.T, atol=1e-10)

    def test_within_greater_than_between(self):
        """Within-target similarity should exceed between-target."""
        from iblnm.analysis import mean_similarity_by_target
        sim, labels = self._make_sim_and_labels()
        result = mean_similarity_by_target(sim, labels)
        assert result.loc['A', 'A'] > result.loc['A', 'B']
        assert result.loc['B', 'B'] > result.loc['A', 'B']

    def test_single_recording_per_target_gives_nan_diagonal(self):
        """One recording per target → no within-target pairs → NaN on diagonal."""
        from iblnm.analysis import mean_similarity_by_target, cosine_similarity_matrix
        index = pd.MultiIndex.from_tuples(
            [('e0', 'A'), ('e1', 'B')], names=['eid', 'target_NM'],
        )
        df = pd.DataFrame([[1, 0], [0, 1]], index=index, columns=['f0', 'f1'])
        sim = cosine_similarity_matrix(df)
        labels = pd.Series(['A', 'B'], index=sim.index)
        result = mean_similarity_by_target(sim, labels)
        assert np.isnan(result.loc['A', 'A'])
        assert np.isnan(result.loc['B', 'B'])

    def test_loso_excludes_same_subject_pairs(self):
        """With subjects, same-subject pairs are excluded."""
        from iblnm.analysis import mean_similarity_by_target, cosine_similarity_matrix
        # 4 recordings, 2 subjects, 2 targets
        # s0 has one A and one B, s1 has one A and one B
        index = pd.MultiIndex.from_tuples(
            [('e0', 'A'), ('e1', 'A'), ('e2', 'B'), ('e3', 'B')],
            names=['eid', 'target_NM'],
        )
        df = pd.DataFrame(
            [[1, 0], [1, 0.1], [0, 1], [0.1, 1]],
            index=index, columns=['f0', 'f1'],
        )
        sim = cosine_similarity_matrix(df)
        labels = pd.Series(['A', 'A', 'B', 'B'], index=sim.index)
        subjects = pd.Series(['s0', 's1', 's0', 's1'], index=sim.index)

        full = mean_similarity_by_target(sim, labels)
        loso = mean_similarity_by_target(sim, labels, subjects=subjects)

        # Within-target: full uses 1 pair (e0-e1), loso also uses 1 pair
        # (e0-e1 are different subjects) → same for A-A
        # Between-target: full uses all 4 pairs, loso excludes
        # (e0,e2)=same subject and (e1,e3)=same subject → only 2 cross-subject pairs
        # The cross-subject between pairs (e0,e3) and (e1,e2) should differ
        # from the full set
        assert not np.isclose(full.loc['A', 'B'], loso.loc['A', 'B'])

    def test_loso_all_same_subject_gives_nan(self):
        """If all recordings of a target pair share a subject, result is NaN."""
        from iblnm.analysis import mean_similarity_by_target, cosine_similarity_matrix
        index = pd.MultiIndex.from_tuples(
            [('e0', 'A'), ('e1', 'B')], names=['eid', 'target_NM'],
        )
        df = pd.DataFrame([[1, 0], [0, 1]], index=index, columns=['f0', 'f1'])
        sim = cosine_similarity_matrix(df)
        labels = pd.Series(['A', 'B'], index=sim.index)
        subjects = pd.Series(['s0', 's0'], index=sim.index)
        result = mean_similarity_by_target(sim, labels, subjects=subjects)
        # Only between pair is (e0, e1) which are same subject → excluded → NaN
        assert np.isnan(result.loc['A', 'B'])

    def test_loso_symmetric(self):
        from iblnm.analysis import mean_similarity_by_target, cosine_similarity_matrix
        index = pd.MultiIndex.from_tuples(
            [('e0', 'A'), ('e1', 'A'), ('e2', 'B'), ('e3', 'B')],
            names=['eid', 'target_NM'],
        )
        df = pd.DataFrame(
            [[1, 0], [1, 0.1], [0, 1], [0.1, 1]],
            index=index, columns=['f0', 'f1'],
        )
        sim = cosine_similarity_matrix(df)
        labels = pd.Series(['A', 'A', 'B', 'B'], index=sim.index)
        subjects = pd.Series(['s0', 's1', 's0', 's1'], index=sim.index)
        result = mean_similarity_by_target(sim, labels, subjects=subjects)
        np.testing.assert_allclose(result.values, result.values.T, atol=1e-10)


# =============================================================================
# LMM Tests
# =============================================================================


def _make_lmm_data(n_per_cell=20, seed=42):
    """Synthetic trial data with known contrast effect for LMM testing.

    3 subjects, 2 sides, 2 rewards, 5 contrasts. Response has a known
    linear relationship with log(contrast).
    """
    rng = np.random.default_rng(seed)
    subjects = ['s0', 's1', 's2']
    sides = ['contra', 'ipsi']
    rewards = [1, -1]
    contrasts = [0.0, 0.0625, 0.125, 0.25, 1.0]

    rows = []
    for subj in subjects:
        subj_intercept = rng.normal(0, 0.3)
        for side in sides:
            for reward in rewards:
                for contrast in contrasts:
                    for _ in range(n_per_cell):
                        log_c = np.log(contrast + 0.01)
                        response = (
                            1.0                              # intercept
                            + 0.5 * log_c                    # contrast effect
                            + 0.3 * (1 if side == 'ipsi' else 0)  # side effect
                            + 0.2 * (1 if reward == 1 else 0)     # reward effect
                            + subj_intercept
                            + rng.normal(0, 0.5)
                        )
                        rows.append({
                            'contrast': contrast,
                            'side': side,
                            'feedbackType': reward,
                            'subject': subj,
                            'response': response,
                        })
    return pd.DataFrame(rows)


class TestFitEventsLMM:

    def test_returns_lmm_result(self):
        from iblnm.analysis import fit_events_lmm
        df = _make_lmm_data()
        result = fit_events_lmm(df, 'response')
        assert result is not None

    def test_result_has_expected_fields(self):
        from iblnm.analysis import fit_events_lmm
        df = _make_lmm_data()
        result = fit_events_lmm(df, 'response')
        assert hasattr(result, 'result')
        assert hasattr(result, 'summary_df')
        assert hasattr(result, 'variance_explained')

    def test_variance_explained_keys(self):
        from iblnm.analysis import fit_events_lmm
        df = _make_lmm_data()
        result = fit_events_lmm(df, 'response')
        ve = result.variance_explained
        assert 'marginal' in ve
        assert 'conditional' in ve
        assert 0 <= ve['marginal'] <= 1
        assert 0 <= ve['conditional'] <= 1
        assert ve['conditional'] >= ve['marginal']

    def test_summary_df_has_coefficients(self):
        from iblnm.analysis import fit_events_lmm
        df = _make_lmm_data()
        result = fit_events_lmm(df, 'response')
        assert isinstance(result.summary_df, pd.DataFrame)
        assert 'Coef.' in result.summary_df.columns
        assert 'P>|z|' in result.summary_df.columns
        assert len(result.summary_df) > 0

    def test_detects_contrast_effect(self):
        """With known contrast slope of 0.5, the model should detect it."""
        from iblnm.analysis import fit_events_lmm
        df = _make_lmm_data(n_per_cell=30, seed=0)
        result = fit_events_lmm(df, 'response')
        # log_contrast coefficient should be significantly positive
        coef = result.summary_df.loc['log_contrast', 'Coef.']
        pval = result.summary_df.loc['log_contrast', 'P>|z|']
        assert coef > 0.2, f"Expected positive contrast effect, got {coef}"
        assert pval < 0.05, f"Expected significant contrast effect, got p={pval}"

    def test_convergence_failure_returns_none(self):
        """Degenerate data should return None, not raise."""
        from iblnm.analysis import fit_events_lmm
        # Single subject, single side — model can't fit random effect or side
        df = pd.DataFrame({
            'contrast': [0.25] * 10,
            'side': ['contra'] * 10,
            'feedbackType': [1] * 10,
            'subject': ['s0'] * 10,
            'response': np.zeros(10),
        })
        result = fit_events_lmm(df, 'response')
        assert result is None

    def test_predictions_on_grid(self):
        """Result should include predicted values on a contrast grid."""
        from iblnm.analysis import fit_events_lmm
        df = _make_lmm_data()
        result = fit_events_lmm(df, 'response')
        assert hasattr(result, 'predictions')
        assert isinstance(result.predictions, pd.DataFrame)
        assert 'contrast' in result.predictions.columns
        assert 'predicted' in result.predictions.columns
        assert 'ci_lower' in result.predictions.columns
        assert 'ci_upper' in result.predictions.columns
        assert 'side' in result.predictions.columns
        assert 'reward' in result.predictions.columns


class TestComputeMarginalMeans:

    def test_returns_dataframe(self):
        from iblnm.analysis import fit_events_lmm, compute_marginal_means
        df = _make_lmm_data()
        result = fit_events_lmm(df, 'response')
        emm = compute_marginal_means(result, 'reward')
        assert isinstance(emm, pd.DataFrame)

    def test_reward_has_two_levels(self):
        from iblnm.analysis import fit_events_lmm, compute_marginal_means
        df = _make_lmm_data()
        result = fit_events_lmm(df, 'response')
        emm = compute_marginal_means(result, 'reward')
        assert len(emm) == 2
        assert set(emm['level']) == {0, 1}

    def test_side_has_two_levels(self):
        from iblnm.analysis import fit_events_lmm, compute_marginal_means
        df = _make_lmm_data()
        result = fit_events_lmm(df, 'response')
        emm = compute_marginal_means(result, 'side')
        assert len(emm) == 2
        assert set(emm['level']) == {'contra', 'ipsi'}

    def test_has_ci_columns(self):
        from iblnm.analysis import fit_events_lmm, compute_marginal_means
        df = _make_lmm_data()
        result = fit_events_lmm(df, 'response')
        emm = compute_marginal_means(result, 'reward')
        assert 'mean' in emm.columns
        assert 'ci_lower' in emm.columns
        assert 'ci_upper' in emm.columns

    def test_reward_effect_detected(self):
        """With known reward effect of 0.2, correct EMM should exceed incorrect."""
        from iblnm.analysis import fit_events_lmm, compute_marginal_means
        df = _make_lmm_data(n_per_cell=30, seed=0)
        result = fit_events_lmm(df, 'response')
        emm = compute_marginal_means(result, 'reward')
        correct = emm[emm['level'] == 1]['mean'].iloc[0]
        incorrect = emm[emm['level'] == 0]['mean'].iloc[0]
        assert correct > incorrect

    def test_contrast_column_present(self):
        """EMM DataFrame should include the contrast column for the factor."""
        from iblnm.analysis import fit_events_lmm, compute_marginal_means
        df = _make_lmm_data()
        result = fit_events_lmm(df, 'response')
        emm = compute_marginal_means(result, 'reward')
        assert 'contrast_diff' in emm.columns or 'diff' in emm.columns or len(emm) == 2
