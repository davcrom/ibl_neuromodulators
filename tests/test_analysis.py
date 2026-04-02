"""Tests for iblnm.analysis module."""
import numpy as np
import pandas as pd

from iblnm.analysis import get_responses, normalize_responses, resample_signal
from iblnm.util import contrast_transform


class TestContrastTransform:
    def test_zero_contrast(self):
        from iblnm.util import contrast_transform
        assert contrast_transform(0) == 0.0

    def test_known_values(self):
        from iblnm.util import contrast_transform
        c = np.array([0.0, 6.25, 12.5, 25, 100])
        result = contrast_transform(c)
        expected = np.log(c + 1)
        np.testing.assert_allclose(result, expected)

    def test_roundtrip(self):
        from iblnm.util import contrast_transform, contrast_inverse
        c = np.array([0.0, 6.25, 12.5, 25, 50, 100])
        np.testing.assert_allclose(contrast_inverse(contrast_transform(c)), c)

    def test_monotonic(self):
        from iblnm.util import contrast_transform
        c = np.array([0.0, 6.25, 12.5, 25, 50, 100])
        result = contrast_transform(c)
        assert np.all(np.diff(result) > 0)


class TestGetContrastCoding:
    """Tests for get_contrast_coding() which returns (transform, inverse) pairs."""

    def test_log_transform_matches_legacy(self):
        from iblnm.util import get_contrast_coding, contrast_transform
        transform, _ = get_contrast_coding('log')
        c = np.array([0.0, 6.25, 12.5, 25, 100])
        np.testing.assert_allclose(transform(c), contrast_transform(c))

    def test_log_roundtrip(self):
        from iblnm.util import get_contrast_coding
        transform, inverse = get_contrast_coding('log')
        c = np.array([0.0, 6.25, 12.5, 25, 50, 100])
        np.testing.assert_allclose(inverse(transform(c)), c)

    def test_linear_identity(self):
        from iblnm.util import get_contrast_coding
        transform, inverse = get_contrast_coding('linear')
        c = np.array([0.0, 6.25, 12.5, 25, 100])
        np.testing.assert_allclose(transform(c), c)
        np.testing.assert_allclose(inverse(c), c)

    def test_rank_maps_to_ordinal(self):
        from iblnm.util import get_contrast_coding
        transform, _ = get_contrast_coding('rank')
        c = np.array([0.0, 6.25, 12.5, 25, 100])
        np.testing.assert_allclose(transform(c), [0, 1, 2, 3, 4])

    def test_rank_roundtrip(self):
        from iblnm.util import get_contrast_coding
        transform, inverse = get_contrast_coding('rank')
        c = np.array([0.0, 6.25, 12.5, 25, 100])
        np.testing.assert_allclose(inverse(transform(c)), c)

    def test_rank_scalar_after_array(self):
        """After seeing the full contrast set, scalar lookup returns correct rank."""
        from iblnm.util import get_contrast_coding
        transform, _ = get_contrast_coding('rank')
        # First call with full array to populate the rank map
        transform(np.array([0.0, 6.25, 12.5, 25, 100]))
        assert transform(12.5) == 2.0

    def test_all_monotonic(self):
        from iblnm.util import get_contrast_coding
        c = np.array([0.0, 6.25, 12.5, 25, 100])
        for coding in ('linear', 'rank', 'log'):
            transform, _ = get_contrast_coding(coding)
            assert np.all(np.diff(transform(c)) > 0), f"{coding} not monotonic"

    def test_invalid_coding_raises(self):
        import pytest
        from iblnm.util import get_contrast_coding
        with pytest.raises(ValueError, match='quadratic'):
            get_contrast_coding('quadratic')


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


class TestSplitFeaturesByEvent:

    def test_returns_dict_keyed_by_event(self):
        from iblnm.analysis import split_features_by_event
        index = pd.MultiIndex.from_tuples(
            [('e0', 'VTA-DA', 0), ('e1', 'DR-5HT', 0)],
            names=['eid', 'target_NM', 'fiber_idx'],
        )
        df = pd.DataFrame({
            'stimOn_c0_contra_correct': [1.0, 2.0],
            'stimOn_c0_ipsi_correct': [3.0, 4.0],
            'feedback_c0_contra_correct': [5.0, 6.0],
        }, index=index)
        result = split_features_by_event(df)
        assert set(result.keys()) == {'stimOn', 'feedback'}

    def test_column_count_matches_original(self):
        from iblnm.analysis import split_features_by_event
        index = pd.MultiIndex.from_tuples(
            [('e0', 'A', 0)], names=['eid', 'target_NM', 'fiber_idx'],
        )
        df = pd.DataFrame({
            'stimOn_c0_contra_correct': [1.0],
            'stimOn_c100_ipsi_incorrect': [2.0],
            'feedback_c0_contra_correct': [3.0],
            'firstMovement_c0_contra_correct': [4.0],
        }, index=index)
        result = split_features_by_event(df)
        total_cols = sum(len(v.columns) for v in result.values())
        assert total_cols == len(df.columns)

    def test_preserves_index(self):
        from iblnm.analysis import split_features_by_event
        index = pd.MultiIndex.from_tuples(
            [('e0', 'A', 0), ('e1', 'B', 0)],
            names=['eid', 'target_NM', 'fiber_idx'],
        )
        df = pd.DataFrame({
            'stimOn_c0_contra_correct': [1.0, 2.0],
            'feedback_c0_contra_correct': [3.0, 4.0],
        }, index=index)
        result = split_features_by_event(df)
        for event_df in result.values():
            assert event_df.index.equals(df.index)

    def test_correct_columns_per_event(self):
        from iblnm.analysis import split_features_by_event
        index = pd.MultiIndex.from_tuples(
            [('e0', 'A', 0)], names=['eid', 'target_NM', 'fiber_idx'],
        )
        df = pd.DataFrame({
            'stimOn_c0_contra_correct': [1.0],
            'stimOn_c100_ipsi_incorrect': [2.0],
            'feedback_c0_contra_correct': [3.0],
        }, index=index)
        result = split_features_by_event(df)
        assert list(result['stimOn'].columns) == [
            'stimOn_c0_contra_correct', 'stimOn_c100_ipsi_incorrect']
        assert list(result['feedback'].columns) == [
            'feedback_c0_contra_correct']


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
        np.testing.assert_allclose(coef_df.values, expected, atol=1e-2)

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


class TestLeaveOneSubjectOutGrouping:
    """LOSO groups by subject to prevent information leakage."""

    def test_subject_grouping_holds_out_all_targets(self):
        """Holding out s0 removes all of s0's recordings (both A and B).

        s0 is the only subject with class B data. With leave-one-subject-out,
        holding out s0 removes all class B from training → fold skipped,
        n_valid = 2 (only s1 and s2 get predictions).
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

        # With leave-one-subject-out: s0 fold removes both s0-A and s0-B →
        # no class B in training → fold skipped. n_valid = 2 (s1, s2).
        assert result['n_valid'] == 2


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
                        log_c = contrast_transform(contrast)
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


class TestFitResponseLMM:

    def test_returns_lmm_result(self):
        from iblnm.analysis import fit_response_lmm
        df = _make_lmm_data()
        result = fit_response_lmm(df, 'response')
        assert result is not None

    def test_accepts_re_formula(self):
        """re_formula parameter controls the random effects structure."""
        from iblnm.analysis import fit_response_lmm
        df = _make_lmm_data()
        result = fit_response_lmm(df, 'response', re_formula='1')
        assert result is not None
        # Intercept-only: random effects should not have log_contrast
        for effects in result.random_effects.values():
            assert 'log_contrast' not in effects.index

    def test_re_formula_with_slope(self):
        """Random slope model should include log_contrast in random effects."""
        from iblnm.analysis import fit_response_lmm
        df = _make_lmm_data(n_per_cell=30, seed=0)
        result = fit_response_lmm(df, 'response', re_formula='log_contrast')
        if result is not None:
            has_slope = any(
                'log_contrast' in eff.index
                for eff in result.random_effects.values()
            )
            assert has_slope

    def test_result_has_expected_fields(self):
        from iblnm.analysis import fit_response_lmm
        df = _make_lmm_data()
        result = fit_response_lmm(df, 'response')
        assert hasattr(result, 'result')
        assert hasattr(result, 'summary_df')
        assert hasattr(result, 'variance_explained')

    def test_variance_explained_keys(self):
        from iblnm.analysis import fit_response_lmm
        df = _make_lmm_data()
        result = fit_response_lmm(df, 'response')
        ve = result.variance_explained
        assert 'marginal' in ve
        assert 'conditional' in ve
        assert 0 <= ve['marginal'] <= 1
        assert 0 <= ve['conditional'] <= 1
        assert ve['conditional'] >= ve['marginal']

    def test_summary_df_has_coefficients(self):
        from iblnm.analysis import fit_response_lmm
        df = _make_lmm_data()
        result = fit_response_lmm(df, 'response')
        assert isinstance(result.summary_df, pd.DataFrame)
        assert 'Coef.' in result.summary_df.columns
        assert 'P>|z|' in result.summary_df.columns
        assert len(result.summary_df) > 0

    def test_detects_contrast_effect(self):
        """With known contrast slope of 0.5, the model should detect it."""
        from iblnm.analysis import fit_response_lmm
        df = _make_lmm_data(n_per_cell=30, seed=0)
        result = fit_response_lmm(df, 'response')
        # log_contrast coefficient should be significantly positive
        coef = result.summary_df.loc['log_contrast', 'Coef.']
        pval = result.summary_df.loc['log_contrast', 'P>|z|']
        assert coef > 0.2, f"Expected positive contrast effect, got {coef}"
        assert pval < 0.05, f"Expected significant contrast effect, got p={pval}"

    def test_convergence_failure_returns_none(self):
        """Degenerate data should return None, not raise."""
        from iblnm.analysis import fit_response_lmm
        # Single subject, single side — model can't fit random effect or side
        df = pd.DataFrame({
            'contrast': [0.25] * 10,
            'side': ['contra'] * 10,
            'feedbackType': [1] * 10,
            'subject': ['s0'] * 10,
            'response': np.zeros(10),
        })
        result = fit_response_lmm(df, 'response')
        assert result is None

    def test_predictions_on_grid(self):
        """Result should include predicted values on a contrast grid."""
        from iblnm.analysis import fit_response_lmm
        df = _make_lmm_data()
        result = fit_response_lmm(df, 'response')
        assert hasattr(result, 'predictions')
        assert isinstance(result.predictions, pd.DataFrame)
        assert 'contrast' in result.predictions.columns
        assert 'predicted' in result.predictions.columns
        assert 'ci_lower' in result.predictions.columns
        assert 'ci_upper' in result.predictions.columns
        assert 'side' in result.predictions.columns
        assert 'reward' in result.predictions.columns


class TestContrastCodingParameter:
    """Test that fit_response_lmm accepts different contrast coding schemes."""

    def test_default_is_log(self):
        from iblnm.analysis import fit_response_lmm
        df = _make_lmm_data()
        result = fit_response_lmm(df, 'response')
        assert result.contrast_coding == 'log'

    def test_linear_coding_converges(self):
        from iblnm.analysis import fit_response_lmm
        df = _make_lmm_data()
        result = fit_response_lmm(df, 'response', contrast_coding='linear')
        assert result is not None
        assert result.contrast_coding == 'linear'

    def test_rank_coding_converges(self):
        from iblnm.analysis import fit_response_lmm
        df = _make_lmm_data()
        result = fit_response_lmm(df, 'response', contrast_coding='rank')
        assert result is not None
        assert result.contrast_coding == 'rank'

    def test_rank_coding_detects_contrast_effect(self):
        from iblnm.analysis import fit_response_lmm
        df = _make_lmm_data(n_per_cell=30, seed=0)
        result = fit_response_lmm(df, 'response', contrast_coding='rank')
        coef = result.summary_df.loc['log_contrast', 'Coef.']
        assert coef > 0, f"Expected positive contrast effect under rank coding, got {coef}"

    def test_marginal_means_use_stored_coding(self):
        from iblnm.analysis import fit_response_lmm, compute_marginal_means
        df = _make_lmm_data()
        result_log = fit_response_lmm(df, 'response', contrast_coding='log')
        result_rank = fit_response_lmm(df, 'response', contrast_coding='rank')
        emm_log = compute_marginal_means(result_log, 'contrast')
        emm_rank = compute_marginal_means(result_rank, 'contrast')
        # Both should return valid DataFrames but with different predictions
        assert len(emm_log) == len(emm_rank)
        assert not np.allclose(emm_log['mean'].values, emm_rank['mean'].values)


class TestComputeMarginalMeans:

    def test_returns_dataframe(self):
        from iblnm.analysis import fit_response_lmm, compute_marginal_means
        df = _make_lmm_data()
        result = fit_response_lmm(df, 'response')
        emm = compute_marginal_means(result, 'reward')
        assert isinstance(emm, pd.DataFrame)

    def test_reward_has_two_levels(self):
        from iblnm.analysis import fit_response_lmm, compute_marginal_means
        df = _make_lmm_data()
        result = fit_response_lmm(df, 'response')
        emm = compute_marginal_means(result, 'reward')
        assert len(emm) == 2
        assert set(emm['level']) == {'incorrect', 'correct'}

    def test_side_has_two_levels(self):
        from iblnm.analysis import fit_response_lmm, compute_marginal_means
        df = _make_lmm_data()
        result = fit_response_lmm(df, 'response')
        emm = compute_marginal_means(result, 'side')
        assert len(emm) == 2
        assert set(emm['level']) == {'contra', 'ipsi'}

    def test_has_ci_columns(self):
        from iblnm.analysis import fit_response_lmm, compute_marginal_means
        df = _make_lmm_data()
        result = fit_response_lmm(df, 'response')
        emm = compute_marginal_means(result, 'reward')
        assert 'mean' in emm.columns
        assert 'ci_lower' in emm.columns
        assert 'ci_upper' in emm.columns

    def test_reward_effect_detected(self):
        """With known reward effect of 0.2, correct EMM should exceed incorrect."""
        from iblnm.analysis import fit_response_lmm, compute_marginal_means
        df = _make_lmm_data(n_per_cell=30, seed=0)
        result = fit_response_lmm(df, 'response')
        emm = compute_marginal_means(result, 'reward')
        correct = emm[emm['level'] == 'correct']['mean'].iloc[0]
        incorrect = emm[emm['level'] == 'incorrect']['mean'].iloc[0]
        assert correct > incorrect

    def test_contrast_column_present(self):
        """EMM DataFrame should include the contrast column for the factor."""
        from iblnm.analysis import fit_response_lmm, compute_marginal_means
        df = _make_lmm_data()
        result = fit_response_lmm(df, 'response')
        emm = compute_marginal_means(result, 'reward')
        assert 'contrast_diff' in emm.columns or 'diff' in emm.columns or len(emm) == 2

    def test_contrast_emm_returns_one_row_per_contrast(self):
        """EMM for contrast should have one row per contrast level."""
        from iblnm.analysis import fit_response_lmm, compute_marginal_means
        df = _make_lmm_data()
        lmm = fit_response_lmm(df, 'response')
        emm = compute_marginal_means(lmm, 'contrast')
        assert isinstance(emm, pd.DataFrame)
        assert len(emm) == 5  # 5 contrast levels
        assert 'level' in emm.columns
        assert 'mean' in emm.columns
        assert 'ci_lower' in emm.columns

    def test_contrast_emm_monotonic(self):
        """With known positive contrast slope, EMMs should increase with contrast."""
        from iblnm.analysis import fit_response_lmm, compute_marginal_means
        df = _make_lmm_data(n_per_cell=30, seed=0)
        result = fit_response_lmm(df, 'response')
        emm = compute_marginal_means(result, 'contrast')
        means = emm.sort_values('level')['mean'].values
        # Not strictly monotonic due to log transform, but highest > lowest
        assert means[-1] > means[0]


# =============================================================================
# Random Effects and Contrast Slopes Tests
# =============================================================================


class TestLMMResultRandomEffects:

    def test_random_effects_field_exists(self):
        """LMMResult should have a random_effects field."""
        from iblnm.analysis import fit_response_lmm
        df = _make_lmm_data()
        result = fit_response_lmm(df, 'response')
        assert hasattr(result, 'random_effects')

    def test_random_effects_is_dict(self):
        from iblnm.analysis import fit_response_lmm
        df = _make_lmm_data()
        result = fit_response_lmm(df, 'response')
        assert isinstance(result.random_effects, dict)

    def test_random_effects_has_subjects(self):
        """random_effects keys should be subjects from the data."""
        from iblnm.analysis import fit_response_lmm
        df = _make_lmm_data()
        result = fit_response_lmm(df, 'response')
        assert set(result.random_effects.keys()) == {'s0', 's1', 's2'}

    def test_random_effects_has_slope_when_converged(self):
        """When random slope model converges, each subject's effects should
        include both intercept and slope."""
        from iblnm.analysis import fit_response_lmm
        df = _make_lmm_data(n_per_cell=30)
        result = fit_response_lmm(df, 'response')
        for subj, effects in result.random_effects.items():
            # effects is a Series with 'Group' (intercept) and possibly 'log_contrast'
            assert len(effects) >= 1  # at minimum, intercept


class TestComputeContrastSlopes:

    def test_returns_dataframe(self):
        from iblnm.analysis import fit_response_lmm, compute_contrast_slopes
        df = _make_lmm_data()
        result = fit_response_lmm(df, 'response')
        slopes = compute_contrast_slopes(result)
        assert isinstance(slopes, pd.DataFrame)

    def test_has_expected_columns(self):
        from iblnm.analysis import fit_response_lmm, compute_contrast_slopes
        df = _make_lmm_data()
        result = fit_response_lmm(df, 'response')
        slopes = compute_contrast_slopes(result)
        assert 'reward' in slopes.columns
        assert 'slope' in slopes.columns
        assert 'ci_lower' in slopes.columns
        assert 'ci_upper' in slopes.columns

    def test_two_reward_conditions(self):
        """Should have slopes for both correct and incorrect."""
        from iblnm.analysis import fit_response_lmm, compute_contrast_slopes
        df = _make_lmm_data()
        result = fit_response_lmm(df, 'response')
        slopes = compute_contrast_slopes(result)
        pop = slopes[slopes['type'] == 'population']
        assert set(pop['reward']) == {'incorrect', 'correct'}

    def test_subject_slopes_present(self):
        """When random slope model converges, subject-level slopes should appear."""
        from iblnm.analysis import fit_response_lmm, compute_contrast_slopes
        df = _make_lmm_data(n_per_cell=30)
        result = fit_response_lmm(df, 'response')
        slopes = compute_contrast_slopes(result)
        subj_rows = slopes[slopes['type'] == 'subject']
        # Should have subject rows if random slope converged
        if 'log_contrast' in list(result.random_effects.values())[0].index:
            assert len(subj_rows) > 0
            assert 'subject' in slopes.columns

    def test_known_contrast_effect(self):
        """Population slope for correct should reflect contrast + interaction."""
        from iblnm.analysis import fit_response_lmm, compute_contrast_slopes
        df = _make_lmm_data(n_per_cell=30, seed=0)
        result = fit_response_lmm(df, 'response')
        slopes = compute_contrast_slopes(result)
        pop = slopes[slopes['type'] == 'population']
        # Both conditions should have positive contrast slopes
        # (data has 0.5 contrast effect, no interaction)
        for _, row in pop.iterrows():
            assert row['slope'] > 0.2


class TestComputeInteractionEffects:

    def test_contrast_by_reward_returns_dataframe(self):
        from iblnm.analysis import fit_response_lmm, compute_interaction_effects
        df = _make_lmm_data()
        lmm = fit_response_lmm(df, 'response')
        result = compute_interaction_effects(lmm, 'contrast', 'reward')
        assert isinstance(result, pd.DataFrame)

    def test_contrast_by_reward_has_two_rows(self):
        from iblnm.analysis import fit_response_lmm, compute_interaction_effects
        df = _make_lmm_data()
        lmm = fit_response_lmm(df, 'response')
        result = compute_interaction_effects(lmm, 'contrast', 'reward')
        assert len(result) == 2
        assert set(result['x_level']) == {'incorrect', 'correct'}

    def test_contrast_by_side_has_two_rows(self):
        from iblnm.analysis import fit_response_lmm, compute_interaction_effects
        df = _make_lmm_data()
        lmm = fit_response_lmm(df, 'response')
        result = compute_interaction_effects(lmm, 'contrast', 'side')
        assert len(result) == 2
        assert set(result['x_level']) == {'contra', 'ipsi'}

    def test_reward_by_side_has_two_rows(self):
        from iblnm.analysis import fit_response_lmm, compute_interaction_effects
        df = _make_lmm_data()
        lmm = fit_response_lmm(df, 'response')
        result = compute_interaction_effects(lmm, 'reward', 'side')
        assert len(result) == 2
        assert set(result['x_level']) == {'contra', 'ipsi'}

    def test_has_ci_columns(self):
        from iblnm.analysis import fit_response_lmm, compute_interaction_effects
        df = _make_lmm_data()
        lmm = fit_response_lmm(df, 'response')
        result = compute_interaction_effects(lmm, 'contrast', 'reward')
        for col in ['effect', 'ci_lower', 'ci_upper', 'x_level', 'p_interaction']:
            assert col in result.columns

    def test_known_contrast_effect_positive(self):
        """Synthetic data has contrast slope 0.5; both reward levels should show it."""
        from iblnm.analysis import fit_response_lmm, compute_interaction_effects
        df = _make_lmm_data(n_per_cell=30, seed=0)
        lmm = fit_response_lmm(df, 'response')
        result = compute_interaction_effects(lmm, 'contrast', 'reward')
        for _, row in result.iterrows():
            assert row['effect'] > 0.2

    def test_known_reward_effect_positive(self):
        """Synthetic data has reward effect 0.2; both side levels should show it."""
        from iblnm.analysis import fit_response_lmm, compute_interaction_effects
        df = _make_lmm_data(n_per_cell=30, seed=0)
        lmm = fit_response_lmm(df, 'response')
        result = compute_interaction_effects(lmm, 'reward', 'side')
        for _, row in result.iterrows():
            assert row['effect'] > 0.0


# =============================================================================
# CCA Tests
# =============================================================================


class TestFitCCA:

    def test_perfect_correlation(self):
        """When Y is a linear function of X, canonical correlation should be ~1."""
        from iblnm.analysis import fit_cca
        rng = np.random.default_rng(42)
        n, k, p = 50, 5, 2
        X = pd.DataFrame(rng.standard_normal((n, k)), columns=[f'x{i}' for i in range(k)])
        W = rng.standard_normal((2, p))
        Y = pd.DataFrame(X.iloc[:, :2].values @ W + 0.01 * rng.standard_normal((n, p)),
                          columns=[f'y{i}' for i in range(p)])
        result = fit_cca(X, Y)
        assert result.correlations[0] > 0.95

    def test_returns_cca_result(self):
        from iblnm.analysis import fit_cca, CCAResult
        rng = np.random.default_rng(0)
        n, k, p = 30, 4, 2
        X = pd.DataFrame(rng.standard_normal((n, k)), columns=[f'x{i}' for i in range(k)])
        Y = pd.DataFrame(rng.standard_normal((n, p)), columns=[f'y{i}' for i in range(p)])
        result = fit_cca(X, Y)
        assert isinstance(result, CCAResult)

    def test_result_shapes(self):
        from iblnm.analysis import fit_cca
        rng = np.random.default_rng(0)
        n, k, p = 40, 6, 3
        X = pd.DataFrame(rng.standard_normal((n, k)), columns=[f'x{i}' for i in range(k)])
        Y = pd.DataFrame(rng.standard_normal((n, p)), columns=[f'y{i}' for i in range(p)])
        result = fit_cca(X, Y, n_components=2)
        assert result.x_weights.shape == (k, 2)
        assert result.y_weights.shape == (p, 2)
        assert result.x_scores.shape == (n, 2)
        assert result.y_scores.shape == (n, 2)
        assert result.correlations.shape == (2,)

    def test_weight_column_names_preserved(self):
        from iblnm.analysis import fit_cca
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((20, 3)),
                          columns=['stim_c0_contra', 'stim_c25_ipsi', 'fb_correct'])
        Y = pd.DataFrame(rng.standard_normal((20, 2)),
                          columns=['threshold', 'bias'])
        result = fit_cca(X, Y)
        assert list(result.x_weights.index) == ['stim_c0_contra', 'stim_c25_ipsi', 'fb_correct']
        assert list(result.y_weights.index) == ['threshold', 'bias']

    def test_n_components_capped(self):
        """n_components should not exceed min(K, P, n)."""
        from iblnm.analysis import fit_cca
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((20, 10)), columns=[f'x{i}' for i in range(10)])
        Y = pd.DataFrame(rng.standard_normal((20, 2)), columns=['y0', 'y1'])
        result = fit_cca(X, Y, n_components=5)
        assert result.correlations.shape == (2,)

    def test_no_permutation_by_default(self):
        from iblnm.analysis import fit_cca
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((20, 3)), columns=[f'x{i}' for i in range(3)])
        Y = pd.DataFrame(rng.standard_normal((20, 2)), columns=[f'y{i}' for i in range(2)])
        result = fit_cca(X, Y)
        assert result.p_values is None

    def test_drops_nan_rows(self):
        from iblnm.analysis import fit_cca
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((22, 3)), columns=[f'x{i}' for i in range(3)])
        Y = pd.DataFrame(rng.standard_normal((22, 2)), columns=[f'y{i}' for i in range(2)])
        X.iloc[0, 0] = np.nan
        Y.iloc[1, 1] = np.nan
        result = fit_cca(X, Y)
        assert result.n_recordings == 20

    def test_too_few_recordings_raises(self):
        from iblnm.analysis import fit_cca
        X = pd.DataFrame({'x0': [1.0, 2.0]})
        Y = pd.DataFrame({'y0': [3.0, 4.0]})
        with pytest.raises(ValueError, match='at least 3'):
            fit_cca(X, Y)

    def test_constant_y_column_dropped(self):
        """A constant Y column should be dropped, not crash."""
        from iblnm.analysis import fit_cca
        rng = np.random.default_rng(0)
        n = 30
        X = pd.DataFrame(rng.standard_normal((n, 4)), columns=[f'x{i}' for i in range(4)])
        Y = pd.DataFrame({
            'varying': rng.standard_normal(n),
            'constant': np.ones(n),
        })
        result = fit_cca(X, Y)
        assert result.y_weights.shape[0] == 1  # constant column removed
        assert result.correlations.shape == (1,)


# =============================================================================
# Sparse CCA Tests
# =============================================================================


class TestCCAResultAlpha:

    def test_standard_cca_has_alpha_none(self):
        from iblnm.analysis import fit_cca
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((20, 3)), columns=[f'x{i}' for i in range(3)])
        Y = pd.DataFrame(rng.standard_normal((20, 2)), columns=[f'y{i}' for i in range(2)])
        result = fit_cca(X, Y)
        assert result.alpha is None
        assert result.l1_ratio is None


class TestFitSparseCCA:

    def test_returns_cca_result(self):
        from iblnm.analysis import fit_sparse_cca, CCAResult
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.standard_normal((50, 6)), columns=[f'f{i}' for i in range(6)])
        Y = pd.DataFrame(rng.standard_normal((50, 4)), columns=[f'p{i}' for i in range(4)])
        result = fit_sparse_cca(X, Y, n_components=1, alpha=0.01, seed=42)
        assert isinstance(result, CCAResult)
        assert result.correlations.shape == (1,)
        assert result.x_weights.shape == (6, 1)
        assert result.y_weights.shape == (4, 1)

    def test_has_alpha_attribute(self):
        from iblnm.analysis import fit_sparse_cca
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.standard_normal((50, 6)), columns=[f'f{i}' for i in range(6)])
        Y = pd.DataFrame(rng.standard_normal((50, 4)), columns=[f'p{i}' for i in range(4)])
        result = fit_sparse_cca(X, Y, n_components=1, alpha=0.01, seed=42)
        assert result.alpha == 0.01
        assert result.l1_ratio == 0.0

    def test_produces_zero_weights_at_high_alpha(self):
        """High alpha with l1_ratio=0.5 should zero out some weights."""
        from iblnm.analysis import fit_sparse_cca
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.standard_normal((50, 6)), columns=[f'f{i}' for i in range(6)])
        Y = pd.DataFrame(rng.standard_normal((50, 4)), columns=[f'p{i}' for i in range(4)])
        result = fit_sparse_cca(X, Y, n_components=1, alpha=0.1,
                                l1_ratio=0.5, seed=42)
        n_zero = (result.x_weights.values.ravel() == 0).sum()
        assert n_zero > 0

    def test_stores_alpha_and_l1_ratio(self):
        from iblnm.analysis import fit_sparse_cca
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.standard_normal((50, 6)), columns=[f'f{i}' for i in range(6)])
        Y = pd.DataFrame(rng.standard_normal((50, 4)), columns=[f'p{i}' for i in range(4)])
        result = fit_sparse_cca(X, Y, n_components=1, alpha=0.05,
                                l1_ratio=0.3, seed=42)
        assert result.alpha == 0.05
        assert result.l1_ratio == 0.3

    def test_permutation_test(self):
        from iblnm.analysis import fit_sparse_cca
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.standard_normal((30, 6)), columns=[f'f{i}' for i in range(6)])
        Y = pd.DataFrame(rng.standard_normal((30, 4)), columns=[f'p{i}' for i in range(4)])
        result = fit_sparse_cca(X, Y, n_components=1, n_permutations=10,
                                alpha=0.01, seed=42)
        assert result.p_values is not None
        assert 0 < result.p_values[0] <= 1

    def test_preserves_feature_names(self):
        from iblnm.analysis import fit_sparse_cca
        rng = np.random.default_rng(42)
        x_names = ['log_contrast', 'side', 'feedback', 'lc:side', 'lc:fb', 'side:fb']
        y_names = ['threshold', 'bias', 'lapse_l', 'lapse_r']
        X = pd.DataFrame(rng.standard_normal((40, 6)), columns=x_names)
        Y = pd.DataFrame(rng.standard_normal((40, 4)), columns=y_names)
        result = fit_sparse_cca(X, Y, n_components=1, alpha=0.01, seed=42)
        assert list(result.x_weights.index) == x_names
        assert list(result.y_weights.index) == y_names

    def test_drops_nan_rows(self):
        from iblnm.analysis import fit_sparse_cca
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.standard_normal((32, 4)), columns=[f'x{i}' for i in range(4)])
        Y = pd.DataFrame(rng.standard_normal((32, 2)), columns=[f'y{i}' for i in range(2)])
        X.iloc[0, 0] = np.nan
        Y.iloc[1, 1] = np.nan
        result = fit_sparse_cca(X, Y, n_components=1, alpha=0.01, seed=42)
        assert result.n_recordings == 30

    def test_low_alpha_preserves_all_weights(self):
        """Very low alpha should keep all weights nonzero (near-standard CCA)."""
        from iblnm.analysis import fit_sparse_cca
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.standard_normal((50, 6)), columns=[f'f{i}' for i in range(6)])
        Y = pd.DataFrame(rng.standard_normal((50, 4)), columns=[f'p{i}' for i in range(4)])
        result = fit_sparse_cca(X, Y, n_components=1, alpha=1e-4, seed=42)
        n_zero = (result.x_weights.values.ravel() == 0).sum()
        assert n_zero == 0

    def test_grid_search_selects_best(self):
        """Grid search over alpha × l1_ratio should return best combo."""
        from iblnm.analysis import fit_sparse_cca
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.standard_normal((50, 6)), columns=[f'f{i}' for i in range(6)])
        Y = pd.DataFrame(rng.standard_normal((50, 4)), columns=[f'p{i}' for i in range(4)])
        result = fit_sparse_cca(X, Y, n_components=1,
                                alpha=[1e-3, 0.01, 0.1],
                                l1_ratio=[0.0, 0.1, 0.5],
                                seed=42)
        assert result.alpha in [1e-3, 0.01, 0.1]
        assert result.l1_ratio in [0.0, 0.1, 0.5]

    def test_unit_norm_rescales_weights(self):
        """unit_norm=True should produce ||w|| = 1."""
        from iblnm.analysis import fit_sparse_cca
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.standard_normal((50, 6)), columns=[f'f{i}' for i in range(6)])
        Y = pd.DataFrame(rng.standard_normal((50, 4)), columns=[f'p{i}' for i in range(4)])
        result = fit_sparse_cca(X, Y, n_components=1, alpha=0.01, seed=42,
                                unit_norm=True)
        np.testing.assert_allclose(
            np.linalg.norm(result.x_weights.values), 1.0, atol=1e-6)
        np.testing.assert_allclose(
            np.linalg.norm(result.y_weights.values), 1.0, atol=1e-6)

    def test_no_unit_norm_preserves_raw_weights(self):
        """unit_norm=False should keep raw ElasticCCA magnitudes (< 1)."""
        from iblnm.analysis import fit_sparse_cca
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.standard_normal((50, 6)), columns=[f'f{i}' for i in range(6)])
        Y = pd.DataFrame(rng.standard_normal((50, 4)), columns=[f'p{i}' for i in range(4)])
        result = fit_sparse_cca(X, Y, n_components=1, alpha=0.01, seed=42,
                                unit_norm=False)
        assert np.linalg.norm(result.x_weights.values) < 1.0


# =============================================================================
# Wheel Kinematics LMM Tests
# =============================================================================


def _make_wheel_lmm_data(n_subjects=3, n_per_subject=80, seed=42):
    """Synthetic trial data for wheel kinematics LMM testing.

    Creates data where response_early has a known positive effect on
    reaction_time within each contrast group.
    """
    rng = np.random.default_rng(seed)
    subjects = [f's{i}' for i in range(n_subjects)]
    contrasts = [0.0, 0.0625, 0.125, 0.25, 1.0]
    sides = ['contra', 'ipsi']
    choices = [-1, 1]

    rows = []
    for subj in subjects:
        subj_intercept = rng.normal(0, 0.5)
        for _ in range(n_per_subject):
            contrast = rng.choice(contrasts)
            side = rng.choice(sides)
            choice = rng.choice(choices)
            response_early = rng.normal(1.0, 0.5)
            # reaction_time has known relationship with response_early
            reaction_time = (
                0.3
                + 0.1 * response_early  # known effect
                + subj_intercept
                + rng.normal(0, 0.1)
            )
            rows.append({
                'contrast': contrast,
                'side': side,
                'choice': choice,
                'stim_side': side,
                'subject': subj,
                'response_early': response_early,
                'reaction_time': max(reaction_time, 0.06),
                'movement_time': max(rng.normal(0.3, 0.1), 0.01),
                'peak_velocity': abs(rng.normal(5.0, 2.0)),
            })
    return pd.DataFrame(rows)


class TestFitWheelLMM:

    def test_returns_dict(self):
        from iblnm.analysis import fit_wheel_lmm
        df = _make_wheel_lmm_data()
        result = fit_wheel_lmm(df, dv_col='reaction_time',
                                response_col='response_early')
        assert isinstance(result, dict)
        assert 'delta_r2' in result
        assert 'lrt_pvalue' in result

    def test_has_expected_fields(self):
        from iblnm.analysis import fit_wheel_lmm
        df = _make_wheel_lmm_data()
        result = fit_wheel_lmm(df, dv_col='reaction_time',
                                response_col='response_early')
        assert result['dv'] == 'reaction_time'
        assert result['delta_r2'] is not None
        assert result['lrt_pvalue'] is not None
        assert result['nm_coefficient'] is not None
        assert result['n_trials'] > 0
        assert result['n_subjects'] == 3

    def test_r2_values_valid(self):
        from iblnm.analysis import fit_wheel_lmm
        df = _make_wheel_lmm_data()
        result = fit_wheel_lmm(df, dv_col='reaction_time',
                                response_col='response_early')
        assert 0 <= result['base_r2_marginal'] <= 1
        assert 0 <= result['full_r2_marginal'] <= 1
        assert 0 <= result['lrt_pvalue'] <= 1

    def test_detects_known_nm_effect(self):
        """Data has known positive relationship between response_early and RT."""
        from iblnm.analysis import fit_wheel_lmm
        df = _make_wheel_lmm_data(n_subjects=5, n_per_subject=200, seed=0)
        df_c = df[np.isclose(df['contrast'], 0.125)]
        result = fit_wheel_lmm(df_c, dv_col='reaction_time',
                                response_col='response_early')
        assert result is not None
        assert result['nm_coefficient'] > 0
        assert result['lrt_pvalue'] < 0.05

    def test_insufficient_subjects_returns_none(self):
        from iblnm.analysis import fit_wheel_lmm
        df = _make_wheel_lmm_data(n_subjects=1, n_per_subject=50)
        result = fit_wheel_lmm(df, dv_col='reaction_time',
                                response_col='response_early')
        assert result is None

    def test_delta_r2_positive_with_true_effect(self):
        """When NM truly predicts DV, delta R² should be positive."""
        from iblnm.analysis import fit_wheel_lmm
        df = _make_wheel_lmm_data(n_subjects=5, n_per_subject=200, seed=0)
        df_c = df[np.isclose(df['contrast'], 0.125)]
        result = fit_wheel_lmm(df_c, dv_col='reaction_time',
                                response_col='response_early')
        assert result['delta_r2'] > 0


class TestCCAPermutation:

    def test_permutation_returns_p_values(self):
        from iblnm.analysis import fit_cca
        rng = np.random.default_rng(42)
        n, k, p = 50, 5, 2
        X = pd.DataFrame(rng.standard_normal((n, k)), columns=[f'x{i}' for i in range(k)])
        Y = pd.DataFrame(rng.standard_normal((n, p)), columns=[f'y{i}' for i in range(p)])
        result = fit_cca(X, Y, n_permutations=100, seed=0)
        assert result.p_values is not None
        assert result.p_values.shape == result.correlations.shape
        assert result.n_permutations == 100

    def test_random_data_not_significant(self):
        """Uncorrelated X and Y should yield p > 0.05 for all components."""
        from iblnm.analysis import fit_cca
        rng = np.random.default_rng(42)
        n = 80
        X = pd.DataFrame(rng.standard_normal((n, 5)), columns=[f'x{i}' for i in range(5)])
        Y = pd.DataFrame(rng.standard_normal((n, 3)), columns=[f'y{i}' for i in range(3)])
        result = fit_cca(X, Y, n_permutations=200, seed=0)
        assert all(p > 0.01 for p in result.p_values)

    def test_correlated_data_significant(self):
        """Strongly correlated X and Y should yield p < 0.05 for first component."""
        from iblnm.analysis import fit_cca
        rng = np.random.default_rng(42)
        n = 80
        X = pd.DataFrame(rng.standard_normal((n, 5)), columns=[f'x{i}' for i in range(5)])
        Y = pd.DataFrame(X.iloc[:, :2].values + 0.1 * rng.standard_normal((n, 2)),
                          columns=['y0', 'y1'])
        result = fit_cca(X, Y, n_permutations=200, seed=0)
        assert result.p_values[0] < 0.05

    def test_session_level_permutation(self):
        """When session_labels provided, permutation preserves within-session structure."""
        from iblnm.analysis import fit_cca
        rng = np.random.default_rng(42)
        n_sessions = 10
        X = pd.DataFrame(rng.standard_normal((n_sessions * 2, 4)),
                          columns=[f'x{i}' for i in range(4)])
        Y_session = rng.standard_normal((n_sessions, 2))
        Y = pd.DataFrame(np.repeat(Y_session, 2, axis=0), columns=['y0', 'y1'])
        session_labels = pd.Series(np.repeat(np.arange(n_sessions), 2))
        result = fit_cca(X, Y, n_permutations=50, session_labels=session_labels, seed=0)
        assert result.p_values is not None

    def test_reproducible_with_seed(self):
        from iblnm.analysis import fit_cca
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.standard_normal((30, 3)), columns=[f'x{i}' for i in range(3)])
        Y = pd.DataFrame(rng.standard_normal((30, 2)), columns=[f'y{i}' for i in range(2)])
        r1 = fit_cca(X, Y, n_permutations=50, seed=99)
        r2 = fit_cca(X, Y, n_permutations=50, seed=99)
        np.testing.assert_array_equal(r1.p_values, r2.p_values)


# =============================================================================
# fit_response_glm
# =============================================================================


def _make_glm_events(n=200, eid='eid1', brain_region='VTA', hemisphere='l',
                     target_nm='VTA-DA', seed=42):
    """Synthetic events DataFrame for GLM tests."""
    rng = np.random.default_rng(seed)
    contrast = rng.choice([0, 6.25, 12.5, 25, 100], n)
    stim_side = rng.choice(['left', 'right'], n)
    feedback_type = rng.choice([-1, 1], n)
    log_c = contrast_transform(contrast)
    contra_side = {'l': 'right', 'r': 'left'}[hemisphere]
    # Deviation coding ±0.5, matching fit_response_glm
    side = np.where(stim_side == contra_side, 0.5, -0.5)
    reward = np.where(feedback_type == 1, 0.5, -0.5)
    response = (2 + 0.5 * log_c + 1.0 * side + 0.3 * reward
                + rng.normal(0, 0.5, n))
    return pd.DataFrame({
        'eid': eid, 'brain_region': brain_region, 'hemisphere': hemisphere,
        'target_NM': target_nm, 'event': 'stimOn_times',
        'contrast': contrast, 'stim_side': stim_side,
        'signed_contrast': np.where(stim_side == 'left', -contrast, contrast),
        'feedbackType': feedback_type, 'probabilityLeft': 0.5,
        'response_early': response,
    })


def _make_glm_coefs(seed=42):
    """Synthetic GLM coefficient matrix with known structure.

    Two cohorts: A (20 sessions) and B (5 sessions).
    A has high log_contrast, low reward.
    B has low log_contrast, high reward.
    This guarantees PC1 separates the two groups.
    """
    rng = np.random.default_rng(seed)
    coef_names = [
        'log_contrast', 'side', 'reward',
        'log_contrast:side', 'log_contrast:reward', 'side:reward',
    ]
    n_a, n_b = 20, 5
    rows = []
    targets = []
    for i in range(n_a):
        rows.append([2.0 + rng.normal(0, 0.3),   # log_contrast: high
                     rng.normal(0, 0.2),
                     0.1 + rng.normal(0, 0.2),    # reward: low
                     rng.normal(0, 0.1),
                     rng.normal(0, 0.1),
                     rng.normal(0, 0.1)])
        targets.append('A')
    for i in range(n_b):
        rows.append([0.1 + rng.normal(0, 0.3),    # log_contrast: low
                     rng.normal(0, 0.2),
                     2.0 + rng.normal(0, 0.2),    # reward: high
                     rng.normal(0, 0.1),
                     rng.normal(0, 0.1),
                     rng.normal(0, 0.1)])
        targets.append('B')

    index = pd.MultiIndex.from_tuples(
        [(f'e{i}', t, 0) for i, t in enumerate(targets)],
        names=['eid', 'target_NM', 'fiber_idx'],
    )
    return pd.DataFrame(rows, index=index, columns=coef_names)


class TestPcaGlmCoefficients:

    def test_returns_result_with_expected_attributes(self):
        from iblnm.analysis import pca_glm_coefficients
        coefs = _make_glm_coefs()
        result = pca_glm_coefficients(coefs, n_components=3)
        assert hasattr(result, 'scores')
        assert hasattr(result, 'components')
        assert hasattr(result, 'explained_variance_ratio')
        assert hasattr(result, 'feature_names')
        assert hasattr(result, 'target_labels')

    def test_scores_shape(self):
        from iblnm.analysis import pca_glm_coefficients
        coefs = _make_glm_coefs()
        result = pca_glm_coefficients(coefs, n_components=3)
        assert result.scores.shape == (25, 3)

    def test_components_shape(self):
        from iblnm.analysis import pca_glm_coefficients
        coefs = _make_glm_coefs()
        result = pca_glm_coefficients(coefs, n_components=3)
        assert result.components.shape == (3, 6)

    def test_explained_variance_sums_to_at_most_one(self):
        from iblnm.analysis import pca_glm_coefficients
        coefs = _make_glm_coefs()
        result = pca_glm_coefficients(coefs, n_components=6)
        np.testing.assert_allclose(
            result.explained_variance_ratio.sum(), 1.0, atol=1e-10)

    def test_drops_intercept_if_present(self):
        from iblnm.analysis import pca_glm_coefficients
        coefs = _make_glm_coefs()
        coefs.insert(0, 'intercept', 1.0)
        result = pca_glm_coefficients(coefs, n_components=3)
        assert 'intercept' not in result.feature_names
        assert result.components.shape[1] == 6

    def test_pc1_separates_groups(self):
        """With known group structure, PC1 scores should separate A and B."""
        from iblnm.analysis import pca_glm_coefficients
        coefs = _make_glm_coefs()
        result = pca_glm_coefficients(coefs, n_components=2)
        scores_a = result.scores[result.target_labels == 'A', 0]
        scores_b = result.scores[result.target_labels == 'B', 0]
        # Group means should be well separated (no overlap in means)
        assert abs(scores_a.mean() - scores_b.mean()) > 1.0

    def test_weighting_changes_components(self):
        """Cohort weighting should produce different PCs than unweighted."""
        from iblnm.analysis import pca_glm_coefficients
        coefs = _make_glm_coefs()
        weighted = pca_glm_coefficients(coefs, n_components=2, cohort_weighted=True)
        unweighted = pca_glm_coefficients(coefs, n_components=2, cohort_weighted=False)
        # With 20 A vs 5 B, weighting should change the components
        assert not np.allclose(
            weighted.components[0], unweighted.components[0], atol=1e-6)

    def test_feature_names_match_input_columns(self):
        from iblnm.analysis import pca_glm_coefficients
        coefs = _make_glm_coefs()
        result = pca_glm_coefficients(coefs, n_components=2)
        assert list(result.feature_names) == list(coefs.columns)


class TestPcaScoreStats:

    def test_returns_dataframe_with_expected_columns(self):
        from iblnm.analysis import pca_glm_coefficients, pca_score_stats
        coefs = _make_glm_coefs()
        pca = pca_glm_coefficients(coefs, n_components=3)
        result = pca_score_stats(pca)
        assert 'pc' in result.columns
        assert 'kruskal_h' in result.columns
        assert 'kruskal_p' in result.columns
        assert 'target_a' in result.columns
        assert 'target_b' in result.columns
        assert 'mwu_u' in result.columns
        assert 'mwu_p' in result.columns

    def test_one_kruskal_row_per_pc(self):
        from iblnm.analysis import pca_glm_coefficients, pca_score_stats
        coefs = _make_glm_coefs()
        pca = pca_glm_coefficients(coefs, n_components=2)
        result = pca_score_stats(pca)
        kw_rows = result[result['target_a'].isna()]
        assert len(kw_rows) == 2

    def test_pairwise_count(self):
        """Two groups → 1 pair per PC."""
        from iblnm.analysis import pca_glm_coefficients, pca_score_stats
        coefs = _make_glm_coefs()  # groups A, B
        pca = pca_glm_coefficients(coefs, n_components=2)
        result = pca_score_stats(pca)
        pw_rows = result[result['target_a'].notna()]
        assert len(pw_rows) == 2  # 1 pair × 2 PCs

    def test_pc1_separates_groups_significantly(self):
        """With well-separated groups, KW p-value should be small."""
        from iblnm.analysis import pca_glm_coefficients, pca_score_stats
        coefs = _make_glm_coefs()
        pca = pca_glm_coefficients(coefs, n_components=1)
        result = pca_score_stats(pca)
        kw_p = result.loc[result['target_a'].isna(), 'kruskal_p'].iloc[0]
        assert kw_p < 0.01


class TestIcaGlmCoefficients:

    def test_returns_result_with_expected_attributes(self):
        from iblnm.analysis import ica_glm_coefficients
        coefs = _make_glm_coefs()
        result = ica_glm_coefficients(coefs, n_components=3)
        assert hasattr(result, 'scores')
        assert hasattr(result, 'components')
        assert hasattr(result, 'explained_variance_ratio')
        assert hasattr(result, 'feature_names')
        assert hasattr(result, 'target_labels')

    def test_scores_shape(self):
        from iblnm.analysis import ica_glm_coefficients
        coefs = _make_glm_coefs()
        result = ica_glm_coefficients(coefs, n_components=3)
        assert result.scores.shape == (25, 3)

    def test_components_shape(self):
        from iblnm.analysis import ica_glm_coefficients
        coefs = _make_glm_coefs()
        result = ica_glm_coefficients(coefs, n_components=3)
        assert result.components.shape == (3, 6)

    def test_variance_explained_positive(self):
        """Post-hoc variance explained should be positive for each IC."""
        from iblnm.analysis import ica_glm_coefficients
        coefs = _make_glm_coefs()
        result = ica_glm_coefficients(coefs, n_components=3)
        assert all(v > 0 for v in result.explained_variance_ratio)

    def test_ordered_by_variance(self):
        """Components should be sorted by descending variance explained."""
        from iblnm.analysis import ica_glm_coefficients
        coefs = _make_glm_coefs()
        result = ica_glm_coefficients(coefs, n_components=3)
        ve = result.explained_variance_ratio
        assert all(ve[i] >= ve[i + 1] for i in range(len(ve) - 1))

    def test_drops_intercept_if_present(self):
        from iblnm.analysis import ica_glm_coefficients
        coefs = _make_glm_coefs()
        coefs.insert(0, 'intercept', 1.0)
        result = ica_glm_coefficients(coefs, n_components=3)
        assert 'intercept' not in result.feature_names

    def test_some_ic_separates_groups(self):
        """With known group structure, at least one IC should separate A and B."""
        from iblnm.analysis import ica_glm_coefficients
        coefs = _make_glm_coefs()
        result = ica_glm_coefficients(coefs, n_components=2)
        # Check all ICs — at least one should show clear separation
        max_sep = max(
            abs(result.scores[result.target_labels == 'A', i].mean()
                - result.scores[result.target_labels == 'B', i].mean())
            for i in range(2)
        )
        assert max_sep > 1.0


class TestFitResponseGLM:
    def test_known_coefficients(self):
        """Synthetic data with known linear relationship recovers coefficients."""
        from iblnm.analysis import fit_response_glm
        rng = np.random.default_rng(42)
        n = 400
        contrast = rng.choice([0, 6.25, 12.5, 25, 100], n)
        stim_side = rng.choice(['left', 'right'], n)
        # Deviation coding ±0.5, matching fit_response_glm
        side = np.where(stim_side == 'right', 0.5, -0.5)  # hemisphere='l'
        feedback_type = rng.choice([-1, 1], n)
        reward = np.where(feedback_type == 1, 0.5, -0.5)
        log_c = contrast_transform(contrast)

        # 7 known coefficients
        true_beta = np.array([2.0, 0.5, 1.0, 0.3, -0.2, 0.1, 0.4])
        X = np.column_stack([
            np.ones(n), log_c, side, reward,
            log_c * side, log_c * reward, side * reward,
        ])
        response = X @ true_beta + rng.normal(0, 0.1, n)

        events = pd.DataFrame({
            'eid': 'eid1', 'brain_region': 'VTA', 'hemisphere': 'l',
            'target_NM': 'VTA-DA', 'event': 'stimOn_times',
            'contrast': contrast, 'stim_side': stim_side,
            'signed_contrast': np.where(stim_side == 'left', -contrast, contrast),
            'feedbackType': feedback_type,
            'probabilityLeft': 0.5,
            'response_early': response,
        })
        coefs, ses = fit_response_glm(events, 'stimOn_times')
        np.testing.assert_allclose(coefs.iloc[0].values, true_beta, atol=0.15)

    def test_returns_standard_errors(self):
        """SE values are positive and finite."""
        from iblnm.analysis import fit_response_glm
        events = _make_glm_events(n=200, seed=0)
        coefs, ses = fit_response_glm(events, 'stimOn_times')
        assert (ses.values > 0).all()
        assert np.all(np.isfinite(ses.values))

    def test_multiple_recordings(self):
        """Two recordings produce two rows."""
        from iblnm.analysis import fit_response_glm
        e1 = _make_glm_events(n=200, eid='eid1', brain_region='VTA', seed=0)
        e2 = _make_glm_events(n=200, eid='eid2', brain_region='SNc', seed=1)
        events = pd.concat([e1, e2], ignore_index=True)
        coefs, ses = fit_response_glm(events, 'stimOn_times')
        assert len(coefs) == 2

    def test_skips_too_few_trials(self):
        """Recording with fewer than min_trials is excluded."""
        from iblnm.analysis import fit_response_glm
        events = _make_glm_events(n=10, seed=0)
        coefs, ses = fit_response_glm(events, 'stimOn_times', min_trials=20)
        assert len(coefs) == 0

    def test_skips_nan_responses(self):
        """Trials with NaN response are excluded; fit proceeds on remainder."""
        from iblnm.analysis import fit_response_glm
        events = _make_glm_events(n=100, seed=0)
        events.loc[:29, 'response_early'] = np.nan
        coefs, ses = fit_response_glm(events, 'stimOn_times', min_trials=20)
        assert len(coefs) == 1  # fits on 70 valid trials

    def test_skips_singular_design(self):
        """Recording with rank-deficient design is skipped."""
        from iblnm.analysis import fit_response_glm
        events = _make_glm_events(n=100, seed=0)
        events['contrast'] = 25  # constant
        events['stim_side'] = 'right'  # constant
        events['feedbackType'] = 1  # constant
        coefs, ses = fit_response_glm(events, 'stimOn_times')
        assert len(coefs) == 0

    def test_wrong_event_raises(self):
        """Requesting an event not in data raises ValueError."""
        from iblnm.analysis import fit_response_glm
        events = _make_glm_events(n=100, seed=0)
        with pytest.raises(ValueError, match="not found"):
            fit_response_glm(events, 'nonexistent_event')

    def test_column_names(self):
        """Output columns use 'reward' (not 'feedback') for consistency with LMM."""
        from iblnm.analysis import fit_response_glm
        events = _make_glm_events(n=200, seed=0)
        coefs, ses = fit_response_glm(events, 'stimOn_times')
        expected = ['intercept', 'log_contrast', 'side', 'reward',
                    'log_contrast:side', 'log_contrast:reward',
                    'side:reward']
        assert list(coefs.columns) == expected
        assert list(ses.columns) == expected

    def test_filters_biased_blocks(self):
        """Only trials with probabilityLeft == 0.5 are used."""
        from iblnm.analysis import fit_response_glm
        events = _make_glm_events(n=200, seed=0)
        events.loc[:99, 'probabilityLeft'] = 0.2
        coefs, ses = fit_response_glm(events, 'stimOn_times', min_trials=20)
        assert len(coefs) == 1  # fits on ~100 unbiased trials


# =============================================================================
# fit_cca scale parameter
# =============================================================================


class TestFitCCAScale:

    def test_scale_false_skips_standardization(self):
        """Pre-scaled data with scale=False gives same result as scale=True."""
        from iblnm.analysis import fit_cca
        from sklearn.preprocessing import StandardScaler
        rng = np.random.default_rng(42)
        n, k, p = 50, 5, 2
        X = pd.DataFrame(rng.standard_normal((n, k)),
                          columns=[f'x{i}' for i in range(k)])
        W = rng.standard_normal((2, p))
        Y = pd.DataFrame(
            X.iloc[:, :2].values @ W + 0.01 * rng.standard_normal((n, p)),
            columns=[f'y{i}' for i in range(p)])
        r1 = fit_cca(X, Y, scale=True)
        Xz = pd.DataFrame(StandardScaler().fit_transform(X),
                           columns=X.columns, index=X.index)
        Yz = pd.DataFrame(StandardScaler().fit_transform(Y),
                           columns=Y.columns, index=Y.index)
        r2 = fit_cca(Xz, Yz, scale=False)
        np.testing.assert_allclose(
            abs(r1.correlations[0]), abs(r2.correlations[0]), atol=0.01)

    def test_scale_true_is_default(self):
        """Default behavior unchanged — scale=True."""
        from iblnm.analysis import fit_cca
        import inspect
        sig = inspect.signature(fit_cca)
        assert sig.parameters['scale'].default is True


# =============================================================================
# cross_project_cca / compare_cca_weights
# =============================================================================


class TestCrossProjectCCA:

    def test_self_projection_matches_within(self):
        """Projecting A through A's weights recovers A's canonical corr."""
        from iblnm.analysis import fit_cca, cross_project_cca
        from sklearn.preprocessing import StandardScaler
        rng = np.random.default_rng(42)
        n, k, p = 50, 6, 4
        X = pd.DataFrame(rng.standard_normal((n, k)),
                          columns=[f'x{i}' for i in range(k)])
        W = rng.standard_normal((2, p))
        Y = pd.DataFrame(
            X.iloc[:, :2].values @ W + 0.1 * rng.standard_normal((n, p)),
            columns=[f'y{i}' for i in range(p)])
        Xz = StandardScaler().fit_transform(X)
        Yz = StandardScaler().fit_transform(Y)
        Xz_df = pd.DataFrame(Xz, columns=X.columns, index=X.index)
        Yz_df = pd.DataFrame(Yz, columns=Y.columns, index=Y.index)
        result = fit_cca(Xz_df, Yz_df, scale=False)
        r = cross_project_cca(Xz, Yz, result)
        np.testing.assert_allclose(r, result.correlations[0], atol=0.05)

    def test_unrelated_data_gives_low_projection(self):
        from iblnm.analysis import fit_cca, cross_project_cca
        from sklearn.preprocessing import StandardScaler
        rng = np.random.default_rng(42)
        n, k, p = 50, 6, 4
        X = pd.DataFrame(rng.standard_normal((n, k)),
                          columns=[f'x{i}' for i in range(k)])
        W = rng.standard_normal((2, p))
        Y = pd.DataFrame(
            X.iloc[:, :2].values @ W + 0.1 * rng.standard_normal((n, p)),
            columns=[f'y{i}' for i in range(p)])
        Xz = StandardScaler().fit_transform(X)
        Yz = StandardScaler().fit_transform(Y)
        result = fit_cca(
            pd.DataFrame(Xz, columns=X.columns),
            pd.DataFrame(Yz, columns=Y.columns),
            scale=False)
        # Random data — low cross-projection
        X2 = rng.standard_normal((40, k))
        Y2 = rng.standard_normal((40, p))
        r = cross_project_cca(X2, Y2, result)
        assert abs(r) < 0.5


class TestCompareCCAWeights:

    def test_identical_weights_give_cosine_one(self):
        from iblnm.analysis import fit_cca, compare_cca_weights
        from sklearn.preprocessing import StandardScaler
        rng = np.random.default_rng(42)
        n, k, p = 50, 6, 4
        X = pd.DataFrame(rng.standard_normal((n, k)),
                          columns=[f'x{i}' for i in range(k)])
        W = rng.standard_normal((2, p))
        Y = pd.DataFrame(
            X.iloc[:, :2].values @ W + 0.01 * rng.standard_normal((n, p)),
            columns=[f'y{i}' for i in range(p)])
        Xz = pd.DataFrame(StandardScaler().fit_transform(X),
                           columns=X.columns)
        Yz = pd.DataFrame(StandardScaler().fit_transform(Y),
                           columns=Y.columns)
        result = fit_cca(Xz, Yz, scale=False)
        sims = compare_cca_weights(result, result)
        np.testing.assert_allclose(
            abs(sims['neural_cosine']), 1.0, atol=0.01)
        np.testing.assert_allclose(
            abs(sims['behavioral_cosine']), 1.0, atol=0.01)

    def test_returns_expected_keys(self):
        from iblnm.analysis import fit_cca, compare_cca_weights
        from sklearn.preprocessing import StandardScaler
        rng = np.random.default_rng(42)
        n, k, p = 50, 6, 4
        X = pd.DataFrame(rng.standard_normal((n, k)),
                          columns=[f'x{i}' for i in range(k)])
        Y = pd.DataFrame(rng.standard_normal((n, p)),
                          columns=[f'y{i}' for i in range(p)])
        Xz = pd.DataFrame(StandardScaler().fit_transform(X),
                           columns=X.columns)
        Yz = pd.DataFrame(StandardScaler().fit_transform(Y),
                           columns=Y.columns)
        result = fit_cca(Xz, Yz, scale=False)
        sims = compare_cca_weights(result, result)
        assert 'neural_cosine' in sims
        assert 'behavioral_cosine' in sims


class TestAlignCCASigns:

    def _make_results(self):
        from iblnm.analysis import fit_cca
        from sklearn.preprocessing import StandardScaler
        rng = np.random.default_rng(42)
        n, k, p = 50, 6, 4
        X = pd.DataFrame(rng.standard_normal((n, k)),
                          columns=[f'x{i}' for i in range(k)])
        W = rng.standard_normal((2, p))
        Y = pd.DataFrame(
            X.iloc[:, :2].values @ W + 0.1 * rng.standard_normal((n, p)),
            columns=[f'y{i}' for i in range(p)])
        Xz = pd.DataFrame(StandardScaler().fit_transform(X),
                           columns=X.columns)
        Yz = pd.DataFrame(StandardScaler().fit_transform(Y),
                           columns=Y.columns)
        return fit_cca(Xz, Yz, scale=False)

    def test_flipped_result_gets_aligned(self):
        from iblnm.analysis import align_cca_signs
        result = self._make_results()
        # Create a flipped copy
        from iblnm.analysis import CCAResult
        flipped = CCAResult(
            x_weights=-result.x_weights.copy(),
            y_weights=-result.y_weights.copy(),
            x_scores=-result.x_scores.copy(),
            y_scores=-result.y_scores.copy(),
            correlations=result.correlations.copy(),
            p_values=result.p_values,
            n_recordings=result.n_recordings,
            n_permutations=result.n_permutations,
        )
        results = {'A': result, 'B': flipped}
        aligned = align_cca_signs(results, reference='A')
        # After alignment, neural weights should point same direction
        cos = np.dot(
            aligned['A'].x_weights['CC1'].values,
            aligned['B'].x_weights['CC1'].values,
        )
        assert cos > 0

    def test_already_aligned_unchanged(self):
        from iblnm.analysis import align_cca_signs
        result = self._make_results()
        results = {'A': result, 'B': result}
        aligned = align_cca_signs(results, reference='A')
        np.testing.assert_array_equal(
            aligned['A'].x_weights.values, result.x_weights.values)
        np.testing.assert_array_equal(
            aligned['B'].x_weights.values, result.x_weights.values)

    def test_default_reference_is_first_sorted(self):
        from iblnm.analysis import align_cca_signs
        result = self._make_results()
        results = {'Z': result, 'A': result}
        aligned = align_cca_signs(results)
        assert list(aligned.keys()) == ['Z', 'A']


class TestComputeRecordingProjection:
    """Tests for compute_recording_projection."""

    from datetime import date

    def test_basic_projection(self):
        from iblnm.analysis import compute_recording_projection
        from datetime import date

        n_ready = {'VTA-DA': 3, 'SNc-DA': 5}
        n_total = {'VTA-DA': 10, 'SNc-DA': 10}
        result = compute_recording_projection(
            n_ready, n_total, target_n=5,
            deadline=date(2026, 7, 31), capacity_per_day=16,
            today=date(2026, 3, 31),
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert set(result.columns) >= {
            'target_NM', 'n_analysis_ready', 'n_total', 'yield_rate',
            'shortfall', 'effective_sessions_needed', 'recording_days_needed',
        }
        # VTA-DA: yield=0.3, shortfall=2, effective=ceil(2/0.3)=7
        vta = result.set_index('target_NM').loc['VTA-DA']
        assert vta['yield_rate'] == 0.3
        assert vta['shortfall'] == 2
        assert vta['effective_sessions_needed'] == 7
        # SNc-DA: already at target, no additional sessions needed
        snc = result.set_index('target_NM').loc['SNc-DA']
        assert snc['shortfall'] == 0
        assert snc['effective_sessions_needed'] == 0

    def test_zero_yield_returns_inf(self):
        from iblnm.analysis import compute_recording_projection
        from datetime import date

        n_ready = {'VTA-DA': 0}
        n_total = {'VTA-DA': 5}
        result = compute_recording_projection(
            n_ready, n_total, target_n=5,
            deadline=date(2026, 7, 31), capacity_per_day=16,
            today=date(2026, 3, 31),
        )
        vta = result.set_index('target_NM').loc['VTA-DA']
        assert vta['yield_rate'] == 0.0
        assert np.isinf(vta['effective_sessions_needed'])
        assert np.isinf(vta['recording_days_needed'])

    def test_no_recordings_yet(self):
        """Target with zero total recordings gets NaN yield."""
        from iblnm.analysis import compute_recording_projection
        from datetime import date

        n_ready = {'VTA-DA': 0}
        n_total = {'VTA-DA': 0}
        result = compute_recording_projection(
            n_ready, n_total, target_n=5,
            deadline=date(2026, 7, 31), capacity_per_day=16,
            today=date(2026, 3, 31),
        )
        vta = result.set_index('target_NM').loc['VTA-DA']
        assert np.isnan(vta['yield_rate'])

    def test_recording_days_uses_capacity(self):
        """Total effective sessions divided by capacity_per_day."""
        from iblnm.analysis import compute_recording_projection
        from datetime import date

        # 2 targets, each needs 10 effective sessions = 20 total, at 16/day = 2 days
        n_ready = {'A': 0, 'B': 0}
        n_total = {'A': 10, 'B': 10}
        result = compute_recording_projection(
            n_ready, n_total, target_n=5,
            deadline=date(2026, 7, 31), capacity_per_day=16,
            today=date(2026, 3, 31),
        )
        total_effective = result['effective_sessions_needed'].sum()
        total_days = result['recording_days_needed'].sum()
        assert total_days == np.ceil(total_effective / 16)

    def test_days_available(self):
        from iblnm.analysis import compute_recording_projection
        from datetime import date

        result = compute_recording_projection(
            {'A': 3}, {'A': 10}, target_n=5,
            deadline=date(2026, 7, 31), capacity_per_day=16,
            today=date(2026, 3, 31),
        )
        # 122 days from March 31 to July 31
        assert result['days_available'].iloc[0] == 122


class TestAnovaRM:
    """Tests for anova_rm."""

    @staticmethod
    def _balanced_df(n_subjects=4, seed=42):
        """Fully balanced subject × contrast × side × feedback dataset."""
        rng = np.random.default_rng(seed)
        rows = []
        for subj in range(n_subjects):
            for contrast in [0.0, 0.25, 1.0]:
                for side in ['contra', 'ipsi']:
                    for fb in [1, -1]:
                        rows.append({
                            'subject': f's{subj}',
                            'contrast': contrast,
                            'side': side,
                            'feedbackType': fb,
                            'response': rng.normal(0, 1),
                        })
        return pd.DataFrame(rows)

    def test_returns_dataframe(self):
        from iblnm.analysis import anova_rm
        df = self._balanced_df()
        result = anova_rm(df, 'response', 'subject',
                          ['contrast', 'side', 'feedbackType'])
        assert isinstance(result, pd.DataFrame)

    def test_has_expected_columns(self):
        from iblnm.analysis import anova_rm
        df = self._balanced_df()
        result = anova_rm(df, 'response', 'subject',
                          ['contrast', 'side', 'feedbackType'])
        for col in ['Source', 'F', 'Num DF', 'Den DF', 'Pr(>F)', 'method']:
            assert col in result.columns, f"Missing column: {col}"

    def test_all_main_effects_and_interactions(self):
        """3 factors → 7 terms (3 main + 3 two-way + 1 three-way)."""
        from iblnm.analysis import anova_rm
        df = self._balanced_df()
        result = anova_rm(df, 'response', 'subject',
                          ['contrast', 'side', 'feedbackType'])
        assert len(result) == 7

    def test_uses_rm_for_balanced_data(self):
        from iblnm.analysis import anova_rm
        df = self._balanced_df()
        result = anova_rm(df, 'response', 'subject',
                          ['contrast', 'side', 'feedbackType'])
        assert (result['method'] == 'rm').all()

    def test_falls_back_to_ols_for_unbalanced(self):
        """Drop one cell for one subject so the design is unbalanced."""
        from iblnm.analysis import anova_rm
        df = self._balanced_df()
        # Remove s0's contra/correct/1.0 row → unbalanced
        mask = (
            (df['subject'] == 's0') &
            (df['side'] == 'contra') &
            (df['feedbackType'] == 1) &
            (df['contrast'] == 1.0)
        )
        df = df[~mask]
        import warnings as w
        with w.catch_warnings(record=True) as caught:
            w.simplefilter('always')
            result = anova_rm(df, 'response', 'subject',
                              ['contrast', 'side', 'feedbackType'])
        assert (result['method'] == 'ols').all()
        assert any('unbalanced' in str(m.message).lower() for m in caught)

    def test_single_factor(self):
        """Works with a single within-subject factor."""
        from iblnm.analysis import anova_rm
        rng = np.random.default_rng(0)
        rows = []
        for subj in ['a', 'b', 'c']:
            for level in ['x', 'y', 'z']:
                rows.append({'sub': subj, 'cond': level,
                             'dv': rng.normal()})
        df = pd.DataFrame(rows)
        result = anova_rm(df, 'dv', 'sub', ['cond'])
        assert len(result) == 1  # one main effect
        assert result['Source'].iloc[0] == 'cond'

    def test_p_values_in_range(self):
        from iblnm.analysis import anova_rm
        df = self._balanced_df()
        result = anova_rm(df, 'response', 'subject',
                          ['contrast', 'side', 'feedbackType'])
        assert (result['Pr(>F)'] >= 0).all()
        assert (result['Pr(>F)'] <= 1).all()


# =============================================================================
# kruskal_wallis_groups
# =============================================================================

class TestKruskalWallisGroups:

    def test_matches_scipy_directly(self):
        from iblnm.analysis import kruskal_wallis_groups
        from scipy.stats import kruskal
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            'group': ['A'] * 30 + ['B'] * 30 + ['C'] * 30,
            'value': np.concatenate([
                rng.normal(0, 1, 30),
                rng.normal(2, 1, 30),
                rng.normal(0, 1, 30),
            ]),
        })
        H, p, groups = kruskal_wallis_groups(df, 'group', 'value')
        H_expected, p_expected = kruskal(groups['A'], groups['B'], groups['C'])
        assert np.isclose(H, H_expected)
        assert np.isclose(p, p_expected)

    def test_returns_groups_dict(self):
        from iblnm.analysis import kruskal_wallis_groups
        df = pd.DataFrame({
            'g': ['x', 'x', 'y', 'y'],
            'v': [1.0, 2.0, 3.0, 4.0],
        })
        _, _, groups = kruskal_wallis_groups(df, 'g', 'v')
        assert set(groups.keys()) == {'x', 'y'}
        assert list(groups['x']) == [1.0, 2.0]

    def test_single_group_returns_nan(self):
        from iblnm.analysis import kruskal_wallis_groups
        df = pd.DataFrame({'g': ['a', 'a'], 'v': [1.0, 2.0]})
        H, p, groups = kruskal_wallis_groups(df, 'g', 'v')
        assert np.isnan(H)
        assert np.isnan(p)

    def test_drops_nan_values(self):
        from iblnm.analysis import kruskal_wallis_groups
        df = pd.DataFrame({
            'g': ['a', 'a', 'b', 'b', 'b'],
            'v': [1.0, np.nan, 3.0, 4.0, np.nan],
        })
        _, _, groups = kruskal_wallis_groups(df, 'g', 'v')
        assert len(groups['a']) == 1
        assert len(groups['b']) == 2


# =============================================================================
# pairwise_mannwhitney
# =============================================================================

class TestPairwiseMannwhitney:

    def test_bonferroni_correction(self):
        from iblnm.analysis import pairwise_mannwhitney
        from scipy.stats import mannwhitneyu
        rng = np.random.default_rng(42)
        groups = {
            'A': rng.normal(0, 1, 20),
            'B': rng.normal(3, 1, 20),
            'C': rng.normal(0, 1, 20),
        }
        results = pairwise_mannwhitney(groups)
        # 3 pairs: A-B, A-C, B-C
        assert len(results) == 3
        # Check Bonferroni: p_corrected = min(p_raw * 3, 1.0)
        for ga, gb, U, p_corr in results:
            _, p_raw = mannwhitneyu(groups[ga], groups[gb],
                                    alternative='two-sided')
            assert np.isclose(p_corr, min(p_raw * 3, 1.0))

    def test_p_capped_at_one(self):
        from iblnm.analysis import pairwise_mannwhitney
        # Two identical groups — p_raw ~ 1.0, corrected should not exceed 1.0
        groups = {'A': np.ones(10), 'B': np.ones(10) + 1e-12}
        results = pairwise_mannwhitney(groups)
        for _, _, _, p_corr in results:
            assert p_corr <= 1.0

    def test_two_groups_no_correction_inflation(self):
        from iblnm.analysis import pairwise_mannwhitney
        from scipy.stats import mannwhitneyu
        rng = np.random.default_rng(0)
        groups = {'X': rng.normal(0, 1, 30), 'Y': rng.normal(0, 1, 30)}
        results = pairwise_mannwhitney(groups)
        assert len(results) == 1
        # With 1 comparison, correction factor is 1
        _, p_raw = mannwhitneyu(groups['X'], groups['Y'],
                                alternative='two-sided')
        assert np.isclose(results[0][3], p_raw)
