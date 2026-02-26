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
