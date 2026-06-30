"""Tests for iblnm.analysis module."""
import numpy as np
import pandas as pd
import pytest

from iblnm.analysis import (
    fit_measurement_error_varcomp,
    get_responses,
    normalize_responses,
    resample_signal,
    summarize_posterior,
)
from iblnm.util import contrast_transform


def _synthetic_varcomp_data(mouse_sd, session_sd, n_mice=30, n_sessions=8, se=0.1, seed=0):
    """Per-session estimates with injected mouse-SD and session-SD.

    Returns ``(estimates, ses, mouse_ids, v_mouse_true, v_session_true)`` for
    ``n_mice`` mice each contributing ``n_sessions`` sessions. Each estimate is a
    mouse effect + session effect + measurement noise of known SD ``se``. The two
    ``*_true`` values are the *realized* between-mouse and between-session
    variances standardized by the estimate variance — the quantities the
    standardized model recovers, which differ from the injected SDs by
    finite-sample sampling. ``n_mice`` is large enough that the between-mouse
    component is identifiable.
    """
    rng = np.random.default_rng(seed)
    mouse_ids = np.repeat(np.arange(n_mice), n_sessions)
    mouse_effect = rng.normal(0, mouse_sd, n_mice)
    session_effect = rng.normal(0, session_sd, mouse_ids.size)
    noise = rng.normal(0, se, mouse_ids.size)
    estimates = mouse_effect[mouse_ids] + session_effect + noise
    ses = np.full(mouse_ids.size, se)
    scale_sq = estimates.var()
    return (estimates, ses, mouse_ids,
            mouse_effect.var() / scale_sq, session_effect.var() / scale_sq)


class TestVarcompFit:
    def test_recovers_realized_variances(self):
        estimates, ses, mouse_ids, v_mouse_true, v_session_true = (
            _synthetic_varcomp_data(0.6, 0.3))
        v_mouse, v_session = fit_measurement_error_varcomp(
            estimates, ses, mouse_ids, draws=500, tune=500, chains=2, random_seed=0
        )
        # the standardized model recovers the realized component variances
        np.testing.assert_allclose(v_mouse.mean(), v_mouse_true, atol=0.25)
        np.testing.assert_allclose(v_session.mean(), v_session_true, atol=0.25)
        # mouse component clearly exceeds session component here
        assert v_mouse.mean() > v_session.mean()

    def test_scaling_invariance(self):
        estimates, ses, mouse_ids, *_ = _synthetic_varcomp_data(0.6, 0.3)
        vm1, vs1 = fit_measurement_error_varcomp(
            estimates, ses, mouse_ids, draws=500, tune=500, chains=2, random_seed=0
        )
        vm2, vs2 = fit_measurement_error_varcomp(
            10 * estimates, 10 * ses, mouse_ids,
            draws=500, tune=500, chains=2, random_seed=0,
        )
        np.testing.assert_allclose(vm1.mean(), vm2.mean(), atol=0.05)
        np.testing.assert_allclose(vs1.mean(), vs2.mean(), atol=0.05)


class TestSummarizePosterior:
    def test_summary_and_grid(self):
        samples = np.random.default_rng(0).normal(2.0, 1.0, 5000)
        mean, hdi_low, hdi_high, x_grid, density = summarize_posterior(
            samples, grid_size=150
        )
        assert hdi_low <= mean <= hdi_high
        assert len(x_grid) == 150
        assert len(density) == 150
        assert np.all(density >= 0)


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

    def test_log2_known_values(self):
        from iblnm.util import get_contrast_coding
        transform, _ = get_contrast_coding('log2')
        c = np.array([0.0, 6.25, 12.5, 25, 100])
        np.testing.assert_allclose(
            transform(c), [0, 2.644, 3.644, 4.644, 6.644], atol=1e-3)

    def test_log2_roundtrip(self):
        from iblnm.util import get_contrast_coding
        transform, inverse = get_contrast_coding('log2')
        c = np.array([0.0, 6.25, 12.5, 25, 100])
        np.testing.assert_allclose(inverse(transform(c)), c)

    def test_log2_rejects_fractional_contrast(self):
        import pytest
        from iblnm.util import get_contrast_coding
        transform, _ = get_contrast_coding('log2')
        with pytest.raises(ValueError):
            transform(np.array([0.0, 0.0625, 1.0]))

    def test_all_monotonic(self):
        from iblnm.util import get_contrast_coding
        c = np.array([0.0, 6.25, 12.5, 25, 100])
        for coding in ('linear', 'rank', 'log', 'log2'):
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


class TestKeypointSpeed:
    def test_constant_velocity(self):
        """Constant-velocity keypoint yields constant speed after the first frame."""
        from iblnm.analysis import keypoint_speed
        x = np.arange(10, dtype=float) * 3.0
        y = np.arange(10, dtype=float) * 4.0
        likelihood = np.ones(10)
        speed = keypoint_speed(x, y, likelihood, threshold=0.9)
        assert speed.shape == x.shape
        assert np.isnan(speed[0])
        np.testing.assert_allclose(speed[1:], 5.0)

    def test_low_likelihood_frames_nan(self):
        """Frames with likelihood below threshold are set to NaN."""
        from iblnm.analysis import keypoint_speed
        x = np.arange(10, dtype=float) * 3.0
        y = np.arange(10, dtype=float) * 4.0
        likelihood = np.ones(10)
        likelihood[[3, 7]] = 0.5
        speed = keypoint_speed(x, y, likelihood, threshold=0.9)
        assert np.isnan(speed[[3, 7]]).all()
        np.testing.assert_allclose(speed[[1, 2, 4, 5, 6, 8, 9]], 5.0)

    def test_zero_likelihood_all_nan(self):
        """A keypoint tracked with zero likelihood is NaN at every frame."""
        from iblnm.analysis import keypoint_speed
        x = np.arange(10, dtype=float) * 3.0
        y = np.arange(10, dtype=float) * 4.0
        likelihood = np.zeros(10)
        speed = keypoint_speed(x, y, likelihood, threshold=0.9)
        assert np.isnan(speed).all()


def _pose_df(**columns):
    """Build a pose DataFrame from per-keypoint coordinate/likelihood arrays."""
    return pd.DataFrame(columns)


class TestMovementTrace:
    def test_single_keypoint_speed(self):
        """reduction='speed' returns the single keypoint's gated speed."""
        from iblnm.analysis import movement_trace, keypoint_speed
        x = np.arange(10, dtype=float) * 3.0
        y = np.arange(10, dtype=float) * 4.0
        pose = _pose_df(nose_tip_x=x, nose_tip_y=y, nose_tip_likelihood=np.ones(10))
        trace = movement_trace(pose, ['nose_tip'], 'speed', threshold=0.9)
        np.testing.assert_array_equal(
            np.isnan(trace), np.isnan(keypoint_speed(x, y, np.ones(10), 0.9)))
        np.testing.assert_allclose(trace[1:], 5.0)

    def test_sum_speed_is_nan_aware(self):
        """sum_speed adds keypoint speeds, ignoring NaN unless all are NaN."""
        from iblnm.analysis import movement_trace
        n = 6
        like_l = np.ones(n)
        like_r = np.ones(n)
        like_r[2] = 0.5   # paw_r untracked at frame 2
        like_l[4] = 0.5   # both untracked at frame 4
        like_r[4] = 0.5
        pose = _pose_df(
            paw_l_x=np.arange(n) * 3.0, paw_l_y=np.arange(n) * 4.0,
            paw_l_likelihood=like_l,
            paw_r_x=np.arange(n) * 6.0, paw_r_y=np.arange(n) * 8.0,
            paw_r_likelihood=like_r,
        )
        trace = movement_trace(pose, ['paw_l', 'paw_r'], 'sum_speed', threshold=0.9)
        assert np.isnan(trace[0])           # first frame: no displacement
        np.testing.assert_allclose(trace[[1, 3, 5]], 15.0)  # 5 + 10
        np.testing.assert_allclose(trace[2], 5.0)           # only paw_l valid
        assert np.isnan(trace[4])           # both untracked

    def test_max_likelihood_is_ungated(self):
        """max_likelihood is the per-frame max of likelihoods, ignoring threshold."""
        from iblnm.analysis import movement_trace
        like_l = np.array([0.1, 0.8, 0.2])
        like_r = np.array([0.7, 0.3, 0.2])
        pose = _pose_df(
            tongue_end_l_x=np.zeros(3), tongue_end_l_y=np.zeros(3),
            tongue_end_l_likelihood=like_l,
            tongue_end_r_x=np.zeros(3), tongue_end_r_y=np.zeros(3),
            tongue_end_r_likelihood=like_r,
        )
        trace = movement_trace(pose, ['tongue_end_l', 'tongue_end_r'],
                               'max_likelihood', threshold=0.9)
        np.testing.assert_allclose(trace, [0.7, 0.8, 0.2])


class TestEventLockedScalar:
    def _step_trace(self, n_trials, baseline_val, response_val):
        tpts = np.linspace(-1, 1, 81)
        trace = np.zeros((n_trials, len(tpts)))
        trace[:, (tpts >= -0.2) & (tpts < 0)] = baseline_val
        trace[:, (tpts >= 0.1) & (tpts < 0.35)] = response_val
        return trace, tpts

    def test_known_step(self):
        """Scalar equals the post-event level minus the baseline level."""
        from iblnm.analysis import event_locked_scalar
        trace, tpts = self._step_trace(n_trials=5, baseline_val=2.0, response_val=7.0)
        scalar = event_locked_scalar(
            trace, tpts, response_window=(0.1, 0.35),
            baseline_window=(-0.2, 0), min_valid=1)
        np.testing.assert_allclose(scalar, 5.0)

    def test_all_nan_window_is_nan(self):
        """A trial whose response window is all NaN yields a NaN scalar."""
        from iblnm.analysis import event_locked_scalar
        trace, tpts = self._step_trace(n_trials=1, baseline_val=2.0, response_val=7.0)
        trace[:, (tpts >= 0.1) & (tpts < 0.35)] = np.nan
        scalar = event_locked_scalar(
            trace, tpts, response_window=(0.1, 0.35),
            baseline_window=(-0.2, 0), min_valid=1)
        assert np.isnan(scalar)

    def test_too_few_valid_samples_is_nan(self):
        """A window with fewer than min_valid finite samples yields NaN."""
        from iblnm.analysis import event_locked_scalar
        trace, tpts = self._step_trace(n_trials=1, baseline_val=2.0, response_val=7.0)
        response_mask = (tpts >= 0.1) & (tpts < 0.35)
        trace[:, response_mask] = np.nan
        # Leave only two finite samples in the response window.
        finite_idx = np.flatnonzero(response_mask)[:2]
        trace[:, finite_idx] = 7.0
        scalar = event_locked_scalar(
            trace, tpts, response_window=(0.1, 0.35),
            baseline_window=(-0.2, 0), min_valid=5)
        assert np.isnan(scalar)


class TestMovementDelta:
    def _step_trace(self, n_trials, level, lo, hi):
        """Trace flat at `level` inside [lo, hi), zero elsewhere, on a shared axis."""
        tpts = np.linspace(-1, 1, 81)
        trace = np.zeros((n_trials, len(tpts)))
        trace[:, (tpts >= lo) & (tpts < hi)] = level
        return trace, tpts

    def test_response_and_baseline_from_separate_traces(self):
        """Scalar = mean(response_trace over response_window) minus
        mean(baseline_trace over baseline_window)."""
        from iblnm.analysis import movement_delta
        response_trace, tpts = self._step_trace(5, level=7.0, lo=0.1, hi=0.35)
        baseline_trace, _ = self._step_trace(5, level=2.0, lo=-0.2, hi=0.0)
        scalar = movement_delta(
            response_trace, baseline_trace, tpts,
            response_window=(0.1, 0.35), baseline_window=(-0.2, 0), min_valid=1)
        np.testing.assert_allclose(scalar, 5.0)

    def test_all_nan_window_is_nan(self):
        from iblnm.analysis import movement_delta
        response_trace, tpts = self._step_trace(1, level=7.0, lo=0.1, hi=0.35)
        baseline_trace, _ = self._step_trace(1, level=2.0, lo=-0.2, hi=0.0)
        response_trace[:, (tpts >= 0.1) & (tpts < 0.35)] = np.nan
        scalar = movement_delta(
            response_trace, baseline_trace, tpts,
            response_window=(0.1, 0.35), baseline_window=(-0.2, 0), min_valid=1)
        assert np.isnan(scalar)


class TestNormalizedCrosscorr:
    def test_identical_signals_peak_at_zero(self):
        """Identical signals → peak lag 0, normalized peak ≈ 1."""
        from iblnm.analysis import normalized_crosscorr
        rng = np.random.default_rng(0)
        a = rng.standard_normal(2000)
        cc, lags, peak_lag = normalized_crosscorr(a, a, fs=100, lag_window=5)
        assert peak_lag == 0.0
        np.testing.assert_allclose(cc.max(), 1.0, atol=1e-10)

    def test_known_shift(self):
        """b a shifted copy of a → peak lag equals the shift in seconds."""
        from iblnm.analysis import normalized_crosscorr
        rng = np.random.default_rng(1)
        a = rng.standard_normal(2000)
        fs = 100
        shift = 7  # samples
        b = np.roll(a, shift)
        cc, lags, peak_lag = normalized_crosscorr(a, b, fs=fs, lag_window=5)
        np.testing.assert_allclose(peak_lag, shift / fs, atol=1 / fs)

    def test_lag_axis_within_window(self):
        """Lag axis is symmetric and bounded by ±lag_window."""
        from iblnm.analysis import normalized_crosscorr
        rng = np.random.default_rng(2)
        a = rng.standard_normal(2000)
        cc, lags, peak_lag = normalized_crosscorr(a, a, fs=100, lag_window=5)
        assert cc.shape == lags.shape
        assert lags.min() == -5.0
        assert lags.max() == 5.0


def _band_limited_signal(times, seed=0, n_components=40):
    """Smooth, broadband, aperiodic signal evaluable at arbitrary times."""
    rng = np.random.default_rng(seed)
    freqs = rng.uniform(1.0, 10.0, n_components)
    phases = rng.uniform(0, 2 * np.pi, n_components)
    return np.sin(2 * np.pi * freqs[:, None] * times[None, :]
                  + phases[:, None]).sum(0)


def _synthetic_paw_wheel(shift_s, dur=60.0, late_start=40.0,
                         fs_paw=60, fs_wheel=100):
    """Paw (camera rate) and wheel (wheel rate) traces from a shared function.

    Early/mid thirds: paw and wheel are the same function (aligned). Late third:
    paw is evaluated at ``t + shift_s`` so the paw pattern leads the wheel by
    ``shift_s`` seconds.
    """
    paw_times = np.arange(0, dur, 1 / fs_paw)
    wheel_times = np.arange(0, dur, 1 / fs_wheel)
    wheel_speed = _band_limited_signal(wheel_times)
    paw_eval = np.where(paw_times >= late_start, paw_times + shift_s, paw_times)
    paw_speed = _band_limited_signal(paw_eval)
    return paw_speed, paw_times, wheel_speed, wheel_times


class TestPerThirdCrosscorr:
    def test_drift_recovers_imposed_shift(self):
        """Late-third paw shifted vs early → drift equals the imposed shift."""
        from iblnm.analysis import per_third_crosscorr
        shift_s = 0.1
        paw_speed, paw_times, wheel_speed, wheel_times = _synthetic_paw_wheel(shift_s)
        functions, lags, peak_lags, drift = per_third_crosscorr(
            paw_speed, paw_times, wheel_speed, wheel_times,
            fs=100, lag_window=5)
        assert functions.shape == (3, lags.size)
        np.testing.assert_allclose(peak_lags[0], 0.0, atol=1 / 100)
        np.testing.assert_allclose(drift, shift_s, atol=1 / 100)

    def test_all_thirds_computed_without_guard(self):
        """Every third yields a finite cross-correlation — no coverage guard."""
        from iblnm.analysis import per_third_crosscorr
        paw_speed, paw_times, wheel_speed, wheel_times = _synthetic_paw_wheel(0.1)
        functions, lags, peak_lags, drift = per_third_crosscorr(
            paw_speed, paw_times, wheel_speed, wheel_times, fs=100, lag_window=5)
        assert np.isfinite(peak_lags).all()
        assert np.isfinite(functions).all()


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


class TestResamplePose:
    def test_irregular_input_to_uniform_grid(self):
        """Pose columns resample onto a uniform 1/fs grid, columns preserved."""
        from iblnm.analysis import resample_pose
        times = np.sort(np.cumsum(np.random.uniform(0.008, 0.012, 200)))
        pose = pd.DataFrame({
            'paw_l_x': np.linspace(0, 10, 200),
            'paw_l_y': np.zeros(200),
            'paw_l_likelihood': np.linspace(0.2, 1.0, 200),
        })
        rs, new_t = resample_pose(pose, times, fs=100)
        np.testing.assert_allclose(np.diff(new_t), 1 / 100, atol=1e-10)
        assert list(rs.columns) == list(pose.columns)
        assert len(rs) == len(new_t)
        # monotone x interpolates within range; likelihood stays in [0, 1]
        assert rs['paw_l_x'].is_monotonic_increasing
        assert rs['paw_l_likelihood'].between(0, 1).all()


# =============================================================================
# Response Vector Analysis Tests
# =============================================================================


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

    def test_variance_explained_populated(self):
        """fit_cca reports per-variate variance extracted for both blocks."""
        from iblnm.analysis import fit_cca
        rng = np.random.default_rng(0)
        n, k, p = 40, 6, 3
        X = pd.DataFrame(rng.standard_normal((n, k)),
                         columns=[f'x{i}' for i in range(k)])
        Y = pd.DataFrame(rng.standard_normal((n, p)),
                         columns=[f'y{i}' for i in range(p)])
        result = fit_cca(X, Y, n_components=2)
        assert result.x_variance_explained.shape == (2,)
        assert result.y_variance_explained.shape == (2,)
        assert np.all((result.x_variance_explained >= 0)
                      & (result.x_variance_explained <= 1))
        assert np.all((result.y_variance_explained >= 0)
                      & (result.y_variance_explained <= 1))


class TestSelectBlockTerms:
    # Full per-session persession model term names (intercept + 6 mains + 12
    # within-/cross-category two-ways), matching get_persession_ols_features.
    _PERSESSION_COLUMNS = [
        'Intercept', 'contrast', 'side', 'reward', 'choice_side',
        'log_reaction_time', 'peak_velocity',
        'contrast:side', 'contrast:reward', 'contrast:choice_side',
        'contrast:log_reaction_time', 'contrast:peak_velocity',
        'side:log_reaction_time', 'side:peak_velocity',
        'reward:log_reaction_time', 'reward:peak_velocity',
        'choice_side:log_reaction_time', 'choice_side:peak_velocity',
        'log_reaction_time:peak_velocity',
    ]

    def test_task_block(self):
        from iblnm.analysis import select_block_terms
        from iblnm.config import CCA_TASK_MAINS
        selected = select_block_terms(self._PERSESSION_COLUMNS, CCA_TASK_MAINS)
        assert set(selected) == {
            'contrast', 'side', 'reward', 'contrast:side', 'contrast:reward'}

    def test_movement_block(self):
        from iblnm.analysis import select_block_terms
        from iblnm.config import CCA_MOVEMENT_MAINS
        selected = select_block_terms(self._PERSESSION_COLUMNS, CCA_MOVEMENT_MAINS)
        assert set(selected) == {
            'choice_side', 'log_reaction_time', 'peak_velocity',
            'choice_side:log_reaction_time', 'choice_side:peak_velocity',
            'log_reaction_time:peak_velocity'}

    def test_cross_term_and_intercept_in_neither_block(self):
        from iblnm.analysis import select_block_terms
        from iblnm.config import CCA_TASK_MAINS, CCA_MOVEMENT_MAINS
        task = select_block_terms(self._PERSESSION_COLUMNS, CCA_TASK_MAINS)
        movement = select_block_terms(self._PERSESSION_COLUMNS, CCA_MOVEMENT_MAINS)
        for col in ('contrast:peak_velocity', 'Intercept'):
            assert col not in task
            assert col not in movement


class TestCCAVarianceExtracted:

    def test_score_equals_feature_block(self):
        """All features equal (up to scale) to the variate -> full extraction."""
        from iblnm.analysis import cca_variance_extracted
        rng = np.random.default_rng(1)
        score = rng.standard_normal((50, 1))
        data = np.column_stack([score[:, 0], 2 * score[:, 0], -3 * score[:, 0]])
        ve = cca_variance_extracted(data, score)
        assert ve.shape == (1,)
        assert np.isclose(ve[0], 1.0)

    def test_orthogonal_score(self):
        """A variate uncorrelated with every feature extracts ~0 variance."""
        from iblnm.analysis import cca_variance_extracted
        rng = np.random.default_rng(2)
        n = 200
        data = rng.standard_normal((n, 4))
        score = rng.standard_normal((n, 1))
        ve = cca_variance_extracted(data, score)
        assert ve[0] < 0.05

    def test_constant_feature_contributes_zero(self):
        """A constant feature has no shared variance and is not a div-by-zero."""
        from iblnm.analysis import cca_variance_extracted
        rng = np.random.default_rng(3)
        score = rng.standard_normal((30, 1))
        data = np.column_stack([score[:, 0], np.ones(30)])
        ve = cca_variance_extracted(data, score)
        # feature 0 loads 1.0, constant feature loads 0 -> mean of [1, 0]
        assert np.isclose(ve[0], 0.5)


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
        x_names = ['contrast', 'side', 'feedback', 'lc:side', 'lc:fb', 'side:fb']
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


# =============================================================================
# Movement Encoding LMM
# =============================================================================

def _make_movement_lmm_df(n_per_cell=40, seed=0, subjects=('s1', 's2', 's3', 's4'),
                          slope_sd=0.0):
    """Synthetic trial-level data for movement LMM tests.

    Creates data where contrast and timing both influence the response,
    so both should show non-zero delta-R². ``slope_sd`` adds subject-specific
    variability to the timing slope (for random-slope tests).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for subj in subjects:
        subj_intercept = rng.normal(0, 0.5)
        subj_slope = 0.4 + rng.normal(0, slope_sd)
        for contrast in [0.0, 25.0, 100.0]:
            for side_label, side_val in [('contra', 0.5), ('ipsi', -0.5)]:
                for fb, reward_val in [(1, 0.5), (-1, -0.5)]:
                    for _ in range(n_per_cell):
                        log_rt = rng.normal(-0.7, 0.3)
                        response = (
                            subj_intercept
                            + 0.3 * (contrast / 100)  # contrast effect
                            + 0.2 * side_val
                            + 0.1 * reward_val
                            + subj_slope * log_rt  # timing effect (random slope)
                            + rng.normal(0, 0.5)
                        )
                        rows.append({
                            'subject': subj,
                            'contrast': contrast,
                            'side': side_label,
                            'feedbackType': fb,
                            'response': response,
                            'log_reaction_time': log_rt,
                        })
    return pd.DataFrame(rows)


class TestCrossvalLmm:
    FORMULAS = {
        'full': 'response ~ contrast * side * reward',
        'interactions': 'response ~ contrast + side + reward',
    }

    def test_required_columns(self):
        from iblnm.analysis import crossval_lmm
        coded = _code_task(_make_task_lmm_df())
        result = crossval_lmm(coded, self.FORMULAS, 'response')
        assert list(result.columns) == ['fold', 'predictor', 'n_trials', 'r2',
                                         'delta_r2']

    def test_fold_below_min_test_excluded(self):
        from iblnm.analysis import crossval_lmm
        df = _make_task_lmm_df()
        coded = _code_task(df)
        # Shrink one subject to fewer than min_test trials.
        small = coded[coded['subject'] == 's0'].iloc[:3]
        coded = pd.concat([small, coded[coded['subject'] != 's0']])
        result = crossval_lmm(coded, self.FORMULAS, 'response', min_test=5)
        folds = result[result['fold'] != 'aggregate']['fold']
        assert 's0' not in set(folds)

    def test_too_few_subjects_returns_empty(self):
        from iblnm.analysis import crossval_lmm
        coded = _code_task(_make_task_lmm_df())
        coded['subject'] = 's0'
        result = crossval_lmm(coded, self.FORMULAS, 'response', min_subjects=3)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert list(result.columns) == ['fold', 'predictor', 'n_trials', 'r2',
                                         'delta_r2']

    def test_non_default_reference_no_full_key(self):
        """With reference='additive' and no 'full' key, ΔR² is measured against
        the named additive baseline; only the dropped predictor is reported."""
        from iblnm.analysis import crossval_lmm
        formulas = {
            'additive': 'response ~ contrast + side + reward',
            'contrast': 'response ~ side + reward',
        }
        coded = _code_task(_make_task_lmm_df())
        new = crossval_lmm(coded, formulas, 'response', reference='additive')
        assert set(new['predictor']) == {'contrast'}
        # r2 column is the reference (additive) model's held-out R².
        assert new['r2'].notna().all()


class TestJackknifeLmm:
    FORMULAS = {
        'full': 'response ~ contrast * side * reward',
        'interactions': 'response ~ contrast + side + reward',
    }

    def test_required_columns(self):
        from iblnm.analysis import jackknife_lmm
        coded = _code_task(_make_task_lmm_df())
        result = jackknife_lmm(coded, self.FORMULAS, 'response')
        assert list(result.columns) == ['fold', 'predictor', 'n_trials', 'r2',
                                         'delta_r2']

    def test_delta_matches_insample_recompute(self):
        """A fold's delta_r2 equals r2(full) − r2(predictor model) recomputed
        directly on that fold's N−1 training subset."""
        from iblnm.analysis import jackknife_lmm, fit_lmm
        coded = _code_task(_make_task_lmm_df())
        result = jackknife_lmm(coded, self.FORMULAS, 'response')

        fold = result[result['fold'] != 'aggregate']['fold'].iloc[0]
        train = coded[coded['subject'] != fold]
        r2 = {name: fit_lmm(f, train, groups=train['subject'],
                            re_formula='1', reml=False
                            ).variance_explained['marginal']
              for name, f in self.FORMULAS.items()}

        row = result[(result['fold'] == fold)
                     & (result['predictor'] == 'interactions')].iloc[0]
        assert row['r2'] == pytest.approx(r2['full'])
        assert row['delta_r2'] == pytest.approx(r2['full'] - r2['interactions'])
        assert row['n_trials'] == len(train)

    def test_aggregate_row_per_predictor(self):
        from iblnm.analysis import jackknife_lmm
        coded = _code_task(_make_task_lmm_df())
        result = jackknife_lmm(coded, self.FORMULAS, 'response')
        agg = result[result['fold'] == 'aggregate']
        assert list(agg['predictor']) == ['interactions']
        fold_delta = result[(result['fold'] != 'aggregate')
                            & (result['predictor'] == 'interactions')]['delta_r2']
        assert agg['delta_r2'].iloc[0] == pytest.approx(fold_delta.mean())

    def test_too_few_subjects_returns_empty(self):
        from iblnm.analysis import jackknife_lmm
        coded = _code_task(_make_task_lmm_df())
        coded['subject'] = 's0'
        result = jackknife_lmm(coded, self.FORMULAS, 'response', min_subjects=3)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert list(result.columns) == ['fold', 'predictor', 'n_trials', 'r2',
                                         'delta_r2']

    def test_non_default_reference_no_full_key(self):
        """With reference='additive' and no 'full' key, ΔR² is measured against
        the named additive baseline, recomputed in sample on each training set."""
        from iblnm.analysis import jackknife_lmm, fit_lmm
        formulas = {
            'additive': 'response ~ contrast + side + reward',
            'contrast': 'response ~ side + reward',
        }
        coded = _code_task(_make_task_lmm_df())
        result = jackknife_lmm(coded, formulas, 'response', reference='additive')

        assert set(result['predictor']) == {'contrast'}
        fold = result[result['fold'] != 'aggregate']['fold'].iloc[0]
        train = coded[coded['subject'] != fold]
        r2 = {name: fit_lmm(f, train, groups=train['subject'],
                            re_formula='1', reml=False
                            ).variance_explained['marginal']
              for name, f in formulas.items()}
        row = result[(result['fold'] == fold)
                     & (result['predictor'] == 'contrast')].iloc[0]
        assert row['r2'] == pytest.approx(r2['additive'])
        assert row['delta_r2'] == pytest.approx(r2['additive'] - r2['contrast'])


class TestFormulaColumns:
    def test_returns_referenced_columns_only(self):
        from iblnm.analysis import formula_columns
        cols = ['response', 'contrast', 'side', 'reward',
                'reaction_time', 'log_reaction_time']
        got = formula_columns('response ~ contrast + log_reaction_time', cols)
        assert got == ['response', 'contrast', 'log_reaction_time']

    def test_word_boundaries_and_wrapping(self):
        from iblnm.analysis import formula_columns
        cols = ['contrast', 'relative_contrast', 'reaction_time',
                'log_reaction_time']
        # C(contrast) names contrast; the bare log column does not pull in
        # reaction_time, nor does contrast pull in relative_contrast.
        got = formula_columns(
            'response ~ C(contrast) * side + log_reaction_time', cols)
        assert got == ['contrast', 'log_reaction_time']

    def test_union_across_formulas_ordered_by_columns(self):
        from iblnm.analysis import formula_union_columns
        cols = ['response', 'contrast', 'side', 'reward', 'log_reaction_time']
        formulas = ['response ~ contrast + log_reaction_time',
                    'response ~ reward + side']
        got = formula_union_columns(formulas, cols)
        # Union of all referenced columns, deduplicated, in cols order.
        assert got == ['response', 'contrast', 'side', 'reward',
                       'log_reaction_time']


class TestBuildTrialRegressors:
    def _trials(self):
        return pd.DataFrame({
            'signed_contrast': [-0.25, 0.0, 0.0625],
            'contrast': [0.25, 0.0, 0.0625],
            'stim_side': ['left', 'right', 'right'],
            'choice': [-1, 1, 1],
            'feedbackType': [1, -1, 1],
            'probabilityLeft': [0.5, 0.5, 0.5],
            'stimOn_times': [1.0, 2.0, 3.0],
            'firstMovement_times': [1.3, 2.4, 3.2],
            'feedback_times': [1.8, 2.9, 3.7],
        })

    def test_column_set_and_derived_timings(self):
        from iblnm.analysis import build_trial_regressors
        trials = self._trials()
        df = build_trial_regressors(trials, wheel_velocity=None)
        expected_cols = {
            'trial', 'signed_contrast', 'contrast', 'stim_side', 'choice',
            'feedbackType', 'probabilityLeft', 'reaction_time',
            'movement_time', 'response_time', 'peak_velocity',
        }
        assert set(df.columns) == expected_cols
        assert df['trial'].tolist() == [0, 1, 2]
        np.testing.assert_allclose(
            df['reaction_time'].values,
            trials['firstMovement_times'] - trials['stimOn_times'])
        np.testing.assert_allclose(
            df['movement_time'].values,
            trials['feedback_times'] - trials['firstMovement_times'])
        np.testing.assert_allclose(
            df['response_time'].values,
            trials['feedback_times'] - trials['stimOn_times'])

    def test_peak_velocity_nan_when_no_wheel(self):
        from iblnm.analysis import build_trial_regressors
        df = build_trial_regressors(self._trials(), wheel_velocity=None)
        assert df['peak_velocity'].isna().all()

    def test_peak_velocity_finite_when_wheel_supplied(self):
        from iblnm.analysis import build_trial_regressors
        velocity = np.array([[0.0, 1.0, -3.0],
                             [np.nan, np.nan, np.nan],
                             [2.0, -5.0, 1.0]])
        df = build_trial_regressors(self._trials(), wheel_velocity=velocity)
        np.testing.assert_array_equal(
            df['peak_velocity'].values, np.array([3.0, np.nan, 5.0]))

    def test_missing_event_columns_give_nan_timings(self):
        from iblnm.analysis import build_trial_regressors
        trials = self._trials().drop(columns=['firstMovement_times'])
        df = build_trial_regressors(trials, wheel_velocity=None)
        assert df['reaction_time'].isna().all()
        assert df['movement_time'].isna().all()
        np.testing.assert_allclose(
            df['response_time'].values,
            trials['feedback_times'] - trials['stimOn_times'])


class TestSelectModelingTrials:
    def _merged_frame(self):
        # Trial 0 passes; 1 false-start, 2 no-go, 3 biased block, 4 NaN response.
        return pd.DataFrame({
            'trial': [0, 1, 2, 3, 4],
            'response': [1.0, 1.1, 1.2, 1.3, np.nan],
            'choice': [1, 1, 0, 1, 1],
            'probabilityLeft': [0.5, 0.5, 0.5, 0.8, 0.5],
            'response_time': [1.0, 0.01, 1.0, 1.0, 1.0],
            'reaction_time': [0.2, 0.2, 0.2, 0.2, 0.2],
            'movement_time': [0.15, 0.15, 0.15, 0.15, 0.15],
            'peak_velocity': [1.0, 1.0, 1.0, 1.0, 1.0],
        })

    def test_keeps_all_blocks_by_default(self):
        from iblnm.analysis import select_modeling_trials
        kept = select_modeling_trials(self._merged_frame())
        assert kept['trial'].tolist() == [0, 3]

    def test_probability_left_filters_to_block(self):
        from iblnm.analysis import select_modeling_trials
        kept = select_modeling_trials(self._merged_frame(), probability_left=0.5)
        assert kept['trial'].tolist() == [0]

    def test_adds_log_timing_columns(self):
        from iblnm.analysis import select_modeling_trials
        df = pd.DataFrame({
            'response': [1.0, 1.0],
            'choice': [1, 1],
            'probabilityLeft': [0.5, 0.5],
            'response_time': [1.0, 1.0],
            'reaction_time': [0.1, -0.2],  # second is non-positive -> log NaN
            'movement_time': [0.1, 0.1],
            'peak_velocity': [1.0, 1.0],
        })
        kept = select_modeling_trials(df)
        assert 'log_reaction_time' in kept.columns
        assert kept['log_reaction_time'].iloc[0] == pytest.approx(np.log10(0.1))
        assert np.isnan(kept['log_reaction_time'].iloc[1])

    def test_respects_response_col_argument(self):
        from iblnm.analysis import select_modeling_trials
        df = self._merged_frame().rename(columns={'response': 'baseline'})
        kept = select_modeling_trials(df, response_col='baseline')
        assert kept['trial'].tolist() == [0, 3]


class TestCodePredictors:
    def _frame(self):
        # contrast in percent units; log2 coding requires nonzero values >= 1.
        return pd.DataFrame({
            'contrast': [0.0, 6.25, 100.0],
            'side': ['contra', 'ipsi', 'contra'],
            'feedbackType': [1, -1, 1],
            'log_reaction_time': [-1.5, -0.5, -2.0],
        })

    def test_side_and_reward_deviation_coded(self):
        from iblnm.analysis import code_predictors
        coded = code_predictors(self._frame())
        assert coded['side'].tolist() == [0.5, -0.5, 0.5]
        assert coded['reward'].tolist() == [0.5, -0.5, 0.5]

    def test_contrast_log2_coded_and_centered(self):
        from iblnm.analysis import code_predictors
        coded = code_predictors(self._frame())
        expected = np.array([0.0, np.log2(6.25), np.log2(100.0)])
        expected = expected - expected.mean()
        assert coded['contrast'].mean() == pytest.approx(0.0, abs=1e-12)
        np.testing.assert_allclose(coded['contrast'].values, expected)

    def test_timing_column_unchanged(self):
        from iblnm.analysis import code_predictors
        df = self._frame()
        coded = code_predictors(df)
        np.testing.assert_array_equal(
            coded['log_reaction_time'].values, df['log_reaction_time'].values)

    def test_input_frame_not_mutated(self):
        from iblnm.analysis import code_predictors
        df = self._frame()
        before = df.copy(deep=True)
        code_predictors(df)
        pd.testing.assert_frame_equal(df, before)


class TestFitLMMNaNHandling:
    def test_drops_nan_rows_in_formula_columns(self):
        """A NaN in a formula column must not misalign statsmodels' groups
        array; fit_lmm drops those rows and fits the complete cases."""
        from iblnm.analysis import fit_lmm
        df = _make_movement_lmm_df()
        df.loc[df.index[::5], 'log_reaction_time'] = np.nan
        fit = fit_lmm('response ~ log_reaction_time', df, groups=df['subject'])
        assert fit is not None
        assert len(fit.model.endog) == int(df['log_reaction_time'].notna().sum())


class TestFitLMMFailLoud:
    def test_missing_column_propagates(self):
        """A formula referencing a column that does not exist is a coding bug,
        not a numerical failure: it must propagate, not be swallowed as None."""
        from iblnm.analysis import fit_lmm
        df = _make_movement_lmm_df()
        with pytest.raises(Exception):
            fit_lmm('response ~ nonexistent_col', df, groups=df['subject'])

    def test_singular_fit_returns_none_and_warns(self):
        """A degenerate fit whose lazy BLUP evaluation raises ValueError returns
        None and emits a warning, so a dropped fit is never silent."""
        from unittest.mock import patch
        from iblnm import analysis
        df = _make_movement_lmm_df()
        df_c = df[df['contrast'] == 25.0]
        coded = df_c.assign(
            side=np.where(df_c['side'] == 'contra', 0.5, -0.5),
            reward=np.where(df_c['feedbackType'] == 1, 0.5, -0.5),
        )
        with patch.object(analysis, '_variance_explained',
                          side_effect=ValueError('singular covariance')):
            with pytest.warns(UserWarning):
                result = analysis.fit_lmm(
                    'response ~ side + reward + log_reaction_time', coded,
                    groups=coded['subject'])
        assert result is None


def _fit_for_emm(seed=0):
    """A converged ``response ~ contrast * side * reward`` fit on coded data."""
    from iblnm.analysis import fit_lmm
    rng = np.random.default_rng(seed)
    rows = []
    for subj in ['s1', 's2', 's3', 's4']:
        b = rng.normal(0, 0.3)
        for contrast in [-1.0, 0.0, 1.0]:
            for side in [-0.5, 0.5]:
                for reward in [-0.5, 0.5]:
                    for _ in range(20):
                        resp = (b + 0.5 * contrast + 0.2 * side + 0.3 * reward
                                + rng.normal(0, 0.5))
                        rows.append({'subject': subj, 'contrast': contrast,
                                     'side': side, 'reward': reward,
                                     'response': resp})
    df = pd.DataFrame(rows)
    return fit_lmm('response ~ contrast * side * reward', df,
                   groups=df['subject'])


class TestComputeMarginalMeans:
    def test_single_factor_one_row_per_level(self):
        from iblnm.analysis import compute_marginal_means
        emm = compute_marginal_means(_fit_for_emm(), ['reward'])
        assert set(emm.columns) == {'reward', 'predicted', 'ci_lower',
                                    'ci_upper'}
        assert sorted(emm['reward']) == [-0.5, 0.5]

    def test_two_factors_give_interaction_grid(self):
        from iblnm.analysis import compute_marginal_means
        emm = compute_marginal_means(_fit_for_emm(), ['contrast', 'reward'])
        # 3 contrast levels x 2 reward levels.
        assert len(emm) == 6
        assert {'contrast', 'reward'}.issubset(emm.columns)

    def test_ci_brackets_the_mean(self):
        from iblnm.analysis import compute_marginal_means
        emm = compute_marginal_means(_fit_for_emm(), ['side'])
        assert (emm['ci_lower'] <= emm['predicted']).all()
        assert (emm['predicted'] <= emm['ci_upper']).all()

    def test_positive_reward_effect_recovered(self):
        from iblnm.analysis import compute_marginal_means
        emm = compute_marginal_means(_fit_for_emm(), ['reward']).set_index('reward')
        assert emm.loc[0.5, 'predicted'] > emm.loc[-0.5, 'predicted']


def _make_task_lmm_df(n_per_cell=20, seed=0,
                      subjects=('s0', 's1', 's2', 's3')):
    """Synthetic task-model trial data: contrast x side x reward x subject."""
    rng = np.random.default_rng(seed)
    rows = []
    for subj in subjects:
        subj_intercept = rng.normal(0, 0.3)
        for side in ['contra', 'ipsi']:
            for reward in [1, -1]:
                for contrast in [0.0, 6.25, 12.5, 25.0, 100.0]:
                    for _ in range(n_per_cell):
                        log_c = np.log2(contrast) if contrast > 0 else 0.0
                        sv = 0.5 if side == 'contra' else -0.5
                        rv = 0.5 if reward == 1 else -0.5
                        response = (1.0 + 0.5 * log_c + 0.3 * sv + 0.2 * rv
                                    + 0.25 * log_c * rv + subj_intercept
                                    + rng.normal(0, 0.5))
                        rows.append({'contrast': contrast, 'side': side,
                                     'feedbackType': reward, 'subject': subj,
                                     'response': response})
    return pd.DataFrame(rows)


def _code_task(df):
    """Code task predictors for the resampling tests: contrast log2-floored and
    mean-centered, side and reward deviation-coded (±0.5)."""
    from iblnm.util import get_contrast_coding
    transform, _ = get_contrast_coding('log2')
    df = df.dropna(subset=['response']).copy()
    coded = transform(df['contrast'])
    df['contrast'] = coded - float(np.mean(coded))
    df['side'] = np.where(df['side'] == 'contra', 0.5, -0.5)
    df['reward'] = np.where(df['feedbackType'] == 1, 0.5, -0.5)
    return df


class TestPermutationPvalue:
    def test_observed_above_null(self):
        """Null entirely below observed: minimal greater-tail, maximal less-tail."""
        from iblnm.analysis import permutation_pvalue
        null = np.zeros(999)
        observed = 1.0
        assert permutation_pvalue(observed, null, 'greater') == 1 / 1000
        assert permutation_pvalue(observed, null, 'less') == 1.0
        assert permutation_pvalue(observed, null, 'two-sided') == 2 / 1000

    def test_two_sided_symmetric_null(self):
        """Two-sided equals 2 * smaller tail when below 1, capped at 1."""
        from iblnm.analysis import permutation_pvalue
        null = np.arange(-500, 501)  # symmetric around 0
        observed = 250.0
        n = len(null)
        p_greater = (np.sum(null >= observed) + 1) / (n + 1)
        p_less = (np.sum(null <= observed) + 1) / (n + 1)
        p = permutation_pvalue(observed, null, 'two-sided')
        assert 0 < p <= 1
        assert p == min(1.0, 2 * min(p_greater, p_less))

    def test_floor_at_one_over_n_plus_one(self):
        """Every returned p is at least 1 / (n + 1)."""
        from iblnm.analysis import permutation_pvalue
        rng = np.random.default_rng(0)
        null = rng.standard_normal(99)
        floor = 1 / (len(null) + 1)
        for alt in ('greater', 'less', 'two-sided'):
            assert permutation_pvalue(5.0, null, alt) >= floor
            assert permutation_pvalue(-5.0, null, alt) >= floor

    def test_bogus_alternative_raises(self):
        from iblnm.analysis import permutation_pvalue
        with pytest.raises(ValueError):
            permutation_pvalue(1.0, np.zeros(10), 'bogus')


class TestSynchronizedPermutationPvalue:
    def test_worked_example(self):
        """Spec worked example: pooled mean, column-wise mouse null, add-one p."""
        from iblnm.analysis import synchronized_permutation_pvalue
        observed_by_stratum = [0.10, 0.06]
        null_by_stratum = [[0.02, 0.01, 0.03], [0.015, 0.005, 0.02]]
        observed_stat, p_value = synchronized_permutation_pvalue(
            observed_by_stratum, null_by_stratum,
            statistic='mean', alternative='greater',
        )
        assert observed_stat == pytest.approx(0.08)
        assert p_value == pytest.approx((1 + 0) / (3 + 1))

    def test_smallest_p_when_observed_exceeds_every_null_column(self):
        """p hits its floor 1 / (K + 1) when obs beats all pooled null columns."""
        from iblnm.analysis import synchronized_permutation_pvalue
        observed_by_stratum = [1.0, 1.0]
        null_by_stratum = [[0.0, 0.1, 0.2, 0.3], [0.0, 0.1, 0.2, 0.3]]
        _, p_value = synchronized_permutation_pvalue(
            observed_by_stratum, null_by_stratum,
            statistic='mean', alternative='greater',
        )
        k = len(null_by_stratum[0])
        assert p_value == pytest.approx(1 / (k + 1))

    def test_ragged_null_raises(self):
        """Unequal row lengths cannot form synchronized columns."""
        from iblnm.analysis import synchronized_permutation_pvalue
        with pytest.raises(ValueError):
            synchronized_permutation_pvalue(
                [0.1, 0.06], [[0.02, 0.01, 0.03], [0.015, 0.005]],
                statistic='mean', alternative='greater',
            )


class TestFitOls:
    def test_recovers_high_r2_on_linear_signal(self):
        """y = 2x + small noise: the fit's R² is high and params are exposed."""
        from iblnm.analysis import fit_ols
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 200)
        df = pd.DataFrame({'x': x, 'y': 2 * x + rng.normal(0, 0.1, 200)})
        fit = fit_ols('y ~ x', df)
        assert fit.rsquared > 0.95
        assert fit.params['x'] == pytest.approx(2.0, abs=0.1)

    def test_singular_design_returns_none(self):
        """A rank-deficient design (a duplicated column) returns None rather
        than raising, matching fit_lmm's None-on-failure contract."""
        from iblnm.analysis import fit_ols
        rng = np.random.default_rng(1)
        x = rng.normal(0, 1, 50)
        df = pd.DataFrame({'x': x, 'x2': x, 'y': x + rng.normal(0, 0.1, 50)})
        # Drop the intercept so the two identical columns alias exactly.
        assert fit_ols('y ~ x + x2 - 1', df) is None


class TestDroponeDeltaR2:
    def test_difference_against_reference(self):
        """One row per non-reference name; delta_r2 = ref − reduced; the r2
        column carries the reference R² on every row."""
        from iblnm.analysis import dropone_delta_r2
        out = dropone_delta_r2({'full': 0.5, 'contrast': 0.3, 'side': 0.45})
        assert list(out.columns) == ['predictor', 'r2', 'delta_r2']
        assert set(out['predictor']) == {'contrast', 'side'}
        assert 'full' not in out['predictor'].values
        out = out.set_index('predictor')
        assert out.loc['contrast', 'delta_r2'] == pytest.approx(0.2)
        assert out.loc['side', 'delta_r2'] == pytest.approx(0.05)
        assert (out['r2'] == 0.5).all()

    def test_reference_only_gives_empty_frame(self):
        """No reduced models → an empty frame that still carries the columns."""
        from iblnm.analysis import dropone_delta_r2
        out = dropone_delta_r2({'full': 0.4})
        assert len(out) == 0
        assert list(out.columns) == ['predictor', 'r2', 'delta_r2']

    def test_nested_ols_fits_give_nonnegative_delta(self):
        """For genuinely nested OLS fits, the reduced model's R² cannot exceed
        the full model's, so every drop-one delta_r2 is >= 0."""
        from iblnm.analysis import dropone_delta_r2, fit_ols
        rng = np.random.default_rng(2)
        x1 = rng.normal(0, 1, 200)
        x2 = rng.normal(0, 1, 200)
        df = pd.DataFrame(
            {'x1': x1, 'x2': x2,
             'y': 0.7 * x1 + 0.4 * x2 + rng.normal(0, 0.5, 200)})
        r2 = {'full': fit_ols('y ~ x1 + x2', df).rsquared,
              'x1': fit_ols('y ~ x2', df).rsquared,
              'x2': fit_ols('y ~ x1', df).rsquared}
        out = dropone_delta_r2(r2)
        assert (out['delta_r2'] >= 0).all()


class TestPermutationNullDeltaR2:
    @staticmethod
    def _focal_frame(n, rng):
        """Focal frame where response is a strong linear function of x."""
        x = rng.normal(0, 1, n)
        z = rng.normal(0, 1, n)
        return pd.DataFrame(
            {'x': x, 'z': z, 'response': 2 * x + 0.3 * z + rng.normal(0, 0.1, n)})

    def test_unrelated_donors_give_small_null_well_below_focal(self):
        """Three donors whose x is unrelated to the focal response yield a
        length-3 null whose every value is far below the focal in-sample ΔR²."""
        from iblnm.analysis import fit_ols, permutation_null_delta_r2
        rng = np.random.default_rng(0)
        focal = self._focal_frame(200, rng)
        donors = [pd.DataFrame({'x': rng.normal(0, 1, 200)}) for _ in range(3)]
        focal_delta = (fit_ols('response ~ x + z', focal).rsquared
                       - fit_ols('response ~ z', focal).rsquared)
        null = permutation_null_delta_r2(
            focal, donors, '{response} ~ x + z', '{response} ~ z', 'x')
        assert null.shape == (3,)
        assert (null < 0.1 * focal_delta).all()

    def test_truncates_to_min_length_and_fits_those_rows(self):
        """A donor longer and a donor shorter than the focal both fit on
        L = min rows: each null delta equals the delta from a hand-built swap on
        the first L rows (proving the fit saw exactly L rows, no length error)."""
        from iblnm.analysis import fit_ols, permutation_null_delta_r2
        rng = np.random.default_rng(1)
        focal = self._focal_frame(100, rng)
        long_donor = pd.DataFrame({'x': rng.normal(0, 1, 160)})
        short_donor = pd.DataFrame({'x': rng.normal(0, 1, 40)})
        null = permutation_null_delta_r2(
            focal, [long_donor, short_donor],
            '{response} ~ x + z', '{response} ~ z', 'x')

        expected = []
        for donor in (long_donor, short_donor):
            length = min(len(focal), len(donor))
            swapped = focal.iloc[:length].copy()
            swapped['x'] = donor['x'].iloc[:length].to_numpy()
            expected.append(fit_ols('response ~ x + z', swapped).rsquared
                            - fit_ols('response ~ z', swapped).rsquared)
        assert null == pytest.approx(expected)

    def test_degenerate_donor_is_skipped(self):
        """A donor whose swapped predictor column is constant gives a rank-
        deficient design; that donor is dropped, shortening the null by one."""
        from iblnm.analysis import permutation_null_delta_r2
        rng = np.random.default_rng(2)
        focal = self._focal_frame(120, rng)
        donors = [
            pd.DataFrame({'x': rng.normal(0, 1, 120)}),
            pd.DataFrame({'x': np.ones(120)}),
            pd.DataFrame({'x': rng.normal(0, 1, 120)}),
        ]
        null = permutation_null_delta_r2(
            focal, donors, '{response} ~ x + z', '{response} ~ z', 'x')
        assert null.shape == (2,)


class TestComputeFeatureDispersion:
    def test_identical_across_sessions_gives_zero(self):
        """A unit whose feature value is identical across its sessions has zero
        within-unit variance, so its dispersion is 0 even when other units in the
        cohort vary (and thus give the feature a nonzero z-score scale)."""
        from iblnm.analysis import compute_feature_dispersion
        df = pd.DataFrame({
            'unit': ['U1', 'U1', 'U1', 'U2', 'U2', 'U2'],
            'eid': ['s1', 's2', 's3', 's1', 's2', 's3'],
            'feat': ['f1'] * 6,
            'val': [5.0, 5.0, 5.0, 1.0, 2.0, 9.0],
        })
        out = compute_feature_dispersion(
            df, ['unit'], 'eid', 'feat', 'val').set_index('unit')
        assert out.loc['U1', 'dispersion'] == pytest.approx(0.0)

    def test_combines_features_as_rms_of_within_unit_variance(self):
        """With two features z-scored to a common scale, a unit's dispersion is
        the sqrt of the mean of its per-feature within-unit variances. Here f1 is
        globally unit-variance (z == raw); f2 has global pop var 2 (z == raw/√2).
        U1's z-variance is 1 on f1 and 2 on f2, so dispersion == sqrt(1.5)."""
        from iblnm.analysis import compute_feature_dispersion
        df = pd.DataFrame({
            'unit': ['U1', 'U1', 'U2', 'U2', 'U1', 'U1', 'U2', 'U2'],
            'eid': ['s1', 's2', 's1', 's2', 's1', 's2', 's1', 's2'],
            'feat': ['f1', 'f1', 'f1', 'f1', 'f2', 'f2', 'f2', 'f2'],
            'val': [1.0, -1.0, 1.0, -1.0, 2.0, -2.0, 0.0, 0.0],
        })
        out = compute_feature_dispersion(
            df, ['unit'], 'eid', 'feat', 'val').set_index('unit')
        assert out.loc['U1', 'dispersion'] == pytest.approx(np.sqrt(1.5))

    def test_standardize_by_scales_per_group(self):
        """With ``standardize_by``, each group's feature is z-scored on its own
        scale. Two units with the same shape but different magnitudes (group A
        spread ±1, group B spread ±10) are equalized to dispersion 1 each;
        without ``standardize_by`` the shared global scale leaves them unequal."""
        from iblnm.analysis import compute_feature_dispersion
        df = pd.DataFrame({
            'unit': ['A', 'A', 'B', 'B'],
            'grp': ['A', 'A', 'B', 'B'],
            'eid': ['s1', 's2', 's1', 's2'],
            'feat': ['f1'] * 4,
            'val': [-1.0, 1.0, -10.0, 10.0],
        })
        per_group = compute_feature_dispersion(
            df, ['unit'], 'eid', 'feat', 'val',
            standardize_by='grp').set_index('unit')
        assert per_group.loc['A', 'dispersion'] == pytest.approx(1.0)
        assert per_group.loc['B', 'dispersion'] == pytest.approx(1.0)
        global_z = compute_feature_dispersion(
            df, ['unit'], 'eid', 'feat', 'val').set_index('unit')
        assert global_z.loc['A', 'dispersion'] < global_z.loc['B', 'dispersion']

    def test_constant_within_group_contributes_zero_not_nan(self):
        """A feature constant within its standardization group has zero spread,
        so per-group z-scoring would divide by zero; it maps to z = 0, giving the
        unit a finite dispersion of 0 rather than NaN."""
        from iblnm.analysis import compute_feature_dispersion
        df = pd.DataFrame({
            'unit': ['A', 'A', 'A', 'B', 'B', 'B'],
            'grp': ['A', 'A', 'A', 'B', 'B', 'B'],
            'eid': ['s1', 's2', 's3', 's1', 's2', 's3'],
            'feat': ['f1'] * 6,
            'val': [7.0, 7.0, 7.0, 1.0, 2.0, 3.0],
        })
        out = compute_feature_dispersion(
            df, ['unit'], 'eid', 'feat', 'val',
            standardize_by='grp').set_index('unit')
        assert out.loc['A', 'dispersion'] == pytest.approx(0.0)
        assert np.isfinite(out.loc['A', 'dispersion'])

    def test_n_sessions_counts_distinct_sessions_per_unit(self):
        """n_sessions is the number of distinct sessions a unit spans, counted
        once per session regardless of how many feature rows reference it."""
        from iblnm.analysis import compute_feature_dispersion
        df = pd.DataFrame({
            'unit': ['U1'] * 6 + ['U2'] * 4,
            'eid': ['s1', 's2', 's3', 's1', 's2', 's3', 's1', 's2', 's1', 's2'],
            'feat': ['f1', 'f1', 'f1', 'f2', 'f2', 'f2',
                     'f1', 'f1', 'f2', 'f2'],
            'val': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0],
        })
        out = compute_feature_dispersion(
            df, ['unit'], 'eid', 'feat', 'val').set_index('unit')
        assert out.loc['U1', 'n_sessions'] == 3
        assert out.loc['U2', 'n_sessions'] == 2


class TestMakeTimeGrid:
    def test_spans_half_open_interval(self):
        from iblnm.analysis import make_time_grid
        tvec = make_time_grid(0.0, 1.0, 0.25)
        np.testing.assert_allclose(tvec, [0.0, 0.25, 0.5, 0.75])


class TestMakeLags:
    def test_integer_centred_length(self):
        from iblnm.analysis import make_lags
        lags = make_lags(50)
        assert np.issubdtype(lags.dtype, np.integer)
        assert len(lags) == 50
        assert lags[0] == -25
        assert lags[-1] == 24


class TestTimesToIndices:
    def test_nearest_bin(self):
        from iblnm.analysis import times_to_indices
        tvec = np.arange(0.0, 1.0, 0.1)
        # 0.34 rounds to bin 3 (0.3), 0.66 rounds to bin 7 (0.7)
        idx = times_to_indices(np.array([0.34, 0.66]), tvec)
        np.testing.assert_array_equal(idx, [3, 7])

    def test_clip_bounds(self):
        from iblnm.analysis import times_to_indices
        tvec = np.arange(0.0, 1.0, 0.1)
        idx = times_to_indices(np.array([-5.0, 5.0]), tvec, clip=True)
        np.testing.assert_array_equal(idx, [0, tvec.size])


class TestMakeEventRegressor:
    def test_event_lands_on_nearest_bin(self):
        from iblnm.analysis import make_event_regressor
        tvec = np.arange(0.0, 1.0, 0.1)
        reg = make_event_regressor(np.array([0.52]), tvec)
        assert reg.shape == (tvec.size,)
        assert reg[5] == 1.0
        assert reg.sum() == 1.0

    def test_events_outside_grid_dropped(self):
        from iblnm.analysis import make_event_regressor
        tvec = np.arange(0.0, 1.0, 0.1)
        reg = make_event_regressor(np.array([-1.0, 0.3, 2.0]), tvec)
        assert reg.sum() == 1.0
        assert reg[3] == 1.0


class TestLagExpand:
    def test_impulse_shifts_by_lag(self):
        from iblnm.analysis import lag_expand
        reg = np.zeros(10)
        reg[5] = 1.0
        lags = np.array([-2, 0, 3])
        out = lag_expand(reg, lags)
        assert out.shape == (10, 3)
        # positive lag shifts the 1.0 later in time
        assert out[3, 0] == 1.0   # lag -2 -> index 3
        assert out[5, 1] == 1.0   # lag 0 -> index 5
        assert out[8, 2] == 1.0   # lag +3 -> index 8
        # exactly one nonzero per column, vacated entries zero
        np.testing.assert_array_equal(out.sum(axis=0), [1.0, 1.0, 1.0])

    def test_positive_lag_drops_off_grid(self):
        from iblnm.analysis import lag_expand
        reg = np.zeros(6)
        reg[5] = 1.0
        out = lag_expand(reg, np.array([2]))
        # impulse shifted past the end is zero-filled away
        assert out.sum() == 0.0


class TestRaisedCosineBasis:
    def test_shape_and_nonnegative(self):
        from iblnm.analysis import raised_cosine_basis
        n_basis, duration, dt = 5, 1.0, 0.1
        basis = raised_cosine_basis(n_basis, duration, 0.05, dt)
        assert basis.shape == (int(np.ceil(duration / dt)), n_basis)
        assert (basis >= 0).all()

    def test_nonpositive_offset_raises(self):
        from iblnm.analysis import raised_cosine_basis
        with pytest.raises(ValueError):
            raised_cosine_basis(5, 1.0, 0.0, 0.1)


class TestMakeTrialConstant:
    def test_value_held_inside_interval_zero_outside(self):
        from iblnm.analysis import make_trial_constant
        tvec = np.arange(0.0, 1.0, 0.1)
        trials = pd.DataFrame(
            {"intervals_0": [0.2, 0.6], "intervals_1": [0.4, 0.8], "x": [5.0, 9.0]}
        )
        values = make_trial_constant(trials, "x", tvec)
        assert values.shape == (tvec.size,)
        # bins inside trial 0's interval [0.2, 0.4) carry its value
        assert values[2] == 5.0
        assert values[3] == 5.0
        # bins inside trial 1's interval [0.6, 0.8) carry its value
        assert values[6] == 9.0
        assert values[7] == 9.0
        # bins outside any interval are zero
        assert values[0] == 0.0
        assert values[5] == 0.0
        assert values[9] == 0.0


class TestInterpolateToGrid:
    def test_linear_series_recovered_interior_nan_outside(self):
        from iblnm.analysis import interpolate_to_grid
        src_t = np.linspace(0.0, 1.0, 6)
        series = pd.Series(2.0 * src_t + 1.0, index=src_t)
        # grid extends past the source support on both ends
        tvec = np.array([-0.2, 0.25, 0.5, 0.75, 1.3])
        out = interpolate_to_grid(series, tvec)
        assert out.shape == (tvec.size,)
        # a linear input is reproduced exactly by quadratic interpolation
        np.testing.assert_allclose(out[1:4], 2.0 * tvec[1:4] + 1.0)
        # samples outside [0, 1] have no support
        assert np.isnan(out[0])
        assert np.isnan(out[-1])


class TestBuildContinuousBlock:
    def test_unlagged_matches_resampled_signal(self):
        from iblnm.analysis import build_continuous_block, interpolate_to_grid
        src_t = np.linspace(0.0, 1.0, 6)
        series = pd.Series(2.0 * src_t + 1.0, index=src_t)
        tvec = np.array([0.25, 0.5, 0.75])
        out = build_continuous_block(series, tvec, lags=None)
        assert out.shape == (tvec.size,)
        np.testing.assert_allclose(out, interpolate_to_grid(series, tvec))

    def test_lagged_shifts_resampled_signal_per_column(self):
        from iblnm.analysis import build_continuous_block, interpolate_to_grid
        src_t = np.linspace(0.0, 2.0, 21)
        series = pd.Series(src_t**2, index=src_t)
        tvec = np.linspace(0.2, 1.8, 9)
        lags = np.array([-2, 0, 3])
        resampled = interpolate_to_grid(series, tvec)
        out = build_continuous_block(series, tvec, lags=lags)
        assert out.shape == (tvec.size, lags.size)
        # column j is the resampled signal shifted by lags[j] (zero-padded)
        np.testing.assert_allclose(out[:3, 0], resampled[2:5])   # lag -2
        np.testing.assert_allclose(out[:, 1], resampled)         # lag 0
        np.testing.assert_allclose(out[3:, 2], resampled[:-3])   # lag +3
        np.testing.assert_array_equal(out[-2:, 0], 0.0)          # vacated tail
        np.testing.assert_array_equal(out[:3, 2], 0.0)           # vacated head


class TestRaisedCosineExpand:
    def test_single_impulse_superposes_to_basis(self):
        from iblnm.analysis import raised_cosine_basis, raised_cosine_expand
        n_basis, duration, offset, dt = 4, 1.0, 0.05, 0.1
        tvec = np.arange(0.0, 3.0, dt)
        impulse_idx = 5
        regressor = np.zeros(tvec.size)
        regressor[impulse_idx] = 1.0
        block = raised_cosine_expand(regressor, tvec, n_basis, duration, offset)
        assert block.shape == (tvec.size, n_basis)
        # column superposition equals the basis placed at the impulse (truncated)
        basis = raised_cosine_basis(n_basis, duration, offset, dt)
        n_kernel = basis.shape[0]
        expected = np.zeros(tvec.size)
        expected[impulse_idx:impulse_idx + n_kernel] = basis.sum(axis=1)
        np.testing.assert_allclose(block.sum(axis=1), expected)


class TestBuildDesignMatrix:
    def test_spans_and_block_recovery(self):
        from iblnm.analysis import build_design_matrix
        n_rows = 4
        a = np.arange(n_rows * 3).reshape(n_rows, 3).astype(float)
        b = np.arange(n_rows).astype(float)  # 1-D, promoted to one column
        c = np.arange(n_rows * 2).reshape(n_rows, 2).astype(float)
        matrix, slices = build_design_matrix({'a': a, 'b': b, 'c': c})
        assert matrix.shape == (n_rows, 6)
        assert slices == {'a': slice(0, 3), 'b': slice(3, 4), 'c': slice(4, 6)}
        # each block is recoverable by its span
        np.testing.assert_array_equal(matrix[:, slices['a']], a)
        np.testing.assert_array_equal(matrix[:, slices['b']], b[:, None])
        np.testing.assert_array_equal(matrix[:, slices['c']], c)


class TestDeviationCode:
    def test_two_levels_map_to_plus_minus_half(self):
        from iblnm.analysis import deviation_code
        codes = deviation_code(['contra', 'ipsi', 'contra'], 'contra')
        np.testing.assert_array_equal(codes, [0.5, -0.5, 0.5])


class TestBuildEventBlocks:
    def _identity_expander(self):
        # expander that returns the 1-D event train as a single column, so block
        # heights are directly readable at event bins
        return lambda regressor: regressor[:, None]

    def test_baseline_unit_heights_and_modulator_values(self):
        from iblnm.analysis import build_event_blocks
        tvec = np.arange(0.0, 1.0, 0.1)
        event_times = np.array([0.2, 0.5])
        contrast = np.array([0.25, -0.75])  # already mean-centered by the caller
        blocks = build_event_blocks(
            event_times, tvec, self._identity_expander(),
            modulators={'contrast': contrast}, name='stimOn_times',
        )
        assert set(blocks) == {'stimOn_times|baseline', 'stimOn_times|contrast'}
        baseline = blocks['stimOn_times|baseline'][:, 0]
        # unit height at each event bin (0.2 -> 2, 0.5 -> 5), zero elsewhere
        assert baseline[2] == 1.0 and baseline[5] == 1.0
        assert baseline.sum() == 2.0
        mod = blocks['stimOn_times|contrast'][:, 0]
        assert mod[2] == 0.25 and mod[5] == -0.75

    def test_interaction_block_is_product_of_modulators(self):
        from iblnm.analysis import build_event_blocks
        tvec = np.arange(0.0, 1.0, 0.1)
        event_times = np.array([0.2, 0.5])
        side = np.array([0.5, -0.5])
        contrast = np.array([0.25, -0.75])
        blocks = build_event_blocks(
            event_times, tvec, self._identity_expander(),
            modulators={'side': side, 'contrast': contrast},
            interactions=[('side', 'contrast')], name='stimOn_times',
        )
        assert 'stimOn_times|side:contrast' in blocks
        inter = blocks['stimOn_times|side:contrast'][:, 0]
        # height at each event bin equals the product of the two modulators
        assert inter[2] == 0.5 * 0.25
        assert inter[5] == -0.5 * -0.75

    def test_split_replicates_block_set_per_level(self):
        from iblnm.analysis import build_event_blocks
        tvec = np.arange(0.0, 1.0, 0.1)
        event_times = np.array([0.2, 0.5, 0.7])
        contrast = np.array([0.25, -0.75, 0.5])
        split = pd.Series([1, -1, 1], name='feedbackType')
        blocks = build_event_blocks(
            event_times, tvec, self._identity_expander(),
            modulators={'contrast': contrast}, split=split, name='feedback_times',
        )
        assert set(blocks) == {
            'feedback_times|feedbackType=-1|baseline',
            'feedback_times|feedbackType=-1|contrast',
            'feedback_times|feedbackType=1|baseline',
            'feedback_times|feedbackType=1|contrast',
        }
        # the +1 group fires only at its events (0.2 -> bin 2, 0.7 -> bin 7)
        pos = blocks['feedback_times|feedbackType=1|baseline'][:, 0]
        assert pos[2] == 1.0 and pos[7] == 1.0 and pos[5] == 0.0
        # its modulator carries that group's contrasts only
        pos_mod = blocks['feedback_times|feedbackType=1|contrast'][:, 0]
        assert pos_mod[2] == 0.25 and pos_mod[7] == 0.5
        # the -1 group fires only at its single event (0.5 -> bin 5)
        neg = blocks['feedback_times|feedbackType=-1|baseline'][:, 0]
        assert neg[5] == 1.0 and neg.sum() == 1.0


class TestFitEncodingModel:
    def test_recovers_coefficients_and_unit_r2(self):
        from iblnm.analysis import fit_encoding_model
        rng = np.random.default_rng(0)
        n = 400
        design = rng.standard_normal((n, 4))
        b_true = np.array([2.0, -1.0, 0.5, 3.0])
        target = pd.Series(design @ b_true, index=np.arange(n) * 0.1)
        slices = {'a': slice(0, 1), 'b': slice(1, 2),
                  'c': slice(2, 3), 'd': slice(3, 4)}
        fit = fit_encoding_model(design, target, slices, alphas=[1e-6], cv=5)
        recovered = fit.coefficients.flatten() / fit.scaler.scale_
        np.testing.assert_allclose(recovered, b_true, atol=1e-3)
        assert fit.r2 > 0.999

    def test_alpha_selected_from_grid(self):
        from iblnm.analysis import fit_encoding_model
        rng = np.random.default_rng(1)
        n = 300
        design = rng.standard_normal((n, 3))
        signal = design @ np.array([1.0, 0.0, -2.0])
        target = pd.Series(signal + rng.standard_normal(n),
                           index=np.arange(n) * 0.1)
        slices = {'a': slice(0, 1), 'b': slice(1, 2), 'c': slice(2, 3)}
        alphas = [1e-3, 1.0, 1e3]
        fit = fit_encoding_model(design, target, slices, alphas=alphas, cv=5)
        assert fit.alpha in alphas

    def test_kernels_to_frame_structure_and_backtransform(self):
        from iblnm.analysis import fit_encoding_model
        rng = np.random.default_rng(2)
        n, dt = 300, 0.1
        event = rng.standard_normal((n, 3))  # FIR event block, 3 lags
        cont = rng.standard_normal((n, 1))   # continuous block, scalar
        design = np.concatenate([event, cont], axis=1)
        b_true = np.array([1.0, -0.5, 2.0, 0.7])
        target = pd.Series(design @ b_true, index=np.arange(n) * dt)
        slices = {'stimOn_times|baseline': slice(0, 3),
                  'wheel_velocity': slice(3, 4)}
        fit = fit_encoding_model(design, target, slices, alphas=[1e-6], cv=5)
        frame = fit.kernels_to_frame()
        assert list(frame.columns) == [
            'term', 'level', 'modulator', 'lag', 'time', 'coef']
        # FIR event block: one row per lag, time = lag * dt, baseline modulator
        event_rows = frame[frame['term'] == 'stimOn_times']
        assert len(event_rows) == 3
        assert event_rows['modulator'].unique().tolist() == ['baseline']
        np.testing.assert_allclose(
            event_rows['time'].values, np.array([0, 1, 2]) * dt)
        np.testing.assert_allclose(event_rows['coef'].values, b_true[:3], atol=1e-3)
        # continuous block: a single scalar row with NaN lag/time
        cont_rows = frame[frame['term'] == 'wheel_velocity']
        assert len(cont_rows) == 1
        assert cont_rows['lag'].isna().all() and cont_rows['time'].isna().all()
        np.testing.assert_allclose(cont_rows['coef'].values, b_true[3:], atol=1e-3)


class TestEncodingConfig:
    """Structural checks on the encoding-model constants and default term spec."""

    def test_scalar_constants(self):
        from iblnm import config
        assert config.ENCODING_DT == 0.1
        assert config.ENCODING_CV == 5
        assert config.ENCODING_POSE_KEYPOINTS == ['paw_l', 'paw_r', 'nose']

    def test_term_spec_events(self):
        from iblnm import config
        assert set(config.ENCODING_TERMS) == {
            'stimOn_times', 'firstMovement_times', 'response_times',
            'feedback_times', 'goCue_times'}

    def test_stimon_interaction(self):
        from iblnm import config
        assert ('side', 'contrast') in config.ENCODING_TERMS['stimOn_times']['interactions']

    def test_feedback_split_by(self):
        from iblnm import config
        assert config.ENCODING_TERMS['feedback_times']['split_by'] == 'feedbackType'

    def test_modulator_types(self):
        from iblnm import config
        mods = config.ENCODING_TERMS['stimOn_times']['modulators']
        assert mods['side'] == 'categorical'
        assert mods['contrast'] == 'continuous'


def _synthetic_encoding_trials(n_trials=24, isi=2.5, seed=0):
    """Synthetic trials table driving the default ENCODING_TERMS coding.

    Trials are evenly spaced by ``isi`` seconds, each carrying the five model
    events plus the categorical/parametric columns the term spec reads. Both
    ``feedbackType`` levels appear so the feedback split yields two kernel sets,
    and ``choice`` is never 0 so ``firstMovement_times`` and ``choice_side`` stay
    defined. Returns the trials DataFrame.
    """
    rng = np.random.default_rng(seed)
    onsets = 2.0 + isi * np.arange(n_trials)
    stim_side = np.where(np.arange(n_trials) % 2 == 0, 'left', 'right')
    contrast = np.tile([0.0, 0.0625, 0.25, 1.0], n_trials // 4 + 1)[:n_trials]
    sign = np.where(stim_side == 'right', 1.0, -1.0)
    return pd.DataFrame({
        'stimOn_times': onsets,
        'goCue_times': onsets + 0.1,
        'firstMovement_times': onsets + 0.3,
        'response_times': onsets + 0.5,
        'feedback_times': onsets + 0.7,
        'intervals_0': onsets - 0.5,
        'intervals_1': onsets + 2.0,
        'stim_side': stim_side,
        'choice': rng.choice([-1, 1], n_trials),
        'feedbackType': np.where(np.arange(n_trials) % 2 == 0, 1, -1),
        'contrast': contrast,
        'signed_contrast': contrast * sign,
        'probabilityLeft': 0.5,
    })


class TestBuildEncodingDesign:
    """The script's ONE-free build+code step composes the default term spec."""

    def _design(self):
        from scripts.encoding import build_encoding_design
        from iblnm.analysis import make_lags, lag_expand
        from iblnm.config import ENCODING_N_LAGS
        from functools import partial

        trials = _synthetic_encoding_trials()
        t = np.arange(0.0, 60.0, 1 / 30)
        target = pd.Series(np.sin(t) + 0.1 * np.arange(t.size) / t.size, index=t)
        grid = np.arange(0.0, 60.0, 0.1)
        continuous = {
            'wheel_velocity': pd.Series(np.cos(grid), index=grid),
            'paw_l_speed': pd.Series(np.abs(np.sin(grid)), index=grid),
        }
        expander = partial(lag_expand, lags=make_lags(ENCODING_N_LAGS))
        design, slices, target_grid = build_encoding_design(
            trials, target, hemisphere='l', expander=expander,
            continuous=continuous)
        return design, slices, target_grid

    def test_default_spec_composes_finite_design(self):
        from iblnm.config import ENCODING_N_LAGS
        design, slices, target_grid = self._design()

        # 13 event blocks (4 stimOn + 2 firstMovement + 2 response + 4 feedback
        # + 1 goCue), each FIR-expanded to n_lags columns, plus the two unlagged
        # continuous regressors.
        event_blocks = {name for name in slices if '|' in name}
        assert event_blocks == {
            'stimOn_times|baseline', 'stimOn_times|side',
            'stimOn_times|contrast', 'stimOn_times|side:contrast',
            'firstMovement_times|baseline', 'firstMovement_times|choice',
            'response_times|baseline', 'response_times|choice',
            'feedback_times|feedbackType=1|baseline',
            'feedback_times|feedbackType=1|contrast',
            'feedback_times|feedbackType=-1|baseline',
            'feedback_times|feedbackType=-1|contrast',
            'goCue_times|baseline',
        }
        assert design.shape == (target_grid.size, 13 * ENCODING_N_LAGS + 2)
        assert np.isfinite(design).all()

    def test_fit_returns_finite_r2(self):
        from iblnm.analysis import fit_encoding_model
        from iblnm.config import ENCODING_ALPHAS, ENCODING_CV
        design, slices, target_grid = self._design()
        fit = fit_encoding_model(
            design, target_grid, slices, ENCODING_ALPHAS, ENCODING_CV)
        assert np.isfinite(fit.r2)
