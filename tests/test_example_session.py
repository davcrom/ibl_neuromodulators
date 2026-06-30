"""Tests for scripts/example_session.py helper functions."""
import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba

from iblnm.config import TARGETNM_COLORS
from scripts.example_session import (
    camera_timing_ok, find_snippet_window, _normalize_window, build_traces,
    plot_example_session, contrast_rank_grays, feedback_colors,
)


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def trials():
    """Synthetic trials with known event timing."""
    rng = np.random.default_rng(0)
    n = 40
    stim_times = np.cumsum(rng.uniform(3, 8, n)) + 100.0
    fb_times = stim_times + rng.uniform(0.5, 3.0, n)
    return pd.DataFrame({
        'stimOn_times': stim_times,
        'feedback_times': fb_times,
        'feedbackType': rng.choice([1, -1], n),
        'choice': rng.choice([-1, 1], n),
    })


POSE_KEYPOINTS = ['paw_l', 'paw_r', 'nose_tip', 'tongue_end_l', 'tongue_end_r']


@pytest.fixture
def snippet_data():
    """Synthetic signals for a 60s window starting at t=100.

    Pose is a flat (constant-position) DataFrame with the five keypoints the
    figure plots, so keypoint speeds are a finite constant and likelihoods are
    well-defined. Photometry and wheel are time-indexed Series; trials carry
    in-window stimulus and feedback events.
    """
    rng = np.random.default_rng(42)
    t_start, t_end = 100.0, 160.0

    # Photometry at 30 Hz
    t_phot = np.arange(0, 300, 1 / 30)
    photometry = pd.Series(rng.normal(0, 1, len(t_phot)), index=t_phot)

    # Wheel at 100 Hz
    t_wheel = np.arange(0, 300, 1 / 100)
    wheel = pd.Series(rng.normal(0, 0.5, len(t_wheel)), index=t_wheel)

    # Pose at 30 Hz: flat positions, high likelihood
    n_frames = int(300 * 30)
    pose_times = np.linspace(0, 300, n_frames)
    pose_df = pd.DataFrame({
        col: np.full(n_frames, fill)
        for kp, (xfill, yfill) in zip(
            POSE_KEYPOINTS,
            [(200, 150), (180, 160), (100, 120), (140, 200), (145, 205)])
        for col, fill in [(f'{kp}_x', xfill), (f'{kp}_y', yfill),
                          (f'{kp}_likelihood', 0.95)]
    })

    # Trials in the window
    stim = np.array([105, 112, 120, 128, 135, 142, 150, 155], dtype=float)
    fb = stim + 2.0
    trials = pd.DataFrame({
        'stimOn_times': stim,
        'feedback_times': fb,
        'feedbackType': [1, -1, 1, 1, -1, 1, 1, -1],
        'choice': [1, -1, 1, 1, -1, 1, 1, -1],
        'contrast': [1.0, 0.25, 0.0625, 0.0, 1.0, 0.125, 0.25, 0.0],
    })

    return photometry, wheel, pose_df, pose_times, trials, t_start, t_end


# =========================================================================
# camera_timing_ok
# =========================================================================

class TestCameraTimingOk:
    def test_passes_when_one_third_exceeds_threshold(self):
        functions = np.array([
            [0.1, 0.7, 0.2],   # this third peaks at 0.7
            [0.1, 0.1, 0.05],
            [0.0, 0.1, 0.1],
        ])
        assert camera_timing_ok({'functions': functions}) is True

    def test_fails_when_all_thirds_at_or_below_threshold(self):
        functions = np.array([
            [0.1, 0.5, 0.2],
            [0.3, 0.4, 0.1],
            [0.0, 0.5, 0.1],
        ])
        assert camera_timing_ok({'functions': functions}) is False

    def test_all_nan_third_does_not_raise_or_pass(self):
        functions = np.array([
            [np.nan, np.nan, np.nan],  # quiet/dropped third
            [0.1, 0.2, 0.3],
            [0.0, 0.1, 0.1],
        ])
        assert camera_timing_ok({'functions': functions}) is False


# =========================================================================
# find_snippet_window
# =========================================================================

class TestFindSnippetWindow:
    def test_returns_correct_duration(self, trials):
        t_start, t_end = find_snippet_window(trials, duration=60)
        assert t_end - t_start == pytest.approx(60.0)

    def test_window_within_session_bounds(self, trials):
        t_start, t_end = find_snippet_window(trials, duration=30)
        session_start = trials['stimOn_times'].min()
        session_end = trials['stimOn_times'].max() + 30
        assert t_start >= session_start - 1
        assert t_end <= session_end + 31

    def test_finds_window_with_enough_trials(self, trials):
        t_start, t_end = find_snippet_window(trials, duration=60, min_trials=3)
        mask = (
            (trials['stimOn_times'] >= t_start)
            & (trials['stimOn_times'] <= t_end)
        )
        assert mask.sum() >= 3

    def test_raises_on_insufficient_trials(self):
        tiny = pd.DataFrame({
            'stimOn_times': [1.0, 2.0],
            'feedback_times': [1.5, 2.5],
        })
        with pytest.raises(ValueError, match="Too few trials"):
            find_snippet_window(tiny, min_trials=10)


# =========================================================================
# _normalize_window
# =========================================================================

class TestNormalizeWindow:
    def test_maps_nonconstant_to_unit_range(self):
        t = np.arange(0, 10, dtype=float)
        values = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20], dtype=float)
        out = _normalize_window(values, t, 2.0, 7.0)
        assert np.nanmin(out) == pytest.approx(0.0)
        assert np.nanmax(out) == pytest.approx(1.0)

    def test_constant_window_returns_zeros(self):
        t = np.arange(0, 10, dtype=float)
        values = np.full(10, 5.0)
        out = _normalize_window(values, t, 2.0, 7.0)
        assert np.all(out == 0.0)

    def test_slices_to_window(self):
        t = np.arange(0, 10, dtype=float)
        values = np.arange(0, 10, dtype=float)
        out = _normalize_window(values, t, 3.0, 6.0)
        assert len(out) == 4  # t == 3, 4, 5, 6


# =========================================================================
# build_traces
# =========================================================================

class TestBuildTraces:
    def test_six_traces_in_order_with_labels(self, snippet_data):
        phot, wheel, pose_df, pose_times, _, _, _ = snippet_data
        traces = build_traces(phot, wheel, pose_df, pose_times, 'VTA-DA')
        assert len(traces) == 6
        assert [t['label'] for t in traces] == [
            'Photometry', 'Wheel', 'Left paw', 'Right paw', 'Nose', 'Tongue']

    def test_photometry_keeps_target_color_rest_unique(self, snippet_data):
        phot, wheel, pose_df, pose_times, _, _, _ = snippet_data
        traces = build_traces(phot, wheel, pose_df, pose_times, 'VTA-DA')
        assert traces[0]['color'] == TARGETNM_COLORS['VTA-DA']
        movement_colors = [t['color'] for t in traces[1:]]
        assert movement_colors == list(plt.cm.Set1.colors[1:6])
        assert len({to_rgba(t['color']) for t in traces}) == 6

    def test_pose_trace_lengths_match_pose_times(self, snippet_data):
        phot, wheel, pose_df, pose_times, _, _, _ = snippet_data
        traces = build_traces(phot, wheel, pose_df, pose_times, 'VTA-DA')
        for trace in traces[2:]:
            assert len(trace['values']) == len(pose_times)
            assert len(trace['times']) == len(pose_times)

    def test_tongue_trace_is_likelihood_in_unit_range(self, snippet_data):
        phot, wheel, pose_df, pose_times, _, _, _ = snippet_data
        traces = build_traces(phot, wheel, pose_df, pose_times, 'VTA-DA')
        tongue = traces[5]['values']
        assert np.nanmin(tongue) >= 0.0
        assert np.nanmax(tongue) <= 1.0


# =========================================================================
# contrast_rank_grays / feedback_colors
# =========================================================================

class TestMarkers:
    def test_contrast_rank_grays_monotonic_in_rank(self):
        levels = [0.0, 0.0625, 0.25, 1.0]
        out = contrast_rank_grays([0.0, 1.0, 0.25, 0.0625], levels)
        # lowest contrast (rank 0) -> white, highest -> black
        assert out[0] == pytest.approx(1.0)   # contrast 0.0
        assert out[1] == pytest.approx(0.0)   # contrast 1.0
        # order follows rank, not raw value: 0.0625 lighter than 0.25
        assert out[3] > out[2]

    def test_contrast_rank_grays_single_level(self):
        out = contrast_rank_grays([0.0, 0.0, 0.0], levels=[0.0])
        assert np.all(out == 0.0)

    def test_feedback_colors(self):
        assert feedback_colors([1, -1, 1]) == ['green', 'red', 'green']


# =========================================================================
# plot_example_session
# =========================================================================

class TestPlotExampleSession:
    def _traces(self, snippet_data):
        phot, wheel, pose_df, pose_times, _, _, _ = snippet_data
        return build_traces(phot, wheel, pose_df, pose_times, 'VTA-DA')

    def test_returns_figure_single_axes(self, snippet_data):
        *_, trials, t0, t1 = snippet_data
        fig = plot_example_session(self._traces(snippet_data), trials, t0, t1)
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_axes_frameless(self, snippet_data):
        *_, trials, t0, t1 = snippet_data
        fig = plot_example_session(self._traces(snippet_data), trials, t0, t1)
        ax = fig.axes[0]
        assert all(not sp.get_visible() for sp in ax.spines.values())
        assert len(ax.get_xticklabels()) == 0 or all(
            lbl.get_text() == '' for lbl in ax.get_xticklabels())
        assert len(ax.get_yticklabels()) == 0 or all(
            lbl.get_text() == '' for lbl in ax.get_yticklabels())
        plt.close(fig)

    def test_each_trace_labeled(self, snippet_data):
        *_, trials, t0, t1 = snippet_data
        traces = self._traces(snippet_data)
        fig = plot_example_session(traces, trials, t0, t1)
        ax = fig.axes[0]
        texts = {t.get_text() for t in ax.texts}
        assert {tr['label'] for tr in traces} <= texts
        plt.close(fig)

    def test_six_data_lines_non_overlapping(self, snippet_data):
        *_, trials, t0, t1 = snippet_data
        fig = plot_example_session(self._traces(snippet_data), trials, t0, t1)
        ax = fig.axes[0]
        data_lines = [ln for ln in ax.get_lines() if len(ln.get_ydata()) > 10]
        assert len(data_lines) == 6
        baselines = sorted(np.nanmin(ln.get_ydata()) for ln in data_lines)
        gaps = np.diff(baselines)
        assert np.all(gaps >= 1.0)  # unit-height bands do not overlap
        plt.close(fig)

    def test_photometry_line_color(self, snippet_data):
        *_, trials, t0, t1 = snippet_data
        fig = plot_example_session(self._traces(snippet_data), trials, t0, t1)
        ax = fig.axes[0]
        data_lines = [ln for ln in ax.get_lines() if len(ln.get_ydata()) > 10]
        colors = {ln.get_color() for ln in data_lines}
        assert TARGETNM_COLORS['VTA-DA'] in colors
        plt.close(fig)

    def test_event_line_per_in_window_event(self, snippet_data):
        *_, trials, t0, t1 = snippet_data
        n_stim = int(trials['stimOn_times'].between(t0, t1).sum())
        n_fb = int(trials['feedback_times'].between(t0, t1).sum())
        fig = plot_example_session(self._traces(snippet_data), trials, t0, t1)
        ax = fig.axes[0]
        event_lines = [ln for ln in ax.get_lines() if len(ln.get_ydata()) == 2]
        assert len(event_lines) == n_stim + n_fb
        plt.close(fig)

    def _split_marker_collections(self, ax):
        """Return (stim_scatter, feedback_scatter) by facecolor signature."""
        green_red = {to_rgba('green'), to_rgba('red')}
        stim, feedback = None, None
        for col in ax.collections:
            facecolors = col.get_facecolors()
            if len(facecolors) and all(
                    tuple(fc) in green_red for fc in facecolors):
                feedback = col
            else:
                stim = col
        return stim, feedback

    def test_feedback_circles_match_feedbacktype(self, snippet_data):
        *_, trials, t0, t1 = snippet_data
        fig = plot_example_session(self._traces(snippet_data), trials, t0, t1)
        _, feedback = self._split_marker_collections(fig.axes[0])
        in_window = trials['feedback_times'].between(t0, t1)
        expected = [to_rgba(c)
                    for c in feedback_colors(trials.loc[in_window, 'feedbackType'])]
        assert [tuple(fc) for fc in feedback.get_facecolors()] == expected
        plt.close(fig)

    def test_stim_circles_have_black_edges(self, snippet_data):
        *_, trials, t0, t1 = snippet_data
        fig = plot_example_session(self._traces(snippet_data), trials, t0, t1)
        stim, _ = self._split_marker_collections(fig.axes[0])
        assert all(tuple(e) == to_rgba('black')
                   for e in stim.get_edgecolors())
        plt.close(fig)

    def test_markers_share_y_above_photometry_band(self, snippet_data):
        *_, trials, t0, t1 = snippet_data
        fig = plot_example_session(self._traces(snippet_data), trials, t0, t1)
        ax = fig.axes[0]
        stim, feedback = self._split_marker_collections(ax)
        marker_ys = np.concatenate([
            stim.get_offsets()[:, 1], feedback.get_offsets()[:, 1]])
        assert np.allclose(marker_ys, marker_ys[0])
        data_lines = [ln for ln in ax.get_lines() if len(ln.get_ydata()) > 10]
        top_band_top = max(np.nanmax(ln.get_ydata()) for ln in data_lines)
        assert marker_ys[0] > top_band_top
        plt.close(fig)
