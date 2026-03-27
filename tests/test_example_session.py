"""Tests for scripts/example_session.py helper functions."""
import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from scripts.example_session import find_snippet_window, plot_example_snippet


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


@pytest.fixture
def snippet_data():
    """Synthetic signals for a 60s window starting at t=100."""
    rng = np.random.default_rng(42)
    t_start, t_end = 100.0, 160.0

    # Photometry at 30 Hz
    t_phot = np.arange(0, 300, 1 / 30)
    photometry = pd.Series(rng.normal(0, 1, len(t_phot)), index=t_phot)

    # Wheel at 100 Hz
    t_wheel = np.arange(0, 300, 1 / 100)
    wheel = pd.Series(rng.normal(0, 0.5, len(t_wheel)), index=t_wheel)

    # Pose at 60 Hz
    n_frames = int(300 * 60)
    pose_times = np.linspace(0, 300, n_frames)
    pose = {
        'nose_tip': pd.DataFrame({
            'x': rng.normal(200, 10, n_frames),
            'y': rng.normal(150, 10, n_frames),
            'likelihood': rng.uniform(0.8, 1.0, n_frames),
        }),
        'paw_l': pd.DataFrame({
            'x': rng.normal(180, 15, n_frames),
            'y': rng.normal(300, 20, n_frames),
            'likelihood': rng.uniform(0.5, 1.0, n_frames),
        }),
    }

    # Trials in the window
    stim = np.array([105, 112, 120, 128, 135, 142, 150, 155], dtype=float)
    fb = stim + 2.0
    trials = pd.DataFrame({
        'stimOn_times': stim,
        'feedback_times': fb,
        'feedbackType': [1, -1, 1, 1, -1, 1, 1, -1],
        'choice': [1, -1, 1, 1, -1, 1, 1, -1],
    })

    return photometry, wheel, pose_times, pose, trials, t_start, t_end


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
# plot_example_snippet
# =========================================================================

class TestPlotExampleSnippet:
    def test_returns_figure(self, snippet_data):
        phot, wheel, pt, pose, trials, t0, t1 = snippet_data
        fig = plot_example_snippet(
            phot, wheel, pt, pose, trials, t0, t1,
            brain_region='VTA', target_nm='VTA-DA',
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_always_three_panels(self, snippet_data):
        """Pose traces share a single panel regardless of body part count."""
        phot, wheel, pt, pose, trials, t0, t1 = snippet_data
        fig = plot_example_snippet(
            phot, wheel, pt, pose, trials, t0, t1,
            brain_region='VTA', target_nm='VTA-DA',
        )
        assert len(fig.axes) == 3
        plt.close(fig)

    def test_three_panels_with_subset(self, snippet_data):
        phot, wheel, pt, pose, trials, t0, t1 = snippet_data
        fig = plot_example_snippet(
            phot, wheel, pt, pose, trials, t0, t1,
            brain_region='VTA', target_nm='VTA-DA',
            body_parts=['nose_tip'],
        )
        assert len(fig.axes) == 3
        plt.close(fig)

    def test_axis_frames_removed(self, snippet_data):
        """Top and right spines should be hidden on all axes."""
        phot, wheel, pt, pose, trials, t0, t1 = snippet_data
        fig = plot_example_snippet(
            phot, wheel, pt, pose, trials, t0, t1,
            brain_region='VTA', target_nm='VTA-DA',
        )
        for ax in fig.axes:
            assert not ax.spines['top'].get_visible()
            assert not ax.spines['right'].get_visible()
        plt.close(fig)

    def test_pose_traces_offset(self, snippet_data):
        """Pose body parts should be vertically offset in the single panel."""
        phot, wheel, pt, pose, trials, t0, t1 = snippet_data
        fig = plot_example_snippet(
            phot, wheel, pt, pose, trials, t0, t1,
            brain_region='VTA', target_nm='VTA-DA',
        )
        ax_pose = fig.axes[2]
        lines = ax_pose.get_lines()
        # With 2 body parts we expect 2 data lines plus event lines.
        # The data lines should have different y-ranges (offset).
        data_lines = [l for l in lines
                      if len(l.get_ydata()) > 10]
        if len(data_lines) >= 2:
            means = [np.nanmean(l.get_ydata()) for l in data_lines[:2]]
            assert abs(means[0] - means[1]) > 1.0
        plt.close(fig)
