"""Tests for iblnm.vis module."""
import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from iblnm.vis import plot_relative_contrast


@pytest.fixture
def df_group():
    rng = np.random.default_rng(0)
    n = 60
    return pd.DataFrame({
        'subject': rng.choice(['s1', 's2', 's3'], n),
        'side': rng.choice(['contra', 'ipsi'], n),
        'contrast': rng.choice([100.0, 25.0, 12.5, 0.0], n),
        'feedbackType': rng.choice([1, -1], n),
        'centered_mean': rng.normal(0, 0.01, n),
    })


class TestPlotRelativeContrast:
    def test_returns_figure(self, df_group):
        fig = plot_relative_contrast(df_group, 'centered_mean', 'VTA-DA', 'stimOn_times')
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_has_two_axes(self, df_group):
        fig = plot_relative_contrast(df_group, 'centered_mean', 'VTA-DA', 'stimOn_times')
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_contra_axis_inverted(self, df_group):
        """Contra (left) panel x-axis is inverted: xlim[0] > xlim[1]."""
        fig = plot_relative_contrast(df_group, 'centered_mean', 'VTA-DA', 'stimOn_times')
        ax_contra = fig.axes[0]
        xlim = ax_contra.get_xlim()
        assert xlim[0] > xlim[1], "Contra x-axis should be inverted"
        plt.close(fig)

    def test_ipsi_axis_not_inverted(self, df_group):
        fig = plot_relative_contrast(df_group, 'centered_mean', 'VTA-DA', 'stimOn_times')
        ax_ipsi = fig.axes[1]
        xlim = ax_ipsi.get_xlim()
        assert xlim[0] < xlim[1], "Ipsi x-axis should not be inverted"
        plt.close(fig)

    def test_accepts_existing_figure(self, df_group):
        fig, _ = plt.subplots(1, 2, sharey=True)
        result = plot_relative_contrast(
            df_group, 'centered_mean', 'VTA-DA', 'stimOn_times', fig=fig
        )
        assert result is fig
        plt.close(fig)

    def test_empty_group_no_crash(self):
        df_empty = pd.DataFrame(
            columns=['subject', 'side', 'contrast', 'feedbackType', 'centered_mean']
        )
        fig = plot_relative_contrast(df_empty, 'centered_mean', 'VTA-DA', 'stimOn_times')
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_xticks_at_actual_contrast_values(self):
        """X-ticks should be at actual contrast values (data is already 0-100 scale)."""
        rows = [
            {'subject': s, 'side': side, 'contrast': c,
             'feedbackType': fb, 'centered_mean': 0.0}
            for s in ['s1', 's2', 's3']
            for side in ['contra', 'ipsi']
            for c in [0.0, 25.0, 100.0]
            for fb in [1, -1]
            for _ in range(5)
        ]
        df = pd.DataFrame(rows)
        fig = plot_relative_contrast(df, 'centered_mean', 'VTA-DA', 'stimOn_times')

        for ax in fig.axes:
            ticks = ax.get_xticks()
            assert set(ticks) == {0.0, 25.0, 100.0}, (
                f"Expected {{0.0, 25.0, 100.0}}, got {set(ticks)}"
            )
        plt.close(fig)

    def test_both_panels_share_contrast_set(self):
        """Both panels show ticks for all contrasts in df_group, even if one side has no data."""
        rows = [
            {'subject': s, 'side': 'contra', 'contrast': c,
             'feedbackType': 1, 'centered_mean': 0.1}
            for s in ['s1', 's2', 's3']
            for c in [0.0, 25.0, 100.0]
            for _ in range(5)
        ]  # no ipsi rows at all
        df = pd.DataFrame(rows)
        fig = plot_relative_contrast(df, 'centered_mean', 'VTA-DA', 'stimOn_times')

        expected = {0.0, 25.0, 100.0}
        for ax in fig.axes:
            assert set(ax.get_xticks()) == expected
        plt.close(fig)

    def test_window_label_in_suptitle(self, df_group):
        """window_label parameter should appear in the figure suptitle."""
        fig = plot_relative_contrast(
            df_group, 'centered_mean', 'VTA-DA', 'stimOn_times', window_label='early'
        )
        suptitle_text = fig.texts[0].get_text() if fig.texts else ''
        assert 'early' in suptitle_text, f"'early' not found in suptitle: {suptitle_text!r}"
        plt.close(fig)

    def test_errorbars_not_nan_when_one_subject_has_all_nan(self):
        """If one subject has all-NaN responses at a cell, errorbars should still
        be computed from the remaining subjects (nan_policy='omit')."""
        from matplotlib.container import ErrorbarContainer

        # s3 has all-NaN centered_mean — pandas groupby.mean() returns NaN for s3
        rows = [
            {'subject': s, 'side': 'contra', 'contrast': 25.0,
             'feedbackType': 1, 'centered_mean': val}
            for s, val in [('s1', 0.3), ('s2', -0.3)]
            for _ in range(5)
        ] + [
            {'subject': 's3', 'side': 'contra', 'contrast': 25.0,
             'feedbackType': 1, 'centered_mean': np.nan}
            for _ in range(5)
        ]
        df = pd.DataFrame(rows)

        fig = plot_relative_contrast(df, 'centered_mean', 'VTA-DA', 'stimOn_times')
        ax_c = fig.axes[0]  # contra panel

        eb_containers = [c for c in ax_c.containers if isinstance(c, ErrorbarContainer)]
        assert len(eb_containers) > 0, "No ErrorbarContainer found"

        has_finite = False
        for container in eb_containers:
            for bl in container.lines[2]:
                for seg in bl.get_segments():
                    seg_arr = np.array(seg)
                    if seg_arr.ndim == 2 and np.isfinite(seg_arr).all():
                        if seg_arr[0, 1] != seg_arr[1, 1]:
                            has_finite = True

        assert has_finite, (
            "Errorbars are NaN even though 2 subjects have valid data — "
            "scipy_sem may be propagating NaN from the third subject"
        )
        plt.close(fig)

    def test_errorbars_rendered_when_multiple_subjects(self):
        """Errorbars (± SEM) should be visible when subjects have different responses."""
        from matplotlib.container import ErrorbarContainer

        subject_vals = {'s1': -0.5, 's2': 0.0, 's3': 0.5}
        rows = [
            {'subject': s, 'side': side, 'contrast': 25.0,
             'feedbackType': fb, 'centered_mean': val}
            for s, val in subject_vals.items()
            for side in ['contra', 'ipsi']
            for fb in [1, -1]
            for _ in range(10)
        ]
        df = pd.DataFrame(rows)
        fig = plot_relative_contrast(df, 'centered_mean', 'VTA-DA', 'stimOn_times')

        for ax in fig.axes:
            eb_containers = [c for c in ax.containers if isinstance(c, ErrorbarContainer)]
            assert len(eb_containers) > 0, "No ErrorbarContainer found"

            has_finite = False
            for container in eb_containers:
                for bl in container.lines[2]:  # barlines = vertical error bar lines
                    for seg in bl.get_segments():
                        seg_arr = np.array(seg)
                        # Segment has two y values; if they differ the bar has non-zero height
                        if seg_arr.ndim == 2 and np.isfinite(seg_arr).all() and seg_arr[0, 1] != seg_arr[1, 1]:
                            has_finite = True

            assert has_finite, "No finite, non-zero error bar segments found"
        plt.close(fig)
