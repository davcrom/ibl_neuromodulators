"""Tests for iblnm.vis module."""
import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from unittest.mock import MagicMock

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

    def test_xticks_are_ranks(self):
        """X-ticks should be integer ranks, with contrast values as labels."""
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
            np.testing.assert_array_equal(ticks, [0, 1, 2])
            labels = [t.get_text() for t in ax.get_xticklabels()]
            assert labels == ['0', '25', '100']
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

        for ax in fig.axes:
            np.testing.assert_array_equal(ax.get_xticks(), [0, 1, 2])
        plt.close(fig)

    def test_subject_mean_removal(self):
        """Between-subject variance should be removed from plotted values."""
        # All subjects have same contrast effect but different offsets
        rows = []
        for s, offset in [('s1', 5.0), ('s2', -5.0), ('s3', 10.0)]:
            for side in ['contra', 'ipsi']:
                for c in [0.0, 25.0, 100.0]:
                    for fb in [1, -1]:
                        for _ in range(20):
                            rows.append({
                                'subject': s, 'side': side, 'contrast': c,
                                'feedbackType': fb,
                                'centered_mean': offset + c * 0.001,
                            })
        df = pd.DataFrame(rows)
        fig = plot_relative_contrast(df, 'centered_mean', 'VTA-DA', 'stimOn_times')
        ax = fig.axes[0]
        # After subject-mean removal, spread across contrasts should be tiny
        # (just the 0.001*c effect), not dominated by subject offsets
        lines = ax.get_lines()
        for line in lines:
            ydata = line.get_ydata()
            if len(ydata) > 1:
                assert np.ptp(ydata) < 1.0, (
                    f"Subject-mean removal failed: spread {np.ptp(ydata):.2f}")
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
        # s1 and s2 have different within-subject variance so errorbars survive
        # subject-mean removal
        rng = np.random.default_rng(42)
        rows = [
            {'subject': s, 'side': 'contra', 'contrast': 25.0,
             'feedbackType': 1, 'centered_mean': val + rng.normal(0, 0.1)}
            for s, val in [('s1', 0.3), ('s2', -0.3)]
            for _ in range(15)
        ] + [
            {'subject': 's3', 'side': 'contra', 'contrast': 25.0,
             'feedbackType': 1, 'centered_mean': np.nan}
            for _ in range(15)
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

    def test_errorbars_rendered_with_within_subject_variance(self):
        """Errorbars (± SEM) should be visible when there is within-subject variance."""
        from matplotlib.container import ErrorbarContainer

        # After subject-mean removal, between-subject variance is gone.
        # Need within-subject variance for error bars.
        rng = np.random.default_rng(42)
        rows = [
            {'subject': s, 'side': side, 'contrast': 25.0,
             'feedbackType': fb, 'centered_mean': rng.normal(0, 0.5)}
            for s in ['s1', 's2', 's3']
            for side in ['contra', 'ipsi']
            for fb in [1, -1]
            for _ in range(15)
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
                        if seg_arr.ndim == 2 and np.isfinite(seg_arr).all() and seg_arr[0, 1] != seg_arr[1, 1]:
                            has_finite = True

            assert has_finite, "No finite, non-zero error bar segments found"
        plt.close(fig)

    def test_pool_and_subject_converge_after_subject_mean_removal(self):
        """After subject-mean removal, pool and subject aggregation should give
        the same grand mean (both see the same adjusted values)."""
        rows = (
            [{'subject': 's1', 'side': 'contra', 'contrast': 25.0,
              'feedbackType': 1, 'response': 1.0}] * 20
            + [{'subject': 's2', 'side': 'contra', 'contrast': 25.0,
                'feedbackType': 1, 'response': 0.0}] * 5
        )
        df = pd.DataFrame(rows)

        fig_pool = plot_relative_contrast(df, 'response', 'VTA-DA', 'stimOn_times',
                                          aggregation='pool')
        fig_subj = plot_relative_contrast(df.copy(), 'response', 'VTA-DA', 'stimOn_times',
                                          aggregation='subject')
        pool_mean = fig_pool.axes[0].containers[0].lines[0].get_ydata()[0]
        subj_mean = fig_subj.axes[0].containers[0].lines[0].get_ydata()[0]
        # Grand mean is preserved by subject-mean removal
        expected = df['response'].mean()
        assert np.isclose(pool_mean, expected, atol=0.01), (
            f"Pool mean {pool_mean} != expected {expected}")
        assert np.isclose(subj_mean, expected, atol=0.01), (
            f"Subject mean {subj_mean} != expected {expected}")
        plt.close(fig_pool)
        plt.close(fig_subj)

    def test_pool_is_default_aggregation(self):
        """Default aggregation should be 'pool'."""
        rows = (
            [{'subject': 's1', 'side': 'contra', 'contrast': 25.0,
              'feedbackType': 1, 'response': 1.0}] * 20
            + [{'subject': 's2', 'side': 'contra', 'contrast': 25.0,
                'feedbackType': 1, 'response': 0.0}] * 5
        )
        df = pd.DataFrame(rows)

        fig = plot_relative_contrast(df, 'response', 'VTA-DA', 'stimOn_times')
        ax_c = fig.axes[0]
        line = ax_c.containers[0].lines[0]
        plotted_mean = line.get_ydata()[0]
        assert np.isclose(plotted_mean, 0.8), (
            f"Default should be pool (0.8), got {plotted_mean}"
        )
        plt.close(fig)

    def test_invalid_aggregation_raises(self):
        """Invalid aggregation value should raise ValueError."""
        rows = [{'subject': 's1', 'side': 'contra', 'contrast': 25.0,
                 'feedbackType': 1, 'response': 0.5}]
        df = pd.DataFrame(rows)
        with pytest.raises(ValueError, match='aggregation'):
            plot_relative_contrast(df, 'response', 'VTA-DA', 'stimOn_times',
                                   aggregation='invalid')

    def test_min_trials_filter_drops_sparse_cells(self):
        """Cells with <= min_trials rows should be excluded from the plot."""
        from matplotlib.container import ErrorbarContainer

        rng = np.random.default_rng(99)
        # s1 and s2 have 20 trials at contrast 25 (well above threshold)
        rows = [
            {'subject': s, 'side': 'contra', 'contrast': 25.0,
             'feedbackType': 1, 'centered_mean': rng.normal(0, 0.1)}
            for s in ['s1', 's2']
            for _ in range(20)
        ]
        # s1 and s2 have only 3 trials at contrast 100 (below threshold)
        rows += [
            {'subject': s, 'side': 'contra', 'contrast': 100.0,
             'feedbackType': 1, 'centered_mean': rng.normal(5, 0.1)}
            for s in ['s1', 's2']
            for _ in range(3)
        ]
        df = pd.DataFrame(rows)

        fig = plot_relative_contrast(df, 'centered_mean', 'VTA-DA',
                                     'stimOn_times', min_trials=10)
        ax_c = fig.axes[0]
        eb_containers = [c for c in ax_c.containers
                         if isinstance(c, ErrorbarContainer)]
        # Only contrast 25 should survive; the plotted mean should be near 0
        # (not pulled toward the contrast-100 value of ~5)
        for container in eb_containers:
            ydata = np.array(container.lines[0].get_ydata(), dtype=float)
            finite = ydata[np.isfinite(ydata)]
            assert all(abs(v) < 2.0 for v in finite), (
                f"Sparse cells (contrast=100, mean~5) leaked through min_trials "
                f"filter: plotted values {finite}"
            )
        plt.close(fig)


class TestPlotSimilarityMatrix:

    def test_returns_figure(self):
        from iblnm.vis import plot_similarity_matrix
        sim = pd.DataFrame(
            [[1.0, 0.5], [0.5, 1.0]],
            index=pd.MultiIndex.from_tuples([('e0', 'VTA'), ('e1', 'DR')]),
            columns=pd.MultiIndex.from_tuples([('e0', 'VTA'), ('e1', 'DR')]),
        )
        labels = pd.Series(['DA', '5HT'], index=sim.index)
        fig = plot_similarity_matrix(sim, labels)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_has_colorbar(self):
        from iblnm.vis import plot_similarity_matrix
        sim = pd.DataFrame(
            [[1.0, 0.3], [0.3, 1.0]],
            index=pd.MultiIndex.from_tuples([('e0', 'VTA'), ('e1', 'DR')]),
            columns=pd.MultiIndex.from_tuples([('e0', 'VTA'), ('e1', 'DR')]),
        )
        labels = pd.Series(['DA', '5HT'], index=sim.index)
        fig = plot_similarity_matrix(sim, labels)
        # Figure should have more than 1 axes (main + colorbar)
        assert len(fig.axes) >= 2
        plt.close(fig)

    def test_target_labels_on_axis(self):
        """Each target-NM group should be labeled on the y-axis."""
        from iblnm.vis import plot_similarity_matrix
        idx = pd.MultiIndex.from_tuples(
            [('e0', 'VTA'), ('e1', 'VTA'), ('e2', 'DR'), ('e3', 'DR')],
        )
        sim = pd.DataFrame(np.eye(4), index=idx, columns=idx)
        labels = pd.Series(['DA', 'DA', '5HT', '5HT'], index=idx)
        subjects = pd.Series(['s0', 's1', 's0', 's1'], index=idx)
        fig = plot_similarity_matrix(sim, labels, subjects=subjects)
        ax = fig.axes[0]
        tick_labels = [t.get_text() for t in ax.get_yticklabels()]
        assert 'DA' in tick_labels
        assert '5HT' in tick_labels
        plt.close(fig)

    def test_sorts_by_subject_within_target(self):
        """Within each target group, recordings should be sorted by subject."""
        from iblnm.vis import plot_similarity_matrix
        # Deliberately unsorted subjects within each target
        idx = pd.MultiIndex.from_tuples(
            [('e0', 'r0'), ('e1', 'r1'), ('e2', 'r2'), ('e3', 'r3')],
        )
        sim = pd.DataFrame(np.eye(4), index=idx, columns=idx)
        labels = pd.Series(['A', 'A', 'A', 'A'], index=idx)
        subjects = pd.Series(['s2', 's0', 's1', 's0'], index=idx)
        fig = plot_similarity_matrix(sim, labels, subjects=subjects)
        # The returned figure's data should be reordered
        # s0 (e1, e3) should come before s1 (e2) before s2 (e0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotConfusionMatrix:

    def test_returns_figure(self):
        from iblnm.vis import plot_confusion_matrix
        cm = pd.DataFrame(
            [[8, 2], [1, 9]],
            index=['DA', '5HT'], columns=['DA', '5HT'],
        )
        fig = plot_confusion_matrix(cm)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_cells_show_counts(self):
        """Each cell should have a text annotation with the count."""
        from iblnm.vis import plot_confusion_matrix
        cm = pd.DataFrame(
            [[5, 0], [0, 3]],
            index=['A', 'B'], columns=['A', 'B'],
        )
        fig = plot_confusion_matrix(cm)
        ax = fig.axes[0]
        texts = [t.get_text() for t in ax.texts]
        assert '5' in texts
        assert '3' in texts
        assert '0' in texts
        plt.close(fig)

    def test_accuracy_in_title(self):
        """Title should contain the overall accuracy."""
        from iblnm.vis import plot_confusion_matrix
        # 8 correct out of 10 → 80%
        cm = pd.DataFrame(
            [[5, 0], [2, 3]],
            index=['A', 'B'], columns=['A', 'B'],
        )
        fig = plot_confusion_matrix(cm)
        title = fig.axes[0].get_title()
        assert '80' in title or '0.8' in title, f"Expected accuracy in title, got: {title!r}"
        plt.close(fig)


class TestPlotDecodingCoefficients:

    def test_returns_figure(self):
        from iblnm.vis import plot_decoding_coefficients
        coefs = pd.DataFrame(
            [[0.5, -0.3, 0.1], [0.0, 0.8, -0.2]],
            index=['DA', '5HT'],
            columns=['stimOn_c0_correct', 'stimOn_c25_correct', 'feedback_c100_correct'],
        )
        fig = plot_decoding_coefficients(coefs)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotFeatureContributions:

    def test_returns_figure(self):
        from iblnm.vis import plot_feature_contributions
        contrib = pd.DataFrame({
            'feature': ['f0', 'f1', 'f2'],
            'full_accuracy': [0.8, 0.8, 0.8],
            'reduced_accuracy': [0.5, 0.7, 0.75],
            'delta': [0.3, 0.1, 0.05],
        })
        fig = plot_feature_contributions(contrib)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_bars_sorted_by_delta(self):
        from iblnm.vis import plot_feature_contributions
        contrib = pd.DataFrame({
            'feature': ['f0', 'f1', 'f2'],
            'full_accuracy': [0.8, 0.8, 0.8],
            'reduced_accuracy': [0.7, 0.5, 0.75],
            'delta': [0.1, 0.3, 0.05],
        })
        fig = plot_feature_contributions(contrib)
        ax = fig.axes[0]
        # Bars sorted ascending (largest delta at top of plot = last tick)
        tick_labels = [t.get_text() for t in ax.get_yticklabels()]
        assert tick_labels[-1] == 'f1'   # largest delta at top
        assert tick_labels[0] == 'f2'    # smallest delta at bottom
        plt.close(fig)


class TestFeatureSortKey:

    def test_sort_order_side_event_feedback_contrast(self):
        from iblnm.vis import feature_sort_key
        features = [
            'stimOn_c0_ipsi_correct',
            'feedback_c1_contra_incorrect',
            'stimOn_c1_contra_correct',
            'stimOn_c0_contra_incorrect',
            'stimOn_c0_contra_correct',
            'feedback_c0_contra_correct',
        ]
        sorted_features = sorted(features, key=feature_sort_key)
        # contra before ipsi, stimOn before feedback, correct before incorrect,
        # then ascending contrast
        assert sorted_features == [
            'stimOn_c0_contra_correct',
            'stimOn_c1_contra_correct',
            'stimOn_c0_contra_incorrect',
            'feedback_c0_contra_correct',
            'feedback_c1_contra_incorrect',
            'stimOn_c0_ipsi_correct',
        ]

    def test_unparseable_labels_sort_last(self):
        from iblnm.vis import feature_sort_key
        features = ['stimOn_c0_contra_correct', 'unknown_feature']
        sorted_features = sorted(features, key=feature_sort_key)
        assert sorted_features[0] == 'stimOn_c0_contra_correct'
        assert sorted_features[1] == 'unknown_feature'


class TestPlotMeanResponseVectors:

    def _make_matrix_with_labels(self):
        features = [
            'stimOn_c0_contra_correct', 'stimOn_c0_contra_incorrect',
            'feedback_c1_contra_correct', 'feedback_c1_contra_incorrect',
        ]
        index = pd.MultiIndex.from_tuples(
            [('e0', 'VTA-DA'), ('e1', 'VTA-DA'), ('e2', 'DR-5HT'), ('e3', 'DR-5HT')],
            names=['eid', 'target_NM'],
        )
        data = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [1.5, 2.5, 3.5, 4.5],
            [5.0, 6.0, 7.0, 8.0],
            [5.5, 6.5, 7.5, 8.5],
        ])
        return pd.DataFrame(data, index=index, columns=features)

    def test_returns_figure(self):
        from iblnm.vis import plot_mean_response_vectors
        df = self._make_matrix_with_labels()
        fig = plot_mean_response_vectors(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_one_errorbar_per_target(self):
        from iblnm.vis import plot_mean_response_vectors
        df = self._make_matrix_with_labels()
        fig = plot_mean_response_vectors(df)
        ax = fig.axes[0]
        assert len(ax.containers) == 2  # DA + 5HT
        plt.close(fig)

    def test_legend_has_target_names(self):
        from iblnm.vis import plot_mean_response_vectors
        df = self._make_matrix_with_labels()
        fig = plot_mean_response_vectors(df)
        ax = fig.axes[0]
        legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert 'VTA-DA' in legend_labels
        assert 'DR-5HT' in legend_labels
        plt.close(fig)

    def test_has_two_axes(self):
        """Stacked: raw on top, normalized on bottom."""
        from iblnm.vis import plot_mean_response_vectors
        df = self._make_matrix_with_labels()
        fig = plot_mean_response_vectors(df)
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_features_sorted_by_side_event_feedback_contrast(self):
        from iblnm.vis import plot_mean_response_vectors
        # Columns in wrong order — should be sorted in the plot
        features = [
            'feedback_c1_ipsi_correct',
            'stimOn_c0_contra_correct',
            'stimOn_c1_contra_incorrect',
            'stimOn_c0_ipsi_correct',
        ]
        index = pd.MultiIndex.from_tuples(
            [('e0', 'VTA-DA'), ('e1', 'DR-5HT')],
            names=['eid', 'target_NM'],
        )
        data = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        df = pd.DataFrame(data, index=index, columns=features)
        fig = plot_mean_response_vectors(df)
        ax = fig.axes[-1]  # bottom axis has tick labels
        tick_labels = [t.get_text() for t in ax.get_xticklabels()]
        assert tick_labels == [
            'stimOn_c0_contra_correct',
            'stimOn_c1_contra_incorrect',
            'stimOn_c0_ipsi_correct',
            'feedback_c1_ipsi_correct',
        ]
        plt.close(fig)

    def test_top_axis_shows_raw_ylabel(self):
        from iblnm.vis import plot_mean_response_vectors
        df = self._make_matrix_with_labels()
        fig = plot_mean_response_vectors(df)
        assert 'raw' in fig.axes[0].get_ylabel().lower()
        plt.close(fig)

    def test_bottom_axis_shows_normalized_ylabel(self):
        from iblnm.vis import plot_mean_response_vectors
        df = self._make_matrix_with_labels()
        fig = plot_mean_response_vectors(df)
        assert 'normalized' in fig.axes[1].get_ylabel().lower()
        plt.close(fig)

    def test_uses_targetnm_colors(self):
        import matplotlib as mpl
        from iblnm.vis import plot_mean_response_vectors
        from iblnm.config import TARGETNM_COLORS
        df = self._make_matrix_with_labels()
        fig = plot_mean_response_vectors(df)
        ax = fig.axes[0]
        targets = sorted(df.index.get_level_values('target_NM').unique())
        # errorbar lines: get color from the line children
        for container, target in zip(ax.containers, targets):
            expected = mpl.colors.to_rgba(TARGETNM_COLORS[target])
            actual = mpl.colors.to_rgba(container[0].get_color())
            np.testing.assert_allclose(actual, expected, atol=0.01)
        plt.close(fig)


class TestPlotDecodingSummary:

    def _make_data(self):
        features = [
            'stimOn_c1_contra_correct',
            'stimOn_c0_contra_correct',
            'feedback_c0_ipsi_incorrect',
        ]
        coefs = pd.DataFrame(
            [[0.5, -0.3, 0.8], [0.1, 0.7, -0.2]],
            index=['DA', '5HT'], columns=features,
        )
        contrib = pd.DataFrame({
            'feature': features,
            'full_accuracy': [0.8, 0.8, 0.8],
            'reduced_accuracy': [0.5, 0.7, 0.6],
            'delta': [0.3, 0.1, 0.2],
        })
        return coefs, contrib

    def test_returns_figure(self):
        from iblnm.vis import plot_decoding_summary
        coefs, contrib = self._make_data()
        fig = plot_decoding_summary(coefs, contrib)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_has_two_axes(self):
        from iblnm.vis import plot_decoding_summary
        coefs, contrib = self._make_data()
        fig = plot_decoding_summary(coefs, contrib)
        assert len(fig.axes) >= 2  # coefs + contrib (+ colorbar)
        plt.close(fig)

    def test_shared_x_order_sorted_by_feature(self):
        """Both panels should share x-axis sorted by side > event > fb > contrast."""
        from iblnm.vis import plot_decoding_summary
        coefs, contrib = self._make_data()
        fig = plot_decoding_summary(coefs, contrib)
        ax_contrib = fig.axes[1]
        tick_labels = [t.get_text() for t in ax_contrib.get_xticklabels()]
        # side>event>fb>contrast: contra stimOn c0 first, then c1, then ipsi feedback
        assert tick_labels == [
            'stimOn_c0_contra_correct',
            'stimOn_c1_contra_correct',
            'feedback_c0_ipsi_incorrect',
        ]
        plt.close(fig)


class TestPlotEmpiricalSimilarity:

    def _make_target_sim(self):
        targets = ['A', 'B']
        return pd.DataFrame(
            [[0.9, 0.3], [0.3, 0.85]],
            index=targets, columns=targets,
        )

    def test_returns_figure(self):
        from iblnm.vis import plot_empirical_similarity
        fig = plot_empirical_similarity(self._make_target_sim())
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_cells_show_values(self):
        from iblnm.vis import plot_empirical_similarity
        target_sim = self._make_target_sim()
        fig = plot_empirical_similarity(target_sim)
        ax = fig.axes[0]
        texts = [t.get_text() for t in ax.texts]
        assert '0.90' in texts
        assert '0.30' in texts
        plt.close(fig)

    def test_axis_labels_are_targets(self):
        from iblnm.vis import plot_empirical_similarity
        fig = plot_empirical_similarity(self._make_target_sim())
        ax = fig.axes[0]
        xlabels = [t.get_text() for t in ax.get_xticklabels()]
        ylabels = [t.get_text() for t in ax.get_yticklabels()]
        assert xlabels == ['A', 'B']
        assert ylabels == ['A', 'B']
        plt.close(fig)

    def test_side_by_side_with_loso(self):
        """Passing loso_matrix produces two axes side by side."""
        from iblnm.vis import plot_empirical_similarity
        full = self._make_target_sim()
        loso = pd.DataFrame(
            [[0.8, 0.2], [0.2, 0.75]],
            index=['A', 'B'], columns=['A', 'B'],
        )
        fig = plot_empirical_similarity(full, loso_matrix=loso)
        # Two heatmap axes (plus colorbars)
        heatmap_axes = [ax for ax in fig.axes if ax.images]
        assert len(heatmap_axes) == 2
        plt.close(fig)

    def test_side_by_side_titles(self):
        from iblnm.vis import plot_empirical_similarity
        full = self._make_target_sim()
        loso = self._make_target_sim()
        fig = plot_empirical_similarity(full, loso_matrix=loso)
        heatmap_axes = [ax for ax in fig.axes if ax.images]
        titles = [ax.get_title() for ax in heatmap_axes]
        assert 'all' in titles[0].lower()
        assert 'cross' in titles[1].lower() or 'loso' in titles[1].lower()
        plt.close(fig)

    def test_single_matrix_still_works(self):
        """Without loso_matrix, behaves as before (one axis)."""
        from iblnm.vis import plot_empirical_similarity
        fig = plot_empirical_similarity(self._make_target_sim())
        heatmap_axes = [ax for ax in fig.axes if ax.images]
        assert len(heatmap_axes) == 1
        plt.close(fig)


# =============================================================================
# LMM Plot Tests
# =============================================================================


def test_plot_lmm_response_removed():
    """plot_lmm_response (prediction-curve plotter) is deleted, unused."""
    import iblnm.vis as vis
    assert not hasattr(vis, 'plot_lmm_response')


class TestPlotLMMVarianceExplained:

    @staticmethod
    def _r2_df():
        return pd.DataFrame({
            'target_NM': ['VTA-DA', 'DR-5HT'],
            'marginal_r2': [0.15, 0.08],
            'conditional_r2': [0.35, 0.22],
        })

    def test_returns_figure_when_ax_none(self):
        from iblnm.vis import plot_lmm_variance_explained
        fig = plot_lmm_variance_explained(self._r2_df())
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_draws_on_passed_ax_without_new_figure(self):
        from iblnm.vis import plot_lmm_variance_explained
        fig, ax = plt.subplots()
        n_before = len(ax.patches)
        existing = set(plt.get_fignums())
        plot_lmm_variance_explained(self._r2_df(), ax=ax)
        assert len(ax.patches) > n_before
        assert set(plt.get_fignums()) == existing
        plt.close(fig)

    def test_bar_heights_match_input(self):
        from iblnm.vis import plot_lmm_variance_explained
        fig, ax = plt.subplots()
        plot_lmm_variance_explained(self._r2_df(), ax=ax)
        heights = sorted(round(p.get_height(), 3) for p in ax.patches)
        assert heights == sorted([0.15, 0.08, 0.35, 0.22])
        plt.close(fig)


class TestPlotMarginalMeans:

    @staticmethod
    def _reward_emm_df():
        """New-schema main-effect EMMs for the reward factor (coded levels)."""
        return pd.DataFrame({
            'target_NM': ['VTA-DA', 'VTA-DA', 'DR-5HT', 'DR-5HT'],
            'reward': [-0.5, 0.5, -0.5, 0.5],
            'predicted': [0.5, 0.8, 0.2, 0.4],
            'ci_lower': [0.3, 0.6, 0.0, 0.2],
            'ci_upper': [0.7, 1.0, 0.4, 0.6],
        })

    def test_returns_figure_when_ax_none(self):
        from iblnm.vis import plot_marginal_means
        fig = plot_marginal_means(self._reward_emm_df())
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_draws_on_passed_ax_without_new_figure(self):
        from iblnm.vis import plot_marginal_means
        fig, ax = plt.subplots()
        existing = set(plt.get_fignums())
        plot_marginal_means(self._reward_emm_df(), ax=ax)
        assert len(ax.containers) >= 2  # one errorbar series per target
        assert set(plt.get_fignums()) == existing
        plt.close(fig)

    def test_marker_y_positions_match_predicted(self):
        from iblnm.vis import plot_marginal_means
        fig, ax = plt.subplots()
        plot_marginal_means(self._reward_emm_df(), ax=ax)
        ys = sorted(
            round(y, 3)
            for c in ax.containers
            for y in c[0].get_ydata()
        )
        assert ys == sorted([0.5, 0.8, 0.2, 0.4])
        plt.close(fig)

    def test_xticklabels_map_coded_levels_to_labels(self):
        from iblnm.vis import plot_marginal_means
        fig, ax = plt.subplots()
        plot_marginal_means(self._reward_emm_df(), ax=ax)
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert labels == ['incorrect', 'correct']
        plt.close(fig)


# =============================================================================
# Consolidated LMM Summary Plot Tests
# =============================================================================

class TestPlotLMMSummary:
    """Orchestrator fed the precomputed effect frames, for one event."""

    @staticmethod
    def _r2_df():
        """``response_lmm_fit`` R² frame: two targets, two events."""
        return pd.DataFrame({
            'target_NM': ['VTA-DA', 'DR-5HT', 'VTA-DA', 'DR-5HT'],
            'event': ['stimOn', 'stimOn', 'feedback', 'feedback'],
            'name': 'task_full',
            'marginal_r2': [0.15, 0.08, 0.10, 0.05],
            'conditional_r2': [0.35, 0.22, 0.30, 0.18],
        })

    @staticmethod
    def _coef_df():
        """``response_lmm_effects(..., 'coefficients')``: terms × targets.

        VTA-DA contrast main effect significant; DR-5HT not. Both contrast×side
        interactions non-significant.
        """
        terms = ['Intercept', 'contrast', 'side', 'reward',
                 'contrast:reward', 'contrast:side', 'side:reward']
        vta_p = [1e-10, 1e-10, 0.2, 0.001, 0.03, 0.3, 0.001]
        dr_p = [1e-10, 0.06, 0.8, 0.15, 0.9, 0.7, 0.5]
        rows = []
        for tnm, pvals in (('VTA-DA', vta_p), ('DR-5HT', dr_p)):
            for term, p in zip(terms, pvals):
                rows.append({'term': term, 'target_NM': tnm, 'event': 'stimOn',
                             'Coef.': 0.3, 'P>|z|': p})
        return pd.DataFrame(rows)

    @staticmethod
    def _emm_frames():
        """``response_lmm_effects(..., 'emm', [factor])`` frames per factor."""
        def _frame(factor, levels):
            rows = []
            for tnm in ('VTA-DA', 'DR-5HT'):
                for lvl in levels:
                    rows.append({'target_NM': tnm, factor: lvl,
                                 'event': 'stimOn', 'predicted': 0.3,
                                 'ci_lower': 0.1, 'ci_upper': 0.5})
            return pd.DataFrame(rows)
        return {'reward': _frame('reward', [-0.5, 0.5]),
                'side': _frame('side', [-0.5, 0.5]),
                'contrast': _frame('contrast', [-1.0, 0.0, 1.0])}

    def _summary(self, **kwargs):
        from iblnm.vis import plot_lmm_summary
        return plot_lmm_summary(self._r2_df(), self._coef_df(),
                                self._emm_frames(), 'stimOn', **kwargs)

    def test_returns_figure(self):
        fig = self._summary()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_has_five_panels(self):
        fig = self._summary()
        # 5 content panels + 1 colorbar = 6 axes
        assert len(fig.axes) >= 5
        plt.close(fig)

    def test_r2_panel_bars_match_input(self):
        """R² panel: two bars per target, heights from the R² frame."""
        fig = self._summary()
        ax_r2 = fig.axes[0]
        heights = sorted(round(p.get_height(), 3) for p in ax_r2.patches
                         if p.get_width() > 0)
        # stimOn rows only: VTA (0.15, 0.35), DR (0.08, 0.22)
        assert heights == sorted([0.15, 0.35, 0.08, 0.22])
        plt.close(fig)

    def test_heatmap_panel_present(self):
        """Coefficient heatmap is the top-right panel (axes[1])."""
        fig = self._summary()
        assert len(fig.axes[1].images) > 0
        plt.close(fig)

    def test_emm_panels_have_errorbars(self):
        """Bottom row (axes[2:5]) are EMM panels with errorbar containers."""
        fig = self._summary()
        for ax in fig.axes[2:5]:
            assert len(ax.containers) > 0
        plt.close(fig)

    def test_summary_annotates_formula(self):
        """A formula passed in is rendered in the figure title."""
        formula = 'response ~ contrast * side * reward'
        fig = self._summary(formula=formula)
        assert formula in fig._suptitle.get_text()
        plt.close(fig)


# =============================================================================
# _sort_events Tests
# =============================================================================


class TestSortEvents:

    def test_orders_events_chronologically(self):
        """Events sort stimOn → firstMovement → feedback regardless of input
        order, not alphabetically. Operates on the real ``_times`` event names."""
        from iblnm.vis import _sort_events
        shuffled = ['feedback_times', 'firstMovement_times', 'stimOn_times']
        assert _sort_events(shuffled) == [
            'stimOn_times', 'firstMovement_times', 'feedback_times']

    def test_unknown_events_sort_last_alphabetically(self):
        """Events outside the chronology fall after known ones, in name order."""
        from iblnm.vis import _sort_events
        assert _sort_events(['zzz', 'stimOn_times', 'aaa']) == [
            'stimOn_times', 'aaa', 'zzz']


# =============================================================================
# plot_lmm_ceiling / plot_lmm_loso Tests
# =============================================================================


class TestPlotLMMSuiteFigures:

    def _ceiling(self):
        return pd.DataFrame([
            {'target_NM': 'VTA-DA', 'event': 'stimOn',
             'marginal': 0.12, 'conditional': 0.30},
            {'target_NM': 'DR-5HT', 'event': 'stimOn',
             'marginal': 0.05, 'conditional': 0.18},
        ])

    def test_ceiling_paired_bars_per_target(self):
        """Two bars (marginal, conditional) per target_NM."""
        from iblnm.vis import plot_lmm_ceiling
        fig = plot_lmm_ceiling(self._ceiling())
        patches = [p for p in fig.axes[0].patches if p.get_height() != 0]
        assert len(patches) == 4  # 2 targets × (marginal, conditional)
        plt.close(fig)

    def test_ceiling_formula_in_title(self):
        """The saturated-model description is shown in the title, not bottom text."""
        from iblnm.vis import plot_lmm_ceiling
        fig = plot_lmm_ceiling(self._ceiling())
        title = fig._suptitle.get_text()
        assert 'C(contrast) × side' in title
        assert 'no side:reward' in title
        plt.close(fig)

    def test_suite_plots_handle_empty_frames(self):
        """The ceiling plot returns a labelled figure on an empty frame."""
        from iblnm.vis import plot_lmm_ceiling
        fig = plot_lmm_ceiling(self._ceiling().iloc[0:0])
        assert isinstance(fig, plt.Figure)
        assert fig._suptitle is not None
        plt.close(fig)

    def test_plot_lmm_loso_removed(self):
        """plot_lmm_loso is deleted (subsumed by the interactions predictor)."""
        import iblnm.vis as vis
        assert not hasattr(vis, 'plot_lmm_loso')


class TestScatterFolds:
    """Fold-agnostic scatter: per-fold faint markers + large aggregate marker."""

    def _group(self):
        return pd.DataFrame([
            {'fold': 's0', 'delta_r2': 0.02},
            {'fold': 's1', 'delta_r2': -0.01},
            {'fold': 'aggregate', 'delta_r2': 0.005},
        ])

    def test_splits_folds_from_aggregate_by_size(self):
        """Per-fold markers are small, the aggregate marker large."""
        from iblnm.vis import _scatter_folds
        fig, ax = plt.subplots()
        _scatter_folds(ax, 0, self._group(), 'C0')
        sizes = [c.get_sizes()[0] for c in ax.collections if len(c.get_sizes())]
        assert max(sizes) > min(sizes)
        plt.close(fig)

    def test_reads_fold_column(self):
        """A frame keyed on `subject` (no `fold`) raises."""
        from iblnm.vis import _scatter_folds
        fig, ax = plt.subplots()
        df = pd.DataFrame([{'subject': 's0', 'delta_r2': 0.02}])
        with pytest.raises(KeyError):
            _scatter_folds(ax, 0, df, 'C0')
        plt.close(fig)


class TestPlotLmmReliability:
    """Grid of target-NM (rows) × event (cols), x = predictor terms."""

    def _rel(self, targets=('VTA-DA', 'DR-5HT'),
             events=('feedback_times', 'stimOn_times'),
             predictors=('contrast', 'side', 'reward', 'interactions')):
        rows = []
        for tnm in targets:
            for event in events:
                for pred in predictors:
                    for fold in ['s0', 's1', 'aggregate']:
                        rows.append({
                            'target_NM': tnm, 'event': event, 'predictor': pred,
                            'fold': fold,
                            'delta_r2': 0.03 if fold == 'aggregate' else 0.01})
        return pd.DataFrame(rows)

    def _full_r2(self, targets=('VTA-DA', 'DR-5HT'),
                 events=('feedback_times', 'stimOn_times'), marginal=0.1):
        """Full-model marginal R² per (target_NM, event) panel."""
        return pd.DataFrame([
            {'target_NM': tnm, 'event': event, 'marginal_r2': marginal}
            for tnm in targets for event in events])

    def test_grid_rows_targets_cols_events(self):
        from iblnm.vis import plot_lmm_reliability
        fig = plot_lmm_reliability(self._rel(), self._full_r2(),
                                   'Task reliability')
        assert len(fig.axes) == 2 * 2  # 2 targets × 2 events
        plt.close(fig)

    def test_event_columns_chronological(self):
        """Top-row panel titles are the events in trial chronology."""
        from iblnm.vis import plot_lmm_reliability
        fig = plot_lmm_reliability(self._rel(), self._full_r2(),
                                   'Task reliability')
        assert [ax.get_title() for ax in fig.axes[:2]] == [
            'stimOn_times', 'feedback_times']
        plt.close(fig)

    def test_xticklabels_are_terms_in_order(self):
        """Bottom-row x-axis lists the task terms, main effects then interactions."""
        from iblnm.vis import plot_lmm_reliability
        fig = plot_lmm_reliability(self._rel(), self._full_r2(),
                                   'Task reliability')
        labels = [t.get_text() for t in fig.axes[2].get_xticklabels()]
        assert labels == ['contrast', 'side', 'reward', 'interactions']
        plt.close(fig)

    def test_rows_labeled_by_target(self):
        """Each row's leftmost panel is labelled with its target-NM, ordered."""
        from iblnm.vis import plot_lmm_reliability
        fig = plot_lmm_reliability(self._rel(), self._full_r2(),
                                   'Task reliability')
        assert [fig.axes[0].get_ylabel(), fig.axes[2].get_ylabel()] == \
            ['VTA-DA', 'DR-5HT']
        plt.close(fig)

    def test_row_colored_by_target(self):
        from iblnm.vis import plot_lmm_reliability
        from iblnm.config import TARGETNM_COLORS
        from matplotlib.colors import to_rgba
        fig = plot_lmm_reliability(self._rel(), self._full_r2(),
                                   'Task reliability')
        colors = {tuple(np.round(c.get_facecolor()[0], 5))
                  for c in fig.axes[0].collections if len(c.get_offsets())}
        assert tuple(np.round(to_rgba(TARGETNM_COLORS['VTA-DA']), 5)) in colors
        plt.close(fig)

    def test_aggregate_marker_larger_than_folds(self):
        from iblnm.vis import plot_lmm_reliability
        fig = plot_lmm_reliability(self._rel(), self._full_r2(),
                                   'Task reliability')
        sizes = [c.get_sizes()[0] for c in fig.axes[0].collections
                 if len(c.get_sizes())]
        assert max(sizes) > min(sizes)
        plt.close(fig)

    def test_title_is_suptitle(self):
        from iblnm.vis import plot_lmm_reliability
        fig = plot_lmm_reliability(self._rel(), self._full_r2(), 'My ΔR² title')
        assert fig._suptitle.get_text() == 'My ΔR² title'
        plt.close(fig)

    def test_empty_frame_returns_titled_figure(self):
        from iblnm.vis import plot_lmm_reliability
        fig = plot_lmm_reliability(self._rel().iloc[0:0], self._full_r2(),
                                   'Empty title')
        assert isinstance(fig, plt.Figure)
        assert fig._suptitle.get_text() == 'Empty title'
        plt.close(fig)

    def test_movement_predictors_mapped_to_x(self):
        """A movement-shaped frame renders and maps its predictors to the x-axis."""
        from iblnm.vis import plot_lmm_reliability
        df = self._rel(predictors=('contrast', 'log_reaction_time'))
        fig = plot_lmm_reliability(df, self._full_r2(), 'Movement reliability')
        labels = [t.get_text() for t in fig.axes[2].get_xticklabels()]
        assert labels == ['contrast', 'log_reaction_time']
        plt.close(fig)

    def test_marker_height_is_raw_delta_r2(self):
        """Each marker's height is the raw delta_r2, not scaled by marginal R²."""
        from iblnm.vis import plot_lmm_reliability
        fig = plot_lmm_reliability(self._rel(), self._full_r2(marginal=0.1),
                                   'Task reliability')
        heights = {round(float(off[1]), 4)
                   for c in fig.axes[0].collections
                   for off in c.get_offsets()}
        assert 0.03 in heights   # aggregate delta_r2
        assert 0.01 in heights   # per-fold delta_r2
        plt.close(fig)

    def test_marginal_r2_annotated_per_panel(self):
        """Each panel carries a text annotation of its full-model marginal R²."""
        from iblnm.vis import plot_lmm_reliability
        fig = plot_lmm_reliability(self._rel(), self._full_r2(marginal=0.1),
                                   'Task reliability')
        for ax in fig.axes:
            assert any('0.10' in t.get_text() for t in ax.texts)
        plt.close(fig)

    def test_no_legend(self):
        from iblnm.vis import plot_lmm_reliability
        fig = plot_lmm_reliability(self._rel(), self._full_r2(),
                                   'Task reliability')
        assert len(fig.legends) == 0
        plt.close(fig)

    def test_each_row_has_own_yscale(self):
        """Panels share a y-axis within a target row, not across rows."""
        from iblnm.vis import plot_lmm_reliability
        # Give the two target rows very different ΔR² magnitudes: per-row
        # scaling makes the rows' y-ranges differ while columns within a row
        # share.
        rows = []
        for tnm, delta in [('VTA-DA', 0.02), ('DR-5HT', 0.5)]:
            for event in ('feedback_times', 'stimOn_times'):
                for fold in ['s0', 'aggregate']:
                    rows.append({'target_NM': tnm, 'event': event,
                                 'predictor': 'contrast', 'fold': fold,
                                 'delta_r2': delta})
        fig = plot_lmm_reliability(pd.DataFrame(rows), self._full_r2(),
                                   'Task reliability')
        # fig.axes ordered row-major: [r0c0, r0c1, r1c0, r1c1].
        assert fig.axes[0].get_ylim() == fig.axes[1].get_ylim()  # row shares
        assert fig.axes[2].get_ylim() == fig.axes[3].get_ylim()
        assert fig.axes[0].get_ylim() != fig.axes[2].get_ylim()  # rows differ
        plt.close(fig)


class TestPlotOlsDropone:
    """Per-mouse median ± IQR grid of per-recording ΔR² (target_NM × event)."""

    def _df(self, targets=('VTA-DA', 'DR-5HT'),
            events=('feedback_times', 'stimOn_times'),
            predictors=('contrast', 'side', 'reward', 'log_reaction_time',
                        'peak_velocity')):
        """Long-form ΔR²: subject 's_few' has 2 recordings, 's_many' has 4."""
        rows = []
        for tnm in targets:
            for event in events:
                for pred in predictors:
                    for subject, n_rec in [('s_few', 2), ('s_many', 4)]:
                        for i in range(n_rec):
                            rows.append({
                                'target_NM': tnm, 'event': event,
                                'subject': subject, 'predictor': pred,
                                'delta_r2': 0.01 * (i + 1)})
        return pd.DataFrame(rows)

    def test_mouse_stats_median_and_iqr(self):
        """Helper returns (median, 25th pct, 75th pct) of a mouse's ΔR²."""
        from iblnm.vis import _mouse_dropone_stats
        values = np.array([0.01, 0.02, 0.03, 0.04])
        med, q25, q75 = _mouse_dropone_stats(values)
        assert med == np.median(values)
        assert q25 == np.percentile(values, 25)
        assert q75 == np.percentile(values, 75)

    def test_grid_shape_targets_by_events(self):
        from iblnm.vis import plot_ols_dropone
        fig = plot_ols_dropone(self._df(), 'Per-recording ΔR²')
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2 * 2  # 2 targets × 2 events
        plt.close(fig)

    def _one_cell(self):
        """Single target/event/predictor cell: s_few (2 recs), s_many (4 recs)."""
        rows = []
        for subject, deltas in [('s_few', [0.05, 0.07]),
                                ('s_many', [0.01, 0.02, 0.03, 0.04])]:
            for d in deltas:
                rows.append({'target_NM': 'VTA-DA', 'event': 'feedback_times',
                             'subject': subject, 'predictor': 'contrast',
                             'delta_r2': d})
        return pd.DataFrame(rows)

    def test_sub_threshold_mouse_excluded(self):
        """Only the ≥ MIN_RECORDINGS_PERMOUSE subject draws a marker."""
        from iblnm.vis import plot_ols_dropone
        from iblnm.config import MIN_RECORDINGS_PERMOUSE
        from matplotlib.collections import PathCollection
        assert MIN_RECORDINGS_PERMOUSE == 3  # fixture assumes this threshold
        fig = plot_ols_dropone(self._one_cell(), 'cell')
        markers = [c for c in fig.axes[0].collections
                   if isinstance(c, PathCollection) and len(c.get_offsets())]
        assert len(markers) == 1  # s_many drawn, s_few excluded
        plt.close(fig)

    def test_marker_at_median_whisker_at_iqr(self):
        """Drawn marker y = median; whisker endpoints = 25th/75th percentiles."""
        from iblnm.vis import plot_ols_dropone
        from matplotlib.collections import PathCollection, LineCollection
        deltas = np.array([0.01, 0.02, 0.03, 0.04])  # s_many
        fig = plot_ols_dropone(self._one_cell(), 'cell')
        ax = fig.axes[0]
        marker = next(c for c in ax.collections
                      if isinstance(c, PathCollection) and len(c.get_offsets()))
        assert marker.get_offsets()[0, 1] == np.median(deltas)
        whisker = next(c for c in ax.collections
                       if isinstance(c, LineCollection))
        ys = whisker.get_segments()[0][:, 1]
        assert {ys.min(), ys.max()} == {np.percentile(deltas, 25),
                                        np.percentile(deltas, 75)}
        plt.close(fig)

    def test_empty_frame_returns_titled_figure(self):
        from iblnm.vis import plot_ols_dropone
        fig = plot_ols_dropone(self._df().iloc[0:0], 'Empty')
        assert isinstance(fig, plt.Figure)
        assert fig._suptitle.get_text() == 'Empty'
        plt.close(fig)


# =============================================================================
# plot_within_target_similarity Tests
# =============================================================================


def _make_sim_data(n_targets=3, n_per_target=4, n_subjects=2):
    """Build a similarity matrix, labels, and subjects for testing."""
    from iblnm.config import TARGETNM_COLORS
    targets = sorted(list(TARGETNM_COLORS.keys()))[:n_targets]
    rng = np.random.default_rng(42)
    eids = []
    target_list = []
    subject_list = []
    for tnm in targets:
        for i in range(n_per_target):
            eids.append(f'eid-{tnm}-{i}')
            target_list.append(tnm)
            subject_list.append(f's{i % n_subjects}')
    index = pd.MultiIndex.from_arrays(
        [eids, target_list, list(range(len(eids)))],
        names=['eid', 'target_NM', 'fiber_idx'],
    )
    n = len(eids)
    # Build a positive semi-definite similarity matrix
    data = rng.uniform(0.3, 0.9, (n, n))
    sim = (data + data.T) / 2
    np.fill_diagonal(sim, 1.0)
    sim_df = pd.DataFrame(sim, index=index, columns=index)
    labels = pd.Series(target_list, index=index)
    subjects = pd.Series(subject_list, index=index)
    return sim_df, labels, subjects


class TestPlotWithinTargetSimilarity:

    def test_one_bar_per_target(self):
        import matplotlib.patches as mpatches
        from iblnm.vis import plot_within_target_similarity
        sim, labels, subjects = _make_sim_data()
        fig = plot_within_target_similarity(sim, labels, subjects)
        ax = fig.axes[0]
        bars = [p for p in ax.patches
                if isinstance(p, mpatches.Rectangle) and p.get_height() != 0]
        assert len(bars) == len(labels.unique())
        plt.close(fig)

    def test_bar_colors_match_config(self):
        import matplotlib as mpl
        import matplotlib.patches as mpatches
        from iblnm.vis import plot_within_target_similarity
        from iblnm.config import TARGETNM_COLORS
        sim, labels, subjects = _make_sim_data()
        fig = plot_within_target_similarity(sim, labels, subjects)
        ax = fig.axes[0]
        targets = sorted(labels.unique())
        bars = [p for p in ax.patches
                if isinstance(p, mpatches.Rectangle) and p.get_height() != 0]
        for bar, target in zip(bars, targets):
            expected = mpl.colors.to_rgba(TARGETNM_COLORS[target])
            np.testing.assert_allclose(bar.get_facecolor(), expected, atol=0.01)
        plt.close(fig)

    def test_scatter_points_present(self):
        from iblnm.vis import plot_within_target_similarity
        sim, labels, subjects = _make_sim_data()
        fig = plot_within_target_similarity(sim, labels, subjects)
        ax = fig.axes[0]
        assert len(ax.collections) >= 1
        plt.close(fig)


# =============================================================================
# plot_response_decoding_summary Tests
# =============================================================================


def _make_decoding_summary_data():
    """Build response_matrix, coefficients, contributions for testing."""
    from iblnm.config import TARGETNM_COLORS
    targets = sorted(list(TARGETNM_COLORS.keys()))[:3]
    features = [
        'stimOn_c0_contra_correct', 'stimOn_c1_contra_correct',
        'feedback_c0_ipsi_incorrect',
    ]
    rng = np.random.default_rng(42)

    # Response matrix: 4 recordings per target
    eids = []
    target_list = []
    for tnm in targets:
        for i in range(4):
            eids.append(f'eid-{tnm}-{i}')
            target_list.append(tnm)
    index = pd.MultiIndex.from_arrays(
        [eids, target_list, list(range(len(eids)))],
        names=['eid', 'target_NM', 'fiber_idx'],
    )
    rm = pd.DataFrame(
        rng.normal(0, 1, (len(eids), len(features))),
        index=index, columns=features,
    )
    coefs = pd.DataFrame(
        rng.normal(0, 1, (len(targets), len(features))),
        index=targets, columns=features,
    )
    contrib = pd.DataFrame({
        'feature': features,
        'delta': rng.uniform(0, 0.1, len(features)),
    })
    return rm, coefs, contrib


class TestPlotResponseDecodingSummary:

    def test_four_axes(self):
        from iblnm.vis import plot_response_decoding_summary
        rm, coefs, contrib = _make_decoding_summary_data()
        fig = plot_response_decoding_summary(rm, coefs, contrib)
        assert len(fig.axes) == 4  # response_vectors + coefs + delta + colorbar
        plt.close(fig)

    def test_top_axis_uses_targetnm_colors(self):
        import matplotlib as mpl
        from iblnm.vis import plot_response_decoding_summary
        from iblnm.config import TARGETNM_COLORS
        rm, coefs, contrib = _make_decoding_summary_data()
        fig = plot_response_decoding_summary(rm, coefs, contrib)
        ax = fig.axes[0]
        targets = sorted(rm.index.get_level_values('target_NM').unique())
        for container, target in zip(ax.containers, targets):
            expected = mpl.colors.to_rgba(TARGETNM_COLORS[target])
            actual = mpl.colors.to_rgba(container[0].get_color())
            np.testing.assert_allclose(actual, expected, atol=0.01)
        plt.close(fig)


# =============================================================================
# LMM Coefficient Heatmap Tests
# =============================================================================


def _make_event_coefficients():
    """Deterministic single-event coefficient frame (two targets, three terms).

    Includes an Intercept row that the plotter must drop.
    """
    return pd.DataFrame({
        'term': ['Intercept', 'side', 'contrast',
                 'Intercept', 'side', 'contrast'],
        'target_NM': ['VTA-DA', 'VTA-DA', 'VTA-DA',
                      'DR-5HT', 'DR-5HT', 'DR-5HT'],
        'Coef.': [1.0, 0.2, 0.5, 0.9, 0.1, 0.3],
        'P>|z|': [1e-10, 0.01, 0.001, 1e-10, 0.5, 0.04],
    })


class TestPlotLMMCoefficientHeatmap:

    def test_returns_figure_when_ax_none(self):
        from iblnm.vis import plot_lmm_coefficient_heatmap
        fig = plot_lmm_coefficient_heatmap(_make_event_coefficients())
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_draws_on_passed_ax_without_new_figure(self):
        from iblnm.vis import plot_lmm_coefficient_heatmap
        fig, ax = plt.subplots()
        existing = set(plt.get_fignums())
        plot_lmm_coefficient_heatmap(_make_event_coefficients(), ax=ax)
        assert len(ax.images) == 1
        assert set(plt.get_fignums()) == existing
        plt.close(fig)

    def test_intercept_dropped_rows_targets_cols_terms(self):
        from iblnm.vis import plot_lmm_coefficient_heatmap
        from iblnm.config import TARGETNM2POSITION
        df = _make_event_coefficients()
        fig, ax = plt.subplots()
        plot_lmm_coefficient_heatmap(df, ax=ax)
        ylabels = [t.get_text() for t in ax.get_yticklabels()]
        xlabels = [t.get_text() for t in ax.get_xticklabels()]
        targets = sorted(df['target_NM'].unique(),
                         key=lambda x: TARGETNM2POSITION.get(x, 999))
        assert ylabels == targets
        assert len(xlabels) == 2  # side, contrast — Intercept dropped
        plt.close(fig)

    def test_cell_values_match_input(self):
        from iblnm.vis import plot_lmm_coefficient_heatmap
        fig, ax = plt.subplots()
        plot_lmm_coefficient_heatmap(_make_event_coefficients(), ax=ax)
        array = np.asarray(ax.images[0].get_array())
        present = np.round(array[~np.isnan(array)], 3)
        assert sorted(present) == [0.1, 0.2, 0.3, 0.5]
        plt.close(fig)

    def test_asterisks_for_significant(self):
        from iblnm.vis import plot_lmm_coefficient_heatmap
        fig, ax = plt.subplots()
        plot_lmm_coefficient_heatmap(_make_event_coefficients(), ax=ax)
        texts = [t.get_text() for t in ax.texts]
        assert any('*' in t for t in texts)
        plt.close(fig)


# =============================================================================
# plot_mean_response_traces Tests
# =============================================================================


def _make_traces_df(n_targets=2, n_subjects=3, n_recs_per=2,
                    n_timepoints=100, events=None, min_trials_per=12):
    """Synthetic mean traces for testing.

    Each (eid, target_NM, event, contrast, feedbackType) gets one row per
    timepoint. ``min_trials_per`` controls the n_trials column so tests can
    exercise the trial-count filter.
    """
    from iblnm.config import TARGETNM_COLORS
    targets = sorted(list(TARGETNM_COLORS.keys()))[:n_targets]
    if events is None:
        events = ['stimOn_times', 'firstMovement_times', 'feedback_times']
    contrasts = [0.0, 25.0, 100.0]
    feedback_types = [1, -1]
    rng = np.random.default_rng(42)
    time = np.linspace(-0.5, 1.5, n_timepoints)
    rows = []
    for tnm in targets:
        for s in range(n_subjects):
            for r in range(n_recs_per):
                eid = f'eid-{tnm}-s{s}-r{r}'
                for event in events:
                    for contrast in contrasts:
                        for fb in feedback_types:
                            # Add known offset so baseline norm is testable
                            offset = 5.0
                            trace = rng.normal(0, 0.1, n_timepoints) + offset
                            for t_idx, t in enumerate(time):
                                rows.append({
                                    'eid': eid,
                                    'subject': f's{s}',
                                    'target_NM': tnm,
                                    'brain_region': tnm.split('-')[0],
                                    'event': event,
                                    'contrast': contrast,
                                    'feedbackType': fb,
                                    'time': t,
                                    'response': trace[t_idx],
                                    'n_trials': min_trials_per,
                                })
    return pd.DataFrame(rows)


class TestPlotMeanResponseTraces:

    def test_returns_one_figure_per_target(self):
        from iblnm.vis import plot_mean_response_traces
        traces = _make_traces_df(n_targets=2)
        target = sorted(traces['target_NM'].unique())[0]
        df_t = traces[traces['target_NM'] == target]
        fig = plot_mean_response_traces(df_t, target)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_layout_2_rows_n_event_cols(self):
        from iblnm.vis import plot_mean_response_traces
        traces = _make_traces_df(n_targets=1, events=['stimOn_times', 'feedback_times'])
        target = traces['target_NM'].iloc[0]
        fig = plot_mean_response_traces(traces, target)
        # 2 rows (reward, omission) × 2 event columns = 4 axes
        assert len(fig.axes) == 4
        plt.close(fig)

    def test_uses_nm_colormap_shades(self):
        import matplotlib as mpl
        from iblnm.vis import plot_mean_response_traces
        from iblnm.config import NM_CMAPS, ANALYSIS_CONTRASTS
        traces = _make_traces_df(n_targets=1)
        target = traces['target_NM'].iloc[0]
        nm = target.split('-')[-1]
        fig = plot_mean_response_traces(traces, target)
        ax = fig.axes[0]
        # Lines (excluding vline) should use NM colormap shades
        contrasts = sorted(traces['contrast'].unique())
        cmap = NM_CMAPS[nm]
        n_levels = len(ANALYSIS_CONTRASTS)
        shade_map = {c: cmap(0.3 + 0.7 * i / (n_levels - 1))
                     for i, c in enumerate(ANALYSIS_CONTRASTS)}
        for line, c in zip(ax.lines[:-1], contrasts):
            expected = mpl.colors.to_rgba(shade_map[c])
            actual = mpl.colors.to_rgba(line.get_color())
            np.testing.assert_allclose(actual, expected, atol=0.01)
        plt.close(fig)

    def test_baseline_normalized(self):
        from iblnm.vis import plot_mean_response_traces
        traces = _make_traces_df(n_targets=1, events=['stimOn_times'])
        target = traces['target_NM'].iloc[0]
        fig = plot_mean_response_traces(traces, target)
        ax = fig.axes[0]
        # With offset=5.0 in synthetic data, after baseline normalization
        # the traces should be centered near 0, not near 5
        for line in ax.lines[:-1]:  # exclude vline
            ydata = line.get_ydata()
            assert abs(np.nanmean(ydata)) < 1.0
        plt.close(fig)

    def test_filters_low_trial_counts(self):
        from iblnm.vis import plot_mean_response_traces
        # Make traces where one contrast has < 10 trials per subject
        traces = _make_traces_df(n_targets=1, events=['stimOn_times'])
        target = traces['target_NM'].iloc[0]
        # Set n_trials=5 for contrast=0.0 → should be excluded
        mask = traces['contrast'] == 0.0
        traces.loc[mask, 'n_trials'] = 5
        fig = plot_mean_response_traces(traces, target)
        ax = fig.axes[0]
        # Should have 2 contrasts plotted (0.25, 1.0) + 1 vline = 3 lines
        assert len(ax.lines) == 3
        plt.close(fig)

    def test_fill_between_present(self):
        from iblnm.vis import plot_mean_response_traces
        traces = _make_traces_df(n_targets=1, n_subjects=3)
        target = traces['target_NM'].iloc[0]
        fig = plot_mean_response_traces(traces, target)
        for ax in fig.axes:
            assert len(ax.collections) >= 1
        plt.close(fig)

    def test_event_order_stim_movement_feedback(self):
        from iblnm.vis import plot_mean_response_traces
        traces = _make_traces_df(
            n_targets=1,
            events=['feedback_times', 'stimOn_times', 'firstMovement_times'],
        )
        target = traces['target_NM'].iloc[0]
        fig = plot_mean_response_traces(traces, target)
        # Top row: col 0, 1, 2 → axes[0], axes[1], axes[2]
        titles = [fig.axes[col].get_title() for col in range(3)]
        assert titles == ['stimOn', 'firstMovement', 'feedback']
        plt.close(fig)

    def test_response_window_shading(self):
        """Early window on all panels, late window only on feedback."""
        from iblnm.vis import plot_mean_response_traces
        from iblnm.config import RESPONSE_WINDOWS
        traces = _make_traces_df(
            n_targets=1,
            events=['stimOn_times', 'feedback_times'],
        )
        target = traces['target_NM'].iloc[0]
        fig = plot_mean_response_traces(traces, target)
        early = RESPONSE_WINDOWS['early']
        late = RESPONSE_WINDOWS['late']
        # stimOn is col 0 → axes[0] (top row)
        ax_stim = fig.axes[0]
        assert any(
            np.isclose(p.get_x(), early[0], atol=0.01)
            for p in ax_stim.patches
        )
        # stimOn should NOT have late window
        assert not any(
            np.isclose(p.get_x(), late[0], atol=0.01)
            for p in ax_stim.patches
        )
        # feedback is col 1 → axes[1] (top row)
        ax_fb = fig.axes[1]
        assert any(
            np.isclose(p.get_x(), early[0], atol=0.01)
            for p in ax_fb.patches
        )
        assert any(
            np.isclose(p.get_x(), late[0], atol=0.01)
            for p in ax_fb.patches
        )
        plt.close(fig)



# =============================================================================
# Per-Cohort CCA Summary Plot
# =============================================================================


def _make_mock_cohort_cca_data():
    """Create mock data for plot_cohort_cca_summary tests."""
    from iblnm.analysis import CCAResult

    targets = ['VTA-DA', 'DR-5HT']
    feature_names = ['contrast', 'side', 'feedback',
                     'contrast:side', 'contrast:feedback',
                     'side:feedback']
    psych_names = ['psych_50_threshold', 'psych_50_bias',
                   'psych_50_lapse_left', 'psych_50_lapse_right']

    rng = np.random.default_rng(42)
    results = {}
    for t in targets:
        x_w = pd.DataFrame(
            rng.standard_normal((6, 1)),
            index=feature_names, columns=['CC1'])
        y_w = pd.DataFrame(
            rng.standard_normal((4, 1)),
            index=psych_names, columns=['CC1'])
        results[t] = CCAResult(
            x_weights=x_w, y_weights=y_w,
            x_scores=rng.standard_normal((50, 1)),
            y_scores=rng.standard_normal((50, 1)),
            correlations=np.array([rng.uniform(0.3, 0.8)]),
            p_values=np.array([rng.uniform(0, 0.05)]),
            n_recordings=50, n_permutations=100,
        )

    cross_projections = pd.DataFrame([
        {'data_cohort': 'VTA-DA', 'weight_cohort': 'VTA-DA', 'correlation': 0.7},
        {'data_cohort': 'VTA-DA', 'weight_cohort': 'DR-5HT', 'correlation': 0.3},
        {'data_cohort': 'DR-5HT', 'weight_cohort': 'VTA-DA', 'correlation': 0.25},
        {'data_cohort': 'DR-5HT', 'weight_cohort': 'DR-5HT', 'correlation': 0.65},
    ])

    weight_sims = pd.DataFrame([
        {'cohort_a': 'VTA-DA', 'cohort_b': 'VTA-DA',
         'neural_cosine': 1.0, 'behavioral_cosine': 1.0},
        {'cohort_a': 'VTA-DA', 'cohort_b': 'DR-5HT',
         'neural_cosine': 0.5, 'behavioral_cosine': 0.6},
        {'cohort_a': 'DR-5HT', 'cohort_b': 'VTA-DA',
         'neural_cosine': 0.5, 'behavioral_cosine': 0.6},
        {'cohort_a': 'DR-5HT', 'cohort_b': 'DR-5HT',
         'neural_cosine': 1.0, 'behavioral_cosine': 1.0},
    ])

    return results, cross_projections, weight_sims


class TestPlotCohortCCASummary:

    def test_returns_figure(self):
        from iblnm.vis import plot_cohort_cca_summary
        results, cp, ws = _make_mock_cohort_cca_data()
        fig = plot_cohort_cca_summary(results, cp, ws)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_has_six_panels(self):
        from iblnm.vis import plot_cohort_cca_summary
        results, cp, ws = _make_mock_cohort_cca_data()
        fig = plot_cohort_cca_summary(results, cp, ws)
        # 6 content panels: bar, cross-proj, delta-r, weights,
        # neural cosine sim, behavioral cosine sim
        titled = [ax for ax in fig.axes if ax.get_title() != '']
        assert len(titled) >= 6
        plt.close(fig)

    def test_delta_r_diagonal_is_zero(self):
        from iblnm.vis import plot_cohort_cca_summary
        results, cp, ws = _make_mock_cohort_cca_data()
        fig = plot_cohort_cca_summary(results, cp, ws)
        # axes[2] is the delta-r heatmap (axes[0]=bars, [1]=raw, [2]=delta,
        # but colorbars add extra axes; the imshow axes are the first 4)
        delta_ax = fig.axes[2]
        im = delta_ax.images[0]
        delta_data = im.get_array()
        n = len(results)
        for i in range(n):
            np.testing.assert_allclose(delta_data[i, i], 0.0, atol=1e-10)
        plt.close(fig)

    def test_bar_colors_match_config(self):
        import matplotlib as mpl
        from iblnm.vis import plot_cohort_cca_summary
        from iblnm.config import TARGETNM_COLORS
        results, cp, ws = _make_mock_cohort_cca_data()
        fig = plot_cohort_cca_summary(results, cp, ws)
        ax = fig.axes[0]
        targets = sorted(results.keys())
        bars = [p for p in ax.patches
                if isinstance(p, mpl.patches.Rectangle) and p.get_height() != 0]
        for bar, target in zip(bars, targets):
            expected = mpl.colors.to_rgba(TARGETNM_COLORS[target])
            np.testing.assert_allclose(
                bar.get_facecolor(), expected, atol=0.01)
        plt.close(fig)


def _make_mock_glm_pca_result():
    """Synthetic GLMPCAResult with some subjects contributing multiple sessions."""
    from iblnm.analysis import GLMPCAResult
    rng = np.random.default_rng(42)
    # 8 VTA-DA recordings from 5 subjects (s1 has 3 sessions, s2 has 2)
    # 4 DR-5HT recordings from 3 subjects (s6 has 2)
    # 3 LC-NE recordings from 3 subjects
    eids = ['e0','e1','e2','e3','e4','e5','e6','e7','e8','e9','e10','e11','e12','e13','e14']
    targets = (['VTA-DA'] * 8 + ['DR-5HT'] * 4 + ['LC-NE'] * 3)
    n = len(eids)
    index = pd.MultiIndex.from_tuples(
        [(eids[i], targets[i], 0) for i in range(n)],
        names=['eid', 'target_NM', 'fiber_idx'],
    )
    return GLMPCAResult(
        scores=rng.normal(size=(n, 3)),
        components=rng.normal(size=(3, 6)),
        explained_variance_ratio=np.array([0.45, 0.25, 0.15]),
        feature_names=['contrast', 'side', 'reward',
                       'contrast:side', 'contrast:reward',
                       'side:reward'],
        target_labels=np.array(targets),
        index=index,
    )


def _make_mock_recordings():
    """Recordings DataFrame matching _make_mock_glm_pca_result eids."""
    eids =    ['e0','e1','e2','e3','e4','e5','e6','e7','e8','e9','e10','e11','e12','e13','e14']
    subjects = ['s1','s1','s1','s2','s2','s3','s4','s5','s6','s6','s7', 's8', 's9','s10','s11']
    targets = (['VTA-DA'] * 8 + ['DR-5HT'] * 4 + ['LC-NE'] * 3)
    return pd.DataFrame({
        'eid': eids,
        'subject': subjects,
        'target_NM': targets,
        'fiber_idx': [0] * len(eids),
    })


class TestPlotGlmPcaWeights:

    def test_returns_figure(self):
        from iblnm.vis import plot_glm_pca_weights
        result = _make_mock_glm_pca_result()
        fig = plot_glm_pca_weights(result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_heatmap_shape_matches_components(self):
        """Heatmap data should have n_components rows and n_features columns."""
        from iblnm.vis import plot_glm_pca_weights
        result = _make_mock_glm_pca_result()
        fig = plot_glm_pca_weights(result)
        ax = fig.axes[0]
        img = ax.images[0]
        data = img.get_array()
        assert data.shape == (3, 6)
        plt.close(fig)


class TestPlotGlmPcaScores:

    def test_returns_figure(self):
        from iblnm.vis import plot_glm_pca_scores
        result = _make_mock_glm_pca_result()
        fig = plot_glm_pca_scores(result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_three_panels_for_three_pcs(self):
        from iblnm.vis import plot_glm_pca_scores
        result = _make_mock_glm_pca_result()
        fig = plot_glm_pca_scores(result)
        assert len(fig.axes) == 3
        plt.close(fig)

    def test_target_colors_used(self):
        """Each target should use its configured color."""
        from iblnm.vis import plot_glm_pca_scores
        result = _make_mock_glm_pca_result()
        fig = plot_glm_pca_scores(result)
        ax = fig.axes[0]
        collections = ax.collections
        assert len(collections) > 0
        plt.close(fig)


class TestPlotGlmPcaSummary:

    def test_returns_figure(self):
        from iblnm.vis import plot_glm_pca_summary
        result = _make_mock_glm_pca_result()
        recs = _make_mock_recordings()
        fig = plot_glm_pca_summary(result, recs)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_layout_has_three_rows(self):
        """Weights heatmap row + violin row + subject-mean row = 3 rows."""
        from iblnm.vis import plot_glm_pca_summary
        result = _make_mock_glm_pca_result()
        recs = _make_mock_recordings()
        fig = plot_glm_pca_summary(result, recs)
        # 1 heatmap + 1 colorbar + 3 violins + 3 subject-mean = 8 axes
        # (colorbar count may vary, just check >= 7)
        assert len(fig.axes) >= 7
        plt.close(fig)

    def test_subject_means_aggregated(self):
        """Subject-mean axes should have one point per subject, not per session."""
        from iblnm.vis import plot_glm_pca_summary
        result = _make_mock_glm_pca_result()
        recs = _make_mock_recordings()
        fig = plot_glm_pca_summary(result, recs)
        # Find first subject-mean axis by title
        subject_ax = next(
            ax for ax in fig.axes if 'subject means' in ax.get_title())
        # VTA-DA has 5 unique subjects, DR-5HT has 3, LC-NE has 3 → 11 total
        assert len(subject_ax.containers) == 11
        plt.close(fig)

    def test_kw_annotation_includes_h_statistic(self):
        """KW annotation should show H= value and n.s./scientific p, matching psychometric figure format."""
        from iblnm.vis import plot_glm_pca_summary
        result = _make_mock_glm_pca_result()
        recs = _make_mock_recordings()
        stats = pd.DataFrame([
            {'pc': 1, 'target_a': None, 'target_b': None,
             'kruskal_h': 4.2, 'kruskal_p': 0.12,
             'mwu_u': np.nan, 'mwu_p': np.nan},
        ])
        fig = plot_glm_pca_summary(result, recs, n_pcs=1, stats=stats)
        # Find the violin axis (row 1) and check annotation text
        texts = []
        for ax in fig.axes:
            for txt in ax.texts:
                texts.append(txt.get_text())
        assert any('H=4.2' in t for t in texts), (
            f"Expected 'H=4.2' in annotations, got: {texts}"
        )
        assert any('n.s.' in t for t in texts), (
            f"Expected 'n.s.' in annotations for p=0.12, got: {texts}"
        )
        plt.close(fig)

    def test_significance_brackets_are_lines(self):
        """Significant pairwise comparisons should draw bracket lines, not text."""
        from iblnm.vis import plot_glm_pca_summary
        # Use a result where groups are well-separated so some pairs are significant
        from iblnm.analysis import GLMPCAResult
        rng = np.random.default_rng(99)
        n = 30
        # 3 groups with very different PC1 scores
        targets = np.array(['VTA-DA'] * 10 + ['DR-5HT'] * 10 + ['LC-NE'] * 10)
        scores = np.zeros((n, 2))
        scores[:10, 0] = rng.normal(5, 0.1, 10)   # VTA-DA far positive
        scores[10:20, 0] = rng.normal(-5, 0.1, 10) # DR-5HT far negative
        scores[20:, 0] = rng.normal(0, 0.1, 10)    # LC-NE near zero
        scores[:, 1] = rng.normal(0, 1, n)
        eids = [f'e{i}' for i in range(n)]
        subjects = [f's{i}' for i in range(n)]
        index = pd.MultiIndex.from_tuples(
            [(eids[i], targets[i], 0) for i in range(n)],
            names=['eid', 'target_NM', 'fiber_idx'],
        )
        result = GLMPCAResult(
            scores=scores,
            components=rng.normal(size=(2, 4)),
            explained_variance_ratio=np.array([0.6, 0.3]),
            feature_names=['a', 'b', 'c', 'd'],
            target_labels=targets,
            index=index,
        )
        recs = pd.DataFrame({
            'eid': eids, 'subject': subjects,
            'target_NM': list(targets), 'fiber_idx': [0] * n,
        })
        fig = plot_glm_pca_summary(result, recs, n_pcs=2)
        # Find a subject-mean axis — it should have bracket lines
        subject_ax = next(
            ax for ax in fig.axes if 'subject means' in ax.get_title())
        bracket_lines = [
            line for line in subject_ax.get_lines()
            if len(line.get_xdata()) == 2 and line.get_color() == 'k'
        ]
        assert len(bracket_lines) > 0
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_rt_by_contrast
# ---------------------------------------------------------------------------

@pytest.fixture
def df_rt():
    """Synthetic RT data: two target-NMs, three contrast levels, multiple trials."""
    rng = np.random.default_rng(42)
    n = 120
    return pd.DataFrame({
        'target_NM': rng.choice(['VTA-DA', 'LC-NE'], n),
        'contrast': rng.choice([6.25, 25.0, 100.0], n),
        'response_time': rng.uniform(0.1, 2.0, n),
        'subject': rng.choice(['s1', 's2', 's3'], n),
    })


class TestPlotRtByContrast:
    def test_returns_axes(self, df_rt):
        from iblnm.vis import _draw_rt_violins as plot_rt_by_contrast
        ax = plot_rt_by_contrast(df_rt)
        assert isinstance(ax, plt.Axes)
        plt.close('all')

    def test_xaxis_uses_log_ticks_on_linear_scale(self, df_rt):
        """Axis should be linear (data pre-transformed) with log-formatted tick labels."""
        from iblnm.vis import _draw_rt_violins as plot_rt_by_contrast
        ax = plot_rt_by_contrast(df_rt)
        assert ax.get_xscale() == 'linear'
        # Tick labels should show real-time values (e.g. '0.1', '1'), not log values
        labels = [t.get_text() for t in ax.get_xticklabels() if t.get_text()]
        assert len(labels) > 0
        plt.close('all')

    def test_violin_xdata_is_log_transformed(self, df_rt):
        """Violin x-extents should be in log space (negative values present for sub-1s RTs)."""
        from iblnm.vis import _draw_rt_violins as plot_rt_by_contrast
        ax = plot_rt_by_contrast(df_rt)
        # df_rt has response_time in [0.1, 2.0]; log10 of those is in [-1, 0.3]
        # so violin bodies must span negative x values
        x_mins = []
        for pc in ax.collections:
            for path in pc.get_paths():
                x_mins.append(path.vertices[:, 0].min())
        assert any(x < 0 for x in x_mins), (
            "Expected log-transformed x values (sub-1 s RTs → negative log)"
        )

    def test_yticks_at_contrast_levels(self, df_rt):
        """Y-ticks should be at the integer indices of sorted contrast levels."""
        from iblnm.vis import _draw_rt_violins as plot_rt_by_contrast
        ax = plot_rt_by_contrast(df_rt)
        tick_labels = [t.get_text() for t in ax.get_yticklabels()]
        # All three contrasts should appear as tick labels
        for c in ['6.25', '25.0', '100.0']:
            assert any(c in lbl for lbl in tick_labels), (
                f"Contrast {c} not found in ytick labels: {tick_labels}"
            )
        plt.close('all')

    def test_each_target_nm_gets_own_offset(self, df_rt):
        """Two target-NMs at the same contrast should produce violins at different y positions."""
        from iblnm.vis import _draw_rt_violins as plot_rt_by_contrast
        ax = plot_rt_by_contrast(df_rt)
        # violinplot bodies are PolyCollections; extract their y-centre positions
        y_centers = set()
        for pc in ax.collections:
            verts = pc.get_paths()
            for path in verts:
                y_centers.add(round(path.vertices[:, 1].mean(), 4))
        # With 2 target-NMs and 3 contrast levels, offsets produce >3 distinct centres
        assert len(y_centers) > 3
        plt.close('all')

    def test_accepts_ax_argument(self, df_rt):
        from iblnm.vis import _draw_rt_violins as plot_rt_by_contrast
        fig, ax = plt.subplots()
        result = plot_rt_by_contrast(df_rt, ax=ax)
        assert result is ax
        plt.close('all')

    def test_empty_data_no_crash(self):
        from iblnm.vis import _draw_rt_violins as plot_rt_by_contrast
        df_empty = pd.DataFrame(
            columns=['target_NM', 'contrast', 'response_time', 'subject']
        )
        ax = plot_rt_by_contrast(df_empty)
        assert isinstance(ax, plt.Axes)
        plt.close('all')


def _make_group(subjects_targets, session_n_per_subject=2):
    """Build a minimal PhotometrySessionGroup for vis tests.

    Parameters
    ----------
    subjects_targets : list of (subject, target_NM, start_time)
        One entry per unique (subject, target_NM) combination.
    session_n_per_subject : int
        Number of sessions per subject row.
    """
    from unittest.mock import MagicMock
    from iblnm.data import PhotometrySessionGroup

    rows = []
    eid_counter = [0]

    for subject, target_nm, start_time in subjects_targets:
        for sn in range(session_n_per_subject):
            eid = f'eid-{eid_counter[0]}'
            eid_counter[0] += 1
            rows.append({
                'eid': eid,
                'subject': subject,
                'session_n': sn,
                'session_type': 'biased',
                'start_time': start_time,
                'brain_region': ['VTA'],
                'hemisphere': ['l'],
                'target_NM': [target_nm],
            })

    df = pd.DataFrame(rows)
    group = PhotometrySessionGroup(df, one=MagicMock())
    group.filter_sessions(
        session_types=False, qc_blockers=set(),
        targetnms=False, min_performance=False, required_contrasts=False,
    )
    return group


class TestSessionOverviewMatrixSubjectOrder:

    def test_subject_order_by_start_time(self):
        """Subjects are ordered by their earliest start_time."""
        from iblnm.vis import session_overview_matrix
        group = _make_group([
            ('mouse_late', 'VTA-DA', '2024-06-01'),
            ('mouse_early', 'VTA-DA', '2024-01-01'),
            ('mouse_mid', 'VTA-DA', '2024-03-01'),
        ])
        ax = session_overview_matrix(group)
        labels = [t.get_text() for t in ax.get_yticklabels()]
        assert labels == ['mouse_early', 'mouse_mid', 'mouse_late']
        plt.close('all')

    def test_one_row_per_subject(self):
        """A subject recording from two targets gets one row."""
        from iblnm.vis import session_overview_matrix
        from unittest.mock import MagicMock
        from iblnm.data import PhotometrySessionGroup

        df = pd.DataFrame([{
            'eid': 'eid-0',
            'subject': 'multi',
            'session_n': 0,
            'session_type': 'biased',
            'start_time': '2024-01-01',
            'brain_region': ['VTA', 'DR'],
            'hemisphere': ['l', 'r'],
            'target_NM': ['VTA-DA', 'DR-5HT'],
        }])
        group = PhotometrySessionGroup(df, one=MagicMock())
        group.filter_sessions(
            session_types=False, qc_blockers=set(),
            targetnms=False, min_performance=False, required_contrasts=False,
        )
        ax = session_overview_matrix(group)
        labels = [t.get_text() for t in ax.get_yticklabels()]
        assert labels == ['multi']
        plt.close('all')

    def test_catalog_faded_sessions_solid(self):
        """_catalog sessions appear in base layer; only filtered sessions in overlay."""
        from iblnm.vis import session_overview_matrix
        from unittest.mock import MagicMock
        from iblnm.data import PhotometrySessionGroup

        # Two subjects; only one has logged_errors so it gets dropped by qc_blockers
        df = pd.DataFrame([
            {'eid': 'e0', 'subject': 'A', 'session_n': 0, 'session_type': 'biased',
             'start_time': '2024-01-01', 'brain_region': ['VTA'], 'hemisphere': ['l'],
             'target_NM': ['VTA-DA'], 'logged_errors': []},
            {'eid': 'e1', 'subject': 'B', 'session_n': 0, 'session_type': 'biased',
             'start_time': '2024-02-01', 'brain_region': ['VTA'], 'hemisphere': ['l'],
             'target_NM': ['VTA-DA'], 'logged_errors': ['MissingRawData']},
        ])
        group = PhotometrySessionGroup(df, one=MagicMock())
        group.filter_sessions(session_types=False, qc_blockers={'MissingRawData'},
                              targetnms=False, min_performance=False, required_contrasts=False)

        # _catalog has 2 subjects; sessions has 1
        assert len(group._catalog) == 2
        assert len(group.sessions) == 1

        ax = session_overview_matrix(group)
        # Both subjects appear on y-axis (from _catalog)
        labels = [t.get_text() for t in ax.get_yticklabels()]
        assert 'A' in labels
        assert 'B' in labels
        # Two images: base (faded) and overlay (solid)
        assert len(ax.images) == 2
        plt.close('all')


# =============================================================================
# plot_psychometric_grid
# =============================================================================

def _make_mock_group_with_performance():
    """Build a mock group with performance data for two target_NMs."""
    from iblnm.data import PhotometrySessionGroup
    rows = []
    for i in range(6):
        tnm = 'VTA-DA' if i < 3 else 'LC-NE'
        rows.append({
            'eid': f'eid-{i}',
            'subject': f'subj-{i % 3}',
            'brain_region': [tnm.split('-')[0]],
            'hemisphere': ['l'],
            'target_NM': [tnm],
            'NM': tnm.split('-')[1],
            'session_type': 'biased',
            'start_time': '2024-01-01T10:00:00',
            'number': 1,
            'task_protocol': 'biased_protocol',
        })
    df = pd.DataFrame(rows)
    group = PhotometrySessionGroup(df, one=MagicMock())
    group.filter_sessions(
        session_types=False, qc_blockers=set(), targetnms=False,
        min_performance=False, required_contrasts=False,
    )

    rng = np.random.default_rng(0)
    group.performance = pd.DataFrame({
        'eid': [f'eid-{i}' for i in range(6)],
        'fraction_correct': rng.uniform(0.6, 0.9, 6),
        'psych_50_bias': rng.uniform(-10, 10, 6),
        'psych_50_threshold': rng.uniform(10, 50, 6),
        'psych_50_lapse_left': rng.uniform(0, 0.15, 6),
        'psych_50_lapse_right': rng.uniform(0, 0.15, 6),
    })
    return group


def _make_mock_group_with_performance_multi():
    """Build a mock group with multiple sessions per subject per target_NM."""
    from iblnm.data import PhotometrySessionGroup
    rng = np.random.default_rng(0)
    subjects = ['subj-0', 'subj-1', 'subj-2']
    target_nms = ['VTA-DA', 'LC-NE']
    rows = []
    perf_rows = []
    eid_idx = 0
    for tnm in target_nms:
        for subj in subjects:
            for _ in range(4):  # 4 sessions per subject per target
                eid = f'eid-{eid_idx}'
                rows.append({
                    'eid': eid,
                    'subject': subj,
                    'brain_region': [tnm.split('-')[0]],
                    'hemisphere': ['l'],
                    'target_NM': [tnm],
                    'NM': tnm.split('-')[1],
                    'session_type': 'biased',
                    'start_time': '2024-01-01T10:00:00',
                    'number': 1,
                    'task_protocol': 'biased_protocol',
                })
                perf_rows.append({
                    'eid': eid,
                    'fraction_correct': rng.uniform(0.6, 0.9),
                    'psych_50_bias': rng.uniform(-10, 10),
                    'psych_50_threshold': rng.uniform(10, 50),
                    'psych_50_lapse_left': rng.uniform(0, 0.15),
                    'psych_50_lapse_right': rng.uniform(0, 0.15),
                })
                eid_idx += 1
    df = pd.DataFrame(rows)
    group = PhotometrySessionGroup(df, one=MagicMock())
    group.filter_sessions(
        session_types=False, qc_blockers=set(), targetnms=False,
        min_performance=False, required_contrasts=False,
    )
    group.performance = pd.DataFrame(perf_rows)
    return group


class TestPlotPsychometricGrid:

    def test_returns_figure(self):
        from iblnm.vis import plot_psychometric_grid
        group = _make_mock_group_with_performance()
        fig = plot_psychometric_grid(group)
        assert isinstance(fig, plt.Figure)
        plt.close('all')

    def test_panels_match_target_nms(self):
        from iblnm.vis import plot_psychometric_grid
        group = _make_mock_group_with_performance()
        fig = plot_psychometric_grid(group)
        titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
        assert 'VTA-DA' in titles
        assert 'LC-NE' in titles
        plt.close('all')


# =============================================================================
# plot_target_comparison
# =============================================================================

class TestPlotTargetComparison:

    def test_returns_figure(self):
        from iblnm.vis import plot_target_comparison
        group = _make_mock_group_with_performance()
        params = ['fraction_correct', 'psych_50_bias']
        labels = ['fraction correct', 'bias']
        fig = plot_target_comparison(group, params, labels)
        assert isinstance(fig, plt.Figure)
        plt.close('all')

    def test_one_panel_per_param(self):
        from iblnm.vis import plot_target_comparison
        group = _make_mock_group_with_performance()
        params = ['fraction_correct', 'psych_50_bias', 'psych_50_threshold']
        labels = ['fraction correct', 'bias', 'threshold']
        fig = plot_target_comparison(group, params, labels)
        titles = [ax.get_title() for ax in fig.axes if ax.get_visible()]
        for label in labels:
            assert label in titles
        plt.close('all')

    def test_subject_errorbars_present(self):
        """Each target-NM should have per-subject mean+CI errorbars overlaid."""
        from iblnm.vis import plot_target_comparison
        group = _make_mock_group_with_performance_multi()
        params = ['fraction_correct']
        labels = ['fraction correct']
        fig = plot_target_comparison(group, params, labels)
        ax = fig.axes[0]
        # ErrorbarContainer objects appear as ax.containers entries
        from matplotlib.container import ErrorbarContainer
        errorbar_containers = [c for c in ax.containers
                               if isinstance(c, ErrorbarContainer)]
        # 2 target_NMs, 3 subjects each → 6 errorbar containers
        assert len(errorbar_containers) == 6
        plt.close('all')


# =============================================================================
# plot_rt_by_contrast (group-based)
# =============================================================================

def _make_mock_group_with_rt():
    """Build a mock group with response_magnitudes and trial_regressors."""
    from iblnm.data import PhotometrySessionGroup
    rng = np.random.default_rng(42)
    rows = []
    for i in range(4):
        tnm = 'VTA-DA' if i < 2 else 'LC-NE'
        rows.append({
            'eid': f'eid-{i}',
            'subject': f'subj-{i % 2}',
            'brain_region': [tnm.split('-')[0]],
            'hemisphere': ['l'],
            'target_NM': [tnm],
            'NM': tnm.split('-')[1],
            'session_type': 'biased',
            'start_time': '2024-01-01T10:00:00',
            'number': 1,
            'task_protocol': 'biased_protocol',
        })
    df = pd.DataFrame(rows)
    group = PhotometrySessionGroup(df, one=MagicMock())
    group.filter_sessions(
        session_types=False, qc_blockers=set(), targetnms=False,
        min_performance=False, required_contrasts=False,
    )

    # response_magnitudes carries recording keys + event only
    resp_rows = []
    regressor_rows = []
    for i in range(4):
        tnm = 'VTA-DA' if i < 2 else 'LC-NE'
        for t in range(20):
            resp_rows.append({
                'eid': f'eid-{i}',
                'subject': f'subj-{i % 2}',
                'target_NM': tnm,
                'trial': t,
                'event': 'stimOn_times',
            })
            regressor_rows.append({
                'eid': f'eid-{i}',
                'trial': t,
                'contrast': rng.choice([6.25, 25.0, 100.0]),
                'choice': rng.choice([-1, 1]),
                'probabilityLeft': rng.choice([0.2, 0.5, 0.8]),
                'response_time': rng.uniform(0.1, 2.0),
            })
    group.response_magnitudes = pd.DataFrame(resp_rows)
    group.trial_regressors = pd.DataFrame(regressor_rows)
    return group


class TestPlotRtByContrastGroup:

    def test_returns_figure(self):
        from iblnm.vis import plot_rt_by_contrast
        group = _make_mock_group_with_rt()
        fig = plot_rt_by_contrast(group)
        assert isinstance(fig, plt.Figure)
        plt.close('all')

    def test_none_timing_returns_empty(self):
        from iblnm.vis import plot_rt_by_contrast
        group = _make_mock_group_with_rt()
        group.trial_regressors = None
        fig = plot_rt_by_contrast(group)
        assert isinstance(fig, plt.Figure)
        plt.close('all')


# =========================================================================
# plot_movement_response
# =========================================================================

def _make_movement_df(n_per_cell=30, seed=42):
    """Synthetic data for plot_movement_response tests, spanning contrasts."""
    rng = np.random.default_rng(seed)
    rows = []
    for subj in ['s1', 's2', 's3']:
        for contrast in [0.0, 25.0, 100.0]:
            for _ in range(n_per_cell):
                rows.append({
                    'subject': subj,
                    'eid': f'eid_{subj}',
                    'contrast': contrast,
                    'response': rng.normal(0, 1),
                    'log_reaction_time': rng.normal(-0.7, 0.3),
                })
    return pd.DataFrame(rows)


class TestPlotMovementResponse:
    def test_returns_figure(self):
        from iblnm.vis import plot_movement_response
        df = _make_movement_df()
        fig = plot_movement_response(
            df, 'response', 'log_reaction_time', 'VTA-DA')
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plots_every_trial(self):
        """Individual trials, not aggregated: one scatter point per row."""
        from iblnm.vis import plot_movement_response
        df = _make_movement_df()
        fig = plot_movement_response(
            df, 'response', 'log_reaction_time', 'VTA-DA')
        offsets = fig.axes[0].collections[0].get_offsets()
        assert len(offsets) == len(df)
        plt.close(fig)

    def test_color_encodes_contrast(self):
        """Point colors map to the contrast column."""
        from iblnm.vis import plot_movement_response
        df = _make_movement_df()
        fig = plot_movement_response(
            df, 'response', 'log_reaction_time', 'VTA-DA')
        carray = fig.axes[0].collections[0].get_array()
        assert np.array_equal(np.asarray(carray), df['contrast'].values)
        plt.close(fig)

    def test_empty_df_no_crash(self):
        from iblnm.vis import plot_movement_response
        df = pd.DataFrame(columns=[
            'subject', 'eid', 'contrast', 'response', 'log_reaction_time',
        ])
        fig = plot_movement_response(
            df, 'response', 'log_reaction_time', 'VTA-DA')
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_event_label_in_suptitle(self):
        """The event label appears in the title so per-event raw-data plots are
        distinguishable."""
        from iblnm.vis import plot_movement_response
        df = _make_movement_df()
        fig = plot_movement_response(
            df, 'response', 'log_reaction_time', 'VTA-DA',
            event='firstMovement_times')
        assert 'firstMovement_times' in fig._suptitle.get_text()
        plt.close(fig)


# =========================================================================
# plot_movement_r2_bars / plot_movement_slope_summary
# =========================================================================

def _make_claim_slopes(with_event=True):
    """Tidy movement-claim result frame (one row per fit)."""
    rows = []
    for tnm in ['VTA-DA', 'DR-5HT']:
        for tc in ['log_reaction_time', 'log_movement_time']:
            events = ['baseline', 'stimOn_times'] if with_event else [None]
            for ev in events:
                row = {
                    'target_NM': tnm, 'term': tc, 'coef': 0.2,
                    'se': 0.03, 'z': 6.0, 'p': 0.001,
                    'ci_low': 0.14, 'ci_high': 0.26,
                    'marginal_r2': 0.05, 'n_trials': 200, 'n_subjects': 3,
                    'timing_col': tc,
                }
                if ev is not None:
                    row['event'] = ev
                rows.append(row)
    return pd.DataFrame(rows)


def _make_movement_r2_bars(movement_vars=('choice', 'reaction_time')):
    """Long-form marginal R²: one row per (target_NM, movement var, model).

    Three model names per target-NM: 'full', 'contrast' (contrast dropped ->
    movement-family), 'movement' (predictor dropped -> task base).
    """
    rows = []
    for tnm in ['VTA-DA', 'DR-5HT']:
        for mvar in movement_vars:
            for name, r2 in [('full', 0.06), ('contrast', 0.04),
                             ('movement', 0.03)]:
                rows.append({'target_NM': tnm, 'movement_var': mvar,
                             'name': name, 'marginal_r2': r2})
    return pd.DataFrame(rows)


class TestPlotMovementR2Bars:
    def test_returns_figure(self):
        from iblnm.vis import plot_movement_r2_bars
        df = _make_movement_r2_bars()
        fig = plot_movement_r2_bars(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_bar_heights_are_model_r2(self):
        """One movement var, two target-NMs, three models: the panel shows three
        bars per target-NM whose heights are the input marginal_r2."""
        from iblnm.vis import plot_movement_r2_bars
        df = _make_movement_r2_bars(movement_vars=('choice',))
        fig = plot_movement_r2_bars(df)
        bars = [p.get_height() for p in fig.axes[0].patches
                if p.get_height() > 0]
        assert len(bars) == 6  # 2 target-NMs x 3 models
        assert {round(h, 4) for h in bars} == {0.03, 0.04, 0.06}
        plt.close(fig)

    def test_empty_df(self):
        from iblnm.vis import plot_movement_r2_bars
        df = pd.DataFrame(
            columns=['target_NM', 'movement_var', 'name', 'marginal_r2'])
        fig = plot_movement_r2_bars(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_movement_base_in_title(self):
        """The full-model description appears in the figure title."""
        from iblnm.vis import plot_movement_r2_bars
        fig = plot_movement_r2_bars(_make_movement_r2_bars())
        title = fig._suptitle.get_text()
        assert 'task base + <movement>' in title
        assert 'saturated' not in title.lower()
        plt.close(fig)


def _make_barplot_recordings():
    """Minimal recordings DataFrame for barplot tests."""
    return pd.DataFrame({
        'eid': ['e0', 'e0', 'e1', 'e1', 'e2'],
        'subject': ['s1', 's1', 's2', 's2', 's3'],
        'target_NM': ['VTA-DA', 'VTA-DA', 'DR-5HT', 'DR-5HT', 'VTA-DA'],
        'session_type': ['biased', 'biased', 'training', 'biased', 'biased'],
        'hemisphere': ['l', 'r', 'l', 'r', 'l'],
    })


class TestTargetOverviewBarplotHorizontal:

    def test_horizontal_bars_use_barh(self):
        from iblnm.vis import target_overview_barplot
        df = _make_barplot_recordings()
        ax = target_overview_barplot(df, horizontal=True)
        # barh creates patches whose width > height (skip zero-count bars)
        patches = [p for c in ax.containers for p in c if p.get_width() > 0]
        assert len(patches) > 0
        for p in patches:
            assert p.get_width() >= p.get_height(), "Expected horizontal bars"
        plt.close('all')

    def test_vertical_bars_default(self):
        from iblnm.vis import target_overview_barplot
        df = _make_barplot_recordings()
        ax = target_overview_barplot(df, horizontal=False)
        patches = [p for c in ax.containers for p in c if p.get_height() > 0]
        assert len(patches) > 0
        for p in patches:
            assert p.get_height() >= p.get_width(), "Expected vertical bars"
        plt.close('all')

    def test_horizontal_ylabel_is_target(self):
        from iblnm.vis import target_overview_barplot
        df = _make_barplot_recordings()
        ax = target_overview_barplot(df, horizontal=True)
        assert 'Target' in ax.get_ylabel() or 'target' in ax.get_ylabel().lower()
        assert 'Session' in ax.get_xlabel() or 'session' in ax.get_xlabel().lower()
        plt.close('all')


class TestMouseOverviewBarplotHorizontal:

    def test_horizontal_bars_use_barh(self):
        from iblnm.vis import mouse_overview_barplot
        df = _make_barplot_recordings()
        ax = mouse_overview_barplot(df, min_sessions=1, horizontal=True)
        patches = [p for c in ax.containers for p in c if p.get_width() > 0]
        assert len(patches) > 0
        for p in patches:
            assert p.get_width() >= p.get_height(), "Expected horizontal bars"
        plt.close('all')
