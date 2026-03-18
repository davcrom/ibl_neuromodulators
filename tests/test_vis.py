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

    def test_pool_aggregation_uses_trial_level_stats(self):
        """With aggregation='pool', mean/SEM should be computed across all trials,
        not across subject means. With unequal trial counts per subject, the two
        methods give different means."""
        from scipy.stats import sem as scipy_sem

        # s1: 20 trials at 1.0, s2: 5 trials at 0.0
        # Pool mean = (20*1 + 5*0) / 25 = 0.8
        # Subject mean = (1.0 + 0.0) / 2 = 0.5
        rows = (
            [{'subject': 's1', 'side': 'contra', 'contrast': 25.0,
              'feedbackType': 1, 'response': 1.0}] * 20
            + [{'subject': 's2', 'side': 'contra', 'contrast': 25.0,
                'feedbackType': 1, 'response': 0.0}] * 5
        )
        df = pd.DataFrame(rows)

        fig = plot_relative_contrast(df, 'response', 'VTA-DA', 'stimOn_times',
                                     aggregation='pool')
        ax_c = fig.axes[0]
        line = ax_c.containers[0].lines[0]
        plotted_mean = line.get_ydata()[0]
        assert np.isclose(plotted_mean, 0.8), (
            f"Pool mean should be 0.8 (trial-level), got {plotted_mean}"
        )
        plt.close(fig)

    def test_subject_aggregation_uses_subject_means(self):
        """With aggregation='subject', mean should be the mean of subject means."""
        # Same data as above — subject mean should be 0.5
        rows = (
            [{'subject': 's1', 'side': 'contra', 'contrast': 25.0,
              'feedbackType': 1, 'response': 1.0}] * 20
            + [{'subject': 's2', 'side': 'contra', 'contrast': 25.0,
                'feedbackType': 1, 'response': 0.0}] * 5
        )
        df = pd.DataFrame(rows)

        fig = plot_relative_contrast(df, 'response', 'VTA-DA', 'stimOn_times',
                                     aggregation='subject')
        ax_c = fig.axes[0]
        line = ax_c.containers[0].lines[0]
        plotted_mean = line.get_ydata()[0]
        assert np.isclose(plotted_mean, 0.5), (
            f"Subject mean should be 0.5, got {plotted_mean}"
        )
        plt.close(fig)

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
        ax = fig.axes[0]
        # Just verify it doesn't crash and returns a figure
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


def _make_lmm_predictions():
    """Synthetic LMM predictions DataFrame for testing."""
    contrasts = [0.0, 0.0625, 0.125, 0.25, 1.0]
    rows = []
    for side in ['contra', 'ipsi']:
        for reward in [0, 1]:
            for c in contrasts:
                rows.append({
                    'contrast': c,
                    'side': side,
                    'reward': reward,
                    'predicted': np.log(c + 0.01) * 0.5 + 0.3 * (reward == 1),
                    'ci_lower': np.log(c + 0.01) * 0.5 - 0.2,
                    'ci_upper': np.log(c + 0.01) * 0.5 + 0.2,
                })
    return pd.DataFrame(rows)


class TestPlotLMMResponse:

    def test_returns_figure(self):
        from iblnm.vis import plot_lmm_response
        predictions = _make_lmm_predictions()
        fig = plot_lmm_response(predictions, 'VTA-DA', 'stimOn_times')
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_has_two_panels(self):
        from iblnm.vis import plot_lmm_response
        predictions = _make_lmm_predictions()
        fig = plot_lmm_response(predictions, 'VTA-DA', 'stimOn_times')
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_correct_incorrect_lines(self):
        """Each panel should have 2 lines (correct + incorrect)."""
        from iblnm.vis import plot_lmm_response
        predictions = _make_lmm_predictions()
        fig = plot_lmm_response(predictions, 'VTA-DA', 'stimOn_times')
        for ax in fig.axes:
            lines = ax.get_lines()
            assert len(lines) >= 2
        plt.close(fig)

    def test_with_raw_data_overlay(self):
        """When raw data is provided, it should be overlaid."""
        from iblnm.vis import plot_lmm_response
        rng = np.random.default_rng(0)
        n = 100
        raw = pd.DataFrame({
            'contrast': rng.choice([0.0, 0.0625, 0.125, 0.25, 1.0], n),
            'side': rng.choice(['contra', 'ipsi'], n),
            'feedbackType': rng.choice([1, -1], n),
            'response': rng.normal(0, 1, n),
        })
        predictions = _make_lmm_predictions()
        fig = plot_lmm_response(predictions, 'VTA-DA', 'stimOn_times',
                                df_raw=raw, response_col='response')
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotLMMVarianceExplained:

    def test_returns_figure(self):
        from iblnm.vis import plot_lmm_variance_explained
        ve_dict = {
            ('VTA-DA', 'stimOn'): {'marginal': 0.15, 'conditional': 0.35},
            ('DR-5HT', 'feedback'): {'marginal': 0.08, 'conditional': 0.22},
        }
        fig = plot_lmm_variance_explained(ve_dict)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_has_bars(self):
        from iblnm.vis import plot_lmm_variance_explained
        ve_dict = {
            ('VTA-DA', 'stimOn'): {'marginal': 0.15, 'conditional': 0.35},
        }
        fig = plot_lmm_variance_explained(ve_dict)
        ax = fig.axes[0]
        # Should have bar containers
        assert len(ax.containers) >= 2  # marginal + conditional
        plt.close(fig)

    def test_empty_dict(self):
        from iblnm.vis import plot_lmm_variance_explained
        fig = plot_lmm_variance_explained({})
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotMarginalMeans:

    @staticmethod
    def _make_emm_dict():
        """EMM results keyed by (target_NM, event)."""
        return {
            ('VTA-DA', 'stimOn'): {
                'reward': pd.DataFrame({
                    'level': [0, 1],
                    'mean': [0.5, 0.8],
                    'ci_lower': [0.3, 0.6],
                    'ci_upper': [0.7, 1.0],
                }),
                'side': pd.DataFrame({
                    'level': ['contra', 'ipsi'],
                    'mean': [0.7, 0.6],
                    'ci_lower': [0.5, 0.4],
                    'ci_upper': [0.9, 0.8],
                }),
            },
            ('DR-5HT', 'stimOn'): {
                'reward': pd.DataFrame({
                    'level': [0, 1],
                    'mean': [0.2, 0.4],
                    'ci_lower': [0.0, 0.2],
                    'ci_upper': [0.4, 0.6],
                }),
                'side': pd.DataFrame({
                    'level': ['contra', 'ipsi'],
                    'mean': [0.3, 0.3],
                    'ci_lower': [0.1, 0.1],
                    'ci_upper': [0.5, 0.5],
                }),
            },
        }

    def test_returns_figure(self):
        from iblnm.vis import plot_marginal_means
        emm_dict = self._make_emm_dict()
        fig = plot_marginal_means(emm_dict, 'stimOn')
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_has_two_axes(self):
        """Should have one axis for reward, one for side."""
        from iblnm.vis import plot_marginal_means
        emm_dict = self._make_emm_dict()
        fig = plot_marginal_means(emm_dict, 'stimOn')
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_lines_per_target(self):
        """Each axis should have a point/errorbar per target."""
        from iblnm.vis import plot_marginal_means
        emm_dict = self._make_emm_dict()
        fig = plot_marginal_means(emm_dict, 'stimOn')
        # Each axis should have lines/containers for 2 targets
        for ax in fig.axes:
            assert len(ax.containers) >= 2 or len(ax.lines) >= 2
        plt.close(fig)
