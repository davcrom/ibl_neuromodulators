"""Tests for dataset overview helper functions."""
import pandas as pd
import pytest

from iblnm.util import aggregate_qc_per_session, build_filter_status, merge_failure_logs, add_hemisphere, process_regions


class TestAddHemisphere:
    """Tests for add_hemisphere function."""

    def test_extracts_hemisphere_from_region_name(self):
        """Hemisphere suffix in region name is extracted."""
        df = pd.DataFrame({
            'brain_region': ['LC-r', 'LC-l', 'VTA', 'SNc-R'],
        })
        result = add_hemisphere(df, region_col='brain_region')

        assert result.loc[0, 'hemisphere'] == 'R'
        assert result.loc[1, 'hemisphere'] == 'L'
        assert pd.isna(result.loc[2, 'hemisphere'])  # VTA has no suffix
        assert result.loc[3, 'hemisphere'] == 'R'  # case insensitive

    def test_uses_fiber_coords_when_no_name_suffix(self):
        """Falls back to fiber coordinates when region name has no suffix."""
        df = pd.DataFrame({
            'subject': ['m1', 'm1'],
            'brain_region': ['VTA', 'LC'],
        })
        df_fibers = pd.DataFrame({
            'subject': ['m1', 'm1'],
            'targeted_region': ['VTA', 'LC'],
            'X-ml_um': [-100, 200],  # VTA right (x<0), LC left (x>0)
        })
        result = add_hemisphere(df, region_col='brain_region', df_fibers=df_fibers)

        assert result.loc[0, 'hemisphere'] == 'R'  # from fiber
        assert result.loc[1, 'hemisphere'] == 'L'  # from fiber

    def test_name_priority_over_fiber(self):
        """With priority='name', region name takes precedence over fiber coords."""
        df = pd.DataFrame({
            'subject': ['m1'],
            'brain_region': ['LC-r'],
        })
        df_fibers = pd.DataFrame({
            'subject': ['m1'],
            'targeted_region': ['LC'],
            'X-ml_um': [200],  # fiber says L
        })
        result = add_hemisphere(df, region_col='brain_region', df_fibers=df_fibers, priority='name')

        assert result.loc[0, 'hemisphere'] == 'R'  # name wins

    def test_fiber_priority_over_name(self):
        """With priority='fiber', fiber coords take precedence over region name."""
        df = pd.DataFrame({
            'subject': ['m1'],
            'brain_region': ['LC-r'],
        })
        df_fibers = pd.DataFrame({
            'subject': ['m1'],
            'targeted_region': ['LC'],
            'X-ml_um': [200],  # fiber says L
        })
        result = add_hemisphere(df, region_col='brain_region', df_fibers=df_fibers, priority='fiber')

        assert result.loc[0, 'hemisphere'] == 'L'  # fiber wins

    def test_warns_on_mismatch(self, capsys):
        """Prints warning when name and fiber hemispheres disagree."""
        df = pd.DataFrame({
            'subject': ['m1'],
            'brain_region': ['LC-r'],
        })
        df_fibers = pd.DataFrame({
            'subject': ['m1'],
            'targeted_region': ['LC'],
            'X-ml_um': [200],  # fiber says L, name says R
        })
        add_hemisphere(df, region_col='brain_region', df_fibers=df_fibers)

        captured = capsys.readouterr()
        assert 'mismatch' in captured.out.lower()

    def test_no_fiber_file_returns_name_only(self):
        """When no fiber data available, uses name only."""
        df = pd.DataFrame({
            'subject': ['m1'],
            'brain_region': ['LC-r'],
        })
        result = add_hemisphere(df, region_col='brain_region', df_fibers=None)

        assert result.loc[0, 'hemisphere'] == 'R'


class TestProcessRegions:
    """Tests for process_regions function."""

    def test_normalizes_region_names(self):
        """Non-standard region names are normalized."""
        df = pd.DataFrame({
            'brain_region': ['DRN', 'SNC', 'LC'],
            'NM': ['5HT', 'DA', 'NE'],
        })
        result = process_regions(df, region_col='brain_region', add_hemisphere=False)

        assert list(result['region_base']) == ['DR', 'SNc', 'LC']
        assert list(result['target_NM']) == ['DR-5HT', 'SNc-DA', 'LC-NE']

    def test_strips_hemisphere_suffix(self):
        """Hemisphere suffix is stripped from region name."""
        df = pd.DataFrame({
            'brain_region': ['LC-r', 'VTA-l', 'DR'],
            'NM': ['NE', 'DA', '5HT'],
        })
        result = process_regions(df, region_col='brain_region', add_hemisphere=False)

        assert list(result['region_base']) == ['LC', 'VTA', 'DR']

    def test_adds_hemisphere_column(self):
        """Hemisphere is extracted from region name suffix."""
        df = pd.DataFrame({
            'brain_region': ['LC-r', 'VTA-l', 'DR'],
            'NM': ['NE', 'DA', '5HT'],
        })
        result = process_regions(df, region_col='brain_region', add_hemisphere=True)

        assert result.loc[0, 'hemisphere'] == 'R'
        assert result.loc[1, 'hemisphere'] == 'L'
        assert pd.isna(result.loc[2, 'hemisphere'])

    def test_infers_nm_when_missing(self):
        """NM is inferred from region when NM='none'."""
        df = pd.DataFrame({
            'brain_region': ['LC', 'VTA', 'DR'],
            'NM': ['none', 'none', 'none'],
        })
        result = process_regions(df, region_col='brain_region', add_hemisphere=False)

        assert list(result['NM']) == ['NE', 'DA', '5HT']
        assert list(result['target_NM']) == ['LC-NE', 'VTA-DA', 'DR-5HT']

    def test_filters_to_valid_targets(self):
        """Only valid targets are kept when filter_valid=True."""
        df = pd.DataFrame({
            'brain_region': ['LC', 'VTA', 'MR'],  # MR is not in VALID_TARGETS
            'NM': ['NE', 'DA', '5HT'],
        })
        result = process_regions(df, region_col='brain_region', filter_valid=True)

        assert len(result) == 2
        assert 'MR-5HT' not in result['target_NM'].values

    def test_keeps_all_targets_when_not_filtering(self):
        """All targets kept when filter_valid=False."""
        df = pd.DataFrame({
            'brain_region': ['LC', 'VTA', 'MR'],
            'NM': ['NE', 'DA', '5HT'],
        })
        result = process_regions(df, region_col='brain_region', filter_valid=False)

        assert len(result) == 3
        assert 'MR-5HT' in result['target_NM'].values

    def test_works_with_target_column(self):
        """Works when region_col='target'."""
        df = pd.DataFrame({
            'target': ['LC-r', 'VTA'],
            'NM': ['NE', 'DA'],
        })
        result = process_regions(df, region_col='target')

        assert len(result) == 2
        assert list(result['target_NM']) == ['LC-NE', 'VTA-DA']


class TestAggregateQcPerSession:
    """Tests for aggregate_qc_per_session function."""

    def test_all_signals_must_pass_by_default(self):
        """With require_all=True (default), all signals must pass for session to pass."""
        df_qc = pd.DataFrame({
            'eid': ['a', 'a', 'b', 'b'],
            'brain_region': ['VTA', 'SNc', 'VTA', 'SNc'],
            'n_unique_samples': [0.5, 0.2, 0.05, 0.3],  # b has one < 0.1
            'n_band_inversions': [0, 0, 0, 0],
        })
        result = aggregate_qc_per_session(df_qc)

        assert len(result) == 2
        assert result.loc[result['eid'] == 'a', 'passes_basic_qc'].iloc[0] == True
        assert result.loc[result['eid'] == 'b', 'passes_basic_qc'].iloc[0] == False

    def test_any_signal_can_pass(self):
        """With require_all=False, any signal passing is sufficient."""
        df_qc = pd.DataFrame({
            'eid': ['a', 'a', 'b', 'b'],
            'brain_region': ['VTA', 'SNc', 'VTA', 'SNc'],
            'n_unique_samples': [0.5, 0.05, 0.05, 0.05],  # a has one > 0.1
            'n_band_inversions': [0, 0, 0, 0],
        })
        result = aggregate_qc_per_session(df_qc, require_all=False)

        assert result.loc[result['eid'] == 'a', 'passes_basic_qc'].iloc[0] == True
        assert result.loc[result['eid'] == 'b', 'passes_basic_qc'].iloc[0] == False

    def test_band_inversions_fail_session(self):
        """Any band inversion fails the session (require_all=True)."""
        df_qc = pd.DataFrame({
            'eid': ['a', 'a'],
            'brain_region': ['VTA', 'SNc'],
            'n_unique_samples': [0.5, 0.5],
            'n_band_inversions': [0, 1],  # one has inversion
        })
        result = aggregate_qc_per_session(df_qc, require_all=True)

        assert result.loc[result['eid'] == 'a', 'passes_basic_qc'].iloc[0] == False

    def test_band_inversions_any_mode(self):
        """With require_all=False, session passes if any signal has no inversions."""
        df_qc = pd.DataFrame({
            'eid': ['a', 'a'],
            'brain_region': ['VTA', 'SNc'],
            'n_unique_samples': [0.5, 0.5],
            'n_band_inversions': [0, 1],  # one has no inversion
        })
        result = aggregate_qc_per_session(df_qc, require_all=False)

        assert result.loc[result['eid'] == 'a', 'passes_basic_qc'].iloc[0] == True

    def test_empty_dataframe(self):
        """Empty input returns empty output."""
        df_qc = pd.DataFrame(columns=['eid', 'brain_region', 'n_unique_samples', 'n_band_inversions'])
        result = aggregate_qc_per_session(df_qc)

        assert len(result) == 0
        assert 'passes_basic_qc' in result.columns


class TestBuildFilterStatus:
    """Tests for build_filter_status function."""

    def test_includes_all_filter_columns(self):
        """Result includes all expected filter columns."""
        df_sessions = pd.DataFrame({
            'eid': ['a'],
            'subject': ['m1'],
            'has_raw_task': [True],
            'has_raw_photometry': [True],
            'has_trials': [True],
            'has_photometry': [True],
            'trials_in_photometry_time': [True],
        })
        result = build_filter_status(df_sessions)

        expected_cols = ['eid', 'subject', 'has_raw_task', 'has_raw_photometry',
                         'has_trials', 'has_photometry', 'trials_in_photometry_time',
                         'passes_basic_qc']
        for col in expected_cols:
            assert col in result.columns

    def test_merges_qc_aggregation(self):
        """QC aggregation is merged correctly."""
        df_sessions = pd.DataFrame({
            'eid': ['a', 'b', 'c'],
            'subject': ['m1', 'm1', 'm2'],
            'has_raw_task': [True, True, True],
            'has_raw_photometry': [True, True, True],
            'has_trials': [True, True, True],
            'has_photometry': [True, True, True],
            'trials_in_photometry_time': [True, True, True],
        })
        qc_agg = pd.DataFrame({
            'eid': ['a', 'b'],
            'passes_basic_qc': [True, False],
        })

        result = build_filter_status(df_sessions, qc_agg)

        assert result.loc[result['eid'] == 'a', 'passes_basic_qc'].iloc[0] == True
        assert result.loc[result['eid'] == 'b', 'passes_basic_qc'].iloc[0] == False
        assert result.loc[result['eid'] == 'c', 'passes_basic_qc'].iloc[0] == False  # not in qc_agg

    def test_no_qc_aggregation(self):
        """Without QC aggregation, passes_basic_qc defaults to False."""
        df_sessions = pd.DataFrame({
            'eid': ['a'],
            'subject': ['m1'],
            'has_raw_task': [True],
            'has_raw_photometry': [True],
            'has_trials': [True],
            'has_photometry': [True],
            'trials_in_photometry_time': [True],
        })
        result = build_filter_status(df_sessions, qc_agg=None)

        assert result.loc[result['eid'] == 'a', 'passes_basic_qc'].iloc[0] == False


class TestMergeFailureLogs:
    """Tests for merge_failure_logs function."""

    def test_merges_single_log(self):
        """Single error log is merged correctly."""
        df_failures = pd.DataFrame({
            'eid': ['a', 'b'],
            'subject': ['m1', 'm2'],
        })
        df_log = pd.DataFrame({
            'eid': ['a'],
            'exception_type': ['ValueError'],
            'exception_message': ['bad data'],
        })

        result = merge_failure_logs(df_failures, [('source1', df_log)])

        assert 'exception_type' in result.columns
        assert 'source' in result.columns
        assert result.loc[result['eid'] == 'a', 'source'].iloc[0] == 'source1'
        assert pd.isna(result.loc[result['eid'] == 'b', 'exception_type'].iloc[0])

    def test_merges_multiple_logs(self):
        """Multiple error logs are merged correctly."""
        df_failures = pd.DataFrame({
            'eid': ['a', 'b', 'c'],
            'subject': ['m1', 'm1', 'm2'],
        })
        df_log1 = pd.DataFrame({
            'eid': ['a'],
            'exception_type': ['ValueError'],
            'exception_message': ['bad data'],
        })
        df_log2 = pd.DataFrame({
            'eid': ['c'],
            'exception_type': ['KeyError'],
            'exception_message': ['missing key'],
        })

        result = merge_failure_logs(df_failures, [('query_db', df_log1), ('qc', df_log2)])

        # Session 'a' has error from query_db
        assert result.loc[result['eid'] == 'a', 'exception_type'].iloc[0] == 'ValueError'
        # Session 'c' has error from qc
        assert result.loc[result['eid'] == 'c', 'exception_type'].iloc[0] == 'KeyError'
        # Session 'b' has no errors
        assert pd.isna(result.loc[result['eid'] == 'b', 'exception_type'].iloc[0])

    def test_empty_logs(self):
        """Empty logs list returns original DataFrame with error columns."""
        df_failures = pd.DataFrame({
            'eid': ['a'],
            'subject': ['m1'],
        })

        result = merge_failure_logs(df_failures, [])

        assert len(result) == 1
        assert 'exception_type' in result.columns
        assert pd.isna(result['exception_type'].iloc[0])

    def test_none_log_skipped(self):
        """None logs are skipped gracefully."""
        df_failures = pd.DataFrame({
            'eid': ['a'],
            'subject': ['m1'],
        })

        result = merge_failure_logs(df_failures, [('source1', None)])

        assert len(result) == 1
