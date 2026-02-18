"""Tests for dataset overview helper functions."""
import pandas as pd
import pytest

from iblnm.util import aggregate_qc_per_session, concat_logs, make_log_entry, LOG_COLUMNS


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


class TestConcatLogsIntegration:
    """Tests for concat_logs used in dataset_overview context."""

    def test_upstream_logs_concatenated(self):
        """Upstream script logs are concatenated without information loss."""
        log1 = pd.DataFrame({
            'eid': ['a'], 'error_type': ['ValueError'],
            'error_message': ['bad data'], 'traceback': [None],
        })
        log2 = pd.DataFrame({
            'eid': ['b'], 'error_type': ['KeyError'],
            'error_message': ['missing key'], 'traceback': [None],
        })
        result = concat_logs([log1, log2])

        assert len(result) == 2
        assert list(result.columns) == LOG_COLUMNS

    def test_flag_errors_mixed_with_upstream(self):
        """Flag-derived errors are concatenated with upstream logs."""
        upstream = pd.DataFrame({
            'eid': ['a'], 'error_type': ['ValueError'],
            'error_message': ['bad'], 'traceback': [None],
        })
        flag_entries = [
            make_log_entry('b', error_type='MissingRawTask', error_message='has_raw_task=False'),
            make_log_entry('c', error_type='BandInversion', error_message='n_band_inversions > 0'),
        ]
        df_flags = pd.DataFrame(flag_entries)
        result = concat_logs([upstream, df_flags])

        assert len(result) == 3
        assert set(result['error_type']) == {'ValueError', 'MissingRawTask', 'BandInversion'}

    def test_same_eid_multiple_errors(self):
        """A session can have errors from multiple sources."""
        log1 = pd.DataFrame({
            'eid': ['a'], 'error_type': ['ValueError'],
            'error_message': ['bad data'], 'traceback': [None],
        })
        log2 = pd.DataFrame({
            'eid': ['a'], 'error_type': ['MissingRawTask'],
            'error_message': ['has_raw_task=False'], 'traceback': [None],
        })
        result = concat_logs([log1, log2])

        assert len(result) == 2
        assert (result['eid'] == 'a').all()

    def test_empty_upstream_with_flags(self):
        """Works when upstream logs are empty but flag errors exist."""
        flag_entries = [
            make_log_entry('a', error_type='MissingExtractedTask', error_message='has_extracted_task=False'),
        ]
        result = concat_logs([pd.DataFrame(columns=LOG_COLUMNS), pd.DataFrame(flag_entries)])

        assert len(result) == 1
        assert result.iloc[0]['error_type'] == 'MissingExtractedTask'
