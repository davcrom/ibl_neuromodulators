"""Tests for dataset overview helper functions."""
import pandas as pd

from iblnm.util import concat_logs, LOG_COLUMNS
from iblnm.validation import make_log_entry


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
