"""Tests for iblnm.io module."""
import pandas as pd
import pytest
from unittest.mock import MagicMock

from one.alf.exceptions import ALFObjectNotFound

from iblnm.io import get_session_dict, get_brain_region


@pytest.fixture
def mock_session():
    return pd.Series({
        'eid': 'test-eid-123',
        'subject': 'test_mouse',
        'brain_region': [],
        'hemisphere': [],
    })


@pytest.fixture
def mock_one():
    return MagicMock()


class TestGetSessionDict:
    """get_session_dict populates metadata from the Alyx session dict."""

    def test_populates_end_time_lab_users(self, mock_session, mock_one):
        mock_one.alyx.rest.return_value = {
            'users': ['alice'],
            'lab': 'cortexlab',
            'end_time': '2024-01-01T12:00:00',
        }
        result = get_session_dict(mock_session, one=mock_one)
        assert result['end_time'] == '2024-01-01T12:00:00'
        assert result['lab'] == 'cortexlab'
        assert result['users'] == ['alice']

    def test_missing_key_does_not_crash(self, mock_session, mock_one):
        """Session dict missing optional keys doesn't raise."""
        mock_one.alyx.rest.return_value = {'lab': 'cortexlab'}
        result = get_session_dict(mock_session, one=mock_one)
        assert result['lab'] == 'cortexlab'

    def test_logs_error_with_exlog(self, mock_session, mock_one):
        """Alyx failure logs to exlog instead of raising."""
        mock_one.alyx.rest.side_effect = Exception("connection error")
        exlog = []
        get_session_dict(mock_session, one=mock_one, exlog=exlog)
        assert len(exlog) == 1
        assert exlog[0]['error_type'] == 'Exception'


class TestGetBrainRegion:
    """get_brain_region populates brain_region/hemisphere from experiment desc or locations."""

    def test_from_experiment_description(self, mock_session, mock_one):
        """Experiment description with fibers → brain_region and hemisphere."""
        mock_one.load_dataset.return_value = {
            'devices': {'neurophotometrics': {'fibers': {
                'G0': {'location': 'VTA-r'},
                'G1': {'location': 'DR'},
            }}}
        }
        result = get_brain_region(mock_session, one=mock_one)
        assert result['brain_region'] == ['VTA-r', 'DR']
        assert result['hemisphere'] == ['r', '']

    def test_fallback_to_locations_file(self, mock_session, mock_one):
        """Experiment desc missing → falls back to photometryROI.locations.pqt."""
        mock_one.load_dataset.side_effect = [
            ALFObjectNotFound("experiment description not found"),
            pd.DataFrame({'brain_region': ['VTA-r', 'DR']}, index=['G0', 'G1']),
        ]
        result = get_brain_region(mock_session, one=mock_one)
        assert result['brain_region'] == ['VTA-r', 'DR']
        assert result['hemisphere'] == ['r', '']

    def test_both_missing_raises(self, mock_session, mock_one):
        """Both sources missing → raises ALFObjectNotFound."""
        mock_one.load_dataset.side_effect = ALFObjectNotFound("not found")
        with pytest.raises(ALFObjectNotFound, match="brain_region"):
            get_brain_region(mock_session, one=mock_one)

    def test_both_missing_logs_with_exlog(self, mock_session, mock_one):
        """Both sources missing with exlog → logs descriptive error."""
        mock_one.load_dataset.side_effect = ALFObjectNotFound("not found")
        exlog = []
        get_brain_region(mock_session, one=mock_one, exlog=exlog)
        assert len(exlog) == 1
        assert exlog[0]['error_type'] == 'ALFObjectNotFound'
        assert 'brain_region' in exlog[0]['error_message']

    def test_experiment_desc_no_fibers(self, mock_session, mock_one):
        """Experiment desc exists but has no neurophotometrics fibers → empty lists."""
        mock_one.load_dataset.return_value = {'devices': {}}
        result = get_brain_region(mock_session, one=mock_one)
        assert result['brain_region'] == []
        assert result['hemisphere'] == []

    def test_experiment_desc_uses_exact_filename(self, mock_session, mock_one):
        """load_dataset is called with the exact filename, not a wildcard pattern."""
        mock_one.load_dataset.return_value = {'devices': {}}
        get_brain_region(mock_session, one=mock_one)
        first_call_dataset = mock_one.load_dataset.call_args_list[0][0][1]
        assert first_call_dataset == '_ibl_experiment.description.yaml'
