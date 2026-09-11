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


def load_dataset_by_name(datasets):
    """Build a ``one.load_dataset`` stub that answers by dataset name.

    Parameters
    ----------
    datasets : dict
        Dataset name -> the loaded object. A name absent from the mapping
        raises ``ALFObjectNotFound``, as ONE does for a session lacking it.

    Returns
    -------
    callable
        Stub with ONE's ``(eid, dataset)`` signature, for use as a
        ``MagicMock.side_effect``.
    """
    def _load(eid, dataset, **kwargs):
        if dataset not in datasets:
            raise ALFObjectNotFound(dataset)
        return datasets[dataset]
    return _load


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

    def test_empty_fibers_falls_through_to_locations(self, mock_session, mock_one):
        """Description present but listing no fibers → region from the locations file."""
        mock_one.load_dataset.side_effect = load_dataset_by_name({
            '_ibl_experiment.description.yaml': {
                'devices': {'neurophotometrics': {'fibers': {}}}
            },
            'photometryROI.locations.pqt': pd.DataFrame(
                {'fiber': ['fiber_LC', 'fiber_LC'], 'brain_region': ['LC-r', 'LC-l']},
                index=pd.Index(['Region3G', 'Region6G'], name='ROI'),
            ),
        })
        result = get_brain_region(mock_session, one=mock_one)
        assert result['brain_region'] == ['LC-r', 'LC-l']
        assert result['hemisphere'] == ['r', 'l']

    def test_locations_file_wins_over_description(self, mock_session, mock_one):
        """Both sources name fibers → the locations file supplies the regions."""
        mock_one.load_dataset.side_effect = load_dataset_by_name({
            '_ibl_experiment.description.yaml': {
                'devices': {'neurophotometrics': {'fibers': {
                    'G0': {'location': 'NBM'},
                    'G1': {'location': 'NBM'},
                }}}
            },
            'photometryROI.locations.pqt': pd.DataFrame(
                {'fiber': ['fiber_NBM', 'fiber_NBM'], 'brain_region': ['NBM-r', 'NBM-l']},
                index=pd.Index(['Region3G', 'Region5G'], name='ROI'),
            ),
        })
        result = get_brain_region(mock_session, one=mock_one)
        assert result['brain_region'] == ['NBM-r', 'NBM-l']
        assert result['hemisphere'] == ['r', 'l']

    def test_falls_back_to_experiment_description(self, mock_session, mock_one):
        """No locations file → regions come from the experiment description."""
        mock_one.load_dataset.side_effect = load_dataset_by_name({
            '_ibl_experiment.description.yaml': {
                'devices': {'neurophotometrics': {'fibers': {
                    'G0': {'location': 'VTA-r'},
                    'G1': {'location': 'DR'},
                }}}
            },
        })
        result = get_brain_region(mock_session, one=mock_one)
        assert result['brain_region'] == ['VTA-r', 'DR']
        assert result['hemisphere'] == ['r', '']

    def test_no_source_names_a_region_raises(self, mock_session, mock_one):
        """No locations file and an empty fibers dict → raises, naming both sources."""
        mock_one.load_dataset.side_effect = load_dataset_by_name({
            '_ibl_experiment.description.yaml': {'devices': {}},
        })
        with pytest.raises(ALFObjectNotFound, match="brain_region"):
            get_brain_region(mock_session, one=mock_one)

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

    def test_sources_use_exact_filenames_in_order(self, mock_session, mock_one):
        """Both datasets are requested by exact name, locations file first."""
        mock_one.load_dataset.side_effect = ALFObjectNotFound("not found")
        get_brain_region(mock_session, one=mock_one, exlog=[])
        requested = [call[0][1] for call in mock_one.load_dataset.call_args_list]
        assert requested == [
            'photometryROI.locations.pqt', '_ibl_experiment.description.yaml']


class TestGetExtendedQC:
    """get_extended_qc normalizes Alyx extended-QC outcomes."""

    def test_null_outcome_becomes_not_set(self, mock_session, mock_one):
        from iblnm.io import get_extended_qc
        mock_one.alyx.rest.return_value = {
            'qc': 'FAIL',
            'extended_qc': {'_videoLeft_wheel_alignment': None},
        }
        result = get_extended_qc(mock_session, one=mock_one)
        assert result['qc_videoLeft_wheel_alignment'] == 'NOT_SET'

    def test_enum_outcome_maps_to_label(self, mock_session, mock_one):
        from iblnm.io import get_extended_qc
        mock_one.alyx.rest.return_value = {
            'qc': 'PASS',
            'extended_qc': {'_videoLeft_focus': 10},
        }
        result = get_extended_qc(mock_session, one=mock_one)
        assert result['qc_videoLeft_focus'] == 'PASS'
