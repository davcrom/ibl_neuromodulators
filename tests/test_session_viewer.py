"""Tests for scripts/session_viewer.py helper functions."""
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import scripts.session_viewer as sv
from scripts.session_viewer import (
    find_session,
    load_session_data,
    print_session_errors,
)
from iblnm.validation import MissingRawData, MissingExtractedData


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def mock_one():
    one = MagicMock()
    one.alyx.rest.return_value = []
    return one


@pytest.fixture
def eid_args():
    """Namespace mimicking parse_args() with --eid."""
    return MagicMock(eid='aaa', subject=None, session_index=-1)


@pytest.fixture
def subject_args():
    """Namespace mimicking parse_args() with subject."""
    return MagicMock(eid=None, subject='ZFM-01', session_index=-1)


# =========================================================================
# find_session
# =========================================================================

def test_find_session_by_eid(mock_one, eid_args):
    """Queries REST by eid and returns a PhotometrySession."""
    mock_one.alyx.rest.return_value = [
        {'id': 'aaa', 'subject': 'ZFM-99', 'start_time': '2024-01-01',
         'number': 1}
    ]
    with patch.object(sv, 'PhotometrySession') as MockPS:
        mock_ps = MagicMock()
        MockPS.return_value = mock_ps

        result = find_session(eid_args, mock_one)

    mock_one.alyx.rest.assert_called_once_with(
        'sessions', 'list', id='aaa', project='ibl_fibrephotometry'
    )
    mock_ps.from_alyx.assert_called_once()
    assert result is mock_ps


def test_find_session_by_eid_not_found_exits(mock_one, eid_args):
    """REST returns nothing → sys.exit."""
    with pytest.raises(SystemExit):
        find_session(eid_args, mock_one)


def test_find_session_by_subject_returns_last(mock_one, subject_args):
    """Default index=-1 returns the most recent session."""
    mock_one.alyx.rest.return_value = [
        {'id': 'r1', 'subject': 'ZFM-01', 'start_time': '2023-01-01',
         'number': 1},
        {'id': 'r2', 'subject': 'ZFM-01', 'start_time': '2023-06-01',
         'number': 1},
    ]
    with patch.object(sv, 'PhotometrySession') as MockPS:
        mock_ps = MagicMock()
        MockPS.return_value = mock_ps

        find_session(subject_args, mock_one)

    # Should have been called with the later session (r2 at index -1)
    row_arg = MockPS.call_args[0][0]
    assert row_arg['eid'] == 'r2'


def test_find_session_by_subject_returns_first(mock_one, subject_args):
    """Index=0 returns the earliest session."""
    subject_args.session_index = 0
    mock_one.alyx.rest.return_value = [
        {'id': 'r1', 'subject': 'ZFM-01', 'start_time': '2023-01-01',
         'number': 1},
        {'id': 'r2', 'subject': 'ZFM-01', 'start_time': '2023-06-01',
         'number': 1},
    ]
    with patch.object(sv, 'PhotometrySession') as MockPS:
        mock_ps = MagicMock()
        MockPS.return_value = mock_ps

        find_session(subject_args, mock_one)

    row_arg = MockPS.call_args[0][0]
    assert row_arg['eid'] == 'r1'


def test_find_session_by_subject_not_found_exits(mock_one, subject_args):
    """REST returns nothing → sys.exit."""
    with pytest.raises(SystemExit):
        find_session(subject_args, mock_one)


def test_find_session_by_subject_index_out_of_range_exits(mock_one, subject_args):
    """Index beyond the number of sessions → sys.exit."""
    subject_args.session_index = 99
    mock_one.alyx.rest.return_value = [
        {'id': 'r1', 'subject': 'ZFM-01', 'start_time': '2023-01-01',
         'number': 1},
    ]
    with pytest.raises(SystemExit):
        find_session(subject_args, mock_one)


def test_find_session_by_subject_queries_with_subject_kwarg(mock_one, subject_args):
    """REST is called with subject= kwarg."""
    mock_one.alyx.rest.return_value = [
        {'id': 'r1', 'subject': 'ZFM-01', 'start_time': '2023-01-01',
         'number': 1},
    ]
    with patch.object(sv, 'PhotometrySession') as MockPS:
        MockPS.return_value = MagicMock()
        find_session(subject_args, mock_one)

    mock_one.alyx.rest.assert_called_once_with(
        'sessions', 'list', subject='ZFM-01', project='ibl_fibrephotometry'
    )


def test_find_session_calls_from_alyx(mock_one, eid_args):
    """find_session must call from_alyx() on the PhotometrySession."""
    mock_one.alyx.rest.return_value = [
        {'id': 'aaa', 'subject': 'ZFM-99', 'start_time': '2024-01-01',
         'number': 1}
    ]
    with patch.object(sv, 'PhotometrySession') as MockPS:
        mock_ps = MagicMock()
        MockPS.return_value = mock_ps

        find_session(eid_args, mock_one)

    mock_ps.from_alyx.assert_called_once()


# =========================================================================
# print_session_errors
# =========================================================================

def test_print_session_errors_no_errors(capsys):
    """Session with no errors prints nothing."""
    ps = MagicMock()
    ps.errors = []
    print_session_errors(ps)
    assert capsys.readouterr().out == ''


def test_print_session_errors_prints_errors(capsys):
    """Errors are printed in [ErrorType] message format."""
    ps = MagicMock()
    ps.errors = [
        {'eid': 'aaa', 'error_type': 'InvalidStrain',
         'error_message': 'bad strain', 'traceback': ''},
    ]
    print_session_errors(ps)
    out = capsys.readouterr().out
    assert '[InvalidStrain]' in out
    assert 'bad strain' in out


# =========================================================================
# load_session_data
# =========================================================================

def _make_mock_ps(tmp_path, responses_side_effect=None):
    """Mock PhotometrySession whose H5 path does not exist (empty store)."""
    ps = MagicMock()
    ps.eid = 'test-eid'
    ps.trials = None
    ps.photometry = {}
    ps.photometry_responses = {}
    ps.filepath = tmp_path / 'test-eid.h5'
    if responses_side_effect:
        ps.load_responses.side_effect = responses_side_effect
    return ps


def test_load_session_data_empty_store_builds_through_load_methods(tmp_path):
    """With no H5, the three load methods are called and nothing is read back."""
    ps = _make_mock_ps(tmp_path)

    result = load_session_data(ps)

    assert result is ps
    ps.load_h5.assert_not_called()
    ps.load_raw_photometry.assert_called_once_with()
    ps.load_photometry.assert_called_once_with()
    ps.load_responses.assert_called_once_with('photometry')


def test_load_session_data_reads_stored_trials(tmp_path):
    """An existing H5 supplies the trials table, which has no load method."""
    ps = _make_mock_ps(tmp_path)
    ps.filepath.touch()

    load_session_data(ps)

    ps.load_h5.assert_called_once_with(ps.filepath, groups=['trials'])


def test_load_session_data_missing_raw_data_continues(tmp_path):
    """MissingRawData from load_responses warns but does not raise."""
    ps = _make_mock_ps(tmp_path,
                       responses_side_effect=MissingRawData("no raw data"))

    result = load_session_data(ps)

    assert result is ps
    assert ps.photometry_responses == {}


def test_load_session_data_missing_extracted_data_continues(tmp_path):
    """MissingExtractedData from load_responses warns but does not raise."""
    ps = _make_mock_ps(
        tmp_path, responses_side_effect=MissingExtractedData("not extracted"))

    result = load_session_data(ps)

    assert result is ps
    assert ps.photometry_responses == {}


def test_load_session_data_missing_photometry_exits(tmp_path):
    """MissingRawData from load_raw_photometry -> sys.exit."""
    ps = _make_mock_ps(tmp_path)
    ps.load_raw_photometry.side_effect = MissingRawData("no photometry")

    with pytest.raises(SystemExit):
        load_session_data(ps)


def test_load_session_data_missing_extracted_photometry_exits(tmp_path):
    """MissingExtractedData from load_raw_photometry -> sys.exit."""
    ps = _make_mock_ps(tmp_path)
    ps.load_raw_photometry.side_effect = MissingExtractedData("not extracted")

    with pytest.raises(SystemExit):
        load_session_data(ps)


def test_load_session_data_complete_store_makes_no_one_call(tmp_path, monkeypatch):
    """A store holding every product is read back without touching Alyx."""
    import numpy as np
    from iblnm.data import PhotometrySession

    monkeypatch.setattr('iblnm.data.store_raw', True)
    row = pd.Series({'eid': 'stored-eid', 'subject': 'ZFM-01',
                     'start_time': '2024-01-01T10:00:00', 'number': 1,
                     'brain_region': ['VTA'], 'hemisphere': ['l'],
                     'target_NM': ['VTA-DA']})
    times = np.linspace(0, 600, 6000)
    bands = {band: pd.DataFrame({'VTA': np.linspace(500, 400, 6000)}, index=times)
             for band in ('GCaMP', 'Isosbestic')}
    builder = PhotometrySession(row, one=MagicMock(), load_data=False)
    builder.filepath = tmp_path / 'stored-eid.h5'
    builder.photometry = dict(bands)
    builder.trials = pd.DataFrame({
        'trial': np.arange(20),
        'stimOn_times': np.linspace(50, 550, 20),
        'firstMovement_times': np.linspace(50.3, 550.3, 20),
        'feedback_times': np.linspace(51, 551, 20),
    })
    builder.preprocess()
    builder.load_responses('photometry')
    builder.save_h5()

    ps = PhotometrySession(row, one=MagicMock(), load_data=False)
    ps.filepath = builder.filepath
    ps.one.reset_mock()  # construction resolves the session path via eid2path

    load_session_data(ps)

    assert ps.one.method_calls == []
    assert set(ps.photometry) == {'GCaMP', 'Isosbestic', 'GCaMP_preprocessed'}
    assert 'VTA' in ps.photometry_responses
    assert len(ps.trials) == 20
