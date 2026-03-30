"""Tests for scripts/session_viewer.py helper functions."""
import sys
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

def _make_mock_ps(load_trials_side_effect=None):
    """Build a mock PhotometrySession."""
    ps = MagicMock()
    ps.eid = 'test-eid'
    ps.trials = None
    if load_trials_side_effect:
        ps.load_trials.side_effect = load_trials_side_effect
    else:
        def _set_trials():
            ps.trials = MagicMock()
        ps.load_trials.side_effect = _set_trials
    return ps


def test_load_session_data_from_h5(monkeypatch, tmp_path):
    """When H5 exists, load_h5 is called."""
    ps = _make_mock_ps()
    h5_path = tmp_path / 'test-eid.h5'
    h5_path.touch()
    monkeypatch.setattr(sv, 'SESSIONS_H5_DIR', tmp_path)

    result = load_session_data(ps)

    assert result is ps
    ps.load_h5.assert_called_once_with(h5_path)


def test_load_session_data_missing_raw_data_continues(monkeypatch, tmp_path):
    """MissingRawData from load_trials prints a warning but does not raise."""
    ps = _make_mock_ps(load_trials_side_effect=MissingRawData("no raw data"))
    monkeypatch.setattr(sv, 'SESSIONS_H5_DIR', tmp_path)  # no H5 exists

    result = load_session_data(ps)

    assert result is ps
    ps.extract_responses.assert_not_called()


def test_load_session_data_missing_extracted_data_continues(monkeypatch, tmp_path):
    """MissingExtractedData from load_trials prints a warning but does not raise."""
    ps = _make_mock_ps(load_trials_side_effect=MissingExtractedData("not extracted"))
    monkeypatch.setattr(sv, 'SESSIONS_H5_DIR', tmp_path)

    result = load_session_data(ps)

    assert result is ps
    ps.extract_responses.assert_not_called()


def test_load_session_data_with_trials_calls_extract_responses(monkeypatch, tmp_path):
    """When trials load successfully, extract_responses is called."""
    ps = _make_mock_ps()
    monkeypatch.setattr(sv, 'SESSIONS_H5_DIR', tmp_path)

    load_session_data(ps)

    ps.extract_responses.assert_called_once()


def test_load_session_data_missing_photometry_exits(monkeypatch, tmp_path):
    """MissingRawData from load_photometry → sys.exit."""
    ps = _make_mock_ps()
    ps.load_photometry.side_effect = MissingRawData("no photometry")
    monkeypatch.setattr(sv, 'SESSIONS_H5_DIR', tmp_path)

    with pytest.raises(SystemExit):
        load_session_data(ps)


def test_load_session_data_missing_extracted_photometry_exits(monkeypatch, tmp_path):
    """MissingExtractedData from load_photometry → sys.exit."""
    ps = _make_mock_ps()
    ps.load_photometry.side_effect = MissingExtractedData("not extracted")
    monkeypatch.setattr(sv, 'SESSIONS_H5_DIR', tmp_path)

    with pytest.raises(SystemExit):
        load_session_data(ps)
