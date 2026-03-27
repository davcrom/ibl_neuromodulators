"""Tests for scripts/session_viewer.py helper functions."""
import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import scripts.session_viewer as sv
from scripts.session_viewer import (
    build_session_from_rest,
    find_session_by_eid,
    find_session_by_subject,
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
def patch_pipeline(monkeypatch):
    """Patch all pipeline io/util/validate functions to no-ops."""
    identity = lambda s, one=None, exlog=None: s
    noop = lambda s, exlog=None: None
    monkeypatch.setattr(sv, 'get_subject_info', identity)
    monkeypatch.setattr(sv, 'get_session_dict', identity)
    monkeypatch.setattr(sv, 'get_brain_region', identity)
    monkeypatch.setattr(sv, 'fill_hemisphere_from_fiber_insertion_table', identity)
    monkeypatch.setattr(sv, 'get_datasets', identity)
    monkeypatch.setattr(sv, 'get_session_type', identity)
    monkeypatch.setattr(sv, 'get_targetNM', identity)
    monkeypatch.setattr(sv, 'get_session_length', identity)
    monkeypatch.setattr(sv, 'validate_subject', noop)
    monkeypatch.setattr(sv, 'validate_strain', noop)
    monkeypatch.setattr(sv, 'validate_line', noop)
    monkeypatch.setattr(sv, 'validate_neuromodulator', noop)
    monkeypatch.setattr(sv, 'validate_brain_region', noop)
    monkeypatch.setattr(sv, 'validate_hemisphere', noop)
    monkeypatch.setattr(sv, 'validate_datasets', noop)


# =========================================================================
# build_session_from_rest
# =========================================================================

def test_build_session_from_rest_renames_id(mock_one, patch_pipeline):
    """'id' key must be renamed to 'eid' in the output Series."""
    result = build_session_from_rest(
        {'id': 'xyz-eid', 'subject': 'ZFM-99', 'start_time': '2024-01-01'}, mock_one
    )
    assert result['eid'] == 'xyz-eid'
    assert 'id' not in result.index


def test_build_session_from_rest_captures_errors(mock_one, monkeypatch):
    """Validation errors are collected into logged_errors and _rest_errors."""
    identity = lambda s, one=None, exlog=None: s
    noop = lambda s, exlog=None: None
    for name in ('get_subject_info', 'get_session_dict', 'get_brain_region',
                 'fill_hemisphere_from_fiber_insertion_table', 'get_datasets',
                 'get_session_type', 'get_targetNM', 'get_session_length'):
        monkeypatch.setattr(sv, name, identity)
    for name in ('validate_subject', 'validate_line', 'validate_neuromodulator',
                 'validate_brain_region', 'validate_hemisphere', 'validate_datasets'):
        monkeypatch.setattr(sv, name, noop)

    def erroring_strain(session, exlog=None):
        if exlog is not None:
            exlog.append({
                'eid': session.get('eid', ''),
                'error_type': 'InvalidStrain',
                'error_message': 'Strain unknown not valid',
                'traceback': '',
            })
    monkeypatch.setattr(sv, 'validate_strain', erroring_strain)

    result = build_session_from_rest(
        {'id': 'abc', 'subject': 'ZFM-99', 'start_time': '2024-01-01'}, mock_one
    )
    assert result['logged_errors'] == ['InvalidStrain']
    assert len(result['_rest_errors']) == 1
    assert result['_rest_errors'][0]['error_message'] == 'Strain unknown not valid'


def test_build_session_from_rest_no_errors_empty_lists(mock_one, patch_pipeline):
    """When no validation errors occur, logged_errors and _rest_errors are empty."""
    result = build_session_from_rest(
        {'id': 'abc', 'subject': 'ZFM-99', 'start_time': '2024-01-01'}, mock_one
    )
    assert result['logged_errors'] == []
    assert result['_rest_errors'] == []


# =========================================================================
# find_session_by_eid
# =========================================================================

def test_find_session_by_eid_queries_rest(mock_one, patch_pipeline):
    """Always queries REST; returns the built session for the given EID."""
    mock_one.alyx.rest.return_value = [
        {'id': 'aaa', 'subject': 'ZFM-99', 'start_time': '2024-01-01'}
    ]
    result = find_session_by_eid('aaa', mock_one)
    assert result['eid'] == 'aaa'
    mock_one.alyx.rest.assert_called_once_with(
        'sessions', 'list', id='aaa', project='ibl_fibrephotometry'
    )


def test_find_session_by_eid_not_found_exits(mock_one):
    """REST returns nothing → sys.exit."""
    with pytest.raises(SystemExit):
        find_session_by_eid('missing', mock_one)


# =========================================================================
# find_session_by_subject
# =========================================================================

def test_find_session_by_subject_returns_last(mock_one, patch_pipeline):
    """Default index=-1 returns the most recent session for a subject."""
    mock_one.alyx.rest.return_value = [
        {'id': 'r1', 'subject': 'ZFM-01', 'start_time': '2023-01-01'},
        {'id': 'r2', 'subject': 'ZFM-01', 'start_time': '2023-06-01'},
    ]
    result = find_session_by_subject('ZFM-01', -1, mock_one)
    assert result['eid'] == 'r2'


def test_find_session_by_subject_returns_first(mock_one, patch_pipeline):
    """Index=0 returns the earliest session."""
    mock_one.alyx.rest.return_value = [
        {'id': 'r1', 'subject': 'ZFM-01', 'start_time': '2023-01-01'},
        {'id': 'r2', 'subject': 'ZFM-01', 'start_time': '2023-06-01'},
    ]
    result = find_session_by_subject('ZFM-01', 0, mock_one)
    assert result['eid'] == 'r1'


def test_find_session_by_subject_not_found_exits(mock_one):
    """REST returns nothing → sys.exit."""
    with pytest.raises(SystemExit):
        find_session_by_subject('ZFM-99', -1, mock_one)


def test_find_session_by_subject_index_out_of_range_exits(mock_one, patch_pipeline):
    """Index beyond the number of sessions → sys.exit."""
    mock_one.alyx.rest.return_value = [
        {'id': 'r1', 'subject': 'ZFM-01', 'start_time': '2023-01-01'},
    ]
    with pytest.raises(SystemExit):
        find_session_by_subject('ZFM-01', 99, mock_one)


def test_find_session_by_subject_queries_with_subject_kwarg(mock_one, patch_pipeline):
    """REST is called with subject= kwarg."""
    mock_one.alyx.rest.return_value = [
        {'id': 'r1', 'subject': 'ZFM-01', 'start_time': '2023-01-01'},
    ]
    find_session_by_subject('ZFM-01', 0, mock_one)
    mock_one.alyx.rest.assert_called_once_with(
        'sessions', 'list', subject='ZFM-01', project='ibl_fibrephotometry'
    )


# =========================================================================
# print_session_errors
# =========================================================================

def test_print_session_errors_no_errors(capsys):
    """Session with no errors prints nothing."""
    session = pd.Series({'eid': 'aaa', '_rest_errors': []})
    print_session_errors(session)
    assert capsys.readouterr().out == ''


def test_print_session_errors_prints_rest_errors(capsys):
    """Errors in _rest_errors are printed in [ErrorType] message format."""
    session = pd.Series({
        'eid': 'aaa',
        '_rest_errors': [
            {'eid': 'aaa', 'error_type': 'InvalidStrain',
             'error_message': 'bad strain', 'traceback': ''},
        ],
    })
    print_session_errors(session)
    out = capsys.readouterr().out
    assert '[InvalidStrain]' in out
    assert 'bad strain' in out


# =========================================================================
# load_session_data
# =========================================================================

def _make_mock_ps(load_trials_side_effect=None):
    """Build a mock PhotometrySession."""
    ps = MagicMock()
    ps.trials = None
    if load_trials_side_effect:
        ps.load_trials.side_effect = load_trials_side_effect
    else:
        def _set_trials():
            ps.trials = MagicMock()
        ps.load_trials.side_effect = _set_trials
    return ps


@pytest.fixture
def session_row(tmp_path):
    return pd.Series({'eid': 'test-eid', 'subject': 'ZFM-01'})


def test_load_session_data_missing_raw_data_continues(session_row, monkeypatch, tmp_path):
    """MissingRawData from load_trials prints a warning but does not raise."""
    ps = _make_mock_ps(load_trials_side_effect=MissingRawData("no raw data"))
    monkeypatch.setattr(sv, 'PhotometrySession', lambda row, one: ps)
    monkeypatch.setattr(sv, 'SESSIONS_H5_DIR', tmp_path)  # no H5 exists

    result = load_session_data(session_row, MagicMock())

    assert result is ps
    ps.extract_responses.assert_not_called()


def test_load_session_data_missing_extracted_data_continues(session_row, monkeypatch, tmp_path):
    """MissingExtractedData from load_trials prints a warning but does not raise."""
    ps = _make_mock_ps(load_trials_side_effect=MissingExtractedData("not extracted"))
    monkeypatch.setattr(sv, 'PhotometrySession', lambda row, one: ps)
    monkeypatch.setattr(sv, 'SESSIONS_H5_DIR', tmp_path)

    result = load_session_data(session_row, MagicMock())

    assert result is ps
    ps.extract_responses.assert_not_called()


def test_load_session_data_with_trials_calls_extract_responses(session_row, monkeypatch, tmp_path):
    """When trials load successfully, extract_responses is called."""
    ps = _make_mock_ps()
    monkeypatch.setattr(sv, 'PhotometrySession', lambda row, one: ps)
    monkeypatch.setattr(sv, 'SESSIONS_H5_DIR', tmp_path)

    load_session_data(session_row, MagicMock())

    ps.extract_responses.assert_called_once()


def test_load_session_data_missing_photometry_exits(session_row, monkeypatch, tmp_path):
    """MissingRawData from load_photometry → sys.exit."""
    ps = _make_mock_ps()
    ps.load_photometry.side_effect = MissingRawData("no photometry")
    monkeypatch.setattr(sv, 'PhotometrySession', lambda row, one: ps)
    monkeypatch.setattr(sv, 'SESSIONS_H5_DIR', tmp_path)

    with pytest.raises(SystemExit):
        load_session_data(session_row, MagicMock())


def test_load_session_data_missing_extracted_photometry_exits(session_row, monkeypatch, tmp_path):
    """MissingExtractedData from load_photometry → sys.exit."""
    ps = _make_mock_ps()
    ps.load_photometry.side_effect = MissingExtractedData("not extracted")
    monkeypatch.setattr(sv, 'PhotometrySession', lambda row, one: ps)
    monkeypatch.setattr(sv, 'SESSIONS_H5_DIR', tmp_path)

    with pytest.raises(SystemExit):
        load_session_data(session_row, MagicMock())
