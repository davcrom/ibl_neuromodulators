"""Tests for scripts/rebuild_responses.py — the photometry response rebuild."""
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pandas as pd
import pytest

import scripts.rebuild_responses as rebuild
from iblnm.config import RESPONSE_EVENTS, WHEEL_FS, WHEEL_RESPONSE_EVENTS
from iblnm.data import PhotometrySession

N_TRIALS = 5
TRIAL_SPACING = 4.0  # seconds between stimulus onsets, wide enough for the cut


@pytest.fixture
def session_series():
    return pd.Series({
        'eid': 'test-eid-rebuild',
        'subject': 'test_mouse',
        'start_time': '2024-01-01T10:00:00',
        'number': 1,
        'lab': 'test_lab',
        'projects': [],
        'url': None,
        'session_n': 1,
        'task_protocol': '_iblrig_tasks_trainingChoiceWorld6.4.2',
        'session_type': 'training',
        'brain_region': ['VTA'],
        'hemisphere': ['l'],
        'datasets': [],
    })


def _raw_bands(duration=60.0, fs=30.0, seed=42):
    """Raw GCaMP and Isosbestic bands over one region, bleaching together.

    The preprocessing the fallback runs needs both bands on the same clock; the
    shared exponential decay is what the isosbestic regression has to remove.
    """
    rng = np.random.default_rng(seed)
    times = np.arange(0, duration, 1 / fs)
    bleaching = 1000 * np.exp(-times / 300)
    return {
        'GCaMP': pd.DataFrame(
            {'VTA': bleaching + rng.normal(0, 1, times.size) + 500},
            index=times),
        'Isosbestic': pd.DataFrame(
            {'VTA': 0.8 * bleaching + rng.normal(0, 0.5, times.size) + 400},
            index=times),
    }


def _trials(n_trials=N_TRIALS):
    """Trials spaced widely enough that no response window leaves the signal.

    `response_times` is the wheel cut's per-trial window end, so it sits inside
    the spacing too.
    """
    stim_on = 10.0 + TRIAL_SPACING * np.arange(n_trials)
    return pd.DataFrame({
        'trial': np.arange(n_trials),
        'stimOnTrigger_times': stim_on,
        'response_times': stim_on + 1.0,
        'feedback_times': stim_on + 1.0,
    })


def _wheel_velocity(duration=60.0):
    """A preprocessed wheel velocity on the `WHEEL_FS` grid, in rad/s."""
    times = np.arange(0, duration, 1 / WHEEL_FS)
    return pd.Series(np.sin(times), index=times)


@pytest.fixture
def stored_session(session_series, tmp_path):
    """A written H5 holding trials, a preprocessed band and velocity, and QC.

    No responses group: what the rebuild writes is the product under test, and
    a session that never had one is the same case as one whose cut changed.
    """
    session = PhotometrySession(session_series, one=MagicMock(), load_data=False)
    session.filepath = tmp_path / f'{session.eid}.h5'
    session.trials = _trials()
    times = np.arange(0, 60, 0.05)
    session.photometry['GCaMP_preprocessed'] = pd.DataFrame(
        {'VTA': np.sin(times)}, index=times)
    # `extract_preprocessed_photometry` leaves these beside the band it built,
    # so a stored preprocessed signal always carries them.
    session.preprocessing_diagnostics = {'VTA': {'bleaching_tau': 300.0}}
    session.photometry_qc = {'VTA': {'n_unique_samples_GCaMP': 1200.0}}
    session.wheel_velocity = _wheel_velocity()
    # Metadata is named explicitly, as the download build does: it is not a
    # product, so the auto-detected group list does not carry it.
    session.save_h5(groups=['metadata'], mode='w')
    session.save_h5()
    return session.filepath


def _reload(filepath):
    return PhotometrySession.from_h5(filepath, one=MagicMock())


@pytest.fixture
def trials_only_session(session_series, tmp_path):
    """A written H5 holding the trials table and nothing else.

    The case the Alyx fallback exists for: no stored band to cut from, so the
    rebuild has to build one before it can cut.
    """
    session = PhotometrySession(session_series, one=MagicMock(), load_data=False)
    session.filepath = tmp_path / f'{session.eid}.h5'
    session.trials = _trials()
    session.save_h5(groups=['metadata'], mode='w')
    session.save_h5()
    return session.filepath


def _stub_store(tmp_path, monkeypatch, n_sessions=2):
    """Stand a store of metadata-only sessions up in front of `main`.

    `process` is replaced by a recorder, so `main` runs the catalog read and
    the filters and stops at the rebuild.

    Returns
    -------
    tuple of (list of str, list, unittest.mock.MagicMock)
        The eids written, the list each processed group and its `fn` are
        appended to, and the mock ONE the run was given.
    """
    from iblnm.data import PhotometrySessionGroup

    eids = [f'eid-rebuild-{n}' for n in range(1, n_sessions + 1)]
    one = MagicMock()
    for n, eid in enumerate(eids, start=1):
        row = pd.Series({
            'eid': eid,
            # Not `test_mouse`: that name is in config.SUBJECTS_TO_EXCLUDE and
            # the filter would drop it before the rebuild saw it.
            'subject': 'catalog_mouse',
            'start_time': f'2024-01-0{n}T10:00:00',
            'number': 1,
            'lab': 'test_lab',
            'task_protocol': '_iblrig_tasks_trainingChoiceWorld6.4.2',
            'session_type': 'training',
            'brain_region': ['VTA'],
            'hemisphere': ['l'],
            'target_NM': ['VTA-DA'],
        })
        session = PhotometrySession(row, one=one, load_data=False)
        session.save_h5(tmp_path / f'{eid}.h5', groups=['metadata'], mode='w')

    monkeypatch.setattr(rebuild, 'SESSIONS_H5_DIR', tmp_path)
    monkeypatch.setattr(rebuild, '_get_default_connection', lambda: one)

    processed = []
    monkeypatch.setattr(PhotometrySessionGroup, 'process',
                        lambda self, fn, **kwargs: processed.append((self, fn)))
    monkeypatch.setattr(PhotometrySessionGroup, 'collect_errors',
                        lambda self: pd.DataFrame())
    return eids, processed, one


class TestMain:
    """The pass reads the store, keeps every session, and re-cuts each one."""

    def test_every_stored_session_is_re_cut(self, tmp_path, monkeypatch):
        """The analysis filters are off, so a bare metadata store survives them."""
        eids, processed, _ = _stub_store(tmp_path, monkeypatch)

        rebuild.main([])

        group, fn = processed[0]
        assert sorted(group.sessions['eid']) == eids
        assert fn is rebuild.rebuild_responses

    def test_the_session_list_is_the_store_not_alyx(self, tmp_path, monkeypatch):
        """No session query: what is on disk is what is re-cut."""
        _, _, one = _stub_store(tmp_path, monkeypatch)

        rebuild.main([])

        assert not one.alyx.rest.called

    def test_the_session_type_flag_narrows_the_pass(self, tmp_path, monkeypatch):
        """Every stored session is `training`, so a `biased` run re-cuts none."""
        _, processed, _ = _stub_store(tmp_path, monkeypatch)

        rebuild.main(['--session-type', 'biased'])

        assert processed[0][0].sessions.empty


class TestParseArgs:
    def test_the_session_filters_parse(self):
        args = rebuild.parse_args(['--session-type', 'biased', '--workers', '4',
                                   '--target-NM', 'LC-NE', 'NBM-ACh'])

        assert args.session_type == ['biased']
        assert args.workers == 4
        assert args.target_NM == ['LC-NE', 'NBM-ACh']


class TestRebuildResponses:

    def test_cuts_the_responses_from_the_stored_band(self, stored_session):
        """Every event, every trial, cut from what the file already holds."""
        session = _reload(stored_session)

        rebuild.rebuild_responses(session)

        responses = session.photometry_responses['VTA']
        assert responses.coords['event'].values.tolist() == list(RESPONSE_EVENTS)
        assert responses.coords['trial'].values.tolist() == list(range(N_TRIALS))
        assert not np.isnan(responses.values).any()

    def test_writes_the_responses_and_keeps_the_other_products(
            self, stored_session):
        """The rebuilt cut lands in the file; the QC beside it is untouched."""
        session = _reload(stored_session)

        rebuild.rebuild_responses(session)

        with h5py.File(stored_session, 'r') as h5:
            assert 'photometry/VTA/responses' in h5
            assert (h5['photometry/VTA/raw/qc'].attrs['n_unique_samples_GCaMP']
                    == 1200.0)
        assert not session.errors

    def test_a_failed_input_is_logged_not_raised(self, session_series, tmp_path):
        """One session with nothing to cut from must not end the pass.

        The trials are the input both modalities read, so a session without
        them logs once against each rather than failing silently for the wheel.
        """
        from iblnm.validation import MissingRawData

        session = PhotometrySession(session_series, one=MagicMock(),
                                    load_data=False)
        session.filepath = tmp_path / f'{session.eid}.h5'
        with patch.object(PhotometrySession, 'fetch_trials',
                          side_effect=MissingRawData('_ibl_trials.table.pqt')):
            rebuild.rebuild_responses(session)

        assert [(e['product'], e['error_type']) for e in session.errors] == [
            ('photometry', 'MissingRawData'), ('wheel', 'MissingRawData')]
        assert not hasattr(session, 'photometry_responses')

    def test_builds_the_band_when_the_file_holds_none(self, trials_only_session):
        """With no stored preprocessed signal, the raw bands come from Alyx."""
        session = _reload(trials_only_session)
        with patch.object(PhotometrySession, 'load_raw_photometry',
                          side_effect=lambda: session.photometry.update(
                              _raw_bands())) as fetch:
            rebuild.rebuild_responses(session)

        assert fetch.call_count == 1
        assert session.photometry_responses['VTA'].sizes['trial'] == N_TRIALS
        with h5py.File(trials_only_session, 'r') as h5:
            assert 'photometry/VTA/responses' in h5


class TestRebuildWheelResponses:

    def test_cuts_the_wheel_responses_from_the_stored_velocity(
            self, stored_session):
        """The wheel is re-cut on its own event and its own per-trial window."""
        session = _reload(stored_session)

        rebuild.rebuild_responses(session)

        responses = session.wheel_responses['velocity']
        assert (responses.coords['event'].values.tolist()
                == list(WHEEL_RESPONSE_EVENTS))
        assert responses.coords['trial'].values.tolist() == list(range(N_TRIALS))

    def test_writes_the_wheel_responses_and_the_peak_velocity(
            self, stored_session):
        """Both wheel products land in the file, the reduction beside the cut."""
        session = _reload(stored_session)

        rebuild.rebuild_responses(session)

        assert session.wheel_peak_velocity.shape == (N_TRIALS,)
        with h5py.File(stored_session, 'r') as h5:
            assert 'wheel/velocity/responses' in h5
            assert 'wheel/velocity/peak_velocity' in h5
        assert not session.errors

    def test_a_failed_wheel_is_logged_against_the_wheel(self, stored_session):
        """A wheel with nothing to cut from must not cost the photometry."""
        from iblnm.validation import MissingRawData

        session = _reload(stored_session)
        with patch.object(PhotometrySession, 'load_wheel',
                          side_effect=MissingRawData('_ibl_wheel.position.npy')):
            del session.wheel_velocity
            rebuild.rebuild_responses(session)

        assert [(e['product'], e['error_type']) for e in session.errors] == [
            ('wheel', 'MissingRawData')]
        assert 'VTA' in session.photometry_responses
