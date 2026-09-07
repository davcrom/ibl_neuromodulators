"""Tests for scripts/download.py — the catalog phase, the build, and the CLI."""
from collections import Counter
from unittest.mock import MagicMock

import h5py
import numpy as np
import pandas as pd
import pytest
from one.alf.exceptions import ALFObjectNotFound

import scripts.download as download
from iblnm.config import MIN_NTRIALS
from iblnm.data import WHEEL_LABEL

# The synthetic session every build test runs on. The camera spans the session,
# the photometry spans much more of it, and every trial event falls inside both,
# so each product is cut from real samples rather than from NaN padding.
CAMERA_FS = 60.0
N_FRAMES = 3600            # one minute of video: long enough for the paw–wheel
SESSION_LENGTH = 60.0      # cross-correlation to fit its ±5 s lags into a third
KEYPOINTS = ['paw_l', 'paw_r', 'nose_tip', 'tongue_end_l', 'tongue_end_r']

# The ONE dataset names the build must not fetch twice.
CAMERA_TIMES = '_ibl_leftCamera.times.npy'
POSE = '_ibl_leftCamera.lightningPose.pqt'
MOTION_ENERGY = 'leftCamera.ROIMotionEnergy.npy'
PHOTOMETRY_SIGNAL = 'photometry.signal.pqt'
NEUROPHOTOMETRICS = '_neurophotometrics_fpData.raw.pqt'
TRIALS = '_ibl_trials.table.pqt'


@pytest.fixture
def session_series():
    return pd.Series({
        'eid': 'test-eid-build',
        'subject': 'test_mouse',
        'start_time': '2024-01-01T10:00:00',
        'number': 1,
        'lab': 'test_lab',
        'projects': [],
        'url': None,
        'session_n': 1,
        'task_protocol': '_iblrig_tasks_trainingChoiceWorld6.4.2',
        'session_type': 'training',
        'datasets': [],
    })


class _ZeroMetrics:
    """Stand-in for `iblphotometry.metrics`, scoring a clean recording.

    Every metric returns 0.0, so `validate_qc` passes: what the build tests
    check is which call happens when, not what the metrics say about synthetic
    data.
    """

    def __getattr__(self, _name):
        return lambda *args, **kwargs: 0.0


def _trials(n=MIN_NTRIALS):
    """Trials whose events sit inside both the camera's and the wheel's span.

    `MIN_NTRIALS` of them, so `validate_n_trials` passes and the blocks below
    it run.
    """
    rng = np.random.default_rng(0)
    stim_on = np.linspace(5.0, 50.0, n)
    contrast = rng.choice([0.0, 0.25, 1.0], size=n)
    left = rng.choice([True, False], size=n)
    return pd.DataFrame({
        'contrastLeft': np.where(left, contrast, np.nan),
        'contrastRight': np.where(left, np.nan, contrast),
        'choice': np.where(left, -1, 1),
        'feedbackType': np.ones(n),
        'probabilityLeft': np.full(n, 0.5),
        'stimOnTrigger_times': stim_on,
        'firstMovement_times': stim_on + 0.1,
        'response_times': stim_on + 0.4,
        'feedback_times': stim_on + 0.6,
    })


def _photometry_bands():
    """Two bands with a shared bleaching decay, on one region."""
    t = np.linspace(0, 600, 18000)
    rng = np.random.default_rng(1)
    bleaching = 1000 * np.exp(-t / 300)
    return {
        'GCaMP': pd.DataFrame({'VTA': bleaching + rng.standard_normal(len(t)) + 500},
                              index=t),
        'Isosbestic': pd.DataFrame(
            {'VTA': 0.8 * bleaching + 0.5 * rng.standard_normal(len(t)) + 400},
            index=t),
    }


def _camera_times():
    return np.arange(N_FRAMES) / CAMERA_FS


def _pose_frame():
    rng = np.random.default_rng(2)
    return pd.DataFrame({
        f'{keypoint}_{field}': rng.standard_normal(N_FRAMES)
        for keypoint in KEYPOINTS for field in ('x', 'y', 'likelihood')
    })


def _raw_wheel():
    """Irregular encoder samples spanning the session, ramping at 1 rad/s."""
    rng = np.random.default_rng(3)
    steps = rng.uniform(0.5, 1.5, int(SESSION_LENGTH * 250)) / 250
    timestamps = np.cumsum(steps)
    return {'timestamps': timestamps, 'position': timestamps.copy()}


def _tidy_qc(metric='n_unique_samples', region='VTA'):
    """One tidy `qc_signals` frame: a whole-signal row plus two window rows."""
    return pd.DataFrame({
        'band': ['GCaMP'] * 3,
        'brain_region': [region] * 3,
        'metric': [metric] * 3,
        'value': [0.85, 0.8, 0.9],
        'window': [np.nan, 0, 1],
    })


@pytest.fixture
def one_calls(monkeypatch):
    """A mock ONE serving every dataset, counting the fetches it is asked for.

    The photometry and trials fetches reach Alyx through their loader parent
    rather than through `one.load_dataset`, so the parent is replaced by a stub
    that fetches the dataset it stands for — leaving one counter for every raw
    source the build touches. The third-party QC computations are stubbed too:
    what is under test is which fetch happens how often, not what the metrics
    say about synthetic data.
    """
    from iblphotometry.fpio import PhotometrySessionLoader

    calls = Counter()

    def load_dataset(_eid, name, **_kwargs):
        calls[name.split('/')[-1]] += 1
        if CAMERA_TIMES in name:
            return _camera_times()
        if POSE in name:
            return _pose_frame()
        if MOTION_ENERGY in name:
            return np.random.default_rng(4).standard_normal(N_FRAMES)
        if NEUROPHOTOMETRICS in name:
            return pd.DataFrame({'times': [0.0, 1.0], 'raw': [1.0, 2.0]})
        if PHOTOMETRY_SIGNAL in name:
            return _photometry_bands()['GCaMP']
        if TRIALS in name:
            return _trials()
        raise ALFObjectNotFound(name)

    def load_object(_eid, name, **_kwargs):
        calls[name] += 1
        return _raw_wheel()

    def fetch_photometry(session, **_kwargs):
        session.one.load_dataset(session.eid, PHOTOMETRY_SIGNAL)
        session.photometry.update(_photometry_bands())

    def fetch_trials(session):
        session.one.load_dataset(session.eid, TRIALS)
        session.trials = _trials()

    monkeypatch.setattr(PhotometrySessionLoader, 'load_photometry',
                        fetch_photometry)
    monkeypatch.setattr(PhotometrySessionLoader, 'load_trials', fetch_trials)
    monkeypatch.setattr('iblnm.data.from_neurophotometrics_df_to_photometry_df',
                        lambda raw: raw)
    monkeypatch.setattr('iblnm.data.metrics', _ZeroMetrics())
    monkeypatch.setattr('iblnm.data.qc_signals', lambda *a, **k: _tidy_qc())

    one = MagicMock()
    one.load_dataset.side_effect = load_dataset
    one.load_object.side_effect = load_object
    return one, calls


@pytest.fixture
def session(session_series, tmp_path, one_calls):
    """A session whose H5 lives under `tmp_path` and whose ONE serves Alyx."""
    from iblnm.data import PhotometrySession

    one, _ = one_calls
    ps = PhotometrySession(session_series, one=one, load_data=False)
    ps.filepath = tmp_path / f'{ps.eid}.h5'
    ps.session_length = SESSION_LENGTH
    return ps


@pytest.fixture
def saved_groups(monkeypatch):
    """Record the `groups` argument of every `save_h5` call, then save for real."""
    from iblnm.data import PhotometrySession

    written = []
    save_h5 = PhotometrySession.save_h5

    def counting_save(self, fpath=None, groups=None, mode='a'):
        written.append(groups)
        return save_h5(self, fpath=fpath, groups=groups, mode=mode)

    monkeypatch.setattr(PhotometrySession, 'save_h5', counting_save)
    return written


class TestParseArgs:
    def test_the_session_filters_parse(self):
        args = download.parse_args(['--session-type', 'biased', '--workers', '4',
                                    '--target-NM', 'LC-NE', 'NBM-ACh'])

        assert args.session_type == ['biased']
        assert args.workers == 4
        assert args.target_NM == ['LC-NE', 'NBM-ACh']


class TestFetchesOncePerDataset:
    """The point of the build path: one Alyx trip per raw dataset per session."""

    def test_every_raw_dataset_is_fetched_once(self, session, one_calls):
        _, calls = one_calls

        download.build_session(session)

        assert calls[CAMERA_TIMES] == 1
        assert calls[POSE] == 1
        assert calls[MOTION_ENERGY] == 1
        assert calls[PHOTOMETRY_SIGNAL] == 1
        assert calls[NEUROPHOTOMETRICS] == 1
        assert calls[TRIALS] == 1
        assert calls['wheel'] == 1

    def test_a_clean_session_builds_every_block(self, session):
        """Nothing fails, and each block leaves its products on the session."""
        download.build_session(session)

        assert [e['error_type'] for e in session.errors] == []
        assert session.performance
        assert 'VTA' in session.photometry_responses
        assert WHEEL_LABEL in session.wheel_responses
        assert session.video_times_qc and session.pose_xcorr

    def test_the_wheel_cut_ends_at_the_choice(self, session):
        """The wheel window closes at `response_times`, 0.2 s before feedback."""
        download.build_session(session)

        matrix = session.wheel_responses[WHEEL_LABEL]
        tpts = matrix.coords['time'].to_numpy()
        # Every trial runs stimOnTrigger → response_times = 0.4 s here, so the
        # shared axis stops short of the 0.6 s feedback lag and nothing is
        # NaN-padded.
        assert tpts.max() < 0.4
        assert not np.isnan(matrix.values).any()

    def test_the_file_is_written_whole(self, session, saved_groups):
        """One truncating write of what has no data of its own, then the rest."""
        download.build_session(session)

        assert saved_groups == [['metadata', 'errors'], None]
        with h5py.File(session.filepath, 'r') as h5:
            assert set(h5) == {'metadata', 'errors', 'trials', 'photometry',
                               'wheel', 'video'}


class TestFailureBlocksItsModality:
    """Blocking comes from call order: a raise abandons the rest of its block."""

    def test_a_failed_fetch_stops_its_modality_only(self, session, monkeypatch):
        """One error against the block, and the blocks below it still run."""
        from iblnm.data import PhotometrySession
        from iblnm.validation import MissingExtractedData

        monkeypatch.setattr(
            PhotometrySession, 'fetch_photometry',
            MagicMock(side_effect=MissingExtractedData('photometry.signal.pqt')))

        download.build_session(session)

        assert [(e['product'], e['error_type']) for e in session.errors] == [
            ('photometry', 'MissingExtractedData')]
        assert not hasattr(session, 'photometry_responses')
        assert session.wheel_velocity is not None
        assert WHEEL_LABEL in session.wheel_responses


class TestMissingPhotometryIsClassified:
    """Absent photometry must log a blocking error, not a bare ALF miss.

    `config.ANALYSIS_QC_BLOCKERS` lists `MissingRawData`, not
    `ALFObjectNotFound`, so a session with no photometry data is excluded from
    analysis only if the block reaches the fetch that tells the two apart.
    """

    def test_absent_photometry_logs_missing_raw_data(self, session, one_calls):
        one, _ = one_calls
        served = one.load_dataset.side_effect

        def without_photometry(eid, name, **kwargs):
            if PHOTOMETRY_SIGNAL in name or NEUROPHOTOMETRICS in name:
                raise ALFObjectNotFound(name)
            return served(eid, name, **kwargs)

        one.load_dataset.side_effect = without_photometry

        download.build_session(session)

        assert [(e['product'], e['error_type']) for e in session.errors] == [
            ('photometry', 'MissingRawData')]


class TestNonFatalSteps:
    """The checks that degrade the build rather than abandoning their block."""

    def test_incomplete_events_are_dropped_from_the_cut(self, session,
                                                        monkeypatch):
        """The events named in the error go; the rest are still cut."""
        from iblnm.data import PhotometrySession
        from iblnm.validation import IncompleteEventTimes

        monkeypatch.setattr(
            PhotometrySession, 'validate_event_completeness',
            MagicMock(side_effect=IncompleteEventTimes(['feedback_times'])))

        download.build_session(session)

        events = session.photometry_responses['VTA'].coords['event']
        assert events.values.tolist() == ['stimOnTrigger_times']

    def test_a_block_structure_bug_is_fixed_and_the_block_runs_on(
            self, session, monkeypatch):
        """Logged, repaired, and the performance scoring below it still runs."""
        from iblnm.data import PhotometrySession
        from iblnm.validation import BlockStructureBug

        monkeypatch.setattr(
            PhotometrySession, 'validate_block_structure',
            MagicMock(side_effect=BlockStructureBug('non-uniform')))
        fix = MagicMock(return_value=True)
        monkeypatch.setattr(PhotometrySession, 'fix_block_structure', fix)

        download.build_session(session)

        assert [(e['product'], e['error_type']) for e in session.errors] == [
            ('trials', 'BlockStructureBug')]
        assert fix.called
        assert session.performance


class TestVideoSourcesFailSeparately:
    """Three independent Alyx queries: one missing source suppresses no other."""

    def test_a_missing_pose_still_fetches_the_motion_energy(self, session,
                                                           one_calls):
        one, calls = one_calls
        served = one.load_dataset.side_effect

        def without_pose(eid, name, **kwargs):
            if POSE in name:
                raise ALFObjectNotFound(name)
            return served(eid, name, **kwargs)

        one.load_dataset.side_effect = without_pose

        download.build_session(session)

        assert calls[MOTION_ENERGY] == 1
        assert session.motion_energy is not None
        assert session.video_times_qc
        assert {e['product'] for e in session.errors} == {'video/pose', 'video'}


def _write_metadata(h5_dir, eid, regions, one):
    """Write one session's `metadata` group, as the Alyx query pass does."""
    from iblnm.data import PhotometrySession

    row = pd.Series({
        'eid': eid,
        # Not `test_mouse`: that name is in config.SUBJECTS_TO_EXCLUDE and
        # `main`'s filter drops it before the build ever sees it.
        'subject': 'catalog_mouse',
        'start_time': f'2024-01-0{eid[-1]}T10:00:00',
        'number': 1,
        'lab': 'test_lab',
        'task_protocol': '_iblrig_tasks_trainingChoiceWorld6.4.2',
        'session_type': 'training',
        'brain_region': regions,
        'hemisphere': ['l'] * len(regions),
        'target_NM': ['VTA-DA'] * len(regions),
    })
    ps = PhotometrySession(row, one=one, load_data=False)
    ps.save_h5(h5_dir / f'{eid}.h5', groups=['metadata'], mode='w')


def _stub_store(tmp_path, monkeypatch, regions_per_session):
    """Stand a store of metadata-only sessions up in front of `download.main`.

    One session per entry of `regions_per_session`, its brain regions taken
    from that entry, written into `tmp_path` and served by a mock ONE. The
    group's `process` is replaced by a recorder, so `main` runs the whole
    catalog phase and stops at the build.

    Returns
    -------
    tuple of (list of str, list)
        The eids written, in order, and the list the recorder appends each
        processed group to.
    """
    from iblnm.data import PhotometrySessionGroup

    eids = [f'eid-catalog-{n}' for n in range(1, len(regions_per_session) + 1)]
    one = MagicMock()
    one.alyx.rest.return_value = [{'id': eid} for eid in eids]
    for eid, regions in zip(eids, regions_per_session):
        _write_metadata(tmp_path, eid, regions, one)

    monkeypatch.setattr(download, 'SESSIONS_H5_DIR', tmp_path)
    monkeypatch.setattr(download, 'SESSIONS_FPATH', tmp_path / 'sessions.pqt')
    monkeypatch.setattr(download, '_get_default_connection', lambda: one)

    processed = []
    monkeypatch.setattr(PhotometrySessionGroup, 'process',
                        lambda self, *a, **k: processed.append(self))
    monkeypatch.setattr(PhotometrySessionGroup, 'collect_errors',
                        lambda self: pd.DataFrame())
    return eids, processed


class TestCatalogPhase:
    """One group is read from the store, fixed, saved, filtered and built."""

    @pytest.fixture
    def stored(self, tmp_path, monkeypatch):
        """Two stored sessions, one of them missing its recorded regions.

        The empty row is the case the phase exists for: the fixups fill its
        parallel columns from its subject's other session, and they have to
        reach the object the build iterates.
        """
        return _stub_store(tmp_path, monkeypatch, (['VTA'], []))

    def test_the_built_group_carries_the_fixups(self, stored):
        """`process` iterates the fixed catalog, not a copy left behind."""
        eids, processed = stored

        download.main([])

        assert len(processed) == 1
        assert processed[0].sessions['target_NM'].tolist() == [['VTA-DA']] * 2
        assert sorted(processed[0].sessions['eid']) == eids

    def test_the_catalog_is_written(self, stored, tmp_path):
        eids, _ = stored

        download.main([])

        catalog = pd.read_parquet(tmp_path / 'sessions.pqt')
        assert sorted(catalog['eid']) == eids

    def test_the_store_is_not_scanned(self, stored, monkeypatch):
        """Nothing in the phase reads what `complete_catalog` collects."""
        from iblnm.data import PhotometrySessionGroup

        def refuse(self):
            raise AssertionError('complete_catalog was called')

        monkeypatch.setattr(PhotometrySessionGroup, 'complete_catalog', refuse)

        download.main([])


class TestTargetNMFilter:
    """`--target-NM` narrows which sessions are built, never what is built."""

    @pytest.fixture
    def stored(self, tmp_path, monkeypatch):
        """Three sessions: LC only, VTA only, and both fibers in one session."""
        return _stub_store(tmp_path, monkeypatch,
                           (['LC'], ['VTA'], ['LC', 'VTA']))

    def test_a_session_is_kept_for_any_of_its_recordings(self, stored):
        """The mixed session is built whole, on the strength of its LC fiber."""
        eids, processed = stored

        download.main(['--target-NM', 'LC-NE'])

        built = processed[0].sessions
        assert sorted(built['eid']) == [eids[0], eids[2]]
        assert built.set_index('eid').loc[eids[2], 'brain_region'] == ['LC', 'VTA']

    def test_no_flag_builds_every_target(self, stored):
        eids, processed = stored

        download.main([])

        assert sorted(processed[0].sessions['eid']) == eids
