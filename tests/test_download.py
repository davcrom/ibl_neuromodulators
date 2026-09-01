"""Tests for scripts/download.py — the catalog phase, the build, and the CLI."""
from collections import Counter
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from one.alf.exceptions import ALFObjectNotFound

import scripts.download as download

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


def _trials(n=8):
    """Trials whose events sit inside both the camera's and the wheel's span."""
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
        'stimOn_times': stim_on,
        'firstMovement_times': stim_on + 0.1,
        'feedback_times': stim_on + 0.4,
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
    monkeypatch.setattr('iblnm.data.metrics', MagicMock())
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
    def test_a_product_name_is_rejected(self):
        """Both flags name modalities now; a product name is not one."""
        with pytest.raises(SystemExit):
            download.parse_args(['--skip', 'video/pose'])
        with pytest.raises(SystemExit):
            download.parse_args(['--rebuild', 'photometry/preprocessed'])

    def test_modalities_and_defaults(self):
        args = download.parse_args(['--skip', 'video', '--workers', '4'])

        assert args.skip == ['video']
        assert args.rebuild == []
        assert args.workers == 4
        assert args.retry_failed is False


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

    def test_every_product_is_built(self, session):
        """A session with everything to fetch ends holding every product."""
        built = download.build_session(session)

        assert [e['error_type'] for e in session.errors] == []
        assert built == {product: 'built' for steps in
                         download.BUILD_STEPS.values() for product, _ in steps}
        assert all(session.product_status(product) == 'current'
                   for product in built)

    def test_each_modality_is_written_once(self, session, saved_groups):
        download.build_session(session)

        assert Counter(group for groups in saved_groups for group in groups) == {
            'trials': 1, 'photometry': 1, 'wheel': 1, 'video': 1}


class TestFailureBlocksItsModality:
    """Blocking comes from call order: a raise abandons the rest of its block."""

    def test_a_failed_fetch_stops_its_modality_only(self, session, monkeypatch):
        """The error names the product being made, and the other blocks run on."""
        from iblnm.data import PhotometrySession
        from iblnm.validation import MissingExtractedData

        monkeypatch.setattr(
            PhotometrySession, 'fetch_photometry',
            MagicMock(side_effect=MissingExtractedData('photometry.signal.pqt')))

        built = download.build_session(session)

        assert built['photometry/raw/qc'] == 'failed'
        assert 'photometry/preprocessed' not in built
        assert 'photometry/responses' not in built
        assert built['trials/performance'] == 'built'
        assert [(e['product'], e['error_type']) for e in session.errors] == [
            ('photometry/raw/qc', 'MissingExtractedData')]


class TestSettledFailures:
    """A product tried, failed and left absent is not attempted again."""

    @staticmethod
    def _drop_pose(one, calls):
        """Serve every dataset but the LightningPose one."""
        served = one.load_dataset.side_effect

        def without_pose(eid, name, **kwargs):
            if POSE in name:
                calls[name.split('/')[-1]] += 1
                raise ALFObjectNotFound(name)
            return served(eid, name, **kwargs)

        one.load_dataset.side_effect = without_pose

    def test_a_recorded_failure_skips_its_modality(self, session, one_calls):
        """A session with no LP settles video on the first pass and skips it."""
        one, calls = one_calls
        self._drop_pose(one, calls)

        download.build_session(session)
        assert {e['product'] for e in session.errors} == {'video/pose',
                                                          'video/pose/qc'}
        assert session.product_status('video/pose') == 'absent'
        calls.clear()

        assert download.build_session(session) == {}
        assert calls[CAMERA_TIMES] == 0

    def test_retry_failed_attempts_it_again(self, session, one_calls):
        """--retry-failed is how the user asks for the refetch anyway."""
        one, calls = one_calls
        self._drop_pose(one, calls)
        download.build_session(session)
        calls.clear()

        built = download.build_session(session, retry_failed=True)

        assert built['video/times/qc'] == 'built'
        assert built['video/pose/qc'] == 'failed'
        assert calls[CAMERA_TIMES] == 1


class TestModalityScope:
    """`modalities` is what the `--skip` flag cuts: whole blocks, not products."""

    def test_only_the_named_modality_is_built(self, session):
        import h5py

        built = download.build_session(session, modalities=('photometry',))

        assert set(built) == {product for product, _
                              in download.BUILD_STEPS['photometry']}
        with h5py.File(session.filepath, 'r') as h5:
            assert 'photometry' in h5
            assert 'wheel' not in h5
            assert 'video' not in h5
