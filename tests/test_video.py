"""Tests for the video products: the three raw datasets, preprocessed, responses."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from one.alf.exceptions import ALFObjectNotFound


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_session_series():
    return pd.Series({
        'eid': 'test-eid-video',
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


CAMERA_FS = 60.0
N_FRAMES = 600
KEYPOINTS = ['paw_l', 'paw_r', 'nose_tip', 'tongue_end_l', 'tongue_end_r']


def _camera_times():
    """Frame times at CAMERA_FS, deliberately not the POSE_FS grid."""
    return np.arange(N_FRAMES) / CAMERA_FS


def _pose_frame(seed=0):
    """LightningPose columns: x, y and likelihood for each keypoint."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        f'{keypoint}_{field}': rng.standard_normal(N_FRAMES)
        for keypoint in KEYPOINTS for field in ('x', 'y', 'likelihood')
    })


def _motion_energy(seed=1):
    return np.random.default_rng(seed).standard_normal(N_FRAMES)


def _raw_wheel(seed=2):
    """Irregular encoder samples spanning the camera's frames, ramping at 1 rad/s."""
    rng = np.random.default_rng(seed)
    steps = rng.uniform(0.5, 1.5, int(N_FRAMES / CAMERA_FS * 250)) / 250
    timestamps = np.cumsum(steps)
    return {'timestamps': timestamps, 'position': timestamps.copy()}


def _one_serving(times=None, pose=None, motion_energy=None, wheel=None):
    """Mock ONE dispatching on the dataset name, raising where a source is False.

    Each of the three video datasets is fetched by its own `load_dataset` call,
    so a test drops one source by passing False for it and leaves the others
    served. The wheel arrives through `load_object` instead, because
    `video/pose/qc` is cross-modal and needs it alongside the pose.
    """
    payloads = {
        'times': _camera_times() if times is None else times,
        'lightningPose': _pose_frame() if pose is None else pose,
        'ROIMotionEnergy': _motion_energy() if motion_energy is None else motion_energy,
    }

    def load_dataset(_eid, name, **_kwargs):
        for key, payload in payloads.items():
            if key in name:
                if payload is False:
                    raise ALFObjectNotFound(name)
                return payload
        raise ALFObjectNotFound(name)

    def load_object(_eid, name, **_kwargs):
        if wheel is False:
            raise ALFObjectNotFound(name)
        return _raw_wheel() if wheel is None else wheel

    one = MagicMock()
    one.load_dataset.side_effect = load_dataset
    one.load_object.side_effect = load_object
    return one


def _make_session(mock_session_series, tmp_path, one=None):
    """Session whose H5 lives in `tmp_path` and whose ONE serves the video data."""
    from iblnm.data import PhotometrySession
    ps = PhotometrySession(mock_session_series,
                           one=_one_serving() if one is None else one,
                           load_data=False)
    ps.filepath = tmp_path / f'{ps.eid}.h5'
    return ps


# ─────────────────────────────────────────────────────────────────────────────
# video/times, video/pose, video/motion_energy — three independent raw products
# ─────────────────────────────────────────────────────────────────────────────

class TestFetchVideoTier:
    """The three video fetches: Alyx in, session attribute out, nothing else."""

    def test_fetch_pose_writes_nothing(self, mock_session_series, tmp_path):
        """A fetch leaves the session's H5 exactly as it found it."""
        ps = _make_session(mock_session_series, tmp_path)
        ps.session_length = 5.0
        ps.load_video_times_qc()          # gives the file something to hold
        before = ps.filepath.read_bytes()

        pose = ps.fetch_pose()

        assert pose is ps.pose
        assert ps.filepath.read_bytes() == before

    def test_fetch_camera_times_missing_raises(self, mock_session_series, tmp_path):
        from iblnm.validation import MissingVideoTimestamps
        ps = _make_session(mock_session_series, tmp_path, _one_serving(times=False))

        with pytest.raises(MissingVideoTimestamps):
            ps.fetch_camera_times()

    def test_fetch_motion_energy_refetches_over_a_stored_product(
            self, mock_session_series, tmp_path, monkeypatch):
        """A fetch never consults the store, however current the store is."""
        monkeypatch.setattr('iblnm.data.store_raw', True)
        ps = _make_session(mock_session_series, tmp_path)
        ps.load_motion_energy()
        ps.save_h5(groups=['video'])
        assert ps.product_status('video/motion_energy') == 'current'

        ps.load_motion_energy()
        stored_reads = ps.one.load_dataset.call_count
        ps.fetch_motion_energy()

        assert ps.one.load_dataset.call_count == stored_reads + 1


class TestRawVideoProducts:
    """What the three datasets do once `config.store_raw` keeps them.

    The three raw loaders read the store but never write it — `build_session`
    saves the modality once, after its block — so these tests save explicitly.
    The gate itself — that nothing is written with `store_raw` off — is
    `tests/test_data.py::TestStoreRawGating`.
    """

    @pytest.fixture(autouse=True)
    def keep_raw(self, monkeypatch):
        monkeypatch.setattr('iblnm.data.store_raw', True)

    def test_each_dataset_stores_and_stamps_on_its_own(self, mock_session_series,
                                                       tmp_path):
        """Fetching all three leaves three separately stamped products."""
        ps = _make_session(mock_session_series, tmp_path)
        ps.load_camera_times()
        ps.load_pose()
        ps.load_motion_energy()
        ps.save_h5(groups=['video'])

        for product in ('video/times', 'video/pose', 'video/motion_energy'):
            assert ps.product_status(product) == 'current'

    def test_roundtrips_each_dataset_unchanged(self, mock_session_series, tmp_path):
        times, pose, motion_energy = _camera_times(), _pose_frame(), _motion_energy()
        ps = _make_session(mock_session_series, tmp_path,
                           _one_serving(times, pose, motion_energy))
        ps.load_camera_times()
        ps.load_pose()
        ps.load_motion_energy()
        ps.save_h5(groups=['video'])

        fresh = _make_session(mock_session_series, tmp_path)
        np.testing.assert_allclose(fresh.load_camera_times(), times)
        # check_like: keypoint columns are addressed by name everywhere, and H5
        # hands them back in its own (alphabetical) order.
        pd.testing.assert_frame_equal(fresh.load_pose(), pose, check_like=True)
        np.testing.assert_allclose(fresh.load_motion_energy(), motion_energy)
        fresh.one.load_dataset.assert_not_called()

    def test_missing_pose_leaves_the_other_two_stored(self, mock_session_series,
                                                      tmp_path):
        """MissingLP takes down `video/pose` alone, not the whole modality."""
        from iblnm.validation import MissingLP
        ps = _make_session(mock_session_series, tmp_path, _one_serving(pose=False))
        ps.load_camera_times()
        with pytest.raises(MissingLP):
            ps.load_pose()
        ps.load_motion_energy()
        ps.save_h5(groups=['video'])

        assert ps.product_status('video/times') == 'current'
        assert ps.product_status('video/motion_energy') == 'current'
        assert ps.product_status('video/pose') == 'absent'

    def test_absent_before_anything_is_stored(self, mock_session_series, tmp_path):
        ps = _make_session(mock_session_series, tmp_path)
        for product in ('video/times', 'video/pose', 'video/motion_energy'):
            assert ps.product_status(product) == 'absent'

    def test_rebuild_skips_the_stored_dataset(self, mock_session_series, tmp_path):
        ps = _make_session(mock_session_series, tmp_path)
        ps.load_pose()
        ps.rebuild.add('video/pose')
        ps.load_pose()

        assert ps.one.load_dataset.call_count == 2

    def test_redownloading_raw_clears_the_manual_verdicts(
            self, mock_session_series, tmp_path):
        """A verdict describes the frames it was passed on, so refetching them
        drops it rather than letting it stand for data nobody looked at."""
        ps = _make_session(mock_session_series, tmp_path)
        ps.load_pose()
        ps.set_manual_qc('qc_lp', 'FAIL')

        ps.rebuild.add('video/pose')
        ps.load_pose()

        assert ps.video_manual_qc == {}
        fresh = _make_session(mock_session_series, tmp_path)
        fresh.load_h5(groups=['video'])
        assert fresh.video_manual_qc == {}

    def test_reading_the_stored_dataset_keeps_the_manual_verdicts(
            self, mock_session_series, tmp_path):
        """Only a fetch clears them: loading the stored frames back is not a
        re-download and leaves the verdict on the data it was set for."""
        ps = _make_session(mock_session_series, tmp_path)
        ps.load_pose()
        ps.save_h5(groups=['video'])
        ps.set_manual_qc('qc_lp', 'FAIL')

        fresh = _make_session(mock_session_series, tmp_path)
        fresh.load_pose()
        fresh.load_h5(groups=['video'])

        assert fresh.video_manual_qc == {'qc_lp': 'FAIL'}


class TestExtractVideoTier:
    """The three video computations: attributes in, attributes out, no I/O.

    Each reads what a fetch already put on the session, assigns its result, and
    leaves the store alone; the matching `load_*` is what writes.
    """

    def test_run_video_times_qc_writes_nothing(self, mock_session_series, tmp_path):
        ps = _make_session(mock_session_series, tmp_path)
        ps.session_length = 5.0
        ps.load_pose()
        ps.save_h5(groups=['video'])       # gives the file something to hold
        ps.fetch_camera_times()
        before = ps.filepath.read_bytes()

        measures = ps.run_video_times_qc()

        assert measures is ps.video_times_qc
        assert set(measures) == {'length_discrepancy', 'framerate_from_tpts'}
        assert ps.filepath.read_bytes() == before

    def test_extract_movement_signals_writes_nothing(self, mock_session_series,
                                                     tmp_path):
        ps = _make_session(mock_session_series, tmp_path)
        ps.session_length = 5.0
        ps.load_video_times_qc()           # gives the file something to hold
        ps._load_raw_video_sources()
        before = ps.filepath.read_bytes()

        signals = ps.extract_movement_signals()

        assert signals is ps.movement_signals
        assert ps.filepath.read_bytes() == before

    def test_run_pose_qc_writes_nothing(self, mock_session_series, tmp_path):
        ps = _make_session(mock_session_series, tmp_path, _xcorr_one())
        ps.session_length = 5.0
        ps.load_video_times_qc()           # gives the file something to hold
        ps.fetch_camera_times()
        ps.fetch_pose()
        ps.load_wheel()
        before = ps.filepath.read_bytes()

        xcorr = ps.run_pose_qc()

        assert xcorr is ps.pose_xcorr
        assert ps.filepath.read_bytes() == before


# ─────────────────────────────────────────────────────────────────────────────
# video/times/qc — the camera-clock measures
# ─────────────────────────────────────────────────────────────────────────────

class TestVideoTimesQcProduct:

    def test_measures_are_stamped_attrs_of_the_times_qc_group(
            self, mock_session_series, tmp_path):
        """Both measures live under video/times/qc, none on the video group."""
        import h5py
        ps = _make_session(mock_session_series, tmp_path)
        ps.session_length = 5.0
        measures = ps.load_video_times_qc()

        # 600 frames at 60 Hz span 599/60 s; session_length is 5 s.
        assert measures['length_discrepancy'] == pytest.approx(599 / 60 - 5.0)
        assert measures['framerate_from_tpts'] == pytest.approx(1 / 60)
        assert ps.product_status('video/times/qc') == 'current'
        with h5py.File(ps.filepath, 'r') as f:
            assert set(measures) <= set(f['video/times/qc'].attrs)
            assert not set(measures) & set(f['video'].attrs)

    def test_a_long_video_is_logged_not_raised(self, mock_session_series, tmp_path):
        """A video outrunning the session records the mismatch and carries on.

        The traces cut from an over-long video have always been kept, so the
        check belongs in the session's error log rather than in a raise.
        """
        from iblnm.config import LENGTH_MISMATCH_THRESHOLD
        ps = _make_session(mock_session_series, tmp_path)
        ps.session_length = (N_FRAMES - 1) / CAMERA_FS - LENGTH_MISMATCH_THRESHOLD - 80
        ps.fetch_camera_times()

        measures = ps.run_video_times_qc()

        assert measures['length_discrepancy'] == pytest.approx(
            LENGTH_MISMATCH_THRESHOLD + 80)
        assert [(e['product'], e['error_type']) for e in ps.errors] == [
            ('video/times/qc', 'VideoLengthError')]

    def test_a_video_within_the_threshold_logs_nothing(self, mock_session_series,
                                                       tmp_path):
        from iblnm.config import LENGTH_MISMATCH_THRESHOLD
        ps = _make_session(mock_session_series, tmp_path)
        ps.session_length = (N_FRAMES - 1) / CAMERA_FS - 10
        ps.fetch_camera_times()

        measures = ps.run_video_times_qc()

        assert measures['length_discrepancy'] == pytest.approx(10)
        assert LENGTH_MISMATCH_THRESHOLD > 10
        assert ps.errors == []

    def test_a_discrepancy_at_the_threshold_is_logged(self, mock_session_series,
                                                      tmp_path):
        """The threshold itself counts as a mismatch — the comparison is `>=`."""
        from iblnm.config import LENGTH_MISMATCH_THRESHOLD
        ps = _make_session(mock_session_series, tmp_path)
        ps.session_length = (N_FRAMES - 1) / CAMERA_FS - LENGTH_MISMATCH_THRESHOLD
        ps.fetch_camera_times()

        ps.run_video_times_qc()

        assert [e['error_type'] for e in ps.errors] == ['VideoLengthError']

    def test_reads_stored_product_without_refetching(self, mock_session_series,
                                                     tmp_path):
        ps = _make_session(mock_session_series, tmp_path)
        ps.session_length = 5.0
        built = ps.load_video_times_qc()

        fresh = _make_session(mock_session_series, tmp_path)
        assert fresh.load_video_times_qc() == built
        fresh.one.load_dataset.assert_not_called()

    def test_alyx_qc_labels_are_never_written(self, mock_session_series, tmp_path,
                                              monkeypatch):
        """The eight VIDEO_QC_COLS are fetched live, so no save path stores them.

        No stamp could tell a stored copy had gone stale: the labels change when
        IBL re-runs its QC and no parameter in this repo feeds them. `store_raw`
        is on so that `video/pose` exists to be checked alongside the rest.
        """
        import h5py
        from iblnm.config import VIDEO_QC_COLS
        monkeypatch.setattr('iblnm.data.store_raw', True)
        ps = _make_session(mock_session_series, tmp_path)
        ps.session_length = 5.0
        ps.video_qc = {col: 'PASS' for col in VIDEO_QC_COLS}
        ps.load_video_times_qc()
        ps.load_pose()
        ps.save_h5(groups=['video'])

        with h5py.File(ps.filepath, 'r') as f:
            written = {name for group in (f['video'], f['video/times/qc'],
                                          f['video/pose'])
                       for name in group.attrs}
        assert not written & set(VIDEO_QC_COLS)


# ─────────────────────────────────────────────────────────────────────────────
# video/pose/qc — the paw–wheel cross-correlation, cross-modal with the wheel
# ─────────────────────────────────────────────────────────────────────────────

XCORR_FIELDS = ('functions', 'lags', 'peak_lags', 'drift')
# Each session third must outlast the ±CROSSCORR_LAG_WINDOW lag range, so the
# cross-correlation needs a longer recording than the other video products do.
XCORR_N_FRAMES = 60 * int(CAMERA_FS)


def _xcorr_one(wheel=None):
    """ONE serving a recording long enough for the per-third cross-correlation."""
    times = np.arange(XCORR_N_FRAMES) / CAMERA_FS
    pose = pd.DataFrame({
        f'{keypoint}_{field}': np.random.default_rng(4).standard_normal(
            XCORR_N_FRAMES)
        for keypoint in KEYPOINTS for field in ('x', 'y', 'likelihood')
    })
    wheel_times = np.cumsum(
        np.random.default_rng(5).uniform(0.5, 1.5, 60 * 250) / 250)
    return _one_serving(
        times=times, pose=pose, motion_energy=np.zeros(XCORR_N_FRAMES),
        wheel=({'timestamps': wheel_times, 'position': wheel_times.copy()}
               if wheel is None else wheel))


class TestPoseQcProduct:

    def test_roundtrips_the_xcorr_fields_and_reports_current(
            self, mock_session_series, tmp_path, monkeypatch):
        monkeypatch.setattr('iblnm.data.store_raw', True)
        ps = _make_session(mock_session_series, tmp_path, _xcorr_one())
        xcorr = ps.load_pose_qc()

        assert set(xcorr) == set(XCORR_FIELDS)
        assert ps.product_status('video/pose/qc') == 'current'
        # The QC group hangs under `video/pose`, so writing it must not leave
        # the raw pose product looking unstamped.
        assert ps.product_status('video/pose') == 'current'

        fresh = _make_session(mock_session_series, tmp_path, _xcorr_one())
        stored = fresh.load_pose_qc()
        for field in ('functions', 'lags', 'peak_lags'):
            np.testing.assert_allclose(stored[field], xcorr[field])
        assert stored['drift'] == pytest.approx(xcorr['drift'])
        fresh.one.load_dataset.assert_not_called()

    def test_goes_stale_when_the_wheel_rate_changes(self, mock_session_series,
                                                    tmp_path):
        """The wheel is an input, so its parameters ride in the pose-QC stamp."""
        import h5py
        from iblnm.data import _write_stamp
        from iblnm.validation import StaleProduct
        ps = _make_session(mock_session_series, tmp_path, _xcorr_one())
        ps.load_pose_qc()
        with h5py.File(ps.filepath, 'a') as f:
            _write_stamp(f['video/pose/qc'],
                         ps.spec['video/pose/qc']
                         | {'wheel/preprocessed.fs': 1000})

        assert ps.product_status('video/pose/qc') == 'stale'
        with pytest.raises(StaleProduct, match='video/pose/qc'):
            ps.load_pose_qc()

    def test_missing_wheel_blocks_the_build_and_names_the_wheel(
            self, mock_session_series, tmp_path):
        """Good pose is not enough: the error must point at the wheel, not pose."""
        from iblnm.validation import MissingRawData
        ps = _make_session(mock_session_series, tmp_path, _xcorr_one(wheel=False))

        with pytest.raises(MissingRawData, match='encoderPositions'):
            ps.load_pose_qc()
        assert ps.product_status('video/pose/qc') == 'absent'


# ─────────────────────────────────────────────────────────────────────────────
# video/preprocessed — the movement channels on the POSE_FS grid
# ─────────────────────────────────────────────────────────────────────────────

class TestPreprocessedVideoProduct:

    def test_every_channel_shares_one_pose_fs_time_base(self, mock_session_series,
                                                        tmp_path):
        """One series per POSE_MEASURES label plus motion_energy, on one grid."""
        from iblnm.config import POSE_FS, POSE_MEASURES
        ps = _make_session(mock_session_series, tmp_path)
        signals = ps._movement_signals()

        assert set(signals) == set(POSE_MEASURES) | {'motion_energy'}
        reference = next(iter(signals.values())).index.to_numpy()
        np.testing.assert_allclose(np.diff(reference), 1 / POSE_FS, atol=1e-9)
        for signal in signals.values():
            np.testing.assert_allclose(signal.index.to_numpy(), reference)

    def test_roundtrips_through_the_preprocessed_groups(self, mock_session_series,
                                                        tmp_path):
        """A second session over the same file reads them and never fetches."""
        ps = _make_session(mock_session_series, tmp_path)
        built = ps._movement_signals()
        assert ps.product_status('video/preprocessed') == 'current'

        fresh = _make_session(mock_session_series, tmp_path)
        reloaded = fresh._movement_signals()

        fresh.one.load_dataset.assert_not_called()
        assert set(reloaded) == set(built)
        for label, signal in built.items():
            pd.testing.assert_series_equal(reloaded[label], signal)

    def test_missing_pose_leaves_only_the_motion_energy_channel(
            self, mock_session_series, tmp_path):
        """No LP → the motion_energy channel alone, and the failure is logged
        against `video/pose`."""
        ps = _make_session(mock_session_series, tmp_path, _one_serving(pose=False))
        signals = ps._movement_signals()

        assert set(signals) == {'motion_energy'}
        assert [(e['product'], e['error_type']) for e in ps.errors] == [
            ('video/pose', 'MissingLP')]

    def test_missing_motion_energy_leaves_only_the_keypoint_channels(
            self, mock_session_series, tmp_path):
        from iblnm.config import POSE_MEASURES
        ps = _make_session(mock_session_series, tmp_path,
                           _one_serving(motion_energy=False))
        signals = ps._movement_signals()

        assert set(signals) == set(POSE_MEASURES)
        assert [(e['product'], e['error_type']) for e in ps.errors] == [
            ('video/motion_energy', 'MissingMotionEnergy')]

    def test_missing_camera_times_is_fatal(self, mock_session_series, tmp_path):
        """Without frame times nothing can be placed on a clock, so it raises."""
        from iblnm.validation import MissingVideoTimestamps
        ps = _make_session(mock_session_series, tmp_path, _one_serving(times=False))
        with pytest.raises(MissingVideoTimestamps):
            ps._movement_signals()

    def test_raises_on_stale_stamp(self, mock_session_series, tmp_path):
        """Channels resampled at a rate that changed are not reused."""
        import h5py
        from iblnm.data import _write_stamp
        from iblnm.validation import StaleProduct
        ps = _make_session(mock_session_series, tmp_path)
        labels = ps._movement_signals()
        # A rate change restamps every channel, so stale them all.
        with h5py.File(ps.filepath, 'a') as h5:
            for label in labels:
                _write_stamp(h5[f'video/{label}/preprocessed'],
                             ps.spec['video/preprocessed'] | {'fs': 1})

        assert ps.product_status('video/preprocessed') == 'stale'
        with pytest.raises(StaleProduct, match='video/preprocessed'):
            ps._movement_signals()


# ─────────────────────────────────────────────────────────────────────────────
# video/responses — every channel cut at every movement event
# ─────────────────────────────────────────────────────────────────────────────

def _make_trials():
    """Three trials whose events all fall inside the synthetic camera window."""
    return pd.DataFrame({
        'trial': [0, 1, 2],
        'stimOn_times':        [2.0, 4.0, 6.0],
        'firstMovement_times': [2.2, 4.2, 6.2],
        'feedback_times':      [2.5, 4.5, 6.5],
    })


class TestVideoResponsesProduct:

    @pytest.fixture
    def video_session(self, mock_session_series, tmp_path):
        ps = _make_session(mock_session_series, tmp_path)
        ps.trials = _make_trials()
        return ps

    def test_every_channel_carries_the_full_event_axis(self, video_session):
        """The event axis is not narrowed per channel at extraction time."""
        from iblnm.config import MOVEMENT_EVENTS, POSE_MEASURES
        responses = video_session.load_responses('video')

        assert set(responses) == set(POSE_MEASURES) | {'motion_energy'}
        for matrix in responses.values():
            assert list(matrix.coords['event'].values) == list(MOVEMENT_EVENTS)

    def test_label2event_selects_a_populated_cell_for_every_channel(
            self, video_session):
        """Each channel's own response event is picked at read time."""
        from iblnm.config import LABEL2EVENT
        responses = video_session.load_responses('video')

        for label, matrix in responses.items():
            cell = matrix.sel(event=LABEL2EVENT[label])
            assert cell.sizes['trial'] == 3
            assert np.isfinite(cell.values).any()

    def test_trial_coord_comes_from_the_trials_table(self, video_session):
        responses = video_session.load_responses('video')
        np.testing.assert_array_equal(
            responses['nose'].coords['trial'].to_numpy(), [0, 1, 2])

    def test_roundtrips_through_h5(self, video_session, mock_session_series,
                                   tmp_path):
        import xarray as xr
        built = video_session.load_responses('video')

        fresh = _make_session(mock_session_series, tmp_path)
        reloaded = fresh.load_responses('video')

        fresh.one.load_dataset.assert_not_called()
        assert set(reloaded) == set(built)
        for label, matrix in built.items():
            xr.testing.assert_allclose(reloaded[label], matrix)

    def test_motion_energy_only_session_has_no_keypoint_channels(
            self, mock_session_series, tmp_path):
        """Motion energy but no pose → one channel, cut like any other."""
        ps = _make_session(mock_session_series, tmp_path, _one_serving(pose=False))
        ps.trials = _make_trials()
        responses = ps.load_responses('video')

        assert set(responses) == {'motion_energy'}
        assert ps.product_status('video/responses') == 'current'
