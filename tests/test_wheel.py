"""Tests for wheel velocity extraction (PhotometrySession methods + wheel.py script)."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_session_series():
    return pd.Series({
        'eid': 'test-eid-wheel',
        'subject': 'test_mouse',
        'start_time': '2024-01-01T10:00:00',
        'number': 1,
        'lab': 'test_lab',
        'projects': [],
        'url': None,
        'session_n': 1,
        'task_protocol': '_iblrig_tasks_trainingChoiceWorld6.4.2',
        'session_type': 'training',
        'datasets': [
            'raw_behavior_data/_iblrig_taskData.raw.jsonable',
            'alf/_ibl_wheel.position.npy',
        ],
    })


def _make_session(mock_session_series):
    from iblnm.data import PhotometrySession
    return PhotometrySession(mock_session_series, one=MagicMock(), load_data=False)


def _make_wheel_df(duration=5.0, fs=1000):
    """Constant-velocity wheel at `fs` Hz."""
    t = np.arange(0, duration, 1 / fs, dtype=np.float32)
    vel = np.ones(len(t), dtype=np.float32)
    return pd.DataFrame({'times': t, 'velocity': vel})


# ─────────────────────────────────────────────────────────────────────────────
# extract_wheel_velocity
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractWheelVelocity:

    def test_output_shape(self, mock_session_series):
        """Matrix shape is (n_trials, longest_trial_samples)."""
        ps = _make_session(mock_session_series)
        ps.wheel = _make_wheel_df(duration=5.0)
        ps.wheel_fs = 1000
        # trial lengths: 500, 500, 1000 samples
        ps.trials = pd.DataFrame({
            'stimOn_times':  [0.5,  1.5, 2.5],
            'feedback_times': [1.0,  2.0, 3.5],
        })
        result = ps.extract_wheel_velocity()
        assert result.shape == (3, 1000)

    def test_valid_values_not_nan(self, mock_session_series):
        """Samples within the trial window are not NaN."""
        ps = _make_session(mock_session_series)
        ps.wheel = _make_wheel_df(duration=5.0)
        ps.wheel_fs = 1000
        ps.trials = pd.DataFrame({
            'stimOn_times':   [1.0],
            'feedback_times': [2.0],
        })
        result = ps.extract_wheel_velocity()
        assert not np.any(np.isnan(result[0]))

    def test_short_trials_padded_with_nan(self, mock_session_series):
        """Rows shorter than the max length are NaN-padded on the right."""
        ps = _make_session(mock_session_series)
        ps.wheel = _make_wheel_df(duration=5.0)
        ps.wheel_fs = 1000
        # trial 0: 0.5 s; trial 1: 1.0 s (longest)
        # get_responses masks tpts > t1_relative (strict), so the sample at
        # exactly t=0.5 (index 500) is included; NaN starts at index 501.
        ps.trials = pd.DataFrame({
            'stimOn_times':   [0.5, 1.5],
            'feedback_times': [1.0, 2.5],
        })
        result = ps.extract_wheel_velocity()
        assert result.shape == (2, 1000)
        assert np.all(np.isnan(result[0, 501:]))  # NaN after boundary sample
        assert not np.any(np.isnan(result[1]))    # trial 1 fully valid

    def test_nan_trial_gives_all_nan_row(self, mock_session_series):
        """Trial with NaN stimOn or feedback produces an all-NaN row."""
        ps = _make_session(mock_session_series)
        ps.wheel = _make_wheel_df(duration=5.0)
        ps.wheel_fs = 1000
        ps.trials = pd.DataFrame({
            'stimOn_times':   [0.5,  np.nan],
            'feedback_times': [1.0,  2.0],
        })
        result = ps.extract_wheel_velocity()
        assert np.all(np.isnan(result[1]))

    def test_stores_on_self(self, mock_session_series):
        """extract_wheel_velocity stores result on self.wheel_velocity."""
        ps = _make_session(mock_session_series)
        ps.wheel = _make_wheel_df(duration=3.0)
        ps.wheel_fs = 1000
        ps.trials = pd.DataFrame({
            'stimOn_times':   [0.5],
            'feedback_times': [1.0],
        })
        result = ps.extract_wheel_velocity()
        assert hasattr(ps, 'wheel_velocity')
        np.testing.assert_array_equal(ps.wheel_velocity, result)

    def test_output_dtype_float32(self, mock_session_series):
        """Output matrix is float32 to match SessionLoader convention."""
        ps = _make_session(mock_session_series)
        ps.wheel = _make_wheel_df(duration=3.0)
        ps.wheel_fs = 1000
        ps.trials = pd.DataFrame({
            'stimOn_times':   [0.5],
            'feedback_times': [1.0],
        })
        result = ps.extract_wheel_velocity()
        assert result.dtype == np.float32


# ─────────────────────────────────────────────────────────────────────────────
# save_h5(mode='a') — wheel group
# ─────────────────────────────────────────────────────────────────────────────

class TestSaveWheelToH5:

    def test_roundtrip_values(self, mock_session_series, tmp_path):
        """velocity matrix is preserved exactly after save/read."""
        import h5py
        ps = _make_session(mock_session_series)
        ps.wheel_velocity = np.array(
            [[1., 2., np.nan], [3., 4., 5.]], dtype=np.float32
        )
        ps.wheel_fs = 1000
        fpath = tmp_path / f'{ps.eid}.h5'
        ps.save_h5(fpath, mode='a')

        with h5py.File(fpath, 'r') as f:
            v = f['wheel/velocity'][:]
        assert v.shape == (2, 3)
        assert v.dtype == np.float32
        assert np.isnan(v[0, 2])
        np.testing.assert_allclose(v[0, :2], [1., 2.])
        np.testing.assert_allclose(v[1], [3., 4., 5.])

    def test_saves_fs_and_event_attrs(self, mock_session_series, tmp_path):
        """HDF5 wheel group carries fs, t0_event, t1_event attributes."""
        import h5py
        ps = _make_session(mock_session_series)
        ps.wheel_velocity = np.ones((2, 500), dtype=np.float32)
        ps.wheel_fs = 1000
        fpath = tmp_path / f'{ps.eid}.h5'
        ps.save_h5(fpath, mode='a')

        with h5py.File(fpath, 'r') as f:
            assert f['wheel'].attrs['fs'] == 1000
            assert f['wheel'].attrs['t0_event'] == 'stimOn_times'
            assert f['wheel'].attrs['t1_event'] == 'feedback_times'

    def test_overwrites_existing_wheel_group(self, mock_session_series, tmp_path):
        """Saving twice replaces the wheel group rather than raising an error."""
        import h5py
        ps = _make_session(mock_session_series)
        fpath = tmp_path / f'{ps.eid}.h5'
        ps.wheel_velocity = np.ones((2, 100), dtype=np.float32)
        ps.wheel_fs = 1000
        ps.save_h5(fpath, mode='a')

        ps.wheel_velocity = np.full((3, 200), 2.0, dtype=np.float32)
        ps.save_h5(fpath, mode='a')  # should not raise

        with h5py.File(fpath, 'r') as f:
            assert f['wheel/velocity'].shape == (3, 200)

    def test_uses_default_path_from_config(self, mock_session_series, tmp_path):
        """Without fpath arg, saves to SESSIONS_H5_DIR / {eid}.h5."""
        ps = _make_session(mock_session_series)
        ps.wheel_velocity = np.ones((1, 10), dtype=np.float32)
        ps.wheel_fs = 1000
        with patch('iblnm.data.SESSIONS_H5_DIR', tmp_path):
            ps.save_h5(mode='a')
            saved = tmp_path / f'{ps.eid}.h5'
            assert saved.exists()


# ─────────────────────────────────────────────────────────────────────────────
# load_wheel
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadWheel:

    def test_raises_missing_raw_data_when_no_task_file(self, mock_session_series):
        """When wheel ALF missing AND raw task data missing → MissingRawData."""
        from iblnm.data import PhotometrySession
        from iblnm.validation import MissingRawData
        from one.alf.exceptions import ALFObjectNotFound
        ps = PhotometrySession(mock_session_series, one=MagicMock(), load_data=False)
        with patch.object(type(ps).__bases__[0].__bases__[0], 'load_wheel',
                          side_effect=ALFObjectNotFound('wheel')):
            ps.one.load_dataset.side_effect = ALFObjectNotFound('taskData')
            with pytest.raises(MissingRawData):
                ps.load_wheel()

    def test_raises_missing_extracted_data_when_raw_present(self, mock_session_series):
        """When wheel ALF missing BUT raw task data exists → MissingExtractedData."""
        from iblnm.data import PhotometrySession
        from iblnm.validation import MissingExtractedData
        from one.alf.exceptions import ALFObjectNotFound
        ps = PhotometrySession(mock_session_series, one=MagicMock(), load_data=False)
        with patch.object(type(ps).__bases__[0].__bases__[0], 'load_wheel',
                          side_effect=ALFObjectNotFound('wheel')):
            ps.one.load_dataset.return_value = MagicMock()  # raw task data found
            with pytest.raises(MissingExtractedData):
                ps.load_wheel()

    def test_sets_wheel_dataframe_on_success(self, mock_session_series):
        """On success, self.wheel is a DataFrame with times and velocity columns."""
        from iblnm.data import PhotometrySession
        ps = PhotometrySession(mock_session_series, one=MagicMock(), load_data=False)
        mock_wheel = pd.DataFrame({
            'times': np.arange(0, 3, 0.001, dtype=np.float32),
            'velocity': np.zeros(3000, dtype=np.float32),
            'position': np.zeros(3000, dtype=np.float32),
            'acceleration': np.zeros(3000, dtype=np.float32),
        })
        with patch.object(type(ps).__bases__[0].__bases__[0], 'load_wheel',
                          side_effect=lambda **kw: setattr(ps, 'wheel', mock_wheel)):
            ps.load_wheel()
        assert isinstance(ps.wheel, pd.DataFrame)
        assert 'times' in ps.wheel.columns
        assert 'velocity' in ps.wheel.columns

