"""Tests for scripts/pose.py process_pose skip logic and collect_pose roll-up."""
from unittest.mock import MagicMock

import h5py
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import scripts.pose as pose
from iblnm.data import PhotometrySession


@pytest.fixture
def mock_session_series():
    return pd.Series({
        'eid': 'eid-0',
        'subject': 'test_mouse',
        'start_time': '2024-01-01T10:00:00',
        'number': 1,
    })


def _write_pose_session(h5_dir, eid, steps, drift, peak_lags, qc_lp,
                        series):
    """Write an H5 with metadata + video groups carrying known step traces.

    ``steps`` maps bodypart -> post-pre step value; NaN yields an all-NaN
    in-window trace (expected to collect to a NaN scalar).
    """
    time = np.linspace(-0.5, 0.5, 101)
    bodyparts = list(steps)
    n_trial = 3
    data = np.empty((len(bodyparts), n_trial, time.size))
    for i, bodypart in enumerate(bodyparts):
        step = steps[bodypart]
        if np.isnan(step):
            data[i] = np.nan
        else:
            trace = np.where(time >= 0.05, step, 0.0)
            data[i] = np.tile(trace, (n_trial, 1))

    series = series.copy()
    series['eid'] = eid
    ps = PhotometrySession(series, one=MagicMock(), load_data=False)
    ps.pose_traces = xr.DataArray(
        data,
        dims=['bodypart', 'trial', 'time'],
        coords={'bodypart': bodyparts, 'trial': np.arange(n_trial),
                'time': time},
    )
    ps.pose_xcorr = {
        'functions': np.zeros((3, 11)),
        'lags': np.linspace(-5, 5, 11),
        'peak_lags': np.asarray(peak_lags),
        'drift': drift,
    }
    ps.qc_lp = qc_lp
    ps.save_h5(h5_dir / f'{eid}.h5', groups=['metadata', 'video'])


@pytest.fixture
def fake_ps():
    """A PhotometrySession mock whose extract methods are tracked."""
    ps = MagicMock()
    ps.eid = 'test-eid'
    return ps


def _write_video_h5(h5_dir, eid):
    with h5py.File(h5_dir / f'{eid}.h5', 'w') as f:
        f.create_group('video')


class TestProcessPoseSkip:
    def test_skips_when_video_group_exists(self, fake_ps, tmp_path, monkeypatch):
        monkeypatch.setattr(pose, 'SESSIONS_H5_DIR', tmp_path)
        _write_video_h5(tmp_path, fake_ps.eid)

        result = pose.process_pose(fake_ps)

        assert result == 'skipped'
        fake_ps.extract_movement_traces.assert_not_called()
        fake_ps.extract_paw_wheel_xcorr.assert_not_called()

    def test_reprocess_extracts_despite_existing_group(self, fake_ps, tmp_path,
                                                       monkeypatch):
        monkeypatch.setattr(pose, 'SESSIONS_H5_DIR', tmp_path)
        _write_video_h5(tmp_path, fake_ps.eid)

        result = pose.process_pose(fake_ps, reprocess=True)

        assert result == 'processed'
        fake_ps.extract_movement_traces.assert_called_once()
        fake_ps.extract_paw_wheel_xcorr.assert_called_once()
        fake_ps.save_h5.assert_called_once_with(groups=['video'])

    def test_missing_lp_logs_and_skips_without_writing(self, fake_ps, tmp_path,
                                                       monkeypatch):
        monkeypatch.setattr(pose, 'SESSIONS_H5_DIR', tmp_path)
        fake_ps.load_pose.side_effect = pose.MissingLP('leftCamera.lightningPose')

        result = pose.process_pose(fake_ps)

        assert result == 'skipped'
        fake_ps.log_error.assert_called_once()
        fake_ps.extract_movement_traces.assert_not_called()
        fake_ps.save_h5.assert_not_called()


class TestReadEids:
    def test_reads_nonempty_stripped_lines(self, tmp_path):
        f = tmp_path / 'eids.csv'
        f.write_text('aaa\nbbb\n\n  ccc  \n')
        assert pose.read_eids(f) == ['aaa', 'bbb', 'ccc']


class TestCollectPose:
    def test_rollup_two_sessions(self, tmp_path, mock_session_series):
        steps_a = {'paw': 1.0, 'nose': 2.0, 'tongue_speed': 3.0,
                   'tongue_likelihood': 0.5}
        steps_b = {'paw': -1.0, 'nose': 0.0, 'tongue_speed': np.nan,
                   'tongue_likelihood': 0.8}
        _write_pose_session(tmp_path, 'eid-a', steps_a, drift=0.3,
                            peak_lags=[0.1, 0.2, 0.4], qc_lp='FAIL',
                            series=mock_session_series)
        _write_pose_session(tmp_path, 'eid-b', steps_b, drift=np.nan,
                            peak_lags=[0.0, np.nan, 0.5], qc_lp='PASS',
                            series=mock_session_series)

        df = pose.collect_pose(tmp_path)

        assert set(df['eid']) == {'eid-a', 'eid-b'}
        row_a = df.set_index('eid').loc['eid-a']
        for bodypart, step in steps_a.items():
            np.testing.assert_allclose(row_a[bodypart], step)
        assert row_a['drift'] == 0.3
        assert row_a['qc_lp'] == 'FAIL'
        np.testing.assert_allclose(
            [row_a['peak_lag_early'], row_a['peak_lag_mid'],
             row_a['peak_lag_late']], [0.1, 0.2, 0.4])

    def test_all_nan_trace_yields_nan_scalar(self, tmp_path,
                                             mock_session_series):
        steps = {'paw': 1.0, 'nose': 2.0, 'tongue_speed': np.nan,
                 'tongue_likelihood': 0.5}
        _write_pose_session(tmp_path, 'eid-c', steps, drift=0.1,
                            peak_lags=[0.0, 0.0, 0.0], qc_lp='NOT_SET',
                            series=mock_session_series)

        df = pose.collect_pose(tmp_path).set_index('eid')

        assert np.isnan(df.loc['eid-c', 'tongue_speed'])
        assert np.isfinite(df.loc['eid-c', 'paw'])
