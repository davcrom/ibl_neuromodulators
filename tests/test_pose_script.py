"""Tests for scripts/pose.py process_pose skip logic and collect_pose roll-up."""
from unittest.mock import MagicMock

import h5py
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import scripts.pose as pose
from iblnm.config import VIDEO_QC_COLS
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
                        series, baselines=None, trials=None, functions=None):
    """Write an H5 with metadata + video groups carrying known traces.

    ``steps`` maps bodypart -> the response-trace level (flat across time); NaN
    yields an all-NaN response trace (expected to collect to a NaN scalar).
    ``baselines`` maps bodypart -> the stimOn-locked baseline-trace level (flat;
    default 0). The collected scalar is ``step - baseline``.
    ``trials``, when given, maps ``stimOn_times`` / ``feedback_times`` to 1D
    arrays written as flat datasets under a ``trials`` group.
    ``functions``, when given, is the (3, n_lags) xcorr array; defaults to zeros.
    """
    baselines = baselines or {}
    time = np.linspace(-0.5, 0.5, 101)
    bodyparts = list(steps)
    n_trial = 3

    def _flat(level):
        if np.isnan(level):
            return np.full((n_trial, time.size), np.nan)
        return np.full((n_trial, time.size), level)

    response = np.stack([_flat(steps[bp]) for bp in bodyparts])
    baseline = np.stack([_flat(baselines.get(bp, 0.0)) for bp in bodyparts])

    series = series.copy()
    series['eid'] = eid
    ps = PhotometrySession(series, one=MagicMock(), load_data=False)
    coords = {'bodypart': bodyparts, 'trial': np.arange(n_trial), 'time': time}
    ps.pose_traces = xr.DataArray(
        response, dims=['bodypart', 'trial', 'time'], coords=coords)
    ps.pose_baseline_traces = xr.DataArray(
        baseline, dims=['bodypart', 'trial', 'time'], coords=coords)
    ps.pose_xcorr = {
        'functions': np.zeros((3, 11)) if functions is None else np.asarray(functions),
        'lags': np.linspace(-5, 5, 11),
        'peak_lags': np.asarray(peak_lags),
        'drift': drift,
    }
    ps.qc_lp = qc_lp
    ps.save_h5(h5_dir / f'{eid}.h5', groups=['metadata', 'video'])

    if trials is not None:
        with h5py.File(h5_dir / f'{eid}.h5', 'a') as f:
            grp = f.create_group('trials')
            for key, values in trials.items():
                grp.create_dataset(key, data=np.asarray(values))


def _write_qc_perf(tmp_path, qc_eids, perf):
    """Write temp ``sessions_qc.pqt`` and ``performance.pqt``, return their paths.

    ``qc_eids`` maps eid -> a fill value assigned to every ``VIDEO_QC_COLS``
    column for that eid. ``perf`` maps eid -> ``fraction_correct`` float.
    """
    qc_fpath = tmp_path / 'sessions_qc.pqt'
    perf_fpath = tmp_path / 'performance.pqt'
    pd.DataFrame(
        [{'eid': eid, **{col: fill for col in VIDEO_QC_COLS}}
         for eid, fill in qc_eids.items()],
        columns=['eid'] + VIDEO_QC_COLS,
    ).to_parquet(qc_fpath)
    pd.DataFrame(
        [{'eid': eid, 'fraction_correct': value}
         for eid, value in perf.items()],
        columns=['eid', 'fraction_correct'],
    ).to_parquet(perf_fpath)
    return qc_fpath, perf_fpath


@pytest.fixture
def empty_qc_perf(tmp_path):
    """Kwargs pointing ``collect_pose`` at empty QC/performance sources.

    Lets trace-focused tests satisfy the required-file check without exercising
    the joins; every joined column comes back NaN.
    """
    qc_fpath, perf_fpath = _write_qc_perf(tmp_path, qc_eids={}, perf={})
    return {'sessions_qc_fpath': qc_fpath, 'performance_fpath': perf_fpath}


@pytest.fixture
def fake_ps():
    """A PhotometrySession mock whose extract methods are tracked.

    ``video_qc`` defaults to all-PASS and ``length_discrepancy`` to 0 so the
    leftCamera QC validations in ``process_pose`` log nothing unless a test
    overrides them.
    """
    ps = MagicMock()
    ps.eid = 'test-eid'
    ps.video_qc = {col: 'PASS' for col in VIDEO_QC_COLS}
    ps.length_discrepancy = 0.0
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

    def test_failing_video_qc_logs_and_proceeds(self, fake_ps, tmp_path,
                                                monkeypatch):
        monkeypatch.setattr(pose, 'SESSIONS_H5_DIR', tmp_path)
        fake_ps.length_discrepancy = 200.0  # >= LENGTH_MISMATCH_THRESHOLD
        fake_ps.video_qc.update({
            'qc_videoLeft_timestamps': 'FAIL',
            'qc_videoLeft_dropped_frames': 'WARNING',
            'qc_videoLeft_pin_state': 'CRITICAL',
        })

        result = pose.process_pose(fake_ps)

        assert result == 'processed'
        logged = {type(call.args[0]).__name__
                  for call in fake_ps.log_error.call_args_list}
        assert logged == {'VideoLengthError', 'VideoTimestampsQCError',
                          'VideoDroppedFramesQCError', 'VideoPinStateQCError'}
        fake_ps.extract_movement_traces.assert_called_once()
        fake_ps.save_h5.assert_called_once_with(groups=['video'])

    def test_clean_video_qc_logs_nothing(self, fake_ps, tmp_path, monkeypatch):
        monkeypatch.setattr(pose, 'SESSIONS_H5_DIR', tmp_path)

        result = pose.process_pose(fake_ps)

        assert result == 'processed'
        fake_ps.log_error.assert_not_called()


class TestReadEids:
    def test_reads_nonempty_stripped_lines(self, tmp_path):
        f = tmp_path / 'eids.csv'
        f.write_text('aaa\nbbb\n\n  ccc  \n')
        assert pose.read_eids(f) == ['aaa', 'bbb', 'ccc']


class TestCollectPose:
    def test_rollup_two_sessions(self, tmp_path, mock_session_series,
                                 empty_qc_perf):
        steps_a = {'paw': 1.0, 'nose': 2.0, 'tongue_speed': 3.0,
                   'tongue_likelihood': 0.5}
        baselines_a = {'paw': 0.4, 'nose': 0.5, 'tongue_speed': 1.0,
                       'tongue_likelihood': 0.2}
        steps_b = {'paw': -1.0, 'nose': 0.0, 'tongue_speed': np.nan,
                   'tongue_likelihood': 0.8}
        _write_pose_session(tmp_path, 'eid-a', steps_a, drift=0.3,
                            peak_lags=[0.1, 0.2, 0.4], qc_lp='FAIL',
                            series=mock_session_series, baselines=baselines_a)
        _write_pose_session(tmp_path, 'eid-b', steps_b, drift=np.nan,
                            peak_lags=[0.0, np.nan, 0.5], qc_lp='PASS',
                            series=mock_session_series)

        df = pose.collect_pose(tmp_path, **empty_qc_perf)

        assert set(df['eid']) == {'eid-a', 'eid-b'}
        row_a = df.set_index('eid').loc['eid-a']
        # scalar is the response level minus the stimOn-locked baseline level
        for bodypart, step in steps_a.items():
            np.testing.assert_allclose(
                row_a[bodypart], step - baselines_a[bodypart])
        assert row_a['drift'] == 0.3
        assert row_a['qc_lp'] == 'FAIL'
        np.testing.assert_allclose(
            [row_a['peak_lag_early'], row_a['peak_lag_mid'],
             row_a['peak_lag_late']], [0.1, 0.2, 0.4])

    def test_peak_values_from_xcorr_functions(self, tmp_path,
                                              mock_session_series, empty_qc_perf):
        steps = {'paw': 1.0, 'nose': 2.0, 'tongue_speed': 3.0,
                 'tongue_likelihood': 0.5}
        # peak of each third's function = its max: 0.2, 0.7, 0.4
        functions = np.array([
            [0.1, 0.2, -0.3], [0.7, 0.1, 0.0], [0.4, -0.5, 0.2]])
        _write_pose_session(tmp_path, 'eid-pk', steps, drift=0.1,
                            peak_lags=[0.0, 0.0, 0.0], qc_lp='PASS',
                            series=mock_session_series, functions=functions)

        df = pose.collect_pose(tmp_path, **empty_qc_perf).set_index('eid')

        np.testing.assert_allclose(
            [df.loc['eid-pk', 'peak_val_early'], df.loc['eid-pk', 'peak_val_mid'],
             df.loc['eid-pk', 'peak_val_late']], [0.2, 0.7, 0.4])

    def test_mean_rt_from_trials_group(self, tmp_path, mock_session_series,
                                       empty_qc_perf):
        steps = {'paw': 1.0, 'nose': 2.0, 'tongue_speed': 3.0,
                 'tongue_likelihood': 0.5}
        trials = {'stimOn_times': [0.0, 0.0, 0.0],
                  'feedback_times': [0.5, 1.0, np.nan]}
        _write_pose_session(tmp_path, 'eid-rt', steps, drift=0.1,
                            peak_lags=[0.0, 0.0, 0.0], qc_lp='PASS',
                            series=mock_session_series, trials=trials)

        df = pose.collect_pose(tmp_path, **empty_qc_perf).set_index('eid')

        np.testing.assert_allclose(df.loc['eid-rt', 'mean_rt'], 0.75)

    def test_mean_rt_nan_without_trials_group(self, tmp_path,
                                              mock_session_series,
                                              empty_qc_perf):
        steps = {'paw': 1.0, 'nose': 2.0, 'tongue_speed': 3.0,
                 'tongue_likelihood': 0.5}
        _write_pose_session(tmp_path, 'eid-nort', steps, drift=0.1,
                            peak_lags=[0.0, 0.0, 0.0], qc_lp='PASS',
                            series=mock_session_series)

        df = pose.collect_pose(tmp_path, **empty_qc_perf).set_index('eid')

        assert np.isnan(df.loc['eid-nort', 'mean_rt'])
        assert np.isfinite(df.loc['eid-nort', 'paw'])

    def test_qc_and_performance_joins(self, tmp_path, mock_session_series):
        steps = {'paw': 1.0, 'nose': 2.0, 'tongue_speed': 3.0,
                 'tongue_likelihood': 0.5}
        for eid in ('eid-a', 'eid-b'):
            _write_pose_session(tmp_path, eid, steps, drift=0.1,
                                peak_lags=[0.0, 0.0, 0.0], qc_lp='PASS',
                                series=mock_session_series)
        qc_fpath, perf_fpath = _write_qc_perf(
            tmp_path, qc_eids={'eid-a': 'WARNING'}, perf={'eid-a': 0.82})

        df = pose.collect_pose(
            tmp_path, sessions_qc_fpath=qc_fpath,
            performance_fpath=perf_fpath).set_index('eid')

        for col in VIDEO_QC_COLS:
            assert df.loc['eid-a', col] == 'WARNING'
            assert pd.isna(df.loc['eid-b', col])
        assert df.loc['eid-a', 'fraction_correct'] == 0.82
        assert pd.isna(df.loc['eid-b', 'fraction_correct'])

    def test_missing_sessions_qc_raises(self, tmp_path, mock_session_series):
        steps = {'paw': 1.0, 'nose': 2.0, 'tongue_speed': 3.0,
                 'tongue_likelihood': 0.5}
        _write_pose_session(tmp_path, 'eid-a', steps, drift=0.1,
                            peak_lags=[0.0, 0.0, 0.0], qc_lp='PASS',
                            series=mock_session_series)
        _, perf_fpath = _write_qc_perf(tmp_path, qc_eids={}, perf={})
        missing = tmp_path / 'does_not_exist.pqt'

        with pytest.raises(FileNotFoundError, match='does_not_exist.pqt'):
            pose.collect_pose(tmp_path, sessions_qc_fpath=missing,
                              performance_fpath=perf_fpath)

    def test_all_nan_trace_yields_nan_scalar(self, tmp_path,
                                             mock_session_series,
                                             empty_qc_perf):
        steps = {'paw': 1.0, 'nose': 2.0, 'tongue_speed': np.nan,
                 'tongue_likelihood': 0.5}
        _write_pose_session(tmp_path, 'eid-c', steps, drift=0.1,
                            peak_lags=[0.0, 0.0, 0.0], qc_lp='NOT_SET',
                            series=mock_session_series)

        df = pose.collect_pose(tmp_path, **empty_qc_perf).set_index('eid')

        assert np.isnan(df.loc['eid-c', 'tongue_speed'])
        assert np.isfinite(df.loc['eid-c', 'paw'])
