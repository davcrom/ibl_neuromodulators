"""Tests for scripts/pose.py process_pose skip logic and collect_pose roll-up."""
from unittest.mock import MagicMock

import h5py
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import scripts.pose as pose
from iblnm.config import QCVAL2NUM, VIDEO_QC_COLS, VIDEO_QC_QUALITY_COLS
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
                        series, baselines=None, trials=None, functions=None,
                        video_qc=None, length_discrepancy=np.nan,
                        framerate_from_tpts=np.nan):
    """Write an H5 with metadata + video groups carrying known traces.

    ``steps`` maps bodypart -> the response-trace level (flat across time); NaN
    yields an all-NaN response trace (expected to collect to a NaN scalar). Pass
    ``steps=None`` to write a video group with measures + QC attrs but no traces
    (the LP-absent case). ``baselines`` maps bodypart -> the stimOn-locked
    baseline-trace level (flat; default 0). The collected scalar is
    ``step - baseline``. ``trials``, when given, maps ``stimOn_times`` /
    ``feedback_times`` to 1D arrays written as flat datasets under a ``trials``
    group. ``functions``, when given, is the (3, n_lags) xcorr array; defaults to
    zeros. ``video_qc`` maps ``VIDEO_QC_COLS`` names to IBL QC labels written as
    video-group attrs (default all ``NOT_SET``).
    """
    baselines = baselines or {}
    series = series.copy()
    series['eid'] = eid
    ps = PhotometrySession(series, one=MagicMock(), load_data=False)
    ps.qc_lp = qc_lp
    ps.video_qc = dict(video_qc) if video_qc else {}
    ps.length_discrepancy = length_discrepancy
    ps.framerate_from_tpts = framerate_from_tpts

    if steps is not None:
        time = np.linspace(-0.5, 0.5, 101)
        bodyparts = list(steps)
        n_trial = 3

        def _flat(level):
            if np.isnan(level):
                return np.full((n_trial, time.size), np.nan)
            return np.full((n_trial, time.size), level)

        response = np.stack([_flat(steps[bp]) for bp in bodyparts])
        baseline = np.stack([_flat(baselines.get(bp, 0.0)) for bp in bodyparts])
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
    ps.save_h5(h5_dir / f'{eid}.h5', groups=['metadata', 'video'])

    if trials is not None:
        with h5py.File(h5_dir / f'{eid}.h5', 'a') as f:
            grp = f.create_group('trials')
            for key, values in trials.items():
                grp.create_dataset(key, data=np.asarray(values))


def _write_errors(h5_dir, eid, error_types):
    """Append an ``errors`` group listing ``error_types`` to ``{eid}.h5``.

    Creates the H5 if absent, so the bare-row case (errors but no ``video``
    group) can be exercised. Mirrors the on-disk schema read by
    ``util.collect_errors`` (datasets ``eid``, ``error_type``,
    ``error_message``, ``traceback``).
    """
    with h5py.File(h5_dir / f'{eid}.h5', 'a') as f:
        if 'errors' in f:
            del f['errors']
        grp = f.create_group('errors')
        n = len(error_types)
        grp.create_dataset('eid', data=[eid] * n, dtype=h5py.string_dtype())
        grp.create_dataset('error_type', data=list(error_types),
                           dtype=h5py.string_dtype())
        grp.create_dataset('error_message', data=[''] * n, dtype=h5py.string_dtype())
        grp.create_dataset('traceback', data=[''] * n, dtype=h5py.string_dtype())


def _write_perf(tmp_path, perf):
    """Write a temp ``performance.pqt`` mapping eid -> ``fraction_correct``."""
    perf_fpath = tmp_path / 'performance.pqt'
    pd.DataFrame(
        [{'eid': eid, 'fraction_correct': value} for eid, value in perf.items()],
        columns=['eid', 'fraction_correct'],
    ).to_parquet(perf_fpath)
    return perf_fpath


@pytest.fixture
def perf_fpath(tmp_path):
    """Path to an empty ``performance.pqt`` for tests not exercising the join."""
    return _write_perf(tmp_path, {})


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

    def test_missing_lp_logs_and_continues_with_motion_energy(self, fake_ps,
                                                              tmp_path, monkeypatch):
        """MissingLP is non-fatal: ME is still extracted (no xcorr) and saved."""
        monkeypatch.setattr(pose, 'SESSIONS_H5_DIR', tmp_path)
        fake_ps.load_pose.side_effect = pose.MissingLP('leftCamera.lightningPose')
        fake_ps.pose = None
        fake_ps.motion_energy = np.arange(10.0)

        result = pose.process_pose(fake_ps)

        assert result == 'processed'
        fake_ps.log_error.assert_called_once()
        fake_ps.extract_movement_traces.assert_called_once()
        fake_ps.extract_paw_wheel_xcorr.assert_not_called()
        fake_ps.save_h5.assert_called_once_with(groups=['video'])

    def test_missing_motion_energy_logs_and_continues_with_lp(self, fake_ps,
                                                             tmp_path, monkeypatch):
        """MissingMotionEnergy is non-fatal: LP traces + xcorr still extracted."""
        monkeypatch.setattr(pose, 'SESSIONS_H5_DIR', tmp_path)
        fake_ps.load_motion_energy.side_effect = \
            pose.MissingMotionEnergy('leftCamera.ROIMotionEnergy')
        fake_ps.motion_energy = None

        result = pose.process_pose(fake_ps)

        assert result == 'processed'
        logged = {type(call.args[0]).__name__
                  for call in fake_ps.log_error.call_args_list}
        assert logged == {'MissingMotionEnergy'}
        fake_ps.extract_movement_traces.assert_called_once()
        fake_ps.extract_paw_wheel_xcorr.assert_called_once()
        fake_ps.save_h5.assert_called_once_with(groups=['video'])

    def test_no_lp_no_motion_energy_writes_basic_group(self, fake_ps, tmp_path,
                                                       monkeypatch):
        """Both sources absent → no trace extraction, but the basic-video group is
        still written (timestamps existed)."""
        monkeypatch.setattr(pose, 'SESSIONS_H5_DIR', tmp_path)
        fake_ps.load_pose.side_effect = pose.MissingLP('leftCamera.lightningPose')
        fake_ps.load_motion_energy.side_effect = \
            pose.MissingMotionEnergy('leftCamera.ROIMotionEnergy')
        fake_ps.pose = None
        fake_ps.motion_energy = None

        result = pose.process_pose(fake_ps)

        assert result == 'processed'
        fake_ps.extract_movement_traces.assert_not_called()
        fake_ps.extract_paw_wheel_xcorr.assert_not_called()
        fake_ps.save_h5.assert_called_once_with(groups=['video'])

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
                                 perf_fpath):
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

        df = pose.collect_pose(tmp_path, performance_fpath=perf_fpath)

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
                                              mock_session_series, perf_fpath):
        steps = {'paw': 1.0, 'nose': 2.0, 'tongue_speed': 3.0,
                 'tongue_likelihood': 0.5}
        # peak of each third's function = its max: 0.2, 0.7, 0.4
        functions = np.array([
            [0.1, 0.2, -0.3], [0.7, 0.1, 0.0], [0.4, -0.5, 0.2]])
        _write_pose_session(tmp_path, 'eid-pk', steps, drift=0.1,
                            peak_lags=[0.0, 0.0, 0.0], qc_lp='PASS',
                            series=mock_session_series, functions=functions)

        df = pose.collect_pose(tmp_path, performance_fpath=perf_fpath).set_index('eid')

        np.testing.assert_allclose(
            [df.loc['eid-pk', 'peak_val_early'], df.loc['eid-pk', 'peak_val_mid'],
             df.loc['eid-pk', 'peak_val_late']], [0.2, 0.7, 0.4])

    def test_mean_rt_from_trials_group(self, tmp_path, mock_session_series,
                                       perf_fpath):
        steps = {'paw': 1.0, 'nose': 2.0, 'tongue_speed': 3.0,
                 'tongue_likelihood': 0.5}
        trials = {'stimOn_times': [0.0, 0.0, 0.0],
                  'feedback_times': [0.5, 1.0, np.nan]}
        _write_pose_session(tmp_path, 'eid-rt', steps, drift=0.1,
                            peak_lags=[0.0, 0.0, 0.0], qc_lp='PASS',
                            series=mock_session_series, trials=trials)

        df = pose.collect_pose(tmp_path, performance_fpath=perf_fpath).set_index('eid')

        np.testing.assert_allclose(df.loc['eid-rt', 'mean_rt'], 0.75)

    def test_mean_rt_nan_without_trials_group(self, tmp_path,
                                              mock_session_series,
                                              perf_fpath):
        steps = {'paw': 1.0, 'nose': 2.0, 'tongue_speed': 3.0,
                 'tongue_likelihood': 0.5}
        _write_pose_session(tmp_path, 'eid-nort', steps, drift=0.1,
                            peak_lags=[0.0, 0.0, 0.0], qc_lp='PASS',
                            series=mock_session_series)

        df = pose.collect_pose(tmp_path, performance_fpath=perf_fpath).set_index('eid')

        assert np.isnan(df.loc['eid-nort', 'mean_rt'])
        assert np.isfinite(df.loc['eid-nort', 'paw'])

    def test_performance_join(self, tmp_path, mock_session_series):
        steps = {'paw': 1.0, 'nose': 2.0, 'tongue_speed': 3.0,
                 'tongue_likelihood': 0.5}
        for eid in ('eid-a', 'eid-b'):
            _write_pose_session(tmp_path, eid, steps, drift=0.1,
                                peak_lags=[0.0, 0.0, 0.0], qc_lp='PASS',
                                series=mock_session_series)
        perf_fpath = _write_perf(tmp_path, {'eid-a': 0.82})

        df = pose.collect_pose(
            tmp_path, performance_fpath=perf_fpath).set_index('eid')

        assert df.loc['eid-a', 'fraction_correct'] == 0.82
        assert pd.isna(df.loc['eid-b', 'fraction_correct'])

    def test_runs_without_sessions_qc_file(self, tmp_path, mock_session_series,
                                           perf_fpath):
        """No ``sessions_qc.pqt`` anywhere: QC is sourced from the H5 group."""
        steps = {'paw': 1.0, 'nose': 2.0, 'tongue_speed': 3.0,
                 'tongue_likelihood': 0.5}
        _write_pose_session(tmp_path, 'eid-a', steps, drift=0.1,
                            peak_lags=[0.0, 0.0, 0.0], qc_lp='PASS',
                            series=mock_session_series)

        df = pose.collect_pose(tmp_path, performance_fpath=perf_fpath)

        assert list(df['eid']) == ['eid-a']

    def test_all_nan_trace_yields_nan_scalar(self, tmp_path,
                                             mock_session_series,
                                             perf_fpath):
        steps = {'paw': 1.0, 'nose': 2.0, 'tongue_speed': np.nan,
                 'tongue_likelihood': 0.5}
        _write_pose_session(tmp_path, 'eid-c', steps, drift=0.1,
                            peak_lags=[0.0, 0.0, 0.0], qc_lp='NOT_SET',
                            series=mock_session_series)

        df = pose.collect_pose(tmp_path, performance_fpath=perf_fpath).set_index('eid')

        assert np.isnan(df.loc['eid-c', 'tongue_speed'])
        assert np.isfinite(df.loc['eid-c', 'paw'])

    def test_video_qc_score_from_h5_quality_cols(self, tmp_path,
                                                 mock_session_series, perf_fpath):
        """Traces + motion energy + clean QC: lp_exists, finite ME, scored QC."""
        steps = {'paw': 1.0, 'nose': 2.0, 'tongue_speed': 3.0,
                 'tongue_likelihood': 0.5, 'motion_energy': 4.0}
        clean_qc = {col: 'PASS' for col in VIDEO_QC_COLS}
        _write_pose_session(tmp_path, 'eid-q', steps, drift=0.1,
                            peak_lags=[0.0, 0.0, 0.0], qc_lp='PASS',
                            series=mock_session_series, video_qc=clean_qc)

        df = pose.collect_pose(tmp_path, performance_fpath=perf_fpath).set_index('eid')

        assert df.loc['eid-q', 'lp_exists']
        assert np.isfinite(df.loc['eid-q', 'motion_energy'])
        expected = np.nanmean([QCVAL2NUM['PASS']] * len(VIDEO_QC_QUALITY_COLS))
        np.testing.assert_allclose(df.loc['eid-q', 'video_qc_score'], expected)

    def test_lp_absent_row_present_with_nan_traces(self, tmp_path,
                                                   mock_session_series, perf_fpath):
        """Video group with measures + QC but no traces: row present, scored."""
        qc = {col: 'PASS' for col in VIDEO_QC_COLS}
        _write_pose_session(tmp_path, 'eid-nolp', steps=None, drift=np.nan,
                            peak_lags=None, qc_lp='NOT_SET',
                            series=mock_session_series, video_qc=qc,
                            length_discrepancy=12.0, framerate_from_tpts=30.0)

        df = pose.collect_pose(tmp_path, performance_fpath=perf_fpath).set_index('eid')

        assert not df.loc['eid-nolp', 'lp_exists']
        assert np.isnan(df.loc['eid-nolp', 'paw'])
        assert df.loc['eid-nolp', 'length_discrepancy'] == 12.0
        expected = np.nanmean([QCVAL2NUM['PASS']] * len(VIDEO_QC_QUALITY_COLS))
        np.testing.assert_allclose(df.loc['eid-nolp', 'video_qc_score'], expected)

    def test_disqualifying_error_forces_score_minus_one(self, tmp_path,
                                                        mock_session_series,
                                                        perf_fpath):
        steps = {'paw': 1.0, 'nose': 2.0, 'tongue_speed': 3.0,
                 'tongue_likelihood': 0.5}
        clean_qc = {col: 'PASS' for col in VIDEO_QC_COLS}
        _write_pose_session(tmp_path, 'eid-err', steps, drift=0.1,
                            peak_lags=[0.0, 0.0, 0.0], qc_lp='PASS',
                            series=mock_session_series, video_qc=clean_qc)
        _write_errors(tmp_path, 'eid-err', ['VideoLengthError'])

        df = pose.collect_pose(tmp_path, performance_fpath=perf_fpath).set_index('eid')

        assert df.loc['eid-err', 'video_qc_score'] == -1

    def test_missing_timestamps_no_video_group_emits_bare_row(
            self, tmp_path, mock_session_series, perf_fpath):
        _write_errors(tmp_path, 'eid-bare', ['MissingVideoTimestamps'])

        df = pose.collect_pose(tmp_path, performance_fpath=perf_fpath).set_index('eid')

        assert 'eid-bare' in df.index
        assert df.loc['eid-bare', 'video_qc_score'] == -1
        assert np.isnan(df.loc['eid-bare', 'paw'])
        assert not df.loc['eid-bare', 'lp_exists']
