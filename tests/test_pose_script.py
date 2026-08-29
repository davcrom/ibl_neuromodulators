"""Tests for scripts/pose.py process_pose skip logic."""
from unittest.mock import MagicMock

import h5py
import numpy as np
import pytest

import scripts.pose as pose
from iblnm.config import VIDEO_QC_COLS


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
    ps.video_times_qc = {'length_discrepancy': 0.0}
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
        fake_ps.extract_responses.assert_not_called()
        fake_ps.load_pose_qc.assert_not_called()

    def test_reprocess_extracts_despite_existing_group(self, fake_ps, tmp_path,
                                                       monkeypatch):
        monkeypatch.setattr(pose, 'SESSIONS_H5_DIR', tmp_path)
        _write_video_h5(tmp_path, fake_ps.eid)

        result = pose.process_pose(fake_ps, reprocess=True)

        assert result == 'processed'
        fake_ps.extract_responses.assert_called_once()
        fake_ps.load_pose_qc.assert_called_once()
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
        fake_ps.extract_responses.assert_called_once()
        fake_ps.load_pose_qc.assert_not_called()
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
        fake_ps.extract_responses.assert_called_once()
        fake_ps.load_pose_qc.assert_called_once()
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
        fake_ps.extract_responses.assert_not_called()
        fake_ps.load_pose_qc.assert_not_called()
        fake_ps.save_h5.assert_called_once_with(groups=['video'])

    def test_failing_video_qc_logs_and_proceeds(self, fake_ps, tmp_path,
                                                monkeypatch):
        monkeypatch.setattr(pose, 'SESSIONS_H5_DIR', tmp_path)
        fake_ps.video_times_qc = {'length_discrepancy': 200.0}  # >= threshold
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
        fake_ps.extract_responses.assert_called_once()
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


