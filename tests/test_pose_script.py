"""Tests for scripts/pose.py process_pose skip logic."""
from unittest.mock import MagicMock

import h5py
import pytest

import scripts.pose as pose


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
