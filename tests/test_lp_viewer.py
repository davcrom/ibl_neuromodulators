"""Tests for LPViewer pure helper functions (iblnm/lp_viewer.py)."""
import h5py
import numpy as np
import pandas as pd
import pytest

from iblnm.lp_viewer import (
    apply_label,
    filter_sessions_table,
    frames_in_trial,
    likelihood_to_alpha,
    persist_labels,
)


@pytest.fixture
def pose_table():
    return pd.DataFrame({
        'eid': ['a', 'b', 'c', 'd'],
        'paw_speed': [0.1, 0.5, 0.9, 0.5],
        'session_type': ['biased', 'biased', 'ephys', 'training'],
    })


# ─────────────────────────────────────────────────────────────────────────────
# filter_sessions_table
# ─────────────────────────────────────────────────────────────────────────────

def test_filter_sessions_table_range_and_type(pose_table):
    eids = filter_sessions_table(
        pose_table, 'paw_speed', (0.4, 0.6), ('biased', 'training'))
    assert eids == ['b', 'd']


def test_filter_sessions_table_type_excludes(pose_table):
    eids = filter_sessions_table(
        pose_table, 'paw_speed', (0.0, 1.0), ('ephys',))
    assert eids == ['c']


def test_filter_sessions_table_empty_range(pose_table):
    eids = filter_sessions_table(
        pose_table, 'paw_speed', (2.0, 3.0), ('biased', 'ephys', 'training'))
    assert eids == []


# ─────────────────────────────────────────────────────────────────────────────
# likelihood_to_alpha
# ─────────────────────────────────────────────────────────────────────────────

def test_likelihood_to_alpha_endpoints():
    assert likelihood_to_alpha(0.0) == pytest.approx(0.0)
    assert likelihood_to_alpha(1.0) == pytest.approx(1.0)


def test_likelihood_to_alpha_monotonic():
    likelihoods = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
    alphas = likelihood_to_alpha(likelihoods)
    assert np.all(np.diff(alphas) >= 0)
    assert np.all((alphas >= 0) & (alphas <= 1))


def test_likelihood_to_alpha_clips_out_of_range():
    alphas = likelihood_to_alpha(np.array([-0.5, 1.5]))
    assert alphas[0] == pytest.approx(0.0)
    assert alphas[1] == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# frames_in_trial
# ─────────────────────────────────────────────────────────────────────────────

def test_frames_in_trial_contiguous_span():
    camera_times = np.arange(0.0, 1.0, 0.1)  # 0.0, 0.1, ..., 0.9
    idx = frames_in_trial(camera_times, 0.25, 0.65)
    assert idx.tolist() == [3, 4, 5, 6]  # times 0.3, 0.4, 0.5, 0.6


def test_frames_in_trial_inclusive_bounds():
    camera_times = np.array([0.0, 0.2, 0.4, 0.6])
    idx = frames_in_trial(camera_times, 0.2, 0.4)
    assert idx.tolist() == [1, 2]


def test_frames_in_trial_empty_window():
    camera_times = np.array([0.0, 0.5, 1.0])
    idx = frames_in_trial(camera_times, 0.6, 0.9)
    assert idx.tolist() == []


# ─────────────────────────────────────────────────────────────────────────────
# apply_label
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def label_table():
    return pd.DataFrame({
        'eid': ['a', 'b'],
        'qc_lp': ['NOT_SET', 'NOT_SET'],
        'qc_movement': ['NOT_SET', 'NOT_SET'],
    })


def test_apply_label_sets_valid(label_table):
    out = apply_label(label_table, 'b', 'qc_lp', 'FAIL')
    assert out.loc[out['eid'] == 'b', 'qc_lp'].item() == 'FAIL'
    # other rows/fields untouched
    assert out.loc[out['eid'] == 'a', 'qc_lp'].item() == 'NOT_SET'
    assert out.loc[out['eid'] == 'b', 'qc_movement'].item() == 'NOT_SET'


def test_apply_label_does_not_mutate_input(label_table):
    apply_label(label_table, 'a', 'qc_movement', 'PASS')
    assert label_table.loc[label_table['eid'] == 'a', 'qc_movement'].item() == 'NOT_SET'


def test_apply_label_rejects_out_of_vocab_value(label_table):
    with pytest.raises(ValueError):
        apply_label(label_table, 'a', 'qc_lp', 'GOOD')


def test_apply_label_rejects_unknown_field(label_table):
    with pytest.raises(ValueError):
        apply_label(label_table, 'a', 'qc_other', 'PASS')


# ─────────────────────────────────────────────────────────────────────────────
# persist_labels
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def video_h5(tmp_path):
    """Synthetic video H5: a traces subgroup (automatic data) + label attrs."""
    fpath = tmp_path / 'session.h5'
    paw = np.arange(12, dtype=np.float64).reshape(3, 4)
    with h5py.File(fpath, 'w') as f:
        grp = f.create_group('video')
        grp.attrs['qc_lp'] = 'NOT_SET'
        grp.attrs['qc_movement'] = 'NOT_SET'
        grp.create_group('traces').create_dataset('paw', data=paw)
    return fpath, paw


def test_persist_labels_round_trips(video_h5):
    fpath, _ = video_h5
    persist_labels(fpath, 'CRITICAL', 'PASS')
    with h5py.File(fpath, 'r') as f:
        attrs = f['video'].attrs
        assert _decode(attrs['qc_lp']) == 'CRITICAL'
        assert _decode(attrs['qc_movement']) == 'PASS'


def test_persist_labels_leaves_traces_untouched(video_h5):
    fpath, paw = video_h5
    persist_labels(fpath, 'WARNING', 'FAIL')
    with h5py.File(fpath, 'r') as f:
        np.testing.assert_array_equal(f['video']['traces']['paw'][:], paw)


def _decode(value):
    return value.decode() if isinstance(value, bytes) else value
