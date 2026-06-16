"""Tests for LPViewer pure helper functions (iblnm/lp_viewer.py)."""
import h5py
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from iblnm.data import _save_pose_traces, _save_pose_xcorr
from iblnm.lp_viewer import (
    HISTOGRAM_MEASURES,
    HISTOGRAM_TITLES,
    LPViewerModel,
    apply_label,
    filter_sessions_table,
    format_event_timings,
    format_session_title,
    frames_in_trial,
    histogram_by_type,
    keypoint_colors,
    likelihood_to_alpha,
    persist_labels,
    save_label,
    trial_frame_window,
    update_pose_qc,
)


@pytest.fixture
def pose_table():
    return pd.DataFrame({
        'eid': ['a', 'b', 'c', 'd'],
        'paw_speed': [0.1, 0.5, 0.9, 0.5],
        'nose_speed': [0.5, 0.5, 0.5, 0.9],
        'session_type': ['biased', 'biased', 'ephys', 'training'],
    })


# ─────────────────────────────────────────────────────────────────────────────
# filter_sessions_table
# ─────────────────────────────────────────────────────────────────────────────

def test_filter_sessions_table_single_range_and_type(pose_table):
    eids = filter_sessions_table(
        pose_table, {'paw_speed': (0.4, 0.6)}, ('biased', 'training'))
    assert eids == ['b', 'd']


def test_filter_sessions_table_intersects_ranges(pose_table):
    # paw in [0.4, 0.6] -> b, d; nose in [0.0, 0.6] -> a, b, c; both -> b
    eids = filter_sessions_table(
        pose_table,
        {'paw_speed': (0.4, 0.6), 'nose_speed': (0.0, 0.6)},
        ('biased', 'training'))
    assert eids == ['b']


def test_filter_sessions_table_type_excludes(pose_table):
    eids = filter_sessions_table(
        pose_table, {'paw_speed': (0.0, 1.0)}, ('ephys',))
    assert eids == ['c']


def test_filter_sessions_table_empty_ranges(pose_table):
    eids = filter_sessions_table(
        pose_table, {}, ('biased', 'ephys', 'training'))
    assert eids == []


# ─────────────────────────────────────────────────────────────────────────────
# HISTOGRAM_TITLES
# ─────────────────────────────────────────────────────────────────────────────

def test_histogram_titles_cover_measures():
    assert set(HISTOGRAM_TITLES) == set(HISTOGRAM_MEASURES)
    assert HISTOGRAM_TITLES == {
        'paw': 'paw speed @ firstMovement',
        'nose': 'nose speed @ stimOn',
        'tongue_speed': 'tongue speed @ feedback',
        'tongue_likelihood': 'tongue likelihood @ feedback',
    }


# ─────────────────────────────────────────────────────────────────────────────
# format_session_title
# ─────────────────────────────────────────────────────────────────────────────

def test_format_session_title_typical():
    title = format_session_title(
        'a1b2c3d4e5f6', 'SWC_065', '2023-05-12T09:30:00', 'biased', 0.84)
    assert title == 'a1b2c3d4 · SWC_065 · 2023-05-12 · biased · performance: 84%'


def test_format_session_title_accepts_timestamp():
    title = format_session_title(
        'a1b2c3d4e5f6', 'SWC_065', pd.Timestamp('2023-05-12 09:30'),
        'biased', 0.84)
    assert '2023-05-12' in title


def test_format_session_title_no_performance():
    for fraction in (None, float('nan')):
        title = format_session_title(
            'a1b2c3d4e5f6', 'SWC_065', '2023-05-12', 'biased', fraction)
        assert title.endswith('performance: —')


# ─────────────────────────────────────────────────────────────────────────────
# histogram_by_type
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def split_table():
    return pd.DataFrame({
        'paw_speed': [0.0, 1.0, 2.0, 3.0, 4.0, np.nan],
        'session_type': ['biased', 'biased', 'ephys', 'ephys', 'training', 'biased'],
    })


def test_histogram_by_type_shared_edges_span_full_range(split_table):
    edges, _ = histogram_by_type(split_table, 'paw_speed', ('biased', 'ephys'), bins=4)
    # Edges fixed to the full measure range [0, 4] regardless of types shown.
    assert edges[0] == pytest.approx(0.0)
    assert edges[-1] == pytest.approx(4.0)
    assert len(edges) == 5  # bins + 1


def test_histogram_by_type_only_requested_types_and_density(split_table):
    edges, per_type = histogram_by_type(
        split_table, 'paw_speed', ('biased', 'ephys'), bins=4)
    assert set(per_type) == {'biased', 'ephys'}
    widths = np.diff(edges)
    for counts in per_type.values():
        # Density integrates to 1 over the shared edges.
        assert np.sum(counts * widths) == pytest.approx(1.0)


def test_histogram_by_type_drops_nan(split_table):
    # The NaN-valued 'biased' row must not contribute; biased has 2 finite values.
    edges, per_type = histogram_by_type(split_table, 'paw_speed', ('biased',), bins=4)
    widths = np.diff(edges)
    assert np.sum(per_type['biased'] * widths) == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# format_event_timings
# ─────────────────────────────────────────────────────────────────────────────

def test_format_event_timings_signed_deltas():
    labels = format_event_timings(1.0, stimOn=0.7, firstMovement=1.2, feedback=1.5)
    assert labels == [
        'stimOn: +0.30 s', 'firstMovement: -0.20 s', 'feedback: -0.50 s']


def test_format_event_timings_nan_first_movement():
    labels = format_event_timings(
        1.0, stimOn=0.7, firstMovement=float('nan'), feedback=1.5)
    assert labels[1] == 'firstMovement: —'


# ─────────────────────────────────────────────────────────────────────────────
# keypoint_colors
# ─────────────────────────────────────────────────────────────────────────────

def test_keypoint_colors_one_distinct_per_keypoint():
    keypoints = ['nose_tip', 'paw_l', 'paw_r', 'tongue_end_l', 'tongue_end_r']
    colors = keypoint_colors(keypoints)
    assert set(colors) == set(keypoints)
    assert len(set(colors.values())) == len(keypoints)


def test_keypoint_colors_stable_by_name():
    first = keypoint_colors(['paw_l', 'paw_r', 'nose_tip'])
    second = keypoint_colors(['paw_l', 'paw_r', 'nose_tip'])
    assert first['paw_l'] == second['paw_l']
    assert first['nose_tip'] == second['nose_tip']


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
# trial_frame_window
# ─────────────────────────────────────────────────────────────────────────────

def test_trial_frame_window_pads_bounds():
    assert trial_frame_window(1.0, 3.0) == pytest.approx((0.9, 3.5))


def test_trial_frame_window_widens_frame_set():
    # frames at every 0.1 s; trial [1.0, 3.0] padded to [0.9, 3.5] picks up
    # the frame 0.1 s before stimOn and frames out to 0.5 s after feedback.
    camera_times = np.round(np.arange(0.0, 4.0, 0.1), 1)
    lo, hi = trial_frame_window(1.0, 3.0)
    padded = frames_in_trial(camera_times, lo, hi)
    tight = frames_in_trial(camera_times, 1.0, 3.0)
    assert set(tight) < set(padded)
    assert camera_times[padded].min() == pytest.approx(0.9)
    assert camera_times[padded].max() == pytest.approx(3.5)


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


# ─────────────────────────────────────────────────────────────────────────────
# update_pose_qc / save_label
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def pose_pqt(tmp_path):
    """A 2-session pose roll-up parquet with both QC labels at NOT_SET."""
    fpath = tmp_path / 'pose.pqt'
    pd.DataFrame({
        'eid': ['eid0', 'eid1'],
        'qc_lp': ['NOT_SET', 'NOT_SET'],
        'qc_movement': ['NOT_SET', 'NOT_SET'],
    }).to_parquet(fpath)
    return fpath


def test_update_pose_qc_sets_only_target_eid(pose_pqt):
    update_pose_qc(pose_pqt, 'eid1', 'FAIL', 'PASS')
    df = pd.read_parquet(pose_pqt)
    target = df.loc[df['eid'] == 'eid1'].iloc[0]
    assert target['qc_lp'] == 'FAIL'
    assert target['qc_movement'] == 'PASS'
    other = df.loc[df['eid'] == 'eid0'].iloc[0]
    assert other['qc_lp'] == 'NOT_SET'
    assert other['qc_movement'] == 'NOT_SET'


def test_save_label_writes_both_stores(video_h5, pose_pqt):
    h5_path, _ = video_h5
    msg = save_label(h5_path, pose_pqt, 'eid1', 'FAIL', 'PASS')
    with h5py.File(h5_path, 'r') as f:
        assert _decode(f['video'].attrs['qc_lp']) == 'FAIL'
        assert _decode(f['video'].attrs['qc_movement']) == 'PASS'
    row = pd.read_parquet(pose_pqt).set_index('eid').loc['eid1']
    assert row['qc_lp'] == 'FAIL'
    assert row['qc_movement'] == 'PASS'
    assert h5_path.name in msg


def test_save_label_reports_failure_without_raising(tmp_path, pose_pqt):
    # 'a'-mode opens/creates the file but it has no 'video' group -> KeyError.
    missing = tmp_path / 'absent.h5'
    msg = save_label(missing, pose_pqt, 'eid1', 'FAIL', 'PASS')
    assert 'eid1' in msg
    unchanged = pd.read_parquet(pose_pqt).set_index('eid').loc['eid1']
    assert unchanged['qc_lp'] == 'NOT_SET'


# ─────────────────────────────────────────────────────────────────────────────
# LPViewerModel
# ─────────────────────────────────────────────────────────────────────────────

BODYPARTS = ['paw', 'nose', 'tongue_speed', 'tongue_likelihood']
N_TRIALS, N_TIME, N_LAGS = 5, 20, 51


@pytest.fixture
def cohort_model(tmp_path):
    """An LPViewerModel over a 2-session cohort with one session on disk."""
    rng = np.random.default_rng(0)
    tpts = np.linspace(-1.0, 1.0, N_TIME)
    traces = xr.DataArray(
        rng.random((len(BODYPARTS), N_TRIALS, N_TIME)),
        dims=['bodypart', 'trial', 'time'],
        coords={'bodypart': BODYPARTS, 'trial': np.arange(N_TRIALS), 'time': tpts},
    )
    xcorr = {
        'functions': rng.random((3, N_LAGS)),
        'lags': np.linspace(-5.0, 5.0, N_LAGS),
        'peak_lags': np.array([0.1, 0.2, 0.3]),
        'drift': 0.2,
    }
    h5_dir = tmp_path / 'sessions'
    h5_dir.mkdir()
    with h5py.File(h5_dir / 'eid1.h5', 'w') as f:
        grp = f.create_group('video')
        _save_pose_traces(grp, traces)
        _save_pose_xcorr(grp, xcorr)

    df_cohort = pd.DataFrame({
        'eid': ['eid1', 'eid2'],
        'paw': [0.3, 0.8],
        'session_type': ['biased', 'ephys'],
    })
    df_performance = pd.DataFrame({'eid': ['eid1'], 'fraction_correct': [0.77]})
    return LPViewerModel(df_cohort, h5_dir, df_performance), traces


def test_model_filter_couples_range_and_type(cohort_model):
    model, _ = cohort_model
    assert model.filter({'paw': (0.0, 0.5)}, ('biased', 'ephys')) == ['eid1']
    assert model.filter({'paw': (0.0, 1.0)}, ('ephys',)) == ['eid2']


def test_session_panels_trace_shapes(cohort_model):
    model, traces = cohort_model
    panels = model.session_panels('eid1')
    assert set(panels.traces) == set(BODYPARTS)
    assert panels.times.shape == (N_TIME,)
    for trace in panels.traces.values():
        assert trace.shape == (N_TIME,)
    # paw panel is the trial-mean of the stored paw trace
    expected = traces.sel(bodypart='paw').mean('trial').values
    np.testing.assert_allclose(panels.traces['paw'], expected)


def test_session_panels_xcorr_and_performance(cohort_model):
    model, _ = cohort_model
    panels = model.session_panels('eid1')
    assert panels.xcorr['functions'].shape == (3, N_LAGS)
    assert panels.xcorr['lags'].shape == (N_LAGS,)
    assert panels.fraction_correct == pytest.approx(0.77)


# ─────────────────────────────────────────────────────────────────────────────
# build_cohort (launcher glue)
# ─────────────────────────────────────────────────────────────────────────────

def test_build_cohort_enriches_with_session_metadata():
    import scripts.lp_viewer as launcher

    df_pose = pd.DataFrame({'eid': ['a', 'b'], 'paw': [0.1, 0.2]})
    df_sessions = pd.DataFrame({
        'eid': ['a', 'b', 'c'],
        'subject': ['m1', 'm2', 'm3'],
        'start_time': ['t1', 't2', 't3'],
        'number': [1, 1, 1],
        'session_type': ['biased', 'ephys', 'training'],
    })
    out = launcher.build_cohort(df_pose, df_sessions)
    # one row per pose eid, no duplication, only pose sessions kept
    assert out['eid'].tolist() == ['a', 'b']
    assert out['paw'].tolist() == [0.1, 0.2]
    assert out['session_type'].tolist() == ['biased', 'ephys']
    assert 'subject' in out.columns
