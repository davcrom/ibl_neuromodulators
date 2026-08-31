"""Tests for the wheel products: raw position, preprocessed velocity, responses."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock


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


def _raw_wheel(duration=10.0, rate=250.0, seed=0):
    """Encoder samples at an irregular rate, position ramping at 1 rad/s.

    Real encoder timestamps are event-driven rather than uniform, so the sample
    spacing is jittered — that irregularity is what `wheel/raw` must preserve.
    """
    rng = np.random.default_rng(seed)
    steps = rng.uniform(0.5, 1.5, int(duration * rate)) / rate
    timestamps = np.cumsum(steps)
    return {'timestamps': timestamps, 'position': timestamps.copy()}


def _make_session(mock_session_series, tmp_path, raw_wheel=None):
    """Session whose H5 lives in `tmp_path` and whose ONE serves `raw_wheel`."""
    from iblnm.data import PhotometrySession
    ps = PhotometrySession(mock_session_series, one=MagicMock(), load_data=False)
    ps.filepath = tmp_path / f'{ps.eid}.h5'
    ps.one.load_object.return_value = (
        _raw_wheel() if raw_wheel is None else raw_wheel)
    return ps


# ─────────────────────────────────────────────────────────────────────────────
# wheel/raw — the irregular encoder samples
# ─────────────────────────────────────────────────────────────────────────────

class TestRawWheelProduct:

    def test_keeps_the_irregular_sample_times(self, mock_session_series, tmp_path):
        """The fetched position keeps ONE's own timestamps, ungridded."""
        raw = _raw_wheel()
        ps = _make_session(mock_session_series, tmp_path, raw)
        position = ps.load_raw_wheel()

        np.testing.assert_allclose(position.index.to_numpy(), raw['timestamps'])
        np.testing.assert_allclose(position.to_numpy(), raw['position'])
        assert ps.wheel_position is position

    def test_roundtrips_through_the_raw_group(self, mock_session_series, tmp_path,
                                              monkeypatch):
        """With `store_raw` on, position and timestamps come back unchanged."""
        monkeypatch.setattr('iblnm.data.store_raw', True)
        raw = _raw_wheel()
        ps = _make_session(mock_session_series, tmp_path, raw)
        ps.load_raw_wheel()
        ps.save_h5(groups=['wheel'])

        fresh = _make_session(mock_session_series, tmp_path)
        fresh.load_h5(groups=['wheel'])
        np.testing.assert_allclose(fresh.wheel_position.index.to_numpy(),
                                   raw['timestamps'])
        np.testing.assert_allclose(fresh.wheel_position.to_numpy(),
                                   raw['position'])

    def test_fetch_wheel_goes_to_alyx_exactly_once(self, mock_session_series,
                                                    tmp_path):
        """`fetch_wheel` is the whole Alyx trip: one `load_object`, no store."""
        raw = _raw_wheel()
        ps = _make_session(mock_session_series, tmp_path, raw)
        position = ps.fetch_wheel()

        assert ps.one.load_object.call_count == 1
        assert isinstance(position, pd.Series)
        np.testing.assert_allclose(position.index.to_numpy(), raw['timestamps'])
        assert ps.wheel_position is position
        assert not ps.filepath.exists()

    def test_fetch_wheel_refetches_on_every_call(self, mock_session_series,
                                                  tmp_path, monkeypatch):
        """Unlike `load_raw_wheel`, a fetch never consults the store."""
        monkeypatch.setattr('iblnm.data.store_raw', True)
        ps = _make_session(mock_session_series, tmp_path)
        ps.load_raw_wheel()
        ps.save_h5(groups=['wheel'])
        ps.fetch_wheel()

        assert ps.one.load_object.call_count == 2

    def test_missing_extracted_data_when_raw_ssv_present(self, mock_session_series,
                                                         tmp_path):
        """Wheel ALF missing but the raw encoder file present → not extracted."""
        from iblnm.validation import MissingExtractedData
        from one.alf.exceptions import ALFObjectNotFound
        ps = _make_session(mock_session_series, tmp_path)
        ps.one.load_object.side_effect = ALFObjectNotFound('wheel')
        ps.one.load_dataset.return_value = MagicMock()

        with pytest.raises(MissingExtractedData):
            ps.load_raw_wheel()

    def test_missing_raw_data_when_encoder_file_absent(self, mock_session_series,
                                                       tmp_path):
        """Neither the ALF nor the raw encoder file → nothing was recorded."""
        from iblnm.validation import MissingRawData
        from one.alf.exceptions import ALFObjectNotFound
        ps = _make_session(mock_session_series, tmp_path)
        ps.one.load_object.side_effect = ALFObjectNotFound('wheel')
        ps.one.load_dataset.side_effect = ALFObjectNotFound('encoderPositions')

        with pytest.raises(MissingRawData):
            ps.load_raw_wheel()


# ─────────────────────────────────────────────────────────────────────────────
# wheel/preprocessed — velocity on the WHEEL_FS grid
# ─────────────────────────────────────────────────────────────────────────────

class TestPreprocessedWheelProduct:

    def test_velocity_is_uniformly_sampled_at_wheel_fs(self, mock_session_series,
                                                       tmp_path):
        """Position ramping at 1 rad/s differentiates to 1 rad/s at WHEEL_FS."""
        from iblnm.config import WHEEL_FS
        ps = _make_session(mock_session_series, tmp_path)
        velocity = ps.load_wheel()

        steps = np.diff(velocity.index.to_numpy())
        np.testing.assert_allclose(steps, 1 / WHEEL_FS, atol=1e-9)
        # The filter's edges ring, so score the settled interior only.
        interior = velocity.iloc[WHEEL_FS:-WHEEL_FS]
        np.testing.assert_allclose(interior.to_numpy(), 1.0, atol=1e-3)
        assert ps.wheel_velocity is velocity

    def test_matches_the_brainbox_differentiation(self, mock_session_series,
                                                  tmp_path):
        """The velocity is what brainbox computed before the product split."""
        from brainbox.behavior.wheel import interpolate_position, velocity_filtered
        from iblnm.config import WHEEL_FS
        raw = _raw_wheel(seed=3)
        ps = _make_session(mock_session_series, tmp_path, raw)
        velocity = ps.load_wheel()

        position, times = interpolate_position(
            raw['timestamps'], raw['position'], freq=WHEEL_FS)
        expected, _ = velocity_filtered(position, fs=WHEEL_FS)
        np.testing.assert_allclose(velocity.index.to_numpy(), times)
        np.testing.assert_allclose(velocity.to_numpy(), expected)

    def test_extract_computes_from_the_position_and_saves_nothing(
            self, mock_session_series, tmp_path):
        """`extract_wheel_velocity` differentiates in memory; only loads write."""
        from brainbox.behavior.wheel import interpolate_position, velocity_filtered
        from iblnm.config import WHEEL_FS
        raw = _raw_wheel(seed=5)
        ps = _make_session(mock_session_series, tmp_path, raw)
        ps.load_raw_wheel()
        velocity = ps.extract_wheel_velocity()

        position, times = interpolate_position(
            raw['timestamps'], raw['position'], freq=WHEEL_FS)
        expected, _ = velocity_filtered(position, fs=WHEEL_FS)
        np.testing.assert_allclose(velocity.index.to_numpy(), times)
        np.testing.assert_allclose(velocity.to_numpy(), expected)
        assert ps.wheel_velocity is velocity
        assert not ps.filepath.exists()

    def test_load_writes_and_stamps_the_preprocessed_group(
            self, mock_session_series, tmp_path):
        """The load is what saves: one velocity dataset under its own label."""
        import h5py
        from iblnm.data import WHEEL_LABEL
        ps = _make_session(mock_session_series, tmp_path)
        velocity = ps.load_wheel()

        with h5py.File(ps.filepath, 'r') as h5:
            stored = h5[f'wheel/{WHEEL_LABEL}/preprocessed/signal'][:]
        np.testing.assert_allclose(stored, velocity.to_numpy())
        assert ps.product_status('wheel/preprocessed') == 'current'

    def test_reads_stored_product_without_fetching(self, mock_session_series,
                                                   tmp_path):
        """A second session over the same file reads it and never fetches."""
        ps = _make_session(mock_session_series, tmp_path)
        built = ps.load_wheel()
        assert ps.product_status('wheel/preprocessed') == 'current'

        fresh = _make_session(mock_session_series, tmp_path)
        velocity = fresh.load_wheel()

        fresh.one.load_object.assert_not_called()
        pd.testing.assert_series_equal(velocity, built)

    def test_raises_on_stale_stamp(self, mock_session_series, tmp_path):
        """A velocity computed at a rate that changed is not reused."""
        import h5py
        from iblnm.data import WHEEL_LABEL, _write_stamp
        from iblnm.validation import StaleProduct
        ps = _make_session(mock_session_series, tmp_path)
        ps.load_wheel()
        with h5py.File(ps.filepath, 'a') as h5:
            _write_stamp(h5[f'wheel/{WHEEL_LABEL}/preprocessed'],
                         ps.spec['wheel/preprocessed'] | {'fs': 1000})

        assert ps.product_status('wheel/preprocessed') == 'stale'
        with pytest.raises(StaleProduct, match='wheel/preprocessed'):
            ps.load_wheel()

    def test_rebuild_skips_the_stored_product(self, mock_session_series, tmp_path):
        """A product named in self.rebuild is refetched even when current."""
        ps = _make_session(mock_session_series, tmp_path)
        ps.load_wheel()
        ps.rebuild.add('wheel/preprocessed')
        ps.load_wheel()

        assert ps.one.load_object.call_count == 2

    def test_absent_before_anything_is_stored(self, mock_session_series, tmp_path):
        ps = _make_session(mock_session_series, tmp_path)
        assert ps.product_status('wheel/raw') == 'absent'
        assert ps.product_status('wheel/preprocessed') == 'absent'


# ─────────────────────────────────────────────────────────────────────────────
# wheel/responses — the stimOn → feedback cut
# ─────────────────────────────────────────────────────────────────────────────

def _make_trials():
    """Two trials whose windows differ, so the shared time axis is testable."""
    return pd.DataFrame({
        'trial': [7, 9],
        'stimOn_times':   [1.0, 3.0],
        'feedback_times': [1.5, 4.0],
    })


class TestWheelResponsesProduct:

    @pytest.fixture
    def wheeled_session(self, mock_session_series, tmp_path):
        ps = _make_session(mock_session_series, tmp_path)
        ps.trials = _make_trials()
        return ps

    def test_nan_tail_starts_at_each_trials_feedback(self, wheeled_session):
        """One time axis spanning the longest trial; each trial masked at its own end."""
        from iblnm.config import WHEEL_FS
        responses = wheeled_session.load_responses('wheel')
        matrix = responses['velocity'].sel(event='stimOn_times')

        tpts = matrix.coords['time'].to_numpy()
        np.testing.assert_allclose(tpts[0], 0.0)
        np.testing.assert_allclose(tpts[-1], 1.0 - 1 / WHEEL_FS)
        assert matrix.shape == (2, WHEEL_FS)
        # Trial 0 runs 0.5 s; get_responses masks tpts strictly greater than
        # that, so the boundary sample survives and NaN starts after it.
        assert not np.isnan(matrix.values[0, :51]).any()
        assert np.isnan(matrix.values[0, 51:]).all()
        assert not np.isnan(matrix.values[1]).any()

    def test_trial_coord_comes_from_the_trials_table(self, wheeled_session):
        responses = wheeled_session.load_responses('wheel')
        np.testing.assert_array_equal(
            responses['velocity'].coords['trial'].to_numpy(), [7, 9])

    def test_events_live_in_the_stamp_not_standalone_attrs(self, wheeled_session):
        """t0_event / t1_event are read back from the product's spec stamp."""
        import h5py
        from iblnm.data import WHEEL_LABEL, _read_stamp
        wheeled_session.load_responses('wheel')

        with h5py.File(wheeled_session.filepath, 'r') as h5:
            group = h5[f'wheel/{WHEEL_LABEL}/responses']
            stamp = _read_stamp(group)
            assert 't0_event' not in group.attrs
            assert 't1_event' not in group.attrs
        assert stamp['t0_event'] == 'stimOn_times'
        assert stamp['t1_event'] == 'feedback_times'
        assert wheeled_session.product_status('wheel/responses') == 'current'

    def test_roundtrips_through_h5(self, wheeled_session, mock_session_series,
                                   tmp_path):
        import xarray as xr
        built = wheeled_session.load_responses('wheel')

        fresh = _make_session(mock_session_series, tmp_path)
        reloaded = fresh.load_responses('wheel')

        assert set(reloaded) == {'velocity'}
        xr.testing.assert_allclose(reloaded['velocity'], built['velocity'])

    def test_raises_on_stale_stamp(self, wheeled_session):
        import h5py
        from iblnm.data import WHEEL_LABEL, _write_stamp
        from iblnm.validation import StaleProduct
        wheeled_session.load_responses('wheel')
        with h5py.File(wheeled_session.filepath, 'a') as h5:
            _write_stamp(h5[f'wheel/{WHEEL_LABEL}/responses'],
                         wheeled_session.spec['wheel/responses']
                         | {'t1_event': 'response_times'})

        with pytest.raises(StaleProduct, match='wheel/responses'):
            wheeled_session.load_responses('wheel')

    def test_peak_velocity_matches_the_old_matrix(self, wheeled_session):
        """`_peak_velocity` on the responses product reproduces the old values."""
        from iblnm.analysis import _peak_velocity, get_responses
        velocity = wheeled_session.load_wheel()
        old_matrix, _ = get_responses(
            velocity,
            events=wheeled_session.trials['stimOn_times'].to_numpy(),
            t0=0.0,
            t1=wheeled_session.trials['feedback_times'].to_numpy(),
        )
        responses = wheeled_session.load_responses('wheel')
        new_matrix = responses['velocity'].sel(event='stimOn_times').values

        np.testing.assert_allclose(_peak_velocity(new_matrix, 2),
                                   _peak_velocity(old_matrix, 2))
