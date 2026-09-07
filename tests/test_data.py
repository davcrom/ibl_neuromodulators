"""Tests for iblnm.data module."""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from unittest.mock import MagicMock, patch

from iblnm.config import REQUIRED_CONTRASTS
from iblnm.data import WHEEL_LABEL
from iblnm.util import LOG_COLUMNS, contrast_transform

# Session-lifetime scratch directory for the helpers that must write a real H5
# (a load method reads its own stored product back, so an in-memory attribute
# is not enough). Cleaned up when the interpreter exits.
_SCRATCH_H5_DIR = tempfile.TemporaryDirectory()


# =============================================================================
# Exception Tests
# =============================================================================

class TestCustomExceptions:
    """Custom exception classes exist and behave correctly."""

    def test_insufficient_trials_is_exception(self):
        from iblnm.validation import InsufficientTrials
        assert issubclass(InsufficientTrials, Exception)

    def test_block_structure_bug_is_exception(self):
        from iblnm.validation import BlockStructureBug
        assert issubclass(BlockStructureBug, Exception)

    def test_incomplete_event_times_is_exception(self):
        from iblnm.validation import IncompleteEventTimes
        assert issubclass(IncompleteEventTimes, Exception)

    def test_incomplete_event_times_stores_missing_events(self):
        from iblnm.validation import IncompleteEventTimes
        exc = IncompleteEventTimes(['goCue_times', 'feedback_times'])
        assert exc.missing_events == ['goCue_times', 'feedback_times']
        assert 'goCue_times' in str(exc)

    def test_trials_not_in_photometry_time_is_exception(self):
        from iblnm.validation import TrialsNotInPhotometryTime
        assert issubclass(TrialsNotInPhotometryTime, Exception)

    def test_missing_extracted_data_is_exception(self):
        from iblnm.validation import MissingExtractedData
        assert issubclass(MissingExtractedData, Exception)

    def test_missing_raw_data_is_exception(self):
        from iblnm.validation import MissingRawData
        assert issubclass(MissingRawData, Exception)

    def test_band_inversion_is_exception(self):
        from iblnm.validation import BandInversion
        assert issubclass(BandInversion, Exception)

    def test_early_samples_is_exception(self):
        from iblnm.validation import EarlySamples
        assert issubclass(EarlySamples, Exception)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_session_series():
    """Mock session metadata."""
    return pd.Series({
        'eid': 'test-eid-123',
        'subject': 'test_mouse',
        'start_time': '2024-01-01T10:00:00',
        'number': 1,
        'lab': 'test_lab',
        'projects': ['test_project'],
        'url': 'https://example.com',
        'session_n': 1,
        'task_protocol': 'test_protocol',
        'session_type': 'training',
    })


@pytest.fixture
def mock_photometry_data():
    """Synthetic photometry data with known bleaching and correlation."""
    np.random.seed(42)
    t = np.linspace(0, 600, 18000)  # 30 min at ~30 Hz

    # Known bleaching decay: tau = 300
    # Use low noise so exponential fit recovers tau accurately
    bleaching = 1000 * np.exp(-t / 300)
    noise_gcamp = 1 * np.random.randn(len(t))  # Low noise for accurate tau recovery

    gcamp = pd.DataFrame({
        'VTA': bleaching + noise_gcamp + 500,
    }, index=t)

    # Isosbestic: correlated with GCaMP bleaching (same decay, different scale)
    noise_iso = 0.5 * np.random.randn(len(t))
    iso = pd.DataFrame({
        'VTA': 0.8 * bleaching + noise_iso + 400,
    }, index=t)

    return {'GCaMP': gcamp, 'Isosbestic': iso}


@pytest.fixture
def mock_photometry_session(mock_session_series, mock_photometry_data, tmp_path):
    """PhotometrySession with injected mock data, writing into tmp_path.

    `filepath` is redirected away from SESSIONS_H5_DIR because the load
    methods write the products they build — a test calling `preprocess`
    would otherwise land in the real store.
    """
    from iblnm.data import PhotometrySession

    mock_one = MagicMock()
    session = PhotometrySession(mock_session_series, one=mock_one, load_data=False)
    session.filepath = tmp_path / f'{session.eid}.h5'

    # Inject mock photometry
    session.photometry = mock_photometry_data

    return session

@pytest.fixture
def minimal_session_series():
    """Minimal session metadata — only fields available from REST list."""
    return pd.Series({
        'eid': 'test-eid-minimal',
        'subject': 'test_mouse',
        'start_time': '2024-01-01T10:00:00',
        'number': 1,
    })


@pytest.fixture
def full_session_series():
    """Full session metadata — all SESSION_SCHEMA fields populated."""
    return pd.Series({
        'eid': 'test-eid-full',
        'subject': 'test_mouse',
        'start_time': '2024-01-01T10:00:00',
        'number': 1,
        'lab': 'test_lab',
        'projects': ['test_project'],
        'url': 'https://example.com',
        'session_n': 3,
        'task_protocol': '_iblrig_tasks_biasedChoiceWorld',
        'session_type': 'biased',
        'NM': 'DA',
        'strain': 'Thy1-GCaMP6s',
        'line': 'Thy1',
        'genotype': ['Thy1-GCaMP6s/wt'],
        'users': ['user1', 'user2'],
        'end_time': '2024-01-01T11:00:00',
        'brain_region': ['VTA', 'SNc'],
        'hemisphere': ['l', 'r'],
        'target_NM': ['VTA-DA', 'SNc-DA'],
        'datasets': ['_ibl_trials.table.pqt'],
        'session_length': 3600,
        'day_n': 5,
    })


# =============================================================================
# Init and Serialization Tests
# =============================================================================

class TestInit:
    """Tests for PhotometrySession.__init__."""

    def test_minimal_init(self, minimal_session_series):
        """Init with only required fields; optional fields get defaults."""
        from iblnm.data import PhotometrySession
        mock_one = MagicMock()
        ps = PhotometrySession(minimal_session_series, one=mock_one, load_data=False)

        assert ps.eid == 'test-eid-minimal'
        assert ps.subject == 'test_mouse'
        assert ps.number == 1
        # Optional fields should have safe defaults
        assert ps.task_protocol == ''
        assert ps.session_type == ''
        assert ps.NM is None
        assert ps.strain is None
        assert ps.line is None
        assert ps.genotype == []
        assert ps.users == []
        assert ps.end_time is None
        assert ps.brain_region == []
        assert ps.hemisphere == []
        assert ps.target_NM == []
        assert ps.datasets == []
        assert ps.session_length is None
        assert ps.day_n is None
        assert ps.errors == []

    def test_init_sets_default_filepath(self, minimal_session_series):
        """filepath defaults to SESSIONS_H5_DIR / {eid}.h5 on init."""
        from iblnm.data import PhotometrySession
        from iblnm.config import SESSIONS_H5_DIR
        mock_one = MagicMock()
        ps = PhotometrySession(minimal_session_series, one=mock_one, load_data=False)
        assert ps.filepath == SESSIONS_H5_DIR / 'test-eid-minimal.h5'

    def test_full_init(self, full_session_series):
        """Init with all fields populated."""
        from iblnm.data import PhotometrySession
        mock_one = MagicMock()
        ps = PhotometrySession(full_session_series, one=mock_one, load_data=False)

        assert ps.eid == 'test-eid-full'
        assert ps.strain == 'Thy1-GCaMP6s'
        assert ps.line == 'Thy1'
        assert ps.genotype == ['Thy1-GCaMP6s/wt']
        assert ps.NM == 'DA'
        assert ps.users == ['user1', 'user2']
        assert ps.end_time == '2024-01-01T11:00:00'
        assert ps.target_NM == ['VTA-DA', 'SNc-DA']
        assert ps.session_length == 3600
        assert ps.day_n == 5
        assert ps.session_type == 'biased'
        assert ps.task_protocol == '_iblrig_tasks_biasedChoiceWorld'

    def test_errors_initialized_empty(self, mock_session_series):
        """errors list is initialized empty."""
        from iblnm.data import PhotometrySession
        mock_one = MagicMock()
        ps = PhotometrySession(mock_session_series, one=mock_one, load_data=False)
        assert ps.errors == []
        assert isinstance(ps.errors, list)

    def test_scalar_region_fields_normalized_to_lists(self):
        """Scalar region fields (a recording row) become length-1 lists."""
        from iblnm.data import PhotometrySession
        recording_row = pd.Series({
            'eid': 'test-eid-rec',
            'subject': 'test_mouse',
            'start_time': '2024-01-01T10:00:00',
            'number': 1,
            'brain_region': 'VTA',
            'hemisphere': 'l',
            'target_NM': 'VTA-DA',
        })
        mock_one = MagicMock()
        ps = PhotometrySession(recording_row, one=mock_one, load_data=False)
        assert ps.brain_region == ['VTA']
        assert ps.hemisphere == ['l']
        assert ps.target_NM == ['VTA-DA']

    def test_data_attributes_absent_until_loaded(self, minimal_session_series):
        """A data attribute exists only once that product has been built.

        The load methods guard on the attribute being there, so a session that
        has fetched nothing must carry none of them — including the ones the
        loader parent declares as dataclass fields.
        """
        from iblnm.data import PhotometrySession
        ps = PhotometrySession(minimal_session_series, one=MagicMock(),
                               load_data=False)
        for attribute in ('trials', 'performance', 'neurophotometrics',
                          'neurophotometrics_qc', 'photometry_responses',
                          'photometry_qc', 'wheel_position', 'wheel_velocity',
                          'wheel_responses', 'pose', 'pose_times',
                          'motion_energy', 'movement_signals',
                          'movement_responses', 'pose_xcorr'):
            assert not hasattr(ps, attribute), attribute

    def test_init_without_one(self, full_session_series):
        """Constructing with no ONE connection populates all metadata."""
        from iblnm.data import PhotometrySession
        ps = PhotometrySession(full_session_series)

        assert ps.one is None
        assert ps.eid == 'test-eid-full'
        assert ps.subject == 'test_mouse'
        assert ps.brain_region == ['VTA', 'SNc']
        assert ps.hemisphere == ['l', 'r']
        assert ps.target_NM == ['VTA-DA', 'SNc-DA']


@pytest.fixture
def stored_session(minimal_session_series, tmp_path):
    """Session whose H5 file is built by hand, with a group-writing helper."""
    import h5py
    from iblnm.data import PhotometrySession

    ps = PhotometrySession(minimal_session_series)
    ps.filepath = tmp_path / f'{ps.eid}.h5'

    def write(path, data=True):
        """Create `path` in the session's H5 file, with data unless told not to."""
        with h5py.File(ps.filepath, 'a') as h5:
            group = h5.require_group(path)
            if data:
                group.create_dataset('values', data=np.arange(3.0))

    return ps, write


class TestStoredProductExists:
    """Tests for PhotometrySession.stored_product_exists."""

    def test_absent_when_file_missing(self, stored_session):
        """No H5 file at all means every product is absent."""
        ps, _ = stored_session
        assert not ps.filepath.exists()
        assert not ps.stored_product_exists('trials/table')

    def test_absent_when_modality_group_missing(self, stored_session):
        """A file holding other modalities still reports this one absent."""
        ps, write = stored_session
        write('trials/table')
        assert not ps.stored_product_exists('photometry/preprocessed')

    def test_absent_when_product_group_missing(self, stored_session):
        """The modality exists but not this product under it."""
        ps, write = stored_session
        write('photometry/VTA/preprocessed')
        assert not ps.stored_product_exists('photometry/responses')

    def test_present_for_unlabelled_product(self, stored_session):
        """A product with no label level is found directly under its modality."""
        ps, write = stored_session
        write('trials/table')
        assert ps.stored_product_exists('trials/table')

    def test_present_for_labelled_product(self, stored_session):
        """A labelled product is found by walking the label level."""
        ps, write = stored_session
        write('photometry/VTA/raw/qc')
        write('photometry/SNc/raw/qc')
        assert ps.stored_product_exists('photometry/raw/qc')

    def test_present_for_a_group_carrying_only_attrs(self, stored_session):
        """QC products are flat attrs with no datasets, and still count."""
        import h5py
        ps, _ = stored_session
        with h5py.File(ps.filepath, 'a') as h5:
            h5.require_group('video/times/qc').attrs['framerate_from_tpts'] = 60.0
        assert ps.stored_product_exists('video/times/qc')

    def test_absent_for_a_container_holding_only_a_subgroup(self, stored_session):
        """`photometry/{region}/raw` with `store_raw` off holds only its qc/."""
        ps, write = stored_session
        write('photometry/VTA/raw/qc')
        assert not ps.stored_product_exists('photometry/raw')

    def test_absent_for_a_group_carrying_only_an_old_stamp(self, stored_session):
        """A stamp left by an earlier version is not data: the product is absent."""
        import h5py
        import json
        ps, write = stored_session
        write('photometry/VTA/raw', data=False)
        with h5py.File(ps.filepath, 'a') as h5:
            h5['photometry/VTA/raw'].attrs['spec_json'] = json.dumps({'fs': 30})
        assert not ps.stored_product_exists('photometry/raw')


class TestLoadingPrimitives:
    """Round-trip tests for the three save/load pairs keyed by data structure."""

    @pytest.fixture
    def writing_session(self, minimal_session_series, tmp_path):
        """Session with an H5 path to write the primitives into."""
        from iblnm.data import PhotometrySession
        ps = PhotometrySession(minimal_session_series)
        ps.filepath = tmp_path / f'{ps.eid}.h5'
        return ps

    def test_time_series_roundtrip_series(self, writing_session):
        """A time-indexed Series survives the round trip as float64."""
        import h5py
        from iblnm.data import _load_time_series, _save_time_series
        ps = writing_session
        signal = pd.Series(np.array([1.0, 2.5, -3.25]),
                           index=np.array([0.0, 0.1, 0.2]))

        with h5py.File(ps.filepath, 'w') as h5:
            _save_time_series(h5.create_group('photometry/VTA/preprocessed'),
                              signal)
        with h5py.File(ps.filepath, 'r') as h5:
            out = _load_time_series(h5['photometry/VTA/preprocessed'])

        assert isinstance(out, pd.Series)
        assert out.dtype == np.float64
        np.testing.assert_array_equal(out.index.values, signal.index.values)
        np.testing.assert_array_equal(out.values, signal.values)

    def test_time_series_roundtrip_dataframe(self, writing_session):
        """A multi-column time-indexed DataFrame survives with its columns."""
        import h5py
        from iblnm.data import _load_time_series, _save_time_series
        ps = writing_session
        frame = pd.DataFrame(
            {'VTA': [1.0, 2.0, 3.0], 'SNc': [-1.0, -2.0, -3.0]},
            index=np.array([0.0, 0.1, 0.2]),
        )

        with h5py.File(ps.filepath, 'w') as h5:
            _save_time_series(h5.create_group('photometry/preprocessed'),
                              frame)
        with h5py.File(ps.filepath, 'r') as h5:
            out = _load_time_series(h5['photometry/preprocessed'])

        assert isinstance(out, pd.DataFrame)
        pd.testing.assert_frame_equal(out[frame.columns], frame)

    def test_time_series_save_is_found_by_the_presence_check(self, writing_session):
        """What the save writes is what `stored_product_exists` looks for."""
        import h5py
        from iblnm.data import _save_time_series
        ps = writing_session
        signal = pd.Series([1.0, 2.0], index=[0.0, 0.1])

        with h5py.File(ps.filepath, 'w') as h5:
            _save_time_series(h5.create_group('photometry/VTA/preprocessed'),
                              signal)

        assert ps.stored_product_exists('photometry/preprocessed')

    def test_peri_event_matrix_roundtrip(self, writing_session):
        """A DataArray(event, trial, time) survives with its coords intact.

        Trial coords are non-contiguous integers: the trial axis is keyed by
        the raw ONE trials-table index, so extraction that drops trials leaves
        gaps that must not be re-indexed away.
        """
        import h5py
        from iblnm.data import _load_peri_event_matrix, _save_peri_event_matrix
        ps = writing_session
        trials = np.array([0, 3, 7, 12])
        responses = xr.DataArray(
            np.arange(2 * 4 * 5, dtype=np.float64).reshape(2, 4, 5),
            dims=['event', 'trial', 'time'],
            coords={'event': ['stimOnTrigger_times', 'feedback_times'],
                    'trial': trials,
                    'time': np.linspace(-0.2, 0.8, 5)},
        )

        with h5py.File(ps.filepath, 'w') as h5:
            _save_peri_event_matrix(h5.create_group('photometry/VTA/responses'),
                                    responses)
        with h5py.File(ps.filepath, 'r') as h5:
            out = _load_peri_event_matrix(h5['photometry/VTA/responses'])

        xr.testing.assert_allclose(out.sortby('event'), responses.sortby('event'))
        np.testing.assert_array_equal(out.coords['trial'].values, trials)
        assert ps.stored_product_exists('photometry/responses')

    def test_scalars_roundtrip_with_nan(self, writing_session):
        """A flat scalar mapping survives as attrs, NaN included.

        A metric that could not be computed is stored as NaN rather than
        omitted, so the reader must not confuse it with a missing key.
        """
        import h5py
        from iblnm.data import _load_scalars, _save_scalars
        ps = writing_session
        qc = {'n_unique_samples_GCaMP': 0.031,
              'median_absolute_deviance_GCaMP': np.nan,
              'ar_score_GCaMP': -1.5}

        with h5py.File(ps.filepath, 'w') as h5:
            _save_scalars(h5.create_group('photometry/VTA/raw/qc'), qc)
        with h5py.File(ps.filepath, 'r') as h5:
            out = _load_scalars(h5['photometry/VTA/raw/qc'])

        assert set(out) == set(qc)
        assert np.isnan(out['median_absolute_deviance_GCaMP'])
        assert out['n_unique_samples_GCaMP'] == pytest.approx(0.031)
        assert out['ar_score_GCaMP'] == pytest.approx(-1.5)
        assert ps.stored_product_exists('photometry/raw/qc')


class TestStoreRawGating:
    """`config.store_raw` decides whether the raw/ groups are written at all."""

    @pytest.fixture
    def raw_session(self, mock_session_series, mock_photometry_data, tmp_path):
        """Session holding raw photometry bands and one region's raw QC."""
        from iblnm.data import PhotometrySession
        session = PhotometrySession(mock_session_series, one=MagicMock(),
                                    load_data=False)
        session.filepath = tmp_path / f'{session.eid}.h5'
        session.photometry = dict(mock_photometry_data)
        session.photometry_qc = {'VTA': {'n_unique_samples_GCaMP': 0.5}}
        return session

    def test_photometry_raw_absent_but_qc_stored_when_off(self, raw_session):
        """Raw bands are dropped; the QC scored from them is kept regardless."""
        raw_session.save_h5(groups=['photometry'])

        assert not raw_session.stored_product_exists('photometry/raw')
        assert raw_session.stored_product_exists('photometry/raw/qc')

    def test_photometry_raw_roundtrips_when_on(self, raw_session, monkeypatch,
                                               mock_session_series,
                                               mock_photometry_data):
        """Flipping the constant stores each band, stamped, and reads it back."""
        from iblnm.data import PhotometrySession
        monkeypatch.setattr('iblnm.data.store_raw', True)
        raw_session.save_h5(groups=['photometry'])

        assert raw_session.stored_product_exists('photometry/raw')
        fresh = PhotometrySession(mock_session_series, one=MagicMock(),
                                  load_data=False)
        fresh.load_h5(raw_session.filepath)
        for band, frame in mock_photometry_data.items():
            pd.testing.assert_frame_equal(fresh.photometry[band], frame)

    def test_wheel_raw_gated(self, raw_session, monkeypatch,
                             mock_session_series):
        """The encoder position is written only with the constant flipped on."""
        from iblnm.data import PhotometrySession
        raw_session.wheel_position = pd.Series([0.0, 0.1, 0.3],
                                               index=[0.0, 0.5, 1.7])
        raw_session.save_h5(groups=['wheel'])
        assert not raw_session.stored_product_exists('wheel/raw')

        monkeypatch.setattr('iblnm.data.store_raw', True)
        raw_session.save_h5(groups=['wheel'])
        assert raw_session.stored_product_exists('wheel/raw')

        fresh = PhotometrySession(mock_session_series, one=MagicMock(),
                                  load_data=False)
        fresh.load_h5(raw_session.filepath)
        pd.testing.assert_series_equal(fresh.wheel_position,
                                       raw_session.wheel_position)

    def test_load_raw_wheel_reads_stored_position_without_fetching(
            self, raw_session, monkeypatch, mock_session_series):
        """Stored encoder samples make the raw wheel readable without Alyx."""
        from iblnm.data import PhotometrySession
        monkeypatch.setattr('iblnm.data.store_raw', True)
        raw_session.wheel_position = pd.Series([0.0, 0.1, 0.3],
                                               index=[0.0, 0.5, 1.7])
        raw_session.save_h5(groups=['wheel'])

        fresh = PhotometrySession(mock_session_series, one=MagicMock(),
                                  load_data=False)
        fresh.filepath = raw_session.filepath
        position = fresh.load_raw_wheel()

        fresh.one.load_object.assert_not_called()
        pd.testing.assert_series_equal(position, raw_session.wheel_position)

    def test_load_raw_wheel_fetches_when_position_not_stored(
            self, raw_session, mock_session_series):
        """With the encoder samples dropped, the same read goes back to Alyx."""
        from iblnm.data import PhotometrySession
        raw_session.wheel_position = pd.Series([0.0, 0.1, 0.3],
                                               index=[0.0, 0.5, 1.7])
        raw_session.save_h5(groups=['wheel'])

        fresh = PhotometrySession(mock_session_series, one=MagicMock(),
                                  load_data=False)
        fresh.filepath = raw_session.filepath
        fresh.one.load_object.return_value = {'position': [0.0, 0.2],
                                              'timestamps': [0.0, 0.4]}
        position = fresh.load_raw_wheel()

        assert fresh.one.load_object.call_count == 1
        pd.testing.assert_series_equal(
            position, pd.Series([0.0, 0.2], index=[0.0, 0.4]))

    def test_video_raw_datasets_gated(self, raw_session, monkeypatch,
                                      mock_session_series):
        """Each of the three independently-fetched video datasets is gated."""
        from iblnm.data import PhotometrySession
        products = ('video/times', 'video/pose', 'video/motion_energy')
        pose = pd.DataFrame({'paw_l_x': [1.0, 2.0, 3.0],
                             'paw_l_likelihood': [0.9, 0.8, 0.99]})
        raw_session.pose_times = np.array([0.0, 0.05, 0.10])
        raw_session.pose = pose
        raw_session.motion_energy = np.array([0.5, 1.5, 2.5])
        raw_session.save_h5(groups=['video'])
        assert not any(raw_session.stored_product_exists(p) for p in products)

        monkeypatch.setattr('iblnm.data.store_raw', True)
        raw_session.save_h5(groups=['video'])
        assert all(raw_session.stored_product_exists(p) for p in products)

        fresh = PhotometrySession(mock_session_series, one=MagicMock(),
                                  load_data=False)
        fresh.load_h5(raw_session.filepath)
        np.testing.assert_array_equal(fresh.pose_times, raw_session.pose_times)
        pd.testing.assert_frame_equal(fresh.pose[pose.columns], pose)
        np.testing.assert_array_equal(fresh.motion_energy,
                                      raw_session.motion_energy)

    @pytest.fixture
    def counted_fetch(self, monkeypatch, mock_photometry_data):
        """Patch the Alyx photometry fetch; returns the list of trips made."""
        from iblphotometry.fpio import PhotometrySessionLoader
        fetches = []

        def fake_fetch(session, **kwargs):
            fetches.append(kwargs)
            session.photometry.update(mock_photometry_data)

        monkeypatch.setattr(PhotometrySessionLoader, 'load_photometry',
                            fake_fetch)
        return fetches

    def test_load_photometry_reads_stored_raw_without_fetching(
            self, raw_session, monkeypatch, mock_session_series, counted_fetch):
        """Stored raw is enough to rebuild the preprocessed signal offline."""
        from iblnm.data import PhotometrySession
        monkeypatch.setattr('iblnm.data.store_raw', True)
        raw_session.save_h5(groups=['photometry'])

        fresh = PhotometrySession(mock_session_series, one=MagicMock(),
                                  load_data=False)
        fresh.filepath = raw_session.filepath
        signal = fresh.load_photometry()

        assert counted_fetch == []
        assert list(signal.columns) == ['VTA']

    def test_load_photometry_fetches_when_raw_not_stored(
            self, raw_session, mock_session_series, counted_fetch):
        """With the raw dropped, the same rebuild goes back to Alyx."""
        from iblnm.data import PhotometrySession
        raw_session.save_h5(groups=['photometry'])

        fresh = PhotometrySession(mock_session_series, one=MagicMock(),
                                  load_data=False)
        fresh.filepath = raw_session.filepath
        fresh.load_photometry()

        assert len(counted_fetch) == 1


class TestToDict:
    """Tests for PhotometrySession.to_dict and to_series."""

    def test_to_dict_includes_all_metadata(self, full_session_series):
        """to_dict includes all metadata fields."""
        from iblnm.data import PhotometrySession
        mock_one = MagicMock()
        ps = PhotometrySession(full_session_series, one=mock_one, load_data=False)
        d = ps.to_dict()

        assert d['eid'] == 'test-eid-full'
        assert d['subject'] == 'test_mouse'
        assert d['strain'] == 'Thy1-GCaMP6s'
        assert d['line'] == 'Thy1'
        assert d['genotype'] == ['Thy1-GCaMP6s/wt']
        assert d['NM'] == 'DA'
        assert d['brain_region'] == ['VTA', 'SNc']
        assert d['hemisphere'] == ['l', 'r']
        assert d['target_NM'] == ['VTA-DA', 'SNc-DA']
        assert d['users'] == ['user1', 'user2']
        assert d['session_type'] == 'biased'
        assert d['datasets'] == ['_ibl_trials.table.pqt']
        assert d['session_length'] == 3600

    def test_to_series_roundtrip(self, full_session_series):
        """to_series produces a Series that can reconstruct the session."""
        from iblnm.data import PhotometrySession
        mock_one = MagicMock()
        ps1 = PhotometrySession(full_session_series, one=mock_one, load_data=False)
        s = ps1.to_series()
        ps2 = PhotometrySession(s, one=mock_one, load_data=False)

        assert ps2.eid == ps1.eid
        assert ps2.strain == ps1.strain
        assert ps2.brain_region == ps1.brain_region
        assert ps2.target_NM == ps1.target_NM
        assert ps2.session_type == ps1.session_type


# =============================================================================
# H5 Metadata and Error Persistence Tests
# =============================================================================

class TestH5Metadata:
    """Tests for metadata save/load in H5."""

    def test_save_load_default_to_filepath(self, full_session_series, tmp_path):
        """save_h5/load_h5 with no fpath use self.filepath."""
        from iblnm.data import PhotometrySession
        mock_one = MagicMock()
        ps = PhotometrySession(full_session_series, one=mock_one, load_data=False)
        ps.filepath = tmp_path / f'{ps.eid}.h5'

        ps.save_h5(groups=['metadata'])
        ps2 = PhotometrySession(full_session_series, one=mock_one, load_data=False)
        ps2.filepath = ps.filepath
        ps2.strain = None
        ps2.load_h5(groups=['metadata'])

        assert ps2.strain == ps.strain

    def test_save_load_metadata_roundtrip(self, full_session_series, tmp_path):
        """Metadata survives H5 roundtrip with all field types."""
        from iblnm.data import PhotometrySession
        mock_one = MagicMock()
        ps = PhotometrySession(full_session_series, one=mock_one, load_data=False)
        fpath = tmp_path / f'{ps.eid}.h5'

        ps.save_h5(fpath, groups=['metadata'])
        ps2 = PhotometrySession(full_session_series, one=mock_one, load_data=False)
        # Clear fields that should be restored from H5
        ps2.strain = None
        ps2.brain_region = []
        ps2.load_h5(fpath, groups=['metadata'])

        assert ps2.eid == ps.eid
        assert ps2.subject == ps.subject
        assert ps2.strain == 'Thy1-GCaMP6s'
        assert ps2.line == 'Thy1'
        assert ps2.genotype == ['Thy1-GCaMP6s/wt']
        assert ps2.NM == 'DA'
        assert ps2.brain_region == ['VTA', 'SNc']
        assert ps2.hemisphere == ['l', 'r']
        assert ps2.target_NM == ['VTA-DA', 'SNc-DA']
        assert ps2.users == ['user1', 'user2']
        assert ps2.session_type == 'biased'
        assert ps2.session_length == 3600
        assert ps2.day_n == 5
        assert ps2.number == 1

    def test_save_metadata_creates_metadata_group(self, full_session_series, tmp_path):
        """H5 file contains a /metadata group."""
        import h5py
        from iblnm.data import PhotometrySession
        mock_one = MagicMock()
        ps = PhotometrySession(full_session_series, one=mock_one, load_data=False)
        fpath = tmp_path / f'{ps.eid}.h5'

        ps.save_h5(fpath, groups=['metadata'])
        with h5py.File(fpath, 'r') as f:
            assert 'metadata' in f
            assert f['metadata'].attrs['eid'] == 'test-eid-full'
            assert f['metadata'].attrs['strain'] == 'Thy1-GCaMP6s'
            # List fields stored as datasets
            assert list(f['metadata']['brain_region'][:]) == [b'VTA', b'SNc']

    def test_load_metadata_backward_compat(self, mock_session_series, tmp_path):
        """load_h5 with groups=['metadata'] on old H5 (no /metadata) is a no-op."""
        import h5py
        from iblnm.data import PhotometrySession
        mock_one = MagicMock()
        fpath = tmp_path / 'old.h5'
        # Create old-style H5 with root attrs only
        with h5py.File(fpath, 'w') as f:
            f.attrs['eid'] = 'test-eid-123'
        ps = PhotometrySession(mock_session_series, one=mock_one, load_data=False)
        ps.load_h5(fpath, groups=['metadata'])  # should not raise
        assert ps.eid == 'test-eid-123'

    def test_metadata_none_values_handled(self, minimal_session_series, tmp_path):
        """None scalar values are stored and restored correctly."""
        from iblnm.data import PhotometrySession
        mock_one = MagicMock()
        ps = PhotometrySession(minimal_session_series, one=mock_one, load_data=False)
        fpath = tmp_path / f'{ps.eid}.h5'

        ps.save_h5(fpath, groups=['metadata'])
        ps2 = PhotometrySession(minimal_session_series, one=mock_one, load_data=False)
        ps2.strain = 'should_be_overwritten'
        ps2.load_h5(fpath, groups=['metadata'])
        assert ps2.strain is None
        assert ps2.brain_region == []

    def test_genotype_list_roundtrip(self, tmp_path):
        """genotype (a list from Alyx) survives H5 save/load and from_h5."""
        from iblnm.data import PhotometrySession
        mock_one = MagicMock()
        row = pd.Series({
            'eid': 'gt-test', 'subject': 'ZFM-01',
            'start_time': '2024-01-01T10:00:00', 'number': 1,
            'genotype': ['ChAT-IRES-Cre +/-', 'Ai148-G6f +/-'],
        })
        ps = PhotometrySession(row, one=mock_one, load_data=False)
        assert ps.genotype == ['ChAT-IRES-Cre +/-', 'Ai148-G6f +/-']

        fpath = tmp_path / 'gt-test.h5'
        ps.save_h5(fpath, groups=['metadata'])

        # load_h5 path
        ps2 = PhotometrySession(row, one=mock_one, load_data=False)
        ps2.genotype = []
        ps2.load_h5(fpath, groups=['metadata'])
        assert ps2.genotype == ['ChAT-IRES-Cre +/-', 'Ai148-G6f +/-']

        # from_h5 path (no ONE)
        ps3 = PhotometrySession.from_h5(fpath)
        assert ps3.genotype == ['ChAT-IRES-Cre +/-', 'Ai148-G6f +/-']

    def test_genotype_empty_list_roundtrip(self, tmp_path):
        """Empty genotype list survives H5 roundtrip."""
        from iblnm.data import PhotometrySession
        mock_one = MagicMock()
        row = pd.Series({
            'eid': 'gt-empty', 'subject': 'ZFM-01',
            'start_time': '2024-01-01T10:00:00', 'number': 1,
            'genotype': [],
        })
        ps = PhotometrySession(row, one=mock_one, load_data=False)
        assert ps.genotype == []

        fpath = tmp_path / 'gt-empty.h5'
        ps.save_h5(fpath, groups=['metadata'])

        ps2 = PhotometrySession.from_h5(fpath)
        assert ps2.genotype == []


class TestLogError:
    """Tests for PhotometrySession.log_error."""

    def test_log_error_accumulates(self, mock_session_series):
        """Multiple errors are accumulated in order."""
        from iblnm.data import PhotometrySession
        from iblnm.validation import InvalidStrain, InvalidLine
        mock_one = MagicMock()
        ps = PhotometrySession(mock_session_series, one=mock_one, load_data=False)

        try:
            raise InvalidStrain("bad strain")
        except InvalidStrain as e:
            ps.log_error(e)
        try:
            raise InvalidLine("bad line")
        except InvalidLine as e:
            ps.log_error(e)

        assert len(ps.errors) == 2
        assert ps.errors[0]['error_type'] == 'InvalidStrain'
        assert ps.errors[1]['error_type'] == 'InvalidLine'
        assert ps.errors[0]['eid'] == 'test-eid-123'

    def test_log_error_records_product(self, mock_session_series):
        """The product being built is carried on the entry; default is None."""
        from iblnm.data import PhotometrySession
        mock_one = MagicMock()
        ps = PhotometrySession(mock_session_series, one=mock_one, load_data=False)

        try:
            raise ValueError("pose failed")
        except ValueError as e:
            ps.log_error(e, product='video/pose')
        try:
            raise ValueError("no product")
        except ValueError as e:
            ps.log_error(e)

        assert ps.errors[0]['product'] == 'video/pose'
        assert ps.errors[1]['product'] is None

    def test_log_error_preserves_traceback(self, mock_session_series):
        """Traceback string is captured."""
        from iblnm.data import PhotometrySession
        mock_one = MagicMock()
        ps = PhotometrySession(mock_session_series, one=mock_one, load_data=False)

        try:
            raise ValueError("test error")
        except ValueError as e:
            ps.log_error(e)

        assert ps.errors[0]['traceback'] is not None
        assert 'ValueError' in ps.errors[0]['traceback']


class TestH5Errors:
    """Tests for error save/load in H5."""

    def test_save_load_errors_roundtrip(self, mock_session_series, tmp_path):
        """Errors survive H5 roundtrip."""
        from iblnm.data import PhotometrySession
        from iblnm.validation import InvalidStrain
        mock_one = MagicMock()
        ps = PhotometrySession(mock_session_series, one=mock_one, load_data=False)
        fpath = tmp_path / f'{ps.eid}.h5'

        try:
            raise InvalidStrain("bad strain")
        except InvalidStrain as e:
            ps.log_error(e)

        ps.save_h5(fpath, groups=['metadata', 'errors'])

        ps2 = PhotometrySession(mock_session_series, one=mock_one, load_data=False)
        ps2.load_h5(fpath, groups=['errors'])
        assert len(ps2.errors) == 1
        assert ps2.errors[0]['error_type'] == 'InvalidStrain'
        assert ps2.errors[0]['error_message'] == 'bad strain'
        assert 'InvalidStrain' in ps2.errors[0]['traceback']

    def test_errors_sort_into_their_product_groups(self, mock_session_series,
                                                   tmp_path):
        """Each entry lands under errors/{product} with all five fields intact."""
        import h5py
        from iblnm.data import PhotometrySession
        from iblnm.validation import MissingRawData, MissingLP
        mock_one = MagicMock()
        ps = PhotometrySession(mock_session_series, one=mock_one, load_data=False)
        fpath = tmp_path / f'{ps.eid}.h5'

        try:
            raise MissingRawData("no photometry")
        except MissingRawData as e:
            ps.log_error(e, product='photometry/raw')
        try:
            raise MissingLP("no pose")
        except MissingLP as e:
            ps.log_error(e, product='video/pose')

        ps.save_h5(fpath, groups=['metadata', 'errors'])

        with h5py.File(fpath, 'r') as f:
            assert f['errors/photometry/raw/error_type'][0].decode() \
                == 'MissingRawData'
            assert f['errors/video/pose/error_type'][0].decode() \
                == 'MissingLP'

        ps2 = PhotometrySession(mock_session_series, one=mock_one, load_data=False)
        ps2.load_h5(fpath, groups=['errors'])
        by_product = {e['product']: e for e in ps2.errors}
        assert set(by_product) == {'photometry/raw', 'video/pose'}
        entry = by_product['photometry/raw']
        assert entry['eid'] == ps.eid
        assert entry['error_type'] == 'MissingRawData'
        assert entry['error_message'] == 'no photometry'
        assert 'MissingRawData' in entry['traceback']

    def test_save_errors_empty_list(self, mock_session_series, tmp_path):
        """Saving with no errors creates empty /errors group."""
        import h5py
        from iblnm.data import PhotometrySession
        mock_one = MagicMock()
        ps = PhotometrySession(mock_session_series, one=mock_one, load_data=False)
        fpath = tmp_path / f'{ps.eid}.h5'

        ps.save_h5(fpath, groups=['metadata', 'errors'])
        with h5py.File(fpath, 'r') as f:
            assert 'errors' in f

        ps2 = PhotometrySession(mock_session_series, one=mock_one, load_data=False)
        ps2.load_h5(fpath, groups=['errors'])
        assert ps2.errors == []

    def test_save_errors_append_mode(self, mock_session_series, tmp_path):
        """Errors can be saved in append mode to existing H5."""
        from iblnm.data import PhotometrySession
        mock_one = MagicMock()
        ps = PhotometrySession(mock_session_series, one=mock_one, load_data=False)
        fpath = tmp_path / f'{ps.eid}.h5'

        # First write metadata
        ps.save_h5(fpath, groups=['metadata'])
        # Then append errors
        try:
            raise ValueError("test")
        except ValueError as e:
            ps.log_error(e)
        ps.save_h5(fpath, groups=['errors'], mode='a')

        ps2 = PhotometrySession(mock_session_series, one=mock_one, load_data=False)
        ps2.load_h5(fpath, groups=['metadata', 'errors'])
        assert ps2.strain is None  # metadata loaded
        assert len(ps2.errors) == 1  # errors loaded

    def test_rebuild_replaces_that_products_group_only(self, mock_session_series,
                                                       tmp_path):
        """Last attempt wins for the rebuilt product; other products survive."""
        from iblnm.data import PhotometrySession
        from iblnm.validation import MissingRawData, MissingLP
        mock_one = MagicMock()
        fpath = tmp_path / f"{mock_session_series['eid']}.h5"

        # First pass: photometry/raw and video/pose both fail.
        ps1 = PhotometrySession(mock_session_series, one=mock_one, load_data=False)
        try:
            raise MissingRawData("first attempt")
        except MissingRawData as e:
            ps1.log_error(e, product='photometry/raw')
        try:
            raise MissingLP("no pose")
        except MissingLP as e:
            ps1.log_error(e, product='video/pose')
        ps1.save_h5(fpath, groups=['metadata', 'errors'], mode='w')

        # Second pass: only photometry/raw is retried, and fails differently.
        ps2 = PhotometrySession(mock_session_series, one=mock_one, load_data=False)
        try:
            raise MissingRawData("second attempt")
        except MissingRawData as e:
            ps2.log_error(e, product='photometry/raw')
        ps2.save_h5(fpath, groups=['errors'], mode='a')

        ps3 = PhotometrySession(mock_session_series, one=mock_one, load_data=False)
        ps3.load_h5(fpath, groups=['errors'])
        by_product = {e['product']: e for e in ps3.errors}
        assert len(ps3.errors) == 2
        assert by_product['photometry/raw']['error_message'] == 'second attempt'
        assert by_product['video/pose']['error_message'] == 'no pose'

    def test_error_without_data_group_roundtrips(self, mock_session_series,
                                                 tmp_path):
        """A failed build leaves an error group and no data group."""
        import h5py
        from iblnm.data import PhotometrySession
        from iblnm.validation import MissingLP
        mock_one = MagicMock()
        ps = PhotometrySession(mock_session_series, one=mock_one, load_data=False)
        fpath = tmp_path / f'{ps.eid}.h5'

        try:
            raise MissingLP("no pose")
        except MissingLP as e:
            ps.log_error(e, product='video/pose')
        ps.save_h5(fpath, groups=['metadata', 'errors'])

        with h5py.File(fpath, 'r') as f:
            assert 'errors/video/pose' in f
            assert 'video' not in f

        ps2 = PhotometrySession(mock_session_series, one=mock_one, load_data=False)
        ps2.load_h5(fpath, groups=['errors'])
        assert [e['product'] for e in ps2.errors] == ['video/pose']

    def test_reads_group_written_before_product_field(self, mock_session_series,
                                                      tmp_path):
        """Groups in the store predating the product field still load."""
        import h5py
        from iblnm.data import PhotometrySession
        mock_one = MagicMock()
        fpath = tmp_path / f"{mock_session_series['eid']}.h5"

        with h5py.File(fpath, 'w') as f:
            grp = f.create_group('errors')
            for col, value in [('eid', 'test-eid-123'),
                               ('error_type', 'MissingRawData'),
                               ('error_message', 'no raw data'),
                               ('traceback', '')]:
                grp.create_dataset(col, data=[value],
                                   dtype=h5py.string_dtype())

        ps = PhotometrySession(mock_session_series, one=mock_one, load_data=False)
        ps.load_h5(fpath, groups=['errors'])
        assert len(ps.errors) == 1
        assert ps.errors[0]['error_type'] == 'MissingRawData'
        assert ps.errors[0]['product'] is None

class TestFromAlyx:
    """Tests for PhotometrySession.from_alyx instance method."""

    def _setup_mock_one(self, mock_one):
        """Configure mock ONE to return valid session data."""
        # get_session_dict: sessions/read
        mock_one.alyx.rest.return_value = {
            'users': ['alice'],
            'lab': 'cortexlab',
            'end_time': '2024-01-01T11:00:00',
            'data_dataset_session_related': [
                {'name': '_ibl_trials.table.pqt'},
            ],
        }
        # get_subject_info: subjects/list
        def rest_side_effect(endpoint, action, **kwargs):
            if endpoint == 'subjects':
                return [{
                    'strain': 'Ai148xDATCre',
                    'line': 'Ai148xDat',
                    'genotype': 'Ai148xDATCre/wt',
                }]
            if endpoint == 'sessions' and action == 'read':
                return {
                    'users': ['alice'], 'lab': 'cortexlab',
                    'end_time': '2024-01-01T11:00:00',
                    'data_dataset_session_related': [
                        {'name': '_ibl_trials.table.pqt'},
                    ],
                }
            return []
        mock_one.alyx.rest.side_effect = rest_side_effect
        # get_brain_region: load_dataset
        mock_one.load_dataset.return_value = {
            'devices': {'neurophotometrics': {'fibers': {
                'G0': {'location': 'VTA-l'},
            }}}
        }
        # get_datasets: list_datasets
        mock_one.list_datasets.return_value = [
            '_ibl_trials.table.pqt',
            '_iblrig_taskData.raw.jsonable',
        ]

    def test_from_alyx_populates_metadata(self, mock_session_series):
        """from_alyx enriches session with Alyx data."""
        from iblnm.data import PhotometrySession
        mock_one = MagicMock()
        self._setup_mock_one(mock_one)
        ps = PhotometrySession(mock_session_series, one=mock_one, load_data=False)
        ps.from_alyx()

        assert ps.strain == 'Ai148xDATCre'
        assert ps.line == 'Ai148xDat'
        assert ps.genotype == ['Ai148xDATCre/wt']
        assert ps.NM == 'DA'
        assert ps.lab == 'cortexlab'
        assert ps.brain_region == ['VTA-l']

    def test_from_alyx_logs_validation_errors(self, mock_session_series):
        """Validation failures are logged, not raised."""
        from iblnm.data import PhotometrySession
        mock_one = MagicMock()
        self._setup_mock_one(mock_one)
        # Make subject info return unknown strain
        def rest_side_effect(endpoint, action, **kwargs):
            if endpoint == 'subjects':
                return [{
                    'strain': 'UNKNOWN_STRAIN',
                    'line': 'UNKNOWN_LINE',
                    'genotype': 'xx',
                }]
            if endpoint == 'sessions' and action == 'read':
                return {
                    'users': ['alice'], 'lab': 'cortexlab',
                    'end_time': '2024-01-01T11:00:00',
                    'data_dataset_session_related': [],
                }
            return []
        mock_one.alyx.rest.side_effect = rest_side_effect

        ps = PhotometrySession(mock_session_series, one=mock_one, load_data=False)
        ps.from_alyx()

        # Errors logged but not raised
        error_types = [e['error_type'] for e in ps.errors]
        assert 'InvalidStrain' in error_types
        assert 'InvalidLine' in error_types

    def test_from_alyx_returns_self(self, mock_session_series):
        """from_alyx returns self for chaining."""
        from iblnm.data import PhotometrySession
        mock_one = MagicMock()
        self._setup_mock_one(mock_one)
        ps = PhotometrySession(mock_session_series, one=mock_one, load_data=False)
        result = ps.from_alyx()
        assert result is ps


class TestFromH5:
    """Tests for PhotometrySession.from_h5 classmethod."""

    def test_from_h5_restores_metadata(self, full_session_series, tmp_path):
        """from_h5 creates session with correct metadata."""
        from iblnm.data import PhotometrySession
        mock_one = MagicMock()
        ps = PhotometrySession(full_session_series, one=mock_one, load_data=False)
        fpath = tmp_path / f'{ps.eid}.h5'
        ps.save_h5(fpath, groups=['metadata'])

        ps2 = PhotometrySession.from_h5(fpath)
        assert ps2.eid == 'test-eid-full'
        assert ps2.subject == 'test_mouse'
        assert ps2.strain == 'Thy1-GCaMP6s'
        assert ps2.brain_region == ['VTA', 'SNc']
        assert ps2.target_NM == ['VTA-DA', 'SNc-DA']
        assert ps2.session_type == 'biased'
        assert ps2.number == 1

    def test_from_h5_sets_filepath(self, full_session_series, tmp_path):
        """from_h5 sets filepath to the file it loaded from."""
        from iblnm.data import PhotometrySession
        mock_one = MagicMock()
        ps = PhotometrySession(full_session_series, one=mock_one, load_data=False)
        fpath = tmp_path / f'{ps.eid}.h5'
        ps.save_h5(fpath, groups=['metadata'])

        ps2 = PhotometrySession.from_h5(fpath)
        assert ps2.filepath == fpath

    def test_from_h5_restores_errors(self, mock_session_series, tmp_path):
        """from_h5 loads errors from H5."""
        from iblnm.data import PhotometrySession
        mock_one = MagicMock()
        ps = PhotometrySession(mock_session_series, one=mock_one, load_data=False)
        fpath = tmp_path / f'{ps.eid}.h5'
        try:
            raise ValueError("test error")
        except ValueError as e:
            ps.log_error(e)
        ps.save_h5(fpath, groups=['metadata', 'errors'])

        ps2 = PhotometrySession.from_h5(fpath)
        assert len(ps2.errors) == 1
        assert ps2.errors[0]['error_type'] == 'ValueError'

    def test_from_h5_with_one(self, full_session_series, tmp_path):
        """from_h5 accepts an optional ONE connection."""
        from iblnm.data import PhotometrySession
        mock_one = MagicMock()
        ps = PhotometrySession(full_session_series, one=mock_one, load_data=False)
        fpath = tmp_path / f'{ps.eid}.h5'
        ps.save_h5(fpath, groups=['metadata'])

        ps2 = PhotometrySession.from_h5(fpath, one=mock_one)
        assert ps2.one is mock_one

    @pytest.mark.parametrize('with_one', [False, True])
    def test_from_h5_restores_same_metadata_with_or_without_one(
            self, full_session_series, tmp_path, with_one):
        """Metadata read back from H5 matches what was saved, ONE or not."""
        from iblnm.data import PhotometrySession
        ps = PhotometrySession(full_session_series, one=MagicMock(), load_data=False)
        fpath = tmp_path / f'{ps.eid}.h5'
        ps.save_h5(fpath, groups=['metadata'])

        ps2 = PhotometrySession.from_h5(fpath, one=MagicMock() if with_one else None)
        assert ps2.to_dict() == ps.to_dict()

    @pytest.mark.parametrize('with_one', [False, True])
    def test_from_h5_leaves_unstored_products_off_the_session(
            self, full_session_series, tmp_path, with_one):
        """A metadata-only file yields a session carrying no data attributes.

        The load handlers read a mapping per label, and an empty one means the
        file held no such product. Assigning it anyway would leave the session
        holding an empty product its load method would then never build.
        """
        from iblnm.data import PhotometrySession
        ps = PhotometrySession(full_session_series, one=MagicMock(), load_data=False)
        fpath = tmp_path / f'{ps.eid}.h5'
        ps.save_h5(fpath, groups=['metadata'])

        ps2 = PhotometrySession.from_h5(fpath, one=MagicMock() if with_one else None)
        for attribute in ('trials', 'performance', 'photometry_responses',
                          'photometry_qc', 'wheel_responses',
                          'movement_responses'):
            assert not hasattr(ps2, attribute), attribute


# =============================================================================
# Load Method Tests
# =============================================================================

class TestLoadTrials:
    """Tests for PhotometrySession.load_trials."""

    def test_reads_the_stored_table_without_fetching(self, mock_session_series,
                                                     tmp_path):
        """A stored `trials/table` is read off the H5, not refetched from Alyx.

        The load tiers are memory, then the store, then Alyx. Trials are stored
        like every other product, so a session holding one must not go to the
        network for it.
        """
        from iblnm.data import PhotometrySession

        ps = PhotometrySession(mock_session_series, one=MagicMock(),
                               load_data=False)
        ps.filepath = tmp_path / 'session.h5'
        ps.trials = pd.DataFrame({
            'trial': [0, 1],
            'choice': [1, -1],
            'feedbackType': [1, -1],
        })
        ps.save_h5(groups=['trials'])
        del ps.trials

        with patch.object(PhotometrySession, 'fetch_trials',
                          side_effect=AssertionError('fetched from Alyx')):
            trials = ps.load_trials()

        assert list(trials['trial']) == [0, 1]
        assert list(trials['choice']) == [1, -1]
        assert trials is ps.trials

    def test_propagates_exception(self, mock_session_series):
        """load_trials should let exceptions propagate."""
        from iblnm.data import PhotometrySession

        mock_one = MagicMock()
        session = PhotometrySession(mock_session_series, one=mock_one, load_data=False)

        with patch.object(
            PhotometrySession.__bases__[0], 'load_trials',
            side_effect=Exception("ALF object not found")
        ):
            with pytest.raises(Exception, match="ALF object not found"):
                session.load_trials()


class TestFetchTier:
    """The `fetch_*` methods: Alyx in, session attribute out, nothing else."""

    def test_fetch_trials_adds_the_derived_columns(self, mock_session_series):
        """`fetch_trials` is the whole trials fetch, derived columns included."""
        from iblnm.data import PhotometrySession
        session = PhotometrySession(mock_session_series, one=MagicMock(),
                                    load_data=False)
        table = pd.DataFrame({
            'contrastLeft': [0.25, np.nan],
            'contrastRight': [np.nan, 1.0],
            'feedbackType': [1, -1],
        })

        def _populate(*args, **kwargs):
            session.trials = table

        with patch.object(PhotometrySession.__bases__[0], 'load_trials',
                          side_effect=_populate):
            trials = session.fetch_trials()

        assert {'trial', 'stim_side', 'signed_contrast', 'contrast'} <= set(
            trials.columns)
        assert trials is session.trials

    def test_fetch_photometry_refetches_over_a_stored_product(
            self, mock_session_series, mock_photometry_data, tmp_path,
            monkeypatch):
        """A fetch never consults the store, however current the store is."""
        from iblnm.data import PhotometrySession
        monkeypatch.setattr('iblnm.data.store_raw', True)
        session = PhotometrySession(mock_session_series, one=MagicMock(),
                                    load_data=False)
        session.filepath = tmp_path / f'{session.eid}.h5'

        def _populate(*args, **kwargs):
            session.photometry = dict(mock_photometry_data)

        with patch.object(PhotometrySession.__bases__[0], 'load_photometry',
                          side_effect=_populate) as fetch:
            session.load_raw_photometry()
            session.save_h5(groups=['photometry'])
            assert session.stored_product_exists('photometry/raw')
            session.load_raw_photometry()
            assert fetch.call_count == 1
            session.fetch_photometry()
            assert fetch.call_count == 2

    def test_fetch_neurophotometrics_assigns_the_source_table(
            self, mock_session_series):
        """The source table lands on the session, not just in the QC scorer."""
        from iblnm.data import PhotometrySession
        session = PhotometrySession(mock_session_series, one=MagicMock(),
                                    load_data=False)
        raw = pd.DataFrame({'times': [0.0, 1.0], 'signal': [1.0, 2.0]})
        with patch('iblnm.data.from_neurophotometrics_df_to_photometry_df',
                   return_value=raw) as convert:
            table = session.fetch_neurophotometrics()

        convert.assert_called_once()
        assert table is session.neurophotometrics
        assert table.index.name == 'times'


class TestLoadRawPhotometry:
    """Tests for PhotometrySession.load_raw_photometry — the Alyx fetch."""

    def test_populates_raw_bands(self, mock_session_series, mock_photometry_data):
        """The fetched bands land in self.photometry under their band names."""
        from iblnm.data import PhotometrySession

        session = PhotometrySession(mock_session_series, one=MagicMock(),
                                    load_data=False)

        def _populate(*args, **kwargs):
            session.photometry = mock_photometry_data

        with patch.object(PhotometrySession.__bases__[0], 'load_photometry',
                          side_effect=_populate):
            session.load_raw_photometry()

        assert set(session.photometry) == {'GCaMP', 'Isosbestic'}

    def test_propagates_exception(self, mock_session_series):
        """load_raw_photometry should let exceptions propagate."""
        from iblnm.data import PhotometrySession

        mock_one = MagicMock()
        session = PhotometrySession(mock_session_series, one=mock_one, load_data=False)

        with patch.object(
            PhotometrySession.__bases__[0], 'load_photometry',
            side_effect=Exception("No photometry data")
        ):
            with pytest.raises(Exception, match="No photometry data"):
                session.load_raw_photometry()

    def test_no_flat_aliases(self, mock_photometry_session):
        """load_raw_photometry should not create self.channels or self.targets."""
        session = mock_photometry_session
        assert not hasattr(session, 'channels')
        assert not hasattr(session, 'targets')


class TestLoadPhotometry:
    """`load_photometry` returns the preprocessed product, building if absent.

    `load_raw_photometry` is the session's only route to Alyx, so patching it
    both supplies the raw bands and counts the trips a real run would make.
    """

    @pytest.fixture
    def fetching_session(self, mock_session_series, mock_photometry_data, tmp_path):
        """(session, fetch) where `fetch` is the patched Alyx call, un-started."""
        from iblnm.data import PhotometrySession
        session = PhotometrySession(mock_session_series, one=MagicMock(),
                                    load_data=False)
        session.filepath = tmp_path / f'{session.eid}.h5'

        def _populate():
            session.photometry.update(mock_photometry_data)

        return session, patch.object(PhotometrySession, 'load_raw_photometry',
                                     side_effect=_populate)

    def test_builds_and_stores_when_absent(self, fetching_session):
        """With nothing stored, the raw signal is fetched once and written."""
        session, fetch = fetching_session
        with fetch as fetch_mock:
            signal = session.load_photometry()

        assert fetch_mock.call_count == 1
        assert list(signal.columns) == ['VTA']
        assert session.photometry['GCaMP_preprocessed'] is signal
        assert session.stored_product_exists('photometry/preprocessed')

    def test_saves_what_the_extraction_returned(self, fetching_session,
                                                mock_photometry_session):
        """The load is what writes: same signal as the bare extraction, stored."""
        import h5py
        session, fetch = fetching_session
        with fetch:
            loaded = session.load_photometry()

        extracted = mock_photometry_session.extract_preprocessed_photometry()
        np.testing.assert_allclose(loaded['VTA'].values, extracted['VTA'].values)
        with h5py.File(session.filepath, 'r') as h5:
            assert 'photometry/VTA/preprocessed' in h5

    def test_writes_the_diagnostics_as_attrs(self, fetching_session):
        """The diagnostics ride as attrs on the preprocessed group it wrote.

        They describe the preprocessing run, not the raw signal, so they hang
        off `preprocessed/` and leave `qc/` depending only on `raw/`.
        """
        import h5py
        from iblnm.data import _load_scalars
        session, fetch = fetching_session
        with fetch:
            session.load_photometry()

        with h5py.File(session.filepath, 'r') as h5:
            stored = _load_scalars(h5['photometry/VTA/preprocessed'])
        assert set(stored) == {'bleaching_tau', 'iso_correlation'}
        assert stored == pytest.approx(session.preprocessing_diagnostics['VTA'])

    def test_reads_stored_product_without_fetching(self, fetching_session,
                                                   mock_session_series):
        """A second session over the same file reads it and never fetches."""
        from iblnm.data import PhotometrySession
        session, fetch = fetching_session
        with fetch:
            built = session.load_photometry()

        fresh = PhotometrySession(mock_session_series, one=MagicMock(),
                                  load_data=False)
        fresh.filepath = session.filepath
        with patch.object(PhotometrySession, 'load_raw_photometry') as fetch_mock:
            signal = fresh.load_photometry()

        fetch_mock.assert_not_called()
        np.testing.assert_allclose(signal['VTA'].values, built['VTA'].values)
        assert fresh.preprocessing_diagnostics['VTA'] == pytest.approx(
            session.preprocessing_diagnostics['VTA'])

    def test_reads_a_stored_product_built_with_other_parameters(
            self, fetching_session):
        """Parameters differing from config.py do not stop the stored read.

        Files written before this change carry a `spec_json` attr naming the
        parameters that produced them. Nothing compares it against `config.py`
        any more: a stored product is present or absent, so the read returns
        the signal rather than raising.
        """
        import json
        import h5py
        from iblnm.data import PhotometrySession
        session, fetch = fetching_session
        with fetch:
            built = session.load_photometry()
        with h5py.File(session.filepath, 'a') as h5:
            h5['photometry/VTA/preprocessed'].attrs['spec_json'] = json.dumps(
                {'fs': 15})

        # A session holding the signal answers from memory, so the stored copy
        # is only met by one that has to read it.
        fresh = PhotometrySession(session.to_dict(), one=MagicMock(),
                                  load_data=False)
        fresh.filepath = session.filepath
        with patch.object(PhotometrySession, 'load_raw_photometry') as fetch_mock:
            signal = fresh.load_photometry()

        fetch_mock.assert_not_called()
        np.testing.assert_allclose(signal['VTA'].values, built['VTA'].values)
        assert fresh.errors == []

    def test_returns_the_held_signal_without_reading_the_file(self,
                                                              fetching_session):
        """A second load answers from memory: the file it wrote is not needed."""
        session, fetch = fetching_session
        with fetch as fetch_mock:
            signal = session.load_photometry()
            session.filepath.unlink()

            assert session.load_photometry() is signal
        assert fetch_mock.call_count == 1

    def test_never_returns_raw(self, fetching_session, mock_photometry_data):
        """Holding the raw bands is not enough: the fetch failure propagates.

        The raw signal is never a stand-in for the preprocessed one — an
        analysis silently running on raw data is the failure this split exists
        to prevent.
        """
        from iblnm.data import PhotometrySession
        from iblnm.validation import MissingExtractedData
        session, _ = fetching_session
        session.photometry.update(mock_photometry_data)

        with patch.object(PhotometrySession, 'load_raw_photometry',
                          side_effect=MissingExtractedData('photometry.signal.pqt')):
            with pytest.raises(MissingExtractedData):
                session.load_photometry()


# =============================================================================
# Validation Tests
# =============================================================================

class TestValidateNTrials:
    """Tests for PhotometrySession.validate_n_trials."""

    def test_raises_when_insufficient(self, mock_session_series):
        from iblnm.data import PhotometrySession
        from iblnm.validation import InsufficientTrials
        session = PhotometrySession(mock_session_series, one=MagicMock(), load_data=False)
        session.trials = pd.DataFrame({'a': range(50)})  # 50 < MIN_NTRIALS (90)
        with pytest.raises(InsufficientTrials, match='n_trials=50'):
            session.validate_n_trials()

    def test_does_not_raise_when_sufficient(self, mock_session_series):
        from iblnm.data import PhotometrySession
        session = PhotometrySession(mock_session_series, one=MagicMock(), load_data=False)
        session.trials = pd.DataFrame({'a': range(200)})
        session.validate_n_trials()  # should not raise


class TestValidateBlockStructure:
    """Tests for PhotometrySession.validate_block_structure."""

    def test_raises_with_corrupted_blocks(self, mock_session_series):
        """Flipping blocks with JSON mismatch raises BlockStructureBug."""
        from iblnm.data import PhotometrySession
        from iblnm.validation import BlockStructureBug
        series = mock_session_series.copy()
        series['session_type'] = 'biased'
        session = PhotometrySession(series, one=MagicMock(), load_data=False)
        session.trials = pd.DataFrame({'probabilityLeft': np.tile([0.8, 0.2], 50)})

        # JSON says blocks should be [50, 50] — doesn't match trials
        rng = np.random.default_rng(0)
        positions = np.empty(100)
        positions[:50] = rng.choice([-35, 35], size=50, p=[0.8, 0.2])
        positions[50:] = rng.choice([-35, 35], size=50, p=[0.2, 0.8])
        session._block_info = {
            'len_blocks': [50, 50],
            'positions': positions,
            'block_probability_set': [0.2, 0.8],
        }
        with pytest.raises(BlockStructureBug):
            session.validate_block_structure()

    def test_does_not_raise_with_valid_blocks(self, mock_session_series):
        from iblnm.data import PhotometrySession
        series = mock_session_series.copy()
        series['session_type'] = 'biased'
        session = PhotometrySession(series, one=MagicMock(), load_data=False)
        session.trials = pd.DataFrame({
            'probabilityLeft': np.concatenate([np.full(100, 0.8), np.full(100, 0.2)]),
        })
        session.validate_block_structure()  # should not raise

    def test_raises_for_training_with_non_uniform_pleft(self, mock_session_series):
        from iblnm.data import PhotometrySession
        from iblnm.validation import BlockStructureBug
        series = mock_session_series.copy()
        series['session_type'] = 'training'
        session = PhotometrySession(series, one=MagicMock(), load_data=False)
        session.trials = pd.DataFrame({'probabilityLeft': np.tile([0.8, 0.2], 50)})
        with pytest.raises(BlockStructureBug, match="Training session"):
            session.validate_block_structure()

    def test_does_not_raise_for_training_with_uniform_pleft(self, mock_session_series):
        from iblnm.data import PhotometrySession
        series = mock_session_series.copy()
        series['session_type'] = 'training'
        session = PhotometrySession(series, one=MagicMock(), load_data=False)
        session.trials = pd.DataFrame({'probabilityLeft': np.full(100, 0.5)})
        session.validate_block_structure()  # should not raise

    def test_missing_block_info_logs_and_raises(self, mock_session_series):
        """When LEN_BLOCKS is None, logs MissingBlockInfo and raises BlockStructureBug."""
        from iblnm.data import PhotometrySession
        from iblnm.validation import BlockStructureBug
        series = mock_session_series.copy()
        series['session_type'] = 'biased'
        session = PhotometrySession(series, one=MagicMock(), load_data=False)
        session.trials = pd.DataFrame({'probabilityLeft': np.tile([0.8, 0.2], 50)})
        session._block_info = {
            'len_blocks': None, 'positions': None, 'block_probability_set': None,
        }
        with pytest.raises(BlockStructureBug):
            session.validate_block_structure()
        error_types = [e['error_type'] for e in session.errors]
        assert 'MissingBlockInfo' in error_types

    def test_json_match_no_raise(self, mock_session_series):
        """Short last block that matches JSON should not raise."""
        from iblnm.data import PhotometrySession
        series = mock_session_series.copy()
        series['session_type'] = 'biased'
        session = PhotometrySession(series, one=MagicMock(), load_data=False)
        # Short last block — cheap check flags, but JSON matches
        prob_left = np.concatenate([
            np.full(50, 0.5), np.full(30, 0.8), np.full(3, 0.2),
        ])
        session.trials = pd.DataFrame({'probabilityLeft': prob_left})

        rng = np.random.default_rng(0)
        positions = np.empty(100)
        positions[:50] = rng.choice([-35, 35], size=50, p=[0.5, 0.5])
        positions[50:80] = rng.choice([-35, 35], size=30, p=[0.8, 0.2])
        positions[80:] = rng.choice([-35, 35], size=20, p=[0.2, 0.8])
        session._block_info = {
            'len_blocks': [50, 30, 20],
            'positions': positions,
            'block_probability_set': [0.2, 0.8],
        }
        session.validate_block_structure()  # should not raise


class TestFixBlockStructure:
    """Tests for PhotometrySession.fix_block_structure."""

    def test_fixes_corrupted_trials(self, mock_session_series):
        from iblnm.data import PhotometrySession
        series = mock_session_series.copy()
        series['session_type'] = 'biased'
        session = PhotometrySession(series, one=MagicMock(), load_data=False)
        # Corrupted: trial-by-trial flipping
        session.trials = pd.DataFrame({'probabilityLeft': np.tile([0.8, 0.2], 50)})

        rng = np.random.default_rng(0)
        positions = np.empty(100)
        positions[:50] = rng.choice([-35, 35], size=50, p=[0.8, 0.2])
        positions[50:] = rng.choice([-35, 35], size=50, p=[0.2, 0.8])
        session._block_info = {
            'len_blocks': [50, 50],
            'positions': positions,
            'block_probability_set': [0.2, 0.8],
        }
        assert session.fix_block_structure() is True
        np.testing.assert_array_equal(session.trials['probabilityLeft'][:50], 0.8)
        np.testing.assert_array_equal(session.trials['probabilityLeft'][50:], 0.2)

    def test_fixes_training_to_uniform_pleft(self, mock_session_series):
        from iblnm.data import PhotometrySession
        series = mock_session_series.copy()
        series['session_type'] = 'training'
        session = PhotometrySession(series, one=MagicMock(), load_data=False)
        session.trials = pd.DataFrame({'probabilityLeft': np.tile([0.8, 0.2], 50)})
        assert session.fix_block_structure() is True
        np.testing.assert_array_equal(session.trials['probabilityLeft'], 0.5)

    def test_returns_false_without_block_info(self, mock_session_series):
        from iblnm.data import PhotometrySession
        series = mock_session_series.copy()
        series['session_type'] = 'biased'
        session = PhotometrySession(series, one=MagicMock(), load_data=False)
        session.trials = pd.DataFrame({'probabilityLeft': np.tile([0.8, 0.2], 50)})
        session._block_info = {
            'len_blocks': None, 'positions': None, 'block_probability_set': None,
        }
        assert session.fix_block_structure() is False


class TestValidateEventCompleteness:
    """Tests for PhotometrySession.validate_event_completeness."""

    def test_raises_with_incomplete_events(self, mock_session_series):
        from iblnm.data import PhotometrySession
        from iblnm.validation import IncompleteEventTimes
        from iblnm.config import RESPONSE_EVENTS
        session = PhotometrySession(mock_session_series, one=MagicMock(), load_data=False)
        data = {e: np.random.rand(100) for e in RESPONSE_EVENTS}
        data[RESPONSE_EVENTS[0]][:85] = np.nan   # 15% present — below threshold
        data[RESPONSE_EVENTS[1]][:85] = np.nan
        session.trials = pd.DataFrame(data)
        with pytest.raises(IncompleteEventTimes) as exc_info:
            session.validate_event_completeness()
        assert RESPONSE_EVENTS[0] in exc_info.value.missing_events
        assert RESPONSE_EVENTS[1] in exc_info.value.missing_events

    def test_does_not_raise_when_all_complete(self, mock_session_series):
        from iblnm.data import PhotometrySession
        from iblnm.config import RESPONSE_EVENTS
        session = PhotometrySession(mock_session_series, one=MagicMock(), load_data=False)
        session.trials = pd.DataFrame({e: np.random.rand(100) for e in RESPONSE_EVENTS})
        session.validate_event_completeness()  # should not raise

    def test_missing_column_included_in_missing_events(self, mock_session_series):
        from iblnm.data import PhotometrySession
        from iblnm.validation import IncompleteEventTimes
        from iblnm.config import RESPONSE_EVENTS
        session = PhotometrySession(mock_session_series, one=MagicMock(), load_data=False)
        session.trials = pd.DataFrame({'stimOnTrigger_times': np.random.rand(100)})
        with pytest.raises(IncompleteEventTimes) as exc_info:
            session.validate_event_completeness()
        for event in RESPONSE_EVENTS:
            if event != 'stimOnTrigger_times':
                assert event in exc_info.value.missing_events


class TestValidateTrialsInPhotometryTime:
    """Tests for PhotometrySession.validate_trials_in_photometry_time."""

    def test_raises_when_trials_outside(self, mock_session_series, mock_photometry_data):
        from iblnm.data import PhotometrySession
        from iblnm.validation import TrialsNotInPhotometryTime
        session = PhotometrySession(mock_session_series, one=MagicMock(), load_data=False)
        session.photometry = mock_photometry_data
        session.trials = pd.DataFrame({
            'stimOnTrigger_times': [-10.0, 100.0],
            'feedback_times': [100.0, 200.0],
        })
        with pytest.raises(TrialsNotInPhotometryTime):
            session.validate_trials_in_photometry_time()

    def test_does_not_raise_when_trials_inside(self, mock_session_series, mock_photometry_data):
        from iblnm.data import PhotometrySession
        session = PhotometrySession(mock_session_series, one=MagicMock(), load_data=False)
        session.photometry = mock_photometry_data
        session.trials = pd.DataFrame({
            'stimOnTrigger_times': [10.0, 100.0],
            'feedback_times': [100.0, 500.0],
        })
        session.validate_trials_in_photometry_time()  # should not raise

    def test_uses_preprocessed_band_when_no_raw(self, mock_photometry_session):
        """Should fall back to GCaMP_preprocessed when GCaMP is not available."""
        session = mock_photometry_session
        session.extract_preprocessed_photometry()
        del session.photometry['GCaMP']
        del session.photometry['Isosbestic']
        session.trials = pd.DataFrame({
            'stimOnTrigger_times': [10.0, 100.0],
            'feedback_times': [100.0, 500.0],
        })
        session.validate_trials_in_photometry_time()  # should not raise


class TestValidateQc:
    """Tests for PhotometrySession.validate_qc."""

    def test_does_not_raise_when_qc_clean(self, mock_session_series):
        from iblnm.data import PhotometrySession
        session = PhotometrySession(mock_session_series, one=MagicMock(), load_data=False)
        session.neurophotometrics_qc = {'n_band_inversions': 0.0,
                                        'n_early_samples': 0.0}
        session.validate_qc()  # should not raise

    def test_raises_on_band_inversions(self, mock_session_series):
        from iblnm.data import PhotometrySession
        from iblnm.validation import QCValidationError
        session = PhotometrySession(mock_session_series, one=MagicMock(), load_data=False)
        session.neurophotometrics_qc = {'n_band_inversions': 3.0,
                                        'n_early_samples': 0.0}
        with pytest.raises(QCValidationError, match='band inversions'):
            session.validate_qc()

    def test_raises_on_early_samples(self, mock_session_series):
        from iblnm.data import PhotometrySession
        from iblnm.validation import QCValidationError
        session = PhotometrySession(mock_session_series, one=MagicMock(), load_data=False)
        session.neurophotometrics_qc = {'n_band_inversions': 0.0,
                                        'n_early_samples': 5.0}
        with pytest.raises(QCValidationError, match='early samples'):
            session.validate_qc()

    def test_raises_with_both_issues_in_message(self, mock_session_series):
        from iblnm.data import PhotometrySession
        from iblnm.validation import QCValidationError
        session = PhotometrySession(mock_session_series, one=MagicMock(), load_data=False)
        session.neurophotometrics_qc = {'n_band_inversions': 3.0,
                                        'n_early_samples': 5.0}
        with pytest.raises(QCValidationError) as exc_info:
            session.validate_qc()
        msg = str(exc_info.value)
        assert 'band inversions' in msg
        assert 'early samples' in msg

    def test_raises_on_attrs_read_back_from_h5(self, mock_photometry_session):
        """The verdict comes from the stored attrs, not from a live QC run."""
        from iblnm.data import PhotometrySession
        from iblnm.validation import QCValidationError
        session = mock_photometry_session
        session.neurophotometrics_qc = {'n_band_inversions': 2.0}
        session.save_h5(groups=['photometry'])

        reopened = PhotometrySession(session.to_dict(), one=MagicMock(),
                                     load_data=False)
        reopened.load_h5(session.filepath, groups=['photometry'])
        with pytest.raises(QCValidationError, match='band inversions'):
            reopened.validate_qc()

    def test_does_not_raise_when_qc_empty(self, mock_session_series):
        from iblnm.data import PhotometrySession
        session = PhotometrySession(mock_session_series, one=MagicMock(), load_data=False)
        session.validate_qc()  # should not raise


# =============================================================================
# Preprocess Tests
# =============================================================================

class TestExtractPreprocessedPhotometry:
    """Tests for PhotometrySession.extract_preprocessed_photometry."""

    def test_preprocess_adds_new_band(self, mock_photometry_session):
        """Preprocess should add preprocessed signal as new band in photometry dict."""
        session = mock_photometry_session

        session.extract_preprocessed_photometry()

        assert 'GCaMP_preprocessed' in session.photometry
        assert isinstance(session.photometry['GCaMP_preprocessed'], pd.DataFrame)
        assert 'VTA' in session.photometry['GCaMP_preprocessed'].columns

    def test_preprocess_computes_diagnostics(self, mock_photometry_session):
        """Preprocess reports bleaching_tau and iso_correlation per region."""
        session = mock_photometry_session

        session.extract_preprocessed_photometry()

        diagnostics = session.preprocessing_diagnostics['VTA']
        tau = diagnostics['bleaching_tau']
        assert 100 < tau < 600  # Known fixture tau=300, allow wide margin for fit
        assert 0.8 < diagnostics['iso_correlation'] <= 1.0

    def test_preprocess_raises_when_no_photometry(self, mock_session_series):
        """Should raise if photometry not loaded (no explicit guard — natural error)."""
        from iblnm.data import PhotometrySession

        mock_one = MagicMock()
        session = PhotometrySession(mock_session_series, one=mock_one, load_data=False)

        with pytest.raises((AttributeError, KeyError, TypeError)):
            session.extract_preprocessed_photometry()

    def test_preprocess_single_band_pipeline(self, mock_photometry_session):
        """Single-band pipeline should work without reference."""
        from iblphotometry.pipelines import sliding_mad_pipeline

        session = mock_photometry_session

        session.extract_preprocessed_photometry(
            pipeline=sliding_mad_pipeline,
            reference_band=None
        )

        assert 'GCaMP_preprocessed' in session.photometry
        diagnostics = session.preprocessing_diagnostics['VTA']
        assert 'iso_correlation' not in diagnostics
        assert not pd.isna(diagnostics['bleaching_tau'])

    def test_preprocess_raises_when_dual_band_no_reference(self, mock_photometry_session):
        """Should raise ValueError if dual-band pipeline but no reference."""
        from iblphotometry.pipelines import isosbestic_correction_pipeline

        with pytest.raises(ValueError, match="requires reference"):
            mock_photometry_session.extract_preprocessed_photometry(
                pipeline=isosbestic_correction_pipeline,
                reference_band=None
            )

    def test_preprocess_custom_output_band(self, mock_photometry_session):
        """Can specify custom output band name."""
        mock_photometry_session.extract_preprocessed_photometry(
            output_band='corrected')

        assert 'corrected' in mock_photometry_session.photometry
        assert 'GCaMP_preprocessed' not in mock_photometry_session.photometry

    def test_preprocess_resamples_to_target_fs(self, mock_photometry_session):
        """Preprocessed signal should be resampled to TARGET_FS."""
        from iblnm.config import TARGET_FS
        session = mock_photometry_session
        session.extract_preprocessed_photometry()
        signal = session.photometry['GCaMP_preprocessed']['VTA']
        dt = np.diff(signal.index.values)
        np.testing.assert_allclose(dt, 1 / TARGET_FS, atol=1e-10)

    def test_preprocess_zscores_signal(self, mock_photometry_session):
        """Preprocessed signal should be z-scored (mean≈0, std≈1)."""
        session = mock_photometry_session
        session.extract_preprocessed_photometry()
        signal = session.photometry['GCaMP_preprocessed']['VTA'].values
        np.testing.assert_allclose(np.mean(signal), 0, atol=0.01)
        np.testing.assert_allclose(np.std(signal), 1, atol=0.01)

    def test_preprocess_puts_the_signal_on_the_target_grid(self, mock_photometry_session):
        """Resampling still happens; it moved into the pipeline, ahead of the
        z-score, so that the z-score is what the stored signal was last through."""
        from iblnm.config import TARGET_FS
        session = mock_photometry_session
        session.extract_preprocessed_photometry()

        times = session.photometry['GCaMP_preprocessed'].index.values
        np.testing.assert_allclose(np.diff(times), 1 / TARGET_FS)

    def test_preprocess_accepts_regression_method(self, mock_photometry_session):
        """The extraction should accept regression_method without error."""
        mock_photometry_session.extract_preprocessed_photometry(
            regression_method='mse')
        assert 'GCaMP_preprocessed' in mock_photometry_session.photometry

    def test_extract_assigns_the_band_and_writes_nothing(
            self, mock_photometry_session):
        """The computation is pure: it assigns, and the H5 file stays absent."""
        session = mock_photometry_session
        assert not session.filepath.exists()

        signal = session.extract_preprocessed_photometry()

        assert signal is session.photometry['GCaMP_preprocessed']
        assert 'VTA' in signal.columns
        assert not session.filepath.exists()


# =============================================================================
# Extract Responses and Trial Data Tests
# =============================================================================

def _make_trials(n=50, **columns):
    """Trials table spanning the synthetic photometry signal.

    Carries the `trial` identity column that `load_trials` adds, since
    `extract_responses` reads trial identity from it. Extra columns are
    appended as given.
    """
    return pd.DataFrame({
        'trial': np.arange(n),
        'stimOnTrigger_times': np.linspace(99.5, 499.5, n),
        'firstMovement_times': np.linspace(100.3, 500.3, n),
        'feedback_times': np.linspace(101, 501, n),
        **columns,
    })


class TestLoadResponses:
    """`load_responses(modality)` reads the stored matrices or cuts them."""

    @pytest.fixture
    def preprocessed_session(self, mock_photometry_session):
        """Session with a stored preprocessed signal and trials in hand.

        The extraction no longer writes, so the save is explicit here: without
        the stored product `load_responses` would send `load_photometry` back
        to the mocked Alyx.
        """
        session = mock_photometry_session
        session.extract_preprocessed_photometry()
        session.save_h5(groups=['photometry'])
        session.trials = _make_trials()
        return session

    def test_builds_and_stores_when_absent(self, preprocessed_session):
        """Nothing stored: the matrices are cut, assigned, and written."""
        session = preprocessed_session
        assert not session.stored_product_exists('photometry/responses')

        responses = session.load_responses('photometry')

        assert isinstance(responses['VTA'], xr.DataArray)
        assert session.photometry_responses is responses
        assert session.stored_product_exists('photometry/responses')

    def test_roundtrips_a_dataarray_per_region(self, preprocessed_session,
                                               mock_session_series):
        """A fresh session reads back the same matrix, coords included."""
        from iblnm.data import PhotometrySession
        session = preprocessed_session
        built = session.load_responses('photometry')

        fresh = PhotometrySession(mock_session_series, one=MagicMock(),
                                  load_data=False)
        fresh.filepath = session.filepath
        reloaded = fresh.load_responses('photometry')

        assert set(reloaded) == set(built)
        xr.testing.assert_allclose(reloaded['VTA'].sortby('event'),
                                   built['VTA'].sortby('event'))

    def test_returns_the_held_matrices_without_reading_the_file(
            self, preprocessed_session):
        """A second call answers from memory: the stored matrices go unread."""
        session = preprocessed_session
        responses = session.load_responses('photometry')
        session.filepath.unlink()

        assert session.load_responses('photometry') is responses

class TestExtractResponses:
    def test_returns_dict_without_assigning_attribute(self, mock_photometry_session):
        """The engine returns its dict; the caller owns the attribute."""
        import xarray as xr
        session = mock_photometry_session
        session.extract_preprocessed_photometry()
        session.trials = _make_trials()
        responses = session.extract_responses(
            session.photometry['GCaMP_preprocessed']
        )
        assert isinstance(responses, dict)
        assert isinstance(responses['VTA'], xr.DataArray)
        assert not hasattr(session, 'photometry_responses')

    def test_labels_come_from_signals_mapping(self, mock_photometry_session):
        """Any mapping of label -> time-indexed Series is a valid signal source."""
        session = mock_photometry_session
        session.extract_preprocessed_photometry()
        session.trials = _make_trials()
        signal = session.photometry['GCaMP_preprocessed']['VTA']
        responses = session.extract_responses({'paw_speed': signal})
        assert list(responses) == ['paw_speed']
        assert set(responses['paw_speed'].dims) == {'event', 'trial', 'time'}

    def test_returns_xarray_dataarray(self, mock_photometry_session):
        import xarray as xr
        session = mock_photometry_session
        session.extract_preprocessed_photometry()
        session.trials = _make_trials()
        session.photometry_responses = session.extract_responses(
            session.photometry['GCaMP_preprocessed']
        )
        assert isinstance(session.photometry_responses, dict)
        assert isinstance(session.photometry_responses['VTA'], xr.DataArray)

    def test_has_correct_dims(self, mock_photometry_session):
        from iblnm.config import RESPONSE_EVENTS
        session = mock_photometry_session
        session.extract_preprocessed_photometry()
        n = 50
        session.trials = _make_trials(n)
        session.photometry_responses = session.extract_responses(
            session.photometry['GCaMP_preprocessed'])
        assert 'VTA' in session.photometry_responses
        region_responses = session.photometry_responses['VTA']
        assert set(region_responses.dims) == {'event', 'trial', 'time'}
        for event in RESPONSE_EVENTS:
            assert event in region_responses.coords['event'].values
        assert region_responses.sizes['trial'] == n

    def test_sel_region_event(self, mock_photometry_session):
        """Selecting by event returns (trial, time) array."""
        session = mock_photometry_session
        session.extract_preprocessed_photometry()
        n = 50
        session.trials = _make_trials(n)
        session.photometry_responses = session.extract_responses(
            session.photometry['GCaMP_preprocessed'])
        sel = session.photometry_responses['VTA'].sel(event='stimOnTrigger_times')
        assert sel.dims == ('trial', 'time')
        assert sel.shape[0] == n

    def test_time_coord_matches_window(self, mock_photometry_session):
        """Time coordinate should span RESPONSE_WINDOW."""
        from iblnm.config import RESPONSE_WINDOW
        session = mock_photometry_session
        session.extract_preprocessed_photometry()
        session.trials = _make_trials()
        session.photometry_responses = session.extract_responses(
            session.photometry['GCaMP_preprocessed'], events=['feedback_times'])
        tpts = session.photometry_responses['VTA'].coords['time'].values
        assert tpts[0] == pytest.approx(RESPONSE_WINDOW[0], abs=0.05)
        assert tpts[-1] == pytest.approx(RESPONSE_WINDOW[1], abs=0.05)

    def test_custom_events(self, mock_photometry_session):
        session = mock_photometry_session
        session.extract_preprocessed_photometry()
        session.trials = _make_trials()
        session.photometry_responses = session.extract_responses(
            session.photometry['GCaMP_preprocessed'], events=['feedback_times'])
        region_responses = session.photometry_responses['VTA']
        assert list(region_responses.coords['event'].values) == ['feedback_times']
        assert region_responses.sizes['event'] == 1

    def test_per_trial_window_end_named_by_a_trials_column(
            self, mock_photometry_session):
        """A window end given as a column name ends each trial at its own event.

        This is the mechanism behind the wheel's cut: every trial shares one
        time axis spanning to the longest, and is NaN-padded from its own
        endpoint onward.
        """
        session = mock_photometry_session
        session.extract_preprocessed_photometry()
        n = 50
        session.trials = _make_trials(
            n, feedback_times=np.linspace(99.5, 499.5, n) + np.linspace(0.5, 2.5, n))
        responses = session.extract_responses(
            session.photometry['GCaMP_preprocessed'],
            events=['stimOnTrigger_times'], window=(0.0, 'feedback_times'))

        cut = responses['VTA'].sel(event='stimOnTrigger_times')
        tpts = cut.coords['time'].values
        assert tpts[-1] == pytest.approx(2.5, abs=0.1)
        durations = (session.trials['feedback_times']
                     - session.trials['stimOnTrigger_times']).to_numpy()
        for trial, duration in enumerate(durations):
            values = cut.values[trial]
            assert not np.any(np.isnan(values[tpts <= duration]))
            assert np.all(np.isnan(values[tpts > duration]))

    def test_trial_coord_comes_from_trial_column(self, mock_photometry_session):
        """Trial identity is the 'trial' column, not the row position.

        A trials table filtered before extraction keeps its original trial
        numbers, so responses stay aligned to the trials they came from.
        """
        session = mock_photometry_session
        session.extract_preprocessed_photometry()
        trial_numbers = np.array([3, 7, 8, 15, 40])
        session.trials = pd.DataFrame({
            'trial': trial_numbers,
            'feedback_times': np.linspace(101, 501, len(trial_numbers)),
        })
        responses = session.extract_responses(
            session.photometry['GCaMP_preprocessed'], events=['feedback_times'])
        np.testing.assert_array_equal(
            responses['VTA'].coords['trial'].values, trial_numbers)



# =============================================================================
# HDF5 Save/Load Tests
# =============================================================================

class TestWriteReadDataframe:
    def test_string_dtype_column_roundtrips(self, tmp_path):
        """A pandas string-dtype column survives _write_dataframe/_read_dataframe.

        pandas 3.0 makes a dedicated ``string`` dtype the default for text
        columns, whose ``.values`` is a StringArray that is not numpy ``object``.
        The writer must still encode it to bytes rather than handing h5py a
        non-native dtype.
        """
        import h5py
        from iblnm.data import _write_dataframe, _read_dataframe

        df = pd.DataFrame({
            'probabilityLeft': np.full(3, 0.5),
            'stim_side': pd.array(['left', 'right', 'left'], dtype='string'),
            'contrast': np.array([50., 100., 25.]),
        })
        fpath = tmp_path / 'frame.h5'
        with h5py.File(fpath, 'w') as f:
            _write_dataframe(f.create_group('trials'), df)
        with h5py.File(fpath, 'r') as f:
            out = _read_dataframe(f['trials'])

        assert set(out.columns) == {'probabilityLeft', 'stim_side', 'contrast'}
        assert out['stim_side'].tolist() == ['left', 'right', 'left']


def _full_one_trials_frame(n=6, index=None):
    """Trials frame carrying every ONE table column plus the derived four.

    The 13 ONE columns are the verbatim `_ibl_trials.table.pqt` schema; the
    derived columns (`contrast`, `signed_contrast`, `stim_side`, `trial`) are
    what `load_trials` adds. `index` sets the `trial` values, defaulting to a
    contiguous range.
    """
    trial = np.arange(n) if index is None else np.asarray(index)
    return pd.DataFrame({
        'intervals_0':         np.linspace(99.0, 499.0, n),
        'intervals_1':         np.linspace(102.0, 502.0, n),
        'goCue_times':         np.linspace(99.6, 499.6, n),
        'response_times':      np.linspace(100.4, 500.4, n),
        'choice':              np.tile([-1.0, 1.0], n // 2),
        'stimOnTrigger_times':        np.linspace(99.5, 499.5, n),
        'contrastLeft':        np.tile([0.25, np.nan], n // 2),
        'contrastRight':       np.tile([np.nan, 1.0], n // 2),
        'feedback_times':      np.linspace(101.0, 501.0, n),
        'feedbackType':        np.tile([1.0, -1.0], n // 2),
        'rewardVolume':        np.tile([0.0, 1.5], n // 2),
        'probabilityLeft':     np.full(n, 0.5),
        'firstMovement_times': np.linspace(100.3, 500.3, n),
        'contrast':            np.tile([25.0, 100.0], n // 2),
        'signed_contrast':     np.tile([-25.0, 100.0], n // 2),
        'stim_side':           np.tile(['left', 'right'], n // 2),
        'trial':               trial,
    })


class TestTrialsTableProduct:
    """The `trials/table` product: verbatim ONE table plus trial identity."""

    def _session(self, mock_session_series, tmp_path):
        from iblnm.data import PhotometrySession
        ps = PhotometrySession(mock_session_series, one=MagicMock(),
                               load_data=False)
        ps.filepath = tmp_path / f'{ps.eid}.h5'
        return ps

    def test_full_table_roundtrips_with_noncontiguous_trial(
            self, mock_session_series, tmp_path):
        """Every ONE column plus the derived four survive, trial values intact."""
        ps = self._session(mock_session_series, tmp_path)
        ps.trials = _full_one_trials_frame(n=6, index=[0, 3, 7, 12, 13, 20])
        original = ps.trials.copy()
        ps.save_h5(groups=['trials'])

        reloaded = self._session(mock_session_series, tmp_path)
        reloaded.load_h5(groups=['trials'])

        assert set(reloaded.trials.columns) == set(original.columns)
        pd.testing.assert_frame_equal(
            reloaded.trials[original.columns], original, check_dtype=False)

    def test_saved_table_is_found_in_the_store(self, mock_session_series, tmp_path):
        """save_h5 writes trials/table where the presence check looks for it."""
        ps = self._session(mock_session_series, tmp_path)
        assert not ps.stored_product_exists('trials/table')
        ps.trials = _full_one_trials_frame()
        ps.save_h5(groups=['trials'])
        assert ps.stored_product_exists('trials/table')

    def test_load_trials_records_one_index_as_trial(self, mock_session_series,
                                                    tmp_path):
        """The raw ONE index becomes the `trial` column, before contrasts."""
        from iblnm.data import PhotometrySession
        ps = self._session(mock_session_series, tmp_path)
        one_index = [0, 3, 7, 12]
        raw = pd.DataFrame({
            'contrastLeft':  [0.25, np.nan, np.nan, 0.0625],
            'contrastRight': [np.nan, 1.0, 0.0, np.nan],
        }, index=one_index)

        def _set_trials():
            ps.trials = raw

        with patch.object(PhotometrySession.__bases__[0], 'load_trials',
                          side_effect=_set_trials):
            ps.load_trials()

        np.testing.assert_array_equal(ps.trials['trial'].values, one_index)
        assert {'contrast', 'signed_contrast', 'stim_side'} <= set(ps.trials.columns)

    def test_signed_zero_survives_roundtrip(self, mock_session_series, tmp_path):
        """Zero-contrast stimulus side rides on the sign bit of signed_contrast."""
        ps = self._session(mock_session_series, tmp_path)
        ps.trials = _full_one_trials_frame(n=4)
        ps.trials['signed_contrast'] = np.array([-0.0, 0.0, -0.0, 25.0])
        expected = np.signbit(ps.trials['signed_contrast'].values)
        ps.save_h5(groups=['trials'])

        reloaded = self._session(mock_session_series, tmp_path)
        reloaded.load_h5(groups=['trials'])
        np.testing.assert_array_equal(
            np.signbit(reloaded.trials['signed_contrast'].values), expected)


class TestSaveLoadH5:
    def test_auto_detect_writes_only_the_groups_held(self, mock_session_series,
                                                     tmp_path):
        """A session holding only trials writes the trials group and no other.

        `save_h5` with no `groups` asks the session which products it carries,
        which is a question of the attribute being there — a session that never
        fetched photometry must not raise on the way to writing its trials.
        """
        from iblnm.data import PhotometrySession
        session = PhotometrySession(mock_session_series, one=MagicMock(),
                                    load_data=False)
        session.trials = _make_trials()
        fpath = tmp_path / f'{session.eid}.h5'
        session.save_h5(fpath)

        import h5py
        with h5py.File(fpath, 'r') as h5:
            assert 'trials/table' in h5
            assert 'photometry' not in h5
            assert 'wheel' not in h5
            assert 'video' not in h5

    def test_save_preprocessed_float64(self, mock_photometry_session, tmp_path):
        """save_h5 should write preprocessed signal as float64 with timestamps."""
        session = mock_photometry_session
        session.extract_preprocessed_photometry()
        fpath = tmp_path / f'{session.eid}.h5'
        session.save_h5(fpath)

        import h5py
        with h5py.File(fpath, 'r') as f:
            pp_grp = f['photometry/VTA/preprocessed']
            assert pp_grp['signal'].dtype == np.float64
            np.testing.assert_allclose(
                pp_grp['signal'][:],
                session.photometry['GCaMP_preprocessed']['VTA'].values,
                rtol=1e-10
            )
            np.testing.assert_allclose(
                pp_grp['times'][:],
                session.photometry['GCaMP_preprocessed'].index.values,
                rtol=1e-10
            )

    def test_saved_photometry_products_are_found_in_the_store(
            self, mock_photometry_session, tmp_path):
        """save_h5 writes each product where the presence check looks for it.

        Covers the orchestrator wiring, not the primitives: `_save_photometry`
        must write each product into its own group, or the file it just wrote
        reads back as empty.
        """
        session = mock_photometry_session
        session.extract_preprocessed_photometry()
        session.trials = _make_trials()
        session.photometry_responses = session.extract_responses(
            session.photometry['GCaMP_preprocessed'])
        session.filepath = tmp_path / f'{session.eid}.h5'
        session.save_h5()

        assert session.stored_product_exists('photometry/preprocessed')
        assert session.stored_product_exists('photometry/responses')

    def test_preprocessed_band_comes_from_the_product(self, mock_photometry_session,
                                                       mock_session_series, tmp_path):
        """Which self.photometry key holds the preprocessed signal is fixed.

        It is named by the `photometry/preprocessed` product, so neither the
        save nor the load takes it as an argument.
        """
        from iblnm.data import PREPROCESSED_BAND, PhotometrySession
        session = mock_photometry_session
        session.extract_preprocessed_photometry()
        fpath = tmp_path / f'{session.eid}.h5'
        session.save_h5(fpath)

        reloaded = PhotometrySession(mock_session_series, one=MagicMock(),
                                     load_data=False)
        reloaded.load_h5(fpath, groups=['photometry'])

        assert PREPROCESSED_BAND in reloaded.photometry
        np.testing.assert_allclose(
            reloaded.photometry[PREPROCESSED_BAND]['VTA'].values,
            session.photometry[PREPROCESSED_BAND]['VTA'].values)

    def test_save_trials_and_responses(self, mock_photometry_session, tmp_path):
        """save_h5 in append mode should add trials and xarray responses."""
        session = mock_photometry_session
        session.extract_preprocessed_photometry()
        n = 50
        session.trials = _make_trials(
            n,
            goCue_times=np.linspace(100, 500, n),
            response_times=np.linspace(100.5, 500.5, n),
            intervals_0=np.linspace(99, 499, n),
            intervals_1=np.linspace(102, 502, n),
            choice=np.random.choice([-1, 1], n),
            feedbackType=np.random.choice([-1, 1], n),
            probabilityLeft=np.random.choice([0.2, 0.5, 0.8], n),
            signed_contrast=np.random.choice([-100, -25, 0, 25, 100], n).astype(float),
            contrast=np.random.choice([0, 25, 100], n).astype(float),
        )
        session.photometry_responses = session.extract_responses(
            session.photometry['GCaMP_preprocessed'], events=['stimOnTrigger_times', 'feedback_times'])

        fpath = tmp_path / f'{session.eid}.h5'
        session.save_h5(fpath)  # Create with preprocessed
        session.save_h5(fpath, mode='a')  # Append trials + responses

        import h5py
        with h5py.File(fpath, 'r') as f:
            assert 'photometry/VTA/preprocessed/signal' in f
            assert 'trials/table/choice' in f
            assert 'photometry/VTA/responses/stimOnTrigger_times' in f
            assert 'photometry/VTA/responses/feedback_times' in f
            # fs is never read back (the time axis is rebuilt from `times`)
            assert 'fs' not in f['photometry/VTA/responses'].attrs
            # Verify response data matches xarray content
            resp_h5 = f['photometry/VTA/responses/stimOnTrigger_times'][:]
            resp_xr = session.photometry_responses['VTA'].sel(event='stimOnTrigger_times').values
            np.testing.assert_allclose(resp_h5, resp_xr, rtol=1e-5)
            np.testing.assert_array_equal(
                f['trials/table/choice'][:],
                session.trials['choice'].values
            )

    def test_load_h5_restores_xarray_responses(self, mock_photometry_session, tmp_path):
        """load_h5 should restore responses as xarray DataArray."""
        import xarray as xr
        session = mock_photometry_session
        session.extract_preprocessed_photometry()
        session.trials = _make_trials()
        session.photometry_responses = session.extract_responses(
            session.photometry['GCaMP_preprocessed'], events=['stimOnTrigger_times', 'feedback_times'])
        original = {r: da.copy() for r, da in session.photometry_responses.items()}

        fpath = tmp_path / f'{session.eid}.h5'
        session.save_h5(fpath)
        session.save_h5(fpath, mode='a')

        # Clear and reload
        session.photometry_responses = {}
        session.load_h5(fpath)
        assert isinstance(session.photometry_responses, dict)
        assert isinstance(session.photometry_responses['VTA'], xr.DataArray)
        assert set(session.photometry_responses['VTA'].dims) == {'event', 'trial', 'time'}
        np.testing.assert_allclose(
            session.photometry_responses['VTA'].sel(event='stimOnTrigger_times').values,
            original['VTA'].sel(event='stimOnTrigger_times').values,
            rtol=1e-5,
        )

    def test_load_h5_restores_trials(self, mock_photometry_session, tmp_path):
        """load_h5 should restore trials saved in the HDF5 trials group."""
        session = mock_photometry_session
        session.extract_preprocessed_photometry()
        n = 50
        session.trials = pd.DataFrame({
            'stimOnTrigger_times':        np.linspace(99.5, 499.5, n),
            'firstMovement_times': np.linspace(100.3, 500.3, n),
            'feedback_times':      np.linspace(101.0, 501.0, n),
            'goCue_times':         np.linspace(99.6, 499.6, n),
            'response_times':      np.linspace(100.4, 500.4, n),
            'intervals_0':         np.linspace(99.0, 499.0, n),
            'intervals_1':         np.linspace(102.0, 502.0, n),
            'choice':              np.random.choice([-1, 1], n).astype(float),
            'feedbackType':        np.random.choice([-1, 1], n).astype(float),
            'probabilityLeft':     np.full(n, 0.5),
            'signed_contrast':     np.zeros(n),
            'contrast':            np.zeros(n),
        })
        saved_stim = session.trials['stimOnTrigger_times'].values.copy()
        fpath = tmp_path / f'{session.eid}.h5'
        session.save_h5(fpath)
        session.save_h5(fpath, mode='a')
        session.trials = None
        session.load_h5(fpath)
        assert session.trials is not None
        assert 'stimOnTrigger_times' in session.trials.columns
        np.testing.assert_allclose(session.trials['stimOnTrigger_times'].values, saved_stim)

    def test_save_load_qc_roundtrip(self, mock_session_series, tmp_path):
        """Per-region QC attrs under photometry/<region>/raw/qc/ survive H5 roundtrip."""
        from iblnm.data import PhotometrySession
        mock_one = MagicMock()
        ps = PhotometrySession(mock_session_series, one=mock_one, load_data=False)
        ps.photometry_qc = {
            'VTA': {'n_unique_samples_GCaMP': 0.4,
                    'n_unique_samples_Isosbestic': 0.6},
            'SNc': {'n_unique_samples_GCaMP': 0.2,
                    'n_unique_samples_Isosbestic': 0.3},
        }
        ps.neurophotometrics_qc = {'n_band_inversions': 0.0}
        fpath = tmp_path / f'{ps.eid}.h5'
        ps.save_h5(fpath, groups=['metadata', 'photometry'])

        ps2 = PhotometrySession(mock_session_series, one=mock_one, load_data=False)
        ps2.load_h5(fpath, groups=['photometry'])
        assert ps2.photometry_qc == ps.photometry_qc
        assert ps2.neurophotometrics_qc == ps.neurophotometrics_qc

    def test_load_h5_roundtrip(self, mock_photometry_session, tmp_path):
        """load_h5 should restore preprocessed signal from saved file."""
        session = mock_photometry_session
        session.extract_preprocessed_photometry()
        fpath = tmp_path / f'{session.eid}.h5'
        session.save_h5(fpath)

        original = session.photometry['GCaMP_preprocessed']['VTA'].values.copy()

        # Clear and reload
        session.photometry.pop('GCaMP_preprocessed')
        session.load_h5(fpath)
        reloaded = session.photometry['GCaMP_preprocessed']['VTA'].values
        np.testing.assert_allclose(reloaded, original, rtol=1e-10)

    def _make_video_session(self, mock_session_series, manual_qc=None):
        """PhotometrySession carrying synthetic movement responses + xcorr."""
        import xarray as xr
        from iblnm.config import MOVEMENT_EVENTS
        from iblnm.data import PhotometrySession
        ps = PhotometrySession(mock_session_series, one=MagicMock(),
                               load_data=False)
        n_trial, n_time = 4, 20
        labels = ['paw', 'nose', 'tongue_speed', 'tongue_likelihood']
        rng = np.random.default_rng(0)
        ps.movement_responses = {
            label: xr.DataArray(
                rng.standard_normal((len(MOVEMENT_EVENTS), n_trial, n_time)),
                dims=['event', 'trial', 'time'],
                coords={
                    'event': list(MOVEMENT_EVENTS),
                    'trial': np.arange(n_trial),
                    'time': np.linspace(-1, 1, n_time),
                },
            )
            for label in labels
        }
        n_lags = 11
        ps.pose_xcorr = {
            'functions': rng.standard_normal((3, n_lags)),
            'lags': np.linspace(-5, 5, n_lags),
            'peak_lags': np.array([0.1, 0.2, 0.4]),
            'drift': 0.3,
        }
        ps.video_manual_qc = dict(manual_qc or {})
        return ps

    def test_save_load_video_roundtrip(self, mock_session_series, tmp_path):
        """video group round-trips movement responses, xcorr, and QC labels."""
        from iblnm.data import PhotometrySession
        ps = self._make_video_session(
            mock_session_series,
            manual_qc={'qc_lp': 'FAIL', 'qc_movement': 'WARNING'})
        fpath = tmp_path / f'{ps.eid}.h5'
        ps.save_h5(fpath, groups=['video'])

        ps2 = PhotometrySession(mock_session_series, one=MagicMock(),
                                load_data=False)
        ps2.load_h5(fpath, groups=['video'])

        assert set(ps2.movement_responses) == set(ps.movement_responses)
        for label, responses in ps.movement_responses.items():
            for event in responses.coords['event'].values:
                np.testing.assert_allclose(
                    ps2.movement_responses[label].sel(event=event).values,
                    responses.sel(event=event).values,
                )
        np.testing.assert_allclose(ps2.pose_xcorr['functions'],
                                   ps.pose_xcorr['functions'])
        np.testing.assert_allclose(ps2.pose_xcorr['lags'], ps.pose_xcorr['lags'])
        np.testing.assert_allclose(ps2.pose_xcorr['peak_lags'],
                                   ps.pose_xcorr['peak_lags'])
        assert ps2.pose_xcorr['drift'] == ps.pose_xcorr['drift']
        assert ps2.video_manual_qc == {'qc_lp': 'FAIL', 'qc_movement': 'WARNING'}

    def test_save_video_writes_per_label_responses_groups(
            self, mock_session_series, tmp_path):
        """Movement responses persist under video/{label}/responses, the same
        layout photometry uses — the old flat trace groups are gone."""
        import h5py
        ps = self._make_video_session(mock_session_series)
        fpath = tmp_path / f'{ps.eid}.h5'
        ps.save_h5(fpath, groups=['video'])

        with h5py.File(fpath, 'r') as f:
            video = f['video']
            assert 'traces' not in video and 'baseline_traces' not in video
            for label in ps.movement_responses:
                assert 'responses' in video[label]
                assert set(ps.movement_responses[label].coords['event'].values) \
                    <= set(video[label]['responses'].keys())

    def test_rebuilding_responses_leaves_manual_qc(self, mock_session_series,
                                                   tmp_path):
        """Re-cutting the responses of a session holding no verdicts leaves the
        stored ones standing — manual_qc is its own group, not part of any
        derived product."""
        from iblnm.data import PhotometrySession
        ps = self._make_video_session(
            mock_session_series,
            manual_qc={'qc_lp': 'FAIL', 'qc_movement': 'CRITICAL'})
        fpath = tmp_path / f'{ps.eid}.h5'
        ps.save_h5(fpath, groups=['video'])

        # Fresh session with only automatic data, holding no verdicts at all.
        ps_auto = self._make_video_session(mock_session_series)
        assert ps_auto.video_manual_qc == {}
        ps_auto.save_h5(fpath, groups=['video'])

        ps2 = PhotometrySession(mock_session_series, one=MagicMock(),
                                load_data=False)
        ps2.load_h5(fpath, groups=['video'])
        assert ps2.video_manual_qc == {'qc_lp': 'FAIL',
                                       'qc_movement': 'CRITICAL'}


class TestManualQC:
    """Manual QC labels, per photometry region and per session for video."""

    def _session(self, mock_session_series, tmp_path):
        from iblnm.data import PhotometrySession
        ps = PhotometrySession(mock_session_series, one=MagicMock(),
                               load_data=False)
        ps.filepath = tmp_path / f'{ps.eid}.h5'
        return ps

    def test_set_manual_qc_round_trips_for_region_and_video(
            self, mock_session_series, tmp_path):
        """A label set through the session is what a fresh session reads back,
        under video/manual_qc for the session and photometry/{region}/manual_qc
        for one recording."""
        ps = self._session(mock_session_series, tmp_path)
        ps.set_manual_qc('qc_lp', 'FAIL')
        ps.set_manual_qc('qc_movement', 'PASS', region='VTA')

        ps2 = self._session(mock_session_series, tmp_path)
        ps2.load_h5(groups=['photometry', 'video'])
        assert ps2.video_manual_qc == {'qc_lp': 'FAIL'}
        assert ps2.photometry_manual_qc == {'VTA': {'qc_movement': 'PASS'}}

    @pytest.mark.parametrize('field, value', [('qc_lp', 'GOOD'),
                                              ('qc_other', 'PASS')])
    def test_set_manual_qc_rejects_bad_input_before_writing(
            self, mock_session_series, tmp_path, field, value):
        """An out-of-vocabulary verdict or field raises and writes nothing, so a
        typo in a viewer cannot leave a half-written file behind."""
        ps = self._session(mock_session_series, tmp_path)
        with pytest.raises(ValueError):
            ps.set_manual_qc(field, value)
        assert not ps.filepath.exists()

    def test_refetching_raw_photometry_clears_every_region(
            self, mock_session_series, tmp_path):
        """The raw fetch brings back all regions at once, so it invalidates the
        verdicts passed on all of them."""
        from iblnm.data import PhotometrySession
        ps = self._session(mock_session_series, tmp_path)
        ps.set_manual_qc('qc_lp', 'FAIL', region='VTA')

        with patch.object(PhotometrySession.__bases__[0], 'load_photometry',
                          side_effect=lambda *a, **k: None):
            ps.load_raw_photometry()

        assert ps.photometry_manual_qc == {}
        fresh = self._session(mock_session_series, tmp_path)
        fresh.load_h5(groups=['photometry'])
        assert fresh.photometry_manual_qc == {}


class TestFetchVideoQC:
    """PhotometrySession.fetch_video_qc selects the 8 VIDEO_QC_COLS, unstored."""

    def test_returns_eight_qc_cols(self, mock_session_series):
        from iblnm.config import VIDEO_QC_COLS
        from iblnm.data import PhotometrySession
        ps = PhotometrySession(mock_session_series, one=MagicMock(),
                               load_data=False)
        fetched = pd.Series({col: 'PASS' for col in VIDEO_QC_COLS})
        fetched['qc_videoLeft_timestamps'] = 'FAIL'
        with patch('iblnm.io.get_extended_qc', return_value=fetched):
            video_qc = ps.fetch_video_qc()
        assert set(video_qc) == set(VIDEO_QC_COLS)
        assert video_qc['qc_videoLeft_timestamps'] == 'FAIL'
        assert video_qc['qc_videoLeft_focus'] == 'PASS'
        assert ps.video_qc == video_qc

    def test_missing_cols_default_not_set(self, mock_session_series):
        from iblnm.config import VIDEO_QC_COLS
        from iblnm.data import LP_QC_NOT_SET, PhotometrySession
        ps = PhotometrySession(mock_session_series, one=MagicMock(),
                               load_data=False)
        with patch('iblnm.io.get_extended_qc',
                   return_value=pd.Series({'eid': ps.eid})):
            ps.fetch_video_qc()
        assert set(ps.video_qc) == set(VIDEO_QC_COLS)
        assert all(v == LP_QC_NOT_SET for v in ps.video_qc.values())


# =============================================================================
# Pose (LightningPose) Method Tests
# =============================================================================

class TestPoseMethods:
    """PhotometrySession LP pose loading and extraction."""

    def test_load_pose_loads_only_lightningpose(self, mock_session_series, tmp_path):
        """load_pose pulls only lightningPose via load_dataset (never the whole
        leftCamera object) and no longer loads camera times."""
        from iblnm.data import PhotometrySession
        pose_df = pd.DataFrame({'paw_l_x': [0.0, 1.0], 'paw_l_y': [0.0, 0.0],
                                'paw_l_likelihood': [1.0, 1.0]})

        one = MagicMock()
        one.load_dataset.return_value = pose_df
        ps = PhotometrySession(mock_session_series, one=one, load_data=False)
        ps.filepath = tmp_path / f'{ps.eid}.h5'
        ps.load_pose()
        one.load_object.assert_not_called()
        pd.testing.assert_frame_equal(ps.pose, pose_df)
        assert not hasattr(ps, 'pose_times')
        loaded_names = [call.args[1] for call in one.load_dataset.call_args_list]
        assert all('lightningPose' in name for name in loaded_names)

    def test_load_pose_missing_raises_missing_lp(self, mock_session_series, tmp_path):
        """load_pose raises MissingLP when the pose dataset is absent."""
        from one.alf.exceptions import ALFObjectNotFound
        from iblnm.data import PhotometrySession
        from iblnm.validation import MissingLP

        one = MagicMock()
        one.load_dataset.side_effect = ALFObjectNotFound('leftCamera.lightningPose')
        ps = PhotometrySession(mock_session_series, one=one, load_data=False)
        ps.filepath = tmp_path / f'{ps.eid}.h5'
        with pytest.raises(MissingLP):
            ps.load_pose()

    def test_load_camera_times_sets_pose_times(self, mock_session_series, tmp_path):
        """load_camera_times loads only leftCamera.times into pose_times."""
        from iblnm.data import PhotometrySession
        times = np.array([0.0, 0.1, 0.2])

        one = MagicMock()
        one.load_dataset.return_value = times
        ps = PhotometrySession(mock_session_series, one=one, load_data=False)
        ps.filepath = tmp_path / f'{ps.eid}.h5'
        ps.load_camera_times()
        np.testing.assert_array_equal(ps.pose_times, times)
        loaded_names = [call.args[1] for call in one.load_dataset.call_args_list]
        assert all('times' in name for name in loaded_names)

    def test_load_camera_times_missing_raises(self, mock_session_series, tmp_path):
        """load_camera_times raises MissingVideoTimestamps when times are absent."""
        from one.alf.exceptions import ALFObjectNotFound
        from iblnm.data import PhotometrySession
        from iblnm.validation import MissingVideoTimestamps

        one = MagicMock()
        one.load_dataset.side_effect = ALFObjectNotFound('leftCamera.times')
        ps = PhotometrySession(mock_session_series, one=one, load_data=False)
        ps.filepath = tmp_path / f'{ps.eid}.h5'
        with pytest.raises(MissingVideoTimestamps):
            ps.load_camera_times()

    def test_load_motion_energy_sets_array(self, mock_session_series, tmp_path):
        """load_motion_energy loads ROIMotionEnergy (no _ibl_ prefix) into
        self.motion_energy."""
        from iblnm.data import PhotometrySession
        me = np.array([0.0, 1.0, 2.0, 3.0])

        one = MagicMock()
        one.load_dataset.return_value = me
        ps = PhotometrySession(mock_session_series, one=one, load_data=False)
        ps.filepath = tmp_path / f'{ps.eid}.h5'
        ps.load_motion_energy()
        np.testing.assert_array_equal(ps.motion_energy, me)
        loaded_names = [call.args[1] for call in one.load_dataset.call_args_list]
        assert all('ROIMotionEnergy' in name and '_ibl_' not in name
                   for name in loaded_names)

    def test_load_motion_energy_missing_raises(self, mock_session_series, tmp_path):
        """load_motion_energy raises MissingMotionEnergy when the dataset is absent."""
        from one.alf.exceptions import ALFObjectNotFound
        from iblnm.data import PhotometrySession
        from iblnm.validation import MissingMotionEnergy

        one = MagicMock()
        one.load_dataset.side_effect = ALFObjectNotFound('leftCamera.ROIMotionEnergy')
        ps = PhotometrySession(mock_session_series, one=one, load_data=False)
        ps.filepath = tmp_path / f'{ps.eid}.h5'
        with pytest.raises(MissingMotionEnergy):
            ps.load_motion_energy()

    def test_run_video_times_qc(self, mock_session_series, tmp_path):
        """run_video_times_qc yields hand-computed discrepancy and framerate."""
        from iblnm.data import PhotometrySession
        ps = PhotometrySession(mock_session_series, one=MagicMock(),
                               load_data=False)
        ps.filepath = tmp_path / f'{ps.eid}.h5'
        ps.pose_times = np.array([1.0, 1.1, 1.2, 1.35])
        ps.session_length = 0.3
        measures = ps.run_video_times_qc()
        # video span 0.35 - session_length 0.3 = 0.05
        assert measures['length_discrepancy'] == pytest.approx(0.05)
        # diffs: [0.1, 0.1, 0.15] -> median 0.1
        assert measures['framerate_from_tpts'] == pytest.approx(0.1)

    def test_video_measures_round_trip_without_responses(self, mock_session_series,
                                                          tmp_path):
        """save_h5(['video']) with no responses but measures set writes a video
        group that load_h5 restores both measures from."""
        from iblnm.data import PhotometrySession
        ps = PhotometrySession(mock_session_series, one=MagicMock(),
                               load_data=False)
        ps.video_times_qc = {'length_discrepancy': 0.05,
                             'framerate_from_tpts': 0.0333}
        fpath = tmp_path / f'{ps.eid}.h5'
        ps.save_h5(fpath, groups=['video'])

        ps2 = PhotometrySession(mock_session_series, one=MagicMock(),
                                load_data=False)
        ps2.load_h5(fpath, groups=['video'])
        assert not hasattr(ps2, 'movement_responses')
        assert ps2.video_times_qc == pytest.approx(ps.video_times_qc)

    def test_available_save_groups_includes_video_for_measures_only(
            self, mock_session_series):
        """A traceless session with only basic-video measures still saves the
        video group (so MissingLP sessions persist their measures)."""
        from iblnm.data import PhotometrySession
        ps = PhotometrySession(mock_session_series, one=MagicMock(),
                               load_data=False)
        ps.video_times_qc = {'length_discrepancy': 0.05}
        assert 'video' in ps._available_save_groups()

    def _make_pose_session(self, mock_session_series, tmp_path, fs=30, dur=60.0,
                           tongue_like=(0.2, 0.9), accelerate=False,
                           motion_energy=False):
        """PhotometrySession with injected synthetic pose + camera times + trials.

        With ``accelerate``, keypoint positions grow quadratically so speed rises
        over time and the event a window is locked to changes its value. With
        ``motion_energy``, a per-frame ME ramp is injected on the camera time base.

        `filepath` points into `tmp_path`: `extract_movement_signals` writes the
        product it builds, which would otherwise land in the real store.
        """
        from iblnm.data import PhotometrySession
        ps = PhotometrySession(mock_session_series, one=MagicMock(),
                               load_data=False)
        ps.filepath = tmp_path / f'{ps.eid}.h5'
        t = np.arange(0, dur, 1 / fs)
        n = t.size
        ramp = np.arange(n, dtype=float)
        if accelerate:
            ramp = ramp ** 2
        ones = np.ones(n)
        if motion_energy:
            ps.motion_energy = ramp.copy()
        ps.pose = pd.DataFrame({
            'paw_l_x': ramp * 3.0, 'paw_l_y': ramp * 4.0, 'paw_l_likelihood': ones,
            'paw_r_x': ramp * 6.0, 'paw_r_y': ramp * 8.0, 'paw_r_likelihood': ones,
            'nose_tip_x': ramp, 'nose_tip_y': ramp, 'nose_tip_likelihood': ones,
            'tongue_end_l_x': ramp, 'tongue_end_l_y': ramp,
            'tongue_end_l_likelihood': np.full(n, tongue_like[0]),
            'tongue_end_r_x': ramp, 'tongue_end_r_y': ramp,
            'tongue_end_r_likelihood': np.full(n, tongue_like[1]),
        })
        ps.pose_times = t
        ps.trials = pd.DataFrame({
            'trial': [0, 1, 2],
            'stimOnTrigger_times': [9.0, 19.0, 29.0],
            'firstMovement_times': [10.0, 20.0, 30.0],
            'feedback_times': [12.0, 22.0, 32.0],
        })
        return ps

    def test_movement_responses_shapes_and_labels(self, mock_session_series,
                                                  tmp_path):
        """One (event, trial, time) grid per movement label, full event axis."""
        from iblnm.config import MOVEMENT_EVENTS, POSE_MEASURES
        ps = self._make_pose_session(mock_session_series, tmp_path, fs=30)
        responses = ps.extract_responses(ps.extract_movement_signals(),
                                         events=MOVEMENT_EVENTS)
        assert set(responses) == set(POSE_MEASURES)
        assert responses['paw'].sizes == {'event': 3, 'trial': 3, 'time': 60}
        assert (list(responses['paw'].coords['event'].values)
                == list(MOVEMENT_EVENTS))

    def _movement_responses(self, ps):
        """Movement responses for `ps` through the unified extraction engine."""
        from iblnm.config import MOVEMENT_EVENTS
        return ps.extract_responses(ps.extract_movement_signals(),
                                    events=MOVEMENT_EVENTS)

    def test_movement_responses_own_event_and_stimon_cells(self, mock_session_series,
                                                           tmp_path):
        """The stimOn baseline is a read-time cell of the same grid: the
        stimOn-locked nose channel has identical own-event and stimOn cells,
        while the firstMovement-locked paw channel does not."""
        from iblnm.config import LABEL2EVENT
        ps = self._make_pose_session(mock_session_series, tmp_path, fs=30,
                                     accelerate=True)
        responses = self._movement_responses(ps)
        np.testing.assert_allclose(
            responses['nose'].sel(event=LABEL2EVENT['nose']).values,
            responses['nose'].sel(event='stimOnTrigger_times').values)
        assert not np.allclose(
            responses['paw'].sel(event=LABEL2EVENT['paw']).values,
            responses['paw'].sel(event='stimOnTrigger_times').values,
            equal_nan=True)

    def test_movement_responses_tongue_likelihood_is_max(self, mock_session_series,
                                                         tmp_path):
        """tongue_likelihood trace equals the per-frame max of the two tips."""
        ps = self._make_pose_session(mock_session_series, tmp_path,
                                     tongue_like=(0.2, 0.9))
        responses = self._movement_responses(ps)
        np.testing.assert_allclose(
            responses['tongue_likelihood'].sel(event='feedback_times').values, 0.9)

    def test_movement_responses_common_timebase_across_fps(self, mock_session_series,
                                                           tmp_path):
        """Different camera fps → identical trace time length (resampled to POSE_FS)."""
        r30 = self._movement_responses(
            self._make_pose_session(mock_session_series, tmp_path / '30', fs=30))
        r99 = self._movement_responses(
            self._make_pose_session(mock_session_series, tmp_path / '99', fs=99))
        assert r30['paw'].sizes['time'] == r99['paw'].sizes['time'] == 60

    def test_movement_responses_include_motion_energy(self, mock_session_series,
                                                      tmp_path):
        """pose + ME present → the LP labels plus a motion_energy channel."""
        from iblnm.config import POSE_MEASURES
        ps = self._make_pose_session(mock_session_series, tmp_path,
                                     motion_energy=True)
        responses = self._movement_responses(ps)
        assert set(responses) == set(POSE_MEASURES) | {'motion_energy'}

    def test_movement_responses_motion_energy_only(self, mock_session_series,
                                                   tmp_path):
        """ME present, no pose → exactly ['motion_energy']."""
        ps = self._make_pose_session(mock_session_series, tmp_path,
                                     motion_energy=True)
        del ps.pose
        assert list(self._movement_responses(ps)) == ['motion_energy']

    def test_movement_responses_lp_only_when_no_motion_energy(self,
                                                              mock_session_series,
                                                              tmp_path):
        """pose present, no motion energy → only the LP labels."""
        from iblnm.config import POSE_MEASURES
        ps = self._make_pose_session(mock_session_series, tmp_path)
        assert not hasattr(ps, 'motion_energy')
        assert set(self._movement_responses(ps)) == set(POSE_MEASURES)

    def test_movement_signals_empty_without_sources(self, mock_session_series,
                                                    tmp_path):
        """Neither pose nor motion energy → no signals, hence no responses."""
        ps = self._make_pose_session(mock_session_series, tmp_path)
        del ps.pose
        assert ps.extract_movement_signals() == {}
        assert self._movement_responses(ps) == {}

    @staticmethod
    def _xcorr_session(mock_session_series, wheel_times=None):
        """Build a session with an imposed late-third paw/wheel shift.

        ``wheel_times`` overrides the wheel sample times (default: the uniform
        grid); used to exercise non-uniform / float32 timestamps.
        """
        from iblnm.data import PhotometrySession
        fs, dur, shift = 100, 60.0, 8  # shift in samples
        t = np.arange(0, dur, 1 / fs)
        rng = np.random.default_rng(0)
        freqs = rng.uniform(1.0, 10.0, 40)
        phases = rng.uniform(0, 2 * np.pi, 40)
        def base(tt):
            return np.sin(2 * np.pi * freqs[:, None] * tt[None, :]
                          + phases[:, None]).sum(0)
        # Wheel speed used by the method is |velocity|, so build both signals
        # from the same non-negative pattern for a lag-0 match in aligned thirds.
        wheel_velocity = base(t)
        # Late third: paw pattern leads the wheel by `shift` samples.
        paw_eval = np.where(t >= (2 / 3) * dur, t + shift / fs, t)
        paw_speed = np.abs(base(paw_eval))
        # Integrate paw speed to x positions so keypoint_speed recovers it.
        paw_x = np.concatenate([[0.0], np.cumsum(paw_speed[1:])])

        ps = PhotometrySession(mock_session_series, one=MagicMock(),
                               load_data=False)
        n = t.size
        ps.pose = pd.DataFrame({
            'paw_l_x': paw_x, 'paw_l_y': np.zeros(n), 'paw_l_likelihood': np.ones(n),
            'paw_r_x': np.zeros(n), 'paw_r_y': np.zeros(n),
            'paw_r_likelihood': np.zeros(n),  # untracked → does not contribute
        })
        ps.pose_times = t
        ps.wheel_velocity = pd.Series(
            wheel_velocity, index=t if wheel_times is None else wheel_times)
        return ps, shift, fs

    def test_run_pose_qc_recovers_drift(self, mock_session_series):
        """An imposed late-third paw/wheel shift surfaces in pose_xcorr['drift']."""
        ps, shift, fs = self._xcorr_session(mock_session_series)
        ps.run_pose_qc()
        np.testing.assert_allclose(ps.pose_xcorr['peak_lags'][0], 0.0, atol=1 / fs)
        np.testing.assert_allclose(ps.pose_xcorr['drift'], shift / fs, atol=1 / fs)

    def test_run_pose_qc_float32_wheel_times(self, mock_session_series):
        """float32 (non-uniform) wheel times do not raise — regression for the
        movements() even-sampling crash; drift stays finite."""
        ps, shift, fs = self._xcorr_session(mock_session_series)
        # float32 storage makes consecutive dt non-uniform well above the 1e-10
        # tolerance that brainbox.movements() asserted on (the crash class).
        ps.wheel_velocity.index = ps.wheel_velocity.index.to_numpy().astype(
            np.float32)
        assert not np.all(np.abs(np.diff(ps.wheel_velocity.index.to_numpy())
                                 - (1 / fs)) < 1e-10)  # genuinely non-uniform
        ps.run_pose_qc()
        assert np.isfinite(ps.pose_xcorr['drift'])


# =============================================================================
# Task Performance Method Tests
# =============================================================================

def _make_training_trials(n=200, seed=42):
    """Mock training trials (single 0.5 block)."""
    np.random.seed(seed)
    contrasts = np.random.choice([0, 0.0625, 0.125, 0.25, 0.5, 1.0], size=n)
    sides = np.random.choice([-1, 1], size=n)
    contrast_left = np.where(sides == -1, contrasts, 0).astype(float)
    contrast_right = np.where(sides == 1, contrasts, 0).astype(float)
    choice = sides.copy()
    nogo_idx = np.random.choice(n, size=10, replace=False)
    choice[nogo_idx] = 0
    feedback_type = np.where(choice == sides, 1, -1)
    feedback_type[choice == 0] = -1
    return pd.DataFrame({
        'contrastLeft': contrast_left,
        'contrastRight': contrast_right,
        'contrast': contrasts,
        'choice': choice,
        'feedbackType': feedback_type,
        'probabilityLeft': np.full(n, 0.5),
    })


def _make_biased_trials(seed=42):
    """Mock biased trials with 20/50/80 blocks."""
    np.random.seed(seed)
    probability_left = np.concatenate([
        np.full(50, 0.5), np.full(50, 0.2), np.full(50, 0.8), np.full(50, 0.5),
    ])
    n = len(probability_left)
    contrasts = np.random.choice([0, 0.0625, 0.125, 0.25, 0.5, 1.0], size=n)
    sides = np.random.choice([-1, 1], size=n)
    contrast_left = np.where(sides == -1, contrasts, 0).astype(float)
    contrast_right = np.where(sides == 1, contrasts, 0).astype(float)
    choice = sides.copy()
    nogo_idx = np.random.choice(n, size=10, replace=False)
    choice[nogo_idx] = 0
    feedback_type = np.where(choice == sides, 1, -1)
    feedback_type[choice == 0] = -1
    return pd.DataFrame({
        'contrastLeft': contrast_left,
        'contrastRight': contrast_right,
        'contrast': contrasts,
        'choice': choice,
        'feedbackType': feedback_type,
        'probabilityLeft': probability_left,
    })


class TestExtractPerformance:
    """Tests for PhotometrySession.extract_performance()."""

    def _session(self, series, session_type, trials):
        from iblnm.data import PhotometrySession
        series = series.copy()
        series['session_type'] = session_type
        session = PhotometrySession(series, one=MagicMock(), load_data=False)
        session.trials = trials
        return session

    def test_biased_returns_basic_and_block_keys(self, mock_session_series):
        """A blocked session is scored with both halves in one call."""
        session = self._session(mock_session_series, 'biased', _make_biased_trials())
        result = session.extract_performance()
        for key in ['n_trials', 'contrasts', 'fraction_correct',
                    'fraction_correct_easy', 'nogo_fraction', 'psych_50_bias',
                    'psych_50_threshold', 'psych_50_r_squared', 'psych_50_n_trials',
                    'psych_20_bias', 'psych_80_bias', 'bias_shift']:
            assert key in result, f"Missing key: {key}"
        assert result is session.performance

    def test_training_omits_block_keys(self, mock_session_series):
        """An unblocked session type is scored with the basic half only."""
        session = self._session(mock_session_series, 'training',
                                _make_training_trials())
        result = session.extract_performance()
        assert not any(key.startswith(('psych_20', 'psych_80')) for key in result)
        assert 'bias_shift' not in result

    def test_no_bias_shift_without_both_blocks(self, mock_session_series):
        """A blocked session type presenting only the 0.5 block has no shift."""
        session = self._session(mock_session_series, 'ephys',
                                _make_training_trials())
        result = session.extract_performance()
        assert 'bias_shift' not in result

    def test_writes_nothing(self, mock_session_series, tmp_path):
        """Scoring is pure: the H5 file absent before the call stays absent."""
        session = self._session(mock_session_series, 'biased', _make_biased_trials())
        session.filepath = tmp_path / f'{session.eid}.h5'
        session.extract_performance()
        assert not session.filepath.exists()

    def test_raises_without_trials(self, mock_session_series):
        """Scoring a session that never fetched its trials raises.

        The processing tier reads its input straight off the session, so a
        missing input surfaces as an `AttributeError` for the caller's error
        handler rather than as an empty score.
        """
        from iblnm.data import PhotometrySession
        session = PhotometrySession(mock_session_series, one=MagicMock(),
                                    load_data=False)
        with pytest.raises(AttributeError):
            session.extract_performance()


# =============================================================================
# QC Method Tests
# =============================================================================

def _score_neurophotometrics(session, n_band_inversions=0, n_early_samples=0):
    """Score a fetched source table with both metrics mocked out."""
    from unittest.mock import patch
    session.neurophotometrics = pd.DataFrame({'col1': [1.0, 2.0]},
                                             index=[0.0, 1.0])
    with patch('iblnm.data.metrics') as mock_metrics:
        mock_metrics.n_band_inversions.return_value = n_band_inversions
        mock_metrics.n_early_samples.return_value = n_early_samples
        return session.run_neurophotometrics_qc()


class TestRunNeurophotometricsQc:
    """Tests for PhotometrySession.run_neurophotometrics_qc."""

    def test_stores_metric_values(self, mock_photometry_session):
        session = mock_photometry_session
        _score_neurophotometrics(session, n_band_inversions=3, n_early_samples=5)
        assert session.neurophotometrics_qc == {'n_band_inversions': 3.0,
                                                'n_early_samples': 5.0}

    def test_scores_the_fetched_table_without_fetching(self,
                                                       mock_photometry_session):
        """It reads `self.neurophotometrics`; the fetch is the caller's job."""
        from unittest.mock import patch
        session = mock_photometry_session
        with patch.object(session, 'fetch_neurophotometrics') as fetch:
            _score_neurophotometrics(session, n_early_samples=2)
        fetch.assert_not_called()

    def test_writes_nothing(self, mock_photometry_session):
        """The computation is pure: scoring alone leaves no H5 file behind."""
        session = mock_photometry_session
        assert not session.filepath.exists()
        _score_neurophotometrics(session, n_band_inversions=1)
        assert not session.filepath.exists()


def _tidy_qc(metric='n_unique_samples', band='GCaMP', region='VTA',
             values=(0.8, 0.9)):
    """One tidy `qc_signals` frame: a whole-signal row plus one row per window."""
    return pd.DataFrame({
        'band': [band] * (len(values) + 1),
        'brain_region': [region] * (len(values) + 1),
        'metric': [metric] * (len(values) + 1),
        'value': [np.mean(values), *values],
        'window': [np.nan, *range(len(values))],
    })


class TestRunPhotometryQc:
    """Tests for PhotometrySession.run_photometry_qc."""

    def test_stores_band_suffixed_metrics_per_region(self, mock_photometry_session):
        from unittest.mock import patch
        session = mock_photometry_session
        tidy = pd.concat([
            _tidy_qc(band='GCaMP', values=(0.8, 0.9)),
            _tidy_qc(band='Isosbestic', values=(0.2, 0.4)),
        ])
        with patch('iblnm.data.qc_signals', return_value=tidy):
            scored = session.run_photometry_qc(sliding_metrics=['n_unique_samples'])
        assert scored is session.photometry_qc
        assert set(session.photometry_qc) == {'VTA'}
        assert set(session.photometry_qc['VTA']) == {
            'n_unique_samples_GCaMP', 'n_unique_samples_Isosbestic'}

    def test_writes_nothing(self, mock_photometry_session):
        """The computation is pure: scoring alone leaves no H5 file behind."""
        from unittest.mock import patch
        session = mock_photometry_session
        assert not session.filepath.exists()
        with patch('iblnm.data.qc_signals', return_value=_tidy_qc()):
            session.run_photometry_qc(sliding_metrics=['n_unique_samples'])
        assert not session.filepath.exists()

    def test_splits_undetrended_metrics_into_own_call(self, mock_photometry_session):
        """n_unique_samples is scored without detrending, the rest with it."""
        from unittest.mock import patch
        session = mock_photometry_session
        with patch('iblnm.data.qc_signals', return_value=_tidy_qc()) as mock_qc:
            session.run_photometry_qc(
                sliding_metrics=['n_unique_samples', 'ar_score'],
                sliding_kwargs={'w_len': 120, 'step_len': 60, 'detrend': True},
            )
        assert mock_qc.call_count == 2
        calls = {call.kwargs['sliding_kwargs']['detrend']: call.kwargs
                 for call in mock_qc.call_args_list}
        assert [m.__name__ for m in calls[False]['metrics']] == ['n_unique_samples']
        assert [m.__name__ for m in calls[True]['metrics']] == ['ar_score']
        for kwargs in calls.values():
            assert kwargs['sliding_kwargs']['w_len'] == 120
            assert kwargs['sliding_kwargs']['step_len'] == 60

    def test_aggregates_each_metric_by_its_own_reducer(self, mock_photometry_session):
        """n_unique_samples takes the windows' q10, ar_score their mean."""
        from unittest.mock import patch
        session = mock_photometry_session
        windows = (0.1, 0.8, 0.9, 1.0)
        # One frame per call, in the order run_photometry_qc issues them:
        # un-detrended first, detrended second.
        per_call = [_tidy_qc(metric='n_unique_samples', values=windows),
                    _tidy_qc(metric='ar_score', values=windows)]
        with patch('iblnm.data.qc_signals', side_effect=per_call):
            session.run_photometry_qc(
                sliding_metrics=['n_unique_samples', 'ar_score'])
        stored = session.photometry_qc['VTA']
        assert stored['n_unique_samples_GCaMP'] == pytest.approx(0.31)
        assert stored['ar_score_GCaMP'] == pytest.approx(0.7)

    def test_propagates_qc_signals_failure(self, mock_photometry_session):
        from unittest.mock import patch
        session = mock_photometry_session
        with patch('iblnm.data.qc_signals', side_effect=Exception("qc_signals failed")):
            with pytest.raises(Exception, match="qc_signals failed"):
                session.run_photometry_qc()


class TestLoadPhotometryQc:
    """`load_photometry_qc` returns the stored product, scoring if absent."""

    def test_scores_when_absent(self, mock_photometry_session):
        from unittest.mock import patch
        session = mock_photometry_session
        with patch('iblnm.data.qc_signals', return_value=_tidy_qc()):
            with patch.object(type(session), 'load_raw_photometry') as fetch:
                stored = session.load_photometry_qc()
        fetch.assert_called_once()
        assert set(stored) == {'VTA'}
        assert 'n_unique_samples_GCaMP' in stored['VTA']
        assert session.stored_product_exists('photometry/raw/qc')

    def test_reads_stored_product_without_rescoring(self, mock_photometry_session,
                                                    mock_session_series):
        from unittest.mock import patch
        from iblnm.data import PhotometrySession
        session = mock_photometry_session
        with patch('iblnm.data.qc_signals', return_value=_tidy_qc()):
            with patch.object(type(session), 'load_raw_photometry'):
                built = session.load_photometry_qc()

        fresh = PhotometrySession(mock_session_series, one=MagicMock(),
                                  load_data=False)
        fresh.filepath = session.filepath
        with patch('iblnm.data.qc_signals') as score:
            assert fresh.load_photometry_qc() == built
        score.assert_not_called()

class TestLoadNeurophotometricsQc:
    """`load_neurophotometrics_qc` returns the stored product, scoring if absent."""

    def _build(self, session, **metric_values):
        """Fetch-and-score through the load path, the source table mocked out."""
        from unittest.mock import patch
        table = pd.DataFrame({'col1': [1.0, 2.0]}, index=[0.0, 1.0])

        def _assign():
            session.neurophotometrics = table

        with patch.object(session, 'fetch_neurophotometrics',
                          side_effect=_assign):
            with patch('iblnm.data.metrics') as mock_metrics:
                mock_metrics.n_band_inversions.return_value = metric_values.get(
                    'n_band_inversions', 0)
                mock_metrics.n_early_samples.return_value = metric_values.get(
                    'n_early_samples', 0)
                return session.load_neurophotometrics_qc()

    def test_fetches_and_scores_when_absent(self, mock_photometry_session):
        session = mock_photometry_session
        assert self._build(session, n_early_samples=4) == {
            'n_band_inversions': 0.0, 'n_early_samples': 4.0}

    def test_writes_group_although_source_table_is_never_stored(
            self, mock_photometry_session):
        """The neurophotometrics QC group persists without its source table."""
        import h5py
        session = mock_photometry_session
        self._build(session, n_band_inversions=1)
        with h5py.File(session.filepath, 'r') as h5:
            group = h5['photometry/neurophotometrics/qc']
            assert group.attrs['n_band_inversions'] == 1.0
            assert list(h5['photometry/neurophotometrics']) == ['qc']
        assert session.stored_product_exists('photometry/neurophotometrics/qc')

    def test_propagates_fetch_failure(self, mock_photometry_session):
        from unittest.mock import patch
        session = mock_photometry_session
        with patch.object(session, 'fetch_neurophotometrics',
                          side_effect=Exception("load failed")):
            with pytest.raises(Exception, match="load failed"):
                session.load_neurophotometrics_qc()

    def test_reads_stored_product_without_rescoring(self, mock_photometry_session,
                                                    mock_session_series):
        from unittest.mock import patch
        from iblnm.data import PhotometrySession
        session = mock_photometry_session
        built = self._build(session, n_early_samples=4)

        fresh = PhotometrySession(mock_session_series, one=MagicMock(),
                                  load_data=False)
        fresh.filepath = session.filepath
        with patch.object(PhotometrySession, 'fetch_neurophotometrics') as fetch:
            assert fresh.load_neurophotometrics_qc() == built
        fetch.assert_not_called()


# =============================================================================
# Baseline Subtraction Tests
# =============================================================================

def _make_responses(tpts, vals, region='R', event='e'):
    """Build a minimal (1 event, n_trials, n_times) DataArray for one region."""
    import xarray as xr
    data = np.array([[vals]] if vals.ndim == 1 else [vals])
    return xr.DataArray(
        data,
        dims=['event', 'trial', 'time'],
        coords={
            'event':  [event],
            'trial':  np.arange(data.shape[1]),
            'time':   tpts,
        },
    )


class TestSubtractBaseline:
    """Tests for PhotometrySession.subtract_baseline."""

    def _session(self, mock_session_series):
        from iblnm.data import PhotometrySession
        return PhotometrySession(mock_session_series, one=MagicMock(), load_data=False)

    def test_subtracts_pretrial_mean(self, mock_session_series):
        session = self._session(mock_session_series)
        tpts = np.array([-1.0, -0.5, -0.1, 0.5, 1.0])
        # window=(-1, 0) → tpts[0:3]=[-1, -0.5, -0.1], mean=4.0
        vals = np.array([2., 4., 6., 8., 10.])
        responses = _make_responses(tpts, vals)
        result = session.subtract_baseline(responses, window=(-1.0, 0.0))
        np.testing.assert_allclose(result.values[0, 0], [-2., 0., 2., 4., 6.])

    def test_subtracts_per_trial(self, mock_session_series):
        """Each trial gets its own baseline removed."""
        session = self._session(mock_session_series)
        tpts = np.array([-1.0, -0.5, 0.5, 1.0])
        # window=(-1, 0) → tpts[0:2]=[-1, -0.5]
        # trial 0: [2, 4, 6, 8], baseline=3.0 → [-1, 1, 3, 5]
        # trial 1: [10, 20, 30, 40], baseline=15.0 → [-5, 5, 15, 25]
        vals = np.array([[2., 4., 6., 8.], [10., 20., 30., 40.]])
        responses = _make_responses(tpts, vals)
        result = session.subtract_baseline(responses, window=(-1.0, 0.0))
        np.testing.assert_allclose(result.values[0, 0], [-1., 1., 3., 5.])
        np.testing.assert_allclose(result.values[0, 1], [-5., 5., 15., 25.])

    def test_does_not_modify_input(self, mock_session_series):
        """Returns new DataArray; input unchanged."""
        session = self._session(mock_session_series)
        tpts = np.array([-1.0, -0.5, 0.5, 1.0])
        vals = np.array([1., 2., 3., 4.])
        responses = _make_responses(tpts, vals)
        original_vals = responses.values.copy()
        result = session.subtract_baseline(responses, window=(-1.0, 0.0))
        np.testing.assert_array_equal(responses.values, original_vals)
        assert result is not responses

    def test_empty_window_produces_nan(self, mock_session_series):
        """Window entirely outside time axis → baseline NaN → output all NaN."""
        session = self._session(mock_session_series)
        tpts = np.array([0.5, 1.0, 1.5])
        vals = np.array([1., 2., 3.])
        responses = _make_responses(tpts, vals)
        result = session.subtract_baseline(responses, window=(-1.0, 0.0))
        assert np.all(np.isnan(result.values))

    def test_default_window_uses_config_baseline(self, mock_session_series):
        """Default window should be BASELINE_WINDOW from config, not RESPONSE_WINDOW."""
        from iblnm.config import BASELINE_WINDOW
        session = self._session(mock_session_series)
        tpts = np.array([-0.3, -0.15, -0.05, 0.0, 0.5, 1.0])
        vals = np.array([1., 2., 3., 4., 5., 6.])
        responses = _make_responses(tpts, vals)
        # Call without explicit window — should use BASELINE_WINDOW
        result_default = session.subtract_baseline(responses)
        result_explicit = session.subtract_baseline(responses, window=BASELINE_WINDOW)
        np.testing.assert_array_equal(result_default.values, result_explicit.values)


# =============================================================================
# Event Masking Tests
# =============================================================================

class TestMaskSubsequentEvents:
    """Tests for PhotometrySession.mask_subsequent_events."""

    def _make_session_and_responses(self, mock_session_series):
        from iblnm.data import PhotometrySession
        import xarray as xr
        session = PhotometrySession(mock_session_series, one=MagicMock(), load_data=False)
        tpts = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        data = np.ones((2, 2, 5))  # (event, trial, time)
        responses = xr.DataArray(
            data,
            dims=['event', 'trial', 'time'],
            coords={
                'event':  ['stimOnTrigger_times', 'firstMovement_times'],
                'trial':  [0, 1],
                'time':   tpts,
            },
        )
        # trial 0: dt = 0.3 - 0.0 = 0.3 → mask tpts > 0.3 (indices 3, 4)
        # trial 1: firstMovement = NaN → no masking
        session.trials = pd.DataFrame({
            'stimOnTrigger_times':        [0.0, 0.0],
            'firstMovement_times': [0.3, np.nan],
            'feedback_times':      [1.5, 1.5],
        })
        return session, responses

    def test_masks_times_after_next_event(self, mock_session_series):
        session, responses = self._make_session_and_responses(mock_session_series)
        result = session.mask_subsequent_events(
            responses,
            event_order=['stimOnTrigger_times', 'firstMovement_times', 'feedback_times'],
        )
        mat = result.sel(event='stimOnTrigger_times').values
        assert np.isnan(mat[0, 3])       # trial 0, t=0.5 > 0.3 → NaN
        assert np.isnan(mat[0, 4])       # trial 0, t=1.0 > 0.3 → NaN
        assert not np.isnan(mat[0, 2])   # trial 0, t=0.0 ≤ 0.3 → kept
        assert not np.isnan(mat[1, 3])   # trial 1, NaN dt → not masked

    def test_last_event_not_masked(self, mock_session_series):
        """firstMovement event matrix is unchanged (no event after it in responses)."""
        session, responses = self._make_session_and_responses(mock_session_series)
        result = session.mask_subsequent_events(
            responses,
            event_order=['stimOnTrigger_times', 'firstMovement_times'],
        )
        mat = result.sel(event='firstMovement_times').values
        assert not np.any(np.isnan(mat))

    def test_nan_dt_not_masked(self, mock_session_series):
        """Trial 1 has NaN firstMovement → stimOn response fully intact."""
        session, responses = self._make_session_and_responses(mock_session_series)
        result = session.mask_subsequent_events(
            responses,
            event_order=['stimOnTrigger_times', 'firstMovement_times', 'feedback_times'],
        )
        mat = result.sel(event='stimOnTrigger_times').values
        assert not np.any(np.isnan(mat[1]))

    def test_no_trials_returns_unchanged(self, mock_session_series):
        """If self.trials is None, return responses unchanged."""
        from iblnm.data import PhotometrySession
        session = PhotometrySession(mock_session_series, one=MagicMock(), load_data=False)
        session.trials = None
        tpts = np.array([-1.0, 0.0, 1.0])
        vals = np.array([1., 2., 3.])
        responses = _make_responses(tpts, vals)
        result = session.mask_subsequent_events(responses)
        np.testing.assert_array_equal(result.values, responses.values)

    def test_event_not_in_responses_skipped(self, mock_session_series):
        """Event in event_order but not in DataArray coords → no error."""
        from iblnm.data import PhotometrySession
        import xarray as xr
        session = PhotometrySession(mock_session_series, one=MagicMock(), load_data=False)
        tpts = np.array([-1.0, 0.0, 1.0])
        data = np.ones((1, 2, 3))
        responses = xr.DataArray(
            data,
            dims=['event', 'trial', 'time'],
            coords={'event': ['feedback_times'],
                    'trial': [0, 1], 'time': tpts},
        )
        session.trials = pd.DataFrame({
            'stimOnTrigger_times':        [0.0, 0.0],
            'firstMovement_times': [0.3, 0.4],
            'feedback_times':      [1.5, 1.5],
        })
        # stimOnTrigger_times not in responses → skip without error
        result = session.mask_subsequent_events(
            responses,
            event_order=['stimOnTrigger_times', 'firstMovement_times', 'feedback_times'],
        )
        np.testing.assert_array_equal(result.values, responses.values)

    def test_default_event_order_masks_stimon_at_feedback(self, mock_session_series):
        """With the default event_order (RESPONSE_EVENTS), stimOn is masked at
        feedback: firstMovement is no longer the event between them."""
        from iblnm.data import PhotometrySession
        import xarray as xr
        session = PhotometrySession(mock_session_series, one=MagicMock(), load_data=False)
        tpts = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        responses = xr.DataArray(
            np.ones((1, 2, 5)),
            dims=['event', 'trial', 'time'],
            coords={'event': ['stimOnTrigger_times'], 'trial': [0, 1], 'time': tpts},
        )
        session.trials = pd.DataFrame({
            'stimOnTrigger_times':        [0.0, 0.0],
            'firstMovement_times': [np.nan, np.nan],
            'feedback_times':      [0.3, np.nan],
        })
        result = session.mask_subsequent_events(responses)  # default event_order
        mat = result.sel(event='stimOnTrigger_times').values
        assert np.isnan(mat[0, 3])      # t=0.5 > feedback-stimOn=0.3 → masked
        assert np.isnan(mat[0, 4])      # t=1.0 > 0.3 → masked
        assert not np.isnan(mat[0, 2])  # t=0.0 ≤ 0.3 → kept
        assert not np.isnan(mat[1, 3])  # feedback NaN → not masked


class TestMatchPhotometryToMetadata:
    """_match_photometry_to_metadata renames columns to match brain_region metadata."""

    def _make_session(self, mock_session_series, brain_region, hemisphere):
        from iblnm.data import PhotometrySession
        series = mock_session_series.copy()
        series['brain_region'] = brain_region
        series['hemisphere'] = hemisphere
        session = PhotometrySession(series, one=MagicMock(), load_data=False)
        return session

    def test_renames_bare_to_suffixed(self, mock_session_series):
        """Bare column 'VTA' + metadata 'VTA-r' → renames to 'VTA-r'."""
        session = self._make_session(mock_session_series, ['VTA-r'], ['r'])
        t = np.linspace(0, 10, 100)
        session.photometry = {
            'GCaMP': pd.DataFrame({'VTA': np.ones(100)}, index=t),
            'Isosbestic': pd.DataFrame({'VTA': np.ones(100)}, index=t),
        }
        session._match_photometry_to_metadata()

        assert list(session.photometry['GCaMP'].columns) == ['VTA-r']
        assert list(session.photometry['Isosbestic'].columns) == ['VTA-r']

    def test_exact_match_no_rename(self, mock_session_series):
        """Columns already match metadata → no rename."""
        session = self._make_session(mock_session_series, ['VTA-r'], ['r'])
        t = np.linspace(0, 10, 100)
        session.photometry = {
            'GCaMP': pd.DataFrame({'VTA-r': np.ones(100)}, index=t),
        }
        session._match_photometry_to_metadata()

        assert list(session.photometry['GCaMP'].columns) == ['VTA-r']

    def test_midline_exact_match(self, mock_session_series):
        """Midline region 'DR' matches metadata 'DR' exactly."""
        session = self._make_session(mock_session_series, ['DR'], [None])
        t = np.linspace(0, 10, 100)
        session.photometry = {
            'GCaMP': pd.DataFrame({'DR': np.ones(100)}, index=t),
        }
        session._match_photometry_to_metadata()

        assert list(session.photometry['GCaMP'].columns) == ['DR']

    def test_bilateral_suffixed_exact_match(self, mock_session_series):
        """Bilateral NBM with suffixed columns matches metadata."""
        session = self._make_session(mock_session_series, ['NBM-l', 'NBM-r'], ['l', 'r'])
        t = np.linspace(0, 10, 100)
        session.photometry = {
            'GCaMP': pd.DataFrame({
                'NBM-l': np.ones(100), 'NBM-r': np.ones(100),
            }, index=t),
        }
        session._match_photometry_to_metadata()

        assert sorted(session.photometry['GCaMP'].columns) == ['NBM-l', 'NBM-r']

    def test_bilateral_bare_raises_ambiguous(self, mock_session_series):
        """Bare 'NBM' with metadata ['NBM-l','NBM-r'] → AmbiguousRegionMapping."""
        from iblnm.validation import AmbiguousRegionMapping
        session = self._make_session(mock_session_series, ['NBM-l', 'NBM-r'], ['l', 'r'])
        t = np.linspace(0, 10, 100)
        session.photometry = {
            'GCaMP': pd.DataFrame(
                np.ones((100, 2)), columns=['NBM', 'NBM'], index=t,
            ),
        }
        with pytest.raises(AmbiguousRegionMapping, match='multiple'):
            session._match_photometry_to_metadata()

    def test_no_match_raises(self, mock_session_series):
        """Column with no matching metadata entry raises AmbiguousRegionMapping."""
        from iblnm.validation import AmbiguousRegionMapping
        session = self._make_session(mock_session_series, ['VTA-r'], ['r'])
        t = np.linspace(0, 10, 100)
        session.photometry = {
            'GCaMP': pd.DataFrame({'XYZ': np.ones(100)}, index=t),
        }
        with pytest.raises(AmbiguousRegionMapping, match='no match'):
            session._match_photometry_to_metadata()

    def test_mixed_regions_rename(self, mock_session_series):
        """Multi-region: bare 'VTA' → 'VTA-r', midline 'DR' stays."""
        session = self._make_session(
            mock_session_series, ['VTA-r', 'DR'], ['r', None],
        )
        t = np.linspace(0, 10, 100)
        session.photometry = {
            'GCaMP': pd.DataFrame({'VTA': np.ones(100), 'DR': np.ones(100)}, index=t),
        }
        session._match_photometry_to_metadata()

        assert sorted(session.photometry['GCaMP'].columns) == ['DR', 'VTA-r']

    def test_empty_photometry_noop(self, mock_session_series):
        """Empty photometry dict → no error."""
        session = self._make_session(mock_session_series, ['VTA-r'], ['r'])
        session.photometry = {}
        session._match_photometry_to_metadata()  # should not raise

    def test_empty_brain_region_noop(self, mock_session_series):
        """Empty brain_region list → no error."""
        session = self._make_session(mock_session_series, [], [])
        t = np.linspace(0, 10, 100)
        session.photometry = {
            'GCaMP': pd.DataFrame({'VTA': np.ones(100)}, index=t),
        }
        session._match_photometry_to_metadata()  # should not raise


class TestTrialsPerformanceProduct:
    """The `trials/performance` product: behavioral scalars built from trials."""

    def _session(self, series, tmp_path, trials=None, session_type='training'):
        from iblnm.data import PhotometrySession
        series = series.copy()
        series['session_type'] = session_type
        ps = PhotometrySession(series, one=MagicMock(), load_data=False)
        ps.filepath = tmp_path / f'{ps.eid}.h5'
        ps.trials = trials
        return ps

    def test_built_product_roundtrips_and_reports_current(
            self, mock_session_series, tmp_path):
        """Building from trials stores a stamped product a reload reproduces."""
        ps = self._session(mock_session_series, tmp_path, _make_training_trials())
        built = ps.load_performance()
        assert ps.stored_product_exists('trials/performance')

        reopened = self._session(mock_session_series, tmp_path)
        reloaded = reopened.load_performance()
        assert reloaded == built

    def test_training_stores_no_block_keys(self, mock_session_series, tmp_path):
        """A training session has one block, so no 20/80 psychometrics are fit."""
        ps = self._session(mock_session_series, tmp_path, _make_training_trials())
        stored = ps.load_performance()
        assert {'n_trials', 'contrasts', 'fraction_correct', 'nogo_fraction',
                'psych_50_bias'} <= set(stored)
        assert not any(key.startswith(('psych_20', 'psych_80')) for key in stored)

    def test_biased_stores_block_keys(self, mock_session_series, tmp_path):
        """A biased session adds the per-block psychometrics and the bias shift."""
        ps = self._session(mock_session_series, tmp_path, _make_biased_trials(),
                           session_type='biased')
        stored = ps.load_performance()
        assert {'psych_20_bias', 'psych_80_bias', 'bias_shift'} <= set(stored)
        assert stored['contrasts'] == [0.0, 0.0625, 0.125, 0.25, 0.5, 1.0]


# =============================================================================
# PhotometrySessionGroup Tests
# =============================================================================

def _make_sessions_df(n_eids=2, regions_per=2):
    """Helper to build a session-level DataFrame with list columns."""
    rows = []
    region_names = ['VTA-r', 'DR-l', 'SNc-r', 'LC-l']
    target_names = ['target-0', 'target-1', 'target-2', 'target-3']
    for i in range(n_eids):
        rows.append({
            'eid': f'eid-{i}',
            'subject': f'subj-{i % 2}',
            'brain_region': [region_names[j] for j in range(regions_per)],
            'hemisphere': [region_names[j][-1] for j in range(regions_per)],
            'target_NM': [target_names[j] for j in range(regions_per)],
            'NM': 'NM-0',
            'session_type': 'biased',
            'start_time': '2024-01-01T10:00:00',
            'number': 1,
            'task_protocol': 'biased_protocol',
        })
    return pd.DataFrame(rows)


def _make_recordings_df(n_eids=2, regions_per=2):
    """Helper to build a recordings DataFrame (exploded)."""
    rows = []
    region_names = ['VTA-r', 'DR-l', 'SNc-r', 'LC-l']
    for i in range(n_eids):
        for j in range(regions_per):
            rows.append({
                'eid': f'eid-{i}',
                'subject': f'subj-{i % 2}',
                'brain_region': region_names[j],
                'hemisphere': region_names[j][-1],
                'target_NM': f'target-{j}',
                'NM': f'NM-{j}',
                'session_type': 'biased',
                'start_time': '2024-01-01T10:00:00',
                'number': 1,
                'task_protocol': 'biased_protocol',
            })
    return pd.DataFrame(rows)


class TestFromCatalog:
    """Tests for PhotometrySessionGroup.from_catalog."""

    def _make_catalog(self):
        """Return a minimal catalog DataFrame with parallel list columns."""
        return pd.DataFrame([
            {
                'eid': 'eid-1', 'subject': 'mouse_A',
                'session_type': 'biased', 'start_time': '2024-01-01T10:00:00',
                'number': 1, 'task_protocol': 'biased_protocol',
                'brain_region': ['VTA', 'SNc'], 'hemisphere': ['l', 'r'],
                'target_NM': ['VTA-DA', 'SNc-DA'], 'NM': 'DA',
            },
            {
                'eid': 'eid-2', 'subject': 'mouse_B',
                'session_type': 'training', 'start_time': '2024-01-02T10:00:00',
                'number': 1, 'task_protocol': 'training_protocol',
                'brain_region': ['DR'], 'hemisphere': ['l'],
                'target_NM': ['DR-5HT'], 'NM': '5HT',
            },
            {   # Mismatched lengths — should be dropped
                'eid': 'eid-3', 'subject': 'mouse_C',
                'session_type': 'biased', 'start_time': '2024-01-03T10:00:00',
                'number': 1, 'task_protocol': 'biased_protocol',
                'brain_region': ['VTA'], 'hemisphere': [],
                'target_NM': ['VTA-DA'], 'NM': 'DA',
            },
        ])

    def test_validates_parallel_columns(self):
        """from_catalog validates parallel list columns and drops mismatched."""
        from iblnm.data import PhotometrySessionGroup
        group = PhotometrySessionGroup.from_catalog(self._make_catalog(), one=MagicMock(), h5_dir=None)
        # eid-3 dropped (mismatched), eid-1 and eid-2 kept
        assert len(group.sessions) == 2
        assert 'eid-3' not in group.sessions['eid'].values

    def test_recordings_from_catalog(self):
        """recordings produces one row per region after from_catalog."""
        from iblnm.data import PhotometrySessionGroup
        group = PhotometrySessionGroup.from_catalog(self._make_catalog(), one=MagicMock(), h5_dir=None)
        group.filter_sessions(session_types=False, targetnms=False,
                              qc_blockers=set(), photometry_qc=False,
                              min_performance=False, required_contrasts=False)
        # eid-1 has 2 regions, eid-2 has 1
        assert len(group.recordings) == 3
        assert 'fiber_idx' in group.recordings.columns

    def test_filter_reflected_in_recordings(self):
        """Filtering by session type is reflected in recordings."""
        from iblnm.data import PhotometrySessionGroup
        group = PhotometrySessionGroup.from_catalog(self._make_catalog(), one=MagicMock(), h5_dir=None)
        group.filter_sessions(session_types=('biased',), targetnms=False,
                              qc_blockers=set(),
                              min_performance=False, required_contrasts=False)
        assert all(group.recordings['session_type'] == 'biased')

    def test_from_catalog_enforces_schema(self):
        """from_catalog fills missing schema columns with typed defaults."""
        from iblnm.data import PhotometrySessionGroup
        group = PhotometrySessionGroup.from_catalog(self._make_catalog(), one=MagicMock(), h5_dir=None)
        # 'lab' is in SESSION_SCHEMA but absent from the catalog fixture
        assert 'lab' in group._catalog.columns

    def test_from_catalog_without_a_connection(self):
        """No `one`: an offline group over stored sessions needs no Alyx."""
        from iblnm.data import PhotometrySessionGroup
        group = PhotometrySessionGroup.from_catalog(self._make_catalog(),
                                                    h5_dir=None)
        connected = PhotometrySessionGroup.from_catalog(
            self._make_catalog(), one=MagicMock(), h5_dir=None)
        assert group.one is None
        assert len(group.sessions) == len(connected.sessions)

    def test_recordings_reflects_refilter(self):
        """recordings updates automatically when filter_sessions is re-called."""
        from iblnm.data import PhotometrySessionGroup
        group = PhotometrySessionGroup.from_catalog(self._make_catalog(), one=MagicMock(), h5_dir=None)
        group.filter_sessions(session_types=('biased',), targetnms=False,
                              qc_blockers=set(),
                              min_performance=False, required_contrasts=False)
        assert all(group.recordings['session_type'] == 'biased')

        group.filter_sessions(session_types=('training',), targetnms=False,
                              qc_blockers=set(),
                              min_performance=False, required_contrasts=False)
        assert all(group.recordings['session_type'] == 'training')

    def test_logged_errors_scanned_from_h5(self, tmp_path):
        """from_catalog(h5_dir=...) scans /errors groups; a qc_blocker drops the session."""
        from iblnm.data import PhotometrySessionGroup
        from iblnm.validation import MissingRawData
        from tests.test_util import _write_session_h5
        _write_session_h5(tmp_path, 'eid-1', 'mouse_A', 'biased',
                          brain_region=['VTA'], errors=[MissingRawData('x')])
        _write_session_h5(tmp_path, 'eid-2', 'mouse_B', 'biased', brain_region=['DR'])
        catalog = pd.DataFrame([
            {'eid': 'eid-1', 'subject': 'mouse_A', 'session_type': 'biased',
             'start_time': '2024-01-01T10:00:00', 'number': 1, 'brain_region': ['VTA'],
             'hemisphere': ['l'], 'target_NM': ['VTA-DA'], 'NM': 'DA'},
            {'eid': 'eid-2', 'subject': 'mouse_B', 'session_type': 'biased',
             'start_time': '2024-01-02T10:00:00', 'number': 1, 'brain_region': ['DR'],
             'hemisphere': ['l'], 'target_NM': ['DR-5HT'], 'NM': '5HT'},
        ])
        group = PhotometrySessionGroup.from_catalog(
            catalog, one=MagicMock(), h5_dir=tmp_path)
        group.filter_sessions(session_types=False, targetnms=False,
                              qc_blockers={'MissingRawData'},
                              min_performance=False, required_contrasts=False)
        assert set(group.sessions['eid']) == {'eid-2'}

    def test_scan_reads_the_h5_afresh(self, tmp_path):
        """Nothing is cached: a group built against an empty directory sees no
        errors, however many times the scan has run before."""
        from iblnm.data import PhotometrySessionGroup
        from iblnm.validation import MissingRawData
        from tests.test_util import _write_session_h5
        _write_session_h5(tmp_path, 'eid-1', 'mouse_A', 'biased',
                          brain_region=['VTA'], errors=[MissingRawData('x')])
        catalog = pd.DataFrame([
            {'eid': 'eid-1', 'subject': 'mouse_A', 'session_type': 'biased',
             'start_time': '2024-01-01T10:00:00', 'number': 1,
             'brain_region': ['VTA'], 'hemisphere': ['l'],
             'target_NM': ['VTA-DA'], 'NM': 'DA'},
        ])
        scanned = PhotometrySessionGroup.from_catalog(
            catalog, one=MagicMock(), h5_dir=tmp_path)
        assert scanned._catalog['logged_errors'].iloc[0] == ['MissingRawData']

        rescanned = PhotometrySessionGroup.from_catalog(
            catalog, one=MagicMock(), h5_dir=tmp_path / 'empty')
        assert rescanned._catalog['logged_errors'].iloc[0] == []

    def test_no_h5_dir_leaves_logged_errors_empty(self):
        """Without h5_dir, from_catalog skips the scan and logged_errors are all empty."""
        from iblnm.data import PhotometrySessionGroup
        group = PhotometrySessionGroup.from_catalog(self._make_catalog(), one=MagicMock(), h5_dir=None)
        assert group._catalog['logged_errors'].apply(lambda x: x == []).all()

    def test_scan_h5_false_reuses_existing_column(self, tmp_path):
        """scan_h5=False skips the H5 scan and keeps a pre-existing
        logged_errors column without a merge collision."""
        from iblnm.data import PhotometrySessionGroup
        catalog = self._make_catalog()
        catalog['logged_errors'] = [['MissingRawData'] for _ in range(len(catalog))]
        group = PhotometrySessionGroup.from_catalog(
            catalog, one=MagicMock(), h5_dir=tmp_path, scan_h5=False)
        assert group._catalog['logged_errors'].apply(
            lambda x: x == ['MissingRawData']).all()


def _collector_catalog(session_types):
    """Catalog of single-region sessions, one row per (eid -> session_type)."""
    return pd.DataFrame([
        {'eid': eid, 'subject': f'mouse_{eid}', 'session_type': session_type,
         'start_time': '2024-01-01T10:00:00', 'number': 1,
         'brain_region': ['VTA'], 'hemisphere': ['l'],
         'target_NM': ['VTA-DA'], 'NM': 'DA'}
        for eid, session_type in session_types.items()
    ])


class TestGroupCollectErrors:
    """PhotometrySessionGroup.collect_errors reads the filtered sessions only."""

    def _group(self, tmp_path, session_types, scan_h5=False):
        from iblnm.data import PhotometrySessionGroup
        return PhotometrySessionGroup.from_catalog(
            _collector_catalog(session_types), one=None, h5_dir=tmp_path,
            scan_h5=scan_h5)

    def test_collects_errors_from_every_session(self, tmp_path):
        from tests.test_util import _write_session_h5
        _write_session_h5(tmp_path, 'eid-1', 'mouse_A',
                          errors=[ValueError("bad value")])
        _write_session_h5(tmp_path, 'eid-2', 'mouse_B',
                          errors=[TypeError("bad type"), KeyError("missing")])
        group = self._group(tmp_path, {'eid-1': 'biased', 'eid-2': 'biased'})

        df = group.collect_errors()

        assert len(df) == 3
        assert set(df['eid']) == {'eid-1', 'eid-2'}
        assert set(df['error_type']) == {'ValueError', 'TypeError', 'KeyError'}
        assert list(df.columns) == LOG_COLUMNS

    def test_filtered_out_session_contributes_nothing(self, tmp_path):
        """The behavior change: the directory walk ignored the group's filters."""
        from tests.test_util import _write_session_h5
        _write_session_h5(tmp_path, 'eid-in', 'mouse_A', 'biased',
                          errors=[ValueError("kept")])
        _write_session_h5(tmp_path, 'eid-out', 'mouse_B', 'training',
                          errors=[TypeError("dropped")])
        group = self._group(tmp_path, {'eid-in': 'biased', 'eid-out': 'training'})
        group.filter_sessions(session_types=('biased',), targetnms=False,
                              qc_blockers=set(), min_performance=False,
                              required_contrasts=False)

        df = group.collect_errors()

        assert set(df['eid']) == {'eid-in'}

    def test_session_without_errors_or_file(self, tmp_path):
        from tests.test_util import _write_session_h5
        _write_session_h5(tmp_path, 'eid-clean', 'mouse_A')
        group = self._group(tmp_path, {'eid-clean': 'biased', 'eid-absent': 'biased'})

        assert len(group.collect_errors()) == 0

    def test_errors_from_product_groups_keep_their_product(self, tmp_path):
        from iblnm.data import PhotometrySession
        series = _collector_catalog({'eid-1': 'biased'}).iloc[0]
        ps = PhotometrySession(series, one=MagicMock(), load_data=False)
        try:
            raise ValueError("bad value")
        except ValueError as e:
            ps.log_error(e, product='photometry/raw')
        ps.save_h5(tmp_path / 'eid-1.h5', groups=['metadata', 'errors'])
        group = self._group(tmp_path, {'eid-1': 'biased'})

        df = group.collect_errors()

        assert len(df) == 1
        assert df.iloc[0]['product'] == 'photometry/raw'
        assert df.iloc[0]['error_type'] == 'ValueError'


class TestGroupFromH5Dir:
    """PhotometrySessionGroup.from_h5_dir rebuilds a catalog from the store."""

    def test_catalog_from_h5_metadata_groups(self, tmp_path):
        from iblnm.data import PhotometrySessionGroup
        from tests.test_util import _write_session_h5
        _write_session_h5(tmp_path, 'eid-1', 'mouse_A', 'biased')
        _write_session_h5(tmp_path, 'eid-2', 'mouse_B', 'training')
        _write_session_h5(tmp_path, 'eid-3', 'mouse_A', 'ephys')

        group = PhotometrySessionGroup.from_h5_dir(tmp_path, one=None)

        assert set(group._catalog['eid']) == {'eid-1', 'eid-2', 'eid-3'}
        assert set(group._catalog['subject']) == {'mouse_A', 'mouse_B'}
        assert group.h5_dir == tmp_path

    def test_every_schema_column_present(self, tmp_path):
        from iblnm.config import SESSION_SCHEMA
        from iblnm.data import PhotometrySessionGroup
        from tests.test_util import _write_session_h5
        _write_session_h5(tmp_path, 'eid-1', 'mouse_A')

        group = PhotometrySessionGroup.from_h5_dir(tmp_path, one=None)

        assert set(SESSION_SCHEMA) <= set(group._catalog.columns)

    def test_empty_directory(self, tmp_path):
        from iblnm.data import PhotometrySessionGroup
        assert len(PhotometrySessionGroup.from_h5_dir(tmp_path, one=None)) == 0

    def test_file_without_metadata_group_skipped(self, tmp_path):
        import h5py
        from iblnm.data import PhotometrySessionGroup
        from tests.test_util import _write_session_h5
        with h5py.File(tmp_path / 'old.h5', 'w') as f:
            f.attrs['eid'] = 'old-eid'
        _write_session_h5(tmp_path, 'eid-1', 'mouse_A')

        group = PhotometrySessionGroup.from_h5_dir(tmp_path, one=None)

        assert list(group._catalog['eid']) == ['eid-1']

    def test_logged_errors_scanned(self, tmp_path):
        """The reconstructed catalog carries the errors its files recorded."""
        from iblnm.data import PhotometrySessionGroup
        from iblnm.validation import MissingRawData
        from tests.test_util import _write_session_h5
        _write_session_h5(tmp_path, 'eid-1', 'mouse_A',
                          errors=[MissingRawData('x')])

        group = PhotometrySessionGroup.from_h5_dir(tmp_path, one=None)

        assert group._catalog['logged_errors'].iloc[0] == ['MissingRawData']

    def test_day_and_session_ranks_derived(self, tmp_path):
        """Both per-subject rankings come off the store, with no fixup step."""
        from iblnm.data import PhotometrySessionGroup
        from tests.test_util import _write_session_h5
        _write_session_h5(tmp_path, 'eid-1', 'mouse_A',
                          start_time='2024-01-01T10:00:00')
        _write_session_h5(tmp_path, 'eid-2', 'mouse_A',
                          start_time='2024-01-03T10:00:00')

        group = PhotometrySessionGroup.from_h5_dir(tmp_path, one=None,
                                                   scan_h5=False)

        assert list(group.sessions['day_n']) == [0, 2]
        assert list(group.sessions['session_n']) == [1, 2]

    def test_scan_skipped(self, tmp_path):
        """scan_h5=False leaves the columns complete_catalog would have filled."""
        from iblnm.data import PhotometrySessionGroup
        from iblnm.validation import MissingRawData
        from tests.test_util import _write_session_h5
        _write_session_h5(tmp_path, 'eid-1', 'mouse_A',
                          errors=[MissingRawData('x')])

        group = PhotometrySessionGroup.from_h5_dir(tmp_path, one=None,
                                                   scan_h5=False)

        assert list(group._catalog['logged_errors']) == [[]]


class TestGroupFixCatalog:
    """fix_catalog repairs the table the group filters and builds from."""

    def test_fixups_reach_the_group(self):
        """The fixed regions and the derived target_NM land on the catalog.

        `group.sessions` returns a copy, so a script cannot repair the table the
        group iterates without this method. Asserting on `group.sessions` after
        the call proves the fix reached `_catalog` and not a detached frame.
        """
        from iblnm.data import PhotometrySessionGroup
        catalog = pd.DataFrame({
            'eid': ['a', 'b'],
            'subject': ['M1', 'M1'],
            'start_time': ['2024-01-01T10:00:00', '2024-01-03T10:00:00'],
            'brain_region': [['SNC'], []],
            'hemisphere': [['l'], []],
            # The store's metadata carries target_NM, and from_catalog drops
            # rows whose parallel columns disagree in length before the fixups
            # ever run — so the populated row needs its (stale) entry here.
            'target_NM': [['SNC-DA'], []],
        })
        group = PhotometrySessionGroup.from_catalog(catalog, one=None,
                                                    h5_dir=None)

        group.fix_catalog()

        assert list(group.sessions['brain_region']) == [['SNc'], ['SNc']]
        assert list(group.sessions['target_NM']) == [['SNc-DA'], ['SNc-DA']]


class TestGroupCollectSessionErrors:
    """collect_session_errors feeds filter_sessions, so it reads the catalog."""

    def _group(self, tmp_path, session_types):
        from iblnm.data import PhotometrySessionGroup
        return PhotometrySessionGroup.from_catalog(
            _collector_catalog(session_types), one=None, h5_dir=tmp_path,
            scan_h5=False)

    def test_error_types_per_eid(self, tmp_path):
        from iblnm.validation import MissingRawData, InvalidStrain
        from tests.test_util import _write_session_h5
        _write_session_h5(tmp_path, 'eid-1', 'mouse_A',
                          errors=[MissingRawData('x'), InvalidStrain('y')])
        _write_session_h5(tmp_path, 'eid-2', 'mouse_B')
        group = self._group(tmp_path, {'eid-1': 'biased', 'eid-2': 'biased'})

        result = group.collect_session_errors().set_index('eid')

        assert set(result.loc['eid-1', 'logged_errors']) == {
            'MissingRawData', 'InvalidStrain'}
        assert result.loc['eid-2', 'logged_errors'] == []

    def test_row_per_catalogued_eid_including_filtered_and_missing(self, tmp_path):
        """Every catalog row gets a row: the mask this feeds does not exist yet."""
        from iblnm.validation import MissingRawData
        from tests.test_util import _write_session_h5
        _write_session_h5(tmp_path, 'eid-1', 'mouse_A', 'training',
                          errors=[MissingRawData('x')])
        group = self._group(tmp_path, {'eid-1': 'training', 'eid-absent': 'biased'})
        group.filter_sessions(session_types=('biased',), targetnms=False,
                              qc_blockers=set(), min_performance=False,
                              required_contrasts=False)

        result = group.collect_session_errors()

        assert list(result['eid']) == ['eid-1', 'eid-absent']
        assert result.set_index('eid').loc['eid-absent', 'logged_errors'] == []

    def test_duplicate_errors_deduplicated(self, tmp_path):
        from iblnm.validation import MissingRawData
        from tests.test_util import _write_session_h5
        _write_session_h5(tmp_path, 'eid-1', 'mouse_A',
                          errors=[MissingRawData('x'), MissingRawData('x')])
        group = self._group(tmp_path, {'eid-1': 'biased'})

        result = group.collect_session_errors()

        assert result.iloc[0]['logged_errors'] == ['MissingRawData']

    def test_joins_the_column_filter_sessions_reads(self, tmp_path):
        from iblnm.validation import MissingRawData
        from tests.test_util import _write_session_h5
        _write_session_h5(tmp_path, 'eid-1', 'mouse_A',
                          errors=[MissingRawData('x')])
        _write_session_h5(tmp_path, 'eid-2', 'mouse_B')
        group = self._group(tmp_path, {'eid-1': 'biased', 'eid-2': 'biased'})

        group.collect_session_errors()
        group.filter_sessions(session_types=False, targetnms=False,
                              qc_blockers={'MissingRawData'},
                              min_performance=False, required_contrasts=False)

        assert set(group.sessions['eid']) == {'eid-2'}


def store_photometry_qc(h5_dir, eid, qc_by_region):
    """Write one session's `photometry/{region}/raw/qc` groups."""
    from iblnm.data import PhotometrySession
    ps = PhotometrySession(pd.Series({
        'eid': eid, 'subject': f'mouse_{eid}', 'number': 1,
        'start_time': '2024-01-01T10:00:00', 'session_type': 'biased',
    }), one=None, load_data=False)
    ps.filepath = h5_dir / f'{eid}.h5'
    ps.photometry_qc = qc_by_region
    ps.save_h5(groups=['photometry'])


def qc_group(h5_dir, regions_by_eid):
    """Group over a catalog of `eid -> [brain_region, ...]`, reading `h5_dir`."""
    from iblnm.data import PhotometrySessionGroup
    catalog = pd.DataFrame([
        {'eid': eid, 'subject': f'mouse_{eid}', 'session_type': 'biased',
         'start_time': '2024-01-01T10:00:00', 'number': 1,
         'brain_region': regions, 'hemisphere': ['l'] * len(regions),
         'target_NM': [f'{r}-DA' for r in regions], 'NM': 'DA'}
        for eid, regions in regions_by_eid.items()
    ])
    return PhotometrySessionGroup.from_catalog(
        catalog, one=None, h5_dir=h5_dir, scan_h5=False)


class TestGroupCollectQc:
    """collect_qc reports every catalogued recording's stored raw-QC metrics."""

    def test_one_row_per_recording_with_band_suffixed_metrics(self, tmp_path):
        store_photometry_qc(tmp_path, 'eid-1', {
            'VTA': {'n_unique_samples_GCaMP': 0.5,
                    'n_unique_samples_Isosbestic': 0.4},
            'SNc': {'n_unique_samples_GCaMP': 0.002,
                    'n_unique_samples_Isosbestic': 0.3}})
        group = qc_group(tmp_path, {'eid-1': ['VTA', 'SNc']})

        df = group.collect_qc().set_index('brain_region')

        assert set(df.index) == {'VTA', 'SNc'}
        assert set(df['eid']) == {'eid-1'}
        assert df.loc['VTA', 'n_unique_samples_GCaMP'] == 0.5
        assert df.loc['SNc', 'n_unique_samples_Isosbestic'] == 0.3

    def test_recording_without_stored_qc_is_nan(self, tmp_path):
        """No value to compare, so the threshold in ticket 18 fails it."""
        store_photometry_qc(tmp_path, 'eid-1',
                            {'VTA': {'n_unique_samples_GCaMP': 0.5}})
        group = qc_group(tmp_path, {'eid-1': ['VTA', 'SNc'], 'eid-2': ['DR']})

        df = group.collect_qc().set_index(['eid', 'brain_region'])

        assert np.isnan(df.loc[('eid-1', 'SNc'), 'n_unique_samples_GCaMP'])
        assert np.isnan(df.loc[('eid-2', 'DR'), 'n_unique_samples_GCaMP'])

    def test_reads_the_catalog_not_the_filtered_view(self, tmp_path):
        """It feeds filter_sessions, so it runs before the mask exists."""
        store_photometry_qc(tmp_path, 'eid-1',
                            {'VTA': {'n_unique_samples_GCaMP': 0.5}})
        store_photometry_qc(tmp_path, 'eid-2',
                            {'DR': {'n_unique_samples_GCaMP': 0.1}})
        group = qc_group(tmp_path, {'eid-1': ['VTA'], 'eid-2': ['DR']})
        group.filter_sessions(session_types=False, targetnms=['VTA-DA'],
                              qc_blockers=set(), min_performance=False,
                              required_contrasts=False)

        assert set(group.collect_qc()['eid']) == {'eid-1', 'eid-2'}

    def test_empty_catalog(self, tmp_path):
        group = qc_group(tmp_path, {})
        assert list(group.collect_qc().columns) == ['eid', 'brain_region']


THRESHOLDS = {'n_unique_samples_GCaMP': ('>=', 0.005),
              'n_unique_samples_Isosbestic': ('>=', 0.005)}


class TestFilterPhotometryQc:
    """filter_sessions(photometry_qc=...) drops recordings, not sessions."""

    @staticmethod
    def _filter(group, photometry_qc=THRESHOLDS):
        group.filter_sessions(session_types=False, targetnms=False,
                              qc_blockers=set(), min_performance=False,
                              required_contrasts=False,
                              photometry_qc=photometry_qc)

    def test_failing_region_dropped_session_kept(self, tmp_path):
        """A session keeps the regions that pass and loses the ones that fail."""
        store_photometry_qc(tmp_path, 'eid-1', {
            'VTA': {'n_unique_samples_GCaMP': 0.5,
                    'n_unique_samples_Isosbestic': 0.4},
            'SNc': {'n_unique_samples_GCaMP': 0.002,
                    'n_unique_samples_Isosbestic': 0.3}})
        group = qc_group(tmp_path, {'eid-1': ['VTA', 'SNc']})

        self._filter(group)

        assert list(group.recordings['brain_region']) == ['VTA']
        assert list(group.sessions['eid']) == ['eid-1']

    def test_one_failing_band_fails_the_recording(self, tmp_path):
        store_photometry_qc(tmp_path, 'eid-1', {
            'VTA': {'n_unique_samples_GCaMP': 0.5,
                    'n_unique_samples_Isosbestic': 0.001}})
        group = qc_group(tmp_path, {'eid-1': ['VTA']})

        self._filter(group)

        assert len(group.recordings) == 0

    def test_recording_without_stored_qc_is_dropped(self, tmp_path):
        """No value to compare against, so the threshold fails it."""
        store_photometry_qc(tmp_path, 'eid-1', {
            'VTA': {'n_unique_samples_GCaMP': 0.5,
                    'n_unique_samples_Isosbestic': 0.4}})
        group = qc_group(tmp_path, {'eid-1': ['VTA', 'SNc'], 'eid-2': ['DR']})

        self._filter(group)

        assert list(zip(group.recordings['eid'],
                        group.recordings['brain_region'])) == [('eid-1', 'VTA')]

    def test_skipped_when_False(self, tmp_path):
        store_photometry_qc(tmp_path, 'eid-1', {
            'VTA': {'n_unique_samples_GCaMP': 0.5,
                    'n_unique_samples_Isosbestic': 0.4},
            'SNc': {'n_unique_samples_GCaMP': 0.002,
                    'n_unique_samples_Isosbestic': 0.3}})
        group = qc_group(tmp_path, {'eid-1': ['VTA', 'SNc']})

        self._filter(group, photometry_qc=False)

        assert list(group.recordings['brain_region']) == ['VTA', 'SNc']

    def test_printout_counts_dropped_recordings(self, tmp_path, capsys):
        store_photometry_qc(tmp_path, 'eid-1', {
            'VTA': {'n_unique_samples_GCaMP': 0.5,
                    'n_unique_samples_Isosbestic': 0.4},
            'SNc': {'n_unique_samples_GCaMP': 0.002,
                    'n_unique_samples_Isosbestic': 0.3}})
        group = qc_group(tmp_path, {'eid-1': ['VTA', 'SNc']})

        self._filter(group)

        printed = capsys.readouterr().out
        assert '-   1 photometry_qc (recordings)' in printed


# Level written into every (label, event, window) cell the pose rollup must not
# read; picked far from any test level so a mis-selection cannot pass.
DECOY_LEVEL = -999.0


def _write_pose_session(h5_dir, eid, steps, drift, peak_lags, qc_lp,
                        series, baselines=None, trials=None, functions=None,
                        length_discrepancy=np.nan, framerate_from_tpts=np.nan):
    """Write an H5 with metadata + video groups carrying known responses.

    ``steps`` maps movement label -> the post-event level of its own event cell
    (``LABEL2EVENT``); NaN yields an all-NaN cell (expected to collect to a NaN
    scalar). ``baselines`` maps label -> the pre-event level of its stimOn cell
    (default 0). Every other (label, event, window) combination is filled with
    ``DECOY_LEVEL``, so a collected scalar of ``step - baseline`` can only come
    from selecting the right two cells and windows. Pass ``steps=None`` to write
    a video group with `video/times/qc` but no responses (LP-absent case).
    ``trials``, when given, maps ``stimOnTrigger_times`` / ``feedback_times`` to 1D
    arrays written as flat datasets under a ``trials`` group. ``functions``,
    when given, is the (3, n_lags) xcorr array; defaults to zeros.

    The eight Alyx QC labels are not written: they are fetched live and handed
    to ``collect_pose`` as its ``video_qc`` argument.
    """
    from iblnm.config import LABEL2EVENT, MOVEMENT_EVENTS
    from iblnm.data import PhotometrySession
    baselines = baselines or {}
    series = series.copy()
    series['eid'] = eid
    ps = PhotometrySession(series, one=MagicMock(), load_data=False)
    ps.video_manual_qc = {'qc_lp': qc_lp}
    ps.video_times_qc = {'length_discrepancy': length_discrepancy,
                         'framerate_from_tpts': framerate_from_tpts}

    if steps is not None:
        time = np.linspace(-0.5, 0.5, 101)
        n_trial = 3

        def _step_cells(label):
            """(n_event, n_trial, n_time) grid; only the cells the collector is
            meant to read carry the label's levels, the rest carry DECOY_LEVEL."""
            cells = [
                np.where(
                    time < 0,
                    baselines.get(label, 0.0) if event == 'stimOnTrigger_times'
                    else DECOY_LEVEL,
                    steps[label] if event == LABEL2EVENT[label] else DECOY_LEVEL,
                )
                for event in MOVEMENT_EVENTS
            ]
            return np.broadcast_to(np.stack(cells)[:, None, :],
                                   (len(MOVEMENT_EVENTS), n_trial, time.size))

        ps.movement_responses = {
            label: xr.DataArray(
                _step_cells(label), dims=['event', 'trial', 'time'],
                coords={'event': list(MOVEMENT_EVENTS),
                        'trial': np.arange(n_trial), 'time': time},
            )
            for label in steps
        }
        ps.pose_xcorr = {
            'functions': np.zeros((3, 11)) if functions is None else np.asarray(functions),
            'lags': np.linspace(-5, 5, 11),
            'peak_lags': np.asarray(peak_lags),
            'drift': drift,
        }
    ps.save_h5(h5_dir / f'{eid}.h5', groups=['metadata', 'video'])

    if trials is not None:
        import h5py
        with h5py.File(h5_dir / f'{eid}.h5', 'a') as f:
            grp = f.create_group('trials/table')
            for key, values in trials.items():
                grp.create_dataset(key, data=np.asarray(values))


def _write_errors(h5_dir, eid, error_types):
    """Append an ``errors`` group listing ``error_types`` to ``{eid}.h5``.

    Creates the H5 if absent, so the bare-row case (errors but no ``video``
    group) can be exercised.
    """
    import h5py
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


class TestGroupCollectPose:
    """collect_pose rolls the filtered sessions' video groups into one table."""

    def _group(self, h5_dir, session_types):
        from iblnm.data import PhotometrySessionGroup
        return PhotometrySessionGroup.from_catalog(
            _collector_catalog(session_types), one=None, h5_dir=h5_dir,
            scan_h5=False)

    def test_rollup_two_sessions(self, tmp_path, mock_session_series):
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
        group = self._group(tmp_path, {'eid-a': 'biased', 'eid-b': 'biased'})

        df = group.collect_pose()

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

    def test_filtered_out_session_excluded(self, tmp_path, mock_session_series):
        """The behavior change: the directory walk ignored the group's filters."""
        steps = {'paw': 1.0, 'nose': 2.0, 'tongue_speed': 3.0,
                 'tongue_likelihood': 0.5}
        for eid in ('eid-in', 'eid-out'):
            _write_pose_session(tmp_path, eid, steps, drift=0.1,
                                peak_lags=[0.0, 0.0, 0.0], qc_lp='PASS',
                                series=mock_session_series)
        group = self._group(tmp_path, {'eid-in': 'biased', 'eid-out': 'training'})
        group.filter_sessions(session_types=('biased',), targetnms=False,
                              qc_blockers=set(), min_performance=False,
                              required_contrasts=False)

        assert list(group.collect_pose()['eid']) == ['eid-in']

    def test_peak_values_from_xcorr_functions(self, tmp_path, mock_session_series):
        steps = {'paw': 1.0, 'nose': 2.0, 'tongue_speed': 3.0,
                 'tongue_likelihood': 0.5}
        # peak of each third's function = its max: 0.2, 0.7, 0.4
        functions = np.array([
            [0.1, 0.2, -0.3], [0.7, 0.1, 0.0], [0.4, -0.5, 0.2]])
        _write_pose_session(tmp_path, 'eid-pk', steps, drift=0.1,
                            peak_lags=[0.0, 0.0, 0.0], qc_lp='PASS',
                            series=mock_session_series, functions=functions)
        group = self._group(tmp_path, {'eid-pk': 'biased'})

        df = group.collect_pose().set_index('eid')

        np.testing.assert_allclose(
            [df.loc['eid-pk', 'peak_val_early'], df.loc['eid-pk', 'peak_val_mid'],
             df.loc['eid-pk', 'peak_val_late']], [0.2, 0.7, 0.4])

    def test_mean_rt_from_trials_group(self, tmp_path, mock_session_series):
        steps = {'paw': 1.0, 'nose': 2.0, 'tongue_speed': 3.0,
                 'tongue_likelihood': 0.5}
        trials = {'stimOnTrigger_times': [0.0, 0.0, 0.0],
                  'feedback_times': [0.5, 1.0, np.nan]}
        _write_pose_session(tmp_path, 'eid-rt', steps, drift=0.1,
                            peak_lags=[0.0, 0.0, 0.0], qc_lp='PASS',
                            series=mock_session_series, trials=trials)
        group = self._group(tmp_path, {'eid-rt': 'biased'})

        df = group.collect_pose().set_index('eid')

        np.testing.assert_allclose(df.loc['eid-rt', 'mean_rt'], 0.75)

    def test_mean_rt_nan_without_trials_group(self, tmp_path, mock_session_series):
        steps = {'paw': 1.0, 'nose': 2.0, 'tongue_speed': 3.0,
                 'tongue_likelihood': 0.5}
        _write_pose_session(tmp_path, 'eid-nort', steps, drift=0.1,
                            peak_lags=[0.0, 0.0, 0.0], qc_lp='PASS',
                            series=mock_session_series)
        group = self._group(tmp_path, {'eid-nort': 'biased'})

        df = group.collect_pose().set_index('eid')

        assert np.isnan(df.loc['eid-nort', 'mean_rt'])
        assert np.isfinite(df.loc['eid-nort', 'paw'])

    def test_session_type_from_the_catalog(self, tmp_path, mock_session_series):
        """The catalog is the source of truth, not the stored metadata group."""
        steps = {'paw': 1.0, 'nose': 2.0, 'tongue_speed': 3.0,
                 'tongue_likelihood': 0.5}
        for eid in ('eid-bi', 'eid-tr'):
            _write_pose_session(tmp_path, eid, steps, drift=0.1,
                                peak_lags=[0.0, 0.0, 0.0], qc_lp='PASS',
                                series=mock_session_series)
        group = self._group(tmp_path, {'eid-bi': 'biased', 'eid-tr': 'training'})

        df = group.collect_pose().set_index('eid')

        assert df.loc['eid-bi', 'session_type'] == 'biased'
        assert df.loc['eid-tr', 'session_type'] == 'training'

    def test_collects_qc_timing(self, tmp_path, mock_session_series):
        import h5py
        steps = {'paw': 1.0, 'nose': 2.0, 'tongue_speed': 3.0,
                 'tongue_likelihood': 0.5}
        _write_pose_session(tmp_path, 'eid-t', steps, drift=0.1,
                            peak_lags=[0.0, 0.0, 0.0], qc_lp='PASS',
                            series=mock_session_series)
        with h5py.File(tmp_path / 'eid-t.h5', 'a') as f:
            f['video/manual_qc'].attrs['qc_timing'] = 'WARNING'
        group = self._group(tmp_path, {'eid-t': 'biased'})

        df = group.collect_pose().set_index('eid')

        assert df.loc['eid-t', 'qc_timing'] == 'WARNING'

    def test_performance_joined_from_the_stored_product(self, tmp_path,
                                                        mock_session_series):
        from iblnm.data import PhotometrySession
        steps = {'paw': 1.0, 'nose': 2.0, 'tongue_speed': 3.0,
                 'tongue_likelihood': 0.5}
        for eid in ('eid-a', 'eid-b'):
            _write_pose_session(tmp_path, eid, steps, drift=0.1,
                                peak_lags=[0.0, 0.0, 0.0], qc_lp='PASS',
                                series=mock_session_series)
        scored = PhotometrySession(pd.Series({
            'eid': 'eid-a', 'subject': 'mouse_A', 'number': 1,
            'start_time': '2024-01-01T10:00:00', 'session_type': 'biased',
        }), one=None, load_data=False)
        scored.filepath = tmp_path / 'eid-a.h5'
        scored.trials = _percent_contrast_trials([0.0, 100.0],
                                                 fraction_correct=0.8)
        scored.load_performance()
        group = self._group(tmp_path, {'eid-a': 'biased', 'eid-b': 'biased'})

        df = group.collect_pose().set_index('eid')

        assert df.loc['eid-a', 'fraction_correct'] == pytest.approx(0.8)
        assert pd.isna(df.loc['eid-b', 'fraction_correct'])

    def test_all_nan_trace_yields_nan_scalar(self, tmp_path, mock_session_series):
        steps = {'paw': 1.0, 'nose': 2.0, 'tongue_speed': np.nan,
                 'tongue_likelihood': 0.5}
        _write_pose_session(tmp_path, 'eid-c', steps, drift=0.1,
                            peak_lags=[0.0, 0.0, 0.0], qc_lp='NOT_SET',
                            series=mock_session_series)
        group = self._group(tmp_path, {'eid-c': 'biased'})

        df = group.collect_pose().set_index('eid')

        assert np.isnan(df.loc['eid-c', 'tongue_speed'])
        assert np.isfinite(df.loc['eid-c', 'paw'])

    def test_video_qc_score_from_fetched_quality_cols(self, tmp_path,
                                                      mock_session_series):
        """Traces + motion energy + clean QC: lp_exists, finite ME, scored QC."""
        from iblnm.config import QCVAL2NUM, VIDEO_QC_COLS, VIDEO_QC_QUALITY_COLS
        steps = {'paw': 1.0, 'nose': 2.0, 'tongue_speed': 3.0,
                 'tongue_likelihood': 0.5, 'motion_energy': 4.0}
        clean_qc = {col: 'PASS' for col in VIDEO_QC_COLS}
        _write_pose_session(tmp_path, 'eid-q', steps, drift=0.1,
                            peak_lags=[0.0, 0.0, 0.0], qc_lp='PASS',
                            series=mock_session_series)
        group = self._group(tmp_path, {'eid-q': 'biased'})

        df = group.collect_pose(video_qc={'eid-q': clean_qc}).set_index('eid')

        assert df.loc['eid-q', 'lp_exists']
        assert np.isfinite(df.loc['eid-q', 'motion_energy'])
        expected = np.nanmean([QCVAL2NUM['PASS']] * len(VIDEO_QC_QUALITY_COLS))
        np.testing.assert_allclose(df.loc['eid-q', 'video_qc_score'], expected)

    def test_not_set_excluded_from_quality_score(self, tmp_path,
                                                 mock_session_series):
        """NOT_SET means the check never ran, so it must not weigh on the score."""
        from iblnm.config import QCVAL2NUM, VIDEO_QC_COLS
        qc = {col: 'PASS' for col in VIDEO_QC_COLS}
        qc['qc_videoLeft_wheel_alignment'] = 'NOT_SET'
        _write_pose_session(tmp_path, 'eid-ns', steps=None, drift=np.nan,
                            peak_lags=None, qc_lp='NOT_SET',
                            series=mock_session_series)
        group = self._group(tmp_path, {'eid-ns': 'biased'})

        df = group.collect_pose(video_qc={'eid-ns': qc}).set_index('eid')

        np.testing.assert_allclose(df.loc['eid-ns', 'video_qc_score'],
                                   QCVAL2NUM['PASS'])

    def test_all_not_set_quality_scores_nan(self, tmp_path, mock_session_series):
        from iblnm.config import VIDEO_QC_COLS, VIDEO_QC_QUALITY_COLS
        qc = {col: 'NOT_SET' if col in VIDEO_QC_QUALITY_COLS else 'PASS'
              for col in VIDEO_QC_COLS}
        _write_pose_session(tmp_path, 'eid-allns', steps=None, drift=np.nan,
                            peak_lags=None, qc_lp='NOT_SET',
                            series=mock_session_series)
        group = self._group(tmp_path, {'eid-allns': 'biased'})

        df = group.collect_pose(video_qc={'eid-allns': qc}).set_index('eid')

        assert np.isnan(df.loc['eid-allns', 'video_qc_score'])

    def test_lp_absent_row_present_with_nan_traces(self, tmp_path,
                                                   mock_session_series):
        """Video group with measures + QC but no traces: row present, scored."""
        from iblnm.config import QCVAL2NUM, VIDEO_QC_COLS, VIDEO_QC_QUALITY_COLS
        qc = {col: 'PASS' for col in VIDEO_QC_COLS}
        _write_pose_session(tmp_path, 'eid-nolp', steps=None, drift=np.nan,
                            peak_lags=None, qc_lp='NOT_SET',
                            series=mock_session_series,
                            length_discrepancy=12.0, framerate_from_tpts=30.0)
        group = self._group(tmp_path, {'eid-nolp': 'biased'})

        df = group.collect_pose(video_qc={'eid-nolp': qc}).set_index('eid')

        assert not df.loc['eid-nolp', 'lp_exists']
        assert np.isnan(df.loc['eid-nolp', 'paw'])
        assert df.loc['eid-nolp', 'length_discrepancy'] == 12.0
        expected = np.nanmean([QCVAL2NUM['PASS']] * len(VIDEO_QC_QUALITY_COLS))
        np.testing.assert_allclose(df.loc['eid-nolp', 'video_qc_score'], expected)

    def test_disqualifying_error_forces_score_minus_one(self, tmp_path,
                                                        mock_session_series):
        from iblnm.config import VIDEO_QC_COLS
        steps = {'paw': 1.0, 'nose': 2.0, 'tongue_speed': 3.0,
                 'tongue_likelihood': 0.5}
        clean_qc = {col: 'PASS' for col in VIDEO_QC_COLS}
        _write_pose_session(tmp_path, 'eid-err', steps, drift=0.1,
                            peak_lags=[0.0, 0.0, 0.0], qc_lp='PASS',
                            series=mock_session_series)
        _write_errors(tmp_path, 'eid-err', ['VideoLengthError'])
        group = self._group(tmp_path, {'eid-err': 'biased'})

        df = group.collect_pose(video_qc={'eid-err': clean_qc}).set_index('eid')

        assert df.loc['eid-err', 'video_qc_score'] == -1

    def test_failing_problem_label_forces_score_minus_one(self, tmp_path,
                                                          mock_session_series):
        """The three problem flags disqualify from the live labels, not from H5."""
        from iblnm.config import VIDEO_QC_COLS
        steps = {'paw': 1.0, 'nose': 2.0, 'tongue_speed': 3.0,
                 'tongue_likelihood': 0.5}
        qc = {col: 'PASS' for col in VIDEO_QC_COLS}
        qc['qc_videoLeft_pin_state'] = 'FAIL'
        _write_pose_session(tmp_path, 'eid-pin', steps, drift=0.1,
                            peak_lags=[0.0, 0.0, 0.0], qc_lp='PASS',
                            series=mock_session_series)
        group = self._group(tmp_path, {'eid-pin': 'biased'})

        df = group.collect_pose(video_qc={'eid-pin': qc}).set_index('eid')

        assert df.loc['eid-pin', 'video_qc_score'] == -1

    def test_not_set_problem_label_forces_score_minus_one(self, tmp_path,
                                                          mock_session_series):
        """NOT_SET is not PASS: an unrun problem check disqualifies the session."""
        from iblnm.config import VIDEO_QC_COLS
        qc = {col: 'PASS' for col in VIDEO_QC_COLS}
        qc['qc_videoLeft_timestamps'] = 'NOT_SET'
        _write_pose_session(tmp_path, 'eid-tsns', steps=None, drift=np.nan,
                            peak_lags=None, qc_lp='NOT_SET',
                            series=mock_session_series)
        group = self._group(tmp_path, {'eid-tsns': 'biased'})

        df = group.collect_pose(video_qc={'eid-tsns': qc}).set_index('eid')

        assert df.loc['eid-tsns', 'video_qc_score'] == -1

    def test_missing_timestamps_no_video_group_emits_bare_row(
            self, tmp_path, mock_session_series):
        _write_errors(tmp_path, 'eid-bare', ['MissingVideoTimestamps'])
        group = self._group(tmp_path, {'eid-bare': 'habituation'})

        df = group.collect_pose().set_index('eid')

        assert 'eid-bare' in df.index
        assert df.loc['eid-bare', 'session_type'] == 'habituation'
        assert df.loc['eid-bare', 'video_qc_score'] == -1
        assert np.isnan(df.loc['eid-bare', 'paw'])
        assert not df.loc['eid-bare', 'lp_exists']

    def test_session_without_video_or_blocking_error_is_absent(
            self, tmp_path, mock_session_series):
        """Nothing to report: no group, and the clock is not what failed."""
        _write_errors(tmp_path, 'eid-quiet', ['MissingLP'])
        steps = {'paw': 1.0, 'nose': 2.0, 'tongue_speed': 3.0,
                 'tongue_likelihood': 0.5}
        _write_pose_session(tmp_path, 'eid-ok', steps, drift=0.1,
                            peak_lags=[0.0, 0.0, 0.0], qc_lp='PASS',
                            series=mock_session_series)
        group = self._group(tmp_path, {'eid-quiet': 'biased', 'eid-ok': 'biased'})

        assert list(group.collect_pose()['eid']) == ['eid-ok']


def _percent_contrast_trials(contrasts, fraction_correct, n_per_contrast=20):
    """Trials presenting `contrasts` (percent) at a known fraction correct.

    Contrast levels are stored as percent, the units `REQUIRED_CONTRASTS` is
    written in. Sides alternate; the first `fraction_correct` of the trials
    choose the stimulus side and the rest choose against it.
    """
    contrast = np.repeat(np.asarray(contrasts, dtype=float), n_per_contrast)
    n = len(contrast)
    side = np.tile([-1.0, 1.0], n // 2)
    choice = np.where(np.arange(n) < fraction_correct * n, side, -side)
    return pd.DataFrame({
        'contrast': contrast,
        'signed_contrast': contrast * side,
        'contrastLeft': np.where(side < 0, contrast / 100, np.nan),
        'contrastRight': np.where(side > 0, contrast / 100, np.nan),
        'choice': choice,
        'feedbackType': np.where(choice == side, 1.0, -1.0),
        'probabilityLeft': np.full(n, 0.5),
    })


def _store_performance(h5_dir, eid, trials):
    """Write one session's `trials/performance` product into `h5_dir`."""
    from iblnm.data import PhotometrySession
    ps = PhotometrySession(pd.Series({
        'eid': eid, 'subject': f'mouse_{eid}', 'number': 1,
        'start_time': '2024-01-01T10:00:00', 'session_type': 'training',
    }), one=None, load_data=False)
    ps.filepath = h5_dir / f'{eid}.h5'
    ps.trials = trials
    return ps.load_performance()


def _performance_group(h5_dir, **kwargs):
    """Group over one passing and one failing session, products stored."""
    from iblnm.data import PhotometrySessionGroup
    _store_performance(h5_dir, 'eid-pass', _percent_contrast_trials(
        sorted(REQUIRED_CONTRASTS), fraction_correct=0.9))
    _store_performance(h5_dir, 'eid-fail', _percent_contrast_trials(
        [0.0, 100.0], fraction_correct=0.4))
    catalog = pd.DataFrame([
        {'eid': eid, 'subject': f'mouse_{eid}', 'session_type': 'training',
         'start_time': '2024-01-01T10:00:00', 'number': 1,
         'brain_region': ['VTA'], 'hemisphere': ['l'],
         'target_NM': ['VTA-DA'], 'NM': 'DA'}
        for eid in ('eid-pass', 'eid-fail')
    ])
    return PhotometrySessionGroup.from_catalog(
        catalog, one=None, h5_dir=h5_dir, **kwargs)


def _rescan(catalog, h5_dir):
    """Build a group over `catalog`, scanning `h5_dir` again from scratch."""
    from iblnm.data import PhotometrySessionGroup
    return PhotometrySessionGroup.from_catalog(
        catalog.drop(columns=['fraction_correct', 'contrasts']),
        one=None, h5_dir=h5_dir)


class TestGroupLoadPerformance:
    """PhotometrySessionGroup.load_performance reads the stored product."""

    def test_returns_every_metric_per_session(self, tmp_path):
        """The full per-session table, whatever the catalog carries."""
        group = _performance_group(tmp_path)
        performance = group.load_performance().set_index('eid')
        assert performance.loc['eid-pass', 'fraction_correct'] == pytest.approx(0.9)
        assert performance.loc['eid-fail', 'contrasts'] == [0.0, 100.0]


class TestCatalogScan:
    """from_catalog completes the catalog with everything its filters read."""

    def test_performance_columns_land_without_load_performance(self, tmp_path):
        """fraction_correct and contrasts come off the scan, per session."""
        group = _performance_group(tmp_path)
        catalog = group._catalog.set_index('eid')
        assert group.performance is None
        assert catalog.loc['eid-pass', 'fraction_correct'] == pytest.approx(0.9)
        assert catalog.loc['eid-pass', 'contrasts'] == sorted(REQUIRED_CONTRASTS)
        assert catalog.loc['eid-fail', 'contrasts'] == [0.0, 100.0]

    def test_filters_drop_the_hand_computed_sessions(self, tmp_path, capsys):
        """min_performance and required_contrasts both bite, and say so."""
        group = _performance_group(tmp_path)
        group.filter_sessions(session_types=False, targetnms=False,
                              qc_blockers=set())
        assert set(group.sessions['eid']) == {'eid-pass'}
        printed = capsys.readouterr().out
        assert '-   1 performance' in printed
        assert '-   1 contrasts' in printed

    def test_session_without_the_product_fails_both_filters(self, tmp_path):
        """No stored performance is not a free pass: NaN and [] fail, not skip."""
        group = _performance_group(tmp_path)
        catalog = pd.concat([group._catalog, group._catalog.iloc[[0]].assign(
            eid='eid-unbuilt')], ignore_index=True)
        rescanned = _rescan(catalog, tmp_path)

        row = rescanned._catalog.set_index('eid').loc['eid-unbuilt']
        assert np.isnan(row['fraction_correct'])
        assert row['contrasts'] == []
        rescanned.filter_sessions(session_types=False, targetnms=False,
                                  qc_blockers=set(), required_contrasts=False)
        assert 'eid-unbuilt' not in set(rescanned.sessions['eid'])
        rescanned.filter_sessions(session_types=False, targetnms=False,
                                  qc_blockers=set(), min_performance=False)
        assert 'eid-unbuilt' not in set(rescanned.sessions['eid'])

    def test_reads_each_session_file_once(self, tmp_path, monkeypatch):
        """Errors, performance and QC come off one open per catalogued file."""
        import iblnm.data as data_module
        group = _performance_group(tmp_path)
        catalog = pd.concat([group._catalog, group._catalog.iloc[[0]].assign(
            eid='eid-third')], ignore_index=True)
        _store_performance(tmp_path, 'eid-third', _percent_contrast_trials(
            [0.0, 100.0], fraction_correct=0.5))

        opens = []
        real_file = data_module.h5py.File
        monkeypatch.setattr(data_module.h5py, 'File', lambda path, *a, **kw: (
            opens.append(str(path)), real_file(path, *a, **kw))[1])
        _rescan(catalog, tmp_path)

        assert sorted(Path(p).stem for p in opens) == [
            'eid-fail', 'eid-pass', 'eid-third']


class TestDeduplicate:
    """Tests for PhotometrySessionGroup.deduplicate."""

    def test_keeps_one_per_subject_day(self):
        """Deduplicate keeps one session per (subject, day_n)."""
        from iblnm.data import PhotometrySessionGroup
        df = pd.DataFrame([
            {'eid': 'e1', 'subject': 'A', 'day_n': 0, 'session_type': 'biased',
             'brain_region': ['VTA'], 'hemisphere': ['l'], 'target_NM': ['VTA-DA'],
             'logged_errors': []},
            {'eid': 'e2', 'subject': 'A', 'day_n': 0, 'session_type': 'biased',
             'brain_region': ['VTA'], 'hemisphere': ['l'], 'target_NM': ['VTA-DA'],
             'logged_errors': ['MissingRawData']},
            {'eid': 'e3', 'subject': 'A', 'day_n': 1, 'session_type': 'biased',
             'brain_region': ['VTA'], 'hemisphere': ['l'], 'target_NM': ['VTA-DA'],
             'logged_errors': []},
        ])
        group = PhotometrySessionGroup(df, one=MagicMock())
        group.deduplicate()
        assert set(group.recordings['eid']) == {'e1', 'e3'}

    def test_keeps_multi_region_recordings(self):
        """Deduplicate preserves all recordings for the kept session."""
        from iblnm.data import PhotometrySessionGroup
        df = pd.DataFrame([
            {'eid': 'e1', 'subject': 'A', 'day_n': 0, 'session_type': 'biased',
             'brain_region': ['VTA', 'SNc'], 'hemisphere': ['l', 'r'],
             'target_NM': ['VTA-DA', 'SNc-DA'], 'logged_errors': []},
            {'eid': 'e2', 'subject': 'A', 'day_n': 0, 'session_type': 'biased',
             'brain_region': ['VTA'], 'hemisphere': ['l'], 'target_NM': ['VTA-DA'],
             'logged_errors': ['MissingRawData']},
        ])
        group = PhotometrySessionGroup(df, one=MagicMock())
        group.deduplicate()
        assert len(group.recordings) == 2
        assert set(group.recordings['brain_region']) == {'VTA', 'SNc'}

    def test_returns_self(self):
        """Deduplicate returns self for chaining."""
        from iblnm.data import PhotometrySessionGroup
        df = pd.DataFrame([
            {'eid': 'e1', 'subject': 'A', 'day_n': 0, 'session_type': 'biased',
             'brain_region': ['VTA'], 'hemisphere': ['l'], 'target_NM': ['VTA-DA'],
             'logged_errors': []},
        ])
        group = PhotometrySessionGroup(df, one=MagicMock())
        result = group.deduplicate()
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ['eid', 'error_type', 'error_message', 'traceback']


class TestGroupProcess:
    """Tests for PhotometrySessionGroup.process."""

    def _make_group_with_h5(self, tmp_path):
        """Create a group backed by H5 files with metadata."""
        from iblnm.data import PhotometrySession, PhotometrySessionGroup

        mock_one = MagicMock()
        for i, eid in enumerate(['eid-0', 'eid-1']):
            series = pd.Series({
                'eid': eid, 'subject': f'subj-{i}',
                'start_time': '2024-01-01T10:00:00', 'number': 1,
                'session_type': 'biased', 'task_protocol': 'biased_protocol',
                'brain_region': ['VTA'], 'hemisphere': ['l'],
                'target_NM': ['VTA-DA'],
            })
            ps = PhotometrySession(series, one=mock_one, load_data=False)
            ps.save_h5(tmp_path / f'{eid}.h5', groups=['metadata', 'errors'])

        recs = _make_recordings_df(n_eids=2, regions_per=1)
        group = PhotometrySessionGroup(recs, one=mock_one, h5_dir=tmp_path)
        return group

    def test_process_collects_results(self, tmp_path):
        """process returns results from each session."""
        group = self._make_group_with_h5(tmp_path)
        results = group.process(lambda ps: ps.eid)
        assert set(results) == {'eid-0', 'eid-1'}

    def test_first_build_session_points_at_the_groups_store(self, tmp_path):
        """A session with no H5 yet writes into the group's directory.

        Nothing is written to `tmp_path` beforehand, so both sessions are
        built from their catalog rows rather than read back.
        """
        from iblnm.data import PhotometrySessionGroup

        group = PhotometrySessionGroup(
            _make_recordings_df(n_eids=2, regions_per=1),
            one=MagicMock(), h5_dir=tmp_path)

        paths = group.process(lambda ps: ps.filepath)

        assert paths == [tmp_path / 'eid-0.h5', tmp_path / 'eid-1.h5']

    def test_process_catches_fatal_errors(self, tmp_path):
        """Fatal errors are caught and logged; processing continues."""
        group = self._make_group_with_h5(tmp_path)
        call_count = 0

        def failing_fn(ps):
            nonlocal call_count
            call_count += 1
            if ps.eid == 'eid-0':
                raise ValueError("intentional failure")
            return ps.eid

        results = group.process(failing_fn)
        assert call_count == 2  # both sessions processed
        assert 'eid-1' in results  # successful result present
        assert any(r is None for r in results)  # failed result is None

    def test_process_writes_errors_to_h5(self, tmp_path):
        """Errors are written to the session's H5 file."""
        import h5py
        group = self._make_group_with_h5(tmp_path)

        def failing_fn(ps):
            raise ValueError("test failure")

        group.process(failing_fn)

        # Check that errors were written to H5
        with h5py.File(tmp_path / 'eid-0.h5', 'r') as f:
            assert 'errors' in f
            assert len(f['errors']['error_type']) > 0

    def test_process_preserves_nonfatal_errors(self, tmp_path):
        """Non-fatal errors logged via ps.log_error are persisted."""
        import h5py
        from iblnm.validation import IncompleteEventTimes
        group = self._make_group_with_h5(tmp_path)

        def fn_with_nonfatal(ps):
            try:
                raise IncompleteEventTimes(['firstMovement_times'])
            except IncompleteEventTimes as e:
                ps.log_error(e)
            return 'ok'

        results = group.process(fn_with_nonfatal)
        assert all(r == 'ok' for r in results)

        # Check non-fatal error was written to H5
        with h5py.File(tmp_path / 'eid-0.h5', 'r') as f:
            error_types = [v.decode() for v in f['errors']['error_type'][:]]
            assert 'IncompleteEventTimes' in error_types

    def _logged_error_types(self, tmp_path):
        """Every error type recorded across both sessions' `errors/` trees."""
        import h5py
        from iblnm.data import read_error_tree

        types = []
        for eid in ['eid-0', 'eid-1']:
            with h5py.File(tmp_path / f'{eid}.h5', 'r') as f:
                types += [entry['error_type'] for entry in read_error_tree(f)]
        return types

    def test_lock_collision_is_retried_at_the_end_of_the_pass(self, tmp_path):
        """A session that hit the file lock once succeeds on the retry pass."""
        group = self._make_group_with_h5(tmp_path)
        seen = set()

        def blocks_once(ps):
            if ps.eid not in seen:
                seen.add(ps.eid)
                raise BlockingIOError("file is locked")
            return ps.eid

        results = group.process(blocks_once)

        assert set(results) == {'eid-0', 'eid-1'}
        assert self._logged_error_types(tmp_path) == []

    def test_persistent_lock_collision_fails_without_being_recorded(self, tmp_path):
        """Retried once, not forever — and a lock is never written to errors/."""
        group = self._make_group_with_h5(tmp_path)
        calls = []

        def always_blocks(ps):
            calls.append(ps.eid)
            raise BlockingIOError("file is locked")

        results = group.process(always_blocks)

        assert results == [None, None]
        assert sorted(calls) == ['eid-0', 'eid-0', 'eid-1', 'eid-1']
        assert self._logged_error_types(tmp_path) == []


class TestPhotometrySessionGroup:

    def test_len_matches_recordings(self):
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=2, regions_per=2)
        group = PhotometrySessionGroup(recs, one=MagicMock())
        assert len(group) == 4

    def test_constructs_without_one(self):
        """A group builds with no ONE connection and still counts recordings."""
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=2, regions_per=2)
        group = PhotometrySessionGroup(recs)
        assert group.one is None
        assert len(group) == 4

    def test_iter_yields_series_and_session(self):
        from iblnm.data import PhotometrySessionGroup, PhotometrySession
        recs = _make_recordings_df(n_eids=1, regions_per=1)
        group = PhotometrySessionGroup(recs, one=MagicMock())
        for rec, ps in group:
            assert isinstance(rec, pd.Series)
            assert isinstance(ps, PhotometrySession)

    def test_getitem_returns_tuple(self):
        from iblnm.data import PhotometrySessionGroup, PhotometrySession
        recs = _make_recordings_df(n_eids=1, regions_per=2)
        group = PhotometrySessionGroup(recs, one=MagicMock())
        rec, ps = group[0]
        assert isinstance(rec, pd.Series)
        assert isinstance(ps, PhotometrySession)
        assert rec['eid'] == 'eid-0'

    def test_iter_deduplicates_sessions_by_eid(self):
        """Two recordings from the same eid should share one PhotometrySession."""
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=1, regions_per=2)
        group = PhotometrySessionGroup(recs, one=MagicMock())
        sessions = [ps for _, ps in group]
        assert sessions[0] is sessions[1]

    def test_filter_returns_subset(self):
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=2, regions_per=2)
        group = PhotometrySessionGroup(recs, one=MagicMock())
        mask = group.recordings['eid'] == 'eid-0'
        subset = group.filter(mask)
        assert len(subset) == 2
        assert all(r['eid'] == 'eid-0' for r, _ in subset)



# =============================================================================
# get_response_vector Tests
# =============================================================================

def _make_session_with_responses(mock_one, n_trials=100, post_event_value=1.0):
    """Create a PhotometrySession with synthetic responses and trials.

    Baseline (t<0) is 0; post-event (t>=0) is post_event_value.
    After baseline subtraction, post-event response = post_event_value.
    """
    import xarray as xr
    from iblnm.data import PhotometrySession

    series = pd.Series({
        'eid': 'test-eid', 'subject': 'mouse1',
        'start_time': '2024-01-01T10:00:00', 'number': 1,
        'task_protocol': 'biased', 'session_type': 'biased',
        'brain_region': ['VTA-r'], 'hemisphere': ['r'],
    })
    ps = PhotometrySession(series, one=mock_one, load_data=False)

    rng = np.random.default_rng(42)
    n_time = 61
    tpts = np.linspace(-1, 1, n_time)
    events = ['stimOnTrigger_times', 'firstMovement_times', 'feedback_times']

    # Baseline = 0, post-event = post_event_value
    data = np.zeros((3, n_trials, n_time))
    post_mask = tpts >= 0
    data[:, :, post_mask] = post_event_value

    ps.photometry_responses = {
        'VTA-r': xr.DataArray(
            data, dims=['event', 'trial', 'time'],
            coords={'event': events,
                    'trial': np.arange(n_trials), 'time': tpts},
        )
    }

    contrasts = np.array([0.0, 0.0625, 0.125, 0.25, 1.0])
    sides = rng.choice(['left', 'right'], n_trials)
    contrast_vals = rng.choice(contrasts, n_trials)
    signed = np.where(sides == 'left', -1, 1) * contrast_vals
    ps.trials = pd.DataFrame({
        'stimOnTrigger_times': np.linspace(10, 10 + n_trials, n_trials),
        'firstMovement_times': np.linspace(10.2, 10.2 + n_trials, n_trials),
        'response_times': np.linspace(11, 11 + n_trials, n_trials),
        'feedback_times': np.linspace(11, 11 + n_trials, n_trials) + 0.0005,
        'signed_contrast': signed,
        'contrast': contrast_vals,
        'stim_side': sides,
        'feedbackType': rng.choice([1, -1], n_trials),
        'choice': rng.choice([-1, 1], n_trials),
        'probabilityLeft': np.full(n_trials, 0.5),
    })
    return ps


class TestGetResponseVector:

    def test_returns_series(self):
        ps = _make_session_with_responses(MagicMock())
        vec = ps.get_response_vector(brain_region='VTA-r', hemisphere='r')
        assert isinstance(vec, pd.Series)

    def test_uses_default_events_only(self):
        """Default events exclude firstMovement."""
        ps = _make_session_with_responses(MagicMock())
        vec = ps.get_response_vector(brain_region='VTA-r', hemisphere='r')
        assert len(vec) > 0
        assert not any('firstMovement' in label for label in vec.index)

    def test_ipsi_contra_labels(self):
        """All contrasts (including zero) have ipsi and contra labels."""
        ps = _make_session_with_responses(MagicMock(), n_trials=200)
        vec = ps.get_response_vector(brain_region='VTA-r', hemisphere='r',
                                     min_trials=1)
        # Non-zero contrasts
        assert 'stimOnTrigger_c0.0625_contra_correct' in vec.index
        assert 'stimOnTrigger_c0.0625_ipsi_correct' in vec.index
        assert 'feedback_c1_contra_incorrect' in vec.index
        assert 'feedback_c1_ipsi_incorrect' in vec.index
        # Zero contrast retains ipsi/contra (side matters for action contingencies)
        assert 'stimOnTrigger_c0_contra_correct' in vec.index
        assert 'stimOnTrigger_c0_ipsi_correct' in vec.index

    def test_custom_events_includes_firstMovement(self):
        """Passing events explicitly can include firstMovement."""
        ps = _make_session_with_responses(MagicMock(), n_trials=200)
        vec = ps.get_response_vector(
            brain_region='VTA-r', hemisphere='r',
            events=['stimOnTrigger_times', 'firstMovement_times', 'feedback_times'],
            min_trials=1,
        )
        assert any('firstMovement' in label for label in vec.index)
        # More features than default (which excludes firstMovement)
        default_vec = ps.get_response_vector(brain_region='VTA-r', hemisphere='r',
                                              min_trials=1)
        assert len(vec) > len(default_vec)

    def test_constant_signal_all_ones(self):
        """Post-event response of 1.0 → all condition means should be 1.0 (ignoring NaN)."""
        ps = _make_session_with_responses(MagicMock(), n_trials=200, post_event_value=1.0)
        vec = ps.get_response_vector(brain_region='VTA-r', hemisphere='r',
                                     min_trials=1)
        finite = vec.dropna()
        assert len(finite) > 0
        np.testing.assert_allclose(finite.values, 1.0, atol=1e-10)

    def test_min_trials_produces_nan(self):
        """Condition with fewer than min_trials should be NaN."""
        # Only 10 trials total → many cells will have < 5 trials
        ps = _make_session_with_responses(MagicMock(), n_trials=10, post_event_value=2.0)
        vec = ps.get_response_vector(brain_region='VTA-r', hemisphere='r',
                                     min_trials=5)
        assert vec.isna().any(), "Some conditions should be NaN with only 10 trials"

    def test_minmax_normalize(self):
        """Min-max normalization should produce values in [0, 1]."""
        import xarray as xr
        from iblnm.data import PhotometrySession

        series = pd.Series({
            'eid': 'test-eid', 'subject': 'mouse1',
            'start_time': '2024-01-01T10:00:00', 'number': 1,
            'task_protocol': 'biased', 'session_type': 'biased',
            'brain_region': ['VTA-r'], 'hemisphere': ['r'],
        })
        ps = PhotometrySession(series, one=MagicMock(), load_data=False)

        n_trials, n_time = 200, 61
        tpts = np.linspace(-1, 1, n_time)
        events = ['stimOnTrigger_times', 'firstMovement_times', 'feedback_times']

        # Baseline (t<0) = 0, post-event varies by event
        rng = np.random.default_rng(0)
        data = np.zeros((3, n_trials, n_time))
        post_mask = tpts >= 0
        data[0, :, :][:, post_mask] = 1.0   # stimOn post-event = 1
        data[1, :, :][:, post_mask] = 2.0   # firstMov post-event = 2
        data[2, :, :][:, post_mask] = 3.0   # feedback post-event = 3

        ps.photometry_responses = {
            'VTA-r': xr.DataArray(
                data, dims=['event', 'trial', 'time'],
                coords={'event': events,
                        'trial': np.arange(n_trials), 'time': tpts},
            )
        }

        contrasts = np.array([0.0, 0.0625, 0.125, 0.25, 1.0])
        sides = rng.choice(['left', 'right'], n_trials)
        contrast_vals = rng.choice(contrasts, n_trials)
        signed = np.where(sides == 'left', -1, 1) * contrast_vals
        ps.trials = pd.DataFrame({
            'stimOnTrigger_times': np.linspace(10, 10 + n_trials, n_trials),
            'firstMovement_times': np.linspace(10.2, 10.2 + n_trials, n_trials),
            'feedback_times': np.linspace(11, 11 + n_trials, n_trials),
            'signed_contrast': signed,
            'contrast': contrast_vals,
            'stim_side': sides,
            'feedbackType': rng.choice([1, -1], n_trials),
            'choice': rng.choice([-1, 1], n_trials),
            'probabilityLeft': np.full(n_trials, 0.5),
        })

        vec = ps.get_response_vector(brain_region='VTA-r', hemisphere='r',
                                     normalize='minmax', min_trials=1)
        finite = vec.dropna()
        assert finite.min() >= -1e-10
        assert finite.max() <= 1.0 + 1e-10
        assert np.isclose(finite.min(), 0.0, atol=1e-10)
        assert np.isclose(finite.max(), 1.0, atol=1e-10)

    def test_invalid_normalize_raises(self):
        ps = _make_session_with_responses(MagicMock())
        with pytest.raises(ValueError, match='normalize'):
            ps.get_response_vector(brain_region='VTA-r', hemisphere='r',
                                   normalize='invalid')

    def test_condition_label_format(self):
        """Labels follow event_cContrast_side_feedback."""
        ps = _make_session_with_responses(MagicMock())
        vec = ps.get_response_vector(brain_region='VTA-r', hemisphere='r')
        assert 'stimOnTrigger_c0_contra_correct' in vec.index
        assert 'stimOnTrigger_c1_ipsi_incorrect' in vec.index
        assert 'feedback_c0.25_contra_correct' in vec.index


# =============================================================================
# PhotometrySessionGroup Analysis Method Tests
# =============================================================================

def _write_h5(path, n_trials=100, regions=('VTA-r',), seed=42,
              all_biased=False, all_nogo=False, fast_response=False):
    """Write a minimal H5 file with trials and responses.

    Parameters
    ----------
    all_biased : bool
        If True, set all probabilityLeft to 0.8 (biased block).
    all_nogo : bool
        If True, set all choice to 0 (no-go).
    fast_response : bool
        If True, set response_times = stimOnTrigger_times + 0.01 (response_time
        < 0.05).
    """
    import h5py

    rng = np.random.default_rng(seed)
    n_time = 61
    tpts = np.linspace(-1, 1, n_time)
    events = ['stimOnTrigger_times', 'firstMovement_times', 'feedback_times']
    contrasts = np.array([0.0, 0.0625, 0.125, 0.25, 1.0])

    # Pre-event = 0, post-event = 1.0
    post_mask = tpts >= 0

    stim_on = np.linspace(10, 10 + n_trials, n_trials)
    response = (stim_on + 0.01 if fast_response
                else np.linspace(11, 11 + n_trials, n_trials))

    with h5py.File(path, 'w') as f:
        grp = f.create_group('trials/table')
        grp.create_dataset('trial', data=np.arange(n_trials))
        grp.create_dataset('stimOnTrigger_times', data=stim_on)
        grp.create_dataset('firstMovement_times',
                           data=stim_on + 0.2)
        grp.create_dataset('response_times', data=response)
        # Feedback delivery lags the choice by the measured correct-trial
        # amount, 0.5 ms.
        grp.create_dataset('feedback_times', data=response + 0.0005)
        sides = rng.choice(['left', 'right'], n_trials)
        contrast_vals = rng.choice(contrasts, n_trials)
        signed = np.where(sides == 'left', -1, 1).astype(float) * contrast_vals
        grp.create_dataset('signed_contrast', data=signed)
        grp.create_dataset('contrast', data=contrast_vals)
        # Store stim_side as fixed-length bytes for HDF5 compatibility
        grp.create_dataset('stim_side', data=np.array(sides, dtype='S5'))
        grp.create_dataset('feedbackType', data=rng.choice([1, -1], n_trials))
        grp.create_dataset('choice', data=np.zeros(n_trials) if all_nogo
                           else rng.choice([-1, 1], n_trials))
        grp.create_dataset('probabilityLeft',
                           data=np.full(n_trials, 0.8) if all_biased
                           else np.full(n_trials, 0.5))

        phot_root = f.create_group('photometry')
        for region in regions:
            region_grp = phot_root.create_group(region)
            resp_grp = region_grp.create_group('responses')
            resp_grp.attrs['fs'] = 30.0
            resp_grp.attrs['response_window'] = (tpts[0], tpts[-1])
            resp_grp.create_dataset('times', data=tpts)
            resp_grp.create_dataset('trials', data=np.arange(n_trials))
            for event in events:
                data = np.zeros((n_trials, n_time))
                data[:, post_mask] = 1.0
                resp_grp.create_dataset(event, data=data)


class TestGetResponseFeatures:

    def test_returns_dataframe_with_correct_index(self, tmp_path):
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=1, regions_per=1)
        _write_h5(tmp_path / 'eid-0.h5', n_trials=200)
        group = PhotometrySessionGroup(recs, one=MagicMock(), h5_dir=tmp_path)
        df = group.get_response_features(min_trials=1)
        assert isinstance(df, pd.DataFrame)
        assert df.index.names == ['eid', 'target_NM', 'fiber_idx']

    def test_stores_response_features(self, tmp_path):
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=1, regions_per=1)
        _write_h5(tmp_path / 'eid-0.h5', n_trials=200)
        group = PhotometrySessionGroup(recs, one=MagicMock(), h5_dir=tmp_path)
        group.get_response_features(min_trials=1)
        assert group.response_features is not None
        assert isinstance(group.response_features, pd.DataFrame)

    def test_multiple_recordings(self, tmp_path):
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=2, regions_per=1)
        _write_h5(tmp_path / 'eid-0.h5', n_trials=200, seed=0)
        _write_h5(tmp_path / 'eid-1.h5', n_trials=200, seed=1)
        group = PhotometrySessionGroup(recs, one=MagicMock(), h5_dir=tmp_path)
        df = group.get_response_features(min_trials=1)
        assert len(df) == 2

    def test_skips_missing_h5(self, tmp_path):
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=2, regions_per=1)
        # Only create H5 for eid-0
        _write_h5(tmp_path / 'eid-0.h5', n_trials=200)
        group = PhotometrySessionGroup(recs, one=MagicMock(), h5_dir=tmp_path)
        df = group.get_response_features(min_trials=1)
        assert len(df) == 1

    def test_discards_raw_data_after_extraction(self, tmp_path):
        """Raw responses should not persist in memory after extraction."""
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=1, regions_per=1)
        _write_h5(tmp_path / 'eid-0.h5', n_trials=200)
        group = PhotometrySessionGroup(recs, one=MagicMock(), h5_dir=tmp_path)
        group.get_response_features(min_trials=1)
        _, ps = group[0]
        assert not hasattr(ps, 'photometry_responses')

    def test_default_min_trials_is_one(self, tmp_path):
        """Default min_trials=1 allows sparse conditions through."""
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=1, regions_per=1)
        # 200 trials: enough to fill most cells, session survives drop_sessions
        _write_h5(tmp_path / 'eid-0.h5', n_trials=200)
        group = PhotometrySessionGroup(recs, one=MagicMock(), h5_dir=tmp_path)
        df = group.get_response_features()
        # Session should survive default drop_sessions with 200 trials
        assert len(df) == 1
        assert df.notna().sum(axis=1).iloc[0] > 0

    def test_drop_sessions_removes_rows_with_nan(self, tmp_path):
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=2, regions_per=1)
        # eid-0: 20 trials → likely has NaN features
        _write_h5(tmp_path / 'eid-0.h5', n_trials=20, seed=0)
        # eid-1: 500 trials → all features populated
        _write_h5(tmp_path / 'eid-1.h5', n_trials=500, seed=1)
        group = PhotometrySessionGroup(recs, one=MagicMock(), h5_dir=tmp_path)
        df = group.get_response_features(nan_handling='drop_sessions')
        # The sparse session should be dropped
        assert df.isna().sum().sum() == 0
        assert len(df) <= 2

    def test_drop_features_removes_sparse_columns(self, tmp_path):
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=3, regions_per=1)
        # Use few trials so some features are frequently NaN
        for i in range(3):
            _write_h5(tmp_path / f'eid-{i}.h5', n_trials=30, seed=i)
        group = PhotometrySessionGroup(recs, one=MagicMock(), h5_dir=tmp_path)
        # First pass: keep all columns (threshold=1.0) to count them
        df_all = group.get_response_features(nan_handling='drop_features',
                                              nan_threshold=1.0)
        n_cols_before = df_all.shape[1]
        nan_rates = df_all.isna().mean()

        # Reset and re-extract with stricter threshold
        group.response_features = None
        df_drop = group.get_response_features(nan_handling='drop_features',
                                               nan_threshold=0.3)
        # Should have fewer columns if any had >30% NaN
        n_expected_drop = (nan_rates > 0.3).sum()
        if n_expected_drop > 0:
            assert df_drop.shape[1] < n_cols_before
        # Remaining columns should have NaN rate <= threshold
        assert (df_drop.isna().mean() <= 0.3 + 1e-10).all()

    def test_invalid_nan_handling_raises(self, tmp_path):
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=1, regions_per=1)
        _write_h5(tmp_path / 'eid-0.h5', n_trials=200)
        group = PhotometrySessionGroup(recs, one=MagicMock(), h5_dir=tmp_path)
        with pytest.raises(ValueError, match='nan_handling'):
            group.get_response_features(nan_handling='invalid')


class TestResponseSimilarityMatrix:

    def test_returns_symmetric_dataframe(self, tmp_path):
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=2, regions_per=1)
        _write_h5(tmp_path / 'eid-0.h5', n_trials=200, seed=0)
        _write_h5(tmp_path / 'eid-1.h5', n_trials=200, seed=1)
        group = PhotometrySessionGroup(recs, one=MagicMock(), h5_dir=tmp_path)
        sim = group.response_similarity_matrix(min_trials=1)
        assert isinstance(sim, pd.DataFrame)
        np.testing.assert_allclose(sim.values, sim.values.T, atol=1e-10)

    def test_auto_calls_get_response_features(self, tmp_path):
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=2, regions_per=1)
        _write_h5(tmp_path / 'eid-0.h5', n_trials=200, seed=0)
        _write_h5(tmp_path / 'eid-1.h5', n_trials=200, seed=1)
        group = PhotometrySessionGroup(recs, one=MagicMock(), h5_dir=tmp_path)
        assert group.response_features is None
        group.response_similarity_matrix(min_trials=1)
        assert group.response_features is not None

    def test_stores_similarity_matrix(self, tmp_path):
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=2, regions_per=1)
        _write_h5(tmp_path / 'eid-0.h5', n_trials=200, seed=0)
        _write_h5(tmp_path / 'eid-1.h5', n_trials=200, seed=1)
        group = PhotometrySessionGroup(recs, one=MagicMock(), h5_dir=tmp_path)
        sim = group.response_similarity_matrix(min_trials=1)
        assert group.similarity_matrix is not None
        pd.testing.assert_frame_equal(sim, group.similarity_matrix)


def _make_decode_recordings(n_per_class=2):
    """Helper: recordings with 2 target_NMs, each with n_per_class subjects."""
    rows = []
    for i in range(n_per_class):
        rows.append({
            'eid': f'eid-A{i}', 'subject': f'subj-A{i}',
            'brain_region': 'VTA-r', 'hemisphere': 'r',
            'target_NM': 'VTA-DA', 'NM': 'DA',
            'session_type': 'biased',
            'start_time': '2024-01-01T10:00:00', 'number': 1,
            'task_protocol': 'biased_protocol',
        })
        rows.append({
            'eid': f'eid-B{i}', 'subject': f'subj-B{i}',
            'brain_region': 'VTA-r', 'hemisphere': 'r',
            'target_NM': 'DR-5HT', 'NM': '5HT',
            'session_type': 'biased',
            'start_time': '2024-01-01T10:00:00', 'number': 1,
            'task_protocol': 'biased_protocol',
        })
    return pd.DataFrame(rows)


class TestDecodeTarget:

    def test_creates_decoder_attribute(self, tmp_path):
        from iblnm.data import PhotometrySessionGroup
        from iblnm.analysis import TargetNMDecoder
        recs = _make_decode_recordings(n_per_class=3)
        for _, rec in recs.iterrows():
            _write_h5(tmp_path / f'{rec["eid"]}.h5', n_trials=200,
                       seed=hash(rec['eid']) % 1000)
        group = PhotometrySessionGroup(recs, one=MagicMock(), h5_dir=tmp_path)
        group.decode_target(min_trials=1)
        assert isinstance(group.decoder, TargetNMDecoder)

    def test_decoder_has_contributions(self, tmp_path):
        from iblnm.data import PhotometrySessionGroup
        recs = _make_decode_recordings(n_per_class=3)
        for _, rec in recs.iterrows():
            _write_h5(tmp_path / f'{rec["eid"]}.h5', n_trials=200,
                       seed=hash(rec['eid']) % 1000)
        group = PhotometrySessionGroup(recs, one=MagicMock(), h5_dir=tmp_path)
        group.decode_target(min_trials=1)
        assert hasattr(group.decoder, 'contributions')
        assert isinstance(group.decoder.contributions, pd.DataFrame)


# =============================================================================
# filter_sessions Tests
# =============================================================================

class TestFilterSessions:

    def test_filters_session_types(self):
        from iblnm.data import PhotometrySessionGroup
        df = _make_sessions_df(n_eids=2, regions_per=1)
        df.loc[0, 'session_type'] = 'habituation'
        group = PhotometrySessionGroup(df, one=MagicMock())
        group.filter_sessions(
            session_types=('biased',), qc_blockers=set(),
            targetnms=False,
            min_performance=False, required_contrasts=False,
        )
        assert 'eid-0' not in group.sessions['eid'].values

    def test_excludes_subjects(self):
        from iblnm.data import PhotometrySessionGroup
        df = _make_sessions_df(n_eids=2, regions_per=1)
        group = PhotometrySessionGroup(df, one=MagicMock())
        group.filter_sessions(
            exclude_subjects=['subj-0'], qc_blockers=set(),
            targetnms=False,
            min_performance=False, required_contrasts=False,
        )
        assert 'subj-0' not in group.sessions['subject'].values

    def test_excludes_eids_to_drop_by_default(self):
        from iblnm.data import PhotometrySessionGroup
        from iblnm.config import EIDS_TO_DROP
        df = _make_sessions_df(n_eids=2, regions_per=1)
        df.loc[0, 'eid'] = EIDS_TO_DROP[0]
        group = PhotometrySessionGroup(df, one=MagicMock())
        group.filter_sessions(
            qc_blockers=set(), targetnms=False,
            min_performance=False, required_contrasts=False,
        )
        assert EIDS_TO_DROP[0] not in group.sessions['eid'].values
        assert 'eid-1' in group.sessions['eid'].values

    def test_exclude_eids_false_keeps_dropped(self):
        from iblnm.data import PhotometrySessionGroup
        from iblnm.config import EIDS_TO_DROP
        df = _make_sessions_df(n_eids=2, regions_per=1)
        df.loc[0, 'eid'] = EIDS_TO_DROP[0]
        group = PhotometrySessionGroup(df, one=MagicMock())
        group.filter_sessions(
            exclude_eids=False, qc_blockers=set(), targetnms=False,
            min_performance=False, required_contrasts=False,
        )
        assert EIDS_TO_DROP[0] in group.sessions['eid'].values

    def test_filters_qc_blockers(self):
        from iblnm.data import PhotometrySessionGroup
        df = _make_sessions_df(n_eids=2, regions_per=1)
        df['logged_errors'] = [['MissingExtractedData'], []]
        group = PhotometrySessionGroup(df, one=MagicMock())
        group.filter_sessions(
            targetnms=False,
            min_performance=False, required_contrasts=False,
        )
        assert 'eid-0' not in group.sessions['eid'].values

    def test_filters_sessions_without_valid_targets(self):
        from iblnm.data import PhotometrySessionGroup
        df = _make_sessions_df(n_eids=2, regions_per=2)
        # eid-0 has ['target-0', 'target-1'], eid-1 same
        group = PhotometrySessionGroup(df, one=MagicMock())
        group.filter_sessions(
            qc_blockers=set(),
            targetnms=['target-0'],
            min_performance=False, required_contrasts=False,
        )
        # Both sessions have target-0, so both survive
        assert len(group.sessions) == 2

    def test_drops_session_with_no_valid_targets(self):
        from iblnm.data import PhotometrySessionGroup
        df = _make_sessions_df(n_eids=2, regions_per=1)
        # eid-0 has ['target-0'], eid-1 has ['target-0']
        # Change eid-1 to have only invalid target
        df.at[1, 'target_NM'] = ['target-invalid']
        group = PhotometrySessionGroup(df, one=MagicMock())
        group.filter_sessions(
            qc_blockers=set(),
            targetnms=['target-0'],
            min_performance=False, required_contrasts=False,
        )
        assert 'eid-1' not in group.sessions['eid'].values
        assert 'eid-0' in group.sessions['eid'].values

    def test_empty_after_filtering(self):
        from iblnm.data import PhotometrySessionGroup
        df = _make_sessions_df(n_eids=2, regions_per=1)
        group = PhotometrySessionGroup(df, one=MagicMock())
        group.filter_sessions(
            session_types=('ephys',),  # none match
            qc_blockers=set(),
            targetnms=False,
            min_performance=False, required_contrasts=False,
        )
        assert len(group.sessions) == 0

    def test_returns_none(self):
        from iblnm.data import PhotometrySessionGroup
        df = _make_sessions_df(n_eids=1, regions_per=1)
        group = PhotometrySessionGroup(df, one=MagicMock())
        result = group.filter_sessions(
            qc_blockers=set(),
            targetnms=False,
            min_performance=False, required_contrasts=False,
        )
        assert result is None

    def test_min_performance_float_applies_to_all(self):
        """Float min_performance filters all session types."""
        from iblnm.data import PhotometrySessionGroup
        df = _make_sessions_df(n_eids=3, regions_per=1)
        df['session_type'] = ['training', 'biased', 'ephys']
        df['fraction_correct'] = [0.6, 0.8, 0.5]
        df['contrasts'] = [[0, 6.25, 12.5, 25, 100]] * 3

        group = PhotometrySessionGroup(df, one=MagicMock())
        group.filter_sessions(
            targetnms=False,
            min_performance=0.7,
            required_contrasts=False,
        )

        remaining = set(group.sessions['eid'].values)
        assert 'eid-1' in remaining  # 0.8 >= 0.7
        assert 'eid-0' not in remaining  # 0.6 < 0.7
        assert 'eid-2' not in remaining  # 0.5 < 0.7

    def test_min_performance_dict_applies_per_type(self):
        """Dict min_performance applies thresholds per session type."""
        from iblnm.data import PhotometrySessionGroup
        df = _make_sessions_df(n_eids=3, regions_per=1)
        df['session_type'] = ['training', 'biased', 'biased']
        df['fraction_correct'] = [0.6, 0.7, 0.9]
        df['contrasts'] = [[0, 6.25, 12.5, 25, 100]] * 3

        group = PhotometrySessionGroup(df, one=MagicMock())
        group.filter_sessions(
            session_types=('training', 'biased'),
            targetnms=False,
            min_performance={'training': 0.5, 'biased': 0.8},
            required_contrasts=False,
        )

        remaining = set(group.sessions['eid'].values)
        assert 'eid-0' in remaining  # training 0.6 >= 0.5
        assert 'eid-1' not in remaining  # biased 0.7 < 0.8
        assert 'eid-2' in remaining  # biased 0.9 >= 0.8

    def test_required_contrasts_exact_match(self):
        """required_contrasts filters sessions whose contrast set doesn't match exactly."""
        from iblnm.data import PhotometrySessionGroup
        df = _make_sessions_df(n_eids=3, regions_per=1)
        df['fraction_correct'] = [0.9, 0.9, 0.9]
        df['contrasts'] = [
            [0, 6.25, 12.5, 25, 100],
            [0, 6.25, 12.5, 25, 50, 100],
            [0, 6.25, 25, 100],
        ]

        group = PhotometrySessionGroup(df, one=MagicMock())
        group.filter_sessions(
            targetnms=False,
            min_performance=False,
            required_contrasts={0, 6.25, 12.5, 25, 100},
        )

        remaining = set(group.sessions['eid'].values)
        assert remaining == {'eid-0'}

    def test_required_contrasts_applies_to_all_session_types(self):
        """Contrast filtering applies to biased and ephys, not just training."""
        from iblnm.data import PhotometrySessionGroup
        df = _make_sessions_df(n_eids=2, regions_per=1)
        df['session_type'] = ['biased', 'ephys']
        df['fraction_correct'] = [0.9, 0.9]
        df['contrasts'] = [
            [0, 6.25, 12.5, 25, 100],
            [0, 6.25, 12.5, 25, 50, 100],
        ]

        group = PhotometrySessionGroup(df, one=MagicMock())
        group.filter_sessions(
            session_types=('biased', 'ephys'),
            targetnms=False,
            min_performance=False,
            required_contrasts={0, 6.25, 12.5, 25, 100},
        )

        remaining = set(group.sessions['eid'].values)
        assert remaining == {'eid-0'}

    def test_catalog_unchanged_after_filter(self):
        """_catalog retains all rows after filter_sessions."""
        from iblnm.data import PhotometrySessionGroup
        df = _make_sessions_df(n_eids=4, regions_per=1)
        group = PhotometrySessionGroup(df, one=MagicMock())
        group.filter_sessions(
            session_types=('ephys',),  # none match (all are biased)
            qc_blockers=set(), targetnms=False,
            min_performance=False, required_contrasts=False,
        )
        assert len(group._catalog) == 4

    def test_filter_sessions_returns_different_views(self):
        """Calling filter_sessions again changes the sessions view."""
        from iblnm.data import PhotometrySessionGroup
        df = _make_sessions_df(n_eids=4, regions_per=1)
        df['session_type'] = ['biased', 'biased', 'ephys', 'ephys']
        group = PhotometrySessionGroup(df, one=MagicMock())
        group.filter_sessions(
            session_types=('biased',), qc_blockers=set(),
            targetnms=False, min_performance=False, required_contrasts=False,
        )
        assert len(group.sessions) == 2
        group.filter_sessions(
            session_types=('ephys',), qc_blockers=set(),
            targetnms=False, min_performance=False, required_contrasts=False,
        )
        assert len(group.sessions) == 2
        assert set(group.sessions['session_type']) == {'ephys'}

    def test_sessions_snapshot_is_independent(self):
        """Snapshot of group.sessions is not affected by subsequent filter calls."""
        from iblnm.data import PhotometrySessionGroup
        df = _make_sessions_df(n_eids=4, regions_per=1)
        df['session_type'] = ['biased', 'biased', 'ephys', 'ephys']
        group = PhotometrySessionGroup(df, one=MagicMock())
        group.filter_sessions(
            session_types=('biased',), qc_blockers=set(),
            targetnms=False, min_performance=False, required_contrasts=False,
        )
        snapshot = group.sessions
        group.filter_sessions(
            session_types=('ephys',), qc_blockers=set(),
            targetnms=False, min_performance=False, required_contrasts=False,
        )
        assert set(snapshot['session_type']) == {'biased'}

    def test_lab_filter(self):
        """lab parameter keeps only sessions from that lab."""
        from iblnm.data import PhotometrySessionGroup
        df = _make_sessions_df(n_eids=4, regions_per=1)
        df['lab'] = ['mainenlab', 'mainenlab', 'cortexlab', 'cortexlab']
        group = PhotometrySessionGroup(df, one=MagicMock())
        group.filter_sessions(
            lab='mainenlab', qc_blockers=set(),
            session_types=False, targetnms=False,
            min_performance=False, required_contrasts=False,
        )
        assert set(group.sessions['lab']) == {'mainenlab'}
        assert len(group.sessions) == 2

    def test_start_time_min_filter(self):
        """start_time_min excludes subjects whose first session is before the cutoff."""
        from iblnm.data import PhotometrySessionGroup
        df = _make_sessions_df(n_eids=4, regions_per=1)
        df['subject'] = ['subj-0', 'subj-0', 'subj-1', 'subj-1']
        df['start_time'] = ['2023-06-01T10:00:00', '2023-07-01T10:00:00',
                            '2024-01-15T10:00:00', '2024-02-01T10:00:00']
        group = PhotometrySessionGroup(df, one=MagicMock())
        group.filter_sessions(
            start_time_min='2024-01-01', qc_blockers=set(),
            session_types=False, targetnms=False,
            min_performance=False, required_contrasts=False,
        )
        assert 'subj-0' not in group.sessions['subject'].values
        assert 'subj-1' in group.sessions['subject'].values
        assert len(group.sessions) == 2


# =============================================================================
# Loader method tests
# =============================================================================

class TestLoaderMethods:
    """Tests for PhotometrySessionGroup.load_* methods."""

    def _make_group(self, n_eids=2, regions_per=1):
        from iblnm.data import PhotometrySessionGroup
        df = _make_sessions_df(n_eids=n_eids, regions_per=regions_per)
        group = PhotometrySessionGroup(df, one=MagicMock())
        group.filter_sessions(
            session_types=False, qc_blockers=set(), targetnms=False,
            min_performance=False, required_contrasts=False,
            photometry_qc=False,
        )
        return group

    def test_load_response_magnitudes(self, tmp_path):
        group = self._make_group()
        df = pd.DataFrame([
            {'eid': 'eid-0', 'trial': 0, 'response': 1.0},
            {'eid': 'eid-0', 'trial': 1, 'response': 2.0},
            {'eid': 'eid-1', 'trial': 0, 'response': 3.0},
            {'eid': 'eid-99', 'trial': 0, 'response': 4.0},  # not in group
        ])
        path = tmp_path / 'responses.pqt'
        df.to_parquet(path, index=False)

        group.load_response_magnitudes(path)
        assert len(group.response_magnitudes) == 3
        assert 'eid-99' not in group.response_magnitudes['eid'].values

    def test_load_response_ols_dropone(self, tmp_path):
        from iblnm.data import RESPONSE_OLS_DROPONE_COLUMNS
        group = self._make_group()
        rows = [
            {'eid': 'eid-0', 'subject': 'subj-0', 'target_NM': 'target-0',
             'brain_region': 'region-0', 'event': 'stimOnTrigger_times',
             'predictor': 'contrast', 'r2': 0.5, 'delta_r2': 0.1, 'n_trials': 80},
            {'eid': 'eid-1', 'subject': 'subj-1', 'target_NM': 'target-0',
             'brain_region': 'region-0', 'event': 'feedback_times',
             'predictor': 'reward', 'r2': 0.4, 'delta_r2': 0.2, 'n_trials': 70},
            {'eid': 'eid-99', 'subject': 'subj-9', 'target_NM': 'target-X',
             'brain_region': 'region-0', 'event': 'stimOnTrigger_times',
             'predictor': 'side', 'r2': 0.3, 'delta_r2': 0.05, 'n_trials': 60},
        ]
        df = pd.DataFrame(rows)[RESPONSE_OLS_DROPONE_COLUMNS]
        path = tmp_path / 'response_ols_persession_dropone.parquet'
        df.to_parquet(path, index=False)

        group.load_response_ols_dropone(path)
        assert set(group.response_ols_dropone_results['eid'].values) == {'eid-0', 'eid-1'}
        assert 'eid-99' not in group.response_ols_dropone_results['eid'].values

        group.load_response_ols_dropone(tmp_path / 'nonexistent.parquet')
        assert group.response_ols_dropone_results is None

    def test_load_response_ols_coefficients(self, tmp_path):
        from iblnm.config import RESPONSE_OLS_COEFS_COLUMNS
        group = self._make_group()
        rows = [
            {'eid': 'eid-0', 'subject': 'subj-0', 'target_NM': 'target-0',
             'brain_region': 'region-0', 'event': 'stimOnTrigger_times',
             'regressor': 'contrast', 'coef': 0.5, 'coef_se': 0.1,
             'n_trials': 80},
            {'eid': 'eid-1', 'subject': 'subj-1', 'target_NM': 'target-0',
             'brain_region': 'region-0', 'event': 'feedback_times',
             'regressor': 'reward', 'coef': 0.4, 'coef_se': 0.2,
             'n_trials': 70},
            {'eid': 'eid-99', 'subject': 'subj-9', 'target_NM': 'target-X',
             'brain_region': 'region-0', 'event': 'stimOnTrigger_times',
             'regressor': 'side', 'coef': 0.3, 'coef_se': 0.05,
             'n_trials': 60},
        ]
        df = pd.DataFrame(rows)[RESPONSE_OLS_COEFS_COLUMNS]
        path = tmp_path / 'response_ols_persession_coefs.parquet'
        df.to_parquet(path, index=False)

        group.load_response_ols_coefficients(path)
        assert set(group.response_ols_coefficients['eid'].values) == {
            'eid-0', 'eid-1'}
        assert 'eid-99' not in group.response_ols_coefficients['eid'].values

        group.load_response_ols_coefficients(tmp_path / 'nonexistent.parquet')
        assert group.response_ols_coefficients is None

    def test_load_response_varcomp_summary(self, tmp_path):
        from iblnm.config import RESPONSE_VARCOMP_SUMMARY_COLUMNS
        group = self._make_group()
        df = pd.DataFrame([
            {'target_NM': 'target-0', 'event': 'stimOnTrigger_times',
             'regressor': 'contrast', 'component': 'V_mouse', 'mean': 0.3,
             'hdi_low': 0.1, 'hdi_high': 0.5, 'n_mice': 4, 'n_sessions': 22},
            {'target_NM': 'target-0', 'event': 'stimOnTrigger_times',
             'regressor': 'contrast', 'component': 'V_session', 'mean': 0.2,
             'hdi_low': 0.05, 'hdi_high': 0.4, 'n_mice': 4, 'n_sessions': 22},
        ])[RESPONSE_VARCOMP_SUMMARY_COLUMNS]
        path = tmp_path / 'response_varcomp_summary.parquet'
        df.to_parquet(path, index=False)

        group.load_response_varcomp_summary(path)
        # No eid column: the loader reads and assigns the frame verbatim.
        pd.testing.assert_frame_equal(group.response_varcomp_summary, df)

        group.load_response_varcomp_summary(tmp_path / 'nonexistent.parquet')
        assert group.response_varcomp_summary is None

    def test_load_response_ols_mouse_pvalues(self, tmp_path):
        from iblnm.data import RESPONSE_OLS_MOUSE_PVAL_COLUMNS
        group = self._make_group()
        df = pd.DataFrame([
            {'target_NM': 'target-0', 'event': 'stimOnTrigger_times',
             'predictor': 'contrast', 'subject': 'subj-0',
             'mean_delta_r2': 0.12, 'p_value': 0.01, 'q_value': 0.03,
             'n_sessions': 3},
            {'target_NM': 'target-0', 'event': 'feedback_times',
             'predictor': 'reward', 'subject': 'subj-1',
             'mean_delta_r2': 0.08, 'p_value': 0.30, 'q_value': 0.45,
             'n_sessions': 2},
        ])[RESPONSE_OLS_MOUSE_PVAL_COLUMNS]
        path = tmp_path / 'response_ols_persession_dropone_mouse_pvalues.parquet'
        df.to_parquet(path, index=False)

        group.load_response_ols_mouse_pvalues(path)
        # No eid column: the loader reads and assigns the frame verbatim.
        pd.testing.assert_frame_equal(group.response_ols_mouse_pvalues, df)

        group.load_response_ols_mouse_pvalues(
            tmp_path / 'nonexistent.parquet')
        assert group.response_ols_mouse_pvalues is None

    def test_load_response_ols_session_pvalues(self, tmp_path):
        """Unlike the per-mouse loader, this table has an eid column, so rows
        outside the group's recordings are filtered out."""
        from iblnm.data import RESPONSE_OLS_SESSION_PVAL_COLUMNS
        group = self._make_group()
        assert group.response_ols_session_pvalues is None
        df = pd.DataFrame([
            {'eid': 'eid-0', 'subject': 'subj-0', 'target_NM': 'target-0',
             'brain_region': 'region-0', 'event': 'stimOnTrigger_times',
             'predictor': 'contrast', 'delta_r2': 0.1, 'p_value': 0.01,
             'q_value': 0.03},
            {'eid': 'eid-99', 'subject': 'subj-9', 'target_NM': 'target-X',
             'brain_region': 'region-0', 'event': 'stimOnTrigger_times',
             'predictor': 'contrast', 'delta_r2': 0.2, 'p_value': 0.02,
             'q_value': 0.04},  # not in group
        ])[RESPONSE_OLS_SESSION_PVAL_COLUMNS]
        path = tmp_path / 'response_ols_persession_dropone_session_pvalues.parquet'
        df.to_parquet(path, index=False)

        group.load_response_ols_session_pvalues(path)
        assert list(group.response_ols_session_pvalues['eid']) == ['eid-0']

        group.load_response_ols_session_pvalues(
            tmp_path / 'nonexistent.parquet')
        assert group.response_ols_session_pvalues is None

    def test_load_response_varcomp_violin(self, tmp_path):
        from iblnm.config import RESPONSE_VARCOMP_VIOLIN_COLUMNS
        group = self._make_group()
        df = pd.DataFrame({
            'target_NM': ['target-0'] * 4,
            'event': ['stimOnTrigger_times'] * 4,
            'regressor': ['contrast'] * 4,
            'component': ['V_mouse', 'V_mouse', 'V_session', 'V_session'],
            'x': [0.0, 0.5, 0.0, 0.5],
            'density': [1.0, 0.2, 0.8, 0.3],
        })[RESPONSE_VARCOMP_VIOLIN_COLUMNS]
        path = tmp_path / 'response_varcomp_violin.parquet'
        df.to_parquet(path, index=False)

        group.load_response_varcomp_violin(path)
        pd.testing.assert_frame_equal(group.response_varcomp_violin, df)

        group.load_response_varcomp_violin(tmp_path / 'nonexistent.parquet')
        assert group.response_varcomp_violin is None

    def test_load_trial_regressors(self, tmp_path):
        group = self._make_group()
        df = pd.DataFrame([
            {'eid': 'eid-0', 'trial': 0, 'reaction_time': 0.1,
             'peak_velocity': 5.0},
            {'eid': 'eid-99', 'trial': 0, 'reaction_time': 0.2,
             'peak_velocity': 6.0},
        ])
        path = tmp_path / 'trial_regressors.pqt'
        df.to_parquet(path, index=False)

        group.load_trial_regressors(path)
        assert len(group.trial_regressors) == 1
        assert group.trial_regressors['eid'].iloc[0] == 'eid-0'

    def test_load_mean_traces(self, tmp_path):
        group = self._make_group()
        df = pd.DataFrame([
            {'eid': 'eid-0', 'target_NM': 'target-0', 'time': 0.0, 'response': 1.0},
            {'eid': 'eid-99', 'target_NM': 'target-X', 'time': 0.0, 'response': 2.0},
        ])
        path = tmp_path / 'traces.pqt'
        df.to_parquet(path, index=False)

        group.load_mean_traces(path)
        assert len(group.mean_traces) == 1

    def test_load_response_features(self, tmp_path):
        group = self._make_group(regions_per=1)
        df = pd.DataFrame({
            'eid': ['eid-0', 'eid-1', 'eid-99'],
            'target_NM': ['target-0', 'target-0', 'target-X'],
            'fiber_idx': [0, 0, 0],
            'feat_a': [1.0, 2.0, 3.0],
            'feat_b': [4.0, 5.0, 6.0],
        }).set_index(['eid', 'target_NM', 'fiber_idx'])
        path = tmp_path / 'features.pqt'
        df.to_parquet(path)

        group.load_response_features(path)
        assert len(group.response_features) == 2
        assert 'eid-99' not in group.response_features.index.get_level_values('eid')

    def test_load_missing_file_is_noop(self, tmp_path):
        group = self._make_group()
        group.load_response_magnitudes(tmp_path / 'nonexistent.pqt')
        assert group.response_magnitudes is None

    def test_load_filters_to_current_recordings(self, tmp_path):
        """After re-filtering, loaded data reflects the new session set."""
        from iblnm.data import PhotometrySessionGroup
        df = _make_sessions_df(n_eids=3, regions_per=1)
        df['session_type'] = ['biased', 'biased', 'ephys']
        group = PhotometrySessionGroup(df, one=MagicMock())
        group.filter_sessions(
            session_types=('biased',), qc_blockers=set(), targetnms=False,
            min_performance=False, required_contrasts=False,
            photometry_qc=False,
        )

        resp = pd.DataFrame([
            {'eid': 'eid-0', 'trial': 0, 'response': 1.0},
            {'eid': 'eid-1', 'trial': 0, 'response': 2.0},
            {'eid': 'eid-2', 'trial': 0, 'response': 3.0},
        ])
        path = tmp_path / 'responses.pqt'
        resp.to_parquet(path, index=False)

        group.load_response_magnitudes(path)
        assert len(group.response_magnitudes) == 2
        assert 'eid-2' not in group.response_magnitudes['eid'].values


# =============================================================================
# response_varcomp Tests
# =============================================================================

def _make_varcomp_coefficients():
    """Per-session coefficients with one includable cell and one that fails.

    Cell A (target-A): 4 mice × 5 sessions → passes the ≥4 mice, ≥5
    sessions/mouse rule. Cell B (target-B): 3 mice × 5 sessions → fails the
    mouse-count check. Coefficients carry injected between-mouse and
    between-session spread so the fit has signal.
    """
    rng = np.random.default_rng(0)
    rows = []
    cells = {'target-A': 4, 'target-B': 3}
    eid = 0
    for target_nm, n_mice in cells.items():
        mouse_means = rng.normal(0, 0.6, n_mice)
        for m in range(n_mice):
            for _ in range(5):
                rows.append({
                    'eid': f'eid-{eid}', 'subject': f'{target_nm}-m{m}',
                    'target_NM': target_nm, 'brain_region': 'region-0',
                    'event': 'stimOnTrigger_times', 'regressor': 'contrast',
                    'coef': mouse_means[m] + rng.normal(0, 0.3),
                    'coef_se': 0.1, 'n_trials': 80})
                eid += 1
    return pd.DataFrame(rows)


class TestResponseVarcomp:
    """Tests for PhotometrySessionGroup.response_varcomp."""

    def _make_group(self):
        from iblnm.data import PhotometrySessionGroup
        return PhotometrySessionGroup(_make_sessions_df(), one=MagicMock())

    def test_inclusion_rule_and_output_shapes(self, monkeypatch):
        # Stub the PyMC fit (its sampling is covered in test_analysis.py): return
        # fixed posterior draws and record which sessions reach it, isolating the
        # orchestration — inclusion rule, frame shapes, survivor counts. The real
        # summarize_posterior runs on the stubbed draws.
        from iblnm.config import (RESPONSE_VARCOMP_SUMMARY_COLUMNS,
                                   RESPONSE_VARCOMP_VIOLIN_COLUMNS)
        rng = np.random.default_rng(1)
        calls = []

        def fake_fit(estimates, ses, mouse_ids, **kwargs):
            calls.append(len(estimates))
            return rng.gamma(2.0, 0.1, 400), rng.gamma(2.0, 0.05, 400)

        monkeypatch.setattr('iblnm.data.fit_measurement_error_varcomp', fake_fit)

        group = self._make_group()
        grid_size = 32
        summary_df, violin_df = group.response_varcomp(
            _make_varcomp_coefficients(),
            mcmc={'draws': 50, 'tune': 50, 'chains': 1, 'target_accept': 0.9,
                  'random_seed': 0},
            tau_prior=('halfnormal', 1.0), min_mice=4,
            min_sessions_per_mouse=5, grid_size=grid_size, hdi_prob=0.94)

        # Only the passing cell (4 mice × 5 sessions) is fit, on all 20 sessions.
        assert calls == [20]
        # The passing cell yields V_mouse and V_session; the failing cell is gone.
        assert list(summary_df.columns) == RESPONSE_VARCOMP_SUMMARY_COLUMNS
        assert set(summary_df['target_NM']) == {'target-A'}
        passing = summary_df[summary_df['target_NM'] == 'target-A']
        assert set(passing['component']) == {'V_mouse', 'V_session'}
        assert (passing['n_mice'] == 4).all()
        assert (passing['n_sessions'] == 20).all()

        # Violin frame: grid_size rows per (cell, component) = 2 components here.
        assert list(violin_df.columns) == RESPONSE_VARCOMP_VIOLIN_COLUMNS
        assert len(violin_df) == 2 * grid_size
        counts = violin_df.groupby(['target_NM', 'component']).size()
        assert (counts == grid_size).all()

    def test_subthreshold_mice_dropped_then_cell_omitted(self):
        """A cell where too few mice clear the per-mouse session floor is omitted."""
        group = self._make_group()
        coefficients = _make_varcomp_coefficients()
        # Thin two of target-A's mice below the 5-session floor: only 2 survive,
        # short of min_mice=4, so the whole cell drops.
        thin = coefficients[~(
            coefficients['subject'].isin(['target-A-m0', 'target-A-m1'])
            & (coefficients.groupby('subject').cumcount() >= 2))]
        summary_df, violin_df = group.response_varcomp(
            thin,
            mcmc={'draws': 50, 'tune': 50, 'chains': 1, 'target_accept': 0.9,
                  'random_seed': 0},
            tau_prior=('halfnormal', 1.0), min_mice=4,
            min_sessions_per_mouse=5, grid_size=16, hdi_prob=0.94)
        assert summary_df.empty
        assert violin_df.empty


# =============================================================================
# get_response_magnitudes Tests
# =============================================================================

class TestGetResponseMagnitudes:

    def test_returns_dataframe(self, tmp_path):
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=1, regions_per=1)
        _write_h5(tmp_path / 'eid-0.h5', n_trials=50)
        group = PhotometrySessionGroup(recs, one=MagicMock(), h5_dir=tmp_path)
        df_events = group.get_response_magnitudes()
        assert isinstance(df_events, pd.DataFrame)
        assert len(df_events) > 0

    def test_stores_response_magnitudes_attribute(self, tmp_path):
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=1, regions_per=1)
        _write_h5(tmp_path / 'eid-0.h5', n_trials=50)
        group = PhotometrySessionGroup(recs, one=MagicMock(), h5_dir=tmp_path)
        group.get_response_magnitudes()
        assert group.response_magnitudes is not None
        assert isinstance(group.response_magnitudes, pd.DataFrame)

    def test_has_expected_columns(self, tmp_path):
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=1, regions_per=1)
        _write_h5(tmp_path / 'eid-0.h5', n_trials=50)
        group = PhotometrySessionGroup(recs, one=MagicMock(), h5_dir=tmp_path)
        df_events = group.get_response_magnitudes()
        expected_cols = {
            'eid', 'subject', 'session_type', 'NM', 'target_NM',
            'brain_region', 'hemisphere', 'event', 'trial', 'response',
        }
        assert expected_cols.issubset(set(df_events.columns))

    def test_response_magnitudes_excludes_predictors(self, tmp_path):
        """Trial-level task/movement predictors live in trial_regressors."""
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=1, regions_per=1)
        _write_h5(tmp_path / 'eid-0.h5', n_trials=50)
        group = PhotometrySessionGroup(recs, one=MagicMock(), h5_dir=tmp_path)
        group.get_response_magnitudes()
        cols = group.response_magnitudes.columns
        for excluded in ['reaction_time', 'movement_time', 'contrast',
                         'signed_contrast', 'choice', 'probabilityLeft']:
            assert excluded not in cols

    def test_one_row_per_trial_per_event(self, tmp_path):
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=1, regions_per=1)
        n_trials = 50
        _write_h5(tmp_path / 'eid-0.h5', n_trials=n_trials)
        group = PhotometrySessionGroup(recs, one=MagicMock(), h5_dir=tmp_path)
        df_events = group.get_response_magnitudes()
        n_events = df_events['event'].nunique()
        assert len(df_events) == n_trials * n_events

    def test_response_magnitude_known_signal(self, tmp_path):
        """Post-event = 1.0, baseline = 0 → response should be ~1.0."""
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=1, regions_per=1)
        _write_h5(tmp_path / 'eid-0.h5', n_trials=50)
        group = PhotometrySessionGroup(recs, one=MagicMock(), h5_dir=tmp_path)
        df_events = group.get_response_magnitudes()
        # After baseline subtraction, post-event signal = 1.0.
        magnitudes = df_events['response'].dropna()
        np.testing.assert_allclose(magnitudes.values, 1.0, atol=0.1)

    def test_skips_missing_h5(self, tmp_path):
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=2, regions_per=1)
        # Only write H5 for eid-0
        _write_h5(tmp_path / 'eid-0.h5', n_trials=50)
        group = PhotometrySessionGroup(recs, one=MagicMock(), h5_dir=tmp_path)
        df_events = group.get_response_magnitudes()
        assert df_events['eid'].nunique() == 1
        assert 'eid-0' in df_events['eid'].values

    def test_multiple_recordings(self, tmp_path):
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=2, regions_per=1)
        _write_h5(tmp_path / 'eid-0.h5', n_trials=50, seed=0)
        _write_h5(tmp_path / 'eid-1.h5', n_trials=50, seed=1)
        group = PhotometrySessionGroup(recs, one=MagicMock(), h5_dir=tmp_path)
        df_events = group.get_response_magnitudes()
        assert df_events['eid'].nunique() == 2

    def test_empty_when_no_h5(self, tmp_path):
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=1, regions_per=1)
        group = PhotometrySessionGroup(recs, one=MagicMock(), h5_dir=tmp_path)
        df_events = group.get_response_magnitudes()
        assert isinstance(df_events, pd.DataFrame)
        assert len(df_events) == 0


# =============================================================================
# fit_lmm Tests
# =============================================================================


def _make_group_with_events():
    """Create a PhotometrySessionGroup with synthetic events for LMM testing.

    3 subjects, 2 target_NMs, 3 events. Events have a known contrast effect.
    """
    from iblnm.data import PhotometrySessionGroup

    rng = np.random.default_rng(0)
    subjects = ['s0', 's1', 's2']
    target_nms = ['VTA-DA', 'DR-5HT']
    events = ['stimOnTrigger_times', 'firstMovement_times', 'feedback_times']
    contrasts = [0.0, 0.0625, 0.125, 0.25, 1.0]
    n_per_cell = 15

    rows = []
    for target_nm in target_nms:
        for subj in subjects:
            subj_intercept = rng.normal(0, 0.3)
            for event in events:
                for side_val in ['left', 'right']:
                    for fb in [1, -1]:
                        for contrast in contrasts:
                            for _ in range(n_per_cell):
                                log_c = contrast_transform(contrast)
                                response = (
                                    1.0 + 0.5 * log_c
                                    + 0.2 * (1 if fb == 1 else 0)
                                    + subj_intercept
                                    + rng.normal(0, 0.5)
                                )
                                rows.append({
                                    'eid': f'eid-{subj}-{target_nm}',
                                    'subject': subj,
                                    'target_NM': target_nm,
                                    'NM': target_nm.split('-')[1],
                                    'brain_region': target_nm.split('-')[0],
                                    'hemisphere': 'r',
                                    'event': event,
                                    'trial': len(rows),
                                    'stim_side': side_val,
                                    'signed_contrast': (
                                        contrast if side_val == 'right'
                                        else -contrast
                                    ),
                                    'contrast': contrast,
                                    'choice': rng.choice([-1, 1]),
                                    'feedbackType': fb,
                                    'probabilityLeft': 0.5,
                                    'reaction_time': 0.2,
                                    'response': response,
                                    'session_type': 'biased',
                                })

    df_events = pd.DataFrame(rows)

    # Split trial-level predictors (trial_regressors) from the response
    # magnitudes (recording keys + response only), per the schema.
    regressor_cols = ['stim_side', 'signed_contrast', 'contrast', 'choice',
                      'feedbackType', 'probabilityLeft', 'reaction_time']
    trial_regressors = (
        df_events[['eid', 'trial'] + regressor_cols]
        .drop_duplicates(subset=['eid', 'trial'])
        .copy()
    )
    trial_regressors['movement_time'] = 0.15
    trial_regressors['response_time'] = 1.0
    # Per-trial variation so peak_velocity and log_reaction_time are not constant
    # (constants are collinear with the intercept and make the persession design
    # singular). Fixed seed keeps the fixture deterministic.
    mvmt_rng = np.random.default_rng(1)
    n_reg = len(trial_regressors)
    trial_regressors['peak_velocity'] = mvmt_rng.uniform(0.5, 2.0, n_reg)
    trial_regressors['reaction_time'] = mvmt_rng.uniform(0.1, 0.5, n_reg)
    response_magnitudes = df_events[[
        'eid', 'subject', 'target_NM', 'NM', 'brain_region', 'hemisphere',
        'event', 'trial', 'session_type', 'response',
    ]].copy()

    # Build minimal recordings DataFrame
    rec_rows = []
    for target_nm in target_nms:
        for subj in subjects:
            rec_rows.append({
                'eid': f'eid-{subj}-{target_nm}',
                'subject': subj,
                'brain_region': target_nm.split('-')[0],
                'hemisphere': 'r',
                'target_NM': target_nm,
                'NM': target_nm.split('-')[1],
                'session_type': 'biased',
                'start_time': '2024-01-01T10:00:00',
                'number': 1,
                'task_protocol': 'biased_protocol',
            })
    recs = pd.DataFrame(rec_rows)

    group = PhotometrySessionGroup(recs, one=MagicMock())
    group.response_magnitudes = response_magnitudes
    group.trial_regressors = trial_regressors
    return group


class TestAnovaResponseMagnitudes:

    def test_returns_dict(self):
        group = _make_group_with_events()
        result = group.response_anovaRM_fit()
        assert isinstance(result, dict)

    def test_keys_are_target_event_tuples(self):
        group = _make_group_with_events()
        result = group.response_anovaRM_fit()
        for key in result:
            assert len(key) == 2
            target_nm, event_label = key
            assert isinstance(target_nm, str)
            assert isinstance(event_label, str)

    def test_values_are_anova_tables(self):
        group = _make_group_with_events()
        result = group.response_anovaRM_fit()
        assert len(result) > 0
        for table in result.values():
            assert isinstance(table, pd.DataFrame)
            for col in ['Source', 'F', 'Pr(>F)', 'method']:
                assert col in table.columns

    def test_seven_terms_per_group(self):
        """3 factors → 7 terms (3 main + 3 two-way + 1 three-way)."""
        group = _make_group_with_events()
        result = group.response_anovaRM_fit()
        for table in result.values():
            assert len(table) == 7

    def test_requires_response_magnitudes(self):
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=1, regions_per=1)
        group = PhotometrySessionGroup(recs, one=MagicMock())
        with pytest.raises(ValueError, match='response_magnitudes'):
            group.response_anovaRM_fit()

    def test_requires_trial_regressors(self):
        group = _make_group_with_events()
        group.trial_regressors = None
        with pytest.raises(ValueError, match='trial_regressors'):
            group.response_anovaRM_fit()

    def test_stores_results_on_self(self):
        group = _make_group_with_events()
        group.response_anovaRM_fit()
        assert hasattr(group, 'anova_results')
        assert isinstance(group.anova_results, dict)


# =============================================================================
# _modeling_frame Tests
# =============================================================================


def _make_group_with_planted_trials():
    """Group with kept trials and two that each violate one filter.

    Single eid, single recording, single event. Trials 0 and 3 pass the modeling
    filters; trial 1 breaks response_time>0.05 (false start) and trial 2 breaks
    choice!=0 (no-go). Trial 3 sits in a biased block (probabilityLeft==0.8),
    which _modeling_frame keeps by default.
    """
    from iblnm.data import PhotometrySessionGroup

    response_magnitudes = pd.DataFrame({
        'eid': 'eid-0',
        'subject': 's0',
        'target_NM': 'VTA-DA',
        'NM': 'DA',
        'brain_region': 'VTA',
        'hemisphere': 'r',
        'event': 'stimOnTrigger_times',
        'trial': [0, 1, 2, 3],
        'session_type': 'biased',
        'response': [1.0, 1.1, 1.2, 1.3],
    })
    trial_regressors = pd.DataFrame({
        'eid': 'eid-0',
        'trial': [0, 1, 2, 3],
        'stim_side': ['right', 'right', 'right', 'right'],
        'signed_contrast': [0.25, 0.25, 0.25, 0.25],
        'contrast': [0.25, 0.25, 0.25, 0.25],
        'choice': [1, 1, 0, 1],          # trial 2: no-go
        'feedbackType': [1, 1, 1, 1],
        'probabilityLeft': [0.5, 0.5, 0.5, 0.8],  # trial 3: biased block
        'reaction_time': [0.2, 0.2, 0.2, 0.2],
        'movement_time': [0.15, 0.15, 0.15, 0.15],
        'response_time': [1.0, 0.01, 1.0, 1.0],   # trial 1: false start
        'peak_velocity': [1.0, 1.0, 1.0, 1.0],
    })
    recs = pd.DataFrame([{
        'eid': 'eid-0', 'subject': 's0', 'brain_region': 'VTA',
        'hemisphere': 'r', 'target_NM': 'VTA-DA', 'NM': 'DA',
        'session_type': 'biased', 'start_time': '2024-01-01T10:00:00',
        'number': 1, 'task_protocol': 'biased_protocol',
    }])
    group = PhotometrySessionGroup(recs, one=MagicMock())
    group.response_magnitudes = response_magnitudes
    group.trial_regressors = trial_regressors
    return group


class TestModelingFrame:

    def test_excludes_filtered_trials(self):
        # Trial 1 (false start) and trial 2 (no-go) are dropped. Trial 3
        # (biased block) is kept: _modeling_frame includes all blocks by
        # default, filtering only on response_time and choice.
        group = _make_group_with_planted_trials()
        df = group._modeling_frame()
        assert df['trial'].tolist() == [0, 3]

    def test_includes_derived_columns(self):
        group = _make_group_with_planted_trials()
        df = group._modeling_frame()
        for col in ('relative_contrast', 'contrast', 'side'):
            assert col in df.columns


class TestCodeLmmPredictors:

    def _frame(self):
        # contrast in percent units (compute_trial_contrasts multiplies by 100);
        # log2 coding requires nonzero values >= 1.
        return pd.DataFrame({
            'contrast': [0.0, 6.25, 100.0],
            'side': ['contra', 'ipsi', 'contra'],
            'choice_side': ['contra', 'ipsi', 'ipsi'],
            'feedbackType': [1, -1, 1],
            'log_reaction_time': [-1.5, -0.5, -2.0],
        })

    def test_side_and_reward_deviation_coded(self):
        group = _make_group_with_planted_trials()
        coded = group._code_lmm_predictors(self._frame())
        assert set(coded['side']) <= {-0.5, 0.5}
        assert set(coded['reward']) <= {-0.5, 0.5}
        assert coded['side'].tolist() == [0.5, -0.5, 0.5]
        assert coded['reward'].tolist() == [0.5, -0.5, 0.5]

    def test_choice_side_deviation_coded(self):
        group = _make_group_with_planted_trials()
        coded = group._code_lmm_predictors(self._frame())
        # contra = +0.5, ipsi = −0.5, same scheme as stimulus side.
        assert coded['choice_side'].tolist() == [0.5, -0.5, -0.5]

    def test_contrast_log2_coded_and_centered(self):
        group = _make_group_with_planted_trials()
        coded = group._code_lmm_predictors(self._frame())
        expected = np.array([0.0, np.log2(6.25), np.log2(100.0)])
        expected = expected - expected.mean()
        assert coded['contrast'].mean() == pytest.approx(0.0, abs=1e-12)
        np.testing.assert_allclose(coded['contrast'].values, expected)

    def test_timing_column_unchanged(self):
        group = _make_group_with_planted_trials()
        df = self._frame()
        coded = group._code_lmm_predictors(df)
        np.testing.assert_array_equal(
            coded['log_reaction_time'].values, df['log_reaction_time'].values)

    def test_input_frame_not_mutated(self):
        group = _make_group_with_planted_trials()
        df = self._frame()
        before = df.copy(deep=True)
        group._code_lmm_predictors(df)
        pd.testing.assert_frame_equal(df, before)


def _make_group_for_response_lmm():
    """``_make_group_with_events`` with percent-unit contrasts.

    ``response_lmm_fit`` codes contrast with the default ``log2`` scheme, which
    requires percent units (nonzero values >= 1); the shared events fixture uses
    fractional contrasts. Rescaling the sign-preserving contrast columns leaves
    ``add_relative_contrast``'s side/relative_contrast derivation unchanged.
    """
    group = _make_group_with_events()
    group.trial_regressors['contrast'] *= 100
    group.trial_regressors['signed_contrast'] *= 100
    return group


class TestResponseLMMFit:

    def test_caches_fit_and_returns_matching_r2(self):
        from iblnm.analysis import LMMResult
        group = _make_group_for_response_lmm()
        r2 = group.response_lmm_fit(
            {'ceiling': '{response} ~ C(contrast) * side * reward'},
            group_by=['target_NM', 'event'])
        # One registry entry and one R² row per (target_NM, event) group.
        assert not r2.empty
        for _, row in r2.iterrows():
            key = ('response', 'ceiling', row['target_NM'], row['event'])
            fit = group.lmm_fits[key]
            assert isinstance(fit, LMMResult)
            assert row['marginal_r2'] == fit.variance_explained['marginal']

    def test_multiple_names_one_entry_and_row_each(self):
        group = _make_group_for_response_lmm()
        formulas = {'ceiling': '{response} ~ C(contrast) * side * reward',
                    'interactions': '{response} ~ contrast + side + reward'}
        r2 = group.response_lmm_fit(formulas, group_by=['target_NM', 'event'])
        groups = r2[['target_NM', 'event']].drop_duplicates()
        # One row per (group, name); one registry entry per (group, name).
        assert len(r2) == len(groups) * len(formulas)
        for _, g in groups.iterrows():
            for name in formulas:
                key = ('response', name, g['target_NM'], g['event'])
                assert key in group.lmm_fits

    def test_distinct_caller_names_no_collision(self):
        group = _make_group_for_response_lmm()
        # Two formulas the caller passes under distinct names: each caches
        # under its own registry key, with no config.LMM_FORMULAS lookup.
        formulas = {'task_full': '{response} ~ contrast * side * reward',
                    'me_full': '{response} ~ contrast + side + reward'}
        r2 = group.response_lmm_fit(formulas, group_by=['target_NM', 'event'])
        groups = r2[['target_NM', 'event']].drop_duplicates()
        for _, g in groups.iterrows():
            for name in formulas:
                key = ('response', name, g['target_NM'], g['event'])
                assert key in group.lmm_fits

    def test_per_name_re_formula_adds_random_slope(self):
        group = _make_group_for_response_lmm()
        group.response_lmm_fit(
            {'interactions': '{response} ~ contrast + side + reward'},
                               group_by=['target_NM', 'event'],
                               re_formula={'interactions': '1 + side'})
        fit = next(iter(group.lmm_fits.values()))
        slopes = next(iter(fit.random_effects.values()))
        assert 'side' in slopes.index

    def test_movement_family_with_nan_timing_fits_shared_trials(self):
        """Regression: a family mixing a timing-using model with one that
        omits the timing predictor must fit every member on the same
        NaN-dropped trials. A NaN ``log_<timing>`` row otherwise misaligns
        statsmodels' ``groups`` array against the design matrix, raising
        ``IndexError`` from ``MixedLM.group_list``.
        """
        group = _make_group_for_response_lmm()
        reg = group.trial_regressors
        rng = np.random.default_rng(0)
        reg['reaction_time'] = rng.uniform(0.1, 2.0, len(reg))
        # Missing movement onsets -> _modeling_frame sets log to NaN.
        reg.loc[reg.index[::5], 'reaction_time'] = np.nan
        formulas = {
            'full': '{response} ~ contrast + log_reaction_time',
            'contrast': '{response} ~ log_reaction_time',
            'movement': '{response} ~ contrast',
        }
        r2 = group.response_lmm_fit(formulas, group_by=['target_NM', 'event'])
        assert not r2.empty
        # Within each fitted group, every member fits the same trial count,
        # below the group's full total (the NaN-timing rows were dropped) — so
        # the drop-one ΔR² shares a denominator even for the timing-free model.
        df = group._modeling_frame()
        checked = 0
        for (target_nm, event), df_group in df.groupby(['target_NM', 'event']):
            keys = [('response', name, target_nm, event) for name in formulas]
            if not all(k in group.lmm_fits for k in keys):
                continue
            nobs = {len(group.lmm_fits[k].model.endog) for k in keys}
            assert len(nobs) == 1
            assert nobs.pop() < len(df_group)
            checked += 1
        assert checked > 0


class TestFitResponseModel:
    """Tests for PhotometrySession.fit_response_model (single OLS fit)."""

    def _coded_frame(self, n=60, seed=0):
        """Synthetic coded trial frame with a `response` driven by `contrast`."""
        rng = np.random.default_rng(seed)
        contrast = rng.uniform(-1, 1, n)
        response = 2.0 * contrast + rng.normal(0, 0.1, n)
        return pd.DataFrame({'contrast': contrast, 'response': response})

    def test_rsquared_matches_direct_fit_ols(self, mock_photometry_session):
        from iblnm.analysis import fit_ols
        df = self._coded_frame()
        fit = mock_photometry_session.fit_response_model(df, '{response} ~ contrast')
        direct = fit_ols('response ~ contrast', df)
        assert fit.rsquared == direct.rsquared

    def test_response_col_substituted_into_formula(self, mock_photometry_session):
        from iblnm.analysis import fit_ols
        df = self._coded_frame().rename(columns={'response': 'magnitude'})
        fit = mock_photometry_session.fit_response_model(
            df, '{response} ~ contrast', response_col='magnitude')
        direct = fit_ols('magnitude ~ contrast', df)
        assert fit.rsquared == direct.rsquared

    def test_returns_none_on_singular_design(self, mock_photometry_session):
        df = self._coded_frame()
        df['contrast'] = 1.0  # constant predictor -> collinear with intercept
        assert mock_photometry_session.fit_response_model(
            df, '{response} ~ contrast') is None

    def test_ols_fits_empty_after_construction(self, minimal_session_series):
        from iblnm.data import PhotometrySession
        ps = PhotometrySession(minimal_session_series, one=MagicMock(),
                               load_data=False)
        assert ps.ols_fits == {}


def _make_session_for_persession(n_trials=120, contrast_gain=2.0, seed=0,
                                 eid='test-eid', subject='mouse1',
                                 region='VTA-r', hemisphere='r',
                                 target_nm='VTA-DA'):
    """PhotometrySession with contrast-driven responses for one recording.

    The early-window magnitude of every event is ``contrast_gain * contrast``
    plus small noise, so the ``contrast`` predictor carries real variance.
    Wheel velocity is finite so ``peak_velocity`` survives complete-case
    filtering. Trials are all unbiased-block go trials with a real response.
    The identity arguments (``eid``/``subject``/``region``/``hemisphere``/
    ``target_nm``) let callers build a multi-recording group; ``wheel_fs`` is
    set so the session round-trips through ``save_h5``.
    """
    import xarray as xr
    from iblnm.data import PhotometrySession

    series = pd.Series({
        'eid': eid, 'subject': subject,
        'start_time': '2024-01-01T10:00:00', 'number': 1,
        'task_protocol': 'biased', 'session_type': 'biased',
        'brain_region': [region], 'hemisphere': [hemisphere],
        'target_NM': [target_nm],
    })
    ps = PhotometrySession(series, one=MagicMock(), load_data=False)

    rng = np.random.default_rng(seed)
    n_time = 61
    tpts = np.linspace(-1, 1, n_time)
    events = ['stimOnTrigger_times', 'firstMovement_times', 'feedback_times']

    # Percent units, as the real `contrast` column is stored (log2 coding
    # expects nonzero values >= 1).
    contrasts = np.array([0.0, 6.25, 12.5, 25.0, 100.0])
    contrast_vals = rng.choice(contrasts, n_trials)
    sides = rng.choice(['left', 'right'], n_trials)
    signed = np.where(sides == 'left', -1, 1) * contrast_vals

    # Magnitude per (event, trial) driven by unsigned contrast; broadcast it
    # across the whole post-event window so the window mean recovers it.
    magnitude = contrast_gain * (contrast_vals / 100) + rng.normal(0, 0.05,
                                                                   n_trials)
    data = np.zeros((len(events), n_trials, n_time))
    data[:, :, tpts >= 0] = magnitude[None, :, None]
    ps.photometry_responses = {
        region: xr.DataArray(
            data, dims=['event', 'trial', 'time'],
            coords={'event': events,
                    'trial': np.arange(n_trials), 'time': tpts},
        )
    }

    # Vary the inter-event gaps so reaction/movement times are not constant
    # (a constant log predictor collinear with the intercept fails the fit).
    stim_on = np.linspace(10, 10 + n_trials, n_trials)
    reaction = rng.uniform(0.1, 0.5, n_trials)
    movement = rng.uniform(0.2, 1.0, n_trials)
    ps.trials = pd.DataFrame({
        'trial': np.arange(n_trials),
        'stimOnTrigger_times': stim_on,
        'firstMovement_times': stim_on + reaction,
        'response_times': stim_on + reaction + movement,
        # Feedback delivery lags the choice by the measured correct-trial
        # amount, 0.5 ms.
        'feedback_times': stim_on + reaction + movement + 0.0005,
        'signed_contrast': signed,
        'contrast': contrast_vals,
        'stim_side': sides,
        'feedbackType': rng.choice([1, -1], n_trials),
        'choice': rng.choice([-1, 1], n_trials),
        'probabilityLeft': np.full(n_trials, 0.5),
    })
    # `_response_modeling_frame` reads the wheel through `load_responses`, so
    # the matrix has to be stored, not just assigned. Each session gets its own
    # H5 under a directory that lives as long as the test session.
    ps.filepath = Path(_SCRATCH_H5_DIR.name) / f'{eid}.h5'
    ps.filepath.unlink(missing_ok=True)
    ps.wheel_responses = {
        WHEEL_LABEL: xr.DataArray(
            rng.normal(0, 1, (1, n_trials, 50)),
            dims=['event', 'trial', 'time'],
            coords={'event': ['stimOnTrigger_times'], 'trial': np.arange(n_trials),
                    'time': np.arange(50) / 100},
        )
    }
    ps.save_h5(groups=['wheel'])
    return ps


class TestCompareResponseModels:
    """Tests for PhotometrySession.compare_response_models (drop-one family)."""

    @property
    def formulas(self):
        from iblnm.config import LMM_FORMULAS
        return LMM_FORMULAS['persession']

    def test_absent_region_returns_empty_frame(self):
        ps = _make_session_for_persession()
        out, _ = ps.compare_response_models('NOT-A-REGION', self.formulas)
        assert out.empty
        assert list(out.columns) == [
            'brain_region', 'target_NM', 'event', 'predictor', 'r2',
            'delta_r2', 'n_trials',
        ]

    def test_informative_contrast_has_positive_delta_r2(self):
        ps = _make_session_for_persession()
        out, _ = ps.compare_response_models('VTA-r', self.formulas)
        contrast_rows = out[out['predictor'] == 'contrast']
        assert not contrast_rows.empty
        assert (contrast_rows['delta_r2'] > 0).all()

    def test_rows_tagged_with_region_and_target_nm(self):
        ps = _make_session_for_persession()
        out, _ = ps.compare_response_models('VTA-r', self.formulas)
        assert (out['brain_region'] == 'VTA-r').all()
        assert (out['target_NM'] == 'VTA-DA').all()
        # No `full` reference row; one row per dropped predictor per event.
        assert 'full' not in set(out['predictor'])

    def test_identical_trial_count_across_models_per_event(self):
        ps = _make_session_for_persession()
        ps.compare_response_models('VTA-r', self.formulas)
        events = {event for _, event in ps.ols_fits}
        assert events
        for event in events:
            nobs = {len(ps.ols_fits[(name, event)].model.endog)
                    for name in self.formulas}
            assert len(nobs) == 1

    def test_ols_fits_cached_per_name_event(self):
        ps = _make_session_for_persession()
        out, _ = ps.compare_response_models('VTA-r', self.formulas)
        for event in set(out['event']):
            for name in self.formulas:
                assert (name, event) in ps.ols_fits

    def test_event_below_min_trials_is_skipped(self):
        ps = _make_session_for_persession(n_trials=120)
        _, coefs = ps.compare_response_models(
            'VTA-r', self.formulas, min_trials=200)
        dropone, _ = ps.compare_response_models(
            'VTA-r', self.formulas, min_trials=200)
        assert dropone.empty
        assert coefs.empty

    def test_coefficients_frame_columns_and_grain(self):
        from iblnm.data import PERSESSION_COEFS_COLUMNS
        from iblnm.config import _PERSESSION_REGRESSORS
        ps = _make_session_for_persession()
        _, coefs = ps.compare_response_models('VTA-r', self.formulas)
        assert list(coefs.columns) == PERSESSION_COEFS_COLUMNS
        # One row per (event, regressor) present in that event's full fit.
        regressors = set(_PERSESSION_REGRESSORS)
        assert set(coefs['regressor']) <= regressors
        per_event = coefs.groupby('event')['regressor'].agg(set)
        for event, present in per_event.items():
            full_params = set(ps.ols_fits[('full', event)].params.index)
            assert present == regressors & full_params

    def test_coefficients_read_full_fit_params_and_bse(self):
        ps = _make_session_for_persession()
        _, coefs = ps.compare_response_models('VTA-r', self.formulas)
        for _, row in coefs.iterrows():
            fit = ps.ols_fits[('full', row['event'])]
            assert row['coef'] == fit.params[row['regressor']]
            assert row['coef_se'] == fit.bse[row['regressor']]
            assert row['brain_region'] == 'VTA-r'
            assert row['target_NM'] == 'VTA-DA'


class TestResponseOlsDropone:
    """Tests for PhotometrySessionGroup.response_ols_dropone (orchestration)."""

    @property
    def formulas(self):
        from iblnm.config import LMM_FORMULAS
        return LMM_FORMULAS['persession']

    def _write_recording(self, h5_dir, eid, subject, region, hemisphere,
                         target_nm, n_trials=120, seed=0):
        """Save a persession PhotometrySession to ``h5_dir/{eid}.h5``."""
        ps = _make_session_for_persession(
            n_trials=n_trials, seed=seed, eid=eid, subject=subject,
            region=region, hemisphere=hemisphere, target_nm=target_nm)
        ps.save_h5(h5_dir / f'{eid}.h5',
                   groups=['metadata', 'trials', 'photometry', 'wheel'])

    def _recordings(self, rows):
        """Build a recordings DataFrame from (eid, subject, region, hemi,
        target_nm) tuples. ``target_NM`` is filled in at query time and is the
        authoritative source — ``response_ols_dropone`` builds each session from
        this row, not the H5 ``/metadata`` (where the field may be missing)."""
        return pd.DataFrame([
            {'eid': eid, 'subject': subject, 'brain_region': region,
             'hemisphere': hemi, 'target_NM': target_nm,
             'session_type': 'biased', 'start_time': '2024-01-01T10:00:00',
             'number': 1, 'task_protocol': 'biased'}
            for eid, subject, region, hemi, target_nm in rows
        ])

    def test_concatenates_per_recording_rows_with_tags(self, tmp_path):
        from iblnm.data import PhotometrySessionGroup
        self._write_recording(tmp_path, 'eid-0', 'subj-0', 'VTA-r', 'r',
                              'VTA-DA', seed=0)
        self._write_recording(tmp_path, 'eid-1', 'subj-1', 'DR-l', 'l',
                              'DR-5HT', seed=1)
        recs = self._recordings([
            ('eid-0', 'subj-0', 'VTA-r', 'r', 'VTA-DA'),
            ('eid-1', 'subj-1', 'DR-l', 'l', 'DR-5HT'),
        ])
        group = PhotometrySessionGroup(recs, one=MagicMock(), h5_dir=tmp_path)
        out, coefs = group.response_ols_dropone(self.formulas)

        assert list(out.columns) == [
            'eid', 'subject', 'target_NM', 'brain_region', 'event',
            'predictor', 'r2', 'delta_r2', 'n_trials',
        ]
        # Both recordings contribute; eid/subject tags come from the row.
        assert set(out['eid']) == {'eid-0', 'eid-1'}
        assert dict(out.groupby('eid')['subject'].first()) == {
            'eid-0': 'subj-0', 'eid-1': 'subj-1'}
        # target_NM comes from the recordings row (query-time fill-in), not the
        # H5 /metadata, which the per-recording session no longer loads.
        assert dict(out.groupby('eid')['target_NM'].first()) == {
            'eid-0': 'VTA-DA', 'eid-1': 'DR-5HT'}
        # One row per dropped regressor per (recording, event); no reference row.
        assert 'full' not in set(out['predictor'])
        dropped = set(self.formulas) - {'full'}
        per_event = out.groupby(['eid', 'event'])['predictor'].agg(set)
        assert (per_event == dropped).all()

        # The coefficients frame is tagged with the same eid/subject pairs.
        from iblnm.config import RESPONSE_OLS_COEFS_COLUMNS
        assert list(coefs.columns) == RESPONSE_OLS_COEFS_COLUMNS
        assert set(coefs['eid']) == {'eid-0', 'eid-1'}
        assert dict(coefs.groupby('eid')['subject'].first()) == {
            'eid-0': 'subj-0', 'eid-1': 'subj-1'}

    def test_insufficient_recording_is_absent(self, tmp_path):
        from iblnm.data import PhotometrySessionGroup
        self._write_recording(tmp_path, 'eid-0', 'subj-0', 'VTA-r', 'r',
                              'VTA-DA', n_trials=120, seed=0)
        # Too few trials to score any event -> contributes no rows.
        self._write_recording(tmp_path, 'eid-1', 'subj-1', 'DR-l', 'l',
                              'DR-5HT', n_trials=20, seed=1)
        recs = self._recordings([
            ('eid-0', 'subj-0', 'VTA-r', 'r', 'VTA-DA'),
            ('eid-1', 'subj-1', 'DR-l', 'l', 'DR-5HT'),
        ])
        group = PhotometrySessionGroup(recs, one=MagicMock(), h5_dir=tmp_path)
        out, coefs = group.response_ols_dropone(self.formulas)

        assert set(out['eid']) == {'eid-0'}
        assert 'eid-1' not in set(out['eid'])
        assert set(coefs['eid']) == {'eid-0'}


class TestResponseLMMEffects:

    def test_coefficients_carry_terms_and_ci(self):
        group = _make_group_for_response_lmm()
        group.response_lmm_fit(
            {'interactions': '{response} ~ contrast + side + reward'},
                               group_by=['target_NM', 'event'])
        effects = group.response_lmm_effects('interactions', 'coefficients')
        # One identity-tagged row per fixed-effects term, with CI columns.
        for col in ('term', 'Coef.', 'ci_lower', 'ci_upper',
                    'target_NM', 'event'):
            assert col in effects.columns
        key = ('response', 'interactions',
               effects.iloc[0]['target_NM'], effects.iloc[0]['event'])
        fit = group.lmm_fits[key]
        row = effects[(effects['target_NM'] == key[2])
                      & (effects['event'] == key[3])
                      & (effects['term'] == 'Intercept')].iloc[0]
        coef = fit.summary_df.loc['Intercept', 'Coef.']
        se = fit.summary_df.loc['Intercept', 'Std.Err.']
        assert row['Coef.'] == coef
        assert row['ci_lower'] == pytest.approx(coef - 1.96 * se)
        assert row['ci_upper'] == pytest.approx(coef + 1.96 * se)

    def test_emm_matches_direct_call(self):
        from iblnm.analysis import compute_marginal_means
        group = _make_group_for_response_lmm()
        group.response_lmm_fit(
            {'interactions': '{response} ~ contrast + side + reward'},
                               group_by=['target_NM', 'event'])
        effects = group.response_lmm_effects(
            'interactions', 'emm', ['reward'])
        # The factor is its own column; identity columns are appended.
        for col in ('reward', 'predicted', 'ci_lower', 'ci_upper',
                    'target_NM', 'event'):
            assert col in effects.columns

        # Reproduce one group's reward EMMs by a direct call on the cached fit.
        df = group._modeling_frame()
        (target_nm, event), _ = next(iter(df.groupby(['target_NM', 'event'])))
        fit = group.lmm_fits[('response', 'interactions', target_nm, event)]
        expected = compute_marginal_means(fit, ['reward'])

        got = effects[(effects['target_NM'] == target_nm)
                      & (effects['event'] == event)].sort_values('reward')
        np.testing.assert_allclose(
            got['predicted'].values,
            expected.sort_values('reward')['predicted'].values)

    def test_emm_two_factors_give_interaction_grid(self):
        group = _make_group_for_response_lmm()
        group.response_lmm_fit(
            {'interactions': '{response} ~ contrast + side + reward'},
                               group_by=['target_NM', 'event'])
        effects = group.response_lmm_effects(
            'interactions', 'emm', ['contrast', 'reward'])
        assert {'contrast', 'reward'}.issubset(effects.columns)

    def test_emm_requires_variables(self):
        group = _make_group_for_response_lmm()
        group.response_lmm_fit(
            {'interactions': '{response} ~ contrast + side + reward'},
                               group_by=['target_NM', 'event'])
        with pytest.raises(ValueError, match='requires a `variables`'):
            group.response_lmm_effects('interactions', 'emm')

    def test_unknown_kind_raises(self):
        group = _make_group_for_response_lmm()
        group.response_lmm_fit(
            {'interactions': '{response} ~ contrast + side + reward'},
                               group_by=['target_NM', 'event'])
        with pytest.raises(ValueError, match='kind must be'):
            group.response_lmm_effects('interactions', 'bogus')


class TestResponseLMMResampling:

    def test_crossval_columns_and_matches_direct_call(self):
        from iblnm.analysis import crossval_lmm
        group = _make_group_for_response_lmm()
        formulas = {'full': '{response} ~ contrast * side * reward',
                    'interactions': '{response} ~ contrast + side + reward'}
        result = group.response_lmm_crossval(
            formulas, group_by=['target_NM', 'event'])
        assert list(result.columns) == [
            'target_NM', 'event', 'predictor', 'fold', 'n_trials',
            'r2', 'delta_r2']
        assert set(result['predictor']) == {'interactions'}

        # Reproduce one group's interactions delta_r2 by a direct call with the
        # same reference.
        df = group._modeling_frame()
        (target_nm, event), df_group = next(
            iter(df.groupby(['target_NM', 'event'])))
        df_coded = group._code_lmm_predictors(df_group)
        coded = {k: v.format(response='response') for k, v in formulas.items()}
        expected = crossval_lmm(df_coded, coded, 'response', reference='full')

        got = result[(result['target_NM'] == target_nm)
                     & (result['event'] == event)
                     & (result['predictor'] == 'interactions')]
        np.testing.assert_allclose(
            got['delta_r2'].values,
            expected[expected['predictor'] == 'interactions']
            ['delta_r2'].values)

    def test_jackknife_columns_and_matches_direct_call(self):
        from iblnm.analysis import jackknife_lmm
        group = _make_group_for_response_lmm()
        formulas = {'full': '{response} ~ contrast * side * reward',
                    'interactions': '{response} ~ contrast + side + reward'}
        result = group.response_lmm_jackknife(
            formulas, group_by=['target_NM', 'event'])
        assert list(result.columns) == [
            'target_NM', 'event', 'predictor', 'fold', 'n_trials',
            'r2', 'delta_r2']

        # Reproduce one group's interactions delta_r2 by a direct call with the
        # same reference.
        df = group._modeling_frame()
        (target_nm, event), df_group = next(
            iter(df.groupby(['target_NM', 'event'])))
        df_coded = group._code_lmm_predictors(df_group)
        coded = {k: v.format(response='response') for k, v in formulas.items()}
        expected = jackknife_lmm(df_coded, coded, 'response', reference='full')

        got = result[(result['target_NM'] == target_nm)
                     & (result['event'] == event)
                     & (result['predictor'] == 'interactions')]
        np.testing.assert_allclose(
            got['delta_r2'].values,
            expected[expected['predictor'] == 'interactions']
            ['delta_r2'].values)

    def _movement_group(self):
        """Events fixture with a varying ``log_reaction_time`` predictor."""
        group = _make_group_for_response_lmm()
        reg = group.trial_regressors
        rng = np.random.default_rng(0)
        reg['reaction_time'] = rng.uniform(0.1, 2.0, len(reg))
        reg['log_reaction_time'] = np.log10(reg['reaction_time'])
        return group

    _MOVEMENT_FORMULAS = {
        'full': '{response} ~ contrast + log_reaction_time',
        'contrast': '{response} ~ log_reaction_time',
        'movement': '{response} ~ contrast',
    }

    def test_movement_set_fits_when_trials_sufficient(self):
        from iblnm.config import MIN_SUBJECTS_MOVEMENT
        # Baseline: with full timing data, every target contributes rows.
        group = self._movement_group()
        result = group.response_lmm_crossval(
            self._MOVEMENT_FORMULAS, group_by=['target_NM', 'event'],
            min_subjects=MIN_SUBJECTS_MOVEMENT)
        assert (result['target_NM'] == 'DR-5HT').sum() > 0
        assert (result['target_NM'] == 'VTA-DA').sum() > 0

    def test_events_filter_restricts_to_named_events(self):
        # The ``events`` filter scopes the modeling frame to the named events
        # before grouping, so the script can run a per-event formula set.
        group = _make_group_for_response_lmm()
        formulas = {'full': '{response} ~ contrast * side',
                    'contrast': '{response} ~ side'}
        result = group.response_lmm_crossval(
            formulas, group_by=['target_NM', 'event'],
            events=['feedback_times'])
        assert set(result['event']) == {'feedback_times'}

    def test_below_min_trials_contributes_no_rows(self):
        from iblnm.config import MIN_SUBJECTS_MOVEMENT, MIN_TRIALS_MOVEMENT
        # Null out all but a handful of one target's timing values so its
        # per-group complete-case count falls below the min_trials floor.
        group = self._movement_group()
        reg = group.trial_regressors
        starved = reg['eid'].str.contains('DR-5HT')
        idx = reg[starved].index
        # ``_modeling_frame`` derives ``log_reaction_time`` from the raw column,
        # so starve the raw ``reaction_time`` to push the group below the floor.
        reg.loc[idx[5:], 'reaction_time'] = np.nan

        result = group.response_lmm_crossval(
            self._MOVEMENT_FORMULAS, group_by=['target_NM', 'event'],
            min_subjects=MIN_SUBJECTS_MOVEMENT, min_trials=MIN_TRIALS_MOVEMENT)
        assert (result['target_NM'] == 'DR-5HT').sum() == 0
        assert (result['target_NM'] == 'VTA-DA').sum() > 0


# =============================================================================
# CCA Tests
# =============================================================================


def _make_group_with_response_features(n_per_target=5, n_features=8, seed=42):
    """Create a PhotometrySessionGroup with synthetic response_features.

    3 subjects, 2 target_NMs (VTA-DA, DR-5HT), n_per_target recordings each.
    """
    from iblnm.data import PhotometrySessionGroup

    rng = np.random.default_rng(seed)
    target_nms = ['VTA-DA', 'DR-5HT']
    subjects = ['s0', 's1', 's2']

    rec_rows = []
    feature_rows = {}
    for tnm in target_nms:
        for i in range(n_per_target):
            subj = subjects[i % len(subjects)]
            eid = f'eid-{subj}-{tnm}-{i}'
            rec_rows.append({
                'eid': eid,
                'subject': subj,
                'brain_region': tnm.split('-')[0],
                'hemisphere': 'r',
                'target_NM': tnm,
                'NM': tnm.split('-')[1],
                'session_type': 'biased',
                'start_time': '2024-01-01T10:00:00',
                'number': 1,
                'task_protocol': 'biased_protocol',
            })
            feature_rows[(eid, tnm)] = rng.standard_normal(n_features)

    recs = pd.DataFrame(rec_rows)
    group = PhotometrySessionGroup(recs, one=MagicMock())

    index = pd.MultiIndex.from_tuples(feature_rows.keys(),
                                       names=['eid', 'target_NM'])
    cols = [f'feat_{i}' for i in range(n_features)]
    group.response_features = pd.DataFrame(
        list(feature_rows.values()), index=index, columns=cols,
    )
    return group


def _make_mock_performance(group, seed=0):
    """Create a mock performance DataFrame matching the eids in group."""
    rng = np.random.default_rng(seed)
    eids = group.response_features.index.get_level_values('eid').unique()
    return pd.DataFrame({
        'eid': eids,
        'psych_50_threshold': rng.uniform(10, 50, len(eids)),
        'psych_50_bias': rng.uniform(-20, 20, len(eids)),
        'psych_50_lapse_left': rng.uniform(0, 0.2, len(eids)),
        'psych_50_lapse_right': rng.uniform(0, 0.2, len(eids)),
        'bias_shift': rng.uniform(-10, 10, len(eids)),
    })


class TestGetPsychometricFeatures:

    def test_returns_aligned_dataframe(self):
        """Output index should match response_features index."""
        import tempfile
        group = _make_group_with_response_features()
        perf = _make_mock_performance(group)
        with tempfile.NamedTemporaryFile(suffix='.pqt', delete=False) as f:
            perf.to_parquet(f.name)
            result = group.get_psychometric_features(performance_path=f.name)
        assert list(result.index) == list(group.response_features.index)

    def test_default_params(self):
        """Default params should be psych_50 threshold, bias, lapse_left, lapse_right."""
        import tempfile
        group = _make_group_with_response_features()
        perf = _make_mock_performance(group)
        with tempfile.NamedTemporaryFile(suffix='.pqt', delete=False) as f:
            perf.to_parquet(f.name)
            result = group.get_psychometric_features(performance_path=f.name)
        assert set(result.columns) == {
            'psych_50_threshold', 'psych_50_bias',
            'psych_50_lapse_contra', 'psych_50_lapse_ipsi',
        }

    def test_custom_params(self):
        import tempfile
        group = _make_group_with_response_features()
        perf = _make_mock_performance(group)
        with tempfile.NamedTemporaryFile(suffix='.pqt', delete=False) as f:
            perf.to_parquet(f.name)
            result = group.get_psychometric_features(
                performance_path=f.name,
                params=['psych_50_threshold', 'bias_shift'],
            )
        assert set(result.columns) == {'psych_50_threshold', 'bias_shift'}

    def test_stored_as_attribute(self):
        import tempfile
        group = _make_group_with_response_features()
        perf = _make_mock_performance(group)
        with tempfile.NamedTemporaryFile(suffix='.pqt', delete=False) as f:
            perf.to_parquet(f.name)
            group.get_psychometric_features(performance_path=f.name)
        assert group.psychometric_features is not None

    def test_values_match_performance_data(self):
        """Merged values should match the source performance data."""
        import tempfile
        group = _make_group_with_response_features()
        perf = _make_mock_performance(group)
        with tempfile.NamedTemporaryFile(suffix='.pqt', delete=False) as f:
            perf.to_parquet(f.name)
            result = group.get_psychometric_features(performance_path=f.name)
        # Check first recording's threshold matches
        first_eid = result.index.get_level_values('eid')[0]
        expected = perf.loc[perf['eid'] == first_eid, 'psych_50_threshold'].iloc[0]
        actual = result.iloc[0]['psych_50_threshold']
        assert np.isclose(actual, expected)

    def test_uses_preloaded_performance(self):
        """When self.performance is already set, no path is needed."""
        group = _make_group_with_response_features()
        perf = _make_mock_performance(group)
        group.performance = perf
        result = group.get_psychometric_features()
        assert list(result.index) == list(group.response_features.index)
        first_eid = result.index.get_level_values('eid')[0]
        expected = perf.loc[perf['eid'] == first_eid, 'psych_50_threshold'].iloc[0]
        assert np.isclose(result.iloc[0]['psych_50_threshold'], expected)

    def test_lateralizes_bias_and_lapse(self):
        """Bias and lapse terms should be converted to contra/ipsi frame."""
        from iblnm.data import PhotometrySessionGroup

        # Two recordings: one left hemisphere, one right hemisphere
        # with known bias and lapse values
        recs = pd.DataFrame([
            {'eid': 'e0', 'subject': 's0', 'brain_region': 'VTA',
             'hemisphere': 'l', 'target_NM': 'VTA-DA', 'NM': 'DA',
             'session_type': 'biased', 'start_time': '2024-01-01T10:00:00',
             'number': 1, 'task_protocol': 'biased'},
            {'eid': 'e1', 'subject': 's1', 'brain_region': 'DR',
             'hemisphere': 'r', 'target_NM': 'DR-5HT', 'NM': '5HT',
             'session_type': 'biased', 'start_time': '2024-01-01T10:00:00',
             'number': 1, 'task_protocol': 'biased'},
        ])
        group = PhotometrySessionGroup(recs, one=MagicMock())

        index = pd.MultiIndex.from_tuples(
            [('e0', 'VTA-DA'), ('e1', 'DR-5HT')],
            names=['eid', 'target_NM'],
        )
        group.response_features = pd.DataFrame(
            np.ones((2, 3)), index=index, columns=['f0', 'f1', 'f2'])

        perf = pd.DataFrame({
            'eid': ['e0', 'e1'],
            'psych_50_threshold': [20.0, 30.0],
            'psych_50_bias': [10.0, 10.0],
            'psych_50_lapse_left': [0.05, 0.10],
            'psych_50_lapse_right': [0.15, 0.20],
        })
        group.performance = perf
        result = group.get_psychometric_features()

        # Column names should be lateralized
        assert 'psych_50_lapse_contra' in result.columns
        assert 'psych_50_lapse_ipsi' in result.columns
        assert 'psych_50_lapse_left' not in result.columns

        # Threshold unchanged for both
        assert result.loc[('e0', 'VTA-DA'), 'psych_50_threshold'] == 20.0
        assert result.loc[('e1', 'DR-5HT'), 'psych_50_threshold'] == 30.0

        # Bias: left hemi uses hemi_sign=1 (no flip), right hemi uses -1 (flip)
        assert result.loc[('e0', 'VTA-DA'), 'psych_50_bias'] == 10.0   # left hemi, no flip
        assert result.loc[('e1', 'DR-5HT'), 'psych_50_bias'] == -10.0  # right hemi, flipped

        # Lapse: left hemi contra=right, right hemi contra=left
        assert result.loc[('e0', 'VTA-DA'), 'psych_50_lapse_contra'] == 0.15  # was lapse_right
        assert result.loc[('e0', 'VTA-DA'), 'psych_50_lapse_ipsi'] == 0.05    # was lapse_left
        assert result.loc[('e1', 'DR-5HT'), 'psych_50_lapse_contra'] == 0.10  # was lapse_left
        assert result.loc[('e1', 'DR-5HT'), 'psych_50_lapse_ipsi'] == 0.20    # was lapse_right


class TestGroupFitCCA:

    def test_returns_cca_result(self):
        from iblnm.analysis import CCAResult
        import tempfile
        group = _make_group_with_response_features()
        perf = _make_mock_performance(group)
        with tempfile.NamedTemporaryFile(suffix='.pqt', delete=False) as f:
            perf.to_parquet(f.name)
            group.get_psychometric_features(performance_path=f.name)
        result = group.fit_cca(n_permutations=0)
        assert isinstance(result, CCAResult)

    def test_stored_as_attribute(self):
        import tempfile
        group = _make_group_with_response_features()
        perf = _make_mock_performance(group)
        with tempfile.NamedTemporaryFile(suffix='.pqt', delete=False) as f:
            perf.to_parquet(f.name)
            group.get_psychometric_features(performance_path=f.name)
        group.fit_cca(n_permutations=0)
        assert group.cca_result is not None

    def test_session_labels_from_eid(self):
        """fit_cca should pass eid as session_labels for permutation."""
        import tempfile
        group = _make_group_with_response_features()
        perf = _make_mock_performance(group)
        with tempfile.NamedTemporaryFile(suffix='.pqt', delete=False) as f:
            perf.to_parquet(f.name)
            group.get_psychometric_features(performance_path=f.name)
        result = group.fit_cca(n_permutations=50, seed=0)
        assert result.p_values is not None


# =============================================================================
# GLM Response Features Tests
# =============================================================================


class TestGetGLMResponseFeatures:
    # Percent-unit fixture: the persession model codes contrast with log2, which
    # requires nonzero contrasts >= 1 (see ``_make_group_for_response_lmm``).

    @staticmethod
    def _formula():
        from iblnm.config import LMM_FORMULAS
        return LMM_FORMULAS['persession']['full']

    def test_returns_persession_coefficient_columns(self):
        """Columns are the persession model's coefficient names."""
        group = _make_group_for_response_lmm()
        result = group.get_persession_ols_features(
            self._formula(), event_name='stimOnTrigger_times')
        assert isinstance(result, pd.DataFrame)
        for col in ('Intercept', 'contrast', 'side', 'reward', 'choice_side',
                    'log_reaction_time', 'peak_velocity', 'contrast:side'):
            assert col in result.columns

    def test_stored_as_attribute(self):
        """Result is stored as self.persession_ols_features."""
        group = _make_group_for_response_lmm()
        group.get_persession_ols_features(self._formula(), event_name='stimOnTrigger_times')
        assert group.persession_ols_features is not None
        assert len(group.persession_ols_features) > 0

    def test_index_structure(self):
        """Index has (eid, target_NM, fiber_idx) levels."""
        group = _make_group_for_response_lmm()
        result = group.get_persession_ols_features(
            self._formula(), event_name='stimOnTrigger_times')
        assert result.index.names == ['eid', 'target_NM', 'fiber_idx']

    def test_weight_by_se(self):
        """With weight_by_se=True, values are t-statistics (coef / SE)."""
        group = _make_group_for_response_lmm()
        coefs = group.get_persession_ols_features(
            self._formula(), event_name='stimOnTrigger_times', weight_by_se=False)
        group2 = _make_group_for_response_lmm()
        tstats = group2.get_persession_ols_features(
            self._formula(), event_name='stimOnTrigger_times', weight_by_se=True)
        assert not np.allclose(coefs.values, tstats.values)

    def test_one_row_per_recording(self):
        """Each scorable recording (eid × brain_region) produces one row."""
        group = _make_group_for_response_lmm()
        result = group.get_persession_ols_features(
            self._formula(), event_name='stimOnTrigger_times')
        # fixture has 6 recordings (3 subjects × 2 targets)
        assert len(result) == 6

    def test_persession_coefficient_count(self):
        """Output has 19 columns (6 mains + 12 interactions + intercept)."""
        group = _make_group_for_response_lmm()
        result = group.get_persession_ols_features(
            self._formula(), event_name='stimOnTrigger_times')
        assert result.shape[1] == 19

    def test_excludes_false_start_trials(self):
        """Trials with response_time <= 0.05 must be excluded; all-fast → empty result."""
        group = _make_group_for_response_lmm()
        group.trial_regressors['response_time'] = 0.01
        result = group.get_persession_ols_features(
            self._formula(), event_name='stimOnTrigger_times')
        assert len(result) == 0

    def test_excludes_nogo_trials(self):
        """Trials with choice == 0 must be excluded; all-nogo → empty result."""
        group = _make_group_for_response_lmm()
        group.trial_regressors = group.trial_regressors.copy()
        group.trial_regressors['choice'] = 0
        result = group.get_persession_ols_features(
            self._formula(), event_name='stimOnTrigger_times')
        assert len(result) == 0


class TestGLMFeaturesCCA:

    def test_cca_with_glm_features(self):
        """fit_cca works with persession_ols_features as X input."""
        import tempfile
        from iblnm.config import LMM_FORMULAS
        group = _make_group_for_response_lmm()
        group.get_persession_ols_features(
            LMM_FORMULAS['persession']['full'], event_name='stimOnTrigger_times')
        group.response_features = group.persession_ols_features
        perf = _make_mock_performance(group)
        with tempfile.NamedTemporaryFile(suffix='.pqt', delete=False) as f:
            perf.to_parquet(f.name)
            group.get_psychometric_features(performance_path=f.name)
        result = group.fit_cca(n_permutations=0)
        assert result.x_weights.shape[1] > 0
        assert 'contrast' in result.x_weights.index


# =============================================================================
# load_response_traces / flush_response_traces Tests
# =============================================================================


class TestLoadResponseTraces:

    def test_loads_traces(self, tmp_path):
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=2, regions_per=1)
        _write_h5(tmp_path / 'eid-0.h5', n_trials=50, seed=0)
        _write_h5(tmp_path / 'eid-1.h5', n_trials=50, seed=1)
        group = PhotometrySessionGroup(recs, one=MagicMock(), h5_dir=tmp_path)
        group.load_response_traces()
        assert group.response_traces is not None
        assert len(group.response_traces) > 0

    def test_cache_structure(self, tmp_path):
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=1, regions_per=1)
        _write_h5(tmp_path / 'eid-0.h5', n_trials=50, seed=0)
        group = PhotometrySessionGroup(recs, one=MagicMock(), h5_dir=tmp_path)
        group.load_response_traces()
        key = list(group.response_traces.keys())[0]
        entry = group.response_traces[key]
        assert 'traces' in entry
        assert 'tpts' in entry
        assert 'meta' in entry
        assert 'trials' in entry
        assert entry['traces'].ndim == 2  # (n_trials, n_timepoints)

    def test_key_is_eid_region_event(self, tmp_path):
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=1, regions_per=1)
        _write_h5(tmp_path / 'eid-0.h5', n_trials=50, seed=0)
        group = PhotometrySessionGroup(recs, one=MagicMock(), h5_dir=tmp_path)
        group.load_response_traces()
        key = list(group.response_traces.keys())[0]
        assert len(key) == 3  # (eid, brain_region, event)
        assert key[0] == 'eid-0'

    def test_traces_are_baseline_subtracted(self, tmp_path):
        """Post-event traces should be ~1.0 after baseline subtraction."""
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=1, regions_per=1)
        _write_h5(tmp_path / 'eid-0.h5', n_trials=50, seed=0)
        group = PhotometrySessionGroup(recs, one=MagicMock(), h5_dir=tmp_path)
        group.load_response_traces()
        key = list(group.response_traces.keys())[0]
        traces = group.response_traces[key]['traces']
        tpts = group.response_traces[key]['tpts']
        post = traces[:, tpts > 0.1]
        np.testing.assert_allclose(np.nanmean(post), 1.0, atol=0.2)

    def test_stores_shared_time_axis(self, tmp_path):
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=1, regions_per=1)
        _write_h5(tmp_path / 'eid-0.h5', n_trials=50, seed=0)
        group = PhotometrySessionGroup(recs, one=MagicMock(), h5_dir=tmp_path)
        group.load_response_traces()
        assert group.response_traces_tpts is not None
        assert len(group.response_traces_tpts) > 0

    def test_flush(self, tmp_path):
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=1, regions_per=1)
        _write_h5(tmp_path / 'eid-0.h5', n_trials=50, seed=0)
        group = PhotometrySessionGroup(recs, one=MagicMock(), h5_dir=tmp_path)
        group.load_response_traces()
        assert group.response_traces is not None
        group.flush_response_traces()
        assert group.response_traces is None

    def test_skips_missing_h5(self, tmp_path):
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=2, regions_per=1)
        _write_h5(tmp_path / 'eid-0.h5', n_trials=50, seed=0)
        # eid-1.h5 not written
        group = PhotometrySessionGroup(recs, one=MagicMock(), h5_dir=tmp_path)
        group.load_response_traces()
        eids = {k[0] for k in group.response_traces.keys()}
        assert 'eid-0' in eids
        assert 'eid-1' not in eids

    def test_multiple_events_per_recording(self, tmp_path):
        """Each recording produces one cache entry per response event."""
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=1, regions_per=1)
        _write_h5(tmp_path / 'eid-0.h5', n_trials=50, seed=0)
        group = PhotometrySessionGroup(recs, one=MagicMock(), h5_dir=tmp_path)
        group.load_response_traces()
        events = {k[2] for k in group.response_traces.keys()}
        assert events == {'stimOnTrigger_times', 'feedback_times'}


class TestGetResponseMagnitudesFromCache:

    def test_uses_cached_traces(self, tmp_path):
        """If traces already loaded, does not re-load H5."""
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=1, regions_per=1)
        _write_h5(tmp_path / 'eid-0.h5', n_trials=50, seed=0)
        group = PhotometrySessionGroup(recs, one=MagicMock(), h5_dir=tmp_path)
        group.load_response_traces()
        # Delete H5 to prove it doesn't re-read
        (tmp_path / 'eid-0.h5').unlink()
        result = group.get_response_magnitudes()
        assert len(result) > 0

    def test_auto_loads_traces_if_not_cached(self, tmp_path):
        """Calling get_response_magnitudes without prior load still works."""
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=1, regions_per=1)
        _write_h5(tmp_path / 'eid-0.h5', n_trials=50, seed=0)
        group = PhotometrySessionGroup(recs, one=MagicMock(), h5_dir=tmp_path)
        result = group.get_response_magnitudes()
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert group.response_traces is not None


class TestGetMeanTraces:

    def test_returns_dataframe(self, tmp_path):
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=2, regions_per=1)
        _write_h5(tmp_path / 'eid-0.h5', n_trials=50, seed=0)
        _write_h5(tmp_path / 'eid-1.h5', n_trials=50, seed=1)
        group = PhotometrySessionGroup(recs, one=MagicMock(), h5_dir=tmp_path)
        result = group.get_mean_traces()
        assert isinstance(result, pd.DataFrame)
        expected_cols = {'eid', 'subject', 'target_NM', 'brain_region',
                         'event', 'time', 'response'}
        assert expected_cols <= set(result.columns)

    def test_one_trace_per_recording_event(self, tmp_path):
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=2, regions_per=1)
        _write_h5(tmp_path / 'eid-0.h5', n_trials=50, seed=0)
        _write_h5(tmp_path / 'eid-1.h5', n_trials=50, seed=1)
        group = PhotometrySessionGroup(recs, one=MagicMock(), h5_dir=tmp_path)
        result = group.get_mean_traces()
        n_rec_events = result.groupby(['eid', 'brain_region', 'event']).ngroups
        assert n_rec_events == 2 * 2  # 2 recordings × 2 events

    def test_stored_as_attribute(self, tmp_path):
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=1, regions_per=1)
        _write_h5(tmp_path / 'eid-0.h5', n_trials=50, seed=0)
        group = PhotometrySessionGroup(recs, one=MagicMock(), h5_dir=tmp_path)
        group.get_mean_traces()
        assert group.mean_traces is not None

    def test_uses_cached_traces(self, tmp_path):
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=1, regions_per=1)
        _write_h5(tmp_path / 'eid-0.h5', n_trials=50, seed=0)
        group = PhotometrySessionGroup(recs, one=MagicMock(), h5_dir=tmp_path)
        group.load_response_traces()
        (tmp_path / 'eid-0.h5').unlink()
        result = group.get_mean_traces()
        assert len(result) > 0

    def test_mean_trace_values(self, tmp_path):
        """Post-event mean trace should be ~1.0 for our test data."""
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=1, regions_per=1)
        _write_h5(tmp_path / 'eid-0.h5', n_trials=50, seed=0)
        group = PhotometrySessionGroup(recs, one=MagicMock(), h5_dir=tmp_path)
        result = group.get_mean_traces()
        post_event = result[result['time'] > 0.1]
        np.testing.assert_allclose(
            post_event['response'].mean(), 1.0, atol=0.2)

    def test_has_contrast_and_feedback_columns(self, tmp_path):
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=1, regions_per=1)
        _write_h5(tmp_path / 'eid-0.h5', n_trials=50, seed=0)
        group = PhotometrySessionGroup(recs, one=MagicMock(), h5_dir=tmp_path)
        result = group.get_mean_traces()
        assert 'contrast' in result.columns
        assert 'feedbackType' in result.columns

    def test_excludes_biased_block_trials(self, tmp_path):
        """Trials with probabilityLeft != 0.5 must be excluded."""
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=1, regions_per=1)
        _write_h5(tmp_path / 'eid-0.h5', n_trials=50, seed=0, all_biased=True)
        group = PhotometrySessionGroup(recs, one=MagicMock(), h5_dir=tmp_path)
        result = group.get_mean_traces()
        assert len(result) == 0, "Expected empty result when all trials are biased"

    def test_excludes_nogo_trials(self, tmp_path):
        """Trials with choice == 0 must be excluded."""
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=1, regions_per=1)
        _write_h5(tmp_path / 'eid-0.h5', n_trials=50, seed=0, all_nogo=True)
        group = PhotometrySessionGroup(recs, one=MagicMock(), h5_dir=tmp_path)
        result = group.get_mean_traces()
        assert len(result) == 0, "Expected empty result when all trials are no-go"

    def test_excludes_fast_response_trials(self, tmp_path):
        """Trials with response_time <= 0.05 must be excluded."""
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=1, regions_per=1)
        _write_h5(tmp_path / 'eid-0.h5', n_trials=50, seed=0, fast_response=True)
        group = PhotometrySessionGroup(recs, one=MagicMock(), h5_dir=tmp_path)
        result = group.get_mean_traces()
        assert len(result) == 0, "Expected empty result when all response_times < 0.05"

    def test_traces_grouped_by_contrast_feedback(self, tmp_path):
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=1, regions_per=1)
        _write_h5(tmp_path / 'eid-0.h5', n_trials=50, seed=0)
        group = PhotometrySessionGroup(recs, one=MagicMock(), h5_dir=tmp_path)
        result = group.get_mean_traces()
        # Should have multiple (contrast, feedbackType) combinations per
        # (eid, brain_region, event)
        n_groups = result.groupby(
            ['eid', 'brain_region', 'event', 'contrast', 'feedbackType']
        ).ngroups
        n_rec_events = result.groupby(['eid', 'brain_region', 'event']).ngroups
        assert n_groups > n_rec_events


# =============================================================================
# Per-Cohort CCA Tests
# =============================================================================


def _make_group_for_cohort_cca(n_per_cohort=None, seed=42):
    """Create a group with pre-populated glm and psychometric features.

    Parameters
    ----------
    n_per_cohort : dict, optional
        Mapping target_NM → number of recordings.
        Default: {'VTA-DA': 50, 'DR-5HT': 50}.
    seed : int
    """
    from iblnm.data import PhotometrySessionGroup

    if n_per_cohort is None:
        n_per_cohort = {'VTA-DA': 50, 'DR-5HT': 50}

    rng = np.random.default_rng(seed)
    subjects = ['s0', 's1', 's2', 's3', 's4']
    glm_cols = [
        'intercept', 'contrast', 'side', 'reward',
        'contrast:side', 'contrast:reward', 'side:reward',
    ]
    psych_cols = [
        'psych_50_threshold', 'psych_50_bias',
        'psych_50_lapse_contra', 'psych_50_lapse_ipsi',
    ]

    rec_rows = []
    glm_rows = {}
    psych_rows = {}

    for tnm, n in n_per_cohort.items():
        for i in range(n):
            subj = subjects[i % len(subjects)]
            eid = f'eid-{subj}-{tnm}-{i}'
            rec_rows.append({
                'eid': eid,
                'subject': subj,
                'brain_region': tnm.split('-')[0],
                'hemisphere': 'r',
                'target_NM': tnm,
                'NM': tnm.split('-')[1],
                'session_type': 'biased',
                'start_time': '2024-01-01T10:00:00',
                'number': 1,
                'task_protocol': 'biased_protocol',
            })
            key = (eid, tnm, 0)
            glm_rows[key] = rng.standard_normal(len(glm_cols))
            psych_rows[key] = rng.uniform(0, 1, len(psych_cols))

    recs = pd.DataFrame(rec_rows)
    group = PhotometrySessionGroup(recs, one=MagicMock())

    glm_index = pd.MultiIndex.from_tuples(
        glm_rows.keys(), names=['eid', 'target_NM', 'fiber_idx'])
    group.persession_ols_features = pd.DataFrame(
        list(glm_rows.values()), index=glm_index, columns=glm_cols)

    psych_index = pd.MultiIndex.from_tuples(
        psych_rows.keys(), names=['eid', 'target_NM', 'fiber_idx'])
    group.psychometric_features = pd.DataFrame(
        list(psych_rows.values()), index=psych_index, columns=psych_cols)

    return group


class TestGroupFitCohortCCA:

    def test_returns_dict_of_cca_results(self):
        from iblnm.analysis import CCAResult
        group = _make_group_for_cohort_cca()
        results = group.fit_cohort_cca(n_permutations=0)
        assert isinstance(results, dict)
        for v in results.values():
            assert isinstance(v, CCAResult)

    def test_one_result_per_target_nm(self):
        group = _make_group_for_cohort_cca()
        results = group.fit_cohort_cca(n_permutations=0)
        assert set(results.keys()) == {'VTA-DA', 'DR-5HT'}

    def test_stores_standardized_data(self):
        group = _make_group_for_cohort_cca()
        group.fit_cohort_cca(n_permutations=0)
        assert group.cohort_cca_data is not None
        for X_z, Y_z in group.cohort_cca_data.values():
            np.testing.assert_allclose(X_z.mean(axis=0), 0, atol=0.01)

    def test_excludes_intercept_by_default(self):
        group = _make_group_for_cohort_cca()
        group.fit_cohort_cca(n_permutations=0)
        for result in group.cohort_cca_results.values():
            assert 'intercept' not in result.x_weights.index

    def test_skips_small_cohort(self):
        group = _make_group_for_cohort_cca(
            n_per_cohort={'VTA-DA': 50, 'DR-5HT': 5})
        results = group.fit_cohort_cca(n_permutations=0, min_recordings=10)
        assert 'VTA-DA' in results
        assert 'DR-5HT' not in results

    def test_feature_cols_subsets_neural_features(self):
        """feature_cols restricts X to the named columns; weights index matches."""
        feature_cols = ['contrast', 'side', 'reward',
                        'contrast:side', 'contrast:reward']
        group = _make_group_for_cohort_cca()
        results = group.fit_cohort_cca(n_permutations=0, feature_cols=feature_cols)
        for result in results.values():
            assert list(result.x_weights.index) == feature_cols


class TestGroupCrossProjectCCA:

    def test_diagonal_matches_within(self):
        group = _make_group_for_cohort_cca()
        group.fit_cohort_cca(n_permutations=0)
        cp = group.cross_project_cca()
        for cohort, result in group.cohort_cca_results.items():
            row = cp[(cp['data_cohort'] == cohort) &
                     (cp['weight_cohort'] == cohort)]
            np.testing.assert_allclose(
                row['correlation'].iloc[0], result.correlations[0], atol=0.05)

    def test_all_pairs(self):
        group = _make_group_for_cohort_cca()
        group.fit_cohort_cca(n_permutations=0)
        cp = group.cross_project_cca()
        n = len(group.cohort_cca_results)
        assert len(cp) == n ** 2

    def test_subset(self):
        group = _make_group_for_cohort_cca()
        group.fit_cohort_cca(n_permutations=0)
        cp = group.cross_project_cca(cohorts=['VTA-DA'])
        assert len(cp) == 1

    def test_stores_result(self):
        group = _make_group_for_cohort_cca()
        group.fit_cohort_cca(n_permutations=0)
        group.cross_project_cca()
        assert group.cohort_cca_cross_projections is not None


class TestGroupCompareCCAWeights:

    def test_self_cosine_one(self):
        group = _make_group_for_cohort_cca()
        group.fit_cohort_cca(n_permutations=0)
        ws = group.compare_cca_weights()
        for cohort in group.cohort_cca_results:
            row = ws[(ws['cohort_a'] == cohort) &
                     (ws['cohort_b'] == cohort)]
            np.testing.assert_allclose(
                abs(row['neural_cosine'].iloc[0]), 1.0, atol=0.01)

    def test_symmetric(self):
        group = _make_group_for_cohort_cca()
        group.fit_cohort_cca(n_permutations=0)
        ws = group.compare_cca_weights()
        ab = ws[(ws['cohort_a'] == 'VTA-DA') &
                (ws['cohort_b'] == 'DR-5HT')]
        ba = ws[(ws['cohort_a'] == 'DR-5HT') &
                (ws['cohort_b'] == 'VTA-DA')]
        np.testing.assert_allclose(
            ab['neural_cosine'].iloc[0], ba['neural_cosine'].iloc[0],
            atol=1e-10)

    def test_stores_result(self):
        group = _make_group_for_cohort_cca()
        group.fit_cohort_cca(n_permutations=0)
        group.compare_cca_weights()
        assert group.cohort_cca_weight_similarities is not None


# =============================================================================
# get_trial_regressors Tests
# =============================================================================

def _write_trial_regressor_h5(path, with_wheel=True):
    """Write a 3-trial H5 with known trials and (optionally) wheel velocity."""
    import h5py

    stim_on = np.array([10.0, 20.0, 30.0])
    first_move = np.array([10.5, 20.7, 31.2])
    response = np.array([11.0, 21.5, 32.0])
    with h5py.File(path, 'w') as f:
        grp = f.create_group('trials/table')
        grp.create_dataset('trial', data=np.arange(3))
        grp.create_dataset('stimOnTrigger_times', data=stim_on)
        grp.create_dataset('firstMovement_times', data=first_move)
        grp.create_dataset('response_times', data=response)
        grp.create_dataset('feedback_times', data=response + 0.3)
        grp.create_dataset('signed_contrast', data=np.array([-0.25, 0.0, 1.0]))
        grp.create_dataset('contrast', data=np.array([0.25, 0.0, 1.0]))
        grp.create_dataset('stim_side', data=np.array(['left', 'right', 'right'],
                                                      dtype='S5'))
        grp.create_dataset('choice', data=np.array([-1, 1, 1]))
        grp.create_dataset('feedbackType', data=np.array([1, -1, 1]))
        grp.create_dataset('probabilityLeft', data=np.full(3, 0.5))
        if with_wheel:
            from iblnm.data import WHEEL_LABEL
            wheel_grp = f.create_group(f'wheel/{WHEEL_LABEL}/responses')
            velocity = np.array([[0.0, 1.0, -3.0],
                                 [np.nan, np.nan, np.nan],
                                 [2.0, -5.0, 1.0]])
            wheel_grp.create_dataset('stimOnTrigger_times', data=velocity)
            wheel_grp.create_dataset('trials', data=np.arange(3))
            wheel_grp.create_dataset('times', data=np.arange(3) / 100)
    return stim_on, first_move, response


class TestGetTrialRegressors:

    def test_trial_regressors_schema_and_values(self, tmp_path):
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=1, regions_per=1)
        stim_on, first_move, response = _write_trial_regressor_h5(
            tmp_path / 'eid-0.h5')
        group = PhotometrySessionGroup(recs, one=MagicMock(), h5_dir=tmp_path)

        df = group.get_trial_regressors()

        expected_cols = {
            'eid', 'trial', 'signed_contrast', 'contrast', 'stim_side',
            'choice', 'feedbackType', 'probabilityLeft', 'reaction_time',
            'movement_time', 'response_time', 'peak_velocity',
        }
        assert set(df.columns) == expected_cols
        assert len(df) == 3
        np.testing.assert_array_equal(
            df['peak_velocity'].values, np.array([3.0, np.nan, 5.0]))
        np.testing.assert_allclose(
            df['reaction_time'].values, first_move - stim_on)
        np.testing.assert_allclose(
            df['movement_time'].values, response - first_move)
        np.testing.assert_allclose(
            df['response_time'].values, response - stim_on)

    def test_trial_regressors_stores_result(self, tmp_path):
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=1, regions_per=1)
        _write_trial_regressor_h5(tmp_path / 'eid-0.h5')
        group = PhotometrySessionGroup(recs, one=MagicMock(), h5_dir=tmp_path)
        result = group.get_trial_regressors()
        assert group.trial_regressors is result

    def test_trial_regressors_no_wheel_nan_peak_velocity(self, tmp_path):
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=1, regions_per=1)
        _write_trial_regressor_h5(tmp_path / 'eid-0.h5', with_wheel=False)
        group = PhotometrySessionGroup(recs, one=MagicMock(), h5_dir=tmp_path)
        df = group.get_trial_regressors()
        assert df['peak_velocity'].isna().all()


# =============================================================================
# PhotometrySessionGroup.session_permutation_test Tests
# =============================================================================

def _make_perm_group(trial_data, target_nm='VTA-DA'):
    """Build a synthetic group of single-region recordings, one per eid.

    Parameters
    ----------
    trial_data : dict[str, dict]
        Maps eid -> column dict of synthetic arrays for that unit. The keys
        become PS attribute names when paired with ``_attr_prep``.
    target_nm : str
        Shared ``target_NM`` for all recordings (one donor pool).

    Returns
    -------
    PhotometrySessionGroup
        Built directly from session rows; ``session_permutation_test``
        constructs each unit's PS inline (no H5/network), and ``_attr_prep``
        supplies its data.
    """
    from iblnm.data import PhotometrySessionGroup

    rows = [{
        'eid': eid, 'subject': f'subj-{i}', 'brain_region': 'VTA',
        'hemisphere': 'l', 'target_NM': target_nm, 'NM': 'DA',
        'session_type': 'biased', 'start_time': '2024-01-01T10:00:00',
        'number': 1, 'task_protocol': 'biased_protocol',
    } for i, eid in enumerate(trial_data)]
    return PhotometrySessionGroup(pd.DataFrame(rows), one=MagicMock())


def _attr_prep(trial_data):
    """Build a ``prep_fn`` that attaches each eid's arrays as PS attributes.

    Keys absent from a unit's dict are left unset, so resolving a
    ``fixed_var``/``swapped_var`` naming them raises (the failure path).
    """
    def prep(ps):
        for name, values in trial_data[ps.eid].items():
            setattr(ps, name, np.asarray(values))
        return ps
    return prep


class TestSessionPermutationTest:
    """Tests for PhotometrySessionGroup.session_permutation_test."""

    def test_observed_matches_hand_computation(self):
        rng = np.random.default_rng(0)
        trial_data = {
            f'eid-{i}': {'rt': rng.random(10), 'signal': rng.random(10)}
            for i in range(3)
        }
        group = _make_perm_group(trial_data)
        result = group.session_permutation_test(
            _attr_prep(trial_data),
            lambda a, b: {'corr': np.corrcoef(a, b)[0, 1]},
            fixed_var=['rt'], swapped_var=['signal'],
            statistic_key='corr', n_iter=10,
        )
        row = result[result['eid'] == 'eid-0'].iloc[0]
        expected = np.corrcoef(trial_data['eid-0']['rt'],
                               trial_data['eid-0']['signal'])[0, 1]
        assert row['observed_corr'] == pytest.approx(expected)

    def test_null_uses_common_min_length(self):
        trial_data = {
            'eid-0': {'rt': np.arange(10), 'signal': np.arange(10)},
            'eid-1': {'rt': np.arange(6), 'signal': np.arange(6)},
        }
        group = _make_perm_group(trial_data)
        result = group.session_permutation_test(
            _attr_prep(trial_data),
            lambda *arrays: {'len': len(arrays[0])},
            fixed_var=['rt'], swapped_var=['signal'],
            statistic_key='len', n_iter=20,
        )
        row = result[result['eid'] == 'eid-0'].iloc[0]
        # target has 10 trials, only donor (eid-1) has 6 → min length 6
        assert np.all(row['null_len'] == 6)

    def test_null_has_n_iter_length(self):
        rng = np.random.default_rng(0)
        trial_data = {
            f'eid-{i}': {'rt': rng.random(8), 'signal': rng.random(8)}
            for i in range(3)
        }
        group = _make_perm_group(trial_data)
        result = group.session_permutation_test(
            _attr_prep(trial_data),
            lambda a, b: {'corr': np.corrcoef(a, b)[0, 1]},
            fixed_var=['rt'], swapped_var=['signal'],
            statistic_key='corr', n_iter=37,
        )
        assert all(len(null) == 37 for null in result['null_corr'])

    def test_reproducible_with_seed(self):
        # each unit's signal is a constant marker = its index, so the null
        # sequence is the sequence of drawn donor markers
        trial_data = {
            f'eid-{i}': {'rt': np.zeros(5), 'signal': np.full(5, i)}
            for i in range(3)
        }
        kwargs = dict(fixed_var=['rt'], swapped_var=['signal'],
                      statistic_key='marker', n_iter=50)
        group = _make_perm_group(trial_data)
        prep, stat = _attr_prep(trial_data), lambda a, b: {'marker': b[0]}
        def target_null(result):
            return result[result['eid'] == 'eid-0'].iloc[0]['null_marker']

        null_a = group.session_permutation_test(prep, stat, seed=42, **kwargs)
        null_b = group.session_permutation_test(prep, stat, seed=42, **kwargs)
        null_c = group.session_permutation_test(prep, stat, seed=7, **kwargs)
        assert np.array_equal(target_null(null_a), target_null(null_b))
        assert not np.array_equal(target_null(null_a), target_null(null_c))

    def test_donor_never_self(self):
        # 2-unit pool: the only valid donor for eid-0 is eid-1
        trial_data = {
            'eid-0': {'rt': np.zeros(5), 'signal': np.zeros(5)},
            'eid-1': {'rt': np.zeros(5), 'signal': np.ones(5)},
        }
        group = _make_perm_group(trial_data)
        result = group.session_permutation_test(
            _attr_prep(trial_data),
            lambda a, b: {'marker': b[0]},
            fixed_var=['rt'], swapped_var=['signal'],
            statistic_key='marker', n_iter=30,
        )
        row = result[result['eid'] == 'eid-0'].iloc[0]
        # self (eid-0) marker is 0; donor (eid-1) marker is 1
        assert np.all(row['null_marker'] == 1)

    def test_p_value_matches_helper(self):
        from iblnm.analysis import permutation_pvalue
        rng = np.random.default_rng(0)
        trial_data = {
            f'eid-{i}': {'rt': rng.random(8), 'signal': rng.random(8)}
            for i in range(3)
        }
        group = _make_perm_group(trial_data)
        result = group.session_permutation_test(
            _attr_prep(trial_data),
            lambda a, b: {'corr': np.corrcoef(a, b)[0, 1]},
            fixed_var=['rt'], swapped_var=['signal'],
            statistic_key='corr', n_iter=40, alternative='greater',
        )
        row = result[result['eid'] == 'eid-0'].iloc[0]
        assert row['p_value'] == pytest.approx(
            permutation_pvalue(row['observed_corr'], row['null_corr'], 'greater'))

    def test_extractor_error_yields_nan_row(self):
        trial_data = {
            'eid-0': {'rt': np.arange(6), 'signal': np.arange(6)},
            'eid-1': {'rt': np.arange(6), 'signal': np.arange(6)},
            'eid-2': {'signal': np.arange(6)},  # no 'rt' → fixed resolve fails
        }
        group = _make_perm_group(trial_data)
        result = group.session_permutation_test(
            _attr_prep(trial_data),
            lambda a, b: {'corr': np.corrcoef(a, b)[0, 1]},
            fixed_var=['rt'], swapped_var=['signal'],
            statistic_key='corr', n_iter=10,
        )
        assert len(result) == 3
        bad = result[result['eid'] == 'eid-2'].iloc[0]
        # failed unit: p_value NaN and the observed/null columns filled NaN
        assert np.isnan(bad['p_value'])
        assert pd.isna(bad['observed_corr']) and pd.isna(bad['null_corr'])
        good = result[result['eid'] == 'eid-0'].iloc[0]
        assert not np.isnan(good['observed_corr'])

    def test_error_column_reports_failing_stage(self):
        trial_data = {
            f'eid-{i}': {'rt': np.arange(6), 'signal': np.arange(6)}
            for i in range(3)
        }
        group = _make_perm_group(trial_data)

        def prep(ps):
            if ps.eid == 'eid-2':
                raise RuntimeError("boom")
            for name, values in trial_data[ps.eid].items():
                setattr(ps, name, np.asarray(values))
            return ps

        result = group.session_permutation_test(
            prep,
            lambda a, b: {'corr': np.corrcoef(a, b)[0, 1]},
            fixed_var=['rt'], swapped_var=['signal'],
            statistic_key='corr', n_iter=5,
        )
        bad = result[result['eid'] == 'eid-2'].iloc[0]
        good = result[result['eid'] == 'eid-0'].iloc[0]
        assert 'prep' in bad['error']
        assert good['error'] is None

    def test_error_column_reports_stat_stage(self):
        trial_data = {
            f'eid-{i}': {'rt': np.arange(6), 'signal': np.arange(6)}
            for i in range(3)
        }
        group = _make_perm_group(trial_data)

        def stat(a, b):
            raise ValueError("nope")

        result = group.session_permutation_test(
            _attr_prep(trial_data), stat,
            fixed_var=['rt'], swapped_var=['signal'],
            statistic_key='corr', n_iter=5,
        )
        assert all('stat' in err for err in result['error'])

    def test_output_columns_for_recordings(self):
        rng = np.random.default_rng(0)
        trial_data = {
            f'eid-{i}': {'rt': rng.random(8), 'signal': rng.random(8)}
            for i in range(3)
        }
        group = _make_perm_group(trial_data)
        result = group.session_permutation_test(
            _attr_prep(trial_data),
            lambda a, b: {'corr': np.corrcoef(a, b)[0, 1]},
            fixed_var=['rt'], swapped_var=['signal'],
            statistic_key='corr', n_iter=5,
        )
        expected = list(group.recordings.columns) + [
            'error', 'p_value', 'observed_corr', 'null_corr']
        assert list(result.columns) == expected
        assert len(result) == 3

    def test_multi_output_records_every_key(self):
        rng = np.random.default_rng(0)
        trial_data = {
            f'eid-{i}': {'rt': rng.random(8), 'signal': rng.random(8)}
            for i in range(3)
        }
        group = _make_perm_group(trial_data)
        result = group.session_permutation_test(
            _attr_prep(trial_data),
            lambda a, b: {'corr': np.corrcoef(a, b)[0, 1], 'n': len(a)},
            fixed_var=['rt'], swapped_var=['signal'],
            statistic_key='corr', n_iter=5,
        )
        row = result[result['eid'] == 'eid-0'].iloc[0]
        expected = np.corrcoef(trial_data['eid-0']['rt'],
                               trial_data['eid-0']['signal'])[0, 1]
        # both returned keys carry observed + null; only 'corr' gets a p_value
        assert row['observed_corr'] == pytest.approx(expected)
        assert row['observed_n'] == 8
        assert len(row['null_corr']) == 5 and len(row['null_n']) == 5
        assert 'p_value' in result.columns
        assert {'observed_corr', 'null_corr',
                'observed_n', 'null_n'} <= set(result.columns)

    def test_missing_statistic_key_raises(self):
        rng = np.random.default_rng(0)
        trial_data = {
            f'eid-{i}': {'rt': rng.random(8), 'signal': rng.random(8)}
            for i in range(3)
        }
        group = _make_perm_group(trial_data)
        with pytest.raises(KeyError):
            group.session_permutation_test(
                _attr_prep(trial_data),
                lambda a, b: {'corr': np.corrcoef(a, b)[0, 1]},
                fixed_var=['rt'], swapped_var=['signal'],
                statistic_key='slope', n_iter=5,
            )


def _two_block_encoding_fit():
    """An EncodingFit on synthetic data: a 'signal' block carries all the
    variance, a 'noise' block is pure noise uncorrelated with the target."""
    from iblnm import analysis

    rng = np.random.default_rng(0)
    n = 600
    tvec = np.linspace(0, 60, n)
    signal = rng.standard_normal((n, 3))
    noise = rng.standard_normal((n, 2))
    weights = np.array([2.0, -1.5, 1.0])
    y = signal @ weights + 0.05 * rng.standard_normal(n)
    design = np.hstack([signal, noise])
    slices = {'signal': slice(0, 3), 'noise': slice(3, 5)}
    target = pd.Series(y, index=tvec)
    return analysis.fit_encoding_model(design, target, slices,
                                       alphas=[1.0], cv=3)


class TestDeltaRSquared:
    """PhotometrySession.delta_r_squared — leave-one-regressor-out ΔR²."""

    def _session(self, mock_session_series):
        from iblnm.data import PhotometrySession
        return PhotometrySession(mock_session_series, one=MagicMock(),
                                 load_data=False)

    def test_signal_block_dominates_noise_block(self, mock_session_series):
        """In-sample: the signal block has the largest ΔR², the noise block ~0."""
        fit = _two_block_encoding_fit()
        deltas = self._session(mock_session_series).delta_r_squared(fit)

        assert list(deltas.index) == ['signal', 'noise']  # sorted descending
        assert deltas['signal'] > 0.5
        assert deltas['noise'] == pytest.approx(0.0, abs=1e-3)

    def test_insample_full_reference_is_fit_r2(self, mock_session_series):
        """cv=None uses fit.r2 as the full reference: ΔR² of a block equals
        fit.r2 minus the independently refit reduced R²."""
        from sklearn.linear_model import Ridge
        from sklearn.metrics import r2_score

        fit = _two_block_encoding_fit()
        deltas = self._session(mock_session_series).delta_r_squared(fit)

        span = fit.slices['noise']
        keep = np.ones(fit.design.shape[1], dtype=bool)
        keep[span] = False
        reduced = fit.design[:, keep]
        model = Ridge(alpha=fit.alpha).fit(reduced, fit.target)
        reduced_r2 = r2_score(fit.target, reduced @ model.coef_.T + model.intercept_)

        assert deltas['noise'] == pytest.approx(fit.r2 - reduced_r2)

    def test_cv_returns_series_over_all_blocks(self, mock_session_series):
        """cv=2 runs and returns a Series indexed by every block name."""
        fit = _two_block_encoding_fit()
        deltas = self._session(mock_session_series).delta_r_squared(fit, cv=2)

        assert set(deltas.index) == set(fit.slices)


class TestAssembleSessionPvalueTable:
    """assemble_session_pvalue_table — per-recording drop-one permutation p."""

    def _observed(self, rows):
        """Build an observed drop-one frame from (eid, subject, delta_r2) rows."""
        return pd.DataFrame(
            [{'eid': eid, 'subject': subject, 'target_NM': 'VTA-DA',
              'brain_region': 'VTA', 'event': 'feedback', 'predictor': 'reward',
              'r2': 0.3, 'delta_r2': delta_r2, 'n_trials': 100}
             for eid, subject, delta_r2 in rows]
        )

    def test_one_row_per_scorable_observed_row(self):
        """Every observed row with a null vector yields one output row carrying
        its identity columns and its observed ΔR² unchanged."""
        from iblnm.data import assemble_session_pvalue_table

        observed = self._observed([('e1', 'm1', 0.10), ('e2', 'm1', 0.06)])
        null_vectors = {
            ('e1', 'feedback', 'reward'): np.full(4, 0.02),
            ('e2', 'feedback', 'reward'): np.full(4, 0.01),
        }

        table = assemble_session_pvalue_table(observed, null_vectors)

        assert list(table['eid']) == ['e1', 'e2']
        assert list(table['subject']) == ['m1', 'm1']
        assert list(table['target_NM']) == ['VTA-DA', 'VTA-DA']
        assert list(table['brain_region']) == ['VTA', 'VTA']
        assert list(table['event']) == ['feedback', 'feedback']
        assert list(table['predictor']) == ['reward', 'reward']
        assert table['delta_r2'].tolist() == pytest.approx([0.10, 0.06])

    def test_row_without_null_vector_is_skipped(self):
        """An unscorable recording — no entry in null_vectors — contributes no
        output row, so its dot has no session p-value to color from."""
        from iblnm.data import assemble_session_pvalue_table

        observed = self._observed([('e1', 'm1', 0.10), ('e2', 'm2', 0.20)])
        null_vectors = {('e2', 'feedback', 'reward'): np.full(4, 0.01)}

        table = assemble_session_pvalue_table(observed, null_vectors)

        assert list(table['eid']) == ['e2']

    def test_pvalue_matches_permutation_primitive(self):
        """p_value is analysis.permutation_pvalue against that row's null. e1's
        ΔR² exceeds every null draw, so it sits at the add-one floor
        1/(len(null)+1); e2's ΔR² is beaten by half its null."""
        from iblnm import analysis
        from iblnm.data import assemble_session_pvalue_table

        observed = self._observed([('e1', 'm1', 0.10), ('e2', 'm2', 0.01)])
        beaten_null = np.array([0.0, 0.0, 0.02, 0.03])
        null_vectors = {
            ('e1', 'feedback', 'reward'): np.full(4, 0.02),
            ('e2', 'feedback', 'reward'): beaten_null,
        }

        table = assemble_session_pvalue_table(observed, null_vectors)

        by_eid = table.set_index('eid')['p_value']
        assert by_eid['e1'] == pytest.approx(1 / 5)
        assert by_eid['e2'] == pytest.approx(
            analysis.permutation_pvalue(0.01, beaten_null, 'greater'))

    def test_alternative_is_forwarded(self):
        """The alternative argument reaches the primitive: the same row scored
        'less' gives the lower-tail p, not the upper-tail one."""
        from iblnm import analysis
        from iblnm.data import assemble_session_pvalue_table

        observed = self._observed([('e1', 'm1', 0.10)])
        null = np.array([0.0, 0.05, 0.2, 0.3])
        null_vectors = {('e1', 'feedback', 'reward'): null}

        table = assemble_session_pvalue_table(observed, null_vectors,
                                              alternative='less')

        assert table.iloc[0]['p_value'] == pytest.approx(
            analysis.permutation_pvalue(0.10, null, 'less'))

    def test_output_columns_match_schema_with_unfilled_qvalue(self):
        """Columns equal RESPONSE_OLS_SESSION_PVAL_COLUMNS in order, and
        q_value is left NaN for the caller's FDR correction to fill."""
        from iblnm.data import (assemble_session_pvalue_table,
                                RESPONSE_OLS_SESSION_PVAL_COLUMNS)

        observed = self._observed([('e1', 'm1', 0.10)])
        null_vectors = {('e1', 'feedback', 'reward'): np.full(4, 0.02)}

        table = assemble_session_pvalue_table(observed, null_vectors)

        assert list(table.columns) == RESPONSE_OLS_SESSION_PVAL_COLUMNS
        assert table['q_value'].isna().all()

    @pytest.mark.parametrize('empty', ['observed', 'null_vectors'])
    def test_empty_input_gives_empty_schema_frame(self, empty):
        """Either input empty gives an empty frame with the schema columns."""
        from iblnm.data import (assemble_session_pvalue_table,
                                RESPONSE_OLS_SESSION_PVAL_COLUMNS)

        observed = self._observed(
            [] if empty == 'observed' else [('e1', 'm1', 0.10)])
        null_vectors = ({} if empty == 'null_vectors'
                        else {('e1', 'feedback', 'reward'): np.full(4, 0.02)})

        table = assemble_session_pvalue_table(observed, null_vectors)

        assert len(table) == 0
        assert list(table.columns) == RESPONSE_OLS_SESSION_PVAL_COLUMNS


class TestAssembleMousePvalueTable:
    """assemble_mouse_pvalue_table — per-mouse drop-one permutation p."""

    def _observed(self, rows):
        """Build an observed drop-one frame from (eid, subject, delta_r2) rows."""
        return pd.DataFrame(
            [{'eid': eid, 'subject': subject, 'target_NM': 'VTA-DA',
              'brain_region': 'VTA', 'event': 'feedback', 'predictor': 'reward',
              'r2': 0.3, 'delta_r2': delta_r2, 'n_trials': 100}
             for eid, subject, delta_r2 in rows]
        )

    def test_single_mouse_two_sessions_pool_by_resampling(self):
        """Two sessions (ΔR² 0.10, 0.06) with ragged, constant null vectors pool
        to mean 0.08; every bootstrap draw is mean(0.02, 0.01) = 0.015 < 0.08, so
        p hits its floor 1/(n_bootstrap+1). Ragged lengths do not raise (the
        crash this fixes)."""
        from iblnm.data import assemble_mouse_pvalue_table

        observed = self._observed([('e1', 'm1', 0.10), ('e2', 'm1', 0.06)])
        null_vectors = {
            ('e1', 'feedback', 'reward'): np.full(3, 0.02),
            ('e2', 'feedback', 'reward'): np.full(2, 0.01),
        }

        table = assemble_mouse_pvalue_table(
            observed, null_vectors, n_bootstrap=99, random_state=0)

        assert len(table) == 1
        row = table.iloc[0]
        assert row['mean_delta_r2'] == pytest.approx(0.08)
        assert row['p_value'] == pytest.approx(1 / 100)
        assert row['n_sessions'] == 2
        assert 'n_donors' not in table.columns

    def test_two_mice_pool_only_their_own_sessions(self):
        """Two mice in the same cell yield two rows; each mouse's mean pools
        only its own sessions."""
        from iblnm.data import assemble_mouse_pvalue_table

        observed = self._observed([('e1', 'm1', 0.10), ('e2', 'm1', 0.06),
                                   ('e3', 'm2', 0.20)])
        null_vectors = {
            ('e1', 'feedback', 'reward'): np.array([0.02, 0.01, 0.03]),
            ('e2', 'feedback', 'reward'): np.array([0.015, 0.005, 0.02]),
            ('e3', 'feedback', 'reward'): np.array([0.04, 0.02, 0.05]),
        }

        table = assemble_mouse_pvalue_table(observed, null_vectors)

        by_subject = table.set_index('subject')
        assert set(by_subject.index) == {'m1', 'm2'}
        assert by_subject.loc['m1', 'mean_delta_r2'] == pytest.approx(0.08)
        assert by_subject.loc['m1', 'n_sessions'] == 2
        assert by_subject.loc['m2', 'mean_delta_r2'] == pytest.approx(0.20)
        assert by_subject.loc['m2', 'n_sessions'] == 1

    def test_output_columns_match_schema(self):
        """Output columns equal RESPONSE_OLS_MOUSE_PVAL_COLUMNS in order."""
        from iblnm.data import (assemble_mouse_pvalue_table,
                                RESPONSE_OLS_MOUSE_PVAL_COLUMNS)

        observed = self._observed([('e1', 'm1', 0.10)])
        null_vectors = {('e1', 'feedback', 'reward'): np.array([0.02, 0.01])}

        table = assemble_mouse_pvalue_table(observed, null_vectors)

        assert list(table.columns) == RESPONSE_OLS_MOUSE_PVAL_COLUMNS
        # q_value sits between p_value and n_sessions and is left for the
        # caller's FDR correction to fill, as at session grain.
        assert RESPONSE_OLS_MOUSE_PVAL_COLUMNS.index('q_value') == (
            RESPONSE_OLS_MOUSE_PVAL_COLUMNS.index('p_value') + 1)
        assert table['q_value'].isna().all()

    def test_group_with_no_null_vectors_is_skipped(self):
        """A cell whose sessions have no null vectors produces no row; sessions
        that do have vectors still pool."""
        from iblnm.data import assemble_mouse_pvalue_table

        observed = self._observed([('e1', 'm1', 0.10), ('e2', 'm2', 0.20)])
        null_vectors = {('e2', 'feedback', 'reward'): np.array([0.04, 0.05])}

        table = assemble_mouse_pvalue_table(observed, null_vectors)

        assert list(table['subject']) == ['m2']
        assert table.iloc[0]['n_sessions'] == 1


class TestResponseOlsDroponePermutation:
    """PhotometrySessionGroup.response_ols_dropone_permutation orchestration."""

    def _group(self):
        """Three single-region recordings, one target_NM, two mice (m1 has
        e1+e2, m2 has e3)."""
        from iblnm.data import PhotometrySessionGroup
        rows = [{
            'eid': eid, 'subject': subject, 'brain_region': 'VTA',
            'hemisphere': 'l', 'target_NM': 'VTA-DA', 'NM': 'DA',
            'session_type': 'biased', 'start_time': '2024-01-01T10:00:00',
            'number': 1, 'task_protocol': 'biased_protocol',
        } for eid, subject in [('e1', 'm1'), ('e2', 'm1'), ('e3', 'm2')]]
        return PhotometrySessionGroup(pd.DataFrame(rows), one=MagicMock())

    def _observed(self):
        return pd.DataFrame(
            [{'eid': eid, 'subject': subject, 'target_NM': 'VTA-DA',
              'brain_region': 'VTA', 'event': 'feedback_times',
              'predictor': 'reward', 'r2': 0.3, 'delta_r2': delta,
              'n_trials': 100}
             for eid, subject, delta in [('e1', 'm1', 0.10),
                                         ('e2', 'm1', 0.06),
                                         ('e3', 'm2', 0.20)]])

    _FORMULAS = {'full': '{response} ~ reward + contrast',
                 'reward': '{response} ~ contrast'}

    def _patch(self, group, monkeypatch):
        """Stub the H5-loading coded-frame gather and the null primitive so
        neither H5 nor statsmodels is invoked. Each canned frame carries a
        ``tag`` column equal to its eid. Returns the list each call records:
        ``(focal_tag, donor_tags, predictor)``."""
        scorable = [(eid, 'VTA-DA', 'feedback_times', pd.DataFrame({'tag': [eid]}))
                    for eid in ['e1', 'e2', 'e3']]
        monkeypatch.setattr(group, '_gather_coded_frames',
                            lambda *a, **k: scorable)
        calls = []

        def fake_null(focal_df, donor_dfs, full_formula, reduced_formula,
                      predictor, response_col='response', *, rng,
                      n_bootstrap=1000):
            calls.append((focal_df['tag'].iloc[0],
                          {d['tag'].iloc[0] for d in donor_dfs}, predictor))
            return np.full(n_bootstrap, 0.01)

        monkeypatch.setattr('iblnm.analysis.permutation_null_delta_r2',
                            fake_null)
        return calls

    def test_donors_span_cohorts_within_event(self, monkeypatch):
        """A focal recording's donors are every other recording at the same
        event, regardless of target_NM; different-event recordings are excluded.
        """
        group = self._group()
        group.response_ols_dropone_results = self._observed()
        # e1/e2 VTA-DA and e3 DR-5HT all at feedback; e4 VTA-DA at stimOn.
        scorable = [
            ('e1', 'VTA-DA', 'feedback_times', pd.DataFrame({'tag': ['e1']})),
            ('e2', 'VTA-DA', 'feedback_times', pd.DataFrame({'tag': ['e2']})),
            ('e3', 'DR-5HT', 'feedback_times', pd.DataFrame({'tag': ['e3']})),
            ('e4', 'VTA-DA', 'stimOnTrigger_times', pd.DataFrame({'tag': ['e4']})),
        ]
        monkeypatch.setattr(group, '_gather_coded_frames',
                            lambda *a, **k: scorable)
        calls = []

        def fake_null(focal_df, donor_dfs, full_formula, reduced_formula,
                      predictor, response_col='response', *, rng,
                      n_bootstrap=1000):
            calls.append((focal_df['tag'].iloc[0],
                          {d['tag'].iloc[0] for d in donor_dfs}, predictor))
            return np.full(n_bootstrap, 0.01)

        monkeypatch.setattr('iblnm.analysis.permutation_null_delta_r2',
                            fake_null)

        group.response_ols_dropone_permutation(
            self._FORMULAS, events=['feedback_times', 'stimOnTrigger_times'])

        donors_for_e1 = next(d for f, d, _ in calls if f == 'e1')
        assert donors_for_e1 == {'e2', 'e3'}   # cross-cohort, same event
        assert 'e4' not in donors_for_e1       # different event excluded

    def test_grain_and_columns(self, monkeypatch):
        """Returns (session, mouse) tables: the first at session grain, one row
        per scorable recording, the second the existing per-mouse table."""
        from iblnm.data import (RESPONSE_OLS_MOUSE_PVAL_COLUMNS,
                                RESPONSE_OLS_SESSION_PVAL_COLUMNS)
        group = self._group()
        group.response_ols_dropone_results = self._observed()
        self._patch(group, monkeypatch)

        session_table, table = group.response_ols_dropone_permutation(
            self._FORMULAS, events=['feedback_times'])

        assert list(session_table.columns) == RESPONSE_OLS_SESSION_PVAL_COLUMNS
        assert list(session_table['eid']) == ['e1', 'e2', 'e3']

        assert list(table.columns) == RESPONSE_OLS_MOUSE_PVAL_COLUMNS
        assert set(table['subject']) == {'m1', 'm2'}
        assert set(zip(table['target_NM'], table['event'],
                       table['predictor'])) == {
            ('VTA-DA', 'feedback_times', 'reward')}

    def test_empty_null_session_excluded(self, monkeypatch):
        """A recording whose primitive returns an empty null vector is dropped
        from both grains: no session row of its own, and it does not count
        toward its mouse's ``n_sessions``."""
        group = self._group()
        group.response_ols_dropone_results = self._observed()
        scorable = [(eid, 'VTA-DA', 'feedback_times', pd.DataFrame({'tag': [eid]}))
                    for eid in ['e1', 'e2', 'e3']]
        monkeypatch.setattr(group, '_gather_coded_frames',
                            lambda *a, **k: scorable)

        def fake_null(focal_df, donor_dfs, full_formula, reduced_formula,
                      predictor, response_col='response', *, rng,
                      n_bootstrap=1000):
            if focal_df['tag'].iloc[0] == 'e2':
                return np.array([])
            return np.full(n_bootstrap, 0.01)

        monkeypatch.setattr('iblnm.analysis.permutation_null_delta_r2',
                            fake_null)

        session_table, table = group.response_ols_dropone_permutation(
            self._FORMULAS, events=['feedback_times'], n_bootstrap=50)

        assert list(session_table['eid']) == ['e1', 'e3']
        m1_sessions = table.loc[table['subject'] == 'm1', 'n_sessions']
        assert m1_sessions.tolist() == [1]

    def test_rng_created_once_and_reproducible(self, monkeypatch):
        """A single rng is threaded through every primitive call — so per-call
        draws differ — and reruns with the same seed reproduce the table."""
        group = self._group()
        group.response_ols_dropone_results = self._observed()
        scorable = [(eid, 'VTA-DA', 'feedback_times', pd.DataFrame({'tag': [eid]}))
                    for eid in ['e1', 'e2', 'e3']]
        monkeypatch.setattr(group, '_gather_coded_frames',
                            lambda *a, **k: scorable)
        draws = {}

        def fake_null(focal_df, donor_dfs, full_formula, reduced_formula,
                      predictor, response_col='response', *, rng,
                      n_bootstrap=1000):
            vector = rng.random(n_bootstrap)
            draws.setdefault(focal_df['tag'].iloc[0], []).append(vector)
            return vector

        monkeypatch.setattr('iblnm.analysis.permutation_null_delta_r2',
                            fake_null)

        session1, table1 = group.response_ols_dropone_permutation(
            self._FORMULAS, events=['feedback_times'], n_bootstrap=32,
            random_state=7)
        first_draws = {eid: v[0] for eid, v in draws.items()}
        draws.clear()
        session2, table2 = group.response_ols_dropone_permutation(
            self._FORMULAS, events=['feedback_times'], n_bootstrap=32,
            random_state=7)

        # One advancing rng: the three per-recording draws are all distinct.
        stacked = np.vstack(list(first_draws.values()))
        assert len({tuple(row) for row in stacked}) == len(first_draws)
        # Same seed reproduces both tables.
        pd.testing.assert_frame_equal(table1, table2)
        pd.testing.assert_frame_equal(session1, session2)


# =============================================================================
# load_states Tests
# =============================================================================

class TestLoadStates:
    """Tests for PhotometrySession.load_states."""

    def _write_posteriors(self, ddm_dir, subject, rows):
        """Write a `{subject}_K2_posteriors.csv` with `rows` (list of dicts)."""
        fpath = ddm_dir / f'{subject}_K2_posteriors.csv'
        pd.DataFrame(rows).to_csv(fpath, index=False)
        return fpath

    def _trials(self):
        """Six trials; two (indices 1, 3) are absent from the fit CSV.

        Kept trials 0, 2, 4, 5 have RTs 0.5, 0.3, 0.7, 0.2 s and |contrast|
        100, 12.5, 25, 6.25 %; the two dropped trials carry RTs (0.4, 10.5)
        that appear in no CSV row, so the ordered rt-alignment skips them.
        H5 ``signed_contrast`` is in percent.
        """
        return pd.DataFrame({
            'choice':          [ 1,    0,     -1,    1,     -1,    1   ],
            'stimOn_times':    [ 0.0,  1.0,   2.0,   3.0,   4.0,   5.0 ],
            'response_times':  [ 0.5,  1.4,   2.3,   13.5,  4.7,   5.2 ],
            'signed_contrast': [ 100,  25.0,  -12.5, 6.25,  -25.0, 6.25],
        })

    def _kept_rows(self, eid):
        """CSV block for the four kept trials, shuffled by trial_in_dataset.

        Kept order (by stimOn_times) is trials 0, 2, 4, 5 with rt 0.5, 0.3,
        0.7, 0.2; trial_in_dataset encodes that chronological order. CSV
        ``signed_contrast`` is a fraction (|value|*100 must equal the H5
        percent |contrast|); its sign may differ from H5 (collaborator coding).
        """
        return [
            {'eid': eid, 'trial_in_dataset': 2, 'rt': 0.7, 'signed_contrast': 0.25,
             'map_state': 2, 'state_1': 0.1, 'state_2': 0.9},
            {'eid': eid, 'trial_in_dataset': 0, 'rt': 0.5, 'signed_contrast': 1.0,
             'map_state': 2, 'state_1': 0.1, 'state_2': 0.9},
            {'eid': eid, 'trial_in_dataset': 3, 'rt': 0.2, 'signed_contrast': 0.0625,
             'map_state': 1, 'state_1': 0.6, 'state_2': 0.4},
            {'eid': eid, 'trial_in_dataset': 1, 'rt': 0.3, 'signed_contrast': -0.125,
             'map_state': 1, 'state_1': 0.6, 'state_2': 0.4},
        ]

    def test_aligns_states_to_kept_trials(self, mock_photometry_session,
                                          tmp_path, monkeypatch):
        """States land on kept trials (by stimOn order), NaN on dropped ones."""
        ps = mock_photometry_session
        ps.trials = self._trials()
        monkeypatch.setattr('iblnm.data.DDM_HMM_DIR', tmp_path)
        rows = self._kept_rows(ps.eid) + [
            # A row for a different eid that must be ignored.
            {'eid': 'other-eid', 'trial_in_dataset': 0, 'rt': 9.9,
             'map_state': 1, 'state_1': 0.5, 'state_2': 0.5},
        ]
        self._write_posteriors(tmp_path, ps.subject, rows)

        ps.load_states()

        assert list(ps.states.index) == list(ps.trials.index)
        assert list(ps.states.columns) == ['map_state', 'state_1', 'state_2']
        # Kept trials 0, 2, 4, 5 carry map_state in stimOn order (0.5→2,
        # 0.3→1... but here mapped by trial_in_dataset: t0→2, t2→1, t4→2, t5→1).
        assert ps.states.loc[[0, 2, 4, 5], 'map_state'].tolist() == [2, 1, 2, 1]
        assert ps.states.loc[0, 'state_2'] == 0.9
        assert ps.states.loc[5, 'state_1'] == 0.6
        # Dropped trials 1 (no-go) and 3 (RT>=10) are NaN.
        assert ps.states.loc[[1, 3]].isna().all().all()

    def test_rt_no_match_raises(self, mock_photometry_session, tmp_path,
                                monkeypatch):
        """A CSV rt with no ordered match in the trials fails loud."""
        ps = mock_photometry_session
        ps.trials = self._trials()
        monkeypatch.setattr('iblnm.data.DDM_HMM_DIR', tmp_path)
        rows = self._kept_rows(ps.eid)
        rows[0]['rt'] = 99.0  # no trial has this RT
        self._write_posteriors(tmp_path, ps.subject, rows)

        with pytest.raises(ValueError):
            ps.load_states()

    def test_contrast_mismatch_raises(self, mock_photometry_session, tmp_path,
                                      monkeypatch):
        """An rt-matched trial whose |contrast| disagrees fails loud."""
        ps = mock_photometry_session
        ps.trials = self._trials()
        monkeypatch.setattr('iblnm.data.DDM_HMM_DIR', tmp_path)
        rows = self._kept_rows(ps.eid)
        rows[0]['signed_contrast'] = 0.5  # 50 % vs H5 25 % on the same trial
        self._write_posteriors(tmp_path, ps.subject, rows)

        with pytest.raises(ValueError):
            ps.load_states()

    def test_eid_absent_leaves_states_none(self, mock_photometry_session,
                                           tmp_path, monkeypatch):
        """A posteriors file with no rows for this eid leaves states None."""
        ps = mock_photometry_session
        ps.trials = self._trials()
        monkeypatch.setattr('iblnm.data.DDM_HMM_DIR', tmp_path)
        self._write_posteriors(tmp_path, ps.subject,
                               self._kept_rows('other-eid'))

        ps.load_states()

        assert ps.states is None
