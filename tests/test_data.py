"""Tests for iblnm.data module."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from iblnm.util import contrast_transform


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
def mock_photometry_session(mock_session_series, mock_photometry_data):
    """PhotometrySession with injected mock data."""
    from iblnm.data import PhotometrySession

    mock_one = MagicMock()
    session = PhotometrySession(mock_session_series, one=mock_one, load_data=False)

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

    def test_save_deduplicates_errors(self, mock_session_series, tmp_path):
        """Duplicate errors (same eid/type/message) are written only once."""
        from iblnm.data import PhotometrySession
        from iblnm.validation import InvalidStrain
        mock_one = MagicMock()
        ps = PhotometrySession(mock_session_series, one=mock_one, load_data=False)
        fpath = tmp_path / f'{ps.eid}.h5'

        # Log the same error three times (simulates repeated pipeline runs)
        for _ in range(3):
            try:
                raise InvalidStrain("bad strain")
            except InvalidStrain as e:
                ps.log_error(e)

        ps.save_h5(fpath, groups=['metadata', 'errors'])

        ps2 = PhotometrySession(mock_session_series, one=mock_one, load_data=False)
        ps2.load_h5(fpath, groups=['errors'])
        assert len(ps2.errors) == 1
        assert ps2.errors[0]['error_type'] == 'InvalidStrain'

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

    def test_save_errors_append_merges_existing(self, mock_session_series, tmp_path):
        """Append mode merges new errors with existing ones in H5."""
        from iblnm.data import PhotometrySession
        from iblnm.validation import InvalidStrain, MissingRawData
        mock_one = MagicMock()
        fpath = tmp_path / f"{mock_session_series['eid']}.h5"

        # First session: write with one error
        ps1 = PhotometrySession(mock_session_series, one=mock_one, load_data=False)
        try:
            raise InvalidStrain("bad strain")
        except InvalidStrain as e:
            ps1.log_error(e)
        ps1.save_h5(fpath, groups=['metadata', 'errors'], mode='w')

        # Second session: append a different error
        ps2 = PhotometrySession(mock_session_series, one=mock_one, load_data=False)
        try:
            raise MissingRawData("no raw data")
        except MissingRawData as e:
            ps2.log_error(e)
        ps2.save_h5(fpath, groups=['errors'], mode='a')

        # Both errors should be present
        ps3 = PhotometrySession(mock_session_series, one=mock_one, load_data=False)
        ps3.load_h5(fpath, groups=['errors'])
        error_types = {e['error_type'] for e in ps3.errors}
        assert error_types == {'InvalidStrain', 'MissingRawData'}
        assert len(ps3.errors) == 2

    def test_save_errors_append_deduplicates(self, mock_session_series, tmp_path):
        """Append mode deduplicates errors across existing and new."""
        from iblnm.data import PhotometrySession
        from iblnm.validation import InvalidStrain
        mock_one = MagicMock()
        fpath = tmp_path / f"{mock_session_series['eid']}.h5"

        # First write
        ps1 = PhotometrySession(mock_session_series, one=mock_one, load_data=False)
        try:
            raise InvalidStrain("bad strain")
        except InvalidStrain as e:
            ps1.log_error(e)
        ps1.save_h5(fpath, groups=['metadata', 'errors'], mode='w')

        # Append the same error again
        ps2 = PhotometrySession(mock_session_series, one=mock_one, load_data=False)
        try:
            raise InvalidStrain("bad strain")
        except InvalidStrain as e:
            ps2.log_error(e)
        ps2.save_h5(fpath, groups=['errors'], mode='a')

        # Should be deduplicated to one
        ps3 = PhotometrySession(mock_session_series, one=mock_one, load_data=False)
        ps3.load_h5(fpath, groups=['errors'])
        assert len(ps3.errors) == 1
        assert ps3.errors[0]['error_type'] == 'InvalidStrain'


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

    def test_from_h5_without_one(self, full_session_series, tmp_path):
        """from_h5 works without ONE for cached data."""
        from iblnm.data import PhotometrySession
        mock_one = MagicMock()
        ps = PhotometrySession(full_session_series, one=mock_one, load_data=False)
        fpath = tmp_path / f'{ps.eid}.h5'
        ps.save_h5(fpath, groups=['metadata'])

        ps2 = PhotometrySession.from_h5(fpath)
        assert ps2.eid == 'test-eid-full'


# =============================================================================
# Load Method Tests
# =============================================================================

class TestLoadTrials:
    """Tests for PhotometrySession.load_trials."""

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


class TestLoadPhotometry:
    """Tests for PhotometrySession.load_photometry."""

    def test_propagates_exception(self, mock_session_series):
        """load_photometry should let exceptions propagate."""
        from iblnm.data import PhotometrySession

        mock_one = MagicMock()
        session = PhotometrySession(mock_session_series, one=mock_one, load_data=False)

        with patch.object(
            PhotometrySession.__bases__[0], 'load_photometry',
            side_effect=Exception("No photometry data")
        ):
            with pytest.raises(Exception, match="No photometry data"):
                session.load_photometry()

    def test_no_flat_aliases(self, mock_photometry_session):
        """load_photometry should not create self.channels or self.targets."""
        session = mock_photometry_session
        assert not hasattr(session, 'channels')
        assert not hasattr(session, 'targets')


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
        session.trials = pd.DataFrame({'stimOn_times': np.random.rand(100)})
        with pytest.raises(IncompleteEventTimes) as exc_info:
            session.validate_event_completeness()
        for event in RESPONSE_EVENTS:
            if event != 'stimOn_times':
                assert event in exc_info.value.missing_events


class TestValidateTrialsInPhotometryTime:
    """Tests for PhotometrySession.validate_trials_in_photometry_time."""

    def test_raises_when_trials_outside(self, mock_session_series, mock_photometry_data):
        from iblnm.data import PhotometrySession
        from iblnm.validation import TrialsNotInPhotometryTime
        session = PhotometrySession(mock_session_series, one=MagicMock(), load_data=False)
        session.photometry = mock_photometry_data
        session.trials = pd.DataFrame({
            'stimOn_times': [-10.0, 100.0],
            'feedback_times': [100.0, 200.0],
        })
        with pytest.raises(TrialsNotInPhotometryTime):
            session.validate_trials_in_photometry_time()

    def test_does_not_raise_when_trials_inside(self, mock_session_series, mock_photometry_data):
        from iblnm.data import PhotometrySession
        session = PhotometrySession(mock_session_series, one=MagicMock(), load_data=False)
        session.photometry = mock_photometry_data
        session.trials = pd.DataFrame({
            'stimOn_times': [10.0, 100.0],
            'feedback_times': [100.0, 500.0],
        })
        session.validate_trials_in_photometry_time()  # should not raise

    def test_uses_preprocessed_band_when_no_raw(self, mock_photometry_session):
        """Should fall back to GCaMP_preprocessed when GCaMP is not available."""
        session = mock_photometry_session
        session.preprocess()
        del session.photometry['GCaMP']
        del session.photometry['Isosbestic']
        session.trials = pd.DataFrame({
            'stimOn_times': [10.0, 100.0],
            'feedback_times': [100.0, 500.0],
        })
        session.validate_trials_in_photometry_time()  # should not raise


class TestValidateFewUniqueSamples:
    def test_raises_below_threshold(self, mock_session_series):
        from iblnm.data import PhotometrySession
        from iblnm.validation import FewUniqueSamples
        session = PhotometrySession(mock_session_series, one=MagicMock(), load_data=False)
        session.qc = pd.DataFrame({
            'brain_region': ['VTA'], 'band': ['GCaMP'],
            'n_unique_samples': [0.01],
        })
        with pytest.raises(FewUniqueSamples, match='VTA/GCaMP'):
            session.validate_few_unique_samples()

    def test_does_not_raise_above_threshold(self, mock_session_series):
        from iblnm.data import PhotometrySession
        session = PhotometrySession(mock_session_series, one=MagicMock(), load_data=False)
        session.qc = pd.DataFrame({
            'brain_region': ['VTA'], 'band': ['GCaMP'],
            'n_unique_samples': [0.5],
        })
        session.validate_few_unique_samples()  # should not raise

    def test_does_not_raise_when_qc_empty(self, mock_session_series):
        from iblnm.data import PhotometrySession
        session = PhotometrySession(mock_session_series, one=MagicMock(), load_data=False)
        session.validate_few_unique_samples()  # should not raise

    def test_does_not_raise_when_column_missing(self, mock_session_series):
        from iblnm.data import PhotometrySession
        session = PhotometrySession(mock_session_series, one=MagicMock(), load_data=False)
        session.qc = pd.DataFrame({'brain_region': ['VTA'], 'band': ['GCaMP']})
        session.validate_few_unique_samples()  # should not raise


class TestValidateQc:
    """Tests for PhotometrySession.validate_qc."""

    def test_does_not_raise_when_qc_clean(self, mock_session_series):
        from iblnm.data import PhotometrySession
        session = PhotometrySession(mock_session_series, one=MagicMock(), load_data=False)
        session.qc = pd.DataFrame({
            'brain_region': ['VTA'], 'band': ['GCaMP'],
            'n_band_inversions': [0], 'n_early_samples': [0],
        })
        session.validate_qc()  # should not raise

    def test_raises_on_band_inversions(self, mock_session_series):
        from iblnm.data import PhotometrySession
        from iblnm.validation import QCValidationError
        session = PhotometrySession(mock_session_series, one=MagicMock(), load_data=False)
        session.qc = pd.DataFrame({
            'brain_region': ['VTA'], 'band': ['GCaMP'],
            'n_band_inversions': [3], 'n_early_samples': [0],
        })
        with pytest.raises(QCValidationError, match='band inversions'):
            session.validate_qc()

    def test_raises_on_early_samples(self, mock_session_series):
        from iblnm.data import PhotometrySession
        from iblnm.validation import QCValidationError
        session = PhotometrySession(mock_session_series, one=MagicMock(), load_data=False)
        session.qc = pd.DataFrame({
            'brain_region': ['VTA'], 'band': ['GCaMP'],
            'n_band_inversions': [0], 'n_early_samples': [5],
        })
        with pytest.raises(QCValidationError, match='early samples'):
            session.validate_qc()

    def test_raises_with_both_issues_in_message(self, mock_session_series):
        from iblnm.data import PhotometrySession
        from iblnm.validation import QCValidationError
        session = PhotometrySession(mock_session_series, one=MagicMock(), load_data=False)
        session.qc = pd.DataFrame({
            'brain_region': ['VTA'], 'band': ['GCaMP'],
            'n_band_inversions': [3], 'n_early_samples': [5],
        })
        with pytest.raises(QCValidationError) as exc_info:
            session.validate_qc()
        msg = str(exc_info.value)
        assert 'band inversions' in msg
        assert 'early samples' in msg

    def test_does_not_raise_when_qc_empty(self, mock_session_series):
        from iblnm.data import PhotometrySession
        session = PhotometrySession(mock_session_series, one=MagicMock(), load_data=False)
        session.validate_qc()  # should not raise


# =============================================================================
# Preprocess Tests
# =============================================================================

class TestPreprocess:
    """Tests for PhotometrySession.preprocess method."""

    def test_preprocess_adds_new_band(self, mock_photometry_session):
        """Preprocess should add preprocessed signal as new band in photometry dict."""
        session = mock_photometry_session

        session.preprocess()

        assert 'GCaMP_preprocessed' in session.photometry
        assert isinstance(session.photometry['GCaMP_preprocessed'], pd.DataFrame)
        assert 'VTA' in session.photometry['GCaMP_preprocessed'].columns

    def test_preprocess_computes_qc_metrics(self, mock_photometry_session):
        """Preprocess should store QC as DataFrame with bleaching_tau and iso_correlation."""
        session = mock_photometry_session

        session.preprocess()

        assert hasattr(session, 'qc')
        assert isinstance(session.qc, pd.DataFrame)
        assert 'brain_region' in session.qc.columns
        assert 'band' in session.qc.columns
        row = session.qc.query("brain_region == 'VTA' and band == 'GCaMP'")
        assert len(row) == 1
        tau = row['bleaching_tau'].iloc[0]
        assert 100 < tau < 600  # Known fixture tau=300, allow wide margin for fit
        iso_corr = row['iso_correlation'].iloc[0]
        assert 0.8 < iso_corr <= 1.0

    def test_preprocess_raises_when_no_photometry(self, mock_session_series):
        """Should raise if photometry not loaded (no explicit guard — natural error)."""
        from iblnm.data import PhotometrySession

        mock_one = MagicMock()
        session = PhotometrySession(mock_session_series, one=mock_one, load_data=False)

        with pytest.raises((AttributeError, KeyError, TypeError)):
            session.preprocess()

    def test_preprocess_single_band_pipeline(self, mock_photometry_session):
        """Single-band pipeline should work without reference."""
        from iblphotometry.pipelines import sliding_mad_pipeline

        session = mock_photometry_session

        session.preprocess(
            pipeline=sliding_mad_pipeline,
            reference_band=None
        )

        assert 'GCaMP_preprocessed' in session.photometry
        row = session.qc.query("brain_region == 'VTA' and band == 'GCaMP'")
        assert 'iso_correlation' not in row.columns or pd.isna(row['iso_correlation'].iloc[0])
        assert not pd.isna(row['bleaching_tau'].iloc[0])

    def test_preprocess_raises_when_dual_band_no_reference(self, mock_photometry_session):
        """Should raise ValueError if dual-band pipeline but no reference."""
        from iblphotometry.pipelines import isosbestic_correction_pipeline

        with pytest.raises(ValueError, match="requires reference"):
            mock_photometry_session.preprocess(
                pipeline=isosbestic_correction_pipeline,
                reference_band=None
            )

    def test_preprocess_custom_output_band(self, mock_photometry_session):
        """Can specify custom output band name."""
        mock_photometry_session.preprocess(output_band='corrected')

        assert 'corrected' in mock_photometry_session.photometry
        assert 'GCaMP_preprocessed' not in mock_photometry_session.photometry

    def test_qc_initialized_as_empty_dataframe(self, mock_session_series):
        """qc should be empty DataFrame on init."""
        from iblnm.data import PhotometrySession

        mock_one = MagicMock()
        session = PhotometrySession(mock_session_series, one=mock_one, load_data=False)

        assert isinstance(session.qc, pd.DataFrame)
        assert len(session.qc) == 0

    def test_preprocess_resamples_to_target_fs(self, mock_photometry_session):
        """Preprocessed signal should be resampled to TARGET_FS."""
        from iblnm.config import TARGET_FS
        session = mock_photometry_session
        session.preprocess()
        signal = session.photometry['GCaMP_preprocessed']['VTA']
        dt = np.diff(signal.index.values)
        np.testing.assert_allclose(dt, 1 / TARGET_FS, atol=1e-10)

    def test_preprocess_zscores_signal(self, mock_photometry_session):
        """Preprocessed signal should be z-scored (mean≈0, std≈1)."""
        session = mock_photometry_session
        session.preprocess()
        signal = session.photometry['GCaMP_preprocessed']['VTA'].values
        np.testing.assert_allclose(np.mean(signal), 0, atol=0.01)
        np.testing.assert_allclose(np.std(signal), 1, atol=0.01)

    def test_preprocess_accepts_regression_method(self, mock_photometry_session):
        """preprocess() should accept regression_method kwarg without error."""
        mock_photometry_session.preprocess(regression_method='mse')
        assert 'GCaMP_preprocessed' in mock_photometry_session.photometry

    def test_qc_is_dataframe_after_preprocess(self, mock_photometry_session):
        """self.qc should be a DataFrame after preprocess."""
        session = mock_photometry_session
        session.preprocess()

        assert isinstance(session.qc, pd.DataFrame)
        assert 'brain_region' in session.qc.columns
        assert 'band' in session.qc.columns
        assert 'bleaching_tau' in session.qc.columns
        assert len(session.qc) == 1  # One row for VTA/GCaMP


# =============================================================================
# Extract Responses and Trial Data Tests
# =============================================================================

class TestExtractResponses:
    def test_returns_xarray_dataarray(self, mock_photometry_session):
        import xarray as xr
        session = mock_photometry_session
        session.preprocess()
        n = 50
        session.trials = pd.DataFrame({
            'stimOn_times': np.linspace(99.5, 499.5, n),
            'firstMovement_times': np.linspace(100.3, 500.3, n),
            'feedback_times': np.linspace(101, 501, n),
        })
        session.extract_responses()
        assert isinstance(session.responses, dict)
        assert isinstance(session.responses['VTA'], xr.DataArray)

    def test_has_correct_dims(self, mock_photometry_session):
        from iblnm.config import RESPONSE_EVENTS
        session = mock_photometry_session
        session.preprocess()
        n = 50
        session.trials = pd.DataFrame({
            'stimOn_times': np.linspace(99.5, 499.5, n),
            'firstMovement_times': np.linspace(100.3, 500.3, n),
            'feedback_times': np.linspace(101, 501, n),
        })
        session.extract_responses()
        assert 'VTA' in session.responses
        region_responses = session.responses['VTA']
        assert set(region_responses.dims) == {'event', 'trial', 'time'}
        for event in RESPONSE_EVENTS:
            assert event in region_responses.coords['event'].values
        assert region_responses.sizes['trial'] == n

    def test_sel_region_event(self, mock_photometry_session):
        """Selecting by event returns (trial, time) array."""
        session = mock_photometry_session
        session.preprocess()
        n = 50
        session.trials = pd.DataFrame({
            'stimOn_times': np.linspace(99.5, 499.5, n),
            'firstMovement_times': np.linspace(100.3, 500.3, n),
            'feedback_times': np.linspace(101, 501, n),
        })
        session.extract_responses()
        sel = session.responses['VTA'].sel(event='stimOn_times')
        assert sel.dims == ('trial', 'time')
        assert sel.shape[0] == n

    def test_time_coord_matches_window(self, mock_photometry_session):
        """Time coordinate should span RESPONSE_WINDOW."""
        from iblnm.config import RESPONSE_WINDOW
        session = mock_photometry_session
        session.preprocess()
        n = 50
        session.trials = pd.DataFrame({
            'feedback_times': np.linspace(101, 501, n),
        })
        session.extract_responses(events=['feedback_times'])
        tpts = session.responses['VTA'].coords['time'].values
        assert tpts[0] == pytest.approx(RESPONSE_WINDOW[0], abs=0.05)
        assert tpts[-1] == pytest.approx(RESPONSE_WINDOW[1], abs=0.05)

    def test_custom_events(self, mock_photometry_session):
        session = mock_photometry_session
        session.preprocess()
        n = 50
        session.trials = pd.DataFrame({
            'feedback_times': np.linspace(101, 501, n),
        })
        session.extract_responses(events=['feedback_times'])
        region_responses = session.responses['VTA']
        assert list(region_responses.coords['event'].values) == ['feedback_times']
        assert region_responses.sizes['event'] == 1



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


class TestSaveLoadH5:
    def test_save_preprocessed_float64(self, mock_photometry_session, tmp_path):
        """save_h5 should write preprocessed signal as float64 with timestamps."""
        session = mock_photometry_session
        session.preprocess()
        fpath = tmp_path / f'{session.eid}.h5'
        session.save_h5(fpath)

        import h5py
        with h5py.File(fpath, 'r') as f:
            pp_grp = f['photometry/VTA/preprocessed']
            assert pp_grp.attrs['fs'] == 30
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

    def test_save_trials_and_responses(self, mock_photometry_session, tmp_path):
        """save_h5 in append mode should add trials and xarray responses."""
        session = mock_photometry_session
        session.preprocess()
        n = 50
        session.trials = pd.DataFrame({
            'stimOn_times': np.linspace(99.5, 499.5, n),
            'firstMovement_times': np.linspace(100.3, 500.3, n),
            'feedback_times': np.linspace(101, 501, n),
            'goCue_times': np.linspace(100, 500, n),
            'response_times': np.linspace(100.5, 500.5, n),
            'intervals_0': np.linspace(99, 499, n),
            'intervals_1': np.linspace(102, 502, n),
            'choice': np.random.choice([-1, 1], n),
            'feedbackType': np.random.choice([-1, 1], n),
            'probabilityLeft': np.random.choice([0.2, 0.5, 0.8], n),
            'signed_contrast': np.random.choice([-100, -25, 0, 25, 100], n).astype(float),
            'contrast': np.random.choice([0, 25, 100], n).astype(float),
        })
        session.extract_responses(events=['stimOn_times', 'feedback_times'])

        fpath = tmp_path / f'{session.eid}.h5'
        session.save_h5(fpath)  # Create with preprocessed
        session.save_h5(fpath, mode='a')  # Append trials + responses

        import h5py
        with h5py.File(fpath, 'r') as f:
            assert 'photometry/VTA/preprocessed/signal' in f
            assert 'trials/choice' in f
            assert 'photometry/VTA/responses/stimOn_times' in f
            assert 'photometry/VTA/responses/feedback_times' in f
            # Verify response data matches xarray content
            resp_h5 = f['photometry/VTA/responses/stimOn_times'][:]
            resp_xr = session.responses['VTA'].sel(event='stimOn_times').values
            np.testing.assert_allclose(resp_h5, resp_xr, rtol=1e-5)
            np.testing.assert_array_equal(
                f['trials/choice'][:],
                session.trials['choice'].values
            )

    def test_load_h5_restores_xarray_responses(self, mock_photometry_session, tmp_path):
        """load_h5 should restore responses as xarray DataArray."""
        import xarray as xr
        session = mock_photometry_session
        session.preprocess()
        n = 50
        session.trials = pd.DataFrame({
            'stimOn_times': np.linspace(99.5, 499.5, n),
            'feedback_times': np.linspace(101, 501, n),
        })
        session.extract_responses(events=['stimOn_times', 'feedback_times'])
        original = {r: da.copy() for r, da in session.responses.items()}

        fpath = tmp_path / f'{session.eid}.h5'
        session.save_h5(fpath)
        session.save_h5(fpath, mode='a')

        # Clear and reload
        session.responses = {}
        session.load_h5(fpath)
        assert isinstance(session.responses, dict)
        assert isinstance(session.responses['VTA'], xr.DataArray)
        assert set(session.responses['VTA'].dims) == {'event', 'trial', 'time'}
        np.testing.assert_allclose(
            session.responses['VTA'].sel(event='stimOn_times').values,
            original['VTA'].sel(event='stimOn_times').values,
            rtol=1e-5,
        )

    def test_load_h5_restores_trials(self, mock_photometry_session, tmp_path):
        """load_h5 should restore trials saved in the HDF5 trials group."""
        session = mock_photometry_session
        session.preprocess()
        n = 50
        session.trials = pd.DataFrame({
            'stimOn_times':        np.linspace(99.5, 499.5, n),
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
        saved_stim = session.trials['stimOn_times'].values.copy()
        fpath = tmp_path / f'{session.eid}.h5'
        session.save_h5(fpath)
        session.save_h5(fpath, mode='a')
        session.trials = None
        session.load_h5(fpath)
        assert session.trials is not None
        assert 'stimOn_times' in session.trials.columns
        np.testing.assert_allclose(session.trials['stimOn_times'].values, saved_stim)

    def test_save_load_qc_roundtrip(self, mock_session_series, tmp_path):
        """QC metrics under photometry/<region>/qc/ survive H5 roundtrip."""
        from iblnm.data import PhotometrySession
        mock_one = MagicMock()
        ps = PhotometrySession(mock_session_series, one=mock_one, load_data=False)
        ps.qc = pd.DataFrame({
            'eid': [ps.eid, ps.eid],
            'brain_region': ['VTA', 'SNc'],
            'band': ['GCaMP', 'GCaMP'],
            'n_band_inversions': [0, 2],
            'bleaching_tau': [150.5, 200.3],
        })
        fpath = tmp_path / f'{ps.eid}.h5'
        ps.save_h5(fpath, groups=['metadata', 'photometry'])

        ps2 = PhotometrySession(mock_session_series, one=mock_one, load_data=False)
        assert ps2.qc.empty
        ps2.load_h5(fpath, groups=['photometry'])
        assert len(ps2.qc) == 2
        assert set(ps2.qc['brain_region']) == {'VTA', 'SNc'}
        qc_sorted = ps2.qc.sort_values('brain_region').reset_index(drop=True)
        assert list(qc_sorted['brain_region']) == ['SNc', 'VTA']
        assert list(qc_sorted['n_band_inversions']) == [2, 0]
        np.testing.assert_allclose(qc_sorted['bleaching_tau'], [200.3, 150.5])

    def test_load_h5_roundtrip(self, mock_photometry_session, tmp_path):
        """load_h5 should restore preprocessed signal from saved file."""
        session = mock_photometry_session
        session.preprocess()
        fpath = tmp_path / f'{session.eid}.h5'
        session.save_h5(fpath)

        original = session.photometry['GCaMP_preprocessed']['VTA'].values.copy()

        # Clear and reload
        session.photometry.pop('GCaMP_preprocessed')
        session.load_h5(fpath)
        reloaded = session.photometry['GCaMP_preprocessed']['VTA'].values
        np.testing.assert_allclose(reloaded, original, rtol=1e-10)

    def _make_video_session(self, mock_session_series, qc_lp='NOT_SET',
                            qc_movement='NOT_SET'):
        """PhotometrySession carrying synthetic pose traces + cross-correlation."""
        import xarray as xr
        from iblnm.data import PhotometrySession
        ps = PhotometrySession(mock_session_series, one=MagicMock(),
                               load_data=False)
        n_trial, n_time = 4, 20
        bodyparts = ['paw', 'nose', 'tongue_speed', 'tongue_likelihood']
        rng = np.random.default_rng(0)
        coords = {
            'bodypart': bodyparts,
            'trial': np.arange(n_trial),
            'time': np.linspace(-1, 1, n_time),
        }
        ps.pose_traces = xr.DataArray(
            rng.standard_normal((len(bodyparts), n_trial, n_time)),
            dims=['bodypart', 'trial', 'time'], coords=coords,
        )
        ps.pose_baseline_traces = xr.DataArray(
            rng.standard_normal((len(bodyparts), n_trial, n_time)),
            dims=['bodypart', 'trial', 'time'], coords=coords,
        )
        n_lags = 11
        ps.pose_xcorr = {
            'functions': rng.standard_normal((3, n_lags)),
            'lags': np.linspace(-5, 5, n_lags),
            'peak_lags': np.array([0.1, 0.2, 0.4]),
            'drift': 0.3,
        }
        ps.qc_lp = qc_lp
        ps.qc_movement = qc_movement
        return ps

    def test_save_load_video_roundtrip(self, mock_session_series, tmp_path):
        """video group round-trips traces, cross-correlation, and QC labels."""
        from iblnm.data import PhotometrySession
        ps = self._make_video_session(mock_session_series, qc_lp='FAIL',
                                      qc_movement='WARNING')
        fpath = tmp_path / f'{ps.eid}.h5'
        ps.save_h5(fpath, groups=['video'])

        ps2 = PhotometrySession(mock_session_series, one=MagicMock(),
                                load_data=False)
        ps2.load_h5(fpath, groups=['video'])

        assert (set(ps2.pose_traces.coords['bodypart'].values)
                == set(ps.pose_traces.coords['bodypart'].values))
        for bodypart in ps.pose_traces.coords['bodypart'].values:
            np.testing.assert_allclose(
                ps2.pose_traces.sel(bodypart=bodypart).values,
                ps.pose_traces.sel(bodypart=bodypart).values,
            )
            np.testing.assert_allclose(
                ps2.pose_baseline_traces.sel(bodypart=bodypart).values,
                ps.pose_baseline_traces.sel(bodypart=bodypart).values,
            )
        np.testing.assert_allclose(ps2.pose_xcorr['functions'],
                                   ps.pose_xcorr['functions'])
        np.testing.assert_allclose(ps2.pose_xcorr['lags'], ps.pose_xcorr['lags'])
        np.testing.assert_allclose(ps2.pose_xcorr['peak_lags'],
                                   ps.pose_xcorr['peak_lags'])
        assert ps2.pose_xcorr['drift'] == ps.pose_xcorr['drift']
        assert ps2.qc_lp == 'FAIL'
        assert ps2.qc_movement == 'WARNING'

    def test_save_video_preserves_manual_qc(self, mock_session_series, tmp_path):
        """Re-saving automatic data with no QC on the object keeps prior labels."""
        from iblnm.data import PhotometrySession
        ps = self._make_video_session(mock_session_series, qc_lp='FAIL',
                                      qc_movement='CRITICAL')
        fpath = tmp_path / f'{ps.eid}.h5'
        ps.save_h5(fpath, groups=['video'])

        # Fresh session with only automatic data, QC fields left at default.
        ps_auto = self._make_video_session(mock_session_series)
        assert ps_auto.qc_lp == 'NOT_SET'
        ps_auto.save_h5(fpath, groups=['video'])

        ps2 = PhotometrySession(mock_session_series, one=MagicMock(),
                                load_data=False)
        ps2.load_h5(fpath, groups=['video'])
        assert ps2.qc_lp == 'FAIL'
        assert ps2.qc_movement == 'CRITICAL'

    def test_save_load_video_qc_roundtrip(self, mock_session_series, tmp_path):
        """The 8 live-fetched VIDEO_QC_COLS round-trip via the video group."""
        from iblnm.config import VIDEO_QC_COLS
        from iblnm.data import PhotometrySession
        ps = self._make_video_session(mock_session_series)
        ps.video_qc = {col: 'PASS' for col in VIDEO_QC_COLS}
        ps.video_qc['qc_videoLeft_pin_state'] = 'FAIL'
        fpath = tmp_path / f'{ps.eid}.h5'
        ps.save_h5(fpath, groups=['video'])

        ps2 = PhotometrySession(mock_session_series, one=MagicMock(),
                                load_data=False)
        ps2.load_h5(fpath, groups=['video'])
        assert ps2.video_qc == ps.video_qc


class TestFetchVideoQC:
    """PhotometrySession.fetch_video_qc selects and stores the 8 VIDEO_QC_COLS."""

    def test_stores_eight_qc_cols(self, mock_session_series):
        from iblnm.config import VIDEO_QC_COLS
        from iblnm.data import PhotometrySession
        ps = PhotometrySession(mock_session_series, one=MagicMock(),
                               load_data=False)
        fetched = pd.Series({col: 'PASS' for col in VIDEO_QC_COLS})
        fetched['qc_videoLeft_timestamps'] = 'FAIL'
        with patch('iblnm.io.get_extended_qc', return_value=fetched):
            ps.fetch_video_qc()
        assert set(ps.video_qc) == set(VIDEO_QC_COLS)
        assert ps.video_qc['qc_videoLeft_timestamps'] == 'FAIL'
        assert ps.video_qc['qc_videoLeft_focus'] == 'PASS'

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

    def test_load_pose_loads_only_lightningpose(self, mock_session_series):
        """load_pose pulls only lightningPose via load_dataset (never the whole
        leftCamera object) and no longer loads camera times."""
        from iblnm.data import PhotometrySession
        pose_df = pd.DataFrame({'paw_l_x': [0.0, 1.0], 'paw_l_y': [0.0, 0.0],
                                'paw_l_likelihood': [1.0, 1.0]})

        one = MagicMock()
        one.load_dataset.return_value = pose_df
        ps = PhotometrySession(mock_session_series, one=one, load_data=False)
        ps.load_pose()
        one.load_object.assert_not_called()
        pd.testing.assert_frame_equal(ps.pose, pose_df)
        assert ps.pose_times is None
        loaded_names = [call.args[1] for call in one.load_dataset.call_args_list]
        assert all('lightningPose' in name for name in loaded_names)

    def test_load_pose_missing_raises_missing_lp(self, mock_session_series):
        """load_pose raises MissingLP when the pose dataset is absent."""
        from one.alf.exceptions import ALFObjectNotFound
        from iblnm.data import PhotometrySession
        from iblnm.validation import MissingLP

        one = MagicMock()
        one.load_dataset.side_effect = ALFObjectNotFound('leftCamera.lightningPose')
        ps = PhotometrySession(mock_session_series, one=one, load_data=False)
        with pytest.raises(MissingLP):
            ps.load_pose()

    def test_load_camera_times_sets_pose_times(self, mock_session_series):
        """load_camera_times loads only leftCamera.times into pose_times."""
        from iblnm.data import PhotometrySession
        times = np.array([0.0, 0.1, 0.2])

        one = MagicMock()
        one.load_dataset.return_value = times
        ps = PhotometrySession(mock_session_series, one=one, load_data=False)
        ps.load_camera_times()
        np.testing.assert_array_equal(ps.pose_times, times)
        loaded_names = [call.args[1] for call in one.load_dataset.call_args_list]
        assert all('times' in name for name in loaded_names)

    def test_load_camera_times_missing_raises(self, mock_session_series):
        """load_camera_times raises MissingVideoTimestamps when times are absent."""
        from one.alf.exceptions import ALFObjectNotFound
        from iblnm.data import PhotometrySession
        from iblnm.validation import MissingVideoTimestamps

        one = MagicMock()
        one.load_dataset.side_effect = ALFObjectNotFound('leftCamera.times')
        ps = PhotometrySession(mock_session_series, one=one, load_data=False)
        with pytest.raises(MissingVideoTimestamps):
            ps.load_camera_times()

    def test_load_motion_energy_sets_array(self, mock_session_series):
        """load_motion_energy loads ROIMotionEnergy (no _ibl_ prefix) into
        self.motion_energy."""
        from iblnm.data import PhotometrySession
        me = np.array([0.0, 1.0, 2.0, 3.0])

        one = MagicMock()
        one.load_dataset.return_value = me
        ps = PhotometrySession(mock_session_series, one=one, load_data=False)
        ps.load_motion_energy()
        np.testing.assert_array_equal(ps.motion_energy, me)
        loaded_names = [call.args[1] for call in one.load_dataset.call_args_list]
        assert all('ROIMotionEnergy' in name and '_ibl_' not in name
                   for name in loaded_names)

    def test_load_motion_energy_missing_raises(self, mock_session_series):
        """load_motion_energy raises MissingMotionEnergy when the dataset is absent."""
        from one.alf.exceptions import ALFObjectNotFound
        from iblnm.data import PhotometrySession
        from iblnm.validation import MissingMotionEnergy

        one = MagicMock()
        one.load_dataset.side_effect = ALFObjectNotFound('leftCamera.ROIMotionEnergy')
        ps = PhotometrySession(mock_session_series, one=one, load_data=False)
        with pytest.raises(MissingMotionEnergy):
            ps.load_motion_energy()

    def test_compute_video_measures(self, mock_session_series):
        """compute_video_measures yields hand-computed discrepancy and framerate."""
        from iblnm.data import PhotometrySession
        ps = PhotometrySession(mock_session_series, one=MagicMock(),
                               load_data=False)
        ps.pose_times = np.array([1.0, 1.1, 1.2, 1.35])
        ps.session_length = 0.3
        ps.compute_video_measures()
        # video span 0.35 - session_length 0.3 = 0.05
        assert ps.length_discrepancy == pytest.approx(0.05)
        # diffs: [0.1, 0.1, 0.15] -> median 0.1
        assert ps.framerate_from_tpts == pytest.approx(0.1)

    def test_video_measures_round_trip_without_traces(self, mock_session_series,
                                                       tmp_path):
        """save_h5(['video']) with no traces but measures set writes a video group
        that load_h5 restores both measures from."""
        from iblnm.data import PhotometrySession
        ps = PhotometrySession(mock_session_series, one=MagicMock(),
                               load_data=False)
        ps.length_discrepancy = 0.05
        ps.framerate_from_tpts = 0.0333
        fpath = tmp_path / f'{ps.eid}.h5'
        ps.save_h5(fpath, groups=['video'])

        ps2 = PhotometrySession(mock_session_series, one=MagicMock(),
                                load_data=False)
        ps2.load_h5(fpath, groups=['video'])
        assert ps2.pose_traces is None
        assert ps2.length_discrepancy == pytest.approx(0.05)
        assert ps2.framerate_from_tpts == pytest.approx(0.0333)

    def test_available_save_groups_includes_video_for_measures_only(
            self, mock_session_series):
        """A traceless session with only basic-video measures still saves the
        video group (so MissingLP sessions persist their measures)."""
        from iblnm.data import PhotometrySession
        ps = PhotometrySession(mock_session_series, one=MagicMock(),
                               load_data=False)
        ps.length_discrepancy = 0.05
        assert 'video' in ps._available_save_groups('GCaMP_preprocessed')

    def _make_pose_session(self, mock_session_series, fs=30, dur=60.0,
                           tongue_like=(0.2, 0.9), accelerate=False,
                           motion_energy=False):
        """PhotometrySession with injected synthetic pose + camera times + trials.

        With ``accelerate``, keypoint positions grow quadratically so speed rises
        over time and the event a window is locked to changes its value. With
        ``motion_energy``, a per-frame ME ramp is injected on the camera time base.
        """
        from iblnm.data import PhotometrySession
        ps = PhotometrySession(mock_session_series, one=MagicMock(),
                               load_data=False)
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
            'stimOn_times': [9.0, 19.0, 29.0],
            'firstMovement_times': [10.0, 20.0, 30.0],
            'feedback_times': [12.0, 22.0, 32.0],
        })
        return ps

    def test_extract_movement_traces_shapes_and_labels(self, mock_session_series):
        """Four bodypart traces with (n_trials, n_time) matrices keyed by label."""
        from iblnm.config import POSE_MEASURES
        ps = self._make_pose_session(mock_session_series, fs=30)
        ps.extract_movement_traces()
        assert set(ps.pose_traces.coords['bodypart'].values) == set(POSE_MEASURES)
        assert ps.pose_traces.sizes == {'bodypart': 4, 'trial': 3, 'time': 60}

    def test_extract_movement_traces_baseline_locked_to_stimon(self, mock_session_series):
        """pose_baseline_traces is stimOn-locked: the nose measure (itself
        stimOn-locked) has identical response and baseline traces, while the
        firstMovement-locked paw measure does not."""
        ps = self._make_pose_session(mock_session_series, fs=30, accelerate=True)
        ps.extract_movement_traces()
        assert ps.pose_baseline_traces.sizes == ps.pose_traces.sizes
        np.testing.assert_allclose(
            ps.pose_baseline_traces.sel(bodypart='nose').values,
            ps.pose_traces.sel(bodypart='nose').values)
        assert not np.allclose(
            ps.pose_baseline_traces.sel(bodypart='paw').values,
            ps.pose_traces.sel(bodypart='paw').values,
            equal_nan=True)

    def test_extract_movement_traces_tongue_likelihood_is_max(self, mock_session_series):
        """tongue_likelihood trace equals the per-frame max of the two tips."""
        ps = self._make_pose_session(mock_session_series, tongue_like=(0.2, 0.9))
        ps.extract_movement_traces()
        tongue = ps.pose_traces.sel(bodypart='tongue_likelihood').values
        np.testing.assert_allclose(tongue, 0.9)

    def test_extract_movement_traces_common_timebase_across_fps(self, mock_session_series):
        """Different camera fps → identical trace time length (resampled to POSE_FS)."""
        ps30 = self._make_pose_session(mock_session_series, fs=30)
        ps99 = self._make_pose_session(mock_session_series, fs=99)
        ps30.extract_movement_traces()
        ps99.extract_movement_traces()
        assert (ps30.pose_traces.sizes['time']
                == ps99.pose_traces.sizes['time'] == 60)

    def test_extract_movement_traces_includes_motion_energy(self, mock_session_series):
        """pose + ME present → bodypart coord adds motion_energy to the LP labels,
        and the ME channel is stimOn-locked (baseline equals event trace)."""
        from iblnm.config import POSE_MEASURES
        ps = self._make_pose_session(mock_session_series, motion_energy=True)
        ps.extract_movement_traces()
        assert (set(ps.pose_traces.coords['bodypart'].values)
                == set(POSE_MEASURES) | {'motion_energy'})
        np.testing.assert_allclose(
            ps.pose_baseline_traces.sel(bodypart='motion_energy').values,
            ps.pose_traces.sel(bodypart='motion_energy').values)

    def test_extract_movement_traces_motion_energy_only(self, mock_session_series):
        """ME present, pose=None → pose_traces has exactly ['motion_energy']."""
        ps = self._make_pose_session(mock_session_series, motion_energy=True)
        ps.pose = None
        ps.extract_movement_traces()
        assert list(ps.pose_traces.coords['bodypart'].values) == ['motion_energy']

    def test_extract_movement_traces_lp_only_when_no_motion_energy(self, mock_session_series):
        """pose present, motion_energy=None → pose_traces has only the LP labels."""
        from iblnm.config import POSE_MEASURES
        ps = self._make_pose_session(mock_session_series)
        assert ps.motion_energy is None
        ps.extract_movement_traces()
        assert (set(ps.pose_traces.coords['bodypart'].values)
                == set(POSE_MEASURES))

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
        ps.wheel = pd.DataFrame({
            'times': t if wheel_times is None else wheel_times,
            'position': np.cumsum(wheel_velocity) / fs, 'velocity': wheel_velocity})
        ps.wheel_fs = fs
        return ps, shift, fs

    def test_extract_paw_wheel_xcorr_recovers_drift(self, mock_session_series):
        """An imposed late-third paw/wheel shift surfaces in pose_xcorr['drift']."""
        ps, shift, fs = self._xcorr_session(mock_session_series)
        ps.extract_paw_wheel_xcorr()
        np.testing.assert_allclose(ps.pose_xcorr['peak_lags'][0], 0.0, atol=1 / fs)
        np.testing.assert_allclose(ps.pose_xcorr['drift'], shift / fs, atol=1 / fs)

    def test_extract_paw_wheel_xcorr_float32_wheel_times(self, mock_session_series):
        """float32 (non-uniform) wheel times do not raise — regression for the
        movements() even-sampling crash; drift stays finite."""
        ps, shift, fs = self._xcorr_session(mock_session_series)
        # float32 storage makes consecutive dt non-uniform well above the 1e-10
        # tolerance that brainbox.movements() asserted on (the crash class).
        ps.wheel['times'] = ps.wheel['times'].values.astype(np.float32)
        assert not np.all(np.abs(np.diff(ps.wheel['times'].values)
                                 - (1 / fs)) < 1e-10)  # genuinely non-uniform
        ps.extract_paw_wheel_xcorr()
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


class TestBasicPerformance:
    """Tests for PhotometrySession.basic_performance()."""

    def test_returns_expected_keys(self, mock_session_series):
        from iblnm.data import PhotometrySession
        session = PhotometrySession(mock_session_series, one=MagicMock(), load_data=False)
        session.trials = _make_training_trials()
        result = session.basic_performance()
        for key in ['fraction_correct', 'fraction_correct_easy', 'nogo_fraction',
                    'psych_50_bias', 'psych_50_threshold', 'psych_50_r_squared',
                    'psych_50_n_trials']:
            assert key in result, f"Missing key: {key}"

    def test_no_block_keys(self, mock_session_series):
        from iblnm.data import PhotometrySession
        session = PhotometrySession(mock_session_series, one=MagicMock(), load_data=False)
        session.trials = _make_training_trials()
        result = session.basic_performance()
        assert not any(k.startswith('psych_20') or k.startswith('psych_80')
                       or k == 'bias_shift' for k in result)

    def test_fraction_correct_reasonable(self, mock_session_series):
        from iblnm.data import PhotometrySession
        session = PhotometrySession(mock_session_series, one=MagicMock(), load_data=False)
        session.trials = _make_training_trials()
        result = session.basic_performance()
        assert 0 < result['fraction_correct'] <= 1
        assert 0 <= result['nogo_fraction'] < 1


# =============================================================================
# QC Method Tests
# =============================================================================

class TestRunRawQc:
    """Tests for PhotometrySession.run_raw_qc."""

    def test_sets_qc_with_raw_metric_columns(self, mock_session_series):
        from iblnm.data import PhotometrySession
        from unittest.mock import patch
        session = PhotometrySession(mock_session_series, one=MagicMock(), load_data=False)
        raw_phot = pd.DataFrame({'col1': [1.0, 2.0]}, index=[0.0, 1.0])
        with patch.object(session, '_load_raw_photometry', return_value=raw_phot):
            with patch('iblnm.data.metrics') as mock_metrics:
                mock_metrics.n_band_inversions.return_value = 0
                mock_metrics.n_early_samples.return_value = 2
                session.run_raw_qc()
        assert 'n_band_inversions' in session.qc.columns
        assert 'n_early_samples' in session.qc.columns

    def test_sets_qc_values_from_metrics(self, mock_session_series):
        from iblnm.data import PhotometrySession
        from unittest.mock import patch
        session = PhotometrySession(mock_session_series, one=MagicMock(), load_data=False)
        raw_phot = pd.DataFrame({'col1': [1.0, 2.0]}, index=[0.0, 1.0])
        with patch.object(session, '_load_raw_photometry', return_value=raw_phot):
            with patch('iblnm.data.metrics') as mock_metrics:
                mock_metrics.n_band_inversions.return_value = 3
                mock_metrics.n_early_samples.return_value = 5
                session.run_raw_qc()
        assert session.qc['n_band_inversions'].iloc[0] == 3
        assert session.qc['n_early_samples'].iloc[0] == 5

    def test_includes_eid(self, mock_session_series):
        from iblnm.data import PhotometrySession
        from unittest.mock import patch
        session = PhotometrySession(mock_session_series, one=MagicMock(), load_data=False)
        with patch.object(session, '_load_raw_photometry', return_value=pd.DataFrame({'c': [1.0]})):
            with patch('iblnm.data.metrics') as mock_metrics:
                mock_metrics.n_band_inversions.return_value = 0
                mock_metrics.n_early_samples.return_value = 0
                session.run_raw_qc()
        assert 'eid' in session.qc.columns
        assert session.qc['eid'].iloc[0] == 'test-eid-123'

    def test_propagates_load_failure(self, mock_session_series):
        from iblnm.data import PhotometrySession
        from unittest.mock import patch
        session = PhotometrySession(mock_session_series, one=MagicMock(), load_data=False)
        with patch.object(session, '_load_raw_photometry', side_effect=Exception("load failed")):
            with pytest.raises(Exception, match="load failed"):
                session.run_raw_qc()


class TestRunSlidingQc:
    """Tests for PhotometrySession.run_sliding_qc."""

    def _make_tidy_qc(self):
        return pd.DataFrame({
            'band': ['GCaMP', 'GCaMP'],
            'brain_region': ['VTA', 'VTA'],
            'metric': ['n_unique_samples', 'n_unique_samples'],
            'value': [0.8, 0.9],
            'window': [0, 1],
        })

    def test_sets_qc_per_region_band(self, mock_photometry_session):
        from unittest.mock import patch
        session = mock_photometry_session
        with patch('iblnm.data.qc_signals', return_value=self._make_tidy_qc()):
            session.run_sliding_qc()
        assert 'brain_region' in session.qc.columns
        assert 'band' in session.qc.columns
        assert 'n_unique_samples' in session.qc.columns

    def test_averages_across_windows(self, mock_photometry_session):
        from unittest.mock import patch
        session = mock_photometry_session
        with patch('iblnm.data.qc_signals', return_value=self._make_tidy_qc()):
            session.run_sliding_qc()
        assert len(session.qc) == 1  # One row for VTA/GCaMP after averaging
        assert session.qc['n_unique_samples'].iloc[0] == pytest.approx((0.8 + 0.9) / 2)

    def test_incorporates_raw_metrics_from_run_raw_qc(self, mock_photometry_session):
        from unittest.mock import patch
        session = mock_photometry_session
        # Simulate state after run_raw_qc()
        session.qc = pd.DataFrame([{'eid': session.eid, 'n_band_inversions': 0, 'n_early_samples': 0}])
        with patch('iblnm.data.qc_signals', return_value=self._make_tidy_qc()):
            session.run_sliding_qc()
        assert 'n_band_inversions' in session.qc.columns
        assert 'n_early_samples' in session.qc.columns

    def test_propagates_qc_signals_failure(self, mock_photometry_session):
        from unittest.mock import patch
        session = mock_photometry_session
        with patch('iblnm.data.qc_signals', side_effect=Exception("qc_signals failed")):
            with pytest.raises(Exception, match="qc_signals failed"):
                session.run_sliding_qc()

    def test_includes_eid(self, mock_photometry_session):
        from unittest.mock import patch
        session = mock_photometry_session
        with patch('iblnm.data.qc_signals', return_value=self._make_tidy_qc()):
            session.run_sliding_qc()
        assert 'eid' in session.qc.columns
        assert session.qc['eid'].iloc[0] == session.eid


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
                'event':  ['stimOn_times', 'firstMovement_times'],
                'trial':  [0, 1],
                'time':   tpts,
            },
        )
        # trial 0: dt = 0.3 - 0.0 = 0.3 → mask tpts > 0.3 (indices 3, 4)
        # trial 1: firstMovement = NaN → no masking
        session.trials = pd.DataFrame({
            'stimOn_times':        [0.0, 0.0],
            'firstMovement_times': [0.3, np.nan],
            'feedback_times':      [1.5, 1.5],
        })
        return session, responses

    def test_masks_times_after_next_event(self, mock_session_series):
        session, responses = self._make_session_and_responses(mock_session_series)
        result = session.mask_subsequent_events(
            responses,
            event_order=['stimOn_times', 'firstMovement_times', 'feedback_times'],
        )
        mat = result.sel(event='stimOn_times').values
        assert np.isnan(mat[0, 3])       # trial 0, t=0.5 > 0.3 → NaN
        assert np.isnan(mat[0, 4])       # trial 0, t=1.0 > 0.3 → NaN
        assert not np.isnan(mat[0, 2])   # trial 0, t=0.0 ≤ 0.3 → kept
        assert not np.isnan(mat[1, 3])   # trial 1, NaN dt → not masked

    def test_last_event_not_masked(self, mock_session_series):
        """firstMovement event matrix is unchanged (no event after it in responses)."""
        session, responses = self._make_session_and_responses(mock_session_series)
        result = session.mask_subsequent_events(
            responses,
            event_order=['stimOn_times', 'firstMovement_times'],
        )
        mat = result.sel(event='firstMovement_times').values
        assert not np.any(np.isnan(mat))

    def test_nan_dt_not_masked(self, mock_session_series):
        """Trial 1 has NaN firstMovement → stimOn response fully intact."""
        session, responses = self._make_session_and_responses(mock_session_series)
        result = session.mask_subsequent_events(
            responses,
            event_order=['stimOn_times', 'firstMovement_times', 'feedback_times'],
        )
        mat = result.sel(event='stimOn_times').values
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
            'stimOn_times':        [0.0, 0.0],
            'firstMovement_times': [0.3, 0.4],
            'feedback_times':      [1.5, 1.5],
        })
        # stimOn_times not in responses → skip without error
        result = session.mask_subsequent_events(
            responses,
            event_order=['stimOn_times', 'firstMovement_times', 'feedback_times'],
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
            coords={'event': ['stimOn_times'], 'trial': [0, 1], 'time': tpts},
        )
        session.trials = pd.DataFrame({
            'stimOn_times':        [0.0, 0.0],
            'firstMovement_times': [np.nan, np.nan],
            'feedback_times':      [0.3, np.nan],
        })
        result = session.mask_subsequent_events(responses)  # default event_order
        mat = result.sel(event='stimOn_times').values
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


class TestBlockPerformance:
    """Tests for PhotometrySession.block_performance()."""

    def test_returns_empty_for_training(self, mock_session_series):
        from iblnm.data import PhotometrySession
        series = mock_session_series.copy()
        series['session_type'] = 'training'
        session = PhotometrySession(series, one=MagicMock(), load_data=False)
        session.trials = _make_training_trials()
        assert session.block_performance() == {}

    def test_returns_block_keys_for_biased(self, mock_session_series):
        from iblnm.data import PhotometrySession
        series = mock_session_series.copy()
        series['session_type'] = 'biased'
        session = PhotometrySession(series, one=MagicMock(), load_data=False)
        session.trials = _make_biased_trials()
        result = session.block_performance()
        assert any(k.startswith('psych_20') for k in result)
        assert any(k.startswith('psych_80') for k in result)

    def test_bias_shift_present_for_biased(self, mock_session_series):
        from iblnm.data import PhotometrySession
        series = mock_session_series.copy()
        series['session_type'] = 'biased'
        session = PhotometrySession(series, one=MagicMock(), load_data=False)
        session.trials = _make_biased_trials()
        result = session.block_performance()
        assert 'bias_shift' in result

    def test_returns_empty_for_ephys_only_with_50_block(self, mock_session_series):
        from iblnm.data import PhotometrySession
        series = mock_session_series.copy()
        series['session_type'] = 'ephys'
        session = PhotometrySession(series, one=MagicMock(), load_data=False)
        # Only 0.5 block — fit_psychometric_by_block returns only '50', no bias_shift
        session.trials = _make_training_trials()
        result = session.block_performance()
        assert 'bias_shift' not in result  # no 20/80 blocks present


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
        group = PhotometrySessionGroup.from_catalog(self._make_catalog(), one=MagicMock())
        # eid-3 dropped (mismatched), eid-1 and eid-2 kept
        assert len(group.sessions) == 2
        assert 'eid-3' not in group.sessions['eid'].values

    def test_recordings_from_catalog(self):
        """recordings produces one row per region after from_catalog."""
        from iblnm.data import PhotometrySessionGroup
        group = PhotometrySessionGroup.from_catalog(self._make_catalog(), one=MagicMock())
        group.filter_sessions(session_types=False, targetnms=False,
                              qc_blockers=set(),
                              min_performance=False, required_contrasts=False)
        # eid-1 has 2 regions, eid-2 has 1
        assert len(group.recordings) == 3
        assert 'fiber_idx' in group.recordings.columns

    def test_filter_reflected_in_recordings(self):
        """Filtering by session type is reflected in recordings."""
        from iblnm.data import PhotometrySessionGroup
        group = PhotometrySessionGroup.from_catalog(self._make_catalog(), one=MagicMock())
        group.filter_sessions(session_types=('biased',), targetnms=False,
                              qc_blockers=set(),
                              min_performance=False, required_contrasts=False)
        assert all(group.recordings['session_type'] == 'biased')

    def test_from_catalog_enforces_schema(self):
        """from_catalog fills missing schema columns with typed defaults."""
        from iblnm.data import PhotometrySessionGroup
        group = PhotometrySessionGroup.from_catalog(self._make_catalog(), one=MagicMock())
        # 'lab' is in SESSION_SCHEMA but absent from the catalog fixture
        assert 'lab' in group._catalog.columns

    def test_recordings_reflects_refilter(self):
        """recordings updates automatically when filter_sessions is re-called."""
        from iblnm.data import PhotometrySessionGroup
        group = PhotometrySessionGroup.from_catalog(self._make_catalog(), one=MagicMock())
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
        group = PhotometrySessionGroup.from_catalog(catalog, one=MagicMock(), h5_dir=tmp_path)
        group.filter_sessions(session_types=False, targetnms=False,
                              qc_blockers={'MissingRawData'},
                              min_performance=False, required_contrasts=False)
        assert set(group.sessions['eid']) == {'eid-2'}

    def test_no_h5_dir_leaves_logged_errors_empty(self):
        """Without h5_dir, from_catalog skips the scan and logged_errors are all empty."""
        from iblnm.data import PhotometrySessionGroup
        group = PhotometrySessionGroup.from_catalog(self._make_catalog(), one=MagicMock())
        assert group._catalog['logged_errors'].apply(lambda x: x == []).all()

    def test_scan_h5_errors_false_reuses_existing_column(self, tmp_path):
        """scan_h5_errors=False skips the H5 scan and keeps a pre-existing
        logged_errors column without a merge collision."""
        from iblnm.data import PhotometrySessionGroup
        catalog = self._make_catalog()
        catalog['logged_errors'] = [['MissingRawData'] for _ in range(len(catalog))]
        group = PhotometrySessionGroup.from_catalog(
            catalog, one=MagicMock(), h5_dir=tmp_path, scan_h5_errors=False)
        assert group._catalog['logged_errors'].apply(
            lambda x: x == ['MissingRawData']).all()


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
        from iblnm.validation import FewUniqueSamples
        group = self._make_group_with_h5(tmp_path)

        def fn_with_nonfatal(ps):
            try:
                raise FewUniqueSamples("low samples")
            except FewUniqueSamples as e:
                ps.log_error(e)
            return 'ok'

        results = group.process(fn_with_nonfatal)
        assert all(r == 'ok' for r in results)

        # Check non-fatal error was written to H5
        with h5py.File(tmp_path / 'eid-0.h5', 'r') as f:
            error_types = [v.decode() for v in f['errors']['error_type'][:]]
            assert 'FewUniqueSamples' in error_types


class TestPhotometrySessionGroup:

    def test_len_matches_recordings(self):
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=2, regions_per=2)
        group = PhotometrySessionGroup(recs, one=MagicMock())
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
    events = ['stimOn_times', 'firstMovement_times', 'feedback_times']

    # Baseline = 0, post-event = post_event_value
    data = np.zeros((3, n_trials, n_time))
    post_mask = tpts >= 0
    data[:, :, post_mask] = post_event_value

    ps.responses = {
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
        'stimOn_times': np.linspace(10, 10 + n_trials, n_trials),
        'firstMovement_times': np.linspace(10.2, 10.2 + n_trials, n_trials),
        'feedback_times': np.linspace(11, 11 + n_trials, n_trials),
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
        assert 'stimOn_c0.0625_contra_correct' in vec.index
        assert 'stimOn_c0.0625_ipsi_correct' in vec.index
        assert 'feedback_c1_contra_incorrect' in vec.index
        assert 'feedback_c1_ipsi_incorrect' in vec.index
        # Zero contrast retains ipsi/contra (side matters for action contingencies)
        assert 'stimOn_c0_contra_correct' in vec.index
        assert 'stimOn_c0_ipsi_correct' in vec.index

    def test_custom_events_includes_firstMovement(self):
        """Passing events explicitly can include firstMovement."""
        ps = _make_session_with_responses(MagicMock(), n_trials=200)
        vec = ps.get_response_vector(
            brain_region='VTA-r', hemisphere='r',
            events=['stimOn_times', 'firstMovement_times', 'feedback_times'],
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
        events = ['stimOn_times', 'firstMovement_times', 'feedback_times']

        # Baseline (t<0) = 0, post-event varies by event
        rng = np.random.default_rng(0)
        data = np.zeros((3, n_trials, n_time))
        post_mask = tpts >= 0
        data[0, :, :][:, post_mask] = 1.0   # stimOn post-event = 1
        data[1, :, :][:, post_mask] = 2.0   # firstMov post-event = 2
        data[2, :, :][:, post_mask] = 3.0   # feedback post-event = 3

        ps.responses = {
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
            'stimOn_times': np.linspace(10, 10 + n_trials, n_trials),
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
        assert 'stimOn_c0_contra_correct' in vec.index
        assert 'stimOn_c1_ipsi_incorrect' in vec.index
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
        If True, set feedback_times = stimOn_times + 0.01 (response_time < 0.05).
    """
    import h5py

    rng = np.random.default_rng(seed)
    n_time = 61
    tpts = np.linspace(-1, 1, n_time)
    events = ['stimOn_times', 'firstMovement_times', 'feedback_times']
    contrasts = np.array([0.0, 0.0625, 0.125, 0.25, 1.0])

    # Pre-event = 0, post-event = 1.0
    post_mask = tpts >= 0

    stim_on = np.linspace(10, 10 + n_trials, n_trials)
    feedback = stim_on + 0.01 if fast_response else np.linspace(11, 11 + n_trials, n_trials)

    with h5py.File(path, 'w') as f:
        grp = f.create_group('trials')
        grp.create_dataset('stimOn_times', data=stim_on)
        grp.create_dataset('firstMovement_times',
                           data=stim_on + 0.2)
        grp.create_dataset('feedback_times', data=feedback)
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
        assert ps.responses == {}

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
             'brain_region': 'region-0', 'event': 'stimOn_times',
             'predictor': 'contrast', 'r2': 0.5, 'delta_r2': 0.1, 'n_trials': 80},
            {'eid': 'eid-1', 'subject': 'subj-1', 'target_NM': 'target-0',
             'brain_region': 'region-0', 'event': 'feedback_times',
             'predictor': 'reward', 'r2': 0.4, 'delta_r2': 0.2, 'n_trials': 70},
            {'eid': 'eid-99', 'subject': 'subj-9', 'target_NM': 'target-X',
             'brain_region': 'region-0', 'event': 'stimOn_times',
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

    def test_load_performance(self, tmp_path):
        group = self._make_group(n_eids=3, regions_per=1)
        df = pd.DataFrame([
            {'eid': 'eid-0', 'fraction_correct': 0.85},
            {'eid': 'eid-1', 'fraction_correct': 0.72},
            {'eid': 'eid-2', 'fraction_correct': 0.91},
            {'eid': 'eid-99', 'fraction_correct': 0.60},  # not in group
        ])
        path = tmp_path / 'performance.pqt'
        df.to_parquet(path, index=False)

        group.load_performance(path)
        assert len(group.performance) == 3
        assert 'eid-99' not in group.performance['eid'].values

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
    events = ['stimOn_times', 'firstMovement_times', 'feedback_times']
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
    """Group with one kept trial and three that each violate one filter.

    Single eid, single recording, single event. Trial 0 passes all filters;
    trials 1-3 each break exactly one of response_time>0.05, choice!=0,
    probabilityLeft==0.5.
    """
    from iblnm.data import PhotometrySessionGroup

    response_magnitudes = pd.DataFrame({
        'eid': 'eid-0',
        'subject': 's0',
        'target_NM': 'VTA-DA',
        'NM': 'DA',
        'brain_region': 'VTA',
        'hemisphere': 'r',
        'event': 'stimOn_times',
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
        group = _make_group_with_planted_trials()
        df = group._modeling_frame()
        assert df['trial'].tolist() == [0]

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
        # Non-positive reaction times -> _modeling_frame sets log to NaN.
        reg.loc[reg.index[::5], 'reaction_time'] = -1.0
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
    events = ['stimOn_times', 'firstMovement_times', 'feedback_times']

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
    ps.responses = {
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
        'stimOn_times': stim_on,
        'firstMovement_times': stim_on + reaction,
        'feedback_times': stim_on + reaction + movement,
        'signed_contrast': signed,
        'contrast': contrast_vals,
        'stim_side': sides,
        'feedbackType': rng.choice([1, -1], n_trials),
        'choice': rng.choice([-1, 1], n_trials),
        'probabilityLeft': np.full(n_trials, 0.5),
    })
    ps.wheel_velocity = rng.normal(0, 1, (n_trials, 50))
    ps.wheel_fs = 30.0
    return ps


class TestCompareResponseModels:
    """Tests for PhotometrySession.compare_response_models (drop-one family)."""

    @property
    def formulas(self):
        from iblnm.config import LMM_FORMULAS
        return LMM_FORMULAS['persession']

    def test_absent_region_returns_empty_frame(self):
        ps = _make_session_for_persession()
        out = ps.compare_response_models('NOT-A-REGION', self.formulas)
        assert out.empty
        assert list(out.columns) == [
            'brain_region', 'target_NM', 'event', 'predictor', 'r2',
            'delta_r2', 'n_trials',
        ]

    def test_informative_contrast_has_positive_delta_r2(self):
        ps = _make_session_for_persession()
        out = ps.compare_response_models('VTA-r', self.formulas)
        contrast_rows = out[out['predictor'] == 'contrast']
        assert not contrast_rows.empty
        assert (contrast_rows['delta_r2'] > 0).all()

    def test_rows_tagged_with_region_and_target_nm(self):
        ps = _make_session_for_persession()
        out = ps.compare_response_models('VTA-r', self.formulas)
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
        out = ps.compare_response_models('VTA-r', self.formulas)
        for event in set(out['event']):
            for name in self.formulas:
                assert (name, event) in ps.ols_fits

    def test_event_below_min_trials_is_skipped(self):
        ps = _make_session_for_persession(n_trials=120)
        out = ps.compare_response_models('VTA-r', self.formulas, min_trials=200)
        assert out.empty


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
        out = group.response_ols_dropone(self.formulas)

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
        out = group.response_ols_dropone(self.formulas)

        assert set(out['eid']) == {'eid-0'}
        assert 'eid-1' not in set(out['eid'])


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
            self._formula(), event_name='stimOn_times')
        assert isinstance(result, pd.DataFrame)
        for col in ('Intercept', 'contrast', 'side', 'reward', 'choice_side',
                    'log_reaction_time', 'peak_velocity', 'contrast:side'):
            assert col in result.columns

    def test_stored_as_attribute(self):
        """Result is stored as self.persession_ols_features."""
        group = _make_group_for_response_lmm()
        group.get_persession_ols_features(self._formula(), event_name='stimOn_times')
        assert group.persession_ols_features is not None
        assert len(group.persession_ols_features) > 0

    def test_index_structure(self):
        """Index has (eid, target_NM, fiber_idx) levels."""
        group = _make_group_for_response_lmm()
        result = group.get_persession_ols_features(
            self._formula(), event_name='stimOn_times')
        assert result.index.names == ['eid', 'target_NM', 'fiber_idx']

    def test_weight_by_se(self):
        """With weight_by_se=True, values are t-statistics (coef / SE)."""
        group = _make_group_for_response_lmm()
        coefs = group.get_persession_ols_features(
            self._formula(), event_name='stimOn_times', weight_by_se=False)
        group2 = _make_group_for_response_lmm()
        tstats = group2.get_persession_ols_features(
            self._formula(), event_name='stimOn_times', weight_by_se=True)
        assert not np.allclose(coefs.values, tstats.values)

    def test_one_row_per_recording(self):
        """Each scorable recording (eid × brain_region) produces one row."""
        group = _make_group_for_response_lmm()
        result = group.get_persession_ols_features(
            self._formula(), event_name='stimOn_times')
        # fixture has 6 recordings (3 subjects × 2 targets)
        assert len(result) == 6

    def test_persession_coefficient_count(self):
        """Output has 19 columns (6 mains + 12 interactions + intercept)."""
        group = _make_group_for_response_lmm()
        result = group.get_persession_ols_features(
            self._formula(), event_name='stimOn_times')
        assert result.shape[1] == 19

    def test_excludes_false_start_trials(self):
        """Trials with response_time <= 0.05 must be excluded; all-fast → empty result."""
        group = _make_group_for_response_lmm()
        group.trial_regressors['response_time'] = 0.01
        result = group.get_persession_ols_features(
            self._formula(), event_name='stimOn_times')
        assert len(result) == 0

    def test_excludes_nogo_trials(self):
        """Trials with choice == 0 must be excluded; all-nogo → empty result."""
        group = _make_group_for_response_lmm()
        group.trial_regressors = group.trial_regressors.copy()
        group.trial_regressors['choice'] = 0
        result = group.get_persession_ols_features(
            self._formula(), event_name='stimOn_times')
        assert len(result) == 0


class TestGLMFeaturesCCA:

    def test_cca_with_glm_features(self):
        """fit_cca works with persession_ols_features as X input."""
        import tempfile
        from iblnm.config import LMM_FORMULAS
        group = _make_group_for_response_lmm()
        group.get_persession_ols_features(
            LMM_FORMULAS['persession']['full'], event_name='stimOn_times')
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
        assert events == {'stimOn_times', 'feedback_times'}


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
    feedback = np.array([11.0, 21.5, 32.0])
    with h5py.File(path, 'w') as f:
        grp = f.create_group('trials')
        grp.create_dataset('stimOn_times', data=stim_on)
        grp.create_dataset('firstMovement_times', data=first_move)
        grp.create_dataset('feedback_times', data=feedback)
        grp.create_dataset('signed_contrast', data=np.array([-0.25, 0.0, 1.0]))
        grp.create_dataset('contrast', data=np.array([0.25, 0.0, 1.0]))
        grp.create_dataset('stim_side', data=np.array(['left', 'right', 'right'],
                                                      dtype='S5'))
        grp.create_dataset('choice', data=np.array([-1, 1, 1]))
        grp.create_dataset('feedbackType', data=np.array([1, -1, 1]))
        grp.create_dataset('probabilityLeft', data=np.full(3, 0.5))
        if with_wheel:
            wheel_grp = f.create_group('wheel/responses')
            velocity = np.array([[0.0, 1.0, -3.0],
                                 [np.nan, np.nan, np.nan],
                                 [2.0, -5.0, 1.0]])
            wheel_grp.create_dataset('velocity', data=velocity)
    return stim_on, first_move, feedback


class TestGetTrialRegressors:

    def test_trial_regressors_schema_and_values(self, tmp_path):
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=1, regions_per=1)
        stim_on, first_move, feedback = _write_trial_regressor_h5(
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
            df['movement_time'].values, feedback - first_move)
        np.testing.assert_allclose(
            df['response_time'].values, feedback - stim_on)

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
