"""Tests for iblnm.data module."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from iblnm.config import contrast_transform


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

    def test_raises_with_flipping_blocks(self, mock_session_series):
        from iblnm.data import PhotometrySession
        from iblnm.validation import BlockStructureBug
        series = mock_session_series.copy()
        series['session_type'] = 'biased'
        session = PhotometrySession(series, one=MagicMock(), load_data=False)
        session.trials = pd.DataFrame({'probabilityLeft': np.tile([0.8, 0.2], 50)})
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

    def test_does_not_raise_for_training(self, mock_session_series):
        from iblnm.data import PhotometrySession
        series = mock_session_series.copy()
        series['session_type'] = 'training'
        session = PhotometrySession(series, one=MagicMock(), load_data=False)
        session.trials = pd.DataFrame({'probabilityLeft': np.tile([0.8, 0.2], 50)})
        session.validate_block_structure()  # should not raise for training


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
        assert isinstance(session.responses, xr.DataArray)

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
        assert set(session.responses.dims) == {'region', 'event', 'trial', 'time'}
        assert 'VTA' in session.responses.coords['region'].values
        for event in RESPONSE_EVENTS:
            assert event in session.responses.coords['event'].values
        assert session.responses.sizes['trial'] == n

    def test_sel_region_event(self, mock_photometry_session):
        """Selecting by region and event returns (trial, time) array."""
        session = mock_photometry_session
        session.preprocess()
        n = 50
        session.trials = pd.DataFrame({
            'stimOn_times': np.linspace(99.5, 499.5, n),
            'firstMovement_times': np.linspace(100.3, 500.3, n),
            'feedback_times': np.linspace(101, 501, n),
        })
        session.extract_responses()
        sel = session.responses.sel(region='VTA', event='stimOn_times')
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
        tpts = session.responses.coords['time'].values
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
        assert list(session.responses.coords['event'].values) == ['feedback_times']
        assert session.responses.sizes['event'] == 1



# =============================================================================
# HDF5 Save/Load Tests
# =============================================================================

class TestSaveLoadH5:
    def test_save_preprocessed_float64(self, mock_photometry_session, tmp_path):
        """save_h5 should write preprocessed signal as float64 with timestamps."""
        session = mock_photometry_session
        session.preprocess()
        fpath = tmp_path / f'{session.eid}.h5'
        session.save_h5(fpath)

        import h5py
        with h5py.File(fpath, 'r') as f:
            assert f.attrs['eid'] == session.eid
            assert f.attrs['subject'] == session.subject
            assert f.attrs['fs'] == 30
            assert 'preprocessed/VTA' in f
            assert 'times' in f
            assert f['preprocessed/VTA'].dtype == np.float64
            np.testing.assert_allclose(
                f['preprocessed/VTA'][:],
                session.photometry['GCaMP_preprocessed']['VTA'].values,
                rtol=1e-10
            )
            np.testing.assert_allclose(
                f['times'][:],
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
            assert 'preprocessed/VTA' in f
            assert 'trials/choice' in f
            assert 'responses/VTA/stimOn_times' in f
            assert 'responses/VTA/feedback_times' in f
            # Verify response data matches xarray content
            resp_h5 = f['responses/VTA/stimOn_times'][:]
            resp_xr = session.responses.sel(region='VTA', event='stimOn_times').values
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
        original = session.responses.copy()

        fpath = tmp_path / f'{session.eid}.h5'
        session.save_h5(fpath)
        session.save_h5(fpath, mode='a')

        # Clear and reload
        session.responses = None
        session.load_h5(fpath)
        assert isinstance(session.responses, xr.DataArray)
        assert set(session.responses.dims) == {'region', 'event', 'trial', 'time'}
        np.testing.assert_allclose(
            session.responses.sel(region='VTA', event='stimOn_times').values,
            original.sel(region='VTA', event='stimOn_times').values,
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
    """Build a minimal (1 region, 1 event, n_trials, n_times) DataArray."""
    import xarray as xr
    data = np.array([[[vals]]] if vals.ndim == 1 else [[vals]])
    return xr.DataArray(
        data,
        dims=['region', 'event', 'trial', 'time'],
        coords={
            'region': [region],
            'event':  [event],
            'trial':  np.arange(data.shape[2]),
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
        np.testing.assert_allclose(result.values[0, 0, 0], [-2., 0., 2., 4., 6.])

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
        np.testing.assert_allclose(result.values[0, 0, 0], [-1., 1., 3., 5.])
        np.testing.assert_allclose(result.values[0, 0, 1], [-5., 5., 15., 25.])

    def test_does_not_modify_self_responses(self, mock_session_series):
        """Returns new DataArray; self.responses unchanged."""
        session = self._session(mock_session_series)
        tpts = np.array([-1.0, -0.5, 0.5, 1.0])
        vals = np.array([1., 2., 3., 4.])
        responses = _make_responses(tpts, vals)
        session.responses = responses.copy()
        original_vals = session.responses.values.copy()
        result = session.subtract_baseline(responses, window=(-1.0, 0.0))
        np.testing.assert_array_equal(session.responses.values, original_vals)
        assert result is not session.responses

    def test_defaults_to_self_responses(self, mock_session_series):
        """Calling without args operates on self.responses."""
        session = self._session(mock_session_series)
        tpts = np.array([-1.0, -0.5, 0.5, 1.0])
        vals = np.array([2., 4., 6., 8.])
        session.responses = _make_responses(tpts, vals)
        result = session.subtract_baseline(window=(-1.0, 0.0))
        # baseline = mean(2, 4) = 3.0; expected = [-1, 1, 3, 5]
        np.testing.assert_allclose(result.values[0, 0, 0], [-1., 1., 3., 5.])

    def test_empty_window_produces_nan(self, mock_session_series):
        """Window entirely outside time axis → baseline NaN → output all NaN."""
        session = self._session(mock_session_series)
        tpts = np.array([0.5, 1.0, 1.5])
        vals = np.array([1., 2., 3.])
        responses = _make_responses(tpts, vals)
        result = session.subtract_baseline(responses, window=(-1.0, 0.0))
        assert np.all(np.isnan(result.values))


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
        data = np.ones((1, 2, 2, 5))  # (region, event, trial, time)
        responses = xr.DataArray(
            data,
            dims=['region', 'event', 'trial', 'time'],
            coords={
                'region': ['R'],
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
        mat = result.sel(region='R', event='stimOn_times').values
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
        mat = result.sel(region='R', event='firstMovement_times').values
        assert not np.any(np.isnan(mat))

    def test_nan_dt_not_masked(self, mock_session_series):
        """Trial 1 has NaN firstMovement → stimOn response fully intact."""
        session, responses = self._make_session_and_responses(mock_session_series)
        result = session.mask_subsequent_events(
            responses,
            event_order=['stimOn_times', 'firstMovement_times', 'feedback_times'],
        )
        mat = result.sel(region='R', event='stimOn_times').values
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
        data = np.ones((1, 1, 2, 3))
        responses = xr.DataArray(
            data,
            dims=['region', 'event', 'trial', 'time'],
            coords={'region': ['R'], 'event': ['feedback_times'],
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

def _make_recordings_df(n_eids=2, regions_per=2):
    """Helper to build a recordings DataFrame."""
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
    data = np.zeros((1, 3, n_trials, n_time))
    post_mask = tpts >= 0
    data[:, :, :, post_mask] = post_event_value

    ps.responses = xr.DataArray(
        data, dims=['region', 'event', 'trial', 'time'],
        coords={'region': ['VTA-r'], 'event': events,
                'trial': np.arange(n_trials), 'time': tpts},
    )

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
        data = np.zeros((1, 3, n_trials, n_time))
        post_mask = tpts >= 0
        data[0, 0, :, :][:, post_mask] = 1.0   # stimOn post-event = 1
        data[0, 1, :, :][:, post_mask] = 2.0   # firstMov post-event = 2
        data[0, 2, :, :][:, post_mask] = 3.0   # feedback post-event = 3

        ps.responses = xr.DataArray(
            data, dims=['region', 'event', 'trial', 'time'],
            coords={'region': ['VTA-r'], 'event': events,
                    'trial': np.arange(n_trials), 'time': tpts},
        )

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

def _write_h5(path, n_trials=100, regions=('VTA-r',), seed=42):
    """Write a minimal H5 file with trials and responses."""
    import h5py

    rng = np.random.default_rng(seed)
    n_time = 61
    tpts = np.linspace(-1, 1, n_time)
    events = ['stimOn_times', 'firstMovement_times', 'feedback_times']
    contrasts = np.array([0.0, 0.0625, 0.125, 0.25, 1.0])

    # Baseline = 0, post-event = 1.0
    post_mask = tpts >= 0

    with h5py.File(path, 'w') as f:
        grp = f.create_group('trials')
        grp.create_dataset('stimOn_times', data=np.linspace(10, 10 + n_trials, n_trials))
        grp.create_dataset('firstMovement_times',
                           data=np.linspace(10.2, 10.2 + n_trials, n_trials))
        grp.create_dataset('feedback_times',
                           data=np.linspace(11, 11 + n_trials, n_trials))
        sides = rng.choice(['left', 'right'], n_trials)
        contrast_vals = rng.choice(contrasts, n_trials)
        signed = np.where(sides == 'left', -1, 1).astype(float) * contrast_vals
        grp.create_dataset('signed_contrast', data=signed)
        grp.create_dataset('contrast', data=contrast_vals)
        # Store stim_side as fixed-length bytes for HDF5 compatibility
        grp.create_dataset('stim_side', data=np.array(sides, dtype='S5'))
        grp.create_dataset('feedbackType', data=rng.choice([1, -1], n_trials))
        grp.create_dataset('choice', data=rng.choice([-1, 1], n_trials))
        grp.create_dataset('probabilityLeft', data=np.full(n_trials, 0.5))

        resp_grp = f.create_group('responses')
        resp_grp.create_dataset('time', data=tpts)
        for region in regions:
            region_grp = resp_grp.create_group(region)
            for event in events:
                data = np.zeros((n_trials, n_time))
                data[:, post_mask] = 1.0
                region_grp.create_dataset(event, data=data)


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
        assert not hasattr(ps, 'responses')

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
# filter_recordings Tests
# =============================================================================

class TestFilterRecordings:

    def test_filters_session_types(self):
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=2, regions_per=1)
        recs.loc[0, 'session_type'] = 'habituation'
        group = PhotometrySessionGroup(recs, one=MagicMock())
        group.filter_recordings(
            session_types=('biased',),
            log_fpaths=[],
            targetnms=['target-0', 'target-1'],
        )
        assert 'eid-0' not in group.recordings['eid'].values

    def test_excludes_subjects(self):
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=2, regions_per=1)
        group = PhotometrySessionGroup(recs, one=MagicMock())
        group.filter_recordings(
            exclude_subjects=['subj-0'],
            log_fpaths=[],
            targetnms=['target-0', 'target-1'],
        )
        assert 'subj-0' not in group.recordings['subject'].values

    def test_filters_qc_blockers(self):
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=2, regions_per=1)
        error_log = pd.DataFrame([{
            'eid': 'eid-0',
            'error_type': 'MissingExtractedData',
            'error_message': 'test',
            'traceback': '',
        }])
        group = PhotometrySessionGroup(recs, one=MagicMock())
        group.filter_recordings(
            log_fpaths=[error_log],
            targetnms=['target-0', 'target-1'],
        )
        assert 'eid-0' not in group.recordings['eid'].values

    def test_filters_targetnms(self):
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=1, regions_per=2)
        group = PhotometrySessionGroup(recs, one=MagicMock())
        group.filter_recordings(
            log_fpaths=[],
            targetnms=['target-0'],
        )
        assert all(group.recordings['target_NM'] == 'target-0')

    def test_empty_after_filtering(self):
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=2, regions_per=1)
        group = PhotometrySessionGroup(recs, one=MagicMock())
        group.filter_recordings(
            session_types=('ephys',),  # none match
            log_fpaths=[],
            targetnms=['target-0'],
        )
        assert len(group) == 0

    def test_returns_self(self):
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=1, regions_per=1)
        group = PhotometrySessionGroup(recs, one=MagicMock())
        result = group.filter_recordings(
            log_fpaths=[],
            targetnms=['target-0'],
        )
        assert result is group


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
            'brain_region', 'hemisphere', 'event', 'trial',
            'stim_side', 'signed_contrast', 'contrast', 'choice',
            'feedbackType', 'probabilityLeft',
            'response_early',
        }
        assert expected_cols.issubset(set(df_events.columns))

    def test_response_magnitudes_excludes_timing(self, tmp_path):
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=1, regions_per=1)
        _write_h5(tmp_path / 'eid-0.h5', n_trials=50)
        group = PhotometrySessionGroup(recs, one=MagicMock(), h5_dir=tmp_path)
        group.get_response_magnitudes()
        assert 'reaction_time' not in group.response_magnitudes.columns
        assert 'movement_time' not in group.response_magnitudes.columns

    def test_trial_timing_populated(self, tmp_path):
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=1, regions_per=1)
        _write_h5(tmp_path / 'eid-0.h5', n_trials=50)
        group = PhotometrySessionGroup(recs, one=MagicMock(), h5_dir=tmp_path)
        group.get_response_magnitudes()
        assert group.trial_timing is not None
        assert {'eid', 'trial', 'reaction_time', 'movement_time'}.issubset(
            group.trial_timing.columns)
        assert 'event' not in group.trial_timing.columns

    def test_trial_timing_keyed_by_eid_trial(self, tmp_path):
        """trial_timing has one row per (eid, trial), not per event."""
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=1, regions_per=1)
        n_trials = 50
        _write_h5(tmp_path / 'eid-0.h5', n_trials=n_trials)
        group = PhotometrySessionGroup(recs, one=MagicMock(), h5_dir=tmp_path)
        group.get_response_magnitudes()
        assert len(group.trial_timing) == n_trials
        assert group.trial_timing.duplicated(subset=['eid', 'trial']).sum() == 0

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
        """Post-event = 1.0, baseline = 0 → response_early should be ~1.0."""
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=1, regions_per=1)
        _write_h5(tmp_path / 'eid-0.h5', n_trials=50)
        group = PhotometrySessionGroup(recs, one=MagicMock(), h5_dir=tmp_path)
        df_events = group.get_response_magnitudes()
        # After baseline subtraction, post-event signal = 1.0
        # Response magnitude in early window should be ~1.0
        magnitudes = df_events['response_early'].dropna()
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

    def test_trial_timing_has_movement_time(self, tmp_path):
        """movement_time should be computed from trial times."""
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=1, regions_per=1)
        _write_h5(tmp_path / 'eid-0.h5', n_trials=50, seed=0)
        group = PhotometrySessionGroup(recs, one=MagicMock(), h5_dir=tmp_path)
        group.get_response_magnitudes()
        assert 'movement_time' in group.trial_timing.columns
        assert group.trial_timing['movement_time'].notna().any()


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
                                    'response_early': response,
                                    'session_type': 'biased',
                                })

    df_events = pd.DataFrame(rows)

    # Separate trial_timing from response_magnitudes
    trial_timing = (
        df_events[['eid', 'trial', 'reaction_time']]
        .drop_duplicates(subset=['eid', 'trial'])
        .copy()
    )
    trial_timing['movement_time'] = 0.15
    df_events = df_events.drop(columns=['reaction_time'])

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
    group.response_magnitudes = df_events
    group.trial_timing = trial_timing
    return group


class TestFitLMM:

    def test_stores_lmm_results(self):
        group = _make_group_with_events()
        group.fit_lmm()
        assert group.lmm_results is not None
        assert isinstance(group.lmm_results, dict)

    def test_keys_are_target_event_tuples(self):
        group = _make_group_with_events()
        group.fit_lmm()
        for key in group.lmm_results:
            assert isinstance(key, tuple)
            assert len(key) == 2

    def test_values_are_lmm_results(self):
        from iblnm.analysis import LMMResult
        group = _make_group_with_events()
        group.fit_lmm()
        for result in group.lmm_results.values():
            assert isinstance(result, LMMResult)

    def test_results_have_emms(self):
        """Each result should have emm_reward and emm_side attributes."""
        group = _make_group_with_events()
        group.fit_lmm()
        for result in group.lmm_results.values():
            assert hasattr(result, 'emm_reward')
            assert hasattr(result, 'emm_side')
            assert isinstance(result.emm_reward, pd.DataFrame)
            assert isinstance(result.emm_side, pd.DataFrame)

    def test_results_have_contrast_slopes(self):
        group = _make_group_with_events()
        group.fit_lmm()
        for result in group.lmm_results.values():
            assert hasattr(result, 'contrast_slopes')
            assert isinstance(result.contrast_slopes, pd.DataFrame)

    def test_saves_coefficients(self):
        """All coefficient summaries should be aggregated."""
        group = _make_group_with_events()
        group.fit_lmm()
        assert group.lmm_coefficients is not None
        assert isinstance(group.lmm_coefficients, pd.DataFrame)
        assert 'target_NM' in group.lmm_coefficients.columns
        assert 'event' in group.lmm_coefficients.columns

    def test_requires_events(self):
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=1, regions_per=1)
        group = PhotometrySessionGroup(recs, one=MagicMock())
        with pytest.raises(ValueError, match='response_magnitudes'):
            group.fit_lmm()

    def test_requires_trial_timing(self):
        group = _make_group_with_events()
        group.trial_timing = None
        with pytest.raises(ValueError, match='trial_timing'):
            group.fit_lmm()

    def test_re_formulas_default_intercept_only(self):
        """Default re_formulas=['1'] should produce intercept-only models."""
        group = _make_group_with_events()
        group.fit_lmm()
        for result in group.lmm_results.values():
            for effects in result.random_effects.values():
                assert 'log_contrast' not in effects.index

    def test_re_formulas_selects_maximal_converging(self):
        """When given multiple re_formulas, selects the most complex
        that converges for all groups."""
        group = _make_group_with_events()
        group.fit_lmm(re_formulas=['log_contrast', '1'])
        # All results should use the same RE structure
        re_structures = set()
        for result in group.lmm_results.values():
            effects = list(result.random_effects.values())[0]
            re_structures.add(tuple(sorted(effects.index)))
        assert len(re_structures) == 1

    def test_re_formulas_stores_selected_formula(self):
        """The selected RE formula should be stored on the group."""
        group = _make_group_with_events()
        group.fit_lmm(re_formulas=['log_contrast', '1'])
        assert hasattr(group, 'lmm_re_formula')
        assert group.lmm_re_formula in ('log_contrast', '1')


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
            'psych_50_lapse_left', 'psych_50_lapse_right',
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

    def test_returns_dataframe(self):
        """Returns a DataFrame with GLM coefficient columns."""
        group = _make_group_with_events()
        result = group.get_glm_response_features(event_name='stimOn_times')
        assert isinstance(result, pd.DataFrame)
        assert 'log_contrast' in result.columns
        assert 'side' in result.columns
        assert 'side:feedback' in result.columns

    def test_stored_as_attribute(self):
        """Result is stored as self.glm_response_features."""
        group = _make_group_with_events()
        group.get_glm_response_features(event_name='stimOn_times')
        assert group.glm_response_features is not None
        assert len(group.glm_response_features) > 0

    def test_index_structure(self):
        """Index has (eid, target_NM, fiber_idx) levels."""
        group = _make_group_with_events()
        result = group.get_glm_response_features(event_name='stimOn_times')
        assert result.index.names == ['eid', 'target_NM', 'fiber_idx']

    def test_weight_by_se(self):
        """With weight_by_se=True, values are t-statistics (coef / SE)."""
        group = _make_group_with_events()
        coefs = group.get_glm_response_features(
            event_name='stimOn_times', weight_by_se=False)
        group2 = _make_group_with_events()
        tstats = group2.get_glm_response_features(
            event_name='stimOn_times', weight_by_se=True)
        assert not np.allclose(coefs.values, tstats.values)

    def test_one_row_per_recording(self):
        """Each recording (eid × brain_region) produces one row."""
        group = _make_group_with_events()
        result = group.get_glm_response_features(event_name='stimOn_times')
        # _make_group_with_events has 6 recordings (3 subjects × 2 targets)
        assert len(result) == 6

    def test_seven_coefficients(self):
        """Output has 7 coefficient columns."""
        group = _make_group_with_events()
        result = group.get_glm_response_features(event_name='stimOn_times')
        assert result.shape[1] == 7


class TestGLMFeaturesCCA:

    def test_cca_with_glm_features(self):
        """fit_cca works with glm_response_features as X input."""
        import tempfile
        group = _make_group_with_events()
        group.get_glm_response_features(event_name='stimOn_times')
        group.response_features = group.glm_response_features
        perf = _make_mock_performance(group)
        with tempfile.NamedTemporaryFile(suffix='.pqt', delete=False) as f:
            perf.to_parquet(f.name)
            group.get_psychometric_features(performance_path=f.name)
        result = group.fit_cca(n_permutations=0)
        assert result.x_weights.shape[1] > 0
        assert 'log_contrast' in result.x_weights.index


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
        """Each recording should produce one cache entry per event."""
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=1, regions_per=1)
        _write_h5(tmp_path / 'eid-0.h5', n_trials=50, seed=0)
        group = PhotometrySessionGroup(recs, one=MagicMock(), h5_dir=tmp_path)
        group.load_response_traces()
        events = {k[2] for k in group.response_traces.keys()}
        assert len(events) == 3  # stimOn, firstMovement, feedback


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
        assert n_rec_events == 2 * 3  # 2 recordings × 3 events

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


# =============================================================================
# Wheel Kinematics Enrichment and LMM Tests
# =============================================================================


def _write_h5_with_wheel(path, n_trials=100, seed=42):
    """Write an H5 file with trials, responses, and wheel velocity."""
    import h5py

    rng = np.random.default_rng(seed)
    n_time = 61
    tpts = np.linspace(-1, 1, n_time)
    events = ['stimOn_times', 'firstMovement_times', 'feedback_times']
    contrasts = np.array([0.0, 0.0625, 0.125, 0.25, 1.0])

    post_mask = tpts >= 0
    stim_times = np.linspace(10, 10 + n_trials, n_trials)
    fm_times = stim_times + rng.uniform(0.1, 0.5, n_trials)
    fb_times = fm_times + rng.uniform(0.1, 0.5, n_trials)

    # Build wheel velocity: (n_trials, max_samples)
    wheel_fs = 100
    max_duration = np.max(fb_times - stim_times)
    max_samples = int(np.ceil(max_duration * wheel_fs)) + 1
    wheel_vel = np.full((n_trials, max_samples), np.nan, dtype=np.float32)
    for t in range(n_trials):
        dur = fb_times[t] - stim_times[t]
        n_samp = int(np.ceil(dur * wheel_fs))
        wheel_vel[t, :n_samp] = rng.normal(0, 5.0, n_samp).astype(np.float32)

    with h5py.File(path, 'w') as f:
        grp = f.create_group('trials')
        grp.create_dataset('stimOn_times', data=stim_times)
        grp.create_dataset('firstMovement_times', data=fm_times)
        grp.create_dataset('feedback_times', data=fb_times)
        sides = rng.choice(['left', 'right'], n_trials)
        contrast_vals = rng.choice(contrasts, n_trials)
        signed = np.where(sides == 'left', -1, 1).astype(float) * contrast_vals
        grp.create_dataset('signed_contrast', data=signed)
        grp.create_dataset('contrast', data=contrast_vals)
        grp.create_dataset('stim_side', data=np.array(sides, dtype='S5'))
        grp.create_dataset('feedbackType', data=rng.choice([1, -1], n_trials))
        grp.create_dataset('choice', data=rng.choice([-1, 1], n_trials))
        grp.create_dataset('probabilityLeft', data=np.full(n_trials, 0.5))

        resp_grp = f.create_group('responses')
        resp_grp.create_dataset('time', data=tpts)
        region = 'VTA-r'
        region_grp = resp_grp.create_group(region)
        for event in events:
            data = np.zeros((n_trials, n_time))
            data[:, post_mask] = 1.0
            region_grp.create_dataset(event, data=data)

        w_grp = f.create_group('wheel')
        w_grp.create_dataset('velocity', data=wheel_vel, compression='gzip')
        w_grp.attrs['fs'] = wheel_fs
        w_grp.attrs['t0_event'] = 'stimOn_times'
        w_grp.attrs['t1_event'] = 'feedback_times'


class TestEnrichPeakVelocity:

    def test_peak_velocity_stored_separately(self, tmp_path):
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=2, regions_per=1)
        for i in range(2):
            _write_h5_with_wheel(tmp_path / f'eid-{i}.h5', n_trials=50, seed=i)
        group = PhotometrySessionGroup(recs, one=MagicMock(), h5_dir=tmp_path)
        group.load_response_traces()
        group.get_response_magnitudes()
        group.enrich_peak_velocity()
        assert 'peak_velocity' not in group.response_magnitudes.columns
        assert group.peak_velocity is not None
        assert {'eid', 'trial', 'peak_velocity'}.issubset(
            group.peak_velocity.columns)

    def test_peak_velocity_is_positive(self, tmp_path):
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=1, regions_per=1)
        _write_h5_with_wheel(tmp_path / 'eid-0.h5', n_trials=50, seed=0)
        group = PhotometrySessionGroup(recs, one=MagicMock(), h5_dir=tmp_path)
        group.load_response_traces()
        group.get_response_magnitudes()
        group.enrich_peak_velocity()
        valid = group.peak_velocity['peak_velocity'].dropna()
        assert (valid >= 0).all()

    def test_does_not_modify_response_magnitudes(self, tmp_path):
        """enrich_peak_velocity should not add columns to response_magnitudes."""
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=1, regions_per=1)
        _write_h5_with_wheel(tmp_path / 'eid-0.h5', n_trials=50, seed=0)
        group = PhotometrySessionGroup(recs, one=MagicMock(), h5_dir=tmp_path)
        group.get_response_magnitudes()
        cols_before = set(group.response_magnitudes.columns)
        group.enrich_peak_velocity()
        cols_after = set(group.response_magnitudes.columns)
        assert cols_before == cols_after

    def test_skips_sessions_without_wheel(self, tmp_path):
        """Sessions without wheel data get NaN for peak_velocity."""
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=2, regions_per=1)
        # Write one with wheel, one without
        _write_h5_with_wheel(tmp_path / 'eid-0.h5', n_trials=50, seed=0)
        _write_h5(tmp_path / 'eid-1.h5', n_trials=50, seed=1)
        group = PhotometrySessionGroup(recs, one=MagicMock(), h5_dir=tmp_path)
        group.load_response_traces()
        group.get_response_magnitudes()
        group.enrich_peak_velocity()
        df = group.peak_velocity
        eid0_pv = df[df['eid'] == 'eid-0']['peak_velocity']
        eid1_pv = df[df['eid'] == 'eid-1']['peak_velocity']
        assert eid0_pv.notna().any()
        assert eid1_pv.isna().all()


class TestFitWheelLMM:

    def test_requires_peak_velocity(self, tmp_path):
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=4, regions_per=1)
        for i in range(4):
            _write_h5_with_wheel(
                tmp_path / f'eid-{i}.h5', n_trials=100, seed=i)
        group = PhotometrySessionGroup(recs, one=MagicMock(), h5_dir=tmp_path)
        group.load_response_traces()
        group.get_response_magnitudes()
        # Don't call enrich_peak_velocity
        with pytest.raises(ValueError, match='peak_velocity'):
            group.fit_wheel_lmm()

    def test_returns_results(self, tmp_path):
        from iblnm.data import PhotometrySessionGroup
        # Need enough subjects for LMM — use 4 eids with 2 subjects
        recs = _make_recordings_df(n_eids=4, regions_per=1)
        for i in range(4):
            _write_h5_with_wheel(
                tmp_path / f'eid-{i}.h5', n_trials=100, seed=i)
        group = PhotometrySessionGroup(recs, one=MagicMock(), h5_dir=tmp_path)
        group.load_response_traces()
        group.get_response_magnitudes()
        group.enrich_peak_velocity()
        group.fit_wheel_lmm()
        assert group.wheel_lmm_results is not None
        assert group.wheel_lmm_summary is not None

    def test_summary_has_expected_columns(self, tmp_path):
        from iblnm.data import PhotometrySessionGroup
        recs = _make_recordings_df(n_eids=4, regions_per=1)
        for i in range(4):
            _write_h5_with_wheel(
                tmp_path / f'eid-{i}.h5', n_trials=100, seed=i)
        group = PhotometrySessionGroup(recs, one=MagicMock(), h5_dir=tmp_path)
        group.load_response_traces()
        group.get_response_magnitudes()
        group.enrich_peak_velocity()
        group.fit_wheel_lmm()
        if len(group.wheel_lmm_summary) > 0:
            expected = {'target_NM', 'contrast', 'dv', 'delta_r2',
                        'lrt_pvalue', 'nm_coefficient', 'nm_pvalue',
                        'n_trials', 'n_subjects'}
            assert expected.issubset(set(group.wheel_lmm_summary.columns))


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
        'intercept', 'log_contrast', 'side', 'feedback',
        'log_contrast:side', 'log_contrast:feedback', 'side:feedback',
    ]
    psych_cols = [
        'psych_50_threshold', 'psych_50_bias',
        'psych_50_lapse_left', 'psych_50_lapse_right',
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
    group.glm_response_features = pd.DataFrame(
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
