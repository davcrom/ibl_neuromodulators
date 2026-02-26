"""Tests for iblnm.data module."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch


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
        from iblnm.validation import TrialsNotInPhotometryTime
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

        result = session.preprocess()

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

        result = session.preprocess(
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
        import xarray as xr
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
