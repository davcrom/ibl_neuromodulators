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
        from iblnm.data import InsufficientTrials
        assert issubclass(InsufficientTrials, Exception)

    def test_block_structure_bug_is_exception(self):
        from iblnm.data import BlockStructureBug
        assert issubclass(BlockStructureBug, Exception)

    def test_incomplete_event_times_is_exception(self):
        from iblnm.data import IncompleteEventTimes
        assert issubclass(IncompleteEventTimes, Exception)

    def test_incomplete_event_times_stores_missing_events(self):
        from iblnm.data import IncompleteEventTimes
        exc = IncompleteEventTimes(['goCue_times', 'feedback_times'])
        assert exc.missing_events == ['goCue_times', 'feedback_times']
        assert 'goCue_times' in str(exc)

    def test_trials_not_in_photometry_time_is_exception(self):
        from iblnm.data import TrialsNotInPhotometryTime
        assert issubclass(TrialsNotInPhotometryTime, Exception)

    def test_missing_extracted_data_is_exception(self):
        from iblnm.data import MissingExtractedData
        assert issubclass(MissingExtractedData, Exception)

    def test_missing_raw_data_is_exception(self):
        from iblnm.data import MissingRawData
        assert issubclass(MissingRawData, Exception)

    def test_band_inversion_is_exception(self):
        from iblnm.data import BandInversion
        assert issubclass(BandInversion, Exception)

    def test_early_samples_is_exception(self):
        from iblnm.data import EarlySamples
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

    def test_returns_error_when_insufficient(self, mock_session_series):
        from iblnm.data import PhotometrySession
        session = PhotometrySession(mock_session_series, one=MagicMock(), load_data=False)
        session.trials = pd.DataFrame({'a': range(50)})  # 50 < MIN_NTRIALS (90)
        errors = session.validate_n_trials()
        assert len(errors) == 1
        assert errors[0]['error_type'] == 'InsufficientTrials'
        assert errors[0]['eid'] == 'test-eid-123'

    def test_returns_empty_when_sufficient(self, mock_session_series):
        from iblnm.data import PhotometrySession
        session = PhotometrySession(mock_session_series, one=MagicMock(), load_data=False)
        session.trials = pd.DataFrame({'a': range(200)})
        assert session.validate_n_trials() == []


class TestValidateBlockStructure:
    """Tests for PhotometrySession.validate_block_structure."""

    def test_returns_error_with_flipping_blocks(self, mock_session_series):
        from iblnm.data import PhotometrySession
        series = mock_session_series.copy()
        series['session_type'] = 'biased'
        session = PhotometrySession(series, one=MagicMock(), load_data=False)
        # Rapidly flipping blocks
        session.trials = pd.DataFrame({
            'probabilityLeft': np.tile([0.8, 0.2], 50),
        })
        errors = session.validate_block_structure()
        assert len(errors) == 1
        assert errors[0]['error_type'] == 'BlockStructureBug'
        assert errors[0]['eid'] == 'test-eid-123'

    def test_returns_empty_with_valid_blocks(self, mock_session_series):
        from iblnm.data import PhotometrySession
        series = mock_session_series.copy()
        series['session_type'] = 'biased'
        session = PhotometrySession(series, one=MagicMock(), load_data=False)
        session.trials = pd.DataFrame({
            'probabilityLeft': np.concatenate([np.full(100, 0.8), np.full(100, 0.2)]),
        })
        assert session.validate_block_structure() == []

    def test_skips_for_training(self, mock_session_series):
        from iblnm.data import PhotometrySession
        series = mock_session_series.copy()
        series['session_type'] = 'training'
        session = PhotometrySession(series, one=MagicMock(), load_data=False)
        session.trials = pd.DataFrame({
            'probabilityLeft': np.tile([0.8, 0.2], 50),  # Would be flagged for biased
        })
        assert session.validate_block_structure() == []


class TestValidateEventCompleteness:
    """Tests for PhotometrySession.validate_event_completeness."""

    def test_returns_error_with_missing_events(self, mock_session_series):
        from iblnm.data import PhotometrySession
        session = PhotometrySession(mock_session_series, one=MagicMock(), load_data=False)
        n = 100
        stim_on = np.random.rand(n)
        stim_on[:85] = np.nan  # 15% present, below threshold
        session.trials = pd.DataFrame({
            'stimOn_times': stim_on,
            'feedback_times': np.random.rand(n),
        })
        errors = session.validate_event_completeness()
        assert len(errors) == 1
        assert errors[0]['error_type'] == 'IncompleteEventTimes'
        assert 'stimOn_times' in errors[0]['error_message']

    def test_returns_empty_when_all_complete(self, mock_session_series):
        from iblnm.data import PhotometrySession
        session = PhotometrySession(mock_session_series, one=MagicMock(), load_data=False)
        n = 100
        session.trials = pd.DataFrame({
            'stimOn_times': np.random.rand(n),
            'feedback_times': np.random.rand(n),
        })
        assert session.validate_event_completeness() == []


class TestValidateTrialsInPhotometryTime:
    """Tests for PhotometrySession.validate_trials_in_photometry_time."""

    def test_returns_error_when_trials_outside(self, mock_session_series, mock_photometry_data):
        from iblnm.data import PhotometrySession
        session = PhotometrySession(mock_session_series, one=MagicMock(), load_data=False)
        session.photometry = mock_photometry_data
        # Trials start before photometry
        session.trials = pd.DataFrame({
            'stimOn_times': [-10.0, 100.0],
            'feedback_times': [100.0, 200.0],
        })
        errors = session.validate_trials_in_photometry_time()
        assert len(errors) == 1
        assert errors[0]['error_type'] == 'TrialsNotInPhotometryTime'
        assert errors[0]['eid'] == 'test-eid-123'

    def test_returns_empty_when_trials_inside(self, mock_session_series, mock_photometry_data):
        from iblnm.data import PhotometrySession
        session = PhotometrySession(mock_session_series, one=MagicMock(), load_data=False)
        session.photometry = mock_photometry_data
        # Trials within photometry window [0, 600]
        session.trials = pd.DataFrame({
            'stimOn_times': [10.0, 100.0],
            'feedback_times': [100.0, 500.0],
        })
        assert session.validate_trials_in_photometry_time() == []

    def test_uses_preprocessed_band_when_no_raw(self, mock_photometry_session):
        """Should fall back to GCaMP_preprocessed when GCaMP is not available."""
        session = mock_photometry_session
        session.preprocess()
        # Remove raw band, keep only preprocessed
        del session.photometry['GCaMP']
        del session.photometry['Isosbestic']
        # Trials within preprocessed window
        session.trials = pd.DataFrame({
            'stimOn_times': [10.0, 100.0],
            'feedback_times': [100.0, 500.0],
        })
        assert session.validate_trials_in_photometry_time() == []


class TestValidateQc:
    """Tests for PhotometrySession.validate_qc."""

    def test_returns_empty_when_qc_clean(self, mock_session_series):
        from iblnm.data import PhotometrySession
        session = PhotometrySession(mock_session_series, one=MagicMock(), load_data=False)
        session.qc = pd.DataFrame({
            'brain_region': ['VTA'],
            'band': ['GCaMP'],
            'n_band_inversions': [0],
            'n_early_samples': [0],
        })
        assert session.validate_qc() == []

    def test_returns_error_on_band_inversions(self, mock_session_series):
        from iblnm.data import PhotometrySession
        session = PhotometrySession(mock_session_series, one=MagicMock(), load_data=False)
        session.qc = pd.DataFrame({
            'brain_region': ['VTA'],
            'band': ['GCaMP'],
            'n_band_inversions': [3],
            'n_early_samples': [0],
        })
        errors = session.validate_qc()
        assert len(errors) == 1
        assert errors[0]['error_type'] == 'BandInversion'

    def test_returns_error_on_early_samples(self, mock_session_series):
        from iblnm.data import PhotometrySession
        session = PhotometrySession(mock_session_series, one=MagicMock(), load_data=False)
        session.qc = pd.DataFrame({
            'brain_region': ['VTA'],
            'band': ['GCaMP'],
            'n_band_inversions': [0],
            'n_early_samples': [5],
        })
        errors = session.validate_qc()
        assert len(errors) == 1
        assert errors[0]['error_type'] == 'EarlySamples'

    def test_returns_both_errors(self, mock_session_series):
        from iblnm.data import PhotometrySession
        session = PhotometrySession(mock_session_series, one=MagicMock(), load_data=False)
        session.qc = pd.DataFrame({
            'brain_region': ['VTA'],
            'band': ['GCaMP'],
            'n_band_inversions': [3],
            'n_early_samples': [5],
        })
        errors = session.validate_qc()
        assert len(errors) == 2
        error_types = {e['error_type'] for e in errors}
        assert error_types == {'BandInversion', 'EarlySamples'}

    def test_returns_empty_when_qc_empty(self, mock_session_series):
        from iblnm.data import PhotometrySession
        session = PhotometrySession(mock_session_series, one=MagicMock(), load_data=False)
        assert session.validate_qc() == []


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


