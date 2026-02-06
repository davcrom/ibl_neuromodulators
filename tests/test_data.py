"""Tests for iblnm.data module."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock


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
    session.has_photometry = True
    session.targets = {'gcamp': ['VTA'], 'isosbestic': ['VTA']}

    return session


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
        """Preprocess should compute bleaching_tau and iso_correlation in qc dict."""
        session = mock_photometry_session

        session.preprocess()

        # QC stored as nested dict: {brain_region: {band: {metric: value}}}
        assert hasattr(session, 'qc')
        assert isinstance(session.qc, dict)
        assert 'VTA' in session.qc
        assert 'GCaMP' in session.qc['VTA']
        # bleaching_tau should be positive and in reasonable range
        tau = session.qc['VTA']['GCaMP']['bleaching_tau']
        assert 100 < tau < 600  # Known fixture tau=300, allow wide margin for fit
        # iso_correlation: R² should be high (signals share bleaching component)
        assert 0.8 < session.qc['VTA']['GCaMP']['iso_correlation'] <= 1.0

    def test_preprocess_raises_when_no_photometry(self, mock_session_series):
        """Should raise ValueError if photometry not loaded."""
        from iblnm.data import PhotometrySession

        mock_one = MagicMock()
        session = PhotometrySession(mock_session_series, one=mock_one, load_data=False)
        session.has_photometry = False

        with pytest.raises(ValueError, match="not loaded"):
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
        # iso_correlation should not be present for single-band pipeline
        assert 'iso_correlation' not in session.qc['VTA']['GCaMP']
        assert 'bleaching_tau' in session.qc['VTA']['GCaMP']

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

    def test_qc_initialized_as_empty_dict(self, mock_session_series):
        """qc should be empty dict on init."""
        from iblnm.data import PhotometrySession

        mock_one = MagicMock()
        session = PhotometrySession(mock_session_series, one=mock_one, load_data=False)

        assert session.qc == {}

    def test_qc_to_dataframe(self, mock_photometry_session):
        """qc_to_dataframe should convert nested dict to DataFrame."""
        session = mock_photometry_session
        session.preprocess()

        df = session.qc_to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert 'brain_region' in df.columns
        assert 'band' in df.columns
        assert 'bleaching_tau' in df.columns
        assert len(df) == 1  # One row for VTA/GCaMP
