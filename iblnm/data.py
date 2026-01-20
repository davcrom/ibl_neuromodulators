import numpy as np
import pandas as pd
import pynapple as nap
from datetime import datetime

from one.api import ONE
from brainbox.io.one import PhotometrySessionLoader
from iblphotometry.fpio import from_neurophotometrics_df_to_photometry_df
from iblphotometry import metrics
from iblphotometry.qc import qc_signals

from iblnm.config import *
from iblnm.analysis import get_responses, get_response_tpts

class PhotometrySession(PhotometrySessionLoader):
    """
    Data class for an IBL photometry session.

    Attributes
    ----------
    has_trials : bool
        Whether trials data loaded successfully
    has_photometry : bool
        Whether photometry data loaded successfully
    trials_in_photometry_time : bool
        Whether all trial times fall within photometry recording time
    """

    RESPONSE_WINDOW = RESPONSE_WINDOW

    def __init__(self, session_series: pd.Series, *args, load_data=True, **kwargs):
        """
        Initialize a PhotometrySession from a pandas Series.

        Parameters:
            session_series (pd.Series): A pandas Series containing session metadata
        """
        self.eid = session_series['eid']
        self.subject = session_series['subject']

        # Parse start_time as datetime if it's a string
        start_time = session_series['start_time']
        if isinstance(start_time, str):
            self.start_time = datetime.fromisoformat(start_time)
        else:
            self.start_time = start_time

        # TODO: add session type
        self.number = int(session_series['number'])
        self.lab = session_series['lab']
        self.projects = session_series['projects']
        self.url = session_series['url']
        self.task_protocol = session_series['task_protocol']

        # Data availability flags
        self.has_trials = False
        self.has_photometry = False
        self.trials_in_photometry_time = False

        super().__init__(*args, eid=self.eid, **kwargs)
        if load_data:
            self.load_session_data()


    def __str__(self) -> str:
        """Return a human-readable string representation."""
        return (f"Session #{self.eid}\n"
                f"Projects: {self.projects}\n"
                f"Lab: {self.lab}\n"
                f"Subject: {self.subject}\n"
                f"Protocol: {self.task_protocol}"
                f"Start: {self.start_time}\n"
                f"URL: {self.url}")


    @property
    def date(self) -> str:
        """Return the session date in YYYY-MM-DD format."""
        return self.start_time.strftime('%Y-%m-%d')


    def to_dict(self) -> dict:
        """Convert the session to a dictionary."""
        return {
            'eid': self.eid,
            'subject': self.subject,
            'start_time': self.start_time.isoformat(),
            'number': self.number,
            'lab': self.lab,
            'projects': self.projects,
            'url': self.url,
            'task_protocol': self.task_protocol
        }


    def to_series(self) -> pd.Series:
        return pd.Series(self.to_dict())


    def load_session_data(self, **kwargs):
        """
        Load trials and photometry data with error handling.

        Sets has_trials, has_photometry, and trials_in_photometry_time flags.
        """
        # Load trials
        try:
            self.load_trials()
            self.has_trials = True
        except Exception:
            self.has_trials = False

        # Load photometry
        try:
            self.load_photometry(restrict_to_session=False)
            self.has_photometry = True
        except Exception:
            self.has_photometry = False

        # Set up convenience attributes if photometry loaded
        if self.has_photometry and isinstance(self.photometry, dict):
            for k, v in self.photometry.items():
                setattr(self, k.lower(), v)
            self.channels = {k.lower() for k in self.photometry.keys()}
            self.targets = {
                k.lower(): list(v.columns) for k, v in self.photometry.items()
            }

        # Check if trials are within photometry time
        self.trials_in_photometry_time = self._check_trials_in_photometry_time()

    def _check_trials_in_photometry_time(self):
        """Check if all trial times fall within the photometry recording time."""
        if not self.has_trials or not self.has_photometry:
            return False

        # Get photometry time range from any band
        phot_times = self.photometry['GCaMP'].index
        t_start = phot_times.min()
        t_stop = phot_times.max()

        # Get trial time range
        trial_start = self.trials['intervals_0'].min()
        trial_stop = self.trials['intervals_1'].max()

        return (trial_start >= t_start) and (trial_stop <= t_stop)

    def get_data_flags(self):
        """Return data availability flags as a dict for updating df_sessions."""
        return {
            'has_trials': self.has_trials,
            'has_photometry': self.has_photometry,
            'trials_in_photometry_time': self.trials_in_photometry_time,
        }

    def _load_raw_photometry(self):
        raw_photometry = self.one.load_dataset(
            self.eid,
            'raw_photometry_data/_neurophotometrics_fpData.raw.pqt'
            )
        # ~ timestamp_col = 'SystemTimestamp' if 'Timestamp' not in raw_photometry.columns else 'Timestamp'
        # ~ raw_photometry = raw_photometry.set_index(timestamp_col)
        return from_neurophotometrics_df_to_photometry_df(raw_photometry).set_index('times')

    def get_responses(self, channel, event, targets=None, **kwargs):
        if event.endswith('_times'):
            event = event.rstrip('_times')
        if not hasattr(self, 'responses'):
            self.responses = {}
        self.responses[event] = {}
        for target in (targets or self.targets[channel]):
            self.responses[event][target] = {}
            responses = get_responses(
                getattr(self, channel)[target],
                self.trials[event + '_times'].values,
                window=self.RESPONSE_WINDOW
                )
            self.responses[event][target][channel] = responses


    def get_response_tpts(self, channel):
        tpts = get_response_tpts(
            getattr(self, channel),
            window=self.RESPONSE_WINDOW
        )
        return tpts


    def run_qc(self, signal_band=None, raw_metrics=None, sliding_metrics=None,
               metrics_kwargs=None, sliding_kwargs=None, brain_region=None, pipeline=None):
        """
        Run quality control on the photometry session.

        Parameters
        ----------
        signal_band : str or list of str, optional
            Signal band(s) to run QC on. If None, runs on all bands.
        raw_metrics : list of str, optional
            List of raw metric names to compute. If None, uses default from config.
        sliding_metrics : list of str, optional
            List of sliding window metric names to compute. If None, uses default from config.
        metrics_kwargs : dict, optional
            Custom kwargs for specific metrics. If None, uses default from config.
        sliding_kwargs : dict, optional
            Sliding window parameters. If None, uses default from config.
        brain_region : str or list of str, optional
            Restrict QC to specific brain region(s)
        pipeline : list of dict, optional
            Processing pipeline to apply before QC

        Returns
        -------
        pd.DataFrame
            QC results with one row per (brain_region, band) combination.
            Columns: eid, band, brain_region, + one column per metric.
        """
        # Get metric names from config if not provided
        if raw_metrics is None:
            raw_metrics = QC_RAW_METRICS
        if sliding_metrics is None:
            sliding_metrics = QC_SLIDING_METRICS
        if metrics_kwargs is None:
            metrics_kwargs = QC_METRICS_KWARGS
        if sliding_kwargs is None:
            sliding_kwargs = QC_SLIDING_KWARGS

        # Run sliding metrics QC - returns tidy data
        qc_tidy = qc_signals(
            self.photometry,
            metrics=[getattr(metrics, m) for m in sliding_metrics],
            metrics_kwargs=metrics_kwargs,
            signal_band=signal_band,
            brain_region=brain_region,
            pipeline=pipeline,
            sliding_kwargs=sliding_kwargs,
        )

        # Average across windows if sliding was used, then pivot to wide format
        # One row per (band, brain_region), one column per metric
        if 'window' in qc_tidy.columns:
            qc_tidy = qc_tidy.groupby(['band', 'brain_region', 'metric'], as_index=False)['value'].mean()

        df_qc = qc_tidy.pivot(index=['band', 'brain_region'], columns='metric', values='value').reset_index()
        df_qc.columns.name = None  # Remove the 'metric' name from columns

        # Add raw metrics (computed on raw photometry, same value for all rows)
        raw_photometry = self._load_raw_photometry()
        for metric_name in raw_metrics:
            metric_func = getattr(metrics, metric_name)
            df_qc[metric_name] = metric_func(raw_photometry)

        # Add session identifier
        df_qc['eid'] = self.eid

        self.qc_results = df_qc
        return df_qc
