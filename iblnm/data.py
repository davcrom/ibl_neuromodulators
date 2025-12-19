import numpy as np
import pandas as pd
import pynapple as nap
from datetime import datetime

from one.api import ONE
from brainbox.io.one import PhotometrySessionLoader
from iblphotometry import metrics
from iblphotometry.qc import qc_signals

from iblnm.config import *
from iblnm.analysis import get_responses, get_response_tpts

class PhotometrySession(PhotometrySessionLoader):
    """
    Data class for an IBL photometry session.
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
        super().load_session_data(**kwargs)
        if isinstance(self.photometry, dict):
            for k, v in self.photometry.items():
                setattr(self, k.lower(), v)
            self.channels = {k.lower() for k in self.photometry.keys()}
            self.targets = {
                k.lower(): list(v.columns) for k, v in self.photometry.items()
                }
            ## TODO: if hasattr(self, 'genotype'): add cell type to targets


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


    def run_qc(self, signal_band='GCaMP', raw_metrics=None, sliding_metrics=None,
               metrics_kwargs=None, sliding_kwargs=None, brain_region=None, pipeline=None):
        """
        Run quality control on the photometry session.

        Parameters
        ----------
        signal_band : str
            Signal band to run QC on (default: 'GCaMP')
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
            QC results dataframe
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

        # Convert metric names to functions
        raw_metric_funcs = [getattr(metrics, m) for m in raw_metrics]
        sliding_metric_funcs = [getattr(metrics, m) for m in sliding_metrics]

        # Run raw metrics QC
        qc_raw = qc_signals(
            self.photometry,
            metrics=raw_metric_funcs,
            metrics_kwargs=metrics_kwargs,
            signal_band=signal_band,
            brain_region=brain_region,
            pipeline=pipeline,
        )
        qc_raw['eid'] = self.eid

        # Run sliding metrics QC
        qc_sliding = qc_signals(
            self.photometry,
            metrics=sliding_metric_funcs,
            metrics_kwargs=metrics_kwargs,
            signal_band=signal_band,
            brain_region=brain_region,
            pipeline=pipeline,
            sliding_kwargs=sliding_kwargs,
        )
        qc_sliding['eid'] = self.eid

        # Combine results
        self.qc = pd.concat([qc_raw, qc_sliding], ignore_index=True)

        return self.qc
