import numpy as np
import pandas as pd
import pynapple as nap
from datetime import datetime
from typing import Optional, List

from matplotlib import pyplot as plt
from matplotlib import gridspec

from one.api import ONE
from brainbox.io.one import PhotometrySessionLoader
from iblphotometry import fpio, metrics
from iblphotometry.analysis import psth_nap

from iblnm.config import *
from iblnm.util import protocol2type, _insert_event_times
from iblnm.resp import get_responses


class PhotometrySession(PhotometrySessionLoader):
    """
    Data class for an IBL photometry session.
    """

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
            signal_ = getattr(self, channel)
            signal = nap.Tsd(
                t=signal_.index.to_numpy(), d=signal_[target].values
                )
            responses = psth_nap(
                signal, self.trials, align_on='_'.join([event, 'times']), **kwargs
                )
            if event == 'feedback':
                responses['correct'] = responses.pop(1.0)
                responses['incorrect'] = responses.pop(-1.0)
            self.responses[event][target][channel] = responses



if __name__ == "__main__":

    one = ONE()

    ## Replace this with a loop
    # Pick a subject
    subject = 'ZFM-08871'
    sessions = one.alyx.rest(
        'sessions',
        'list',
        project='ibl_fibrephotometry',
        subject=subject,
        datasets='photometry.signal.pqt'
    )
    df_sessions = pd.DataFrame(sessions).rename(columns={'id':'eid'})
    print(df_sessions.head())
    # Pick a session
    session_i = 6
    session = PhotometrySession(df_sessions.iloc[session_i], one=one)



    qc = {target: {} for target in session.targets['gcamp']}
    for target in session.targets['gcamp']:
        # Sampling regularity
        # Need raw df to correct this, but metric will tell if error is present
        qc[target]['qc_n_early_samples'] = metrics.n_early_samples(session.gcamp[target])
        # Number of unique samples in the signal
        qc[target]['qc_n_unique_samples'] = metrics.n_unique_samples(session.gcamp[target])
        # Outliers
        qc[target]['qc_n_edges'] = metrics.n_edges(session.gcamp[target])

        # Sliding metrics
        metrics_to_apply = [
            metrics.median_absolute_deviance,
            metrics.percentile_distance,
            metrics.percentile_asymmetry,
            metrics.n_outliers,
            metrics.n_expmax_violations,
            metrics.expmax_violation,
            # metrics.bleaching_tau,  # cant be applied sliding, needs to be series
            metrics.spectral_entropy,
            metrics.ar_score
        ]
        metrics_params = {metric:{} for metric in metrics_to_apply}
        sliding_kwargs = {'w_len': 120}
        qc[target].update(
            metrics.qc_series(session.gcamp['LC'], metrics_params, sliding_kwargs)
        )

        ## TODO: response based qc
        # rename trials kwargs in metrics module to events


        # Signal amplitude
        # qc[target]['qc_deviance'] = metrics.median_absolute_deviance(session.gcamp[target])  # median amplitude
        # qc[target]['qc_percentile_distance'] = metrics.percentile_distance(
            # session.gcamp[target],
            # pc=(50, 95)
            # )  # amplitude of positive transients
        # qc[target]['qc_percentile_asymmetry'] = metrics.percentile_asymmetry(
            # session.gcamp[target],
            # pc_comp=95
            # )  # amplitude of positive versus negative transients
