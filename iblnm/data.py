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
from iblnm import task
from iblnm.task import _get_signed_contrast

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

    def __init__(self, session_series: pd.Series, *args, load_data=False, **kwargs):
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
        self.session_n = session_series['session_n']
        self.task_protocol = session_series['task_protocol']
        self.session_type = session_series['session_type']

        super().__init__(*args, eid=self.eid, **kwargs)
        self.qc = {}
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


    def load_trials(self):
        try:
            super().load_trials()
            self.has_trials = True
        except Exception:
            self.has_trials = False

        if self.has_trials:
            self.trials['signed_contrast'] = _get_signed_contrast(self.trials)
            self.trials['contrast'] = np.abs(self.trials['signed_contrast'])


    def load_photometry(self):
        try:
            super().load_photometry()
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


    def load_session_data(self):
        """
        Load trials and photometry data with error handling.

        Sets has_trials, has_photometry, and trials_in_photometry_time flags.
        """
        # Load trials
        self.load_trials()

        # Load photometry
        self.load_photometry()


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


    def _update_qc(self, brain_region: str, band: str, metrics: dict) -> None:
        """Update QC metrics in nested dict structure."""
        if brain_region not in self.qc:
            self.qc[brain_region] = {}
        if band not in self.qc[brain_region]:
            self.qc[brain_region][band] = {}
        self.qc[brain_region][band].update(metrics)

    def qc_to_dataframe(self) -> pd.DataFrame:
        """Convert nested QC dict to DataFrame for aggregation."""
        rows = []
        for brain_region, bands in self.qc.items():
            for band, metrics in bands.items():
                row = {'brain_region': brain_region, 'band': band, **metrics}
                rows.append(row)
        df = pd.DataFrame(rows)
        if not df.empty:
            df['eid'] = self.eid
        return df

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
        raw_metric_values = {}
        for metric_name in raw_metrics:
            metric_func = getattr(metrics, metric_name)
            raw_metric_values[metric_name] = metric_func(raw_photometry)

        # Store in nested dict structure
        metric_cols = [c for c in df_qc.columns if c not in ['band', 'brain_region']]
        for _, row in df_qc.iterrows():
            qc_metrics = {col: row[col] for col in metric_cols}
            qc_metrics.update(raw_metric_values)
            self._update_qc(row['brain_region'], row['band'], qc_metrics)

        return self.qc_to_dataframe()

    # =========================================================================
    # Task Performance Methods
    # =========================================================================

    def task_performance(self):
        result = {}

        result['fraction_correct'] = self.fraction_correct()
        result['fraction_correct_easy'] = self.fraction_correct_easy()
        result['nogo_fraction'] = self.nogo_fraction()

        block_info = task.validate_block_structure(self.trials)
        result['block_structure_valid'] = block_info['valid']
        result['min_block_length'] = block_info['min_block_length']
        result['n_blocks'] = block_info['n_blocks']

        # Training sessions: fit 50-50 only (no bias blocks)
        if self.session_type == 'training':
            fit_50 = self.fit_psychometric(probability_left=0.5)
            for param, value in fit_50.items():
                result[f'psych_50_{param}'] = value

        # Biased/ephys: fit by block if structure is valid
        elif block_info['valid'] and self.session_type in ['biased', 'ephys']:
            fits = self.fit_psychometric_by_block()
            for block_name, fit in fits.items():
                for param, value in fit.items():
                    result[f'psych_{block_name}_{param}'] = value

            if '20' in fits and '80' in fits:
                result['bias_shift'] = task.compute_bias_shift(fits['20'], fits['80'])

        return result

    def fraction_correct(self, exclude_nogo=True):
        return task.compute_fraction_correct(self.trials, exclude_nogo=exclude_nogo)

    def fraction_correct_by_contrast(self, exclude_nogo=True):
        df = self.trials if not exclude_nogo else self.trials[self.trials['choice'] != 0]
        return df.groupby('contrast')['feedbackType'].apply(lambda x: (x == 1).mean())

    def fraction_correct_easy(self, exclude_nogo=True):
        trials = self.trials[self.trials['contrast'] >= 0.5]
        return task.compute_fraction_correct(trials, exclude_nogo=exclude_nogo)

    def nogo_fraction(self):
        return task.compute_nogo_fraction(self.trials)

    def fit_psychometric(self, probability_left=None):
        return task.fit_psychometric(self.trials, probability_left=probability_left)

    def fit_psychometric_by_block(self):
        return task.fit_psychometric_by_block(self.trials)

    # =========================================================================
    # Preprocessing Methods
    # =========================================================================

    def preprocess(
        self,
        pipeline=None,
        signal_band='GCaMP',
        reference_band='Isosbestic',
        targets=None,
        output_band='GCaMP_preprocessed',
    ):
        """Run preprocessing pipeline and store result as new band."""
        from iblphotometry.pipelines import run_pipeline, isosbestic_correction_pipeline
        from iblphotometry.processing import Regression, LinearModel, ExponDecay

        if not self.has_photometry:
            raise ValueError("Photometry data not loaded")

        if pipeline is None:
            pipeline = isosbestic_correction_pipeline

        if targets is None:
            targets = list(self.photometry[signal_band].columns)

        # Check if pipeline needs reference
        needs_reference = any('reference' in step.get('inputs', ()) for step in pipeline)

        if needs_reference and reference_band is None:
            raise ValueError("Pipeline requires reference_band")

        # Extract regression method from pipeline for iso_correlation
        regression_method = 'mse'
        for step in pipeline:
            if 'regression_method' in step.get('parameters', {}):
                regression_method = step['parameters']['regression_method']
                break

        preprocessed = {}

        for brain_region in targets:
            signal = self.photometry[signal_band][brain_region]

            # Compute bleaching_tau using correct argument order
            # Note: iblphotometry.metrics.bleaching_tau has a bug (calls fit(y, t) instead of fit(t, y))
            reg = Regression(model=ExponDecay())
            reg.fit(signal.index.values, signal.values)  # fit(t, y)
            qc_metrics = {'bleaching_tau': reg.popt[1]}

            if needs_reference:
                reference = self.photometry[reference_band][brain_region]

                # Run pipeline with full_output to get intermediate results
                res = run_pipeline(pipeline, signal=signal, reference=reference, full_output=True)
                result = res['result']

                # Compute iso_correlation on bleach-corrected signals (from pipeline)
                signal_bc = res.get('signal_bleach_corrected', signal)
                reference_bc = res.get('reference_bleach_corrected', reference)

                reg = Regression(model=LinearModel(), method=regression_method)
                reg.fit(reference_bc.values, signal_bc.values)
                # Use model equation directly (predict() sorts x, breaking alignment)
                predicted = reg.model.eq(reference_bc.values, *reg.popt)
                ss_res = np.sum((signal_bc.values - predicted) ** 2)
                ss_tot = np.sum((signal_bc.values - np.mean(signal_bc.values)) ** 2)
                qc_metrics['iso_correlation'] = 1 - (ss_res / ss_tot)
            else:
                result = run_pipeline(pipeline, signal=signal)

            preprocessed[brain_region] = result
            self._update_qc(brain_region, signal_band, qc_metrics)

        self.photometry[output_band] = pd.DataFrame(preprocessed)
        return self.photometry[output_band]

