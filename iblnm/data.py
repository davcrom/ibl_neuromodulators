import numpy as np
import pandas as pd
from datetime import datetime

from one.api import ONE
from brainbox.io.one import PhotometrySessionLoader
from iblphotometry.fpio import from_neurophotometrics_df_to_photometry_df
from iblphotometry import metrics
from iblphotometry.qc import qc_signals

from one.alf.exceptions import ALFObjectNotFound

from iblnm.config import *
from iblnm.analysis import get_responses
from iblnm import task
from iblnm.task import _get_signed_contrast
from iblnm.util import make_log_entry


class MissingExtractedData(Exception):
    """Extracted dataset not found on Alyx (raw data exists)."""


class MissingRawData(Exception):
    """Raw dataset not found on Alyx."""


class InsufficientTrials(Exception):
    """Session has too few trials for analysis."""


class BlockStructureBug(Exception):
    """Biased/ephys session has rapidly flipping blocks."""


class IncompleteEventTimes(Exception):
    """Event times below completeness threshold."""
    def __init__(self, missing_events):
        self.missing_events = missing_events
        super().__init__(f"Incomplete events: {', '.join(missing_events)}")


class TrialsNotInPhotometryTime(Exception):
    """Trial times fall outside photometry recording window."""


class BandInversion(Exception):
    """Photometry signal has band inversions."""


class EarlySamples(Exception):
    """Photometry signal has early samples."""


class FewUniqueSamples(Exception):
    """One or more photometry channels have too few unique samples."""


class QCValidationError(Exception):
    """One or more raw QC checks failed (band inversions, early samples)."""


class PhotometrySession(PhotometrySessionLoader):
    """Data class for an IBL photometry session."""

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

        self.number = int(session_series['number'])
        self.lab = session_series.get('lab')
        self.projects = session_series.get('projects', [])
        self.url = session_series.get('url')
        self.session_n = session_series.get('session_n')
        self.task_protocol = session_series['task_protocol']
        self.session_type = session_series['session_type']
        self.NM = session_series.get('NM')
        self.datasets = session_series.get('datasets', [])

        super().__init__(*args, eid=self.eid, **kwargs)
        if not isinstance(self.photometry, dict):
            self.photometry = {}
        self.qc = pd.DataFrame()
        if load_data:
            self.load_trials()
            self.load_photometry()


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
        except ALFObjectNotFound:
            has_raw = any(d in self.datasets for d in [
                'raw_behavior_data/_iblrig_taskData.raw.jsonable',
                'raw_task_data_00/_iblrig_taskData.raw.jsonable',
            ])
            if has_raw:
                raise MissingExtractedData("Trials could not be downloaded")
            raise MissingRawData("No raw task data")
        self.trials['signed_contrast'] = _get_signed_contrast(self.trials)
        self.trials['contrast'] = np.abs(self.trials['signed_contrast'])

    def load_photometry(
        self,
        pre: int = -5,
        post: int = 5,
        ):
        try:
            super().load_photometry(
                restrict_to_session=True,
                pre=pre,
                post=post
            )
        except ALFObjectNotFound:
            if 'raw_photometry_data/_neurophotometrics_fpData.raw.pqt' in self.datasets:
                raise MissingExtractedData("Photometry could not be downloaded")
            raise MissingRawData("No raw photometry signal found")


    def validate_n_trials(self):
        """Raises InsufficientTrials if n_trials < MIN_NTRIALS."""
        if len(self.trials) < MIN_NTRIALS:
            raise InsufficientTrials(
                f"n_trials={len(self.trials)} < MIN_NTRIALS={MIN_NTRIALS}"
            )

    def validate_block_structure(self):
        """Raises BlockStructureBug if biased/ephys session has rapidly flipping blocks."""
        if self.session_type not in ('biased', 'ephys'):
            return
        block_info = task.validate_block_structure(self.trials)
        if block_info['flagged']:
            raise BlockStructureBug(
                f"Min block length: {block_info['min_block_length']}, "
                f"n_blocks: {block_info['n_blocks']}"
            )

    def validate_event_completeness(self):
        """Raises IncompleteEventTimes with all missing events if any are below threshold."""
        missing = [
            event for event in RESPONSE_EVENTS
            if (event not in self.trials.columns
                or self.trials[event].notna().mean() < EVENT_COMPLETENESS_THRESHOLD)
        ]
        if missing:
            raise IncompleteEventTimes(missing)

    def validate_few_unique_samples(self):
        """Raises FewUniqueSamples listing channels below threshold."""
        if self.qc.empty or 'n_unique_samples' not in self.qc.columns:
            return
        rows = self.qc[self.qc['n_unique_samples'] < N_UNIQUE_SAMPLES_THRESHOLD]
        if not rows.empty:
            channels = ', '.join(
                f"{r['brain_region']}/{r['band']}={r['n_unique_samples']:.3f}"
                for _, r in rows.iterrows()
            )
            raise FewUniqueSamples(f"Few unique samples: {channels}")

    def validate_trials_in_photometry_time(self, band=None):
        """Raises TrialsNotInPhotometryTime if trial times fall outside photometry window."""
        if band is None:
            band = 'GCaMP_preprocessed' if 'GCaMP_preprocessed' in self.photometry else 'GCaMP'
        phot_times = self.photometry[band].index
        trial_start = self.trials['stimOn_times'].min()
        trial_stop = self.trials['feedback_times'].max()
        if not (trial_start >= phot_times.min() and trial_stop <= phot_times.max()):
            raise TrialsNotInPhotometryTime(
                f"Trials [{trial_start:.1f}, {trial_stop:.1f}] outside "
                f"photometry [{phot_times.min():.1f}, {phot_times.max():.1f}]"
            )

    def validate_qc(self):
        """Raises QCValidationError listing all raw QC issues found."""
        if self.qc.empty:
            return
        issues = []
        if 'n_band_inversions' in self.qc.columns and (self.qc['n_band_inversions'] > 0).any():
            issues.append("band inversions detected")
        if 'n_early_samples' in self.qc.columns and (self.qc['n_early_samples'] > 0).any():
            issues.append("early samples detected")
        if issues:
            raise QCValidationError('; '.join(issues))

    def _load_raw_photometry(self):
        raw_photometry = self.one.load_dataset(
            self.eid,
            'raw_photometry_data/_neurophotometrics_fpData.raw.pqt'
            )
        # ~ timestamp_col = 'SystemTimestamp' if 'Timestamp' not in raw_photometry.columns else 'Timestamp'
        # ~ raw_photometry = raw_photometry.set_index(timestamp_col)
        return from_neurophotometrics_df_to_photometry_df(raw_photometry).set_index('times')

    def extract_responses(self, events=None, band='GCaMP_preprocessed',
                          window=None):
        """Extract peri-event response matrices as xarray DataArray.

        Returns
        -------
        xr.DataArray
            dims: (region, event, trial, time)
        """
        import xarray as xr

        if events is None:
            events = RESPONSE_EVENTS
        if window is None:
            window = self.RESPONSE_WINDOW
        regions = list(self.photometry[band].columns)
        n_trials = len(self.trials)

        # Collect response matrices and determine tpts from first call
        data = []
        tpts = None
        for region in regions:
            signal = self.photometry[band][region]
            region_data = []
            for event in events:
                event_times = self.trials[event].values
                resp, t = get_responses(signal, event_times,
                                        t0=window[0], t1=window[1])
                if tpts is None:
                    tpts = t
                region_data.append(resp)
            data.append(region_data)

        # data shape: (n_regions, n_events, n_trials, n_times)
        self.responses = xr.DataArray(
            np.array(data),
            dims=['region', 'event', 'trial', 'time'],
            coords={
                'region': regions,
                'event': events,
                'trial': np.arange(n_trials),
                'time': tpts,
            },
        )
        return self.responses

    def save_h5(self, fpath=None, band='GCaMP_preprocessed', mode='w'):
        """Save session data to HDF5.

        mode='w': Create file with preprocessed signal + session attrs.
        mode='a': Append trials + responses to existing file.
        """
        import h5py
        from pathlib import Path
        if fpath is None:
            fpath = SESSIONS_H5_DIR / f'{self.eid}.h5'
        fpath = Path(fpath)
        fpath.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(fpath, mode) as f:
            if mode == 'w':
                f.attrs['eid'] = self.eid
                f.attrs['subject'] = self.subject
                f.attrs['session_type'] = self.session_type
                f.attrs['date'] = self.date
                f.attrs['fs'] = TARGET_FS
                f.attrs['response_window'] = self.RESPONSE_WINDOW

                preprocessed = self.photometry[band]
                f.create_dataset('times', data=preprocessed.index.values)
                grp = f.create_group('preprocessed')
                for col in preprocessed.columns:
                    grp.create_dataset(col, data=preprocessed[col].values,
                                       compression='gzip', compression_opts=4)

            elif mode == 'a':
                if hasattr(self, 'trials') and self.trials is not None:
                    if 'trials' in f:
                        del f['trials']
                    cols = TRIAL_COLUMNS + ['signed_contrast', 'contrast']
                    available = [c for c in cols if c in self.trials.columns]
                    grp = f.create_group('trials')
                    for col in available:
                        grp.create_dataset(col, data=self.trials[col].values)

                if hasattr(self, 'responses') and self.responses is not None:
                    if 'responses' in f:
                        del f['responses']
                    grp = f.create_group('responses')
                    tpts = self.responses.coords['time'].values
                    grp.create_dataset('time', data=tpts)
                    for region in self.responses.coords['region'].values:
                        region_grp = grp.create_group(region)
                        for event in self.responses.coords['event'].values:
                            resp = self.responses.sel(region=region, event=event).values
                            ds = region_grp.create_dataset(
                                event, data=resp.astype(np.float64),
                                compression='gzip', compression_opts=4
                            )
                            ds.attrs['window_t0'] = tpts[0]
                            ds.attrs['window_t1'] = tpts[-1]

    def load_h5(self, fpath):
        """Load preprocessed signal and responses from HDF5 file."""
        import h5py
        import xarray as xr

        with h5py.File(fpath, 'r') as f:
            times = f['times'][:]
            preprocessed = {}
            for name in f['preprocessed']:
                preprocessed[name] = pd.Series(
                    f[f'preprocessed/{name}'][:].astype(np.float64),
                    index=times
                )
            self.photometry['GCaMP_preprocessed'] = pd.DataFrame(preprocessed)

            if 'responses' in f:
                resp_grp = f['responses']
                tpts = resp_grp['time'][:]
                regions = [k for k in resp_grp.keys() if k != 'time']
                events = list(resp_grp[regions[0]].keys())
                n_trials = resp_grp[regions[0]][events[0]].shape[0]

                data = np.empty((len(regions), len(events), n_trials, len(tpts)),
                                dtype=np.float64)
                for i, region in enumerate(regions):
                    for j, event in enumerate(events):
                        data[i, j] = resp_grp[region][event][:].astype(np.float64)

                self.responses = xr.DataArray(
                    data,
                    dims=['region', 'event', 'trial', 'time'],
                    coords={
                        'region': regions,
                        'event': events,
                        'trial': np.arange(n_trials),
                        'time': tpts,
                    },
                )

    def _append_qc(self, brain_region: str, band: str, metrics: dict) -> None:
        """Append or update a QC row in the DataFrame."""
        mask = (self.qc['brain_region'] == brain_region) & (self.qc['band'] == band) if not self.qc.empty else pd.Series(dtype=bool)
        if mask.any():
            idx = mask.idxmax()
            for k, v in metrics.items():
                self.qc.loc[idx, k] = v
        else:
            row = {'brain_region': brain_region, 'band': band, 'eid': self.eid, **metrics}
            self.qc = pd.concat([self.qc, pd.DataFrame([row])], ignore_index=True)

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

        # Add raw metrics and eid to the DataFrame
        for metric_name, value in raw_metric_values.items():
            df_qc[metric_name] = value
        df_qc['eid'] = self.eid

        # Merge with existing QC (from preprocess)
        if self.qc.empty:
            self.qc = df_qc
        else:
            self.qc = self.qc.merge(
                df_qc, on=['brain_region', 'band'], how='outer', suffixes=('', '_new')
            )
            # Update with new values where they exist
            for col in df_qc.columns:
                if col + '_new' in self.qc.columns:
                    self.qc[col] = self.qc[col + '_new'].combine_first(self.qc[col])
                    self.qc.drop(columns=[col + '_new'], inplace=True)
            if 'eid' not in self.qc.columns:
                self.qc['eid'] = self.eid

        return self.qc

    # =========================================================================
    # Task Performance Methods
    # =========================================================================

    def basic_performance(self) -> dict:
        """Compute session-level performance metrics (all session types)."""
        result = {}
        result['fraction_correct'] = task.compute_fraction_correct(self.trials)
        result['fraction_correct_easy'] = task.compute_fraction_correct(
            self.trials[self.trials['contrast'] >= 0.5]
        )
        result['nogo_fraction'] = task.compute_nogo_fraction(self.trials)
        fit_50 = task.fit_psychometric(self.trials, probability_left=0.5)
        for param, value in fit_50.items():
            result[f'psych_50_{param}'] = value
        return result

    def block_performance(self) -> dict:
        """Compute per-block psychometrics and bias shift (biased/ephys only)."""
        if self.session_type not in ('biased', 'ephys'):
            return {}
        result = {}
        fits = task.fit_psychometric_by_block(self.trials)
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
        regression_method: str = 'mse',
    ):
        """Run preprocessing pipeline and store result as new band.

        Pipeline steps (bleach correct → isosbestic correct → zscore) are defined
        in config.PREPROCESSING_PIPELINES. Resampling to TARGET_FS is applied after
        the pipeline as a separate step.
        """
        from iblphotometry.pipelines import run_pipeline
        from iblnm.analysis import compute_bleaching_tau, compute_iso_correlation, resample_signal

        if pipeline is None:
            pipeline = PREPROCESSING_PIPELINES['isosbestic_correction']

        if targets is None:
            targets = list(self.photometry[signal_band].columns)

        needs_reference = any('reference' in step.get('inputs', ()) for step in pipeline)

        if needs_reference and reference_band is None:
            raise ValueError("Pipeline requires reference_band")

        preprocessed = {}

        for brain_region in targets:
            signal = self.photometry[signal_band][brain_region]
            qc_metrics = {'bleaching_tau': compute_bleaching_tau(signal)}

            if needs_reference:
                reference = self.photometry[reference_band][brain_region]
                res = run_pipeline(pipeline, signal=signal, reference=reference, full_output=True)
                result = res['result']
                # iso_correlation computed on bleach-corrected signals before isosbestic step
                signal_bc = res.get('signal_bleach_corrected', signal)
                reference_bc = res.get('reference_bleach_corrected', reference)
                qc_metrics['iso_correlation'] = compute_iso_correlation(
                    signal_bc, reference_bc, regression_method=regression_method
                )
            else:
                result = run_pipeline(pipeline, signal=signal)

            result = resample_signal(result, target_fs=TARGET_FS)
            preprocessed[brain_region] = result
            self._append_qc(brain_region, signal_band, qc_metrics)

        self.photometry[output_band] = pd.DataFrame(preprocessed)
        return self.photometry[output_band]

