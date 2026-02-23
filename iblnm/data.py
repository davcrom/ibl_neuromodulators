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
from iblnm.validation import (
    MissingExtractedData, MissingRawData, InsufficientTrials, BlockStructureBug,
    IncompleteEventTimes, TrialsNotInPhotometryTime, BandInversion, EarlySamples,
    FewUniqueSamples, QCValidationError,
)


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
            try:
                _ = self.one.load_dataset(self.eid, '_iblrig_taskData.raw.jsonable')
            except ALFObjectNotFound:
                raise MissingRawData("_iblrig_taskData.raw.jsonable")
            raise MissingExtractedData("_ibl_trials.table.pqt")
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
            try:
                _ = self.one.load_dataset(self.eid, '_neurophotometrics_fpData.raw.pqt')
            except ALFObjectNotFound:
                raise MissingRawData("_neurophotometrics_fpData.raw.pqt")
            raise MissingExtractedData("photometry.signal.pqt")


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

    def save_h5(self, fpath=None, groups=None, band='GCaMP_preprocessed', mode='w'):
        """Save session data to HDF5.

        Parameters
        ----------
        fpath : Path or str, optional
            Output path. Defaults to SESSIONS_H5_DIR / {eid}.h5.
        groups : sequence of str, optional
            Which data groups to write. Any subset of:
            'signal', 'trials', 'responses', 'wheel'.
            None applies mode-based defaults:
              mode='w' → ('signal',)
              mode='a' → all available among ('trials', 'responses', 'wheel')
        mode : str
            HDF5 file open mode ('w' creates/truncates, 'a' appends).
        """
        import h5py
        from pathlib import Path
        if fpath is None:
            fpath = SESSIONS_H5_DIR / f'{self.eid}.h5'
        fpath = Path(fpath)
        fpath.parent.mkdir(parents=True, exist_ok=True)

        if groups is None:
            if mode == 'w':
                groups = ('signal',)
            else:
                groups = [g for g, available in (
                    ('trials',    hasattr(self, 'trials') and self.trials is not None),
                    ('responses', hasattr(self, 'responses') and self.responses is not None),
                    ('wheel',     hasattr(self, 'wheel_velocity') and self.wheel_velocity is not None),
                ) if available]

        with h5py.File(fpath, mode) as f:
            if 'signal' in groups:
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

            if 'trials' in groups and hasattr(self, 'trials') and self.trials is not None:
                if 'trials' in f:
                    del f['trials']
                cols = TRIAL_COLUMNS + ['signed_contrast', 'contrast']
                available = [c for c in cols if c in self.trials.columns]
                grp = f.create_group('trials')
                for col in available:
                    grp.create_dataset(col, data=self.trials[col].values)

            if 'responses' in groups and hasattr(self, 'responses') and self.responses is not None:
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

            if 'wheel' in groups and hasattr(self, 'wheel_velocity') and self.wheel_velocity is not None:
                if 'wheel' in f:
                    del f['wheel']
                grp = f.create_group('wheel')
                grp.create_dataset(
                    'velocity', data=self.wheel_velocity,
                    compression='gzip', compression_opts=4,
                )
                grp.attrs['fs'] = self.wheel_fs
                grp.attrs['t0_event'] = getattr(self, '_wheel_t0_event', 'stimOn_times')
                grp.attrs['t1_event'] = getattr(self, '_wheel_t1_event', 'feedback_times')

    def load_h5(self, fpath, groups=None):
        """Load session data from HDF5 file.

        Parameters
        ----------
        fpath : Path or str
            Path to the HDF5 file.
        groups : sequence of str, optional
            Which data groups to load. Any subset of:
            'signal', 'trials', 'responses'.
            None loads all groups present in the file.
        """
        import h5py
        import xarray as xr

        with h5py.File(fpath, 'r') as f:
            if (groups is None or 'signal' in groups) and 'preprocessed' in f:
                times = f['times'][:]
                preprocessed = {}
                for name in f['preprocessed']:
                    preprocessed[name] = pd.Series(
                        f[f'preprocessed/{name}'][:].astype(np.float64),
                        index=times
                    )
                self.photometry['GCaMP_preprocessed'] = pd.DataFrame(preprocessed)

            if (groups is None or 'trials' in groups) and 'trials' in f:
                self.trials = pd.DataFrame(
                    {col: f[f'trials/{col}'][:] for col in f['trials']}
                )

            if (groups is None or 'responses' in groups) and 'responses' in f:
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

    def run_raw_qc(self, raw_metrics=None):
        """Load raw photometry and compute session-level QC metrics.

        Updates self.qc with raw metric columns (n_band_inversions, n_early_samples).
        Call validate_qc() after this to check for band inversions and early samples.
        """
        if raw_metrics is None:
            raw_metrics = QC_RAW_METRICS
        raw_photometry = self._load_raw_photometry()
        raw_metric_values = {m: getattr(metrics, m)(raw_photometry) for m in raw_metrics}
        self.qc = pd.DataFrame([{'eid': self.eid, **raw_metric_values}])

    def run_sliding_qc(self, signal_band=None, sliding_metrics=None,
                       metrics_kwargs=None, sliding_kwargs=None,
                       brain_region=None, pipeline=None):
        """Run sliding-window QC on photometry signals.

        Updates self.qc with per-(band, brain_region) sliding metrics,
        merging in any raw metrics already stored from run_raw_qc().
        """
        if sliding_metrics is None:
            sliding_metrics = QC_SLIDING_METRICS
        if metrics_kwargs is None:
            metrics_kwargs = QC_METRICS_KWARGS
        if sliding_kwargs is None:
            sliding_kwargs = QC_SLIDING_KWARGS

        qc_tidy = qc_signals(
            self.photometry,
            metrics=[getattr(metrics, m) for m in sliding_metrics],
            metrics_kwargs=metrics_kwargs,
            signal_band=signal_band,
            brain_region=brain_region,
            pipeline=pipeline,
            sliding_kwargs=sliding_kwargs,
        )

        if 'window' in qc_tidy.columns:
            qc_tidy = qc_tidy.groupby(['band', 'brain_region', 'metric'], as_index=False)['value'].mean()

        df_qc = qc_tidy.pivot(index=['band', 'brain_region'], columns='metric', values='value').reset_index()
        df_qc.columns.name = None
        df_qc['eid'] = self.eid

        # Incorporate raw metrics from run_raw_qc() if present
        if not self.qc.empty:
            raw_cols = [c for c in self.qc.columns if c not in ('eid', 'brain_region', 'band')]
            for col in raw_cols:
                df_qc[col] = self.qc[col].iloc[0]

        self.qc = df_qc


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


    # =========================================================================
    # Response Convenience Methods
    # =========================================================================

    def subtract_baseline(self, responses=None, window=None):
        """Subtract per-trial pre-event baseline from response traces.

        Parameters
        ----------
        responses : xr.DataArray, optional
            dims (region, event, trial, time). Defaults to self.responses.
        window : tuple(float, float), optional
            Baseline window in seconds [t_start, t_end). Defaults to
            (RESPONSE_WINDOW[0], 0.0).

        Returns
        -------
        xr.DataArray
            Baseline-subtracted responses, same shape and coords as input.
        """
        if responses is None:
            responses = self.responses
        if window is None:
            window = (self.RESPONSE_WINDOW[0], 0.0)
        tpts = responses.coords['time'].values
        i0 = np.searchsorted(tpts, window[0])
        i1 = np.searchsorted(tpts, window[1])
        baseline = responses.isel(time=slice(i0, i1)).mean(dim='time', skipna=True)
        return responses - baseline

    def mask_subsequent_events(self, responses=None, event_order=None):
        """Mask response times that fall after the next event onset.

        For each consecutive pair (e0, e1) in event_order, per-trial times
        t > (trials[e1] - trials[e0]) are replaced with NaN in the e0
        response matrix. Trials where the next event time is NaN are not masked.

        Parameters
        ----------
        responses : xr.DataArray, optional
            dims (region, event, trial, time). Defaults to self.responses.
        event_order : list[str], optional
            Chronologically ordered event names. Defaults to RESPONSE_EVENTS.

        Returns
        -------
        xr.DataArray
            Masked responses, same shape and coords as input.
        """
        import xarray as xr
        if responses is None:
            responses = self.responses
        if event_order is None:
            event_order = list(RESPONSE_EVENTS)
        if not hasattr(self, 'trials') or self.trials is None:
            return responses
        events_present = list(responses.coords['event'].values)
        tpts = responses.coords['time'].values
        result = responses.copy()
        for i, event in enumerate(event_order[:-1]):
            next_event = event_order[i + 1]
            if event not in events_present:
                continue
            if event not in self.trials.columns or next_event not in self.trials.columns:
                continue
            dt = self.trials[next_event].values - self.trials[event].values
            nan_dt = np.isnan(dt)
            keep = (tpts[None, :] <= dt[:, None]) | nan_dt[:, None]
            keep_da = xr.DataArray(
                keep, dims=['trial', 'time'],
                coords={'trial': responses.coords['trial'],
                        'time':  responses.coords['time']},
            )
            result.loc[dict(event=event)] = result.sel(event=event).where(keep_da)
        return result

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
    # Wheel Methods
    # =========================================================================

    def load_wheel(self, fs=None):
        """Load wheel position and velocity from ONE.

        Interpolates to WHEEL_FS Hz and computes velocity via Butterworth filter.
        Stores result in self.wheel (DataFrame: times, position, velocity, acceleration).
        """
        if fs is None:
            fs = WHEEL_FS
        try:
            super().load_wheel(fs=fs)
        except ALFObjectNotFound:
            try:
                self.one.load_dataset(self.eid, '_iblrig_encoderPositions.raw.ssv')
            except ALFObjectNotFound:
                raise MissingRawData("_iblrig_encoderPositions.raw.ssv")
            raise MissingExtractedData("_ibl_wheel.position.npy")
        self.wheel_fs = fs

    def extract_wheel_velocity(self, t0_event='stimOn_times', t1_event='feedback_times'):
        """Extract per-trial wheel velocity from t0_event to t1_event.

        Returns a float32 (T, W) matrix where W is the longest trial in samples.
        Shorter trials and NaN-event trials are NaN-padded on the right.
        Stores result in self.wheel_velocity and returns it.
        """
        wheel_signal = pd.Series(
            self.wheel['velocity'].values,
            index=self.wheel['times'].values,
        )
        t0_times = self.trials[t0_event].values
        t1_times = self.trials[t1_event].values
        velocity_matrix, _ = get_responses(wheel_signal, events=t0_times, t0=0.0, t1=t1_times)
        self.wheel_velocity = velocity_matrix.astype(np.float32)
        self._wheel_t0_event = t0_event
        self._wheel_t1_event = t1_event
        return self.wheel_velocity

