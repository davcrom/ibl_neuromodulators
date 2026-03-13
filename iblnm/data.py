import numpy as np
import pandas as pd
from datetime import datetime

from brainbox.io.one import PhotometrySessionLoader
from iblphotometry.fpio import from_neurophotometrics_df_to_photometry_df
from iblphotometry import metrics
from iblphotometry.qc import qc_signals

from one.alf.exceptions import ALFObjectNotFound

from iblnm.config import (
    EVENT_COMPLETENESS_THRESHOLD, MIN_NTRIALS, N_UNIQUE_SAMPLES_THRESHOLD,
    PREPROCESSING_PIPELINES, QC_METRICS_KWARGS, QC_RAW_METRICS,
    QC_SLIDING_KWARGS, QC_SLIDING_METRICS, RESPONSE_EVENTS, RESPONSE_WINDOW,
    RESPONSE_WINDOWS, SESSIONS_H5_DIR, TARGET_FS, TRIAL_COLUMNS, WHEEL_FS,
)
from iblnm.analysis import get_responses, compute_response_magnitude
from iblnm import task
from iblnm.task import compute_trial_contrasts
from iblnm.validation import (
    MissingExtractedData, MissingRawData, InsufficientTrials, BlockStructureBug,
    IncompleteEventTimes, TrialsNotInPhotometryTime,
    FewUniqueSamples, QCValidationError, AmbiguousRegionMapping,
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
        raw_br = session_series.get('brain_region', [])
        self.brain_region = list(raw_br) if isinstance(raw_br, (list, np.ndarray)) else []
        raw_hm = session_series.get('hemisphere', [])
        self.hemisphere = list(raw_hm) if isinstance(raw_hm, (list, np.ndarray)) else []

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
        contrasts = compute_trial_contrasts(self.trials)
        self.trials['stim_side'] = contrasts['stim_side']
        self.trials['signed_contrast'] = contrasts['signed_contrast']
        self.trials['contrast'] = contrasts['contrast']

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
        self._match_photometry_to_metadata()

    def _match_photometry_to_metadata(self):
        """Rename photometry columns to match brain_region metadata.

        Photometry columns from brainbox may use bare names ('VTA') while
        brain_region metadata includes hemisphere suffixes ('VTA-r').
        This method renames columns to match metadata names.

        Raises AmbiguousRegionMapping if any column matches zero or multiple
        metadata entries (e.g. bare 'NBM' with metadata ['NBM-l', 'NBM-r']).
        """
        if not self.photometry or not self.brain_region:
            return

        ref_band = next(iter(self.photometry))
        phot_cols = list(self.photometry[ref_band].columns)

        # If columns already match metadata, nothing to do
        if sorted(phot_cols) == sorted(self.brain_region):
            return

        # Build rename map: each photometry column must match exactly one
        # metadata entry by name (exact match or bare→suffixed)
        rename = {}
        for col in phot_cols:
            if col in self.brain_region:
                continue  # exact match, no rename needed
            matches = [r for r in self.brain_region if r.rsplit('-', 1)[0] == col]
            if len(matches) == 1:
                rename[col] = matches[0]
            elif len(matches) == 0:
                raise AmbiguousRegionMapping(
                    f"Photometry column '{col}' has no match in "
                    f"brain_region {self.brain_region}"
                )
            else:
                raise AmbiguousRegionMapping(
                    f"Photometry column '{col}' matches multiple entries in "
                    f"brain_region {self.brain_region}: {matches}"
                )

        for band_df in self.photometry.values():
            band_df.rename(columns=rename, inplace=True)

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
                cols = TRIAL_COLUMNS + ['contrast', 'signed_contrast']
                available = [c for c in cols if c in self.trials.columns]
                grp = f.create_group('trials')
                for col in available:
                    vals = self.trials[col].values
                    if vals.dtype == object:
                        vals = vals.astype('S')
                    grp.create_dataset(col, data=vals)

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
            'signal', 'trials', 'responses', 'wheel'.
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
                data = {}
                for col in f['trials']:
                    vals = f[f'trials/{col}'][:]
                    if vals.dtype.kind == 'S':
                        vals = vals.astype(str)
                    data[col] = vals
                self.trials = pd.DataFrame(data)

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

            if (groups is None or 'wheel' in groups) and 'wheel' in f:
                self.wheel_velocity = f['wheel/velocity'][:].astype(np.float32)
                self.wheel_fs = f['wheel'].attrs['fs']
                self._wheel_t0_event = f['wheel'].attrs['t0_event']
                self._wheel_t1_event = f['wheel'].attrs['t1_event']

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

    # =========================================================================
    # Response Vector
    # =========================================================================

    _DEFAULT_FEATURE_EVENTS = ('stimOn_times', 'feedback_times')

    def get_response_vector(self, brain_region, hemisphere,
                            min_trials=5, normalize=None, events=None):
        """Compute a response vector: one scalar per trial-type condition.

        Each condition is defined by event × contrast × side × feedback.

        Parameters
        ----------
        brain_region : str
            Region to extract (must be in responses coords).
        hemisphere : str or None
            'l', 'r', or None (midline). Used to lateralize contrasts.
        events : sequence of str, optional
            Event names to include. Defaults to stimOn_times and
            feedback_times.
        min_trials : int
            Minimum trials per condition cell; fewer → NaN.
        normalize : str or None
            None (default) or 'minmax'.

        Returns
        -------
        pd.Series
            Index = condition labels, values = mean response magnitudes.
        """
        if normalize not in (None, 'minmax'):
            raise ValueError(f"normalize must be None or 'minmax', got {normalize!r}")

        responses = self.mask_subsequent_events(self.responses)
        responses = self.subtract_baseline(responses)
        tpts = responses.coords['time'].values

        # Lateralize using stim_side column
        contra_side = {'l': 'right', 'r': 'left'}.get(hemisphere, 'right')
        stim_side = self.trials['stim_side'].values
        contrast = self.trials['contrast'].values
        feedback = self.trials['feedbackType'].values

        contrasts = sorted(self.trials['contrast'].unique())
        if events is None:
            events = list(self._DEFAULT_FEATURE_EVENTS)
        # Filter to events present in the data
        available = set(responses.coords['event'].values)
        events = [e for e in events if e in available]

        is_contra = (stim_side == contra_side)

        # Build conditions: event × contrast × side × feedback
        win = RESPONSE_WINDOWS['early']
        condition_specs = []
        for event in events:
            for c in contrasts:
                for side, side_contra in [('contra', True), ('ipsi', False)]:
                    for fb, fb_label in [(1, 'correct'), (-1, 'incorrect')]:
                        cfmt = int(c) if c == int(c) else c
                        label = f"{event.replace('_times', '')}_c{cfmt}_{side}_{fb_label}"
                        condition_specs.append((event, c, side_contra, fb, label))

        result = {}
        for event, c, side_contra, fb, label in condition_specs:
            trial_mask = (
                np.isclose(contrast, c)
                & (is_contra == side_contra)
                & (feedback == fb)
            )
            n = trial_mask.sum()
            if n < min_trials:
                result[label] = np.nan
                continue
            resp = responses.sel(region=brain_region, event=event).values[trial_mask]
            magnitudes = compute_response_magnitude(resp, tpts, win)
            result[label] = np.nanmean(magnitudes)

        vec = pd.Series(result)

        if normalize == 'minmax':
            vmin, vmax = vec.min(), vec.max()
            if vmax > vmin:
                vec = (vec - vmin) / (vmax - vmin)

        return vec


class PhotometrySessionGroup:
    """Collection of recordings spanning multiple sessions.

    Parameters
    ----------
    recordings : pd.DataFrame
        One row per recording (session × region).
    one : one.api.One
        ONE connection instance.
    h5_dir : Path, optional
        Directory containing {eid}.h5 files.
    """

    def __init__(self, recordings, one, h5_dir=None):
        self.recordings = recordings.reset_index(drop=True)
        self.one = one
        self.h5_dir = h5_dir if h5_dir is not None else SESSIONS_H5_DIR
        self._sessions = {}  # eid → PhotometrySession
        self.response_features = None
        self.similarity_matrix = None
        self.decoder = None

    def __len__(self):
        return len(self.recordings)

    def _get_session(self, rec):
        """Get or create a PhotometrySession for a recording row."""
        eid = rec['eid']
        if eid not in self._sessions:
            self._sessions[eid] = PhotometrySession(rec, one=self.one, load_data=False)
        return self._sessions[eid]

    def __iter__(self):
        for _, rec in self.recordings.iterrows():
            yield rec, self._get_session(rec)

    def __getitem__(self, idx):
        rec = self.recordings.iloc[idx]
        return rec, self._get_session(rec)

    def filter(self, mask):
        """Return a new group with recordings selected by boolean mask."""
        filtered = self.recordings[mask].copy()
        new_group = PhotometrySessionGroup(filtered, one=self.one, h5_dir=self.h5_dir)
        # Share already-loaded sessions
        for eid, ps in self._sessions.items():
            if eid in filtered['eid'].values:
                new_group._sessions[eid] = ps
        return new_group

    def get_response_features(self, nan_handling='drop_sessions',
                              nan_threshold=0.3, **kwargs):
        """Build response feature vectors for all recordings.

        Loads H5 files one at a time, extracts response vectors, then
        discards raw data to keep memory usage low.

        Parameters
        ----------
        nan_handling : str
            How to handle NaN in the feature matrix:
            - ``'drop_sessions'``: drop recordings with any NaN feature.
            - ``'drop_features'``: drop feature columns whose NaN rate
              exceeds ``nan_threshold``.
        nan_threshold : float
            Fraction of recordings allowed to be NaN before a feature
            column is dropped. Only used when ``nan_handling='drop_features'``.
        **kwargs
            Forwarded to ``PhotometrySession.get_response_vector``.
            ``min_trials`` defaults to 1.

        Returns
        -------
        pd.DataFrame
            Rows indexed by (eid, target_NM), columns = condition labels.
        """
        _valid = ('drop_sessions', 'drop_features')
        if nan_handling not in _valid:
            raise ValueError(
                f"nan_handling must be one of {_valid}, got {nan_handling!r}"
            )

        kwargs.setdefault('min_trials', 1)

        from pathlib import Path
        import logging

        rows = {}
        for _, rec in self.recordings.iterrows():
            eid = rec['eid']
            brain_region = rec['brain_region']
            hemisphere = rec['hemisphere']
            target_nm = rec['target_NM']

            ps = self._get_session(rec)

            # Load H5 if responses not yet available
            if not hasattr(ps, 'responses') or not hasattr(ps, 'trials'):
                h5_path = Path(self.h5_dir) / f'{eid}.h5'
                if not h5_path.exists():
                    logging.warning(f"H5 file not found: {h5_path}")
                    continue
                ps.load_h5(h5_path, groups=['trials', 'responses'])

            if not hasattr(ps, 'responses') or not hasattr(ps, 'trials'):
                continue
            if brain_region not in ps.responses.coords['region'].values:
                continue

            vec = ps.get_response_vector(
                brain_region=brain_region, hemisphere=hemisphere, **kwargs,
            )
            rows[(eid, target_nm)] = vec

            # Discard raw data to free memory
            del ps.responses
            del ps.trials

        if not rows:
            self.response_features = pd.DataFrame()
            return self.response_features

        df = pd.DataFrame(rows).T
        df.index = pd.MultiIndex.from_tuples(df.index, names=['eid', 'target_NM'])

        if nan_handling == 'drop_sessions':
            df = df.dropna()
        elif nan_handling == 'drop_features':
            nan_rate = df.isna().mean()
            df = df.loc[:, nan_rate <= nan_threshold]

        self.response_features = df
        return df

    def response_similarity_matrix(self, **kwargs):
        """Pairwise cosine similarity of response feature vectors.

        Calls ``get_response_features`` if not already computed.

        Parameters
        ----------
        **kwargs
            Forwarded to ``get_response_features`` if needed.

        Returns
        -------
        pd.DataFrame
            Symmetric similarity matrix.
        """
        from iblnm.analysis import cosine_similarity_matrix

        if self.response_features is None:
            self.get_response_features(**kwargs)

        self.similarity_matrix = cosine_similarity_matrix(self.response_features)
        return self.similarity_matrix

    def decode_target(self, **kwargs):
        """Decode target-NM from response features.

        Creates a ``TargetNMDecoder``, fits it with leave-one-subject-target-out CV,
        and computes feature unique contributions. Stores the decoder as
        ``self.decoder``.

        Calls ``get_response_features`` if not already computed.

        Parameters
        ----------
        **kwargs
            Forwarded to ``get_response_features`` if needed.

        Returns
        -------
        TargetNMDecoder
            Fitted decoder with results as attributes.
        """
        from iblnm.analysis import TargetNMDecoder

        if self.response_features is None:
            self.get_response_features(**kwargs)

        # Labels from the index; subjects looked up from recordings
        labels = self.response_features.index.get_level_values('target_NM')
        labels = pd.Series(labels.values, index=self.response_features.index)

        rec_indexed = self.recordings.set_index(['eid', 'target_NM'])
        subjects = rec_indexed['subject'].reindex(self.response_features.index)

        self.decoder = TargetNMDecoder(self.response_features, labels, subjects)
        self.decoder.fit()
        self.decoder.unique_contribution()
        return self.decoder

