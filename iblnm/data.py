import warnings

import numpy as np
import pandas as pd
from datetime import datetime

from brainbox.io.one import PhotometrySessionLoader
from iblphotometry.fpio import from_neurophotometrics_df_to_photometry_df
from iblphotometry import metrics
from iblphotometry.qc import qc_signals

from one.alf.exceptions import ALFObjectNotFound

from iblnm.config import (
    ANALYSIS_QC_BLOCKERS, BASELINE_WINDOW, EVENT_COMPLETENESS_THRESHOLD,
    MIN_NTRIALS, MIN_PERFORMANCE, N_UNIQUE_SAMPLES_THRESHOLD,
    PREPROCESSING_PIPELINES, QC_METRICS_KWARGS, QC_RAW_METRICS,
    QC_SLIDING_KWARGS, QC_SLIDING_METRICS, REQUIRED_CONTRASTS,
    RESPONSE_EVENTS, RESPONSE_WINDOW, RESPONSE_WINDOWS, SESSIONS_H5_DIR,
    SESSION_TYPES_TO_ANALYZE, SUBJECTS_TO_EXCLUDE, TARGETNMS_TO_ANALYZE,
    TARGET_FS, TRIAL_COLUMNS, WHEEL_FS,
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
            session_series (pd.Series): A pandas Series containing session metadata.
                Required fields: eid, subject, start_time, number.
                All other fields are optional and default to safe empty values.
        """
        self.eid = session_series['eid']
        self.subject = session_series['subject']

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
        self.task_protocol = session_series.get('task_protocol', '')
        self.session_type = session_series.get('session_type', '')
        self.NM = session_series.get('NM')
        self.strain = session_series.get('strain')
        self.line = session_series.get('line')
        raw_gt = session_series.get('genotype', [])
        self.genotype = list(raw_gt) if isinstance(raw_gt, (list, np.ndarray)) else (
            [raw_gt] if raw_gt else [])
        self.users = list(session_series.get('users', []))
        self.end_time = session_series.get('end_time')
        self.datasets = list(session_series.get('datasets', []))
        self.session_length = session_series.get('session_length')
        self.day_n = session_series.get('day_n')

        raw_br = session_series.get('brain_region', [])
        self.brain_region = list(raw_br) if isinstance(raw_br, (list, np.ndarray)) else []
        raw_hm = session_series.get('hemisphere', [])
        self.hemisphere = list(raw_hm) if isinstance(raw_hm, (list, np.ndarray)) else []
        raw_tnm = session_series.get('target_NM', [])
        self.target_NM = list(raw_tnm) if isinstance(raw_tnm, (list, np.ndarray)) else []

        self.errors = []

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
        """Convert the session metadata to a dictionary."""
        return {
            'eid': self.eid,
            'subject': self.subject,
            'start_time': self.start_time.isoformat(),
            'number': self.number,
            'task_protocol': self.task_protocol,
            'session_type': self.session_type,
            'projects': self.projects,
            'lab': self.lab,
            'url': self.url,
            'session_n': self.session_n,
            'NM': self.NM,
            'strain': self.strain,
            'line': self.line,
            'genotype': self.genotype,
            'users': self.users,
            'end_time': self.end_time,
            'brain_region': self.brain_region,
            'hemisphere': self.hemisphere,
            'target_NM': self.target_NM,
            'datasets': self.datasets,
            'session_length': self.session_length,
            'day_n': self.day_n,
        }


    def to_series(self) -> pd.Series:
        return pd.Series(self.to_dict())

    @classmethod
    def from_h5(cls, fpath, one=None):
        """Construct a PhotometrySession from a saved H5 file.

        Reads the /metadata group to build the session Series, then loads
        all other available groups (errors, signal, trials, responses, wheel).

        Parameters
        ----------
        fpath : Path or str
            Path to the HDF5 file.
        one : one.api.One, optional
            ONE connection instance. Not required for cached data access.
        """
        import h5py

        # Read metadata to build the init Series
        with h5py.File(fpath, 'r') as f:
            if 'metadata' not in f:
                raise ValueError(f"H5 file has no /metadata group: {fpath}")
            grp = f['metadata']
            data = {}
            for attr, is_list in cls._METADATA_FIELDS:
                if is_list:
                    if attr in grp:
                        data[attr] = [v.decode() if isinstance(v, bytes) else v
                                      for v in grp[attr][:]]
                    else:
                        data[attr] = []
                else:
                    if attr in grp.attrs:
                        val = grp.attrs[attr]
                        if isinstance(val, bytes):
                            val = val.decode()
                        elif hasattr(val, 'item'):
                            val = val.item()  # numpy scalar → native
                        if isinstance(val, str) and val == '__none__':
                            val = None
                        data[attr] = val

        series = pd.Series(data)
        if one is not None:
            ps = cls(series, one=one, load_data=False)
        else:
            # Bypass parent __init__ which requires ONE connection.
            # All attributes are set manually from the H5 metadata.
            ps = object.__new__(cls)
            ps.one = None
            ps.session_path = ''
            ps.eid = series.get('eid', '')
            ps.revision = ''
            ps.photometry = {}
            ps.qc = pd.DataFrame()
            ps.errors = []
            # Set all metadata attrs from series
            ps.subject = series.get('subject', '')
            start_time = series.get('start_time', '')
            if isinstance(start_time, str) and start_time:
                ps.start_time = datetime.fromisoformat(start_time)
            else:
                ps.start_time = start_time
            ps.number = int(series.get('number', 0))
            ps.lab = series.get('lab')
            ps.projects = list(series.get('projects', []))
            ps.url = series.get('url')
            ps.session_n = series.get('session_n')
            ps.task_protocol = series.get('task_protocol', '')
            ps.session_type = series.get('session_type', '')
            ps.NM = series.get('NM')
            ps.strain = series.get('strain')
            ps.line = series.get('line')
            raw_gt = series.get('genotype', [])
            ps.genotype = list(raw_gt) if isinstance(raw_gt, (list, np.ndarray)) else (
                [raw_gt] if raw_gt else [])
            ps.users = list(series.get('users', []))
            ps.end_time = series.get('end_time')
            ps.datasets = list(series.get('datasets', []))
            ps.session_length = series.get('session_length')
            ps.day_n = series.get('day_n')
            raw_br = series.get('brain_region', [])
            ps.brain_region = list(raw_br) if isinstance(raw_br, (list, np.ndarray)) else []
            raw_hm = series.get('hemisphere', [])
            ps.hemisphere = list(raw_hm) if isinstance(raw_hm, (list, np.ndarray)) else []
            raw_tnm = series.get('target_NM', [])
            ps.target_NM = list(raw_tnm) if isinstance(raw_tnm, (list, np.ndarray)) else []

        # Load remaining groups
        ps.load_h5(fpath, groups=['errors', 'signal', 'trials', 'responses', 'wheel'])
        return ps

    def from_alyx(self):
        """Enrich session metadata by querying Alyx.

        Calls io and validation functions to populate subject info,
        brain regions, datasets, session type, and target NM. All errors
        are logged to self.errors rather than raised.

        Returns self for chaining.
        """
        from iblnm.io import (
            get_subject_info, get_session_dict, get_brain_region, get_datasets,
        )
        from iblnm.validation import (
            validate_subject, validate_strain, validate_line,
            validate_neuromodulator, validate_brain_region, validate_hemisphere,
            validate_datasets,
        )
        from iblnm.util import get_session_type, get_targetNM, get_session_length

        exlog = []
        s = self.to_series()

        # Subject info (strain, line, genotype, NM)
        s = get_subject_info(s, one=self.one, exlog=exlog)
        validate_subject(s, exlog=exlog)
        validate_strain(s, exlog=exlog)
        validate_line(s, exlog=exlog)
        validate_neuromodulator(s, exlog=exlog)

        # Session metadata (users, lab, end_time)
        s = get_session_dict(s, one=self.one, exlog=exlog)

        # Brain regions and hemispheres
        s = get_brain_region(s, one=self.one, exlog=exlog)
        validate_brain_region(s, exlog=exlog)
        validate_hemisphere(s, exlog=exlog)

        # Datasets
        s = get_datasets(s, one=self.one, exlog=exlog)
        validate_datasets(s, exlog=exlog)

        # Derived fields
        s = get_session_type(s, exlog=exlog)
        s = get_targetNM(s, exlog=exlog)
        s = get_session_length(s, exlog=exlog)

        # Update self from enriched series
        for attr, _ in self._METADATA_FIELDS:
            if attr in s.index:
                val = s[attr]
                if attr == 'start_time' and isinstance(val, str):
                    val = datetime.fromisoformat(val)
                elif hasattr(val, 'item'):
                    val = val.item()  # numpy scalar → native Python type
                setattr(self, attr, val)

        # Normalize list attrs
        for attr in ('brain_region', 'hemisphere', 'target_NM',
                     'users', 'datasets', 'projects', 'genotype'):
            val = getattr(self, attr, [])
            if isinstance(val, np.ndarray):
                setattr(self, attr, list(val))
            elif isinstance(val, str):
                setattr(self, attr, [val] if val else [])
            elif not isinstance(val, list):
                setattr(self, attr, [])

        self.errors.extend(exlog)
        return self

    def log_error(self, error):
        """Log an exception to the session's error list.

        Parameters
        ----------
        error : Exception
            The exception to log. Type, message, and traceback are captured.
        """
        from iblnm.validation import make_log_entry
        self.errors.append(make_log_entry(self.eid, error=error))

    # Metadata fields: (attr_name, is_list)
    # Scalars are stored as H5 attrs, lists as H5 datasets.
    _METADATA_FIELDS = [
        ('eid', False), ('subject', False), ('start_time', False),
        ('number', False), ('task_protocol', False), ('session_type', False),
        ('lab', False), ('NM', False), ('strain', False), ('line', False),
        ('genotype', True), ('end_time', False), ('session_length', False),
        ('day_n', False), ('session_n', False), ('url', False),
        ('projects', True), ('users', True), ('brain_region', True),
        ('hemisphere', True), ('target_NM', True), ('datasets', True),
    ]

    def load_trials(self):
        try:
            super().load_trials()
        except ALFObjectNotFound:
            try:
                _ = self.one.load_dataset(self.eid, '_iblrig_taskData.raw.jsonable')
            except ALFObjectNotFound:
                raise MissingRawData("_iblrig_taskData.raw.jsonable")
            raise MissingExtractedData("_ibl_trials.table.pqt")
        except Exception as e:
            raise MissingExtractedData(
                f"_ibl_trials.table.pqt ({type(e).__name__}: {e})"
            ) from e
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

    def save_h5(self, fpath=None, groups=None, band='GCaMP_preprocessed', mode='a'):
        """Save session data to HDF5.

        Parameters
        ----------
        fpath : Path or str, optional
            Output path. Defaults to SESSIONS_H5_DIR / {eid}.h5.
        groups : sequence of str, optional
            Which data groups to write. Any subset of:
            'metadata', 'errors', 'signal', 'trials', 'responses', 'wheel',
            'photometry_qc_metrics'.
            None auto-detects all available data groups.
        mode : str
            HDF5 file open mode ('a' creates/appends, 'w' truncates).
        """
        import h5py
        from pathlib import Path
        if fpath is None:
            fpath = SESSIONS_H5_DIR / f'{self.eid}.h5'
        fpath = Path(fpath)
        fpath.parent.mkdir(parents=True, exist_ok=True)

        if groups is None:
            groups = [g for g, available in (
                ('signal',    band in self.photometry),
                ('trials',    hasattr(self, 'trials') and self.trials is not None),
                ('responses', hasattr(self, 'responses') and self.responses is not None),
                ('wheel',     hasattr(self, 'wheel_velocity') and self.wheel_velocity is not None),
                ('photometry_qc_metrics', hasattr(self, 'qc') and self.qc is not None
                 and len(self.qc) > 0),
            ) if available]

        with h5py.File(fpath, mode) as f:
            if 'metadata' in groups:
                if 'metadata' in f:
                    del f['metadata']
                grp = f.create_group('metadata')
                for attr, is_list in self._METADATA_FIELDS:
                    val = getattr(self, attr, None)
                    if is_list:
                        items = list(val) if val else []
                        grp.create_dataset(
                            attr,
                            data=[s.encode() if isinstance(s, str) else s
                                  for s in items],
                            dtype=h5py.string_dtype() if items else h5py.string_dtype(),
                        )
                    else:
                        if attr == 'start_time' and hasattr(val, 'isoformat'):
                            val = val.isoformat()
                        if val is None:
                            grp.attrs[attr] = '__none__'
                        elif isinstance(val, str):
                            grp.attrs[attr] = val
                        else:
                            grp.attrs[attr] = val

            if 'errors' in groups:
                if 'errors' in f:
                    del f['errors']
                grp = f.create_group('errors')
                seen = set()
                unique_errors = []
                for e in self.errors:
                    key = (e.get('eid', ''), e.get('error_type', ''), e.get('error_message', ''))
                    if key not in seen:
                        seen.add(key)
                        unique_errors.append(e)
                if unique_errors:
                    for col in ('eid', 'error_type', 'error_message', 'traceback'):
                        vals = [str(e.get(col, '') or '') for e in unique_errors]
                        grp.create_dataset(col, data=vals,
                                           dtype=h5py.string_dtype())
                # Empty group signals "no errors" — distinguishable from
                # "errors not yet written" (no group at all).

            if 'signal' in groups:
                f.attrs['eid'] = self.eid
                f.attrs['subject'] = self.subject
                f.attrs['session_type'] = self.session_type
                f.attrs['date'] = self.date
                f.attrs['fs'] = TARGET_FS
                f.attrs['response_window'] = self.RESPONSE_WINDOW

                if 'times' in f:
                    del f['times']
                if 'preprocessed' in f:
                    del f['preprocessed']
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

            if 'photometry_qc_metrics' in groups and hasattr(self, 'qc') and self.qc is not None and len(self.qc) > 0:
                if 'photometry_qc_metrics' in f:
                    del f['photometry_qc_metrics']
                grp = f.create_group('photometry_qc_metrics')
                for col in self.qc.columns:
                    vals = self.qc[col].values
                    if vals.dtype == object:
                        vals = vals.astype('S')
                    grp.create_dataset(col, data=vals)

    def load_h5(self, fpath, groups=None):
        """Load session data from HDF5 file.

        Parameters
        ----------
        fpath : Path or str
            Path to the HDF5 file.
        groups : sequence of str, optional
            Which data groups to load. Any subset of:
            'metadata', 'errors', 'signal', 'trials', 'responses', 'wheel',
            'photometry_qc_metrics'.
            None loads all groups present in the file.
        """
        import h5py
        import xarray as xr

        with h5py.File(fpath, 'r') as f:
            if (groups is None or 'metadata' in groups) and 'metadata' in f:
                grp = f['metadata']
                for attr, is_list in self._METADATA_FIELDS:
                    if is_list:
                        if attr in grp:
                            vals = [v.decode() if isinstance(v, bytes) else v
                                    for v in grp[attr][:]]
                            setattr(self, attr, vals)
                    else:
                        if attr in grp.attrs:
                            val = grp.attrs[attr]
                            if isinstance(val, bytes):
                                val = val.decode()
                            elif hasattr(val, 'item'):
                                val = val.item()
                            if isinstance(val, str) and val == '__none__':
                                val = None
                            if attr == 'start_time' and isinstance(val, str):
                                val = datetime.fromisoformat(val)
                            setattr(self, attr, val)

            if (groups is None or 'errors' in groups) and 'errors' in f:
                err_grp = f['errors']
                if 'error_type' in err_grp:
                    n = len(err_grp['error_type'])
                    self.errors = []
                    for i in range(n):
                        entry = {}
                        for col in ('eid', 'error_type', 'error_message', 'traceback'):
                            val = err_grp[col][i]
                            entry[col] = val.decode() if isinstance(val, bytes) else val
                        self.errors.append(entry)
                else:
                    self.errors = []

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

            if (groups is None or 'photometry_qc_metrics' in groups) and 'photometry_qc_metrics' in f:
                grp = f['photometry_qc_metrics']
                data = {}
                for col in grp:
                    vals = grp[col][:]
                    if vals.dtype.kind == 'S':
                        vals = vals.astype(str)
                    data[col] = vals
                self.qc = pd.DataFrame(data)

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
            BASELINE_WINDOW from config.

        Returns
        -------
        xr.DataArray
            Baseline-subtracted responses, same shape and coords as input.
        """
        if responses is None:
            responses = self.responses
        if window is None:
            window = BASELINE_WINDOW
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

    def get_trial_timings(self) -> pd.DataFrame:
        """Compute per-trial reaction, movement, and response times."""
        trials = self.trials
        n = len(trials)

        if ('firstMovement_times' in trials.columns
                and 'stimOn_times' in trials.columns):
            reaction_time = (trials['firstMovement_times'].values
                             - trials['stimOn_times'].values)
        else:
            reaction_time = np.full(n, np.nan)

        if ('feedback_times' in trials.columns
                and 'firstMovement_times' in trials.columns):
            movement_time = (trials['feedback_times'].values
                             - trials['firstMovement_times'].values)
        else:
            movement_time = np.full(n, np.nan)

        if ('feedback_times' in trials.columns
                and 'stimOn_times' in trials.columns):
            response_time = (trials['feedback_times'].values
                             - trials['stimOn_times'].values)
        else:
            response_time = np.full(n, np.nan)

        return pd.DataFrame({
            'eid': self.eid,
            'trial': range(n),
            'reaction_time': reaction_time,
            'movement_time': movement_time,
            'response_time': response_time,
        })

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

        # Lateralize using stim_side column (set by compute_trial_contrasts in load_trials)
        if 'stim_side' not in self.trials.columns:
            raise KeyError(
                "'stim_side' column missing from trials. "
                "Regenerate H5 files by re-running photometry.py."
            )
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


def _process_worker(eid, row_dict, h5_dir, fn, kwargs):
    """Worker function for parallel process(). Runs in a subprocess.

    Creates its own ONE connection, builds a PhotometrySession, calls
    fn(ps, **kwargs), and flushes errors to H5.
    """
    from pathlib import Path
    from iblnm.io import _get_default_connection

    one = _get_default_connection()
    h5_path = Path(h5_dir) / f'{eid}.h5'

    if h5_path.exists():
        ps = PhotometrySession.from_h5(h5_path, one=one)
    else:
        row = pd.Series(row_dict)
        ps = PhotometrySession(row, one=one, load_data=False)

    try:
        result = fn(ps, **kwargs)
    except Exception as e:
        ps.log_error(e)
        result = None
    finally:
        if h5_path.exists() or ps.errors:
            ps.save_h5(h5_path, groups=['errors'])

    return result


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

    def __init__(self, sessions, one, h5_dir=None):
        self._catalog = sessions.reset_index(drop=True)
        self._filter_mask = pd.Series(True, index=self._catalog.index)
        self._dedup_mask = pd.Series(True, index=self._catalog.index)
        self._recordings_targetnms = None
        self.one = one
        self.h5_dir = h5_dir if h5_dir is not None else SESSIONS_H5_DIR
        self._sessions = {}  # eid → PhotometrySession cache
        self.response_traces = None
        self.response_traces_tpts = None
        self.mean_traces = None
        self.response_magnitudes = None
        self.trial_timing = None
        self.peak_velocity = None
        self.response_features = None
        self.psychometric_features = None
        self.similarity_matrix = None
        self.decoder = None
        self.glm_response_features = None
        self.cca_result = None
        self.cohort_cca_results = None
        self.cohort_cca_data = None
        self.cohort_cca_cross_projections = None
        self.cohort_cca_weight_similarities = None
        self.lmm_results = None
        self.lmm_coefficients = None
        self.wheel_lmm_results = None
        self.wheel_lmm_summary = None

    @classmethod
    def from_catalog(cls, catalog, one, h5_dir=None):
        """Build a group from a session catalog DataFrame.

        Validates parallel list columns. Call ``filter_sessions`` separately.

        Parameters
        ----------
        catalog : pd.DataFrame
            Session catalog (one row per session, with list columns for
            brain_region, hemisphere, target_NM). Enrich with
            ``collect_session_errors`` before passing if you need error-based
            filtering.
        one : one.api.One
            ONE connection instance.
        h5_dir : Path, optional
            Directory containing {eid}.h5 files.
        """
        from iblnm.config import SESSION_SCHEMA
        from iblnm.util import enforce_schema, validate_parallel_lists

        df = enforce_schema(catalog.copy(), SESSION_SCHEMA)

        parallel_cols = ['brain_region', 'hemisphere', 'target_NM']
        df = validate_parallel_lists(df, parallel_cols)

        return cls(df, one=one, h5_dir=h5_dir)

    @property
    def sessions(self):
        """Session-level view: _catalog rows passing both dedup and filter masks."""
        combined = self._dedup_mask & self._filter_mask
        return self._catalog[combined].copy().reset_index(drop=True)

    @property
    def recordings(self):
        """Recording-level view: sessions exploded to one row per region.

        Reflects the current filter and dedup masks. Filters to
        _recordings_targetnms (set by filter_sessions) when not None.
        """
        parallel_cols = ['brain_region', 'hemisphere', 'target_NM']
        df = self.sessions.explode(parallel_cols).copy()
        df['fiber_idx'] = df.groupby('eid').cumcount()
        if self._recordings_targetnms is not None:
            df = df[df['target_NM'].isin(self._recordings_targetnms)]
        return df.reset_index(drop=True)

    def filter_sessions(self, session_types=SESSION_TYPES_TO_ANALYZE,
                        exclude_subjects=SUBJECTS_TO_EXCLUDE,
                        qc_blockers=ANALYSIS_QC_BLOCKERS,
                        targetnms=TARGETNMS_TO_ANALYZE,
                        min_performance=MIN_PERFORMANCE,
                        required_contrasts=REQUIRED_CONTRASTS,
                        lab=None, start_time_min=None):
        """Compute a boolean filter mask over _catalog. Non-destructive.

        Stores a new _filter_mask on each call. Access filtered data via the
        ``sessions`` and ``recordings`` properties. Call multiple times to get
        different filtered views.

        Parameters
        ----------
        session_types : tuple of str or False
            Session types to keep. Defaults to config.SESSION_TYPES_TO_ANALYZE.
            False → skip.
        exclude_subjects : list of str
            Subjects to exclude. Defaults to config.SUBJECTS_TO_EXCLUDE.
            Pass ``[]`` to skip.
        qc_blockers : set of str
            Error types that block a session. Defaults to
            config.ANALYSIS_QC_BLOCKERS. Pass ``set()`` to skip.
            Silently skipped if ``logged_errors`` is not present on the catalog.
        targetnms : list of str or None
            Target-NM values to retain in sessions and recordings.
            Defaults to config.TARGETNMS_TO_ANALYZE. Pass None to skip.
        min_performance : float, dict, or False
            Minimum fraction_correct. Defaults to config.MIN_PERFORMANCE.
            False → skip. Requires 'fraction_correct' in the catalog.
        required_contrasts : frozenset of float or False
            Required contrast set. Defaults to config.REQUIRED_CONTRASTS.
            False → skip. Requires 'contrasts' in the catalog.
        lab : str, optional
            Keep only sessions from this lab.
        start_time_min : str or date, optional
            Keep only subjects whose first session is >= this date.

        Returns
        -------
        None
        """
        # Resolve False → None for session_types (disables the filter)
        if session_types is False:
            session_types = None

        df = self._catalog
        true = pd.Series(True, index=df.index)

        # Build individual masks
        type_mask = df['session_type'].isin(session_types) if session_types is not None else true
        subject_mask = ~df['subject'].isin(exclude_subjects) if exclude_subjects else true
        lab_mask = (df['lab'] == lab) if (lab is not None and 'lab' in df.columns) else true

        if start_time_min is not None and 'start_time' in df.columns:
            dt_series = pd.to_datetime(df['start_time'], format='ISO8601')
            first_per_row = dt_series.groupby(df['subject']).transform('min')
            start_mask = first_per_row >= pd.Timestamp(start_time_min)
        else:
            start_mask = true

        if qc_blockers and 'logged_errors' in df.columns:
            qc_mask = df['logged_errors'].apply(
                lambda e: not any(err in qc_blockers for err in e)
            )
        else:
            qc_mask = true

        if targetnms is not None and 'target_NM' in df.columns:
            targetnms_set = set(targetnms)
            target_mask = df['target_NM'].apply(
                lambda ts: any(t in targetnms_set for t in ts)
                if isinstance(ts, (list, np.ndarray)) else ts in targetnms_set
            )
        else:
            target_mask = true

        if min_performance is not False and 'fraction_correct' in df.columns:
            if isinstance(min_performance, dict):
                perf_mask = true.copy()
                for stype, threshold in min_performance.items():
                    is_type = df['session_type'] == stype
                    meets = df['fraction_correct'] >= threshold
                    perf_mask = perf_mask & (~is_type | meets)
            else:
                perf_mask = df['fraction_correct'] >= min_performance
        else:
            perf_mask = true

        if required_contrasts is not False and 'contrasts' in df.columns:
            required_set = set(required_contrasts)
            contrast_mask = df['contrasts'].apply(
                lambda c: set(c) == required_set
                if isinstance(c, (list, np.ndarray)) else False
            )
        else:
            contrast_mask = true

        mask = type_mask & subject_mask & lab_mask & start_mask & qc_mask & target_mask & perf_mask & contrast_mask

        self._filter_mask = mask
        self._recordings_targetnms = targetnms

        n = len(df)
        lines = [f"filter_sessions: {n} -> {int(mask.sum())}"]
        for label, m in [
            ('session_type', type_mask), ('excluded_subjects', subject_mask),
            ('lab', lab_mask), ('start_time', start_mask),
            ('qc_errors', qc_mask), ('target_NM', target_mask),
            ('performance', perf_mask), ('contrasts', contrast_mask),
        ]:
            removed = n - int(m.sum())
            if removed:
                lines.append(f"  -{removed:4d} {label}")
        print('\n'.join(lines))
        return None

    def deduplicate(self):
        """Compute _dedup_mask by resolving duplicate (subject, day_n) sessions.

        Operates on _catalog (full unfiltered table). True duplicates are
        logged as TrueDuplicateSession entries; one row is kept as fallback.
        Updates self.recordings to match.

        Returns
        -------
        self
        """
        from iblnm.util import resolve_duplicate_group

        if 'logged_errors' not in self._catalog.columns:
            self._catalog['logged_errors'] = [[] for _ in range(len(self._catalog))]

        exlog = []
        df_kept = (
            self._catalog.groupby(['subject', 'day_n'], group_keys=False)
            .apply(resolve_duplicate_group, exlog=exlog, include_groups=False)
        )
        if isinstance(df_kept, pd.DataFrame):
            kept_eids = set(df_kept['eid'])
        else:
            kept_eids = {df_kept['eid']} if isinstance(df_kept, pd.Series) else set()

        self._dedup_mask = self._catalog['eid'].isin(kept_eids)
        return pd.DataFrame(exlog) if exlog else pd.DataFrame(
            columns=['eid', 'error_type', 'error_message', 'traceback']
        )

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

    def process(self, fn, workers=1, **kwargs):
        """Apply a function to each unique session in the group.

        For each session, instantiates a PhotometrySession (from H5 if
        available, otherwise from the recording row), calls fn(ps, **kwargs),
        catches any exception as a fatal error, and always flushes accumulated
        errors to the session's H5 file.

        Parameters
        ----------
        fn : callable
            Function taking a PhotometrySession (plus any **kwargs) and
            returning a result. Must be a top-level function (picklable)
            when workers > 1. Non-fatal errors should be logged via
            ps.log_error() inside fn. Fatal errors can be raised and will
            be caught by process().
        workers : int
            Number of parallel workers. 1 = sequential.
        **kwargs
            Extra keyword arguments forwarded to fn(ps, **kwargs).

        Returns
        -------
        list
            Results from fn, one per unique session. None for failed sessions.
        """
        from pathlib import Path
        from tqdm import tqdm

        if workers > 1:
            return self._process_parallel(fn, workers, **kwargs)

        results = []
        for _, row in tqdm(self.sessions.iterrows(), total=len(self.sessions),
                           desc="Processing"):
            eid = row['eid']
            h5_path = Path(self.h5_dir) / f'{eid}.h5'
            if h5_path.exists():
                ps = PhotometrySession.from_h5(h5_path, one=self.one)
            else:
                ps = PhotometrySession(row, one=self.one, load_data=False)

            try:
                result = fn(ps, **kwargs)
                results.append(result)
            except Exception as e:
                ps.log_error(e)
                results.append(None)
            finally:
                # Always flush errors to H5
                if h5_path.exists() or ps.errors:
                    ps.save_h5(h5_path, groups=['errors'])
        return results

    def _process_parallel(self, fn, workers, **kwargs):
        """Parallel implementation of process().

        Each worker creates its own ONE connection and PhotometrySession.
        fn must be a picklable top-level function (not a lambda or closure).
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from tqdm import tqdm

        # Serialize rows as dicts for pickling
        tasks = {row['eid']: row.to_dict()
                 for _, row in self.sessions.iterrows()}

        results = [None] * len(tasks)
        eid_list = list(tasks.keys())
        eid_to_idx = {eid: i for i, eid in enumerate(eid_list)}

        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _process_worker, eid, row_dict,
                    str(self.h5_dir), fn, kwargs,
                ): eid
                for eid, row_dict in tasks.items()
            }
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc="Processing"):
                eid = futures[future]
                try:
                    results[eid_to_idx[eid]] = future.result()
                except Exception as e:
                    print(f"\n  FATAL: {eid}: {type(e).__name__}: {e}")

        return results

    # -----------------------------------------------------------------
    # Parquet loaders — populate group attributes from saved files
    # -----------------------------------------------------------------

    def _load_parquet(self, path):
        """Read a parquet file and filter rows to current recordings.

        Returns None if the file does not exist.
        """
        from pathlib import Path
        path = Path(path)
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        eids = set(self.recordings['eid'])
        return df[df['eid'].isin(eids)].copy()

    def load_response_magnitudes(self, path):
        """Load response magnitudes from parquet, filtered to current recordings."""
        self.response_magnitudes = self._load_parquet(path)

    def load_trial_timing(self, path):
        """Load trial timing from parquet, filtered to current recordings."""
        self.trial_timing = self._load_parquet(path)

    def load_peak_velocity(self, path):
        """Load peak velocity from parquet, filtered to current recordings."""
        self.peak_velocity = self._load_parquet(path)

    def load_mean_traces(self, path):
        """Load mean traces from parquet, filtered to current recordings."""
        self.mean_traces = self._load_parquet(path)

    def load_response_features(self, path):
        """Load response features from parquet, filtered to current recordings."""
        from pathlib import Path
        path = Path(path)
        if not path.exists():
            self.response_features = None
            return
        df = pd.read_parquet(path)
        if 'eid' in df.columns:
            df = df.set_index(['eid', 'target_NM', 'fiber_idx'])
        eids = set(self.recordings['eid'])
        df = df[df.index.get_level_values('eid').isin(eids)]
        self.response_features = df

    # -----------------------------------------------------------------
    # Trace loading and extraction
    # -----------------------------------------------------------------

    def load_response_traces(self):
        """Load and cache per-trial response traces from H5 files.

        For each recording and event, loads the response xarray from H5,
        applies ``mask_subsequent_events`` and ``subtract_baseline``, and
        stores the per-trial traces in ``self.response_traces``.

        The cache is keyed by ``(eid, brain_region, event)`` with values
        containing ``traces``, ``tpts``, ``meta``, and ``trials``.

        Returns
        -------
        self
        """
        from pathlib import Path
        from tqdm import tqdm

        cache = {}

        for rec, ps in tqdm(self, total=len(self),
                            desc="Loading response traces"):
            eid = rec['eid']
            brain_region = rec['brain_region']
            hemisphere = rec['hemisphere']
            h5_path = Path(self.h5_dir) / f'{eid}.h5'

            if not h5_path.exists():
                continue

            ps.load_h5(h5_path, groups=['trials', 'responses'])

            if (getattr(ps, 'responses', None) is None
                    or getattr(ps, 'trials', None) is None):
                continue

            available_regions = list(ps.responses.coords['region'].values)
            if brain_region not in available_regions:
                continue

            responses = ps.mask_subsequent_events(ps.responses)
            responses = ps.subtract_baseline(responses)

            tpts = responses.coords['time'].values
            if self.response_traces_tpts is None:
                self.response_traces_tpts = tpts

            meta = {
                'eid': eid,
                'subject': rec['subject'],
                'session_type': rec.get('session_type'),
                'NM': rec.get('NM'),
                'target_NM': rec['target_NM'],
                'brain_region': brain_region,
                'hemisphere': hemisphere,
                'fiber_idx': int(rec['fiber_idx']) if 'fiber_idx' in rec.index else 0,
            }

            for event in RESPONSE_EVENTS:
                if event not in responses.coords['event'].values:
                    continue
                resp = responses.sel(
                    region=brain_region, event=event,
                ).values  # (n_trials, n_time)
                cache[(eid, brain_region, event)] = {
                    'traces': resp,
                    'tpts': tpts,
                    'meta': {**meta, 'event': event},
                    'trials': ps.trials.copy(),
                }

        self.response_traces = cache
        return self

    def flush_response_traces(self):
        """Free the per-trial trace cache to reclaim memory."""
        self.response_traces = None
        self.response_traces_tpts = None

    def get_response_magnitudes(self):
        """Compute trial-level response magnitudes from the trace cache.

        Calls :meth:`load_response_traces` if traces are not yet cached.
        For each cached (recording × event), computes scalar magnitudes in
        the early response window.

        Returns
        -------
        pd.DataFrame
            One row per (recording × event × trial) with scalar response
            magnitudes and trial metadata.
        """
        from tqdm import tqdm

        if self.response_traces is None:
            self.load_response_traces()

        response_rows = []
        timing_eids_seen = set()
        timing_rows = []
        for (eid, brain_region, event), entry in tqdm(
                self.response_traces.items(),
                desc="Computing response magnitudes"):
            traces = entry['traces']
            tpts = entry['tpts']
            meta = entry['meta']
            trials = entry['trials']
            n_trials = len(trials)

            early = compute_response_magnitude(
                traces, tpts, RESPONSE_WINDOWS['early'],
            )

            # Collect timing once per eid (same across regions and events)
            if eid not in timing_eids_seen:
                timing_eids_seen.add(eid)

                if ('firstMovement_times' in trials.columns
                        and 'stimOn_times' in trials.columns):
                    reaction_time = (
                        trials['firstMovement_times'].values
                        - trials['stimOn_times'].values
                    )
                else:
                    reaction_time = np.full(n_trials, np.nan)

                if ('feedback_times' in trials.columns
                        and 'firstMovement_times' in trials.columns):
                    movement_time = (
                        trials['feedback_times'].values
                        - trials['firstMovement_times'].values
                    )
                else:
                    movement_time = np.full(n_trials, np.nan)

                if ('feedback_times' in trials.columns
                        and 'stimOn_times' in trials.columns):
                    response_time = (
                        trials['feedback_times'].values
                        - trials['stimOn_times'].values
                    )
                else:
                    response_time = np.full(n_trials, np.nan)

                # FIXME: find a way to do this without looping... so inefficient
                for t in range(n_trials):
                    timing_rows.append({
                        'eid': eid,
                        'trial': t,
                        'reaction_time': reaction_time[t],
                        'movement_time': movement_time[t],
                        'response_time': response_time[t]
                    })

            for t in range(n_trials):
                response_rows.append({
                    'eid': meta['eid'],
                    'subject': meta['subject'],
                    'session_type': meta.get('session_type'),
                    'NM': meta.get('NM'),
                    'target_NM': meta['target_NM'],
                    'brain_region': meta['brain_region'],
                    'hemisphere': meta['hemisphere'],
                    'event': event,
                    'trial': t,
                    'stim_side': trials['stim_side'].iloc[t],
                    'signed_contrast': trials['signed_contrast'].iloc[t],
                    'contrast': trials['contrast'].iloc[t],
                    'choice': trials['choice'].iloc[t],
                    'feedbackType': trials['feedbackType'].iloc[t],
                    'probabilityLeft': trials['probabilityLeft'].iloc[t],
                    'response_early': early[t],
                })

        if not response_rows:
            self.response_magnitudes = pd.DataFrame()
            self.trial_timing = pd.DataFrame()
            return self.response_magnitudes

        self.response_magnitudes = pd.DataFrame(response_rows)
        self.trial_timing = pd.DataFrame(timing_rows)
        return self.response_magnitudes

    def get_mean_traces(self):
        """Compute trial-averaged traces from the trace cache.

        Calls :meth:`load_response_traces` if traces are not yet cached.
        For each cached (recording × event), groups trials by
        ``(contrast, feedbackType)`` and computes the mean trace per group.

        Returns
        -------
        pd.DataFrame
            Long-form DataFrame with columns: eid, subject, target_NM,
            brain_region, event, contrast, feedbackType, time, response.
        """
        if self.response_traces is None:
            self.load_response_traces()

        rows = []
        for (_eid, _region, _event), entry in self.response_traces.items():
            traces = entry['traces']  # (n_trials, n_time)
            tpts = entry['tpts']
            meta = entry['meta']
            trials = entry['trials']

            # Apply canonical trial filters
            if ('feedback_times' in trials.columns
                    and 'stimOn_times' in trials.columns):
                response_time = (trials['feedback_times'].values
                                 - trials['stimOn_times'].values)
            else:
                response_time = np.full(len(trials), np.nan)
            keep = (
                (trials['probabilityLeft'] == 0.5)
                & (trials['choice'] != 0)
                & (response_time > 0.05)
            )
            trials = trials[keep]

            for (contrast, fb), idx in trials.groupby(
                    ['contrast', 'feedbackType']).groups.items():
                trial_mask = idx.values
                if len(trial_mask) == 0:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    mean_trace = np.nanmean(traces[trial_mask], axis=0)
                n_trials = int(np.sum(~np.isnan(traces[trial_mask, 0])))

                for i, t in enumerate(tpts):
                    rows.append({
                        'eid': meta['eid'],
                        'subject': meta['subject'],
                        'target_NM': meta['target_NM'],
                        'brain_region': meta['brain_region'],
                        'fiber_idx': meta['fiber_idx'],
                        'event': meta['event'],
                        'contrast': contrast,
                        'feedbackType': fb,
                        'time': t,
                        'response': mean_trace[i],
                        'n_trials': n_trials,
                    })

        self.mean_traces = pd.DataFrame(rows)
        return self.mean_traces

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

        rows = {}
        has_fiber_idx = 'fiber_idx' in self.recordings.columns

        for rec, ps in self:
            eid = rec['eid']
            brain_region = rec['brain_region']
            hemisphere = rec['hemisphere']
            target_nm = rec['target_NM']
            fiber_idx = int(rec['fiber_idx']) if has_fiber_idx else 0

            # Load H5 if responses not yet available
            if not hasattr(ps, 'responses') or not hasattr(ps, 'trials'):
                h5_path = Path(self.h5_dir) / f'{eid}.h5'
                if not h5_path.exists():
                    print(f"  H5 file not found: {h5_path}")
                    continue
                ps.load_h5(h5_path, groups=['trials', 'responses'])

            if not hasattr(ps, 'responses') or not hasattr(ps, 'trials'):
                continue
            if brain_region not in ps.responses.coords['region'].values:
                continue

            vec = ps.get_response_vector(
                brain_region=brain_region, hemisphere=hemisphere, **kwargs,
            )
            rows[(eid, target_nm, fiber_idx)] = vec

            # Discard raw data to free memory
            del ps.responses
            del ps.trials

        if not rows:
            self.response_features = pd.DataFrame()
            return self.response_features

        df = pd.DataFrame(rows).T
        df.index = pd.MultiIndex.from_tuples(
            df.index, names=['eid', 'target_NM', 'fiber_idx'],
        )

        if nan_handling == 'drop_sessions':
            df = df.dropna()
        elif nan_handling == 'drop_features':
            nan_rate = df.isna().mean()
            df = df.loc[:, nan_rate <= nan_threshold]

        self.response_features = df
        return df

    def get_glm_response_features(self, event_name='stimOn_times',
                                   weight_by_se=False, min_trials=20,
                                   contrast_coding='log'):
        """Build GLM coefficient features for each recording.

        Fits a per-recording OLS model on trial-level responses and uses the
        regression coefficients as a compact feature vector.

        Parameters
        ----------
        event_name : str
            Event to model (default ``'stimOn_times'``).
        weight_by_se : bool
            If True, return t-statistics (coef / SE) instead of raw
            coefficients.
        min_trials : int
            Minimum valid trials per recording.

        Returns
        -------
        pd.DataFrame
            (n_recordings, 7) indexed by ``(eid, target_NM, fiber_idx)``.
        """
        from iblnm.analysis import fit_response_glm

        if self.response_magnitudes is None:
            self.get_response_magnitudes()

        # FIXME: this method should iterate over sessions, filter trials and select the event type,  not the underlying function
        events = self.response_magnitudes.copy()
        if self.trial_timing is not None:
            events = events.merge(
                self.trial_timing[['eid', 'trial', 'response_time']],
                on=['eid', 'trial'], how='left',
            )
            events = events.query('choice != 0 and response_time > 0.05')
        else:
            events = events.query('choice != 0')

        if len(events) == 0:
            self.glm_response_features = pd.DataFrame()
            return pd.DataFrame()

        coefs, ses = fit_response_glm(events, event_name,
                                       min_trials=min_trials,
                                       contrast_coding=contrast_coding)

        if len(coefs) == 0:
            self.glm_response_features = coefs
            return coefs

        if weight_by_se:
            result = coefs / ses
        else:
            result = coefs

        # Reindex to (eid, target_NM, fiber_idx) dropping brain_region level
        result = result.droplevel('brain_region')
        self.glm_response_features = result
        return result

    def pca_glm_coefficients(self, event_name='stimOn_times',
                             contrast_coding='log', n_components=3,
                             min_trials=20, cohort_weighted=False):
        """Fit per-session GLM and run PCA on coefficients.

        Parameters
        ----------
        event_name : str
            Event to model.
        contrast_coding : str
            Contrast transform for the GLM ('log', 'linear', 'rank').
        n_components : int
            Number of PCs to retain.
        min_trials : int
            Minimum trials per recording for GLM fitting.
        cohort_weighted : bool
            If True, weight by 1/n_k so each target contributes equally.

        Returns
        -------
        GLMPCAResult
        """
        from iblnm.analysis import pca_glm_coefficients

        self.get_glm_response_features(
            event_name=event_name, min_trials=min_trials,
            contrast_coding=contrast_coding,
        )
        if self.glm_response_features is None or len(self.glm_response_features) == 0:
            return None

        result = pca_glm_coefficients(
            self.glm_response_features, n_components=n_components,
            cohort_weighted=cohort_weighted)
        self.glm_pca_result = result
        return result

    def ica_glm_coefficients(self, event_name='stimOn_times',
                             contrast_coding='log', n_components=3,
                             min_trials=20, cohort_weighted=False):
        """Fit per-session GLM and run ICA on coefficients.

        Parameters
        ----------
        event_name : str
            Event to model.
        contrast_coding : str
            Contrast transform for the GLM ('log', 'linear', 'rank').
        n_components : int
            Number of independent components to extract.
        min_trials : int
            Minimum trials per recording for GLM fitting.
        cohort_weighted : bool
            If True, weight by 1/n_k so each target contributes equally.

        Returns
        -------
        GLMPCAResult
        """
        from iblnm.analysis import ica_glm_coefficients

        self.get_glm_response_features(
            event_name=event_name, min_trials=min_trials,
            contrast_coding=contrast_coding,
        )
        if self.glm_response_features is None or len(self.glm_response_features) == 0:
            return None

        result = ica_glm_coefficients(
            self.glm_response_features, n_components=n_components,
            cohort_weighted=cohort_weighted)
        self.glm_pca_result = result
        return result

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

        Creates a ``TargetNMDecoder``, fits it with leave-one-subject-out CV,
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

        recs = self.recordings.copy()
        if 'fiber_idx' not in recs.columns:
            recs['fiber_idx'] = 0
        idx_cols = ['eid', 'target_NM', 'fiber_idx']
        rec_indexed = (
            recs[idx_cols + ['subject']]
            .drop_duplicates(subset=idx_cols)
            .set_index(idx_cols)
        )
        subjects = rec_indexed['subject'].reindex(self.response_features.index)

        self.decoder = TargetNMDecoder(self.response_features, labels, subjects)
        self.decoder.fit()
        self.decoder.unique_contribution()
        return self.decoder

    def get_psychometric_features(self, performance_path=None, params=None):
        """Build psychometric parameter matrix aligned to response_features.

        Parameters
        ----------
        performance_path : Path or str, optional
            Path to performance.pqt. Default: config.PERFORMANCE_FPATH.
        params : list of str, optional
            Columns to include from performance data. Default:
            ``['psych_50_threshold', 'psych_50_bias',
            'psych_50_lapse_left', 'psych_50_lapse_right']``.

        Returns
        -------
        pd.DataFrame
            (n_recordings, P) aligned to ``self.response_features`` index.
        """
        from iblnm.config import PERFORMANCE_FPATH

        if performance_path is None:
            performance_path = PERFORMANCE_FPATH
        if params is None:
            params = [
                'psych_50_threshold', 'psych_50_bias',
                'psych_50_lapse_left', 'psych_50_lapse_right',
            ]

        perf = pd.read_parquet(performance_path)

        # Extract eid from response_features index and merge
        rf_index = self.response_features.index
        eids = rf_index.get_level_values('eid')
        lookup = perf.set_index('eid')[params]

        # Build aligned DataFrame: one row per recording, matching rf_index
        psych = lookup.reindex(eids)
        psych.index = rf_index

        self.psychometric_features = psych
        return psych

    def fit_cca(self, n_components=None, n_permutations=1000, seed=42,
                **kwargs):
        """Fit CCA between response features and psychometric parameters.

        Parameters
        ----------
        n_components : int or None
            Number of canonical variates. Default: min(K, P, n).
        n_permutations : int
            Permutation test iterations. Default 1000.
        seed : int
            RNG seed. Default 42.
        **kwargs
            Forwarded to ``get_response_features`` /
            ``get_psychometric_features`` if not yet computed.

        Returns
        -------
        CCAResult
        """
        from iblnm.analysis import fit_cca

        if self.response_features is None:
            self.get_response_features(**kwargs)
        if self.psychometric_features is None:
            self.get_psychometric_features(**kwargs)

        X = self.response_features
        Y = self.psychometric_features

        # Align on shared index
        shared = X.index.intersection(Y.index)
        X = X.loc[shared]
        Y = Y.loc[shared]

        session_labels = pd.Series(
            shared.get_level_values('eid'), index=shared,
        )

        self.cca_result = fit_cca(
            X, Y,
            n_components=n_components,
            n_permutations=n_permutations,
            session_labels=session_labels,
            seed=seed,
        )
        return self.cca_result

    def fit_cohort_cca(self, n_permutations=1000, seed=42,
                       min_recordings=10, exclude_intercept=True,
                       sparse=False, alpha=0.01, l1_ratio=0.0,
                       unit_norm=True):
        """Fit CCA separately per target-NM cohort.

        Parameters
        ----------
        n_permutations : int
            Permutation test iterations per cohort.
        seed : int
            RNG seed.
        min_recordings : int
            Minimum recordings to include a cohort.
        exclude_intercept : bool
            If True, drop the ``intercept`` column from neural features.
        sparse : bool
            If True, use sparse CCA (cca-zoo ElasticCCA) instead of sklearn CCA.
        alpha : float or list[float]
            Regularization strength for sparse CCA. When a list,
            grid-searched. Ignored when ``sparse=False``.
        l1_ratio : float or list[float]
            L1/L2 mixing ratio for sparse CCA. When a list,
            grid-searched. Ignored when ``sparse=False``.

        Returns
        -------
        dict[str, CCAResult]
        """
        from sklearn.preprocessing import StandardScaler
        from iblnm.analysis import fit_cca
        if sparse:
            from iblnm.analysis import fit_sparse_cca

        if self.glm_response_features is None:
            raise ValueError("glm_response_features is None")
        if self.psychometric_features is None:
            raise ValueError("psychometric_features is None")

        X = self.glm_response_features.copy()
        Y = self.psychometric_features.copy()

        if exclude_intercept and 'intercept' in X.columns:
            X = X.drop(columns=['intercept'])

        # Align on shared index
        shared = X.index.intersection(Y.index)
        X = X.loc[shared]
        Y = Y.loc[shared]

        # Group by target_NM
        target_nms = X.index.get_level_values('target_NM')

        results = {}
        data = {}

        from tqdm import tqdm
        for tnm in tqdm(target_nms.unique(), desc='Fitting CCA per cohort'):
            mask = target_nms == tnm
            X_cohort = X.loc[mask]
            Y_cohort = Y.loc[mask]

            # Drop NaN rows
            valid = X_cohort.notna().all(axis=1) & Y_cohort.notna().all(axis=1)
            X_cohort = X_cohort.loc[valid]
            Y_cohort = Y_cohort.loc[valid]

            if len(X_cohort) < min_recordings:
                continue

            # Drop constant Y columns
            y_std = Y_cohort.std()
            varying = y_std[y_std > 0].index
            if len(varying) == 0:
                continue
            Y_cohort = Y_cohort[varying]

            # Standardize
            x_scaler = StandardScaler()
            y_scaler = StandardScaler()
            X_z = x_scaler.fit_transform(X_cohort.values)
            Y_z = y_scaler.fit_transform(Y_cohort.values)

            X_z_df = pd.DataFrame(X_z, columns=X_cohort.columns,
                                  index=X_cohort.index)
            Y_z_df = pd.DataFrame(Y_z, columns=Y_cohort.columns,
                                  index=Y_cohort.index)

            session_labels = pd.Series(
                X_cohort.index.get_level_values('eid'),
                index=X_cohort.index,
            )

            cca_func = fit_sparse_cca if sparse else fit_cca
            cca_kwargs = dict(
                n_components=1,
                n_permutations=n_permutations,
                session_labels=session_labels,
                seed=seed,
                scale=False,
            )
            if sparse:
                cca_kwargs['alpha'] = alpha
                cca_kwargs['l1_ratio'] = l1_ratio
                cca_kwargs['unit_norm'] = unit_norm
            result = cca_func(X_z_df, Y_z_df, **cca_kwargs)
            results[tnm] = result
            data[tnm] = (X_z, Y_z)

        if not results:
            raise ValueError("No cohort has enough recordings")

        # Align signs across cohorts for consistent comparison
        from iblnm.analysis import align_cca_signs
        results = align_cca_signs(results)

        self.cohort_cca_results = results
        self.cohort_cca_data = data
        return results

    def cross_project_cca(self, cohorts=None):
        """Cross-project each cohort's data through every other's CCA weights.

        Parameters
        ----------
        cohorts : list of str, optional
            Subset of cohort keys. Default: all fitted cohorts.

        Returns
        -------
        pd.DataFrame
            Columns: ``data_cohort``, ``weight_cohort``, ``correlation``.
        """
        from iblnm.analysis import cross_project_cca as _cross_project

        if self.cohort_cca_results is None:
            raise ValueError("Call fit_cohort_cca first")

        if cohorts is None:
            cohorts = list(self.cohort_cca_results.keys())

        rows = []
        for data_cohort in cohorts:
            X_z, Y_z = self.cohort_cca_data[data_cohort]
            for weight_cohort in cohorts:
                target_result = self.cohort_cca_results[weight_cohort]
                r = _cross_project(X_z, Y_z, target_result)
                rows.append({
                    'data_cohort': data_cohort,
                    'weight_cohort': weight_cohort,
                    'correlation': r,
                })

        df = pd.DataFrame(rows)
        self.cohort_cca_cross_projections = df
        return df

    def compare_cca_weights(self, cohorts=None):
        """Cosine similarity between CC1 weights for all cohort pairs.

        Parameters
        ----------
        cohorts : list of str, optional
            Subset of cohort keys. Default: all fitted cohorts.

        Returns
        -------
        pd.DataFrame
            Columns: ``cohort_a``, ``cohort_b``, ``neural_cosine``,
            ``behavioral_cosine``.
        """
        from iblnm.analysis import compare_cca_weights as _compare_weights

        if self.cohort_cca_results is None:
            raise ValueError("Call fit_cohort_cca first")

        if cohorts is None:
            cohorts = list(self.cohort_cca_results.keys())

        rows = []
        for a in cohorts:
            for b in cohorts:
                sims = _compare_weights(
                    self.cohort_cca_results[a],
                    self.cohort_cca_results[b],
                )
                rows.append({
                    'cohort_a': a,
                    'cohort_b': b,
                    **sims,
                })

        df = pd.DataFrame(rows)
        self.cohort_cca_weight_similarities = df
        return df

    def fit_lmm(self, response_col='response_early',
                 min_subjects=2, re_formulas=None, contrast_coding='log'):
        """Fit LMMs per (target_NM, event) on trial-level events data.

        For each group, fits: response ~ log(contrast) * side * reward | subject.
        Computes estimated marginal means and contrast slopes for each fit.

        When ``re_formulas`` contains multiple entries (ordered most complex to
        simplest), the method selects the most complex formula that converges
        for **all** groups and refits everyone with that formula.

        Requires ``self.response_magnitudes`` to be populated (via
        ``get_response_magnitudes()`` or direct assignment).

        Parameters
        ----------
        response_col : str
            Column name for the response magnitude.
        min_subjects : int
            Minimum subjects per group to attempt fitting.
        re_formulas : list of str, optional
            Random-effects formulas to try, ordered from most complex to
            simplest. Default ``['1']`` (random intercept only).

        Returns
        -------
        dict
            Keys: (target_NM, event_label) tuples.
            Values: LMMResult objects with emm_reward, emm_side,
            emm_contrast, and contrast_slopes populated.
        """
        # CHECK: why imports inside functions? isn't this poor form?
        from tqdm import tqdm
        from iblnm.analysis import (
            fit_response_lmm, compute_marginal_means,
            compute_contrast_slopes, compute_interaction_effects,
        )
        from iblnm.task import add_relative_contrast

        if re_formulas is None:
            re_formulas = ['1']

        if self.response_magnitudes is None:
            raise ValueError(
                "response_magnitudes not populated. Call get_response_magnitudes() first."
            )
        # ~ if self.trial_timing is None:
            # ~ raise ValueError(
                # ~ "trial_timing not populated. Call get_response_magnitudes() first."
            # ~ )

        if self.trial_timing is None:
            raise ValueError(
                "trial_timing not populated. Call get_response_magnitudes() first."
            )

        # FIXME: do this once insead of several different places (see
        # plotting in responses.py)
        df = add_relative_contrast(self.response_magnitudes.copy())
        df = df.merge(
            self.trial_timing[['eid', 'trial', 'response_time']],
            on=['eid', 'trial'], how='left',
        )
        df = df.query('probabilityLeft == 0.5')
        df = df.dropna(subset=[response_col])
        df = df.query('choice != 0 and response_time > 0.05')

        # Identify valid groups
        groups = {}
        for (target_nm, event), df_group in df.groupby(['target_NM', 'event']):
            if df_group['subject'].nunique() < min_subjects:
                continue
            event_label = event.replace('_times', '')
            groups[(target_nm, event_label)] = df_group

        if not groups:
            self.lmm_results = {}
            self.lmm_coefficients = pd.DataFrame()
            self.lmm_re_formula = re_formulas[-1]
            return self.lmm_results

        # Select the most complex RE formula that converges for all groups.
        # Cache fits to avoid refitting with the selected formula.
        selected_re = None
        cached_fits = None
        for re_formula in re_formulas:
            fits = {
                key: fit_response_lmm(df_g, response_col, re_formula=re_formula,
                                      contrast_coding=contrast_coding)
                for key, df_g in tqdm(groups.items(),
                                      desc=f"Fitting LMMs (re={re_formula})")
            }
            if all(lmm is not None for lmm in fits.values()):
                selected_re = re_formula
                cached_fits = fits
                break

        if selected_re is None:
            self.lmm_results = {}
            self.lmm_coefficients = pd.DataFrame()
            self.lmm_re_formula = None
            return self.lmm_results

        self.lmm_re_formula = selected_re

        results = {}
        all_summaries = []

        for (target_nm, event_label), lmm in cached_fits.items():
            if lmm is None:
                continue

            lmm.emm_reward = compute_marginal_means(lmm, 'reward')
            lmm.emm_side = compute_marginal_means(lmm, 'side')
            lmm.emm_contrast = compute_marginal_means(lmm, 'contrast')
            lmm.contrast_slopes = compute_contrast_slopes(lmm)
            lmm.interaction_contrast_reward = compute_interaction_effects(
                lmm, 'contrast', 'reward')
            lmm.interaction_contrast_side = compute_interaction_effects(
                lmm, 'contrast', 'side')
            lmm.interaction_reward_side = compute_interaction_effects(
                lmm, 'reward', 'side')

            results[(target_nm, event_label)] = lmm

            summary = lmm.summary_df.copy()
            summary.insert(0, 'target_NM', target_nm)
            summary.insert(1, 'event', event_label)
            summary.index.name = 'term'
            all_summaries.append(summary.reset_index())

        self.lmm_results = results

        if all_summaries:
            self.lmm_coefficients = pd.concat(all_summaries, ignore_index=True)
        else:
            self.lmm_coefficients = pd.DataFrame()

        return results

    def enrich_peak_velocity(self):
        """Extract peak wheel velocity from H5 files.

        Loads per-trial wheel velocity from H5 and computes the maximum
        absolute velocity for each trial. Stores result in
        ``self.peak_velocity``.

        Requires ``self.response_magnitudes`` to be populated (uses its
        eid/trial index to know which trials to extract).

        Returns
        -------
        self
        """
        import h5py
        from pathlib import Path
        from tqdm import tqdm

        if self.response_magnitudes is None:
            raise ValueError(
                "response_magnitudes not populated. "
                "Call get_response_magnitudes() first."
            )

        df = self.response_magnitudes
        # Get unique (eid, trial) pairs
        trial_keys = df[['eid', 'trial']].drop_duplicates()

        rows = []
        for eid in tqdm(df['eid'].unique(), desc="Enriching peak velocity"):
            h5_path = Path(self.h5_dir) / f'{eid}.h5'
            wheel_vel = None

            if h5_path.exists():
                with h5py.File(h5_path, 'r') as f:
                    if 'wheel' in f:
                        wheel_vel = f['wheel/velocity'][:]

            eid_trials = trial_keys[trial_keys['eid'] == eid]['trial']
            for trial in eid_trials:
                pv = np.nan
                if wheel_vel is not None and trial < len(wheel_vel):
                    trial_vel = wheel_vel[trial]
                    valid = trial_vel[~np.isnan(trial_vel)]
                    if len(valid) > 0:
                        pv = float(np.max(np.abs(valid)))
                rows.append({'eid': eid, 'trial': trial, 'peak_velocity': pv})

        self.peak_velocity = pd.DataFrame(rows)
        return self

    def fit_wheel_lmm(self, response_col='response_early', min_subjects=2):
        """Fit nested LMMs for wheel kinematics predicted by NM activity.

        For each (target_NM, contrast_level, dv), fits:
          Base:  ``dv ~ C(stim_side) * C(choice) + (1 | subject)``
          Full:  ``dv ~ C(stim_side) * C(choice) + response_early + (1|subject)``

        Requires ``response_magnitudes``, ``trial_timing``, and
        ``peak_velocity`` to be populated.

        Parameters
        ----------
        response_col : str
            Column name for NM response magnitude.
        min_subjects : int
            Minimum subjects per group.

        Returns
        -------
        dict
            Keys: ``(target_NM, contrast, dv_name)`` tuples.
            Values: dict with comparison results.
        """
        from iblnm.analysis import fit_wheel_lmm as _fit_wheel_lmm
        from tqdm import tqdm

        if self.response_magnitudes is None:
            raise ValueError(
                "response_magnitudes not populated. "
                "Call get_response_magnitudes() first."
            )
        if self.trial_timing is None:
            raise ValueError(
                "trial_timing not populated. "
                "Call get_response_magnitudes() first."
            )
        if self.peak_velocity is None:
            raise ValueError(
                "peak_velocity not populated. "
                "Call enrich_peak_velocity() first."
            )

        dvs = ['reaction_time', 'movement_time', 'peak_velocity']
        df = self.response_magnitudes.copy()
        timing_cols = ['eid', 'trial'] + [
            c for c in ['reaction_time', 'movement_time']
            if c in self.trial_timing.columns
        ]
        df = df.merge(
            self.trial_timing[timing_cols],
            on=['eid', 'trial'], how='left',
        )
        df = df.merge(
            self.peak_velocity[['eid', 'trial', 'peak_velocity']],
            on=['eid', 'trial'], how='left',
        )

        # Filter to stimOn event, unbiased blocks, valid trials
        df = df[df['event'] == 'stimOn_times']
        df = df[df['probabilityLeft'] == 0.5]
        df = df[df['choice'] != 0]
        df = df.dropna(subset=[response_col])

        # Lateralize stim_side relative to hemisphere
        # stim_side is already 'left'/'right' — convert to contra/ipsi
        if 'hemisphere' in df.columns:
            contra_map = {'l': 'right', 'r': 'left'}
            df['stim_side_lateral'] = df.apply(
                lambda row: 'contra' if row['stim_side'] == contra_map.get(
                    row['hemisphere'], '') else 'ipsi',
                axis=1,
            )
        else:
            df['stim_side_lateral'] = df['stim_side']
        df['stim_side'] = df['stim_side_lateral']

        # Build list of (target_nm, contrast, dv, df_dv) jobs
        jobs = []
        for target_nm in df['target_NM'].unique():
            df_target = df[df['target_NM'] == target_nm]
            for contrast in sorted(df_target['contrast'].unique()):
                df_c = df_target[np.isclose(df_target['contrast'], contrast)]
                if df_c['subject'].nunique() < min_subjects:
                    continue
                for dv in dvs:
                    if dv not in df_c.columns:
                        continue
                    df_dv = df_c.dropna(subset=[dv])
                    if len(df_dv) < 10:
                        continue
                    jobs.append((target_nm, contrast, dv, df_dv))

        results = {}
        summary_rows = []

        for target_nm, contrast, dv, df_dv in tqdm(
                jobs, desc="Fitting wheel LMMs"):
                    result = _fit_wheel_lmm(
                        df_dv, dv_col=dv, response_col=response_col,
                        target_nm=target_nm, contrast=contrast,
                        min_subjects=min_subjects,
                    )

                    if result is not None:
                        key = (target_nm, contrast, dv)
                        results[key] = result
                        summary_rows.append({
                            'target_NM': target_nm,
                            'contrast': contrast,
                            'dv': dv,
                            **{k: result[k] for k in [
                                'delta_r2', 'base_r2_marginal',
                                'full_r2_marginal', 'lrt_chi2', 'lrt_pvalue',
                                'nm_coefficient', 'nm_pvalue', 'n_trials',
                                'n_subjects',
                            ]},
                        })

        self.wheel_lmm_results = results
        self.wheel_lmm_summary = (
            pd.DataFrame(summary_rows) if summary_rows
            else pd.DataFrame(columns=[
                'target_NM', 'contrast', 'dv', 'delta_r2',
                'lrt_pvalue', 'nm_coefficient', 'nm_pvalue',
                'n_trials', 'n_subjects',
            ])
        )
        return results

