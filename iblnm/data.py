import warnings
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import xarray as xr

#from brainbox.io.one import PhotometrySessionLoader
from iblphotometry.fpio import PhotometrySessionLoader, from_neurophotometrics_df_to_photometry_df
from iblphotometry import metrics
from iblphotometry.qc import qc_signals

from one.alf.exceptions import ALFObjectNotFound

from iblnm.config import (
    ANALYSIS_QC_BLOCKERS, BASELINE_WINDOW, EIDS_TO_DROP,
    EVENT_COMPLETENESS_THRESHOLD,
    LP_QC_LABELS,
    MIN_NTRIALS, MIN_PERFORMANCE, MIN_TRIALS_PERSESSION, MOTION_ENERGY_EVENT,
    N_UNIQUE_SAMPLES_THRESHOLD,
    POSE_MEASURES,
    PREPROCESSING_PIPELINES, QC_METRICS_KWARGS, QC_RAW_METRICS,
    QC_SLIDING_KWARGS, QC_SLIDING_METRICS, REQUIRED_CONTRASTS,
    RESPONSE_EVENTS, RESPONSE_WINDOW, RESPONSE_WINDOWS, SESSIONS_H5_DIR,
    SESSION_TYPES_TO_ANALYZE, SUBJECTS_TO_EXCLUDE, TARGETNMS_TO_ANALYZE,
    TARGET_FS, TRIAL_COLUMNS, VIDEO_QC_COLS, WHEEL_FS, POSE_FS,
)
from iblnm.analysis import (
    get_responses, compute_response_magnitude, movement_trace,
    per_third_crosscorr, resample_pose, resample_signal,
)
from iblnm import analysis
from iblnm import task
from iblnm.task import compute_trial_contrasts
from iblnm.validation import (
    MissingExtractedData, MissingRawData, MissingLP, MissingVideoTimestamps,
    MissingMotionEnergy,
    InsufficientTrials, BlockStructureBug, MissingBlockInfo,
    IncompleteEventTimes, TrialsNotInPhotometryTime, FewUniqueSamples,
    QCValidationError, AmbiguousRegionMapping,
)

# Long-form schema returned by per-recording drop-one OLS ΔR² (one row per
# event × dropped predictor for a single recording).
PERSESSION_DROPONE_COLUMNS = [
    'brain_region', 'target_NM', 'event', 'predictor', 'r2', 'delta_r2',
    'n_trials',
]

# Group-level long-form drop-one frame: per-recording rows tagged with their
# eid/subject. target_NM precedes brain_region (the recording identity order).
RESPONSE_OLS_DROPONE_COLUMNS = [
    'eid', 'subject', 'target_NM', 'brain_region', 'event', 'predictor', 'r2',
    'delta_r2', 'n_trials',
]


# =============================================================================
# HDF5 save/load helpers
# =============================================================================
#
# These module-level functions handle the on-disk layout for each top-level
# group. `save_h5` / `load_h5` on PhotometrySession are thin dispatchers over
# the _SAVE_HANDLERS / _LOAD_HANDLERS registries at the bottom of this block.
#
# Photometry sub-handlers (_save_preprocessed, _save_responses, _save_qc and
# their load counterparts) are pure: they take a parent group and a payload,
# with no coupling to PhotometrySession. The region loop and session→payload
# extraction live in the _save_photometry / _load_photometry orchestrators.


def _replace_group(parent, name):
    """Delete `name` under `parent` if present, create and return a fresh group."""
    if name in parent:
        del parent[name]
    return parent.create_group(name)


def _write_dataframe(h5_group, dataframe):
    """Write each column of `dataframe` as a dataset under `h5_group`."""
    for col in dataframe.columns:
        # .to_numpy() (not .values) collapses pandas extension arrays — e.g. the
        # string dtype that is default in pandas 3.0 — to numpy object, so the
        # bytes encoding below catches them instead of handing h5py a non-native
        # dtype.
        values = dataframe[col].to_numpy()
        if values.dtype == object:
            values = values.astype('S')
        h5_group.create_dataset(col, data=values)


def _read_dataframe(h5_group):
    """Read all datasets under `h5_group` into a DataFrame, decoding bytes."""
    data = {}
    for col in h5_group:
        values = h5_group[col][:]
        if values.dtype.kind == 'S':
            values = values.astype(str)
        data[col] = values
    return pd.DataFrame(data)


_METADATA_NONE_SENTINEL = '__none__'
_ERROR_FIELDS = ('eid', 'error_type', 'error_message', 'traceback')
_RESPONSES_RESERVED_KEYS = {'times', 'trials'}


def _save_metadata(session, h5_file, band):
    grp = _replace_group(h5_file, 'metadata')
    for attr, is_list in session._METADATA_FIELDS:
        value = getattr(session, attr, None)
        if is_list:
            items = list(value) if value else []
            grp.create_dataset(
                attr,
                data=[s.encode() if isinstance(s, str) else s for s in items],
                dtype=h5py.string_dtype(),
            )
        else:
            if attr == 'start_time' and hasattr(value, 'isoformat'):
                value = value.isoformat()
            grp.attrs[attr] = _METADATA_NONE_SENTINEL if value is None else value


def _load_metadata(session, h5_file, band):
    if 'metadata' not in h5_file:
        return
    grp = h5_file['metadata']
    for attr, is_list in session._METADATA_FIELDS:
        if is_list:
            if attr in grp:
                setattr(session, attr, [
                    v.decode() if isinstance(v, bytes) else v
                    for v in grp[attr][:]
                ])
            continue
        if attr not in grp.attrs:
            continue
        value = grp.attrs[attr]
        if isinstance(value, bytes):
            value = value.decode()
        elif hasattr(value, 'item'):
            value = value.item()
        if isinstance(value, str) and value == _METADATA_NONE_SENTINEL:
            value = None
        if attr == 'start_time' and isinstance(value, str):
            value = datetime.fromisoformat(value)
        setattr(session, attr, value)


def _read_existing_errors(h5_file):
    if 'errors' not in h5_file or not h5_file['errors'].keys():
        return []
    err_grp = h5_file['errors']
    n = len(err_grp['eid'])
    return [
        {
            col: (err_grp[col][i].decode()
                  if isinstance(err_grp[col][i], bytes)
                  else str(err_grp[col][i]))
            for col in _ERROR_FIELDS
        }
        for i in range(n)
    ]


def _dedup_errors(errors):
    seen = set()
    unique = []
    for entry in errors:
        key = (entry.get('eid', ''), entry.get('error_type', ''),
               entry.get('error_message', ''))
        if key not in seen:
            seen.add(key)
            unique.append(entry)
    return unique


def _save_errors(session, h5_file, band):
    merged = _dedup_errors(_read_existing_errors(h5_file) + session.errors)
    grp = _replace_group(h5_file, 'errors')
    # Empty group signals "no errors" — distinguishable from "not yet written".
    if not merged:
        return
    for col in _ERROR_FIELDS:
        grp.create_dataset(
            col,
            data=[str(e.get(col, '') or '') for e in merged],
            dtype=h5py.string_dtype(),
        )


def _load_errors(session, h5_file, band):
    if 'errors' not in h5_file:
        return
    session.errors = _read_existing_errors(h5_file)


def _save_trials(session, h5_file, band):
    if getattr(session, 'trials', None) is None:
        return
    columns = [c for c in TRIAL_COLUMNS + ['contrast', 'signed_contrast']
               if c in session.trials.columns]
    _write_dataframe(_replace_group(h5_file, 'trials'), session.trials[columns])


def _load_trials(session, h5_file, band):
    if 'trials' not in h5_file:
        return
    session.trials = _read_dataframe(h5_file['trials'])


def _save_wheel(session, h5_file, band):
    if getattr(session, 'wheel_velocity', None) is None:
        return
    wheel_group = h5_file.require_group('wheel')
    responses_group = _replace_group(wheel_group, 'responses')
    responses_group.create_dataset(
        'velocity', data=session.wheel_velocity,
        compression='gzip', compression_opts=4,
    )
    responses_group.attrs['fs'] = session.wheel_fs
    responses_group.attrs['t0_event'] = getattr(
        session, '_wheel_t0_event', 'stimOn_times',
    )
    responses_group.attrs['t1_event'] = getattr(
        session, '_wheel_t1_event', 'feedback_times',
    )


def _load_wheel(session, h5_file, band):
    if 'wheel' not in h5_file or 'responses' not in h5_file['wheel']:
        return
    responses_group = h5_file['wheel/responses']
    session.wheel_velocity = responses_group['velocity'][:].astype(np.float32)
    session.wheel_fs = responses_group.attrs['fs']
    session._wheel_t0_event = responses_group.attrs['t0_event']
    session._wheel_t1_event = responses_group.attrs['t1_event']


# ----- Photometry sub-handlers (pure: parent_group + payload only) -----

def _save_preprocessed(parent_group, signal_series):
    """Write preprocessed/ subgroup from a time-indexed Series."""
    pp_group = _replace_group(parent_group, 'preprocessed')
    pp_group.attrs['fs'] = TARGET_FS
    pp_group.create_dataset('times', data=signal_series.index.values)
    pp_group.create_dataset(
        'signal',
        data=signal_series.values.astype(np.float64),
        compression='gzip', compression_opts=4,
    )


def _load_preprocessed(parent_group):
    """Read preprocessed/ subgroup into a time-indexed Series, or None."""
    if 'preprocessed' not in parent_group:
        return None
    pp_group = parent_group['preprocessed']
    return pd.Series(
        pp_group['signal'][:].astype(np.float64),
        index=pp_group['times'][:],
    )


def _save_responses(parent_group, responses, response_window):
    """Write responses/ subgroup from a DataArray(event, trial, time)."""
    responses_group = _replace_group(parent_group, 'responses')
    responses_group.attrs['fs'] = TARGET_FS
    responses_group.attrs['response_window'] = response_window
    responses_group.create_dataset(
        'times', data=responses.coords['time'].values,
    )
    responses_group.create_dataset(
        'trials', data=responses.coords['trial'].values,
    )
    for event_name in responses.coords['event'].values:
        responses_group.create_dataset(
            event_name,
            data=responses.sel(event=event_name).values.astype(np.float64),
            compression='gzip', compression_opts=4,
        )


def _load_responses(parent_group):
    """Read responses/ subgroup into a DataArray(event, trial, time), or None."""
    if 'responses' not in parent_group:
        return None
    responses_group = parent_group['responses']
    event_names = [k for k in responses_group.keys()
                   if k not in _RESPONSES_RESERVED_KEYS]
    return xr.DataArray(
        np.stack([
            responses_group[name][:].astype(np.float64)
            for name in event_names
        ]),
        dims=['event', 'trial', 'time'],
        coords={
            'event': event_names,
            'trial': responses_group['trials'][:],
            'time':  responses_group['times'][:],
        },
    )


def _save_qc(parent_group, qc_rows):
    """Write qc/ subgroup from a DataFrame."""
    _write_dataframe(_replace_group(parent_group, 'qc'), qc_rows)


def _load_qc(parent_group):
    """Read qc/ subgroup into a DataFrame, or None."""
    if 'qc' not in parent_group:
        return None
    return _read_dataframe(parent_group['qc'])


def _save_photometry(session, h5_file, band):
    photometry_group = h5_file.require_group('photometry')
    preprocessed = session.photometry.get(band)
    has_qc = (getattr(session, 'qc', None) is not None
              and len(session.qc) > 0)

    regions = set()
    if preprocessed is not None:
        regions.update(preprocessed.columns)
    regions.update(session.responses.keys())
    if has_qc:
        regions.update(session.qc['brain_region'].unique())

    for region in sorted(regions):
        region_group = photometry_group.require_group(region)

        if preprocessed is not None and region in preprocessed.columns:
            _save_preprocessed(region_group, preprocessed[region])

        if region in session.responses:
            _save_responses(
                region_group, session.responses[region], session.RESPONSE_WINDOW,
            )

        if has_qc:
            qc_rows = session.qc[session.qc['brain_region'] == region]
            if len(qc_rows) > 0:
                _save_qc(region_group, qc_rows)


def _load_photometry(session, h5_file, band):
    if 'photometry' not in h5_file:
        return
    photometry_group = h5_file['photometry']
    regions = sorted(photometry_group.keys())

    preprocessed_by_region = {
        region: series for region in regions
        if (series := _load_preprocessed(photometry_group[region])) is not None
    }
    if preprocessed_by_region:
        session.photometry[band] = pd.DataFrame(preprocessed_by_region)

    session.responses = {
        region: region_responses for region in regions
        if (region_responses := _load_responses(photometry_group[region])) is not None
    }

    qc_frames = [
        qc_frame for region in regions
        if (qc_frame := _load_qc(photometry_group[region])) is not None
    ]
    if qc_frames:
        session.qc = pd.concat(qc_frames, ignore_index=True)


# ----- Video / LightningPose sub-handlers (pure: parent_group + payload) -----

LP_QC_NOT_SET = 'NOT_SET'
_POSE_TRACES_RESERVED_KEYS = {'times', 'trials'}


def _save_pose_traces(parent_group, traces, name='traces'):
    """Write a `name`/ subgroup from a DataArray(bodypart, trial, time).

    `name` selects the subgroup: ``traces`` for the event-locked response
    traces, ``baseline_traces`` for the stimOn-locked baseline traces.
    """
    traces_group = _replace_group(parent_group, name)
    traces_group.attrs['response_window'] = RESPONSE_WINDOW
    traces_group.create_dataset('times', data=traces.coords['time'].values)
    traces_group.create_dataset('trials', data=traces.coords['trial'].values)
    for bodypart in traces.coords['bodypart'].values:
        traces_group.create_dataset(
            bodypart,
            data=traces.sel(bodypart=bodypart).values.astype(np.float64),
            compression='gzip', compression_opts=4,
        )


def _load_pose_traces(parent_group, name='traces'):
    """Read a `name`/ subgroup into a DataArray(bodypart, trial, time), or None."""
    if name not in parent_group:
        return None
    traces_group = parent_group[name]
    bodyparts = [k for k in traces_group.keys()
                 if k not in _POSE_TRACES_RESERVED_KEYS]
    return xr.DataArray(
        np.stack([traces_group[bp][:].astype(np.float64) for bp in bodyparts]),
        dims=['bodypart', 'trial', 'time'],
        coords={
            'bodypart': bodyparts,
            'trial': traces_group['trials'][:],
            'time':  traces_group['times'][:],
        },
    )


def _save_pose_xcorr(parent_group, xcorr):
    """Write crosscorr/ subgroup from a dict (functions, lags, peak_lags, drift)."""
    grp = _replace_group(parent_group, 'crosscorr')
    grp.create_dataset('functions', data=np.asarray(xcorr['functions'], dtype=np.float64))
    grp.create_dataset('lags', data=np.asarray(xcorr['lags'], dtype=np.float64))
    grp.create_dataset('peak_lags', data=np.asarray(xcorr['peak_lags'], dtype=np.float64))
    grp.attrs['drift'] = xcorr['drift']


def _load_pose_xcorr(parent_group):
    """Read crosscorr/ subgroup into a dict, or None."""
    if 'crosscorr' not in parent_group:
        return None
    grp = parent_group['crosscorr']
    return {
        'functions': grp['functions'][:].astype(np.float64),
        'lags':      grp['lags'][:].astype(np.float64),
        'peak_lags': grp['peak_lags'][:].astype(np.float64),
        'drift':     grp.attrs['drift'],
    }


def _read_video_qc(h5_file):
    """Return existing manual QC label attrs from the video group, if present."""
    if 'video' not in h5_file:
        return {}
    attrs = h5_file['video'].attrs
    return {label: attrs[label] for label in LP_QC_LABELS if label in attrs}


def _save_video(session, h5_file, band):
    # Read manual labels before _replace_group wipes them, so re-saving
    # automatic data (extraction --overwrite) never clobbers a manual verdict.
    preserved_qc = _read_video_qc(h5_file)
    grp = _replace_group(h5_file, 'video')
    grp.attrs['length_discrepancy'] = session.length_discrepancy
    grp.attrs['framerate_from_tpts'] = session.framerate_from_tpts
    if session.pose_traces is not None:
        _save_pose_traces(grp, session.pose_traces)
    if session.pose_baseline_traces is not None:
        _save_pose_traces(grp, session.pose_baseline_traces, name='baseline_traces')
    if session.pose_xcorr is not None:
        _save_pose_xcorr(grp, session.pose_xcorr)
    for label in LP_QC_LABELS:
        value = getattr(session, label)
        if value in (None, LP_QC_NOT_SET):
            value = preserved_qc.get(label, LP_QC_NOT_SET)
        grp.attrs[label] = value
    for col in VIDEO_QC_COLS:
        grp.attrs[col] = session.video_qc.get(col, LP_QC_NOT_SET)


def _load_video(session, h5_file, band):
    if 'video' not in h5_file:
        return
    grp = h5_file['video']
    if 'length_discrepancy' in grp.attrs:
        session.length_discrepancy = grp.attrs['length_discrepancy']
    if 'framerate_from_tpts' in grp.attrs:
        session.framerate_from_tpts = grp.attrs['framerate_from_tpts']
    session.pose_traces = _load_pose_traces(grp)
    session.pose_baseline_traces = _load_pose_traces(grp, name='baseline_traces')
    session.pose_xcorr = _load_pose_xcorr(grp)
    for label in LP_QC_LABELS:
        if label in grp.attrs:
            value = grp.attrs[label]
            setattr(session, label,
                    value.decode() if isinstance(value, bytes) else value)
    session.video_qc = {}
    for col in VIDEO_QC_COLS:
        if col in grp.attrs:
            value = grp.attrs[col]
            session.video_qc[col] = (
                value.decode() if isinstance(value, bytes) else value)


_SAVE_HANDLERS = {
    'metadata':   _save_metadata,
    'errors':     _save_errors,
    'photometry': _save_photometry,
    'trials':     _save_trials,
    'wheel':      _save_wheel,
    'video':      _save_video,
}

_LOAD_HANDLERS = {
    'metadata':   _load_metadata,
    'errors':     _load_errors,
    'photometry': _load_photometry,
    'trials':     _load_trials,
    'wheel':      _load_wheel,
    'video':      _load_video,
}


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
        self.filepath = SESSIONS_H5_DIR / f'{self.eid}.h5'
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

        def _as_list(raw):
            """Normalize a parallel-list field: list/ndarray → list, non-null
            scalar → length-1 list, missing/null → []."""
            if isinstance(raw, (list, np.ndarray)):
                return list(raw)
            return [raw] if pd.notna(raw) else []

        self.brain_region = _as_list(session_series.get('brain_region', []))
        self.hemisphere = _as_list(session_series.get('hemisphere', []))
        self.target_NM = _as_list(session_series.get('target_NM', []))

        self.errors = []

        super().__init__(*args, eid=self.eid, **kwargs)
        if not isinstance(self.photometry, dict):
            self.photometry = {}
        self.responses = {}
        self.ols_fits = {}
        self.qc = pd.DataFrame()
        self.pose = None
        self.pose_times = None
        self.motion_energy = None
        self.length_discrepancy = np.nan
        self.framerate_from_tpts = np.nan
        self.pose_traces = None
        self.pose_baseline_traces = None
        self.pose_xcorr = None
        self.video_qc = {}
        self.qc_lp = LP_QC_NOT_SET
        self.qc_movement = LP_QC_NOT_SET
        self.qc_timing = LP_QC_NOT_SET
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

        # Load remaining groups from the same file
        ps.filepath = Path(fpath)
        ps.load_h5(groups=['errors', 'photometry', 'trials', 'wheel'])
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

    def _fetch_block_info(self):
        """Fetch and cache block structure fields from the Alyx session JSON."""
        if not hasattr(self, '_block_info'):
            from iblnm.io import get_block_info
            self._block_info = get_block_info(self.eid, self.one)
        return self._block_info

    def validate_block_structure(self):
        """Validate probabilityLeft against expected block structure.

        Training sessions must have probabilityLeft == 0.5 uniformly.
        Biased/ephys sessions are checked for short blocks, then validated
        against the session JSON ground truth.

        Raises
        ------
        BlockStructureBug
            If the block structure is corrupted.
        """
        if 'probabilityLeft' not in self.trials.columns:
            return

        if self.session_type == 'training':
            if not (self.trials['probabilityLeft'] == 0.5).all():
                raise BlockStructureBug(
                    "Training session has non-uniform probabilityLeft"
                )
            return

        if self.session_type not in ('biased', 'ephys'):
            return

        block_info = task.validate_block_structure(self.trials)
        if not block_info['flagged']:
            return

        # Cheap check flagged — fetch JSON ground truth
        bi = self._fetch_block_info()
        if bi['len_blocks'] is None:
            self.log_error(MissingBlockInfo(self.eid))
            raise BlockStructureBug(
                f"Min block length: {block_info['min_block_length']}, "
                f"n_blocks: {block_info['n_blocks']}"
            )

        if not task.validate_block_match(
            self.trials, bi['len_blocks'], bi['positions'],
            bi['block_probability_set'],
        ):
            raise BlockStructureBug(
                "probabilityLeft does not match session JSON block structure"
            )

    def fix_block_structure(self):
        """Overwrite probabilityLeft with correct values.

        Training sessions are set to 0.5 uniformly. Biased/ephys sessions
        are reconstructed from the session JSON.

        Returns
        -------
        bool
            True if the fix was applied, False if block info is unavailable.
        """
        if self.session_type == 'training':
            self.trials['probabilityLeft'] = 0.5
            return True
        bi = self._fetch_block_info()
        if bi['len_blocks'] is None:
            return False
        self.trials['probabilityLeft'] = task.reconstruct_probability_left(
            len(self.trials), bi['len_blocks'], bi['positions'],
            bi['block_probability_set'],
        )
        return True

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
        """Extract peri-event response matrices as a dict of xarrays.

        Returns
        -------
        dict[str, xr.DataArray]
            One DataArray per brain region, dims (event, trial, time).
        """


        if events is None:
            events = RESPONSE_EVENTS
        if window is None:
            window = self.RESPONSE_WINDOW

        self.responses = {}
        for region in self.photometry[band].columns:
            signal = self.photometry[band][region]
            per_event = []
            for event in events:
                event_times = self.trials[event].values
                resp, sample_times = get_responses(
                    signal, event_times, t0=window[0], t1=window[1],
                )
                per_event.append(resp)
            self.responses[region] = xr.DataArray(
                np.stack(per_event),
                dims=['event', 'trial', 'time'],
                coords={
                    'event': list(events),
                    'trial': self.trials.index.to_numpy(),
                    'time': sample_times,
                },
            )
        return self.responses

    def save_h5(self, fpath=None, groups=None, band='GCaMP_preprocessed', mode='a'):
        """Save session data to HDF5.

        Parameters
        ----------
        fpath : Path or str, optional
            Output path. Defaults to ``self.filepath``.
        groups : sequence of str, optional
            Which data groups to write. Any subset of:
            'metadata', 'errors', 'photometry', 'trials', 'wheel', 'video'.
            None auto-detects all available data groups.
        mode : str
            HDF5 file open mode ('a' creates/appends, 'w' truncates).
        """
        if fpath is None:
            fpath = self.filepath
        fpath = Path(fpath)
        fpath.parent.mkdir(parents=True, exist_ok=True)

        if groups is None:
            groups = self._available_save_groups(band)

        with h5py.File(fpath, mode) as h5_file:
            for group_name in groups:
                _SAVE_HANDLERS[group_name](self, h5_file, band)

    def _available_save_groups(self, band):
        has_photometry = (
            band in self.photometry
            or bool(getattr(self, 'responses', None))
            or (getattr(self, 'qc', None) is not None and len(self.qc) > 0)
        )
        has_video = (self.pose_traces is not None
                     or self.pose_xcorr is not None
                     or np.isfinite(self.length_discrepancy)
                     or np.isfinite(self.framerate_from_tpts))
        return [name for name, available in (
            ('photometry', has_photometry),
            ('trials',     getattr(self, 'trials', None) is not None),
            ('wheel',      getattr(self, 'wheel_velocity', None) is not None),
            ('video',      has_video),
        ) if available]

    def load_h5(self, fpath=None, groups=None, band='GCaMP_preprocessed'):
        """Load session data from HDF5 file.

        Parameters
        ----------
        fpath : Path or str, optional
            Path to the HDF5 file. Defaults to ``self.filepath``.
        groups : sequence of str, optional
            Which data groups to load. Any subset of:
            'metadata', 'errors', 'photometry', 'trials', 'wheel', 'video'.
            None loads all groups present in the file.
        band : str
            Preprocessed band name used as the key in `self.photometry`
            when loading photometry.
        """
        if fpath is None:
            fpath = self.filepath
        group_names = list(_LOAD_HANDLERS) if groups is None else list(groups)
        with h5py.File(fpath, 'r') as h5_file:
            for group_name in group_names:
                _LOAD_HANDLERS[group_name](self, h5_file, band)

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
        from iblnm.analysis import compute_bleaching_tau, compute_iso_correlation

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

    def subtract_baseline(self, responses, window=None):
        """Subtract per-trial pre-event baseline from response traces.

        Parameters
        ----------
        responses : xr.DataArray
            Single-region DataArray with dims (event, trial, time).
        window : tuple(float, float), optional
            Baseline window in seconds [t_start, t_end). Defaults to
            BASELINE_WINDOW from config.

        Returns
        -------
        xr.DataArray
            Baseline-subtracted responses, same shape and coords as input.
        """
        if window is None:
            window = BASELINE_WINDOW
        sample_times = responses.coords['time'].values
        i0 = np.searchsorted(sample_times, window[0])
        i1 = np.searchsorted(sample_times, window[1])
        baseline = responses.isel(time=slice(i0, i1)).mean(dim='time', skipna=True)
        return responses - baseline

    def mask_subsequent_events(self, responses, event_order=None):
        """Mask response times that fall after the next event onset.

        For each consecutive pair (e0, e1) in event_order, per-trial times
        t > (trials[e1] - trials[e0]) are replaced with NaN in the e0
        response matrix. Trials where the next event time is NaN are not masked.

        Parameters
        ----------
        responses : xr.DataArray
            Single-region DataArray with dims (event, trial, time).
        event_order : list[str], optional
            Chronologically ordered event names. Defaults to RESPONSE_EVENTS.

        Returns
        -------
        xr.DataArray
            Masked responses, same shape and coords as input.
        """

        if event_order is None:
            event_order = list(RESPONSE_EVENTS)
        if self.trials is None:
            return responses
        events_present = list(responses.coords['event'].values)
        sample_times = responses.coords['time'].values
        result = responses.copy()
        for i, event in enumerate(event_order[:-1]):
            next_event = event_order[i + 1]
            if event not in events_present:
                continue
            if event not in self.trials.columns or next_event not in self.trials.columns:
                continue
            dt = self.trials[next_event].values - self.trials[event].values
            nan_dt = np.isnan(dt)
            keep = (sample_times[None, :] <= dt[:, None]) | nan_dt[:, None]
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

    def load_camera_times(self):
        """Load left-camera frame timestamps, independently of LightningPose.

        Stores the per-frame times (session clock, seconds) in
        ``self.pose_times``. Raises ``MissingVideoTimestamps`` when the dataset
        is absent, so the basic-video pass can block the session before LP is
        attempted. Loaded separately from ``load_pose`` because the camera
        timestamps gate the whole session while LP gates only the traces.
        """
        try:
            self.pose_times = np.asarray(self.one.load_dataset(
                self.eid, '_ibl_leftCamera.times.npy', collection='alf'))
        except ALFObjectNotFound:
            raise MissingVideoTimestamps("leftCamera.times")

    def load_pose(self):
        """Load LightningPose keypoint tracking from the left camera.

        Stores the pose DataFrame (columns ``{part}_x``, ``{part}_y``,
        ``{part}_likelihood``) in ``self.pose``. Raises ``MissingLP`` when the
        pose dataset is not available, so batch extraction can log and skip.

        Loads only the ``lightningPose`` dataset via ``load_dataset`` rather than
        the whole ``leftCamera`` object, which would also fetch ``features`` and
        ``ROIMotionEnergy`` that we never use. Camera timestamps are loaded
        separately by ``load_camera_times``.
        """
        try:
            self.pose = self.one.load_dataset(
                self.eid, '_ibl_leftCamera.lightningPose.pqt', collection='alf')
        except ALFObjectNotFound:
            raise MissingLP("leftCamera.lightningPose")

    def load_motion_energy(self):
        """Load per-frame left-camera ROI motion energy, independently of LP.

        Stores the per-frame motion-energy scalar (on the ``leftCamera.times``
        base) in ``self.motion_energy``. Raises ``MissingMotionEnergy`` when the
        dataset is absent, logged non-fatally by the pipeline. Unlike
        ``lightningPose`` and ``times``, the ROIMotionEnergy dataset carries no
        ``_ibl_`` prefix, so its ``load_dataset`` object string differs.
        """
        try:
            self.motion_energy = np.asarray(self.one.load_dataset(
                self.eid, 'leftCamera.ROIMotionEnergy.npy', collection='alf'))
        except ALFObjectNotFound:
            raise MissingMotionEnergy("leftCamera.ROIMotionEnergy")

    def compute_video_measures(self):
        """Compute basic-video scalars from the loaded camera timestamps.

        Sets ``self.length_discrepancy`` (video duration minus
        ``session_length``, seconds) and ``self.framerate_from_tpts`` (median
        inter-frame interval, seconds) from ``self.pose_times``. Requires
        ``load_camera_times`` to have populated ``self.pose_times``.
        """
        self.length_discrepancy = (
            (self.pose_times[-1] - self.pose_times[0]) - self.session_length)
        self.framerate_from_tpts = np.median(np.diff(self.pose_times))

    def fetch_video_qc(self):
        """Live-fetch leftCamera extended QC and store the 8 ``VIDEO_QC_COLS``.

        Queries Alyx via ``io.get_extended_qc`` and retains the eight
        ``config.VIDEO_QC_COLS`` outcome labels in ``self.video_qc`` (a dict
        keyed by column name). Columns absent from the source default to
        ``LP_QC_NOT_SET``. The Alyx call is the only network access in this
        method, keeping the downstream validation/storage logic testable with
        an injected ``video_qc`` dict.
        """
        from iblnm.io import get_extended_qc
        qc = get_extended_qc(self.to_series(), one=self.one)
        self.video_qc = {col: qc.get(col, LP_QC_NOT_SET) for col in VIDEO_QC_COLS}

    def _movement_signals(self):
        """Per-frame movement signals keyed by channel label.

        Returns a dict ``label -> (signal, event)`` where ``signal`` is a
        ``pd.Series`` on a common 1/POSE_FS time base and ``event`` is the trials
        column the response trace locks to. LP keypoint channels
        (``config.POSE_MEASURES``) are included only when ``self.pose`` is set;
        the ``motion_energy`` channel only when ``self.motion_energy`` is set.
        The two sources are independent, so a session may contribute either or
        both. Returns an empty dict when neither source is present.
        """
        signals = {}
        if self.pose is not None:
            # Resample raw pose to a common rate first, so speeds (px per time
            # step) and trace lengths are comparable across camera fps.
            pose, pose_times = resample_pose(self.pose, self.pose_times, POSE_FS)
            for label, (event, keypoints, reduction) in POSE_MEASURES.items():
                signals[label] = (
                    pd.Series(movement_trace(pose, keypoints, reduction),
                              index=pose_times),
                    event,
                )
        if self.motion_energy is not None:
            signals['motion_energy'] = (
                resample_signal(
                    pd.Series(self.motion_energy, index=self.pose_times), POSE_FS),
                MOTION_ENERGY_EVENT,
            )
        return signals

    def extract_movement_traces(self):
        """Extract per-trial peri-event movement traces for each channel.

        Assembles the available per-frame signals (LP keypoint measures and/or
        the resampled motion-energy scalar), event-locks each to its own event,
        and stores them on ``self.pose_traces`` as a DataArray with dims
        ``(bodypart, trial, time)``. ``self.pose_baseline_traces`` holds the same
        channels stimOn-locked for a common pre-stimOn baseline. Both are left
        ``None`` when neither LP nor motion energy is present.
        """
        signals = self._movement_signals()
        if not signals:
            self.pose_traces = None
            self.pose_baseline_traces = None
            return
        stimon = self.trials['stimOn_times'].values
        traces = {}
        baselines = {}
        for label, (signal, event) in signals.items():
            traces[label], tpts = get_responses(
                signal, self.trials[event].values,
                t0=RESPONSE_WINDOW[0], t1=RESPONSE_WINDOW[1],
            )
            # Stimulus-onset-locked trace for a common pre-stimOn baseline, over
            # the same window so its time axis matches the response trace.
            baselines[label], _ = get_responses(
                signal, stimon, t0=RESPONSE_WINDOW[0], t1=RESPONSE_WINDOW[1],
            )
        labels = list(signals)
        coords = {
            'bodypart': labels,
            'trial': self.trials.index.to_numpy(),
            'time': tpts,
        }
        self.pose_traces = xr.DataArray(
            np.stack([traces[label] for label in labels]),
            dims=['bodypart', 'trial', 'time'], coords=coords,
        )
        self.pose_baseline_traces = xr.DataArray(
            np.stack([baselines[label] for label in labels]),
            dims=['bodypart', 'trial', 'time'], coords=coords,
        )

    def extract_paw_wheel_xcorr(self):
        """Compute the per-third paw–wheel cross-correlation timing diagnostic.

        Stores ``functions``, ``lags``, ``peak_lags``, and ``drift`` on
        ``self.pose_xcorr``.
        """
        paw_speed = movement_trace(self.pose, ['paw_l', 'paw_r'], 'sum_speed')
        finite = np.isfinite(paw_speed)  # drop untracked frames (NaN speed)
        functions, lags, peak_lags, drift = per_third_crosscorr(
            paw_speed[finite], self.pose_times[finite],
            np.abs(self.wheel['velocity'].values), self.wheel['times'].values,
        )
        self.pose_xcorr = {'functions': functions, 'lags': lags,
                           'peak_lags': peak_lags, 'drift': drift}

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

        responses = self.mask_subsequent_events(self.responses[brain_region])
        responses = self.subtract_baseline(responses)
        sample_times = responses.coords['time'].values

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
            resp = responses.sel(event=event).values[trial_mask]
            magnitudes = compute_response_magnitude(resp, sample_times, win)
            result[label] = np.nanmean(magnitudes)

        vec = pd.Series(result)

        if normalize == 'minmax':
            vmin, vmax = vec.min(), vec.max()
            if vmax > vmin:
                vec = (vec - vmin) / (vmax - vmin)

        return vec


    def fit_response_model(self, df: pd.DataFrame, formula: str,
                           response_col: str = 'response'):
        """Fit one OLS response model on a prepared trial frame.

        Thin, event/region-agnostic wrapper over ``analysis.fit_ols``: the
        ``{response}`` placeholder in ``formula`` is filled with ``response_col``
        before fitting, so the caller owns which magnitude column is the
        response. This method does not touch ``self.ols_fits`` — the comparison
        caller keys that cache by ``(name, event)``, context this single fit
        does not have.

        Parameters
        ----------
        df : pd.DataFrame
            Coded trial frame carrying ``response_col`` and every predictor the
            formula references.
        formula : str
            Wilkinson formula whose ``{response}`` placeholder (if present) is
            replaced by ``response_col``; passed through unchanged otherwise.
        response_col : str
            Name of the response column substituted into ``formula``.

        Returns
        -------
        statsmodels RegressionResults or None
            The fitted model (exposes ``.rsquared``, ``.params``), or ``None``
            if the design is degenerate (mirrors ``analysis.fit_ols``).
        """
        formula = formula.format(response=response_col)
        return analysis.fit_ols(formula, df)

    def compare_response_models(self, brain_region, formulas,
                                response_col='response', reference='full',
                                min_trials=MIN_TRIALS_PERSESSION,
                                contrast_coding='log2'):
        """Fit a drop-one OLS family per event for one recording, return ΔR².

        Builds this recording's per-trial response frame for ``brain_region``,
        then for each event fits every formula in ``formulas`` on the event's
        complete-case trials and differences each reduced model against the
        ``reference`` model. Every model in an event is fit on the same rows
        (complete cases over the family's column union), so their R² are
        directly comparable. Each fit is cached in ``self.ols_fits`` keyed by
        ``(name, event)``.

        Parameters
        ----------
        brain_region : str
            Recording region; must be a key of ``self.responses``.
        formulas : dict[str, str]
            Drop-one family ``{name: formula_template}``; ``{response}`` is
            filled with ``response_col``. One key equals ``reference``.
        response_col : str
            Name of the per-trial response magnitude column.
        reference : str
            Key of the full model each reduced model's ΔR² is measured against.
        min_trials : int
            An event with fewer complete-case rows is omitted.
        contrast_coding : str
            Passed to :func:`iblnm.analysis.code_predictors`.

        Returns
        -------
        pd.DataFrame
            Long-form rows ``brain_region, target_NM, event, predictor, r2,
            delta_r2, n_trials``. Empty (those columns) if ``brain_region`` is
            absent or no event is scorable.
        """
        if brain_region not in self.responses:
            return pd.DataFrame(columns=PERSESSION_DROPONE_COLUMNS)

        target_NM = self.target_NM[self.brain_region.index(brain_region)]
        df = self._response_modeling_frame(brain_region, response_col)

        frames = []
        for event, df_event in df.groupby('event'):
            df_event = analysis.code_predictors(df_event, contrast_coding)
            union_cols = analysis.formula_union_columns(
                formulas.values(), df_event.columns)
            df_event = df_event.dropna(subset=union_cols)
            if len(df_event) < min_trials:
                continue
            fits = {name: self.fit_response_model(df_event, formula,
                                                  response_col)
                    for name, formula in formulas.items()}
            if any(fit is None for fit in fits.values()):
                continue
            for name, fit in fits.items():
                self.ols_fits[(name, event)] = fit
            r2_by_name = {name: fit.rsquared for name, fit in fits.items()}
            rows = analysis.dropone_delta_r2(r2_by_name, reference)
            rows.insert(0, 'brain_region', brain_region)
            rows.insert(1, 'target_NM', target_NM)
            rows.insert(2, 'event', event)
            rows['n_trials'] = len(df_event)
            frames.append(rows)

        if not frames:
            return pd.DataFrame(columns=PERSESSION_DROPONE_COLUMNS)
        return pd.concat(frames, ignore_index=True)[PERSESSION_DROPONE_COLUMNS]

    def _response_modeling_frame(self, brain_region: str,
                                 response_col: str) -> pd.DataFrame:
        """Per-trial response magnitudes for one region merged with regressors.

        For each event in ``self.responses[brain_region]`` computes the
        early-window magnitude per trial (named ``response_col``), stacks the
        events into one long frame, and merges on ``trial`` with this session's
        :func:`iblnm.analysis.build_trial_regressors`. Adds the recording's
        ``hemisphere`` so :func:`iblnm.task.add_relative_contrast` can resolve
        contra/ipsi, then selects the modeling trials.

        Returns
        -------
        pd.DataFrame
            Long-form ``event, trial, response_col`` plus the coded-ready
            regressor columns, restricted to unbiased-block go trials.
        """
        responses = self.responses[brain_region]
        tpts = responses.coords['time'].values
        magnitude_frames = [
            pd.DataFrame({
                'event': event,
                'trial': responses.coords['trial'].values,
                response_col: compute_response_magnitude(
                    responses.sel(event=event).values, tpts,
                    RESPONSE_WINDOWS['early']),
            })
            for event in responses.coords['event'].values
        ]
        long = pd.concat(magnitude_frames, ignore_index=True)

        regressors = analysis.build_trial_regressors(
            self.trials, getattr(self, 'wheel_velocity', None))
        df = long.merge(regressors, on='trial', how='left')
        df['hemisphere'] = self.hemisphere[
            self.brain_region.index(brain_region)]
        df = task.add_relative_contrast(df)
        return analysis.select_modeling_trials(df, response_col)


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


def _resolve_ps_variable(ps, entry):
    """Resolve a fixed/swapped entry against a PhotometrySession.

    A callable is called as ``entry(ps)``; a str is read as a ``ps.trials``
    column when present, else as a PS attribute. Returns a 1-D numpy array.
    """
    if callable(entry):
        value = entry(ps)
    elif isinstance(ps.trials, pd.DataFrame) and entry in ps.trials.columns:
        value = ps.trials[entry]
    else:
        value = getattr(ps, entry)
    return np.asarray(value).ravel()


class _PermutationStageError(Exception):
    """A failure in one target-path stage of the session permutation test.

    Carries the ``stage`` label ('prep', 'resolve', or 'stat') so the per-unit
    loop can report which component raised. Its string form prefixes the
    original exception's type and message with that stage.
    """

    def __init__(self, stage, original):
        self.stage = stage
        self.original = original
        super().__init__(f"{stage}: {type(original).__name__}: {original}")


@contextmanager
def _permutation_stage(stage):
    """Re-raise any exception from the block as ``_PermutationStageError(stage)``."""
    try:
        yield
    except Exception as exc:
        raise _PermutationStageError(stage, exc)


def _apply_statistic(statistic, arrays):
    """Truncate every array to their common minimum length (from index 0),
    then call ``statistic(*truncated)``."""
    min_len = min(len(a) for a in arrays)
    return statistic(*[a[:min_len] for a in arrays])


def _get_donor_sessions(view, pos, group_by):
    """Integer positions in ``view`` eligible as donors for unit ``pos``.

    Donors share the target's ``group_by`` value (all other units when
    ``group_by`` is None); the target's own position is always excluded.
    """
    positions = np.arange(len(view))
    if group_by is None:
        return view[positions != pos]
    same_group = view[group_by].values == view.iloc[pos][group_by]
    return view[same_group & (positions != pos)]


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
        self._recordings_targetnms = False
        self.one = one
        self.h5_dir = h5_dir if h5_dir is not None else SESSIONS_H5_DIR
        self._sessions = {}  # eid → PhotometrySession cache
        self.response_traces = None
        self.response_traces_tpts = None
        self.mean_traces = None
        self.response_magnitudes = None
        self.trial_regressors = None
        self.response_features = None
        self.performance = None
        self.psychometric_features = None
        self.similarity_matrix = None
        self.decoder = None
        self.glm_response_features = None
        self.cca_result = None
        self.cohort_cca_results = None
        self.cohort_cca_data = None
        self.cohort_cca_cross_projections = None
        self.cohort_cca_weight_similarities = None
        self.lmm_fits = {}
        self._lmm_group_by = None

    @classmethod
    def from_catalog(cls, catalog, one, h5_dir=SESSIONS_H5_DIR):
        """Build a group from a session catalog DataFrame.

        Validates parallel list columns and populates ``logged_errors`` from
        each session's H5 ``/errors`` group when ``h5_dir`` is given (scans all
        files, so it can take a moment). Call ``filter_sessions`` separately.

        Parameters
        ----------
        catalog : pd.DataFrame
            Session catalog (one row per session, with list columns for
            brain_region, hemisphere, target_NM).
        one : one.api.One
            ONE connection instance.
        h5_dir : Path, optional
            Directory containing {eid}.h5 files. When provided, the H5 ``/errors``
            groups are scanned to populate ``logged_errors`` for error-based
            filtering. When ``None``, ``logged_errors`` is left empty (no scan).
        """
        from iblnm.config import SESSION_SCHEMA
        from iblnm.util import (
            enforce_schema, validate_parallel_lists, collect_session_errors,
        )

        df = enforce_schema(catalog.copy(), SESSION_SCHEMA)

        parallel_cols = ['brain_region', 'hemisphere', 'target_NM']
        df = validate_parallel_lists(df, parallel_cols)

        if h5_dir is not None:
            print(f"Scanning H5 files in {h5_dir} for logged errors "
                  "(this may take a moment)...")
            df = df.merge(collect_session_errors(df['eid'], h5_dir),
                          on='eid', how='left')
        elif 'logged_errors' not in df.columns:
            df['logged_errors'] = [[] for _ in range(len(df))]

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
        _recordings_targetnms (set by filter_sessions) when not False.
        """
        parallel_cols = ['brain_region', 'hemisphere', 'target_NM']
        df = self.sessions.explode(parallel_cols).copy()
        df['fiber_idx'] = df.groupby('eid').cumcount()
        if self._recordings_targetnms is not False:
            df = df[df['target_NM'].isin(self._recordings_targetnms)]
        return df.reset_index(drop=True)

    def filter_sessions(self, session_types=SESSION_TYPES_TO_ANALYZE,
                        exclude_subjects=SUBJECTS_TO_EXCLUDE,
                        exclude_eids=EIDS_TO_DROP,
                        qc_blockers=ANALYSIS_QC_BLOCKERS,
                        targetnms=TARGETNMS_TO_ANALYZE,
                        min_performance=MIN_PERFORMANCE,
                        required_contrasts=REQUIRED_CONTRASTS,
                        lab=False, start_time_min=False):
        """Compute a boolean filter mask over _catalog. Non-destructive.

        Stores a new _filter_mask on each call. Access filtered data via the
        ``sessions`` and ``recordings`` properties. Call multiple times to get
        different filtered views.

        All filter parameters accept ``False`` to skip that filter.

        Parameters
        ----------
        session_types : tuple of str or False
            Session types to keep. Defaults to config.SESSION_TYPES_TO_ANALYZE.
        exclude_subjects : list of str or False
            Subjects to exclude. Defaults to config.SUBJECTS_TO_EXCLUDE.
        exclude_eids : list of str or False
            Specific session eids to exclude (curated drop-list). Defaults
            to config.EIDS_TO_DROP.
        qc_blockers : set of str or False
            Error types that block a session. Defaults to
            config.ANALYSIS_QC_BLOCKERS. Silently skipped if
            ``logged_errors`` is not present on the catalog.
        targetnms : list of str or False
            Target-NM values to retain in sessions and recordings.
            Defaults to config.TARGETNMS_TO_ANALYZE.
        min_performance : float, dict, or False
            Minimum fraction_correct. Defaults to config.MIN_PERFORMANCE.
            Requires 'fraction_correct' in the catalog.
        required_contrasts : frozenset of float or False
            Required contrast set. Defaults to config.REQUIRED_CONTRASTS.
            Requires 'contrasts' in the catalog.
        lab : str or False
            Keep only sessions from this lab.
        start_time_min : str, date, or False
            Keep only subjects whose first session is >= this date.

        Returns
        -------
        None
        """
        df = self._catalog
        true = pd.Series(True, index=df.index)

        # Build individual masks — False skips any filter
        type_mask = df['session_type'].isin(session_types) if session_types is not False else true
        subject_mask = ~df['subject'].isin(exclude_subjects) if exclude_subjects is not False else true
        eid_mask = ~df['eid'].isin(exclude_eids) if exclude_eids is not False else true
        lab_mask = (df['lab'] == lab) if (lab is not False and 'lab' in df.columns) else true

        if start_time_min is not False and 'start_time' in df.columns:
            dt_series = pd.to_datetime(df['start_time'], format='ISO8601')
            first_per_row = dt_series.groupby(df['subject']).transform('min')
            start_mask = first_per_row >= pd.Timestamp(start_time_min)
        else:
            start_mask = true

        if qc_blockers is not False and 'logged_errors' in df.columns:
            qc_mask = df['logged_errors'].apply(
                lambda e: not any(err in qc_blockers for err in e)
            )
        else:
            qc_mask = true

        if targetnms is not False and 'target_NM' in df.columns:
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

        mask = type_mask & subject_mask & eid_mask & lab_mask & start_mask & qc_mask & target_mask & perf_mask & contrast_mask

        self._filter_mask = mask
        self._recordings_targetnms = targetnms

        n = len(df)
        lines = [f"filter_sessions: {n} -> {int(mask.sum())}"]
        for label, m in [
            ('session_type', type_mask), ('excluded_subjects', subject_mask),
            ('excluded_eids', eid_mask),
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

    def _permutation_test_unit(self, target, donors, prep_fn, stat_fn,
                               fixed_var, swapped_var, n_iter, rng):
        """Run the session-swap test for one unit; return ``(observed, null)``.

        ``stat_fn`` returns a dict ``{name: value}`` of one or more scalar
        quantities. ``observed`` is that dict computed from the target's own
        data; ``null`` is a dict with the same keys, each a length-``n_iter``
        array of the quantity recomputed with ``swapped`` drawn from a random
        donor. ``fixed`` is always resolved on the target PS; ``swapped`` on
        the target (for ``observed``) or the donor (for each null draw). A
        donor iteration that raises leaves NaN at that index for every key.
        Re-loads the donor PS every iteration — no caching, so this is the
        dominant cost of the test (acceptable per spec).
        """
        # Prepare the photometry session
        with _permutation_stage('prep'):
            target_ps = PhotometrySession(target, one=self.one, load_data=False)
            target_ps = prep_fn(target_ps)

        # Extract data arrays to be used in stat_fn
        with _permutation_stage('resolve'):
            fixed_arrays = [getattr(target_ps, e) for e in fixed_var]
            swapped_arrays = [getattr(target_ps, e) for e in swapped_var]

        # Compute the observed quantities (dict keyed by quantity name)
        with _permutation_stage('stat'):
            observed = _apply_statistic(stat_fn, fixed_arrays + swapped_arrays)

        null = {key: np.full(n_iter, np.nan) for key in observed}
        for i in range(n_iter):
            donor = donors.iloc[rng.integers(len(donors))]
            try:
                donor_ps = PhotometrySession(donor, one=self.one, load_data=False)
                donor_ps = prep_fn(donor_ps)
                donated_arrays = [getattr(donor_ps, e) for e in swapped_var]
                drawn = _apply_statistic(stat_fn, fixed_arrays + donated_arrays)
                for key in null:
                    null[key][i] = drawn[key]
            except Exception:
                continue

        return observed, null

    def session_permutation_test(self, prep_fn, stat_fn, fixed_var, swapped_var,
                                 statistic_key, group_by='target_NM',
                                 unit='recordings', n_iter=1000,
                                 alternative='two-sided', seed=42):
        """Session-swap permutation test of a statistic, per unit.

        For each unit (a recording or session), computes the observed
        quantities from its own data, then builds an ``n_iter`` null by holding
        ``fixed`` with the target and swapping ``swapped`` in from random donor
        units in the same ``group_by`` group, breaking the within-unit
        correspondence. ``stat_fn`` may return several quantities at once
        (e.g. a regression slope and its R²); the p-value is computed for the
        one named by ``statistic_key``, while every quantity's observed value
        and null distribution are recorded.

        Parameters
        ----------
        stat_fn : callable
            ``statistic(*fixed_arrays, *swapped_arrays) -> dict[str, float]``.
            Receives the resolved ``fixed`` arrays followed by the resolved
            ``swapped`` arrays, in list order, each truncated to their common
            length, and returns a dict mapping quantity names to scalars.
        fixed_var : list of str
            Attribute names resolved via ``getattr`` on the target PS (after
            ``prep_fn`` has run) in both the observed run and every null
            iteration. ``prep_fn`` is responsible for setting these attributes.
        swapped_var : list of str
            Same as ``fixed_var``; resolved on the target PS for the observed
            run and on a donor PS for each null draw.
        statistic_key : str
            Which key of the ``stat_fn`` dict is the test statistic; its
            observed value and null drive ``p_value``. A successful unit whose
            ``stat_fn`` output lacks this key raises ``KeyError`` (loud
            misconfiguration, not a silent NaN run).
        group_by : str or None, default 'target_NM'
            Column of the unit view defining the donor pool (donors share the
            target's value). ``None`` pools all other units.
        unit : {'recordings', 'sessions'}, default 'recordings'
            View to iterate: ``self.recordings`` (one row per region) or
            ``self.sessions``.
        n_iter : int, default 1000
            Null iterations per unit.
        alternative : {'two-sided', 'greater', 'less'}, default 'two-sided'
            Passed to ``analysis.permutation_pvalue``.
        seed : int, default 42
            Seeds ``np.random.default_rng`` for the donor draws.

        Returns
        -------
        pandas.DataFrame
            One row per unit, not written to disk. Identifier columns ``eid``
            (and ``brain_region``, ``hemisphere``, ``target_NM``, ``fiber_idx``
            for ``unit='recordings'``) plus ``error`` (None on success, else a
            ``"<stage>: <ExcType>: <msg>"`` string naming the failing target
            stage — ``prep``, ``resolve``, or ``stat``), ``p_value`` (float, for
            ``statistic_key``) and, for every key ``k`` returned by ``stat_fn``,
            ``observed_<k>`` (float) and ``null_<k>`` (object column of 1-D
            arrays). A unit whose run raises yields a non-null ``error``,
            ``p_value`` NaN, and NaN ``observed_*``/``null_*`` entries.
        """
        from tqdm import tqdm

        view = self.recordings if unit == 'recordings' else self.sessions
        rng = np.random.default_rng(seed)

        rows = []
        for pos in tqdm(range(len(view)), desc="Permutation test"):
            session = view.iloc[pos]
            donors = _get_donor_sessions(view, pos, group_by)
            if len(donors) == 0:
                raise ValueError(f"Empty donor pool for unit at position {pos}")
            try:
                observed, null = self._permutation_test_unit(
                    session, donors, prep_fn, stat_fn, fixed_var, swapped_var,
                    n_iter, rng
                    )
                error = None
            except _PermutationStageError as exc:
                observed, null, error = None, None, str(exc)
            row = session.to_dict()
            row['error'] = error
            if observed is None:
                row['p_value'] = np.nan
            else:
                if statistic_key not in observed:
                    raise KeyError(
                        f"statistic_key {statistic_key!r} not in stat_fn output "
                        f"keys {list(observed)}")
                row['p_value'] = analysis.permutation_pvalue(
                    observed[statistic_key], null[statistic_key], alternative)
                for key in observed:
                    row[f'observed_{key}'] = observed[key]
                    row[f'null_{key}'] = null[key]
            rows.append(row)
        return pd.DataFrame(rows)

    def response_ols_dropone(self, formulas, response_col='response',
                             reference='full', min_trials=MIN_TRIALS_PERSESSION,
                             contrast_coding='log2'):
        """Per-recording drop-one OLS ΔR² over the whole group.

        Loops ``self.recordings``; for each row instantiates its
        ``PhotometrySession`` from H5 and runs
        :meth:`PhotometrySession.compare_response_models` for that recording's
        ``brain_region``. Each recording's long-form rows are tagged with the
        row's ``eid`` and ``subject`` (``target_NM`` already comes back from the
        PS method) and concatenated. A recording whose region is absent or whose
        events are all below ``min_trials`` contributes no rows. No disk write.

        Parameters
        ----------
        formulas : dict[str, str]
            Drop-one family ``{name: formula_template}`` passed through to
            ``compare_response_models``; one key equals ``reference``.
        response_col : str
            Per-trial response magnitude column the formulas model.
        reference : str
            Full-model key each reduced model's ΔR² is measured against.
        min_trials : int
            An event with fewer complete-case rows is omitted (per recording).
        contrast_coding : str
            Passed to :func:`iblnm.analysis.code_predictors`.

        Returns
        -------
        pandas.DataFrame
            Long-form ``eid, subject, target_NM, brain_region, event,
            predictor, r2, delta_r2, n_trials``; empty (those columns) when no
            recording is scorable.
        """
        from tqdm import tqdm

        frames = []
        for _, row in tqdm(self.recordings.iterrows(),
                           total=len(self.recordings),
                           desc="Per-recording OLS drop-one"):
            # Build the session from the authoritative recordings row, not the
            # H5 /metadata: query_database fills brain_region/hemisphere/
            # target_NM at the catalog level (TEMPFIX) but never writes them
            # back to the H5, so the per-session /metadata can be empty. Load
            # only the data groups compare_response_models needs.
            ps = PhotometrySession(row, one=self.one, load_data=False)
            ps.load_h5(Path(self.h5_dir) / f"{row['eid']}.h5",
                       groups=['photometry', 'trials', 'wheel'])
            rows = ps.compare_response_models(
                brain_region=row['brain_region'], formulas=formulas,
                response_col=response_col, reference=reference,
                min_trials=min_trials, contrast_coding=contrast_coding)
            if rows.empty:
                continue
            rows.insert(0, 'eid', row['eid'])
            rows.insert(1, 'subject', row['subject'])
            frames.append(rows)

        if not frames:
            return pd.DataFrame(columns=RESPONSE_OLS_DROPONE_COLUMNS)
        return pd.concat(frames, ignore_index=True)[RESPONSE_OLS_DROPONE_COLUMNS]

    # -----------------------------------------------------------------
    # Parquet loaders — populate group attributes from saved files
    # -----------------------------------------------------------------

    def _load_parquet(self, path):
        """Read a parquet file and filter rows to current recordings.

        Returns None if the file does not exist.
        """

        path = Path(path)
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        eids = set(self.recordings['eid'])
        return df[df['eid'].isin(eids)].copy()

    def load_performance(self, path):
        """Load performance data from parquet, filtered to in-scope eids."""
        self.performance = self._load_parquet(path)

    def load_response_magnitudes(self, path):
        """Load response magnitudes from parquet, filtered to current recordings."""
        self.response_magnitudes = self._load_parquet(path)

    def load_trial_regressors(self, path):
        """Load trial regressors from parquet, filtered to current recordings."""
        self.trial_regressors = self._load_parquet(path)

    def load_mean_traces(self, path):
        """Load mean traces from parquet, filtered to current recordings."""
        self.mean_traces = self._load_parquet(path)

    def load_response_features(self, path):
        """Load response features from parquet, filtered to current recordings."""

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

            ps.load_h5(h5_path, groups=['trials', 'photometry'])

            if (getattr(ps, 'responses', None) is None
                    or getattr(ps, 'trials', None) is None):
                continue

            if brain_region not in ps.responses:
                continue

            masked = ps.mask_subsequent_events(ps.responses[brain_region])
            responses = ps.subtract_baseline(masked)

            sample_times = responses.coords['time'].values
            if self.response_traces_tpts is None:
                self.response_traces_tpts = sample_times

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
                resp = responses.sel(event=event).values  # (n_trials, n_time)
                cache[(eid, brain_region, event)] = {
                    'traces': resp,
                    'tpts': sample_times,
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
        For each cached (recording × event), computes the scalar magnitude
        in the early response window. Trial-level task/movement predictors
        are not included here — they live in ``trial_regressors`` (populated
        by :meth:`get_trial_regressors`).

        Returns
        -------
        pd.DataFrame
            One row per (recording × event × trial) with columns
            ``eid, subject, session_type, NM, target_NM, brain_region,
            hemisphere, event, trial, response``.
        """
        from tqdm import tqdm

        if self.response_traces is None:
            self.load_response_traces()

        frames = []
        for (eid, brain_region, event), entry in tqdm(
                self.response_traces.items(),
                desc="Computing response magnitudes"):
            meta = entry['meta']
            magnitude = compute_response_magnitude(
                entry['traces'], entry['tpts'], RESPONSE_WINDOWS['early'],
            )
            frames.append(pd.DataFrame({
                'eid': meta['eid'],
                'subject': meta['subject'],
                'session_type': meta.get('session_type'),
                'NM': meta.get('NM'),
                'target_NM': meta['target_NM'],
                'brain_region': meta['brain_region'],
                'hemisphere': meta['hemisphere'],
                'event': event,
                'trial': range(len(magnitude)),
                'response': magnitude,
            }))

        self.response_magnitudes = (
            pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        )
        return self.response_magnitudes

    def _merge_trial_regressors(self) -> pd.DataFrame:
        """Join ``response_magnitudes`` with ``trial_regressors`` on (eid, trial).

        Raises if either is unpopulated. Trial-level predictors
        (contrast, side, choice, timing, peak velocity) come from
        ``trial_regressors``; recording keys and ``response`` come from
        ``response_magnitudes``.
        """
        if self.response_magnitudes is None:
            raise ValueError(
                "response_magnitudes not populated. "
                "Call get_response_magnitudes() first."
            )
        if self.trial_regressors is None:
            raise ValueError(
                "trial_regressors not populated. "
                "Call get_trial_regressors() first."
            )
        return self.response_magnitudes.merge(
            self.trial_regressors, on=['eid', 'trial'], how='left',
        )

    def _modeling_frame(self, response_col: str = 'response') -> pd.DataFrame:
        """Canonical trial selection shared by every model and plot.

        Merges ``response_magnitudes`` with ``trial_regressors``, adds
        hemisphere-relative contrast/side, then keeps unbiased-block go trials
        (``probabilityLeft == 0.5``, ``choice != 0``) with a real response
        (``response_time > 0.05`` and non-null ``response_col``). Adds a
        ``log_<var>`` column (base-10 log, NaN where the value is ≤ 0) for each
        ``config.TIMING_VARS`` entry present, so movement models can reference
        them; the NaN rows are dropped per family at fit time. Every model and
        plot derives from this frame so they share identical trials.

        Parameters
        ----------
        response_col : str
            Column name for the response magnitude whose NaNs are dropped.
        """
        from iblnm.task import add_relative_contrast

        df = add_relative_contrast(self._merge_trial_regressors())
        return analysis.select_modeling_trials(df, response_col)

    def _code_lmm_predictors(
        self, df: pd.DataFrame, contrast_coding: str = 'log2'
    ) -> pd.DataFrame:
        """Code the trial frame for LMM fitting; do not mutate the input.

        Returns a copy with ``contrast`` transformed (``contrast_coding``) and
        mean-centered, and ``side`` / ``reward`` deviation-coded to ±0.5
        (``side``: contra = +0.5, ipsi = −0.5; ``reward``: ``feedbackType`` 1 =
        +0.5, −1 = −0.5). ``log_<timing>`` columns are left untouched. Coding a
        column a given formula does not use is harmless.

        Parameters
        ----------
        df : pd.DataFrame
            Trial-level frame with columns ``contrast``, ``side``, and
            ``feedbackType``.
        contrast_coding : str
            Coding passed to :func:`iblnm.util.get_contrast_coding`.
        """
        return analysis.code_predictors(df, contrast_coding)

    def response_lmm_fit(self, formulas, group_by, response_col='response',
                         reml=True, re_formula='1', min_subjects=2,
                         events=None):
        """Fit caller-supplied LMMs per ``group_by`` group and cache each fit.

        For every group with at least ``min_subjects`` subjects, codes the
        trials (:meth:`_code_lmm_predictors`) and fits each model in
        ``formulas`` via :func:`iblnm.analysis.fit_lmm`. Each fitted
        ``LMMResult`` is cached in ``self.lmm_fits`` under
        ``(response_col, name, *group_values)`` for later effect extraction.
        Scoring is intrinsic: the returned frame carries each fit's in-sample
        R². The method does no ``config.LMM_FORMULAS`` lookup — names are
        whatever the caller keyed the dict by, so formulas from different
        config sets passed under distinct names never collide.

        Parameters
        ----------
        formulas : dict[str, str]
            Flat ``{name: formula_template}`` mapping; each template may
            contain ``{response}``, filled with ``response_col``.
        group_by : list[str]
            Columns whose unique combinations each get individual fits; their
            values tag the registry keys and the returned rows.
        response_col : str
            Response-magnitude column; also the formula's ``{response}``.
        reml : bool
            REML (True) for reporting fits or ML (False) for nested comparisons.
        re_formula : str or dict[str, str]
            Random-effects formula, shared across names (str) or per name
            (dict). Defaults to a random intercept (``'1'``).
        min_subjects : int
            Minimum subjects per group to attempt fitting.
        events : list[str], optional
            Restrict to these ``event`` values before grouping; ``None`` (the
            default) fits every event. Lets the caller fit a per-event model
            under one cached ``name`` without later events overwriting earlier
            ones.

        Returns
        -------
        pd.DataFrame
            One row per fitted ``(group, name)`` with the ``group_by`` columns,
            ``name``, ``marginal_r2``, and ``conditional_r2``.
        """
        df = self._modeling_frame(response_col)
        if events is not None:
            df = df[df['event'].isin(events)]
        self._lmm_group_by = list(group_by)
        formulas = {name: template.format(response=response_col)
                    for name, template in formulas.items()}
        model_cols = analysis.formula_union_columns(
            formulas.values(), df.columns)

        rows = []
        for keys, df_group in df.groupby(group_by):
            group_values = keys if isinstance(keys, tuple) else (keys,)
            df_coded = self._code_lmm_predictors(df_group)
            # Complete cases across the whole family, so every model fits the
            # same rows: a member whose formula omits a column must still drop
            # the rows where that column is NaN, else statsmodels misaligns
            # ``groups`` against the design matrix and the ΔR² denominators
            # diverge.
            df_coded = df_coded.dropna(subset=model_cols)
            if df_coded['subject'].nunique() < min_subjects:
                continue
            for name, formula in formulas.items():
                rf = re_formula[name] if isinstance(re_formula, dict) \
                    else re_formula
                fit = analysis.fit_lmm(formula, df_coded,
                                       groups=df_coded['subject'],
                                       re_formula=rf, reml=reml)
                if fit is None:
                    continue
                self.lmm_fits[(response_col, name, *group_values)] = fit
                rows.append({
                    **dict(zip(group_by, group_values)),
                    'name': name,
                    'marginal_r2': fit.variance_explained['marginal'],
                    'conditional_r2': fit.variance_explained['conditional'],
                })

        return pd.DataFrame(
            rows, columns=[*group_by, 'name', 'marginal_r2', 'conditional_r2'])

    def response_lmm_crossval(self, formulas, group_by, response_col='response',
                              reference='full', fold_col='subject',
                              min_subjects=3, min_test=5, min_trials=0,
                              events=None):
        """Out-of-sample ΔR² by leave-one-fold-out cross-validation per group.

        See :meth:`_response_lmm_resample` for the orchestration; this binds the
        scoring arguments of :func:`iblnm.analysis.crossval_lmm`.

        Parameters
        ----------
        formulas : dict[str, str]
            One comparison set: a flat ``{name: formula_template}`` mapping (a
            ``reference`` key plus drop-one variants), each template containing
            ``{response}``, filled with ``response_col``.
        group_by : list[str]
            Columns whose unique combinations each get an independent run.
        response_col : str
            Response-magnitude column; also the formula's ``{response}``.
        reference : str
            Key in ``formulas`` naming the baseline each other model's ΔR² is
            measured against.
        fold_col : str
            Column whose unique values define the leave-one-out folds.
        min_subjects : int
            Minimum number of folds required to score a group.
        min_test : int
            Minimum held-out trials for a fold to be scored.
        min_trials : int
            Minimum complete-case rows for a group to be scored (see
            :meth:`_response_lmm_resample`).
        events : list[str], optional
            Restrict to these ``event`` values before grouping; ``None`` (the
            default) uses every event. Lets the caller run a per-event formula
            set without refitting the others.
        """
        def procedure(coded_formulas, df_coded):
            return analysis.crossval_lmm(
                df_coded, coded_formulas, response_col, reference=reference,
                fold_col=fold_col, min_subjects=min_subjects,
                min_test=min_test)

        return self._response_lmm_resample(procedure, formulas, group_by,
                                           response_col, min_trials=min_trials,
                                           events=events)

    def response_lmm_jackknife(self, formulas, group_by, response_col='response',
                               reference='full', fold_col='subject',
                               min_subjects=3, min_trials=0, events=None):
        """In-sample-influence ΔR² by leave-one-fold-out jackknife per group.

        See :meth:`_response_lmm_resample` for the orchestration; this binds the
        scoring arguments of :func:`iblnm.analysis.jackknife_lmm`.

        Parameters
        ----------
        formulas : dict[str, str]
            One comparison set: a flat ``{name: formula_template}`` mapping (a
            ``reference`` key plus drop-one variants), each template containing
            ``{response}``, filled with ``response_col``.
        group_by : list[str]
            Columns whose unique combinations each get an independent run.
        response_col : str
            Response-magnitude column; also the formula's ``{response}``.
        reference : str
            Key in ``formulas`` naming the model each other model's ΔR² is
            measured against.
        fold_col : str
            Column whose unique values define the leave-one-out folds.
        min_subjects : int
            Minimum number of folds required to score a group.
        min_trials : int
            Minimum complete-case rows for a group to be scored (see
            :meth:`_response_lmm_resample`).
        events : list[str], optional
            Restrict to these ``event`` values before grouping; ``None`` (the
            default) uses every event. Lets the caller run a per-event formula
            set without refitting the others.
        """
        def procedure(coded_formulas, df_coded):
            return analysis.jackknife_lmm(
                df_coded, coded_formulas, response_col, reference=reference,
                fold_col=fold_col, min_subjects=min_subjects)

        return self._response_lmm_resample(procedure, formulas, group_by,
                                           response_col, min_trials=min_trials,
                                           events=events)

    def _response_lmm_resample(self, procedure, formulas, group_by,
                               response_col, min_trials=0, events=None):
        """Run a resampling ``procedure`` per ``group_by`` group.

        Shared orchestration for :meth:`response_lmm_crossval` and
        :meth:`response_lmm_jackknife`. Formats the caller's flat
        ``{name: formula}`` dict with ``response_col``, then for each
        ``group_by`` group codes the trials, reduces them to the complete cases
        across the whole family (drop rows null in any referenced column, so
        every model fits the same rows), skips a group with fewer than
        ``min_trials`` such rows, calls ``procedure(formulas, df_coded)``, tags
        the long-form result with the group columns, and concatenates. Reads no
        ``config.LMM_FORMULAS``.

        Parameters
        ----------
        procedure : callable
            ``(formulas, df_coded) -> pd.DataFrame`` wrapping the analysis-level
            resampling function with its scoring arguments bound.
        formulas : dict[str, str]
            Flat ``{name: formula_template}`` mapping for one comparison set.
        group_by : list[str]
            Columns whose unique combinations each get an independent run.
        response_col : str
            Response-magnitude column; also the formula's ``{response}``.
        min_trials : int
            Minimum complete-case rows for a group to be scored. The default 0
            scores every group; callers raise it for high-parameter families
            (e.g. saturated movement models) that need more data to fit stably.
        events : list[str], optional
            Restrict the modeling frame to these ``event`` values before
            grouping; ``None`` (the default) keeps every event.

        Returns
        -------
        pd.DataFrame
            Long-form ΔR² frame with columns ``[*group_by, 'predictor', 'fold',
            'n_trials', 'r2', 'delta_r2']``.
        """
        df = self._modeling_frame(response_col)
        if events is not None:
            df = df[df['event'].isin(events)]
        cols = [*group_by, 'predictor', 'fold', 'n_trials', 'r2', 'delta_r2']
        formulas = {name: template.format(response=response_col)
                    for name, template in formulas.items()}
        model_cols = analysis.formula_union_columns(
            formulas.values(), df.columns)

        frames = []
        for keys, df_group in df.groupby(group_by):
            group_values = keys if isinstance(keys, tuple) else (keys,)
            df_coded = self._code_lmm_predictors(df_group).dropna(
                subset=model_cols)
            if len(df_coded) < min_trials:
                continue
            result = procedure(formulas, df_coded)
            for col, val in zip(group_by, group_values):
                result[col] = val
            frames.append(result)

        return pd.concat(frames, ignore_index=True)[cols] if frames \
            else pd.DataFrame(columns=cols)

    def response_lmm_effects(self, name, kind, variables=None,
                             response_col='response'):
        """Extract a tidy effect frame from the cached fits of one named model.

        Reads the ``LMMResult``s cached by :meth:`response_lmm_fit` under
        ``(response_col, name, *group_values)``, computes the requested effect
        per group, and tags each row with the group identity (the ``group_by``
        columns from the originating fit call). Names no variable itself.

        Parameters
        ----------
        name : str
            Model name whose cached fits to read.
        kind : str
            ``'emm'`` (estimated marginal means for the ``variables`` factor
            set — one factor → main-effect means, two → interaction grid) or
            ``'coefficients'`` (fixed-effects table with ``ci_lower`` /
            ``ci_upper``; ``variables`` ignored).
        variables : sequence of str, optional
            For ``'emm'``, the factor list to cross. Required for ``'emm'``.
        response_col : str
            Response-magnitude column; selects the registry entries.

        Returns
        -------
        pd.DataFrame
            Long-form effect frame; columns include the ``group_by`` identity
            columns recovered from the registry keys.
        """
        df = self._modeling_frame(response_col)

        frames = []
        for keys, _ in df.groupby(self._lmm_group_by):
            group_values = keys if isinstance(keys, tuple) else (keys,)
            fit = self.lmm_fits.get((response_col, name, *group_values))
            if fit is None:
                continue
            effect = self._extract_lmm_effect(fit, kind, variables)
            for col, val in zip(self._lmm_group_by, group_values):
                effect[col] = val
            frames.append(effect)

        return pd.concat(frames, ignore_index=True) if frames \
            else pd.DataFrame()

    @staticmethod
    def _extract_lmm_effect(fit, kind, variables=None):
        """Compute one tidy effect frame from a single cached ``LMMResult``.

        ``'emm'`` returns :func:`analysis.compute_marginal_means` over the
        caller's ``variables`` factor list; ``'coefficients'`` returns the
        fixed-effects table with the term as a column and Wald CIs appended.
        """
        if kind == 'emm':
            if not variables:
                raise ValueError("kind='emm' requires a `variables` factor list")
            return analysis.compute_marginal_means(fit, list(variables))
        if kind == 'coefficients':
            coef = fit.summary_df.copy()
            coef['ci_lower'] = coef['Coef.'] - 1.96 * coef['Std.Err.']
            coef['ci_upper'] = coef['Coef.'] + 1.96 * coef['Std.Err.']
            return coef.rename_axis('term').reset_index()
        raise ValueError(
            f"kind must be 'emm' or 'coefficients', got {kind!r}")

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



        rows = {}
        has_fiber_idx = 'fiber_idx' in self.recordings.columns

        for rec, ps in self:
            eid = rec['eid']
            brain_region = rec['brain_region']
            hemisphere = rec['hemisphere']
            target_nm = rec['target_NM']
            fiber_idx = int(rec['fiber_idx']) if has_fiber_idx else 0

            # Load H5 if responses not yet available
            if not ps.responses or not hasattr(ps, 'trials') or ps.trials is None:
                h5_path = Path(self.h5_dir) / f'{eid}.h5'
                if not h5_path.exists():
                    print(f"  H5 file not found: {h5_path}")
                    continue
                ps.load_h5(h5_path, groups=['trials', 'photometry'])

            if brain_region not in ps.responses:
                continue

            vec = ps.get_response_vector(
                brain_region=brain_region, hemisphere=hemisphere, **kwargs,
            )
            rows[(eid, target_nm, fiber_idx)] = vec

            # Discard raw data to free memory
            ps.responses = {}
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
        events = self._merge_trial_regressors()
        events = events.query('choice != 0 and response_time > 0.05')

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

        Reads from ``self.performance`` if already loaded, otherwise calls
        ``load_performance(performance_path)``.

        Lateralizes bias and lapse terms to the contra/ipsi frame using each
        recording's hemisphere, matching the side coding in the neural GLM
        (contra = positive). Uses the same ``hemi_sign`` convention as
        ``add_relative_contrast``: ``{'l': 1, 'r': -1}``.

        - ``bias``: multiplied by ``hemi_sign``
        - ``lapse_left`` / ``lapse_right`` → ``lapse_contra`` / ``lapse_ipsi``

        Parameters
        ----------
        performance_path : Path or str, optional
            Path to performance.pqt. Only used when ``self.performance`` is
            None. Default: config.PERFORMANCE_FPATH.
        params : list of str, optional
            Columns to include from performance data. Default:
            ``['psych_50_threshold', 'psych_50_bias',
            'psych_50_lapse_left', 'psych_50_lapse_right']``.
            Lapse columns are lateralized to contra/ipsi in the output.

        Returns
        -------
        pd.DataFrame
            (n_recordings, P) aligned to ``self.response_features`` index.
        """
        if self.performance is None:
            from iblnm.config import PERFORMANCE_FPATH
            self.load_performance(
                performance_path if performance_path is not None
                else PERFORMANCE_FPATH
            )
        if params is None:
            params = [
                'psych_50_threshold', 'psych_50_bias',
                'psych_50_lapse_left', 'psych_50_lapse_right',
            ]

        perf = self.performance

        # Extract eid from response_features index and merge
        rf_index = self.response_features.index
        eids = rf_index.get_level_values('eid')
        lookup = perf.set_index('eid')[params]

        # Build aligned DataFrame: one row per recording, matching rf_index
        psych = lookup.reindex(eids)
        psych.index = rf_index

        # Lateralize bias and lapse to contra/ipsi frame
        psych = self._lateralize_psychometric(psych)

        self.psychometric_features = psych
        return psych

    def _lateralize_psychometric(self, psych):
        """Convert left/right psychometric params to contra/ipsi frame.

        Parameters
        ----------
        psych : pd.DataFrame
            Psychometric features indexed like response_features.

        Returns
        -------
        pd.DataFrame
            Same shape, with bias flipped by hemisphere and lapse columns
            renamed to contra/ipsi.
        """
        # Look up hemisphere per recording
        recs = self.recordings
        join_cols = [c for c in ['eid', 'target_NM', 'fiber_idx']
                     if c in psych.index.names]
        hemi_lookup = (
            recs[join_cols + ['hemisphere']]
            .drop_duplicates(subset=join_cols)
            .set_index(join_cols)['hemisphere']
        )
        hemi = hemi_lookup.reindex(psych.index)
        # hemi_sign: same convention as add_relative_contrast
        hemi_sign = hemi.map({'l': 1, 'r': -1}).fillna(1)

        psych = psych.copy()

        # Flip bias sign for right hemisphere
        if 'psych_50_bias' in psych.columns:
            psych['psych_50_bias'] = psych['psych_50_bias'] * hemi_sign.values

        # Swap lapse_left/lapse_right → lapse_contra/lapse_ipsi
        has_left = 'psych_50_lapse_left' in psych.columns
        has_right = 'psych_50_lapse_right' in psych.columns
        if has_left and has_right:
            is_left_hemi = (hemi == 'l').values
            lapse_left = psych['psych_50_lapse_left'].values.copy()
            lapse_right = psych['psych_50_lapse_right'].values.copy()

            # Left hemi: contra=right, ipsi=left
            # Right hemi: contra=left, ipsi=right
            contra = np.where(is_left_hemi, lapse_right, lapse_left)
            ipsi = np.where(is_left_hemi, lapse_left, lapse_right)

            psych = psych.drop(columns=['psych_50_lapse_left',
                                         'psych_50_lapse_right'])
            psych['psych_50_lapse_contra'] = contra
            psych['psych_50_lapse_ipsi'] = ipsi

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

    def response_anovaRM_fit(self, response_col='response',
                                    min_subjects=2, min_trials=10):
        """Run repeated-measures ANOVA on subject-mean response magnitudes.

        For each (target_NM, event) group, aggregates trial-level data to
        subject means by (contrast, side, feedbackType), then runs a 3-way
        repeated-measures ANOVA via ``anova_rm``.

        Requires ``self.response_magnitudes`` and ``self.trial_regressors``
        to be populated.

        Parameters
        ----------
        response_col : str
            Column name for the response magnitude.
        min_subjects : int
            Minimum subjects per group to attempt the ANOVA.
        min_trials : int
            Minimum trials per subject x condition cell. Cells with fewer
            trials are dropped before aggregation.

        Returns
        -------
        dict
            Keys: (target_NM, event_label) tuples.
            Values: ANOVA result DataFrames (from ``anova_rm``).
        """
        from iblnm.analysis import anova_rm

        df = self._modeling_frame(response_col)

        results = {}
        for (target_nm, event), df_group in df.groupby(['target_NM', 'event']):
            if df_group['subject'].nunique() < min_subjects:
                continue
            event_label = event.replace('_times', '')

            # Aggregate to subject means per condition cell
            group_cols = ['subject', 'contrast', 'side', 'feedbackType']
            cell_counts = df_group.groupby(group_cols)[response_col].count()
            # Drop cells with too few trials
            valid_cells = cell_counts[cell_counts >= min_trials].reset_index()
            if len(valid_cells) == 0:
                continue
            subject_means = (
                df_group.merge(valid_cells[group_cols], on=group_cols, how='inner')
                .groupby(group_cols, as_index=False)[response_col]
                .mean()
            )
            if subject_means['subject'].nunique() < min_subjects:
                continue

            table = anova_rm(
                subject_means, response_col, 'subject',
                ['contrast', 'side', 'feedbackType'],
            )
            results[(target_nm, event_label)] = table

        self.anova_results = results
        return results

    def get_trial_regressors(self) -> pd.DataFrame:
        """Collect per-trial predictors for every in-scope session.

        Reads each session's H5 file directly (group ``trials`` and, when
        present, ``wheel/responses/velocity``) and assembles one row per
        ``eid × trial``. Derived timing columns are NaN for a session when
        the underlying event-time columns are absent; ``peak_velocity`` is
        NaN when the wheel group is missing or holds no finite samples.

        Returns
        -------
        pd.DataFrame
            Columns: ``eid, trial, signed_contrast, contrast, stim_side,
            choice, feedbackType, probabilityLeft, reaction_time,
            movement_time, response_time, peak_velocity``. Stored on
            ``self.trial_regressors``.
        """
        from tqdm import tqdm

        frames = []
        for eid in tqdm(self.recordings['eid'].unique(),
                        desc="Collecting trial regressors"):
            h5_path = Path(self.h5_dir) / f'{eid}.h5'
            with h5py.File(h5_path, 'r') as f:
                trials = _read_dataframe(f['trials'])
                wheel_vel = (f['wheel/responses/velocity'][:]
                             if 'wheel/responses/velocity' in f else None)

            df = analysis.build_trial_regressors(trials, wheel_vel)
            df.insert(0, 'eid', eid)
            frames.append(df)

        self.trial_regressors = pd.concat(frames, ignore_index=True)
        return self.trial_regressors

