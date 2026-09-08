import operator
import warnings
from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

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
    ANALYSIS_QC_BLOCKERS, BASELINE_WINDOW, DDM_HMM_DIR, EIDS_TO_DROP,
    EVENT_COMPLETENESS_THRESHOLD, IBL_QC_VALUES,
    LABEL2EVENT, LENGTH_MISMATCH_THRESHOLD, LP_QC_LABELS,
    MIN_NTRIALS, MIN_PERFORMANCE, MIN_TRIALS_PERSESSION,
    MOVEMENT_EVENTS, MOVEMENT_RESPONSE_WINDOW,
    OLS_PERSESSION_COLUMNS, PERSESSION_FDR_GROUP_COLS,
    PERSESSION_PVAL_N_BOOTSTRAP, PERSESSION_PVAL_SEED,
    PHOTOMETRY_BANDS, PHOTOMETRY_QC_THRESHOLDS,
    POSE_MEASURES, QCVAL2NUM,
    PREPROCESSING_PIPELINES, QC_METRICS_KWARGS, QC_RAW_METRICS,
    QC_SLIDING_AGG, QC_SLIDING_KWARGS, QC_SLIDING_METRICS,
    QC_UNDETRENDED_METRICS, REQUIRED_CONTRASTS,
    RESPONSE_EVENTS,
    RESPONSE_VARCOMP_SUMMARY_COLUMNS, RESPONSE_VARCOMP_VIOLIN_COLUMNS,
    RESPONSE_WINDOW,
    RESPONSE_WINDOWS, SESSIONS_H5_DIR, STIM_ONSET_EVENT,
    SESSION_TYPES_TO_ANALYZE, SUBJECTS_TO_EXCLUDE, TARGETNMS_TO_ANALYZE,
    VIDEO_QC_COLS, VIDEO_QC_QUALITY_COLS, VIDEO_QC_PROBLEM_COLS,
    WHEEL_FS, WHEEL_RESPONSE_EVENTS, WHEEL_RESPONSE_WINDOW, POSE_FS,
    store_raw,
    PERSESSION_REGRESSORS,
)
from iblnm.analysis import (
    get_responses, compute_response_magnitude, compute_masked_fraction,
    movement_delta, movement_trace,
    per_third_crosscorr, resample_pose, resample_signal,
    fit_measurement_error_varcomp, summarize_posterior,
)
from tqdm import tqdm

from iblnm import analysis
from iblnm import task
from iblnm.task import compute_trial_contrasts
from iblnm.util import (
    LOG_COLUMNS, deduplicate_log, enforce_schema, fix_catalog,
    resolve_duplicate_group, validate_parallel_lists,
)
from iblnm.validation import (
    MissingExtractedData, MissingRawData, MissingLP, MissingVideoTimestamps,
    MissingMotionEnergy,
    InsufficientTrials, BlockStructureBug, MissingBlockInfo,
    IncompleteEventTimes, MissingFormula, TrialsNotInPhotometryTime,
    QCValidationError, AmbiguousRegionMapping,
    VideoLengthError,
)

# Per-mouse drop-one significance table: one row per (target_NM, event,
# predictor, subject) cell, pooling the cell's sessions by bootstrap resampling
# their per-session donor null ΔR² vectors.
RESPONSE_OLS_MOUSE_PVAL_COLUMNS = [
    'target_NM', 'event', 'predictor', 'subject', 'mean_delta_r2', 'p_value',
    'q_value', 'n_sessions',
]

# One recording's magnitude rows, before the trial regressors are merged onto
# them: the `config.RESPONSE_MAGNITUDE_COLUMNS` entries that come from the
# response cut and the recording's identity rather than from the trials table.
_RECORDING_MAGNITUDE_COLUMNS = [
    'eid', 'subject', 'session_type', 'NM', 'target_NM', 'brain_region',
    'hemisphere', 'event', 'trial', 'response', 'masked_fraction',
]


class DonorFrame(NamedTuple):
    """One session's prepared contribution to the cross-session swap null.

    Built from the trials table alone, so it has no event and no region: one
    frame serves every focal recording-event the session may donate to.
    ``target_NM`` is the whole parallel-list column of the donating session
    rather than one recording's entry, for the same reason.
    """

    eid: str
    subject: str
    target_NM: tuple[str, ...]
    frame: pd.DataFrame


def resolve_event_family(formulas: dict, event: str) -> dict[str, str]:
    """Select the formula family one event is fitted against.

    Two family shapes are in use. A flat family maps model name to formula
    template and is shared by every event (``LMM_FORMULAS['persession']``); an
    event-keyed family nests one such mapping per event, because the predictors
    available differ by event — reward is only known at feedback
    (``LMM_FORMULAS['task_reliability']`` and the movement families).

    Parameters
    ----------
    formulas : dict
        Either ``{name: formula_template}`` or ``{event: {name: template}}``.
    event : str
        Event whose family is wanted.

    Returns
    -------
    dict[str, str]
        That event's ``{name: formula_template}`` mapping. Resolving an
        already-resolved flat family returns it unchanged, so the call is
        idempotent and safe to repeat down a call chain.

    Raises
    ------
    iblnm.validation.MissingFormula
        ``formulas`` is event-keyed and does not name ``event``, which would
        otherwise leave the event fitted against nothing.
    """
    if all(isinstance(template, str) for template in formulas.values()):
        return formulas
    if event not in formulas:
        raise MissingFormula(event, formulas)
    return formulas[event]


# Which sessions may stand in for a focal one in the cross-session swap null.
# Cohort (target_NM) is not filtered by default: the swap replaces trial data,
# not photometry, and the IBL task is standardized and interleaved across
# cohorts, so any session's trial sequence is a valid stand-in. Excluding the
# focal subject rather than only the focal session is the default because a
# subject's own sessions share its behavioral idiosyncrasies, which is the very
# structure the null is meant to be free of. Both sides carry a parallel list of
# target NMs — a session recording two regions has two — so `same_target` asks
# whether the two sessions share any target rather than whether they name the
# same one, which is the same test whenever either side records a single
# region.
_SESSION_DONOR_SCOPES = {
    'exclude_session': lambda focal, donor: donor.eid != focal.eid,
    'exclude_subject': lambda focal, donor: donor.subject != focal.subject,
    'same_target': lambda focal, donor: (
        donor.subject != focal.subject
        and not set(donor.target_NM).isdisjoint(focal.target_NM)),
}


def assemble_mouse_pvalue_table(
    observed: pd.DataFrame,
    n_bootstrap: int = 1000,
    random_state: int | None = 0,
) -> pd.DataFrame:
    """Pool per-session drop-one ΔR² into a per-mouse permutation p-value table.

    Pure assembler: groups the observed drop-one frame by
    ``(target_NM, event, predictor, subject)`` and tests each mouse's pooled
    ΔR² against its sessions' donor null vectors by bootstrap resampling
    (:func:`iblnm.analysis.bootstrap_pooled_pvalue`). A mouse's sessions can
    carry different-length null vectors — each is an arbitrarily ordered donor
    set — so the pooling resamples one draw per session rather than aligning
    columns; the vectors need not share a length.

    Parameters
    ----------
    observed : pd.DataFrame
        Per-session fits (``config.OLS_PERSESSION_COLUMNS``), one row per
        ``(eid, target_NM, brain_region, event, predictor)`` carrying the
        in-sample ``delta_r2`` beside the ``null`` vector it was scored against
        and the ``n_donors`` that null was built from. Vector lengths may differ
        across sessions — each is an arbitrarily ordered donor set — and a row
        whose ``null`` is empty was not scorable and is dropped from its group.
        The pooled null draws one value per session, so it is no better resolved
        than its coarsest session and the cell's p-value floor comes from the
        smallest pooled ``n_donors`` (:func:`_floor_pvalue`).
    n_bootstrap : int
        Pooled-null draws per cell. It does not set the p-value floor; the
        donor counts do.
    random_state : int or None
        Seed for the bootstrap rng, created once and reused across cells.

    Returns
    -------
    pd.DataFrame
        One row per scorable ``(target_NM, event, predictor, subject)`` cell in
        ``RESPONSE_OLS_MOUSE_PVAL_COLUMNS`` order. ``mean_delta_r2`` is the
        pooled observed statistic, ``p_value`` the one-sided (greater) bootstrap
        p floored on the donor counts, and ``n_sessions`` the pooled session
        count. ``q_value`` is present but NaN — the caller fills it with
        :func:`iblnm.analysis.add_fdr_qvalues`, which chooses the correction
        families.
    """
    rng = np.random.default_rng(random_state)
    rows = []
    group_keys = ['target_NM', 'event', 'predictor', 'subject']
    for (target_NM, event, predictor, subject), group in observed.groupby(
            group_keys, sort=True):
        scorable = [
            (row['delta_r2'], np.asarray(row['null']), row['n_donors'])
            for _, row in group.iterrows()
            if np.size(row['null'])
        ]
        if not scorable:
            continue
        observed_by_stratum = [delta_r2 for delta_r2, _, _ in scorable]
        null_by_stratum = [null for _, null, _ in scorable]
        mean_delta_r2, p_value = analysis.bootstrap_pooled_pvalue(
            observed_by_stratum, null_by_stratum, rng=rng,
            n_bootstrap=n_bootstrap, alternative='greater')
        rows.append({
            'target_NM': target_NM, 'event': event, 'predictor': predictor,
            'subject': subject, 'mean_delta_r2': mean_delta_r2,
            'p_value': _floor_pvalue(
                p_value, min(count for _, _, count in scorable)),
            'n_sessions': len(scorable),
        })
    return pd.DataFrame(rows, columns=RESPONSE_OLS_MOUSE_PVAL_COLUMNS)


def _floor_pvalue(p_value: float, n_donors: int) -> float:
    """Lift a permutation p-value to the resolution its donor pool supports.

    A donor null is bootstrap-resampled to a fixed length, so the add-one
    correction inside :func:`iblnm.analysis.permutation_pvalue` floors at
    1 / (n_draws + 1) — a resolution the resampling manufactured. The pool of
    ``n_donors`` distinct donors behind those draws is what the null actually
    resolves, so the reported p is floored at 1 / (n_donors + 1) instead.
    """
    return max(p_value, 1 / (n_donors + 1))


# =============================================================================
# HDF5 save/load helpers
# =============================================================================
#
# These module-level functions handle the on-disk layout for each top-level
# group. `save_h5` / `load_h5` on PhotometrySession are thin dispatchers over
# the _SAVE_HANDLERS / _LOAD_HANDLERS registries at the bottom of this block.
#
# Below them sit three save/load pairs keyed by the data structure they carry
# rather than by modality — _save_time_series, _save_peri_event_matrix,
# _save_scalars and their load counterparts. They are pure: they take one H5
# group and a payload, with no coupling to PhotometrySession. Creating the
# group, looping over regions or labels, and extracting the payload from the
# session are the orchestrators' job (_save_photometry, _save_video, ...).


def _concat_frames(frames: list[pd.DataFrame], columns: list[str]) -> pd.DataFrame:
    """Concatenate collected frames, falling back to an empty typed frame.

    Empty frames are dropped first: a recording that matched nothing carries no
    dtypes for pandas to reconcile, and concatenating it would widen the result
    to object. A walk that matched nothing anywhere still returns the named
    columns, so a caller's column access works either way.
    """
    populated = [frame for frame in frames if not frame.empty]
    if not populated:
        return pd.DataFrame(columns=columns)
    return pd.concat(populated, ignore_index=True)[columns]


# `config.OLS_PERSESSION_COLUMNS` less `q_value`: a session scores its own rows
# against its own null, but the false-discovery-rate correction pools every
# session of an (event, predictor) family, so the group fills that column after
# collecting these frames.
_SESSION_OLS_COLUMNS = [column for column in OLS_PERSESSION_COLUMNS
                        if column != 'q_value']


def _dropone_rows(fits: dict, n_trials: int,
                  reference: str = 'full') -> pd.DataFrame:
    """One cell's drop-one ΔR² carrying each dropped regressor's own weight.

    The dropped ``predictor`` and the reference model's ``regressor`` are the
    same names, so differencing the family
    (:func:`iblnm.analysis.dropone_delta_r2`) and reading its weights
    (:func:`_coefficient_rows`) give one row per predictor once joined. The
    reference's R² repeats across those rows, which ``r2_full`` /
    ``r2_full_adj`` say.

    Parameters
    ----------
    fits : dict
        Model name -> fitted ``statsmodels`` result, every member of one
        event's family fitted on the same complete-case rows. None may be
        ``None``; the caller drops a cell with a degenerate member before
        calling.
    n_trials : int
        Rows every model was fit on, which the adjusted R² is penalized over.
    reference : str
        Full-model key each reduced model's ΔR² is measured against and whose
        weights are read.

    Returns
    -------
    pandas.DataFrame
        ``predictor, r2_full, r2_full_adj, delta_r2, delta_r2_adj, coef,
        coef_se, n_trials``, one row per dropped predictor.
    """
    scores = analysis.dropone_delta_r2(
        {name: (fit.rsquared, fit.df_model) for name, fit in fits.items()},
        n_trials, reference)
    weights = _coefficient_rows(fits[reference], n_trials).rename(
        columns={'regressor': 'predictor'})
    return (scores.rename(columns={'r2': 'r2_full', 'r2_adj': 'r2_full_adj'})
            .merge(weights, on='predictor', how='left'))


def _score_against_null(rows: pd.DataFrame, nulls: dict[str, np.ndarray],
                        n_donors: int) -> pd.DataFrame:
    """Score one cell's observed drop-one ΔR² against its own donor null.

    Parameters
    ----------
    rows : pandas.DataFrame
        One cell's fitted rows, one per dropped ``predictor``, carrying the
        observed ``delta_r2``.
    nulls : dict[str, numpy.ndarray]
        That cell's null ΔR² vector per predictor, from
        :func:`iblnm.analysis.permutation_null_delta_r2`. A predictor no donor
        was scorable for has an empty vector.
    n_donors : int
        Size of the donor pool the nulls were built from, which sets the
        p-value floor (:func:`_floor_pvalue`).

    Returns
    -------
    pandas.DataFrame
        ``rows`` with ``null``, ``p_value`` and ``n_donors`` added. ``null`` is
        an object column of float32 arrays — parquet stores it as a list column
        and the per-mouse pooling reads it back rather than refitting.
        ``p_value`` is the one-sided (greater) permutation p, NaN where the null
        is empty: an unscorable cell keeps its fit rather than dropping out.
    """
    null = [np.asarray(nulls[predictor], dtype=np.float32)
            for predictor in rows['predictor']]
    p_value = [
        _floor_pvalue(analysis.permutation_pvalue(delta_r2, vector, 'greater'),
                      n_donors) if vector.size else np.nan
        for delta_r2, vector in zip(rows['delta_r2'], null)
    ]
    return rows.assign(
        null=pd.Series(null, index=rows.index, dtype=object),
        p_value=p_value, n_donors=n_donors)


def _coefficient_rows(fit, n_trials: int) -> pd.DataFrame:
    """Main-effect weight and SE per regressor from one fitted model.

    Reads ``fit.params`` / ``fit.bse`` for each bare regressor name in
    ``config.PERSESSION_REGRESSORS`` present in the design (a regressor absent
    from this event's model contributes no row). No refit — ``fit`` is the
    already-fitted reference model.
    """
    present = [name for name in PERSESSION_REGRESSORS
               if name in fit.params.index]
    return pd.DataFrame({
        'regressor': present,
        'coef': fit.params[present].values,
        'coef_se': fit.bse[present].values,
        'n_trials': n_trials,
    })


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


def _decode(value):
    """Return an H5 string attr as str, leaving anything else alone."""
    return value.decode() if isinstance(value, bytes) else value


def _read_dataframe(h5_group):
    """Read all datasets under `h5_group` into a DataFrame, decoding bytes."""
    data = {}
    for col in h5_group:
        values = h5_group[col][:]
        if values.dtype.kind == 'S':
            values = values.astype(str)
        data[col] = values
    return pd.DataFrame(data)


# Attrs earlier versions wrote beside every stored product, naming the
# parameters that produced it. Nothing writes them now, but files built before
# that change still carry them, so readers of a group's own attrs skip them.
_STAMP_ATTRS = frozenset({'spec_json', 'built_at'})

# The last path component of every product name. A subgroup carrying one of
# these names is another product, not this one's data — which is what lets
# `stored_product_exists` tell a region's stored raw bands (named after the
# bands) from the `qc/` group left sitting alone when the raw was not kept.
_PRODUCT_SUBGROUPS = frozenset(
    {'raw', 'preprocessed', 'responses', 'qc', 'manual_qc'})


# The self.photometry key holding the preprocessed signal — the payload of the
# 'photometry/preprocessed' product. The raw bands sit beside it under their own
# names ('GCaMP', 'Isosbestic').
PREPROCESSED_BAND = 'GCaMP_preprocessed'
# The wheel's H5 label. It has one channel, so the label level carries no
# information for it, but keeping it makes every modality's handlers walk labels
# the same way. The label names the preprocessed product — velocity — not the
# raw encoder position stored underneath it.
WHEEL_LABEL = 'velocity'
# The event the wheel matrix is cut from, hence its only event coordinate.
_WHEEL_T0_EVENT = WHEEL_RESPONSE_EVENTS[0]
_METADATA_NONE_SENTINEL = '__none__'
_ERROR_FIELDS = ('eid', 'error_type', 'error_message', 'traceback', 'product')
# Never persisted to errors/ — see _save_errors.
# Session columns holding one entry per recording, exploded together so a
# region never parts company with its hemisphere and target NM.
PARALLEL_COLS = ['brain_region', 'hemisphere', 'target_NM']

_UNRECORDED_ERROR_TYPES = frozenset({'BlockingIOError'})
_RESPONSES_RESERVED_KEYS = {'times', 'trials'}


def _save_metadata(session, h5_file):
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


def _read_metadata(h5_file) -> dict:
    """Read the `metadata` group into the session row it was written from.

    Parameters
    ----------
    h5_file : h5py.File
        An open session file. One with no `metadata` group yields `{}`, which
        is how a file written before the group existed drops out of a catalog
        rebuilt from the store.

    Returns
    -------
    dict
        Field -> value for every `PhotometrySession._METADATA_FIELDS` entry the
        group holds. List fields come back as lists of str, empty when the
        dataset is absent; the `__none__` sentinel comes back as None. Dates
        stay ISO strings, the form the catalog carries them in.
    """
    if 'metadata' not in h5_file:
        return {}
    grp = h5_file['metadata']
    row = {}
    for attr, is_list in PhotometrySession._METADATA_FIELDS:
        if is_list:
            row[attr] = [_decode(value) for value in grp[attr][:]] \
                if attr in grp else []
        elif attr in grp.attrs:
            value = grp.attrs[attr]
            value = _decode(value) if isinstance(value, bytes) else (
                value.item() if hasattr(value, 'item') else value)
            row[attr] = None if value == _METADATA_NONE_SENTINEL else value
    return row


def _load_metadata(session, h5_file):
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


def _write_error_entries(group: h5py.Group, entries: list[dict]) -> None:
    """Replace `group`'s error datasets with one row per entry in `entries`.

    Writes `_ERROR_FIELDS` as parallel string datasets. Existing datasets are
    deleted first: an error group records a single build attempt, so the last
    attempt replaces the previous one rather than accumulating alongside it.
    """
    for col in _ERROR_FIELDS:
        if col in group:
            del group[col]
        group.create_dataset(
            col,
            data=[str(entry.get(col, '') or '') for entry in entries],
            dtype=h5py.string_dtype(),
        )


def _read_error_entries(group: h5py.Group) -> list[dict]:
    """Read one group's error rows back as dicts, or [] if it holds none.

    An empty `product` field means the entry was logged without a product and
    comes back as None, as does a `product` dataset missing altogether — the
    shape of every error group written into the store before the field existed.
    """
    if 'eid' not in group:
        return []
    n_entries = len(group['eid'])
    columns = {
        col: ([v.decode() if isinstance(v, bytes) else str(v)
               for v in group[col][:]] if col in group else [None] * n_entries)
        for col in _ERROR_FIELDS
    }
    return [
        {col: (values[i] or None) if col == 'product' else values[i]
         for col, values in columns.items()}
        for i in range(n_entries)
    ]


def _save_errors(session, h5_file):
    """Write `session.errors` into `errors/`, mirroring the product tree.

    Entries are sorted by their `product` field: each product's entries replace
    whatever `errors/{product}` held before, and products absent from this
    attempt keep their existing groups. Entries with no product go to the
    `errors/` root.

    `BlockingIOError` is never written: it is a transient file lock that
    `process()` retries, not a failed build, and recording it would mark the
    session permanently failed under the absent-data + present-error rule.
    """
    # Empty group signals "no errors" — distinguishable from "not yet written".
    grp = h5_file.require_group('errors')
    by_product = defaultdict(list)
    for entry in session.errors:
        if entry['error_type'] not in _UNRECORDED_ERROR_TYPES:
            by_product[entry['product']].append(entry)
    for product, entries in by_product.items():
        _write_error_entries(
            grp if product is None else grp.require_group(product), entries
        )


def read_error_tree(h5_file: h5py.File) -> list[dict]:
    """Read every error entry under `errors/`, walking the product tree.

    Returns entries grouped by product — those logged without a product (the
    `errors/` root) first, then each product group in depth-first order. An
    open file with no `errors/` group yields [].
    """
    if 'errors' not in h5_file:
        return []
    root = h5_file['errors']
    descendants = []
    root.visit(descendants.append)
    return _read_error_entries(root) + [
        entry for path in descendants
        if isinstance(root[path], h5py.Group)
        for entry in _read_error_entries(root[path])
    ]


def _load_errors(session, h5_file):
    if 'errors' not in h5_file:
        return
    session.errors = read_error_tree(h5_file)


def _save_trials(session, h5_file):
    """Write the trials table verbatim to `trials/table`, performance beside it.

    Every column is stored, ONE's own and the ones `load_trials` derives. Trial
    identity travels in the `trial` column, not in the index: `_read_dataframe`
    rebuilds a fresh RangeIndex, so a session whose trials were filtered before
    extraction would otherwise silently realign against its responses.

    `trials/performance` is written whenever the session holds it, independently
    of the table, so reading a stored table and scoring it does not depend on
    both being in memory at once.
    """
    grp = h5_file.require_group('trials')
    if hasattr(session, 'trials'):
        _write_dataframe(_replace_group(grp, 'table'), session.trials)
    if hasattr(session, 'performance'):
        _save_performance(_replace_group(grp, 'performance'),
                          session.performance)


def _load_trials(session, h5_file):
    if 'trials/table' in h5_file:
        session.trials = _read_dataframe(h5_file['trials/table'])
    if 'trials/performance' in h5_file:
        session.performance = _load_performance(h5_file['trials/performance'])


def _save_performance(group: h5py.Group, performance: dict) -> None:
    """Write the per-session behavioral scalars into `group`.

    Every value is a scalar attr except `contrasts`, the sorted list of contrast
    levels the session presented, which becomes a dataset — the one entry that
    `_save_scalars` alone could not carry.

    Parameters
    ----------
    group : h5py.Group
        Destination group, created (and any predecessor replaced) by the caller.
    performance : dict
        Metric name -> value, including the `contrasts` list.
    """
    group.create_dataset('contrasts',
                         data=np.asarray(performance['contrasts'], dtype=np.float64))
    _save_scalars(group, {key: value for key, value in performance.items()
                          if key != 'contrasts'})


def _load_performance(group: h5py.Group) -> dict:
    """Read the behavioral scalars written by `_save_performance`."""
    return _load_scalars(group) | {'contrasts': group['contrasts'][:].tolist()}


# ----- Photometry sub-handlers (pure: parent_group + payload only) -----

def _save_time_series(
    group: h5py.Group, obj: pd.Series | pd.DataFrame,
) -> None:
    """Write a time-indexed pandas object into `group`.

    The index becomes the `times` dataset; each signal becomes its own dataset
    beside it, named after the DataFrame column it came from. A Series has no
    column name to use, so it is written as `signal`.

    Parameters
    ----------
    group : h5py.Group
        Destination group, created (and any predecessor replaced) by the caller.
    obj : pandas.Series or pandas.DataFrame
        Time-indexed signal(s); index and values are stored as float64.
    """
    frame = obj.to_frame('signal') if isinstance(obj, pd.Series) else obj
    group.create_dataset('times', data=frame.index.to_numpy(dtype=np.float64))
    for name, column in frame.items():
        group.create_dataset(
            name, data=column.to_numpy(dtype=np.float64),
            compression='gzip', compression_opts=4,
        )


def _load_time_series(group: h5py.Group) -> pd.Series | pd.DataFrame:
    """Read a time-indexed pandas object written by `_save_time_series`.

    Returns a Series when the group holds a single signal dataset beside
    `times` and a DataFrame when it holds several, so a one-signal product
    (photometry, one group per region) and a multi-signal one both fit. The
    Series is unnamed: the dataset name is the placeholder `signal`, not data.
    """
    signals = {name: group[name][:].astype(np.float64)
               for name in group if name != 'times'}
    times = group['times'][:]
    if len(signals) == 1:
        return pd.Series(next(iter(signals.values())), index=times)
    return pd.DataFrame(signals, index=times)


def _save_peri_event_matrix(
    group: h5py.Group, da: xr.DataArray,
) -> None:
    """Write a peri-event matrix into `group`.

    The `time` and `trial` coords become the `times` and `trials` datasets;
    each event's `(trial, time)` slice becomes a dataset named after the event,
    so the event axis is readable without loading the others.

    Parameters
    ----------
    group : h5py.Group
        Destination group, created (and any predecessor replaced) by the caller.
    da : xarray.DataArray
        Dims `(event, trial, time)`. The `trial` coord is the raw ONE
        trials-table index and need not be contiguous.
    """
    group.create_dataset('times', data=da.coords['time'].values)
    group.create_dataset('trials', data=da.coords['trial'].values)
    for event_name in da.coords['event'].values:
        group.create_dataset(
            event_name, data=da.sel(event=event_name).values.astype(np.float64),
            compression='gzip', compression_opts=4,
        )


def _load_peri_event_matrix(group: h5py.Group) -> xr.DataArray:
    """Read a peri-event matrix written by `_save_peri_event_matrix`.

    Every dataset other than the reserved coord datasets
    (`_RESPONSES_RESERVED_KEYS`) is one event's `(trial, time)` slice, stacked
    back into dims `(event, trial, time)`.
    """
    event_names = [k for k in group.keys() if k not in _RESPONSES_RESERVED_KEYS]
    return xr.DataArray(
        np.stack([group[name][:].astype(np.float64) for name in event_names]),
        dims=['event', 'trial', 'time'],
        coords={
            'event': event_names,
            'trial': group['trials'][:],
            'time':  group['times'][:],
        },
    )


def _read_label_products(
    modality_group: h5py.Group, product: str, read,
) -> dict:
    """Read every `{label}/{product}` subgroup into a label -> payload dict.

    Serves any modality: the label is a brain region under `photometry/`, the
    single wheel channel under `wheel/`, and a movement channel under `video/`.
    Subgroups that do not carry this product are skipped, which is what keeps
    the raw and QC groups sitting beside the labels out of the result.

    Parameters
    ----------
    modality_group : h5py.Group
        A top-level modality group, holding one subgroup per label.
    product : str
        Name of the per-label subgroup to read, e.g. 'responses'.
    read : callable
        Loading primitive applied to each label's subgroup.
    """
    return {label: read(modality_group[f'{label}/{product}'])
            for label in modality_group
            if product in modality_group[label]}


def _read_label_responses(modality_group: h5py.Group) -> dict[str, xr.DataArray]:
    """Read every `{label}/responses` subgroup into a label -> DataArray dict."""
    return _read_label_products(modality_group, 'responses',
                                _load_peri_event_matrix)


def _save_scalars(group: h5py.Group, mapping: dict[str, float]) -> None:
    """Write a flat scalar mapping into `group` as attrs.

    Values are coerced to float, so a metric that could not be computed stays
    a NaN attr rather than a missing key.

    Parameters
    ----------
    group : h5py.Group
        Destination group, created (and any predecessor replaced) by the caller.
    mapping : dict
        Metric name -> value. Where a channel axis exists it is suffixed into
        the name (`n_unique_samples_GCaMP`), so the mapping stays flat.
    """
    for key, value in mapping.items():
        group.attrs[key] = float(value)


def _load_scalars(group: h5py.Group) -> dict[str, float]:
    """Read the scalar attrs written by `_save_scalars`, dropping the stamp."""
    return {key: float(value) for key, value in group.attrs.items()
            if key not in _STAMP_ATTRS}


def _save_manual_qc(group: h5py.Group, labels: dict[str, str]) -> None:
    """Write manual QC verdicts into `group` as attrs.

    One pair serves `photometry/{region}/manual_qc` and `video/manual_qc`: the
    payload is the same small `LP_QC_LABELS`-keyed dict of IBL verdict strings
    either way, and only the parent group differs.

    Attrs are set in place rather than the group being replaced, so writing one
    field leaves the session's other verdicts standing.
    """
    for field, value in labels.items():
        group.attrs[field] = value


def _load_manual_qc(group: h5py.Group) -> dict[str, str]:
    """Read the verdicts written by `_save_manual_qc`, decoding bytes to str."""
    return {field: _decode(group.attrs[field]) for field in LP_QC_LABELS
            if field in group.attrs}


# How a metric's sliding windows are reduced to the one value that is stored.
# Named by string in config.QC_SLIDING_AGG, so the choice reads as data there
# and the callable implementing it lives here.
_QC_AGGREGATORS = {
    'mean': lambda values: values.mean(),
    'q10':  lambda values: values.quantile(0.10),
}


def _aggregate_qc_windows(
    qc_tidy: pd.DataFrame, agg: dict[str, str],
) -> dict[str, dict[str, float]]:
    """Reduce a tidy sliding-QC frame to one scalar per (region, band, metric).

    Parameters
    ----------
    qc_tidy : pandas.DataFrame
        `qc_signals` output: columns `band`, `brain_region`, `metric`, `value`,
        and `window` (the window centre, NaN on the whole-signal row). Only the
        windowed rows are aggregated — the whole-signal row scores a different
        thing and would otherwise be averaged in as if it were a window.
    agg : dict
        Metric name -> aggregator name in `_QC_AGGREGATORS`.

    Returns
    -------
    dict
        Region -> {f'{metric}_{band}': value}. QC is stored per region but not
        per band, so the band is suffixed into the metric name to keep each
        region's mapping flat.
    """
    windows = qc_tidy[qc_tidy['window'].notna()]
    aggregated = {}
    for (region, band, metric), rows in windows.groupby(
            ['brain_region', 'band', 'metric']):
        aggregated.setdefault(region, {})[f'{metric}_{band}'] = (
            _QC_AGGREGATORS[agg[metric]](rows['value']))
    return aggregated


def _read_photometry_preprocessed(
    photometry_group: h5py.Group,
) -> tuple[pd.DataFrame | None, dict[str, dict[str, float]]]:
    """Read every region's preprocessed signal and preprocessing diagnostics.

    Parameters
    ----------
    photometry_group : h5py.Group
        The file's `photometry/` group, holding one subgroup per region.

    Returns
    -------
    pandas.DataFrame or None
        Preprocessed signal, regions as columns on a shared time index. None
        when no region has a stored `preprocessed/` group.
    dict
        Region -> its `bleaching_tau` / `iso_correlation` attrs.
    """
    groups = {region: photometry_group[f'{region}/preprocessed']
              for region in photometry_group
              if 'preprocessed' in photometry_group[region]}
    if not groups:
        return None, {}
    return (pd.DataFrame({region: _load_time_series(group)
                          for region, group in groups.items()}),
            {region: _load_scalars(group) for region, group in groups.items()})


def _raw_bands(photometry: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """The raw bands held in `self.photometry`, without the preprocessed signal.

    The preprocessed product sits in the same dict under `PREPROCESSED_BAND`;
    everything else came from Alyx as a raw band.
    """
    return {band: frame for band, frame in photometry.items()
            if band != PREPROCESSED_BAND}


def _save_raw_bands(
    group: h5py.Group, bands: dict[str, pd.Series],
) -> None:
    """Write one region's raw bands into `group`.

    Each band becomes its own time-series subgroup: acquisition interleaves the
    excitation wavelengths, so the bands of one region carry different sample
    times and cannot share a single index. The bands are written beside — not
    over — the `qc/` subgroup, which is scored from the raw but stored whether
    or not the raw itself is kept.

    Parameters
    ----------
    group : h5py.Group
        The region's `raw` group, required (not replaced) by the caller.
    bands : dict
        Band name (`config.PHOTOMETRY_BANDS`) -> that band's signal for this
        region, indexed by time in seconds.
    """
    for band, signal in bands.items():
        _save_time_series(_replace_group(group, band), signal)


def _load_raw_bands(group: h5py.Group) -> dict[str, pd.Series]:
    """Read one region's raw bands written by `_save_raw_bands`.

    Only the time-series subgroups are read, so the `qc/` group sitting beside
    the bands drops out on its own rather than by name.
    """
    return {band: _load_time_series(group[band]) for band in group
            if 'times' in group[band]}


def _read_photometry_raw(photometry_group: h5py.Group) -> dict[str, pd.DataFrame]:
    """Read every region's raw bands into band -> DataFrame with regions as columns.

    That is the shape `load_raw_photometry` gets from Alyx, so a session reading
    its stored raw is indistinguishable from one that just fetched it.
    """
    by_region = _read_label_products(photometry_group, 'raw', _load_raw_bands)
    bands = {band for signals in by_region.values() for band in signals}
    return {band: pd.DataFrame({region: signals[band]
                                for region, signals in by_region.items()
                                if band in signals})
            for band in bands}


def _read_photometry_qc(
    photometry_group: h5py.Group,
) -> dict[str, dict[str, float]]:
    """Read every region's `raw/qc` attrs into region -> metric -> value.

    Regions with no stored QC are absent from the result rather than present
    with an empty mapping, so a caller can tell "never scored" from "scored".
    """
    return {region: _load_scalars(photometry_group[f'{region}/raw/qc'])
            for region in photometry_group
            if 'raw/qc' in photometry_group[region]}


def _set_if_read(session, attribute: str, mapping: dict) -> None:
    """Assign a label-keyed product to `session`, unless the file had none.

    The label readers return an empty mapping when the modality group holds no
    such product, which is what "absent" looks like on disk. Assigning that
    would put an empty product on the session and stop the load method from
    ever building it.
    """
    if mapping:
        setattr(session, attribute, mapping)


def _save_photometry(session, h5_file):
    photometry_group = h5_file.require_group('photometry')
    preprocessed = session.photometry.get(PREPROCESSED_BAND)

    if hasattr(session, 'neurophotometrics_qc'):
        _save_scalars(
            _replace_group(photometry_group.require_group('neurophotometrics'),
                           'qc'),
            session.neurophotometrics_qc,
        )

    raw_bands = _raw_bands(session.photometry) if store_raw else {}

    responses = getattr(session, 'photometry_responses', {})
    qc = getattr(session, 'photometry_qc', {})
    regions = set(responses) | set(qc) | set(session.photometry_manual_qc)
    if preprocessed is not None:
        regions.update(preprocessed.columns)
    regions.update(*(frame.columns for frame in raw_bands.values()))

    for region in sorted(regions):
        region_group = photometry_group.require_group(region)

        band_signals = {band: frame[region] for band, frame in raw_bands.items()
                        if region in frame.columns}
        if band_signals:
            _save_raw_bands(region_group.require_group('raw'), band_signals)

        if preprocessed is not None and region in preprocessed.columns:
            group = _replace_group(region_group, 'preprocessed')
            _save_time_series(group, preprocessed[region])
            # Diagnostics of the preprocessing run itself, beside the signal.
            for key, value in session.preprocessing_diagnostics.get(
                    region, {}).items():
                group.attrs[key] = float(value)

        if region in responses:
            _save_peri_event_matrix(
                _replace_group(region_group, 'responses'), responses[region],
            )

        # The raw bands themselves may or may not be stored; their QC always is.
        if region in qc:
            _save_scalars(
                _replace_group(region_group.require_group('raw'), 'qc'),
                qc[region],
            )

        if region in session.photometry_manual_qc:
            _save_manual_qc(region_group.require_group('manual_qc'),
                            session.photometry_manual_qc[region])


def _load_photometry(session, h5_file):
    if 'photometry' not in h5_file:
        return
    photometry_group = h5_file['photometry']

    session.photometry.update(_read_photometry_raw(photometry_group))
    preprocessed, diagnostics = _read_photometry_preprocessed(photometry_group)
    if preprocessed is not None:
        session.photometry[PREPROCESSED_BAND] = preprocessed
        session.preprocessing_diagnostics = diagnostics

    # An empty mapping means the file holds no such product, not a product that
    # was built empty, so the attribute is left off rather than set to `{}`.
    _set_if_read(session, 'photometry_responses',
                 _read_label_responses(photometry_group))
    _set_if_read(session, 'photometry_qc',
                 _read_photometry_qc(photometry_group))
    session.photometry_manual_qc = _read_label_products(
        photometry_group, 'manual_qc', _load_manual_qc)
    if 'neurophotometrics/qc' in photometry_group:
        session.neurophotometrics_qc = _load_scalars(
            photometry_group['neurophotometrics/qc'])


def _save_wheel(session, h5_file):
    """Write the wheel's four products under `wheel/{WHEEL_LABEL}/`.

    Raw position and preprocessed velocity sit on different time bases, so each
    is its own time-series group; the responses matrix is written by the same
    peri-event pair every modality uses. The peak velocity is one value per
    trial with no time axis of its own, which is the shape the frame-data pair
    carries.
    """
    wheel_group = h5_file.require_group('wheel')
    label_group = wheel_group.require_group(WHEEL_LABEL)
    if store_raw and hasattr(session, 'wheel_position'):
        _save_time_series(_replace_group(label_group, 'raw'),
                          session.wheel_position)
    if hasattr(session, 'wheel_velocity'):
        _save_time_series(_replace_group(label_group, 'preprocessed'),
                          session.wheel_velocity)
    if hasattr(session, 'wheel_peak_velocity'):
        _save_frame_data(_replace_group(label_group, 'peak_velocity'),
                         session.wheel_peak_velocity)
    for label, responses in getattr(session, 'wheel_responses', {}).items():
        _save_peri_event_matrix(
            _replace_group(wheel_group.require_group(label), 'responses'),
            responses,
        )


def _load_wheel(session, h5_file):
    if 'wheel' not in h5_file:
        return
    wheel_group = h5_file['wheel']
    label_group = wheel_group.get(WHEEL_LABEL)
    if label_group is not None:
        if 'raw' in label_group:
            session.wheel_position = _load_time_series(label_group['raw'])
        if 'preprocessed' in label_group:
            session.wheel_velocity = _load_time_series(
                label_group['preprocessed'])
        if 'peak_velocity' in label_group:
            session.wheel_peak_velocity = _load_frame_data(
                label_group['peak_velocity'])
    _set_if_read(session, 'wheel_responses', _read_label_responses(wheel_group))


# ----- Video / LightningPose sub-handlers (pure: parent_group + payload) -----

LP_QC_NOT_SET = 'NOT_SET'
LP_QC_PASS = 'PASS'


def _save_frame_data(
    group: h5py.Group, data: np.ndarray | pd.DataFrame,
) -> None:
    """Write per-frame or per-trial data into `group`.

    The pair carries values indexed by something other than time, with no index
    of their own: video's three raw datasets, each stored on the camera's own
    frame axis — that is what keeps `video/pose` and `video/motion_energy` free
    of `video/times` as an input — and the wheel's `peak_velocity`, one value
    per trial. A 1-D array becomes the single dataset `values`; a DataFrame
    becomes one dataset per column.

    Only the group's datasets are replaced, not the group itself: the
    `motion_energy` group holds this product's frames alongside the
    `preprocessed`/`responses` subgroups of the movement channel of the same
    name.

    Parameters
    ----------
    group : h5py.Group
        Destination group, required (not replaced) by the caller.
    data : numpy.ndarray or pandas.DataFrame
        One value per camera frame, or one column of them per keypoint field.
    """
    for name in [key for key in group if isinstance(group[key], h5py.Dataset)]:
        del group[name]
    columns = (data.items() if isinstance(data, pd.DataFrame)
               else [('values', data)])
    for name, values in columns:
        group.create_dataset(name, data=np.asarray(values),
                             compression='gzip', compression_opts=4)


def _load_frame_data(group: h5py.Group) -> np.ndarray | pd.DataFrame | None:
    """Read the index-free data written by `_save_frame_data`.

    Returns an array when the group holds the single `values` dataset, a
    DataFrame when it holds one dataset per column, and None when it holds no
    datasets at all — the state of a label group carrying only `preprocessed`
    and `responses`.
    """
    datasets = {name: group[name][:] for name in group
                if isinstance(group[name], h5py.Dataset)}
    if not datasets:
        return None
    if 'values' in datasets:
        return datasets['values']
    return pd.DataFrame(datasets)


def _save_pose_xcorr(group: h5py.Group, xcorr: dict) -> None:
    """Write the paw–wheel cross-correlation into `group`.

    `video/pose/qc` is the one QC product that is not flat scalars — three
    arrays (`functions`, `lags`, `peak_lags`) beside the scalar `drift` — so it
    keeps its own handler pair rather than going through `_save_scalars`.
    """
    for field in ('functions', 'lags', 'peak_lags'):
        group.create_dataset(field, data=np.asarray(xcorr[field],
                                                    dtype=np.float64))
    group.attrs['drift'] = xcorr['drift']


def _load_pose_xcorr(group: h5py.Group) -> dict:
    """Read the cross-correlation written by `_save_pose_xcorr`."""
    return {
        'functions': group['functions'][:].astype(np.float64),
        'lags':      group['lags'][:].astype(np.float64),
        'peak_lags': group['peak_lags'][:].astype(np.float64),
        'drift':     group.attrs['drift'],
    }


def _save_video(session, h5_file):
    """Write the video modality's raw, preprocessed and response products.

    The three raw datasets are fetched independently and are written
    independently: `video/times`, `video/pose` and `video/motion_energy` each
    appear only when that source was loaded, so a session missing one keeps the
    other two. `motion_energy` is both a raw product and a movement channel, so
    its group carries the raw frames beside the channel's `preprocessed` and
    `responses` subgroups.

    `video/manual_qc` sits beside them all and is written only from verdicts
    held on the session, so rebuilding any derived product leaves a stored
    verdict alone.
    """
    grp = h5_file.require_group('video')
    for attribute, name in (('pose_times', 'times'), ('pose', 'pose'),
                            ('motion_energy', 'motion_energy')):
        if store_raw and hasattr(session, attribute):
            _save_frame_data(grp.require_group(name), getattr(session, attribute))
    if hasattr(session, 'video_times_qc'):
        _save_scalars(_replace_group(grp.require_group('times'), 'qc'),
                      session.video_times_qc)
    for label, signal in getattr(session, 'movement_signals', {}).items():
        _save_time_series(_replace_group(grp.require_group(label), 'preprocessed'),
                          signal)
    for label, responses in getattr(session, 'movement_responses', {}).items():
        _save_peri_event_matrix(
            _replace_group(grp.require_group(label), 'responses'),
            responses,
        )
    if hasattr(session, 'pose_xcorr'):
        _save_pose_xcorr(_replace_group(grp.require_group('pose'), 'qc'),
                         session.pose_xcorr)
    if session.video_manual_qc:
        _save_manual_qc(grp.require_group('manual_qc'), session.video_manual_qc)


def _load_video(session, h5_file):
    if 'video' not in h5_file:
        return
    grp = h5_file['video']
    if 'times/qc' in grp:
        session.video_times_qc = _load_scalars(grp['times/qc'])
    for attribute, name in (('pose_times', 'times'), ('pose', 'pose'),
                            ('motion_energy', 'motion_energy')):
        if name in grp:
            setattr(session, attribute, _load_frame_data(grp[name]))
    _set_if_read(session, 'movement_signals',
                 _read_label_products(grp, 'preprocessed', _load_time_series))
    _set_if_read(session, 'movement_responses', _read_label_responses(grp))
    if 'pose/qc' in grp:
        session.pose_xcorr = _load_pose_xcorr(grp['pose/qc'])
    if 'manual_qc' in grp:
        session.video_manual_qc = _load_manual_qc(grp['manual_qc'])


# The QC check run over a session's video, and the error types that disqualify
# it in the rollup: that check, plus a camera clock that was never there. The
# three leftCamera problem labels disqualify too, but they are read live from
# Alyx rather than logged, so they are scored in `_score_video_qc` directly.
VIDEO_QC_ERRORS = (VideoLengthError,)
VIDEO_QC_DISQUALIFYING_ERRORS = frozenset(
    e.__name__ for e in (MissingVideoTimestamps, *VIDEO_QC_ERRORS))

# Trace-derived scalar columns guaranteed in the pose rollup, NaN when their
# source is absent: the LP keypoint measures plus the motion-energy channel.
POSE_TRACE_COLUMNS = [*POSE_MEASURES, 'motion_energy']


def _has_lp_channel(movement_responses: dict) -> bool:
    """True when an LP keypoint channel was extracted (not just motion energy)."""
    return any(label != 'motion_energy' for label in movement_responses)


def _score_video_qc(video_qc: dict, error_types: set[str]) -> float:
    """Video QC score in [0, 1], or ``-1`` when the session is disqualified.

    The five ``VIDEO_QC_QUALITY_COLS`` labels in ``video_qc`` — the extended-QC
    outcomes fetched live from Alyx, never stored in the H5 — are mapped through
    ``config.QCVAL2NUM`` and averaged with ``nanmean``.

    Two things disqualify a session outright. Any error type in
    ``VIDEO_QC_DISQUALIFYING_ERRORS`` logged against it, and any of the three
    ``VIDEO_QC_PROBLEM_COLS`` labels present in ``video_qc`` reading anything
    other than ``PASS`` — a broken camera clock, dropped frames or a bad pin
    state make the video unusable however clean it looks. ``NOT_SET`` is not
    ``PASS`` and disqualifies with the rest, since an unrun problem check is no
    evidence that the problem is absent.

    Among the quality labels ``NOT_SET`` is instead dropped: the check produced
    no outcome, so it carries no evidence either way. Its ``QCVAL2NUM`` value
    exists to place it on the QC colormap, not to weigh in an average. A session
    with no scorable label left scores NaN.
    """
    if error_types & VIDEO_QC_DISQUALIFYING_ERRORS:
        return -1.0
    if any(video_qc[col] != LP_QC_PASS
           for col in VIDEO_QC_PROBLEM_COLS if col in video_qc):
        return -1.0
    quality = [QCVAL2NUM.get(video_qc[col], np.nan)
               for col in VIDEO_QC_QUALITY_COLS
               if col in video_qc and video_qc[col] != LP_QC_NOT_SET]
    return float(np.nanmean(quality)) if quality else np.nan


def _add_trace_deltas(row: dict, movement_responses: dict) -> None:
    """Add one post-minus-pre movement delta per channel; no-op when absent.

    Both terms are read-time slices of the channel's response grid: the mean
    over ``MOVEMENT_RESPONSE_WINDOW`` of its own event cell (``LABEL2EVENT``)
    minus the mean over ``BASELINE_WINDOW`` of its onset-locked cell
    (``STIM_ONSET_EVENT``). The ``motion_energy`` channel flows through this
    loop like any other label.
    """
    for label, responses in movement_responses.items():
        row[label] = movement_delta(
            responses.sel(event=LABEL2EVENT[label]).values,
            responses.sel(event=STIM_ONSET_EVENT).values,
            responses.coords['time'].values,
            MOVEMENT_RESPONSE_WINDOW, BASELINE_WINDOW,
        )


def _add_xcorr_scalars(row: dict, xcorr) -> None:
    """Add drift and per-third peak lag/value; all NaN when no cross-correlation."""
    if xcorr is None:
        for col in ('drift', 'peak_lag_early', 'peak_lag_mid', 'peak_lag_late',
                    'peak_val_early', 'peak_val_mid', 'peak_val_late'):
            row[col] = np.nan
        return
    row['drift'] = xcorr['drift']
    row['peak_lag_early'], row['peak_lag_mid'], row['peak_lag_late'] = \
        xcorr['peak_lags']
    # Peak value of each third's cross-correlation function (alignment strength)
    row['peak_val_early'], row['peak_val_mid'], row['peak_val_late'] = \
        np.nanmax(xcorr['functions'], axis=1)


def _read_mean_rt(h5_file) -> float:
    """Mean reaction time from the H5 ``trials/table`` group, NaN when unavailable.

    Reaction time is ``feedback_times - STIM_ONSET_EVENT`` per trial, averaged
    with ``nanmean``. Returns NaN if the ``trials/table`` group or either
    dataset is absent.
    """
    if 'trials/table' not in h5_file:
        return np.nan
    trials = h5_file['trials/table']
    if STIM_ONSET_EVENT not in trials or 'feedback_times' not in trials:
        return np.nan
    return np.nanmean(trials['feedback_times'][:] - trials[STIM_ONSET_EVENT][:])


def _pose_row(video: h5py.Group, video_qc: dict, error_types: set[str]) -> dict:
    """Roll one session's ``video`` group up into a flat row of scalars.

    Parameters
    ----------
    video : h5py.Group
        The session's ``video`` group. A session whose LP pose is absent still
        has one, with its camera-clock measures and whatever channels were
        extracted.
    video_qc : dict
        That session's eight ``VIDEO_QC_COLS`` labels, fetched from Alyx. Empty
        for a session absent from the fetch, which then scores NaN.
    error_types : set of str
        Error types logged for the session, read for the disqualifying ones.

    Returns
    -------
    dict
        ``lp_exists``, the movement deltas, the cross-correlation scalars, the
        manual QC verdicts, the camera-clock measures and ``video_qc_score``.
    """
    movement_responses = _read_label_responses(video)
    xcorr = _load_pose_xcorr(video['pose/qc']) if 'pose/qc' in video else None
    times_qc = dict(video['times/qc'].attrs) if 'times/qc' in video else {}
    manual_qc = (_load_manual_qc(video['manual_qc'])
                 if 'manual_qc' in video else {})
    row = {
        'lp_exists': _has_lp_channel(movement_responses),
        'length_discrepancy': times_qc.get('length_discrepancy', np.nan),
        'framerate_from_tpts': times_qc.get('framerate_from_tpts', np.nan),
        'video_qc_score': _score_video_qc(video_qc, error_types),
    }
    row.update({col: video_qc[col] for col in VIDEO_QC_COLS if col in video_qc})
    _add_trace_deltas(row, movement_responses)
    _add_xcorr_scalars(row, xcorr)
    row.update({label: manual_qc.get(label, LP_QC_NOT_SET)
                for label in LP_QC_LABELS})
    return row


# The video modality's three raw products, each its own ONE fetch:
# product -> (session attribute, ONE dataset, exception when Alyx lacks it).
# They are listed separately rather than folded into one 'video/raw' because
# each fails on its own, and a session missing only LP still has usable times
# and motion energy.
_RAW_VIDEO_DATASETS = {
    'video/times': ('pose_times', '_ibl_leftCamera.times.npy',
                    MissingVideoTimestamps),
    'video/pose': ('pose', '_ibl_leftCamera.lightningPose.pqt', MissingLP),
    'video/motion_energy': ('motion_energy', 'leftCamera.ROIMotionEnergy.npy',
                            MissingMotionEnergy),
}

# What `load_responses(modality)` needs per modality: the method returning that
# modality's preprocessed signals as a label -> Series mapping, the session
# attribute its response matrices are assigned to, and the extraction arguments
# to fall back on when the caller names none. Photometry's fallback is empty
# because `extract_responses` already defaults to the photometry window; the
# wheel's and the video's cuts are their own, read straight off `config.py`.
_RESPONSE_MODALITIES = {
    'photometry': ('load_photometry', 'photometry_responses', {}),
    'wheel':      ('_wheel_signals', 'wheel_responses', {
        'events': WHEEL_RESPONSE_EVENTS,
        'window': WHEEL_RESPONSE_WINDOW,
    }),
    'video':      ('_movement_signals', 'movement_responses',
                   {'events': MOVEMENT_EVENTS,
                    'window': MOVEMENT_RESPONSE_WINDOW}),
}

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

# The session attributes each save handler writes from, in the order the groups
# are written. `save_h5` consults these to decide which handlers to run when the
# caller names no groups: a session carries a data attribute only once that
# product exists, so the attribute being there is the whole test.
_SAVE_GROUP_PRODUCTS = {
    'photometry': ('photometry_responses', 'photometry_qc',
                   'neurophotometrics_qc'),
    'trials':     ('trials', 'performance'),
    'wheel':      ('wheel_position', 'wheel_velocity', 'wheel_responses'),
    'video':      ('pose_times', 'pose', 'motion_energy', 'movement_signals',
                   'movement_responses', 'pose_xcorr', 'video_times_qc'),
}


def _align_posteriors_to_trials(
    block: pd.DataFrame, trials: pd.DataFrame, atol: float = 1e-3
) -> np.ndarray:
    """Match each DDM-HMM posterior row to its canonical trial by ordered RT.

    The posteriors CSV is a chronological subsequence of the session's trials
    (the fit dropped some trials by a preprocessing rule not recoverable from
    the trial columns). Both are walked in order — ``block`` by
    ``trial_in_dataset``, ``trials`` by ``stimOn_times`` — matching each block
    ``rt`` to the next trial whose ``response_times - stimOn_times`` equals it.
    Order preservation makes RT collisions harmless.

    ``stimOn_times`` here is not the pipeline's onset clock
    (``config.STIM_ONSET_EVENT``, the Bpod trigger) but the column the
    collaborator's fit measured its RTs from. It is a matching key against a
    foreign file, so it tracks that file's definition; the two clocks differ by
    ~60 ms, far more than ``atol``, and nothing would match if this drifted.

    Parameters
    ----------
    block : pandas.DataFrame
        Posterior rows for one eid, sorted by ``trial_in_dataset``; needs an
        ``rt`` column (seconds).
    trials : pandas.DataFrame
        Session trials sorted by ``stimOn_times``; needs ``response_times`` and
        ``stimOn_times`` (seconds).
    atol : float
        RT match tolerance in seconds.

    Returns
    -------
    numpy.ndarray
        ``trials`` index labels, one per ``block`` row, in block order.

    Raises
    ------
    ValueError
        If a block row has no ordered RT match (the CSV is not a subsequence of
        these trials — fail loud).
    """
    block_rt = block['rt'].to_numpy()
    trial_rt = (trials['response_times'] - trials['stimOn_times']).to_numpy()
    trial_labels = trials.index.to_numpy()
    matched = np.empty(len(block_rt), dtype=trial_labels.dtype)
    j = 0
    for i, rt in enumerate(block_rt):
        while j < len(trial_rt) and abs(trial_rt[j] - rt) > atol:
            j += 1
        if j >= len(trial_rt):
            raise ValueError(
                f"posteriors row {i} (rt={rt}) has no ordered RT match in trials")
        matched[i] = trial_labels[j]
        j += 1
    return matched


# Data fields the loader parent declares as dataclass fields, and so sets to an
# empty DataFrame or dict on every construction. `PhotometrySession` guards its
# loads on the attribute being there, so those empty containers are deleted
# rather than mistaken for a product that was built and came back empty.
_PARENT_DATA_FIELDS = ('trials', 'wheel', 'pose', 'motion_energy', 'pupil')


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

        Keyword arguments are forwarded to the loader parent. `one` is optional:
        without it the session still constructs, and only the load methods that
        fetch from Alyx will fail.
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
        # No data attribute is pre-set: a product exists on the session only
        # once it has been loaded or computed, which is what every load method
        # guards on and what makes a missing input raise rather than pass a
        # sentinel downstream. The loader parent is a dataclass and sets four
        # of them to empty containers, so those are dropped here.
        for attribute in _PARENT_DATA_FIELDS:
            delattr(self, attribute)
        # Two dicts survive as namespaces rather than as products. Both raw and
        # preprocessed photometry live in `self.photometry` keyed by band, so
        # the band key is the presence check; `self.ols_fits` accumulates fits
        # keyed by (model, event) and is never stored.
        self.photometry = {}
        self.ols_fits = {}
        # Manual QC verdicts, set by hand in the viewers and computed from
        # nothing: LP_QC_LABELS -> verdict for the camera, which is per session,
        # and region -> the same mapping for photometry, which is per recording.
        self.photometry_manual_qc = {}
        self.video_manual_qc = {}
        if load_data:
            self.load_trials()
            self.load_photometry()


    def __post_init__(self) -> None:
        """Resolve the session path, unless there is no ONE connection.

        ``SessionLoader.__post_init__`` raises when ``one`` is None, because it
        needs the connection to turn an eid into a session path. Sessions read
        back from H5 never touch Alyx, so with no connection leave
        ``session_path`` and ``data_info`` at their dataclass defaults; any
        load method that does need Alyx then fails on ``self.one`` being None.
        """
        if self.one is not None:
            super().__post_init__()


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
            data = _read_metadata(f)
        if not data:
            raise ValueError(f"H5 file has no /metadata group: {fpath}")

        ps = cls(pd.Series(data), one=one, load_data=False)

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

    def log_error(self, error, product=None):
        """Log an exception to the session's error list.

        Parameters
        ----------
        error : Exception
            The exception to log. Type, message, and traceback are captured.
        product : str, optional
            The '{modality}/{product}' name whose build raised, e.g.
            'video/pose'. Decides which `errors/{product}` group the entry is
            saved under; None writes it to the `errors/` root.
        """
        from iblnm.validation import make_log_entry
        self.errors.append(
            make_log_entry(self.eid, error=error, product=product)
        )

    def stored_product_exists(self, product: str) -> bool:
        """Whether the session's H5 already holds `product`.

        The store tier of the read order every load method follows: session
        attribute, else the stored product, else fetch. A product is held when
        its group carries data of its own: a dataset, an attr that is not a
        stamp left by an earlier version, or a subgroup that is not itself
        another product (`_PRODUCT_SUBGROUPS`) — the raw bands of one region,
        which sit one level down because they carry different sample times.
        A group with none of those only contains the products beneath it, which
        is the state `photometry/{region}/raw` is left in when
        `config.store_raw` is off but its `qc/` was written.

        Parameters
        ----------
        product : str
            A '{modality}/{product}' name, e.g. 'photometry/raw/qc'. Names omit
            the label level, so this one is searched for at both
            'photometry/raw/qc' and 'photometry/{region}/raw/qc'.

        Returns
        -------
        bool
            False when the file, the modality group, or the product group is
            missing, or when the group carries no data of its own.

        Reads the H5 structure only: no datasets are loaded and no ONE
        connection is needed.
        """
        modality, _, name = product.partition('/')
        if not self.filepath.exists():
            return False
        with h5py.File(self.filepath, 'r') as h5:
            modality_grp = h5.get(modality)
            if modality_grp is None:
                return False
            for path in [name] + [f'{label}/{name}' for label in modality_grp]:
                grp = modality_grp.get(path)
                if grp is None:
                    continue
                if (set(grp.attrs) - _STAMP_ATTRS
                        or any(isinstance(grp[key], h5py.Dataset)
                               or key not in _PRODUCT_SUBGROUPS for key in grp)):
                    return True
        return False

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

    def load_trials(self) -> pd.DataFrame:
        """Return the trials table, fetching it from Alyx if it is not stored.

        Reads in the order every load method follows — the session attribute,
        else the stored `trials/table`, else Alyx — and writes what it fetched,
        so a session with an empty H5 fills itself.

        Returns
        -------
        pandas.DataFrame
            The trials table, derived columns included. Also assigned to
            ``self.trials``.

        """
        if hasattr(self, 'trials'):
            return self.trials
        if self.stored_product_exists('trials/table'):
            with h5py.File(self.filepath, 'r') as h5:
                self.trials = _read_dataframe(h5['trials/table'])
            return self.trials
        self.fetch_trials()
        self.save_h5(groups=['trials'])
        return self.trials

    def fetch_trials(self) -> pd.DataFrame:
        """Fetch the trials table from Alyx and add the derived columns.

        Returns
        -------
        pandas.DataFrame
            ONE's table plus `trial` (its index, persisted as a column so the
            H5 round-trip cannot lose it), `stim_side`, `signed_contrast` and
            `contrast`. Also assigned to ``self.trials``.

        Raises
        ------
        MissingExtractedData
            The trials table is absent but the raw task data is there.
        MissingRawData
            Neither is there.
        """
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
        # The ONE index is trial identity; persist it as a column, since the H5
        # round-trip rebuilds a fresh RangeIndex and would lose it otherwise.
        self.trials['trial'] = self.trials.index.to_numpy()
        contrasts = compute_trial_contrasts(self.trials)
        self.trials['stim_side'] = contrasts['stim_side']
        self.trials['signed_contrast'] = contrasts['signed_contrast']
        self.trials['contrast'] = contrasts['contrast']
        return self.trials

    def load_states(self) -> None:
        """Attach per-trial DDM-HMM state posteriors to ``self.states``.

        Locates this mouse's ``{subject}_K*_posteriors.csv`` in
        ``config.DDM_HMM_DIR`` and aligns its rows for this eid to the canonical
        H5 trials by ordered RT-subsequence matching (see
        :func:`_align_posteriors_to_trials`). The match is verified by requiring
        ``|signed_contrast|`` to agree on every matched trial (CSV as a fraction,
        H5 as a percent), which catches any RT-collision misalignment.

        Sets ``self.states`` to a DataFrame indexed like ``self.trials`` with
        columns ``map_state, state_1 … state_K`` filled on the trials that are in
        the fit and NaN on those the fit dropped. Leaves it ``None`` when the
        mouse was not modeled or this session is absent from the fit.

        Raises
        ------
        ValueError
            If ``self.trials`` is not loaded, a CSV row has no ordered RT match,
            or a matched trial's ``|contrast|`` disagrees with the CSV (fail
            loud — the alignment is wrong).
        """
        matches = sorted(DDM_HMM_DIR.glob(f'{self.subject}_K*_posteriors.csv'))
        if not matches:
            self.states = None
            return
        posteriors = pd.read_csv(matches[0])
        block = posteriors[posteriors['eid'] == self.eid].sort_values(
            'trial_in_dataset')
        if block.empty:
            self.states = None
            return

        if not hasattr(self, 'trials'):
            raise ValueError(
                f"load_states requires loaded trials (eid {self.eid})")

        trials = self.trials.sort_values('stimOn_times')
        matched = _align_posteriors_to_trials(block, trials)

        csv_abs_contrast = block['signed_contrast'].abs().to_numpy() * 100
        h5_abs_contrast = self.trials.loc[matched, 'signed_contrast'].abs().to_numpy()
        if not np.allclose(csv_abs_contrast, h5_abs_contrast, atol=1e-2):
            raise ValueError(
                f"|contrast| mismatch between posteriors and trials for eid "
                f"{self.eid} (RT-collision misalignment)")

        state_cols = ['map_state'] + [c for c in block.columns
                                      if c.startswith('state_')]
        states = pd.DataFrame(np.nan, index=self.trials.index,
                              columns=state_cols)
        states.loc[matched, state_cols] = block[state_cols].to_numpy()
        self.states = states

    def load_photometry(self) -> pd.DataFrame:
        """Return the preprocessed photometry signal, building it if absent.

        Reads `photometry/{region}/preprocessed` when it is stored; otherwise
        fetches the raw bands from Alyx, preprocesses them, and writes the
        product on the way out.

        Returns
        -------
        pandas.DataFrame
            Regions as columns, time (seconds) as the index. Also assigned to
            ``self.photometry[PREPROCESSED_BAND]``.

        """
        if PREPROCESSED_BAND in self.photometry:
            return self.photometry[PREPROCESSED_BAND]
        signal = None
        if self.stored_product_exists('photometry/preprocessed'):
            with h5py.File(self.filepath, 'r') as h5:
                signal, self.preprocessing_diagnostics = (
                    _read_photometry_preprocessed(h5['photometry']))
        if signal is None:
            self.load_raw_photometry()
            signal = self.extract_preprocessed_photometry()
            self.save_h5(groups=['photometry'])
        self.photometry[PREPROCESSED_BAND] = signal
        return signal

    def load_raw_photometry(
        self,
        pre: int = -5,
        post: int = 5,
        ):
        """Return the raw signal and reference bands, fetching them if absent.

        Populates ``self.photometry`` with one DataFrame per band
        (``'GCaMP'``, ``'Isosbestic'``), brain regions as columns. Separate
        from :meth:`load_photometry`, which returns the preprocessed signal:
        one method returning either would let an analysis run on raw data
        without saying so.

        Reads `photometry/{region}/raw` when it is stored — which only happens
        with `config.store_raw` on, since nothing writes that group otherwise —
        and goes to Alyx in every other case. Only the fetch clears the manual
        QC verdicts: reading the stored bands back replaces no samples.
        """
        if all(band in self.photometry for band in PHOTOMETRY_BANDS):
            return
        if self.stored_product_exists('photometry/raw'):
            with h5py.File(self.filepath, 'r') as h5:
                self.photometry.update(_read_photometry_raw(h5['photometry']))
            return
        self.fetch_photometry(pre=pre, post=post)

    def fetch_photometry(self, pre: int = -5, post: int = 5) -> None:
        """Fetch the raw signal and reference bands from Alyx.

        Populates ``self.photometry`` with one DataFrame per band (``'GCaMP'``,
        ``'Isosbestic'``), brain regions as columns, renamed to the session's
        `brain_region` metadata. Clears the photometry manual QC verdicts: they
        were passed on the samples this fetch has just replaced.

        Parameters
        ----------
        pre, post : int
            Seconds of signal to keep either side of the session, passed to the
            loader parent.

        Raises
        ------
        MissingExtractedData
            The extracted signal is absent but the neurophotometrics source
            table is there.
        MissingRawData
            Neither is there.
        """
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
        self._clear_manual_qc('photometry')

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

    def complete_events(self) -> list[str]:
        """The `config.RESPONSE_EVENTS` whose times are complete enough to cut on.

        :meth:`validate_event_completeness` is non-fatal for a response cut: the
        events it names are dropped and the rest are cut.

        Returns
        -------
        list of str
            The events to cut on, in `RESPONSE_EVENTS` order. Empty when no
            event survives, in which case nothing is cut at all.
        """
        try:
            self.validate_event_completeness()
        except IncompleteEventTimes as error:
            return [event for event in RESPONSE_EVENTS
                    if event not in error.missing_events]
        return list(RESPONSE_EVENTS)

    def validate_trials_in_photometry_time(self, band=None):
        """Raises TrialsNotInPhotometryTime if trial times fall outside photometry window."""
        if band is None:
            band = 'GCaMP_preprocessed' if 'GCaMP_preprocessed' in self.photometry else 'GCaMP'
        phot_times = self.photometry[band].index
        trial_start = self.trials[STIM_ONSET_EVENT].min()
        trial_stop = self.trials['feedback_times'].max()
        if not (trial_start >= phot_times.min() and trial_stop <= phot_times.max()):
            raise TrialsNotInPhotometryTime(
                f"Trials [{trial_start:.1f}, {trial_stop:.1f}] outside "
                f"photometry [{phot_times.min():.1f}, {phot_times.max():.1f}]"
            )

    def validate_qc(self):
        """Raise QCValidationError on a non-zero neurophotometrics QC metric.

        Reads the `photometry/neurophotometrics/qc` attrs held in
        `self.neurophotometrics_qc`. Both metrics are fatal: a band inversion
        means the channels are not the bands they are labelled, and early
        samples mean the recording started before the LEDs settled. A session
        that has not been scored carries no such attribute, and so has nothing
        to fail on.
        """
        qc = getattr(self, 'neurophotometrics_qc', {})
        issues = [message for metric, message in (
            ('n_band_inversions', 'band inversions detected'),
            ('n_early_samples', 'early samples detected'),
        ) if qc.get(metric, 0) > 0]
        if issues:
            raise QCValidationError('; '.join(issues))

    def fetch_neurophotometrics(self) -> pd.DataFrame:
        """Fetch the neurophotometrics source table from Alyx.

        Returns
        -------
        pandas.DataFrame
            The pre-extraction table, times (seconds) as the index, also
            assigned to ``self.neurophotometrics``. It carries the `color`
            column that `n_band_inversions` scores and that extraction drops,
            which is why the QC reads this rather than the extracted bands.
        """
        raw_photometry = self.one.load_dataset(
            self.eid,
            'raw_photometry_data/_neurophotometrics_fpData.raw.pqt'
            )
        self.neurophotometrics = from_neurophotometrics_df_to_photometry_df(
            raw_photometry).set_index('times')
        return self.neurophotometrics

    def load_responses(
        self,
        modality: str,
        events: Sequence[str] | None = None,
        window: Sequence[float | str] | None = None,
    ) -> dict[str, xr.DataArray]:
        """Return one modality's peri-event matrices, cutting them if absent.

        Reads `{modality}/{label}/responses` when it is stored; otherwise loads
        that modality's preprocessed signals plus the trials table, cuts the
        matrices with :meth:`extract_responses`, and writes them.

        Parameters
        ----------
        modality : str
            Key of `_RESPONSE_MODALITIES`, e.g. 'photometry'. Names both the
            product (`{modality}/responses`) and the session attribute the
            result is assigned to.
        events, window : optional
            Passed through to :meth:`extract_responses`; see there. Ignored
            when the stored product is read rather than cut.

        Returns
        -------
        dict[str, xarray.DataArray]
            One DataArray per label, dims (event, trial, time).

        """
        load_signals, attribute, defaults = _RESPONSE_MODALITIES[modality]
        if hasattr(self, attribute):
            return getattr(self, attribute)
        responses = None
        if self.stored_product_exists(f'{modality}/responses'):
            with h5py.File(self.filepath, 'r') as h5:
                responses = _read_label_responses(h5[modality])
        was_cut = responses is None
        if was_cut:
            signals = getattr(self, load_signals)()
            self.load_trials()
            named = {key: value for key, value in
                     (('events', events), ('window', window))
                     if value is not None}
            responses = self.extract_responses(signals, **(defaults | named))
        setattr(self, attribute, responses)
        if was_cut:
            self.save_h5(groups=[modality])
        return responses

    def extract_responses(
        self,
        signals: Mapping[str, pd.Series],
        events: Sequence[str] | None = None,
        window: Sequence[float | str] | None = None,
    ) -> dict[str, xr.DataArray]:
        """Cut peri-event response matrices out of arbitrary time series.

        Signal-source agnostic: photometry passes ``self.photometry[band]``
        (a DataFrame, whose ``.items()`` yields ``(region, Series)``), behavior
        passes a ``label -> Series`` dict. Every label gets the same full
        ``events`` axis. The result is returned, not stored — the caller
        assigns it to the attribute for its modality.

        Parameters
        ----------
        signals : Mapping[str, pd.Series]
            Label to time-indexed signal (index in seconds, same clock as
            ``self.trials``).
        events : sequence of str, optional
            ``self.trials`` columns holding event times. Defaults to
            ``RESPONSE_EVENTS``.
        window : sequence, optional
            ``(t0, t1)``. ``t0`` is seconds relative to each event. ``t1`` is
            either seconds relative to each event, or the name of a
            ``self.trials`` column holding each trial's own window end — the
            wheel's cut runs stimOn to that trial's feedback. Defaults to
            ``self.RESPONSE_WINDOW``.

        Returns
        -------
        dict[str, xr.DataArray]
            One DataArray per label, dims (event, trial, time). With a
            per-trial window end every trial still shares one time axis,
            spanning to the longest trial, and is NaN-padded beyond its own
            endpoint.
        """
        if events is None:
            events = RESPONSE_EVENTS
        if window is None:
            window = self.RESPONSE_WINDOW

        t0, t1 = window
        if isinstance(t1, str):
            t1 = self.trials[t1].to_numpy()

        responses = {}
        for label, signal in signals.items():
            per_event = []
            for event in events:
                resp, sample_times = get_responses(
                    signal, self.trials[event].values, t0=t0, t1=t1,
                )
                per_event.append(resp)
            responses[label] = xr.DataArray(
                np.stack(per_event),
                dims=['event', 'trial', 'time'],
                coords={
                    'event': list(events),
                    'trial': self.trials['trial'].to_numpy(),
                    'time': sample_times,
                },
            )
        return responses

    def save_h5(self, fpath=None, groups=None, mode='a'):
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
            groups = self._available_save_groups()

        with h5py.File(fpath, mode) as h5_file:
            for group_name in groups:
                _SAVE_HANDLERS[group_name](self, h5_file)

    def _available_save_groups(self) -> list[str]:
        """Name the top-level H5 groups this session holds something to write.

        A group is available when any of its products is on the session, which
        is now a question of the attribute existing: a product that was built
        and came back empty is still a product and is still written.
        `photometry` is the exception — it is a band namespace rather than a
        product, so the preprocessed band key is what counts there.
        """
        available = {group: any(hasattr(self, attr) for attr in attributes)
                     for group, attributes in _SAVE_GROUP_PRODUCTS.items()}
        available['photometry'] |= (PREPROCESSED_BAND in self.photometry
                                    or bool(self.photometry_manual_qc))
        available['video'] |= bool(self.video_manual_qc)
        return [group for group, is_available in available.items()
                if is_available]

    def load_h5(self, fpath=None, groups=None):
        """Load session data from HDF5 file.

        Parameters
        ----------
        fpath : Path or str, optional
            Path to the HDF5 file. Defaults to ``self.filepath``; naming another
            path adopts it as ``self.filepath``, so the load methods and
            :meth:`stored_product_exists` go on reading the file this data came
            from rather than the default one for this eid.
        groups : sequence of str, optional
            Which data groups to load. Any subset of:
            'metadata', 'errors', 'photometry', 'trials', 'wheel', 'video'.
            None loads all groups present in the file.
        """
        if fpath is None:
            fpath = self.filepath
        else:
            self.filepath = Path(fpath)
        group_names = list(_LOAD_HANDLERS) if groups is None else list(groups)
        with h5py.File(fpath, 'r') as h5_file:
            for group_name in group_names:
                _LOAD_HANDLERS[group_name](self, h5_file)

    def load_neurophotometrics_qc(self) -> dict[str, float]:
        """Return the neurophotometrics QC metrics, scoring them if absent.

        Reads `photometry/neurophotometrics/qc` when it is stored; otherwise
        fetches the source table from Alyx, scores it with
        :meth:`run_neurophotometrics_qc`, and writes the product.

        Returns
        -------
        dict
            Metric name -> value, also assigned to `self.neurophotometrics_qc`.

        """
        if hasattr(self, 'neurophotometrics_qc'):
            return self.neurophotometrics_qc
        if self.stored_product_exists('photometry/neurophotometrics/qc'):
            with h5py.File(self.filepath, 'r') as h5:
                self.neurophotometrics_qc = _load_scalars(
                    h5['photometry/neurophotometrics/qc'])
            return self.neurophotometrics_qc
        self.fetch_neurophotometrics()
        self.run_neurophotometrics_qc()
        self.save_h5(groups=['photometry'])
        return self.neurophotometrics_qc

    def load_photometry_qc(self) -> dict[str, dict[str, float]]:
        """Return each region's raw-band QC metrics, scoring them if absent.

        Reads `photometry/{region}/raw/qc` when it is stored; otherwise fetches
        the raw bands from Alyx, scores them with :meth:`run_photometry_qc`,
        and writes the product.

        Returns
        -------
        dict
            Region -> {band-suffixed metric name: value}, also assigned to
            `self.photometry_qc`.

        """
        if hasattr(self, 'photometry_qc'):
            return self.photometry_qc
        if self.stored_product_exists('photometry/raw/qc'):
            with h5py.File(self.filepath, 'r') as h5:
                self.photometry_qc = _read_photometry_qc(h5['photometry'])
            return self.photometry_qc
        self.load_raw_photometry()
        self.run_photometry_qc()
        self.save_h5(groups=['photometry'])
        return self.photometry_qc

    def run_neurophotometrics_qc(
        self, raw_metrics: Sequence[str] | None = None
    ) -> dict[str, float]:
        """Score the neurophotometrics source table already fetched.

        Reads `self.neurophotometrics`, assigned by
        :meth:`fetch_neurophotometrics`; it never fetches and never writes.
        `load_neurophotometrics_qc` is what does both.

        Parameters
        ----------
        raw_metrics : sequence of str, optional
            Names of `iblphotometry.metrics` functions taking the whole source
            table. Defaults to `config.QC_RAW_METRICS`.

        Returns
        -------
        dict
            Metric name -> value, also assigned to `self.neurophotometrics_qc`.

        The source table itself is never stored: `n_band_inversions` reads a
        `color` column that only exists before extraction, so the QC is all
        that survives the fetch.
        """
        if raw_metrics is None:
            raw_metrics = QC_RAW_METRICS
        self.neurophotometrics_qc = {
            name: float(getattr(metrics, name)(self.neurophotometrics))
            for name in raw_metrics
        }
        return self.neurophotometrics_qc

    def run_photometry_qc(
        self,
        sliding_metrics: Sequence[str] | None = None,
        metrics_kwargs: dict | None = None,
        sliding_kwargs: dict | None = None,
        agg: dict[str, str] | None = None,
    ) -> dict[str, dict[str, float]]:
        """Score the raw bands in sliding windows, per region.

        Reads `self.photometry`, populated by :meth:`load_raw_photometry`; it
        never fetches and never writes. `load_photometry_qc` is what does both.

        Issues two `qc_signals` calls over the same windows: the metrics named
        in `config.QC_UNDETRENDED_METRICS` with `detrend=False`, the rest with
        `detrend=True`. The split is load-bearing — detrending makes every
        sample of a window a distinct float, which pins `n_unique_samples` at
        1.0 no matter how dead the signal is.

        Parameters
        ----------
        sliding_metrics : sequence of str, optional
            Names of `iblphotometry.metrics` functions. Defaults to
            `config.QC_SLIDING_METRICS`.
        metrics_kwargs : dict, optional
            Per-metric keyword arguments, keyed by metric name. Defaults to
            `config.QC_METRICS_KWARGS`.
        sliding_kwargs : dict, optional
            `w_len` and `step_len` in seconds. Defaults to
            `config.QC_SLIDING_KWARGS`; its `detrend` entry is overridden per
            call by the split above.
        agg : dict, optional
            Metric name -> aggregator name in `_QC_AGGREGATORS`, reducing the
            windows to one value. Defaults to `config.QC_SLIDING_AGG`.

        Returns
        -------
        dict
            Region -> {band-suffixed metric name: value}, also assigned to
            `self.photometry_qc`.
        """
        if sliding_metrics is None:
            sliding_metrics = QC_SLIDING_METRICS
        if metrics_kwargs is None:
            metrics_kwargs = QC_METRICS_KWARGS
        if sliding_kwargs is None:
            sliding_kwargs = QC_SLIDING_KWARGS
        if agg is None:
            agg = QC_SLIDING_AGG

        def score(metric_names, detrend):
            return qc_signals(
                self.photometry,
                metrics=[getattr(metrics, name) for name in metric_names],
                metrics_kwargs=metrics_kwargs,
                sliding_kwargs={**sliding_kwargs, 'detrend': detrend},
            )

        qc_tidy = pd.concat([
            score([m for m in sliding_metrics if m in QC_UNDETRENDED_METRICS],
                  detrend=False),
            score([m for m in sliding_metrics if m not in QC_UNDETRENDED_METRICS],
                  detrend=True),
        ])
        self.photometry_qc = _aggregate_qc_windows(qc_tidy, agg)
        return self.photometry_qc


    # =========================================================================
    # Preprocessing Methods
    # =========================================================================

    def extract_preprocessed_photometry(
        self,
        pipeline=None,
        signal_band='GCaMP',
        reference_band='Isosbestic',
        targets=None,
        output_band=PREPROCESSED_BAND,
        regression_method: str = 'mse',
    ):
        """Run the preprocessing pipeline over the raw bands already loaded.

        Reads `self.photometry[signal_band]` and `[reference_band]`, computes,
        and assigns; it never fetches and never writes. `load_photometry` is
        what saves the result.

        Pipeline steps (bleach correct → isosbestic correct → resample to
        TARGET_FS → zscore) are defined in config.PREPROCESSING_PIPELINES. The
        z-score is last so that the stored signal is the one it was applied to:
        interpolating an already-z-scored signal leaves it short of unit
        variance.

        `bleaching_tau` and `iso_correlation` are computed on the partially
        processed signals and land in `self.preprocessing_diagnostics`: they
        describe this preprocessing run rather than the raw signal, which is
        why `load_photometry` writes them as attrs of the preprocessed group.

        A caller passing a non-default `output_band` is opting out of the
        product: the result is kept in `self.photometry` under that name, and
        `load_photometry` saves only what the default band produced, since only
        `PREPROCESSED_BAND` is what the product names.
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
        diagnostics = {}

        for brain_region in targets:
            signal = self.photometry[signal_band][brain_region]
            region_diagnostics = {'bleaching_tau': compute_bleaching_tau(signal)}

            if needs_reference:
                reference = self.photometry[reference_band][brain_region]
                res = run_pipeline(pipeline, signal=signal, reference=reference, full_output=True)
                result = res['result']
                # iso_correlation computed on bleach-corrected signals before isosbestic step
                signal_bc = res.get('signal_bleach_corrected', signal)
                reference_bc = res.get('reference_bleach_corrected', reference)
                region_diagnostics['iso_correlation'] = compute_iso_correlation(
                    signal_bc, reference_bc, regression_method=regression_method
                )
            else:
                result = run_pipeline(pipeline, signal=signal)

            preprocessed[brain_region] = result
            diagnostics[brain_region] = region_diagnostics

        self.photometry[output_band] = pd.DataFrame(preprocessed)
        self.preprocessing_diagnostics = diagnostics
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
        if not hasattr(self, 'trials'):
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

    def extract_trial_timings(self) -> pd.DataFrame:
        """Add the three event-timing durations to `self.trials`, in seconds.

        `reaction_time` is `firstMovement_times - config.STIM_ONSET_EVENT`,
        `movement_time` is `response_times - firstMovement_times`, and
        `response_time` is `response_times - config.STIM_ONSET_EVENT`. Both
        durations end at the choice rather than at feedback delivery, which lags
        it by an outcome-dependent amount (0.1 ms on correct trials, up to
        1.75 s on errors). A missing event-time column yields an all-NaN
        duration rather than an error, since a session can be extracted without
        one. Writes nothing.

        Returns
        -------
        pandas.DataFrame
            `self.trials` with the three columns assigned in place; the row
            count and every existing column are untouched.
        """
        self.trials['reaction_time'] = analysis._event_diff(
            self.trials, 'firstMovement_times', STIM_ONSET_EVENT)
        self.trials['movement_time'] = analysis._event_diff(
            self.trials, 'response_times', 'firstMovement_times')
        self.trials['response_time'] = analysis._event_diff(
            self.trials, 'response_times', STIM_ONSET_EVENT)
        return self.trials

    def extract_performance(self) -> dict:
        """Score `self.trials` into the `trials/performance` payload.

        Computes the metrics every session type carries — trial count, the
        contrasts presented, correct and no-go fractions, and the unbiased
        psychometric fit — then, where the session type has blocks, the
        per-block psychometrics and the bias shift between the 20 and 80
        blocks. Writes nothing; :meth:`load_performance` saves.

        Returns
        -------
        dict
            Metric name -> value. Block metrics are keyed
            `psych_{block}_{param}`, with `block` one of `20`, `50`, `80`.
            Also assigned to ``self.performance``.
        """
        performance = {
            'n_trials': len(self.trials),
            'contrasts': sorted(self.trials['contrast'].unique().tolist()),
            'fraction_correct': task.compute_fraction_correct(self.trials),
            'fraction_correct_easy': task.compute_fraction_correct(
                self.trials[self.trials['contrast'] >= 0.5]
            ),
            'nogo_fraction': task.compute_nogo_fraction(self.trials),
        }
        fit_50 = task.fit_psychometric(self.trials, probability_left=0.5)
        performance.update(
            {f'psych_50_{param}': value for param, value in fit_50.items()}
        )
        if self.session_type in ('biased', 'ephys'):
            fits = task.fit_psychometric_by_block(self.trials)
            performance.update({
                f'psych_{block}_{param}': value
                for block, fit in fits.items() for param, value in fit.items()
            })
            if '20' in fits and '80' in fits:
                performance['bias_shift'] = task.compute_bias_shift(
                    fits['20'], fits['80']
                )
        self.performance = performance
        return self.performance

    def load_performance(self) -> dict:
        """Return the per-session behavioral scalars, scoring them if absent.

        Reads `trials/performance` when it is stored; otherwise scores the
        trials table with :meth:`extract_performance` and writes the result.

        Returns
        -------
        dict
            Metric name -> value, plus `n_trials` and the sorted `contrasts`
            list the session presented. Also assigned to `self.performance`.

        """
        if hasattr(self, 'performance'):
            return self.performance
        if self.stored_product_exists('trials/performance'):
            with h5py.File(self.filepath, 'r') as h5:
                self.performance = _load_performance(h5['trials/performance'])
            return self.performance
        if not hasattr(self, 'trials'):
            self.fetch_trials()
        self.extract_performance()
        self.save_h5(groups=['trials'])
        return self.performance

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

    def load_raw_wheel(self) -> pd.Series:
        """Return the raw encoder position, fetching it if absent.

        Reads `wheel/{WHEEL_LABEL}/raw` when it is stored — which only happens
        with `config.store_raw` on, since nothing writes that group otherwise —
        and goes to Alyx in every other case.

        Returns
        -------
        pandas.Series
            Wheel position (radians) indexed by the encoder's own timestamps
            (seconds, session clock). The encoder samples on movement, not on a
            clock, so the index is irregular — that is what the `wheel/raw`
            product stores, leaving the uniform grid to
            :meth:`extract_wheel_velocity`. Also assigned to
            ``self.wheel_position``.
        """
        if hasattr(self, 'wheel_position'):
            return self.wheel_position
        if self.stored_product_exists('wheel/raw'):
            with h5py.File(self.filepath, 'r') as h5:
                self.wheel_position = _load_time_series(
                    h5[f'wheel/{WHEEL_LABEL}/raw'])
            return self.wheel_position
        return self.fetch_wheel()

    def fetch_wheel(self) -> pd.Series:
        """Fetch the raw encoder position from Alyx, ignoring the store.

        Returns
        -------
        pandas.Series
            Wheel position (radians) on the encoder's own irregular timestamps
            (seconds, session clock), also assigned to ``self.wheel_position``.

        Raises
        ------
        MissingExtractedData
            The wheel ALF object is absent but the raw encoder file is there.
        MissingRawData
            Neither is there, so nothing was recorded.
        """
        try:
            wheel = self.one.load_object(
                self.eid, 'wheel',
                collection=self._find_behaviour_collection('wheel'),
                revision=self.revision or None,
            )
        except ALFObjectNotFound:
            try:
                self.one.load_dataset(self.eid, '_iblrig_encoderPositions.raw.ssv')
            except ALFObjectNotFound:
                raise MissingRawData("_iblrig_encoderPositions.raw.ssv")
            raise MissingExtractedData("_ibl_wheel.position.npy")
        self.wheel_position = pd.Series(np.asarray(wheel['position'], dtype=float),
                                        index=np.asarray(wheel['timestamps'],
                                                         dtype=float))
        return self.wheel_position

    def load_wheel(self) -> pd.Series:
        """Return the preprocessed wheel velocity, building it if absent.

        Reads `wheel/{WHEEL_LABEL}/preprocessed` when it is stored; otherwise
        fetches the raw encoder position from Alyx and differentiates it, which
        writes the product on the way out.

        Returns
        -------
        pandas.Series
            Velocity (radians per second) on a uniform ``WHEEL_FS`` time index
            (seconds). Also assigned to ``self.wheel_velocity``.

        """
        if hasattr(self, 'wheel_velocity'):
            return self.wheel_velocity
        if self.stored_product_exists('wheel/preprocessed'):
            with h5py.File(self.filepath, 'r') as h5:
                self.wheel_velocity = _load_time_series(
                    h5[f'wheel/{WHEEL_LABEL}/preprocessed'])
            return self.wheel_velocity
        self.load_raw_wheel()
        self.extract_wheel_velocity()
        self.save_h5(groups=['wheel'])
        return self.wheel_velocity

    def extract_wheel_velocity(self, fs: float = WHEEL_FS) -> pd.Series:
        """Differentiate `self.wheel_position` into the velocity product.

        Only the velocity is put on a uniform grid — the position stays raw, in
        its own product. Writes nothing; :meth:`load_wheel` saves.

        Parameters
        ----------
        fs : float
            Grid rate in Hz. Defaults to ``config.WHEEL_FS``, the rate the
            stored `wheel/preprocessed` product is written at.

        Returns
        -------
        pandas.Series
            Velocity (radians per second) indexed by grid time (seconds), also
            assigned to ``self.wheel_velocity``.
        """
        self.wheel_velocity = analysis.differentiate(self.wheel_position, fs=fs)
        return self.wheel_velocity

    def load_peak_velocity(self) -> np.ndarray:
        """Return the per-trial peak wheel speed, building it if absent.

        Reads `wheel/{WHEEL_LABEL}/peak_velocity` when it is stored; otherwise
        cuts the wheel responses — fetching the encoder samples if those are
        missing too — reduces them, and writes the product on the way out.

        Returns
        -------
        numpy.ndarray
            Maximum absolute velocity (radians per second) per trial, also
            assigned to ``self.wheel_peak_velocity``.
        """
        if hasattr(self, 'wheel_peak_velocity'):
            return self.wheel_peak_velocity
        if self.stored_product_exists('wheel/peak_velocity'):
            with h5py.File(self.filepath, 'r') as h5:
                self.wheel_peak_velocity = _load_frame_data(
                    h5[f'wheel/{WHEEL_LABEL}/peak_velocity'])
            return self.wheel_peak_velocity
        self.load_responses('wheel')
        self.extract_peak_velocity()
        self.save_h5(groups=['wheel'])
        return self.wheel_peak_velocity

    def extract_peak_velocity(self) -> np.ndarray:
        """Reduce `self.wheel_responses` to one peak speed per trial.

        The response matrix is cut from stimulus onset to the choice, so its
        per-trial maximum is how fast the mouse turned the wheel on that trial.
        Writes nothing; :meth:`load_peak_velocity` saves.

        Returns
        -------
        numpy.ndarray
            Maximum absolute velocity (radians per second) per trial, aligned
            to the `trial` coordinate of the response matrix. A trial whose row
            is entirely NaN — no wheel samples in its window — scores NaN. Also
            assigned to ``self.wheel_peak_velocity``.
        """
        matrix = self.wheel_responses[WHEEL_LABEL].sel(event=_WHEEL_T0_EVENT)
        self.wheel_peak_velocity = analysis.peak_velocity(
            matrix.values, matrix.sizes['trial'])
        return self.wheel_peak_velocity

    def _wheel_signals(self) -> dict[str, pd.Series]:
        """The wheel's preprocessed velocity keyed by its H5 label.

        The wheel has one channel, so this mapping has one entry; it exists so
        `load_responses` can hand every modality's signals to
        :meth:`extract_responses` in the same shape.
        """
        return {WHEEL_LABEL: self.load_wheel()}

    def _load_raw_video(self, product: str) -> np.ndarray | pd.DataFrame:
        """Return one raw video dataset, fetching it when the store has none.

        Reads the stored product when it is present — which only happens with
        `config.store_raw` on, since nothing writes those groups otherwise —
        and goes to Alyx in every other case.

        Parameters
        ----------
        product : str
            One of the `_RAW_VIDEO_DATASETS` keys. Names the H5 group, the
            session attribute the result is assigned to, the ONE dataset to
            fetch, and the exception raised when Alyx does not have it.

        Returns
        -------
        numpy.ndarray or pandas.DataFrame
            The dataset, also assigned to its session attribute.

        """
        attribute, _, _ = _RAW_VIDEO_DATASETS[product]
        if hasattr(self, attribute):
            return getattr(self, attribute)
        if self.stored_product_exists(product):
            with h5py.File(self.filepath, 'r') as h5:
                data = _load_frame_data(h5[product])
            setattr(self, attribute, data)
            return data
        return self._fetch_video(product)

    def _fetch_video(self, product: str) -> np.ndarray | pd.DataFrame:
        """Fetch one raw video dataset from Alyx, ignoring the store.

        Clears the video manual QC verdict on the way out: it was passed on the
        frames this fetch has just replaced.

        Parameters
        ----------
        product : str
            One of the `_RAW_VIDEO_DATASETS` keys; see :meth:`_load_raw_video`.

        Returns
        -------
        numpy.ndarray or pandas.DataFrame
            The dataset, also assigned to its session attribute.

        Raises
        ------
        MissingVideoTimestamps, MissingLP, MissingMotionEnergy
            The product's own exception, when Alyx does not have its dataset.
        """
        attribute, dataset, missing = _RAW_VIDEO_DATASETS[product]
        try:
            data = self.one.load_dataset(self.eid, dataset, collection='alf')
        except ALFObjectNotFound:
            raise missing(dataset)
        if not isinstance(data, pd.DataFrame):
            data = np.asarray(data)
        setattr(self, attribute, data)
        self._clear_manual_qc('video')
        return data

    def fetch_camera_times(self) -> np.ndarray:
        """Fetch the left-camera frame times from Alyx.

        The per-frame times (session clock, seconds) are also assigned to
        ``self.pose_times``. Raises ``MissingVideoTimestamps`` when Alyx has no
        such dataset, so the basic-video pass can block the session before LP is
        attempted.
        """
        return self._fetch_video('video/times')

    def fetch_pose(self) -> pd.DataFrame:
        """Fetch the LightningPose keypoint traces from Alyx.

        The pose DataFrame (columns ``{part}_x``, ``{part}_y``,
        ``{part}_likelihood``) is also assigned to ``self.pose``. Raises
        ``MissingLP`` when the dataset is absent, so batch extraction can log it
        against `video/pose` and carry on. Fetches only the ``lightningPose``
        dataset rather than the whole ``leftCamera`` object, which would also
        pull ``features`` and ``ROIMotionEnergy`` that we never use.
        """
        return self._fetch_video('video/pose')

    def fetch_motion_energy(self) -> np.ndarray:
        """Fetch the per-frame left-camera ROI motion energy from Alyx.

        The per-frame scalar (on the ``leftCamera.times`` base) is also assigned
        to ``self.motion_energy``. Raises ``MissingMotionEnergy`` when the
        dataset is absent, logged non-fatally by the pipeline.
        """
        return self._fetch_video('video/motion_energy')

    def load_camera_times(self) -> np.ndarray:
        """Return the left-camera frame times, from the store or :meth:`fetch_camera_times`.

        The per-frame times (session clock, seconds) are also assigned to
        ``self.pose_times``. Loaded separately from ``load_pose`` because the
        camera timestamps gate the whole session while LP gates only the traces.
        """
        return self._load_raw_video('video/times')

    def load_pose(self) -> pd.DataFrame:
        """Return the LightningPose traces, from the store or :meth:`fetch_pose`.

        The pose DataFrame (columns ``{part}_x``, ``{part}_y``,
        ``{part}_likelihood``) is also assigned to ``self.pose``.
        """
        return self._load_raw_video('video/pose')

    def load_motion_energy(self) -> np.ndarray:
        """Return the ROI motion energy, from the store or :meth:`fetch_motion_energy`.

        The per-frame scalar (on the ``leftCamera.times`` base) is also assigned
        to ``self.motion_energy``. Unlike ``lightningPose`` and ``times``, the
        ROIMotionEnergy dataset carries no ``_ibl_`` prefix, so its ONE dataset
        name differs.
        """
        return self._load_raw_video('video/motion_energy')

    def load_video_times_qc(self) -> dict[str, float]:
        """Return the camera-clock QC metrics, computing them if absent.

        Reads `video/times/qc` when it is stored; otherwise fetches the camera
        times from Alyx, scores them with :meth:`run_video_times_qc`, and
        writes the product.

        Returns
        -------
        dict
            Metric name -> value, also assigned to `self.video_times_qc`.

        """
        if hasattr(self, 'video_times_qc'):
            return self.video_times_qc
        if self.stored_product_exists('video/times/qc'):
            with h5py.File(self.filepath, 'r') as h5:
                self.video_times_qc = _load_scalars(h5['video/times/qc'])
            return self.video_times_qc
        self.load_camera_times()
        self.run_video_times_qc()
        self.save_h5(groups=['video'])
        return self.video_times_qc

    def run_video_times_qc(self) -> dict[str, float]:
        """Score the camera clock from the frame times already loaded.

        Reads `self.pose_times`, assigned by :meth:`fetch_camera_times`; it
        never fetches and never writes. `load_video_times_qc` is what does both.

        A `length_discrepancy` reaching `config.LENGTH_MISMATCH_THRESHOLD` is
        logged as a `VideoLengthError` against `video/times/qc`, not raised: a
        video outrunning its session has never blocked the traces cut from it,
        and must not start. `framerate_from_tpts` is recorded and not checked.

        Returns
        -------
        dict
            ``length_discrepancy`` (video duration minus ``session_length``,
            seconds) and ``framerate_from_tpts`` (median inter-frame interval,
            seconds), also assigned to ``self.video_times_qc``.
        """
        discrepancy = float(
            (self.pose_times[-1] - self.pose_times[0]) - self.session_length)
        self.video_times_qc = {
            'length_discrepancy': discrepancy,
            'framerate_from_tpts': float(np.median(np.diff(self.pose_times))),
        }
        if discrepancy >= LENGTH_MISMATCH_THRESHOLD:
            self.log_error(
                VideoLengthError(
                    f"Video–session length discrepancy {discrepancy:.0f}s "
                    f"exceeds {LENGTH_MISMATCH_THRESHOLD}s threshold"),
                product='video/times/qc')
        return self.video_times_qc

    def fetch_video_qc(self) -> dict[str, str]:
        """Live-fetch the eight ``VIDEO_QC_COLS`` labels, without storing them.

        Unlike every other QC on this class these labels are not a product:
        they change when IBL re-runs its QC and no repo parameter feeds them,
        so they are refetched rather than cached. Assigned to
        ``self.video_qc`` and returned.
        """
        from iblnm.io import get_video_qc
        self.video_qc = get_video_qc(self.eid, one=self.one)
        return self.video_qc

    def set_manual_qc(self, field: str, value: str,
                      region: str | None = None) -> None:
        """Set one manual QC verdict on this session and write it to the H5.

        The verdict is written on its own rather than through
        :meth:`save_h5`, so what a viewer persists does not depend on which
        products the session happens to be holding in memory.

        Parameters
        ----------
        field : str
            One of `config.LP_QC_LABELS`.
        value : str
            One of `config.IBL_QC_VALUES`. Both arguments are checked before
            the file is opened, so a bad verdict leaves the store untouched.
        region : str, optional
            Names the recording to label, writing `photometry/{region}/
            manual_qc`. Left None the verdict is the camera's, written to
            `video/manual_qc` — video is scored per session because there is
            one camera, photometry per region because there is one fiber each.

        Raises
        ------
        ValueError
            `field` is not an `LP_QC_LABELS` entry, or `value` is not an
            `IBL_QC_VALUES` verdict.
        """
        if field not in LP_QC_LABELS:
            raise ValueError(
                f"Unknown QC field: {field!r} (expected {LP_QC_LABELS})")
        if value not in IBL_QC_VALUES:
            raise ValueError(
                f"Invalid QC value: {value!r} (expected {IBL_QC_VALUES})")
        labels = (self.video_manual_qc if region is None
                  else self.photometry_manual_qc.setdefault(region, {}))
        labels[field] = value
        group = ('video/manual_qc' if region is None
                 else f'photometry/{region}/manual_qc')
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(self.filepath, 'a') as h5:
            _save_manual_qc(h5.require_group(group), labels)

    def _clear_manual_qc(self, modality: str) -> None:
        """Drop one modality's manual QC verdicts, in memory and on disk.

        Called where that modality's raw data is fetched: a verdict was passed
        on the frames or samples that have just been replaced, so it no longer
        describes anything stored. Reading a stored raw product back, or
        rebuilding anything derived from it, does not come through here.

        Parameters
        ----------
        modality : str
            'video', clearing the session's one verdict group, or 'photometry',
            clearing every region's — the raw photometry fetch brings back all
            regions at once, so it invalidates all of them together.
        """
        if modality == 'video':
            self.video_manual_qc = {}
        else:
            self.photometry_manual_qc = {}
        if not self.filepath.exists():
            return
        with h5py.File(self.filepath, 'a') as h5:
            if modality not in h5:
                return
            parents = ([h5['video']] if modality == 'video' else
                       [h5[f'photometry/{region}'] for region in h5['photometry']])
            for parent in parents:
                if 'manual_qc' in parent:
                    del parent['manual_qc']

    def _movement_signals(self) -> dict[str, pd.Series]:
        """Return the preprocessed movement channels, resampling them if absent.

        Reads `video/{label}/preprocessed` when it is stored; otherwise fetches
        the raw video datasets, resamples them with
        :meth:`extract_movement_signals`, and writes the product.

        Returns
        -------
        dict[str, pandas.Series]
            Channel label -> signal on the shared 1/POSE_FS time base, ready to
            hand to :meth:`extract_responses`. Also assigned to
            ``self.movement_signals``.

        """
        if hasattr(self, 'movement_signals'):
            return self.movement_signals
        signals = None
        if self.stored_product_exists('video/preprocessed'):
            with h5py.File(self.filepath, 'r') as h5:
                signals = _read_label_products(h5['video'], 'preprocessed',
                                               _load_time_series)
        if signals is None:
            self._load_raw_video_sources()
            signals = self.extract_movement_signals()
            self.save_h5(groups=['video'])
        self.movement_signals = signals
        return signals

    def _load_raw_video_sources(self) -> None:
        """Load the three raw video datasets, tolerating a missing signal source.

        Camera times are required — without them no channel can be placed on the
        session clock — so a missing `video/times` propagates. Pose and motion
        energy are independent and each is optional: a missing one is logged
        against its own product and leaves the other's channels intact.
        """
        self.load_camera_times()
        for product, error in (('video/pose', MissingLP),
                               ('video/motion_energy', MissingMotionEnergy)):
            try:
                self._load_raw_video(product)
            except error as e:
                self.log_error(e, product=product)

    def extract_movement_signals(self) -> dict[str, pd.Series]:
        """Resample the raw video already loaded onto the POSE_FS grid.

        Reads `self.pose`, `self.pose_times` and `self.motion_energy`, assigned
        by the video fetches; it never fetches and never writes.
        `_movement_signals` is what does both.

        The two signal sources are independent, so a session contributes the LP
        keypoint channels (``config.POSE_MEASURES``), the ``motion_energy``
        channel, or both, depending on which of ``self.pose`` and
        ``self.motion_energy`` is set.

        Returns
        -------
        dict[str, pandas.Series]
            Channel label -> signal on the shared 1/POSE_FS time base, also
            assigned to ``self.movement_signals``.
        """
        signals = {}
        if hasattr(self, 'pose'):
            # Resample raw pose to a common rate first, so speeds (px per time
            # step) and trace lengths are comparable across camera fps.
            pose, pose_times = resample_pose(self.pose, self.pose_times, POSE_FS)
            signals.update({
                label: pd.Series(movement_trace(pose, keypoints, reduction),
                                 index=pose_times)
                for label, (_, keypoints, reduction) in POSE_MEASURES.items()
            })
        if hasattr(self, 'motion_energy'):
            signals['motion_energy'] = resample_signal(
                pd.Series(self.motion_energy, index=self.pose_times), POSE_FS)
        self.movement_signals = signals
        return signals

    def load_pose_qc(self) -> dict:
        """Return the paw–wheel timing diagnostic, computing it if absent.

        Reads `video/pose/qc` when it is stored; otherwise loads the pose, the
        camera times and the wheel velocity and correlates them with
        :meth:`run_pose_qc`, then writes the product.

        This is the one cross-modal QC product: good pose is not enough, so a
        session with no wheel fails here with the wheel's own missing-data
        error rather than a pose one.

        Returns
        -------
        dict
            ``functions``, ``lags``, ``peak_lags`` and ``drift``, also assigned
            to ``self.pose_xcorr``.

        """
        if hasattr(self, 'pose_xcorr'):
            return self.pose_xcorr
        if self.stored_product_exists('video/pose/qc'):
            with h5py.File(self.filepath, 'r') as h5:
                self.pose_xcorr = _load_pose_xcorr(h5['video/pose/qc'])
            return self.pose_xcorr
        self.load_camera_times()
        self.load_pose()
        self.load_wheel()
        self.run_pose_qc()
        self.save_h5(groups=['video'])
        return self.pose_xcorr

    def run_pose_qc(self) -> dict:
        """Correlate paw speed against wheel speed per third.

        Reads `self.pose`, `self.pose_times` and `self.wheel_velocity`, put
        there by the pose and wheel loads; it never fetches and never writes.
        `load_pose_qc` is what does both.

        Returns
        -------
        dict
            ``functions`` (one cross-correlation per third), ``lags``,
            ``peak_lags`` and the scalar ``drift``, also assigned to
            ``self.pose_xcorr``.
        """
        paw_speed = movement_trace(self.pose, ['paw_l', 'paw_r'], 'sum_speed')
        finite = np.isfinite(paw_speed)  # drop untracked frames (NaN speed)
        functions, lags, peak_lags, drift = per_third_crosscorr(
            paw_speed[finite], self.pose_times[finite],
            np.abs(self.wheel_velocity.to_numpy()),
            self.wheel_velocity.index.to_numpy(),
        )
        self.pose_xcorr = {'functions': functions, 'lags': lags,
                           'peak_lags': peak_lags, 'drift': drift}
        return self.pose_xcorr

    # =========================================================================
    # Response Vector
    # =========================================================================

    _DEFAULT_FEATURE_EVENTS = (STIM_ONSET_EVENT, 'feedback_times')

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
            Event names to include. Defaults to `config.STIM_ONSET_EVENT` and
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

        responses = self.mask_subsequent_events(self.photometry_responses[brain_region])
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


    def _recording_magnitudes(self, region: str, hemisphere: str,
                              target_nm: str,
                              events: Sequence[str]) -> pd.DataFrame:
        """One region's per-trial response magnitudes, one row per event x trial.

        The stored cut is masked at the next event and baseline-subtracted
        before the `config.RESPONSE_WINDOWS['early']` mean is taken, so the
        magnitude carries the evoked component alone. Empty when the region's
        cut holds none of ``events``, so a region the analysis does not model
        drops out of the concatenation rather than raising.

        Parameters
        ----------
        region, hemisphere, target_nm : str
            The recording's entries in the session's parallel list columns.
            ``region`` also keys ``self.photometry_responses``.
        events : Sequence[str]
            Events to cut, in the order the rows are emitted. Events the stored
            cut does not carry are skipped.

        Returns
        -------
        pandas.DataFrame
            `_RECORDING_MAGNITUDE_COLUMNS`, one row per event x trial. `trial`
            is the trials table's own trial number, not the row position, so it
            joins to the trial regressors.
        """
        responses = self.subtract_baseline(
            self.mask_subsequent_events(self.photometry_responses[region]))
        tpts = responses.coords['time'].values
        trials = responses.coords['trial'].values
        cut_events = [event for event in events
                      if event in responses.coords['event'].values]
        if not cut_events:
            return pd.DataFrame(columns=_RECORDING_MAGNITUDE_COLUMNS)
        with warnings.catch_warnings():
            # A trial whose window is masked end to end averages an empty
            # slice. NaN is the intended magnitude there, and the null filter
            # downstream drops it; the warning would fire once per recording.
            warnings.simplefilter('ignore', RuntimeWarning)
            return pd.concat([
                pd.DataFrame({
                    'eid': self.eid,
                    'subject': self.subject,
                    'session_type': self.session_type,
                    'NM': self.NM,
                    'target_NM': target_nm,
                    'brain_region': region,
                    'hemisphere': hemisphere,
                    'event': event,
                    'trial': trials,
                    'response': compute_response_magnitude(
                        responses.sel(event=event).values, tpts,
                        RESPONSE_WINDOWS['early']),
                    'masked_fraction': compute_masked_fraction(
                        responses.sel(event=event).values, tpts,
                        RESPONSE_WINDOWS['early']),
                })
                for event in cut_events
            ], ignore_index=True)

    def _trial_regressors(self) -> pd.DataFrame:
        """This session's one-row-per-trial regressor frame.

        Reads the stored `peak_velocity` off the session when it is there; a
        session holding no wheel product gets an all-NaN ``peak_velocity``
        column rather than an error, and the complete-case filter drops those
        rows when a formula referencing it is fitted.
        """
        return analysis.build_trial_regressors(
            self.trials, getattr(self, 'wheel_peak_velocity', None),
            STIM_ONSET_EVENT)

    def _merge_response_magnitudes(self, events: Sequence[str]) -> pd.DataFrame:
        """Every recording's magnitudes with this session's trial regressors.

        The uncoded, unselected frame the modelling and plotting branches share:
        one row per recording x event x trial, carrying the trial-level columns
        beside the magnitude measured on that trial. A region the catalog names
        but the store holds no cut for contributes no rows.

        Parameters
        ----------
        events : Sequence[str]
            Events to cut, `config.RESPONSE_EVENTS` by default.

        Returns
        -------
        pandas.DataFrame
            `config.RESPONSE_MAGNITUDE_COLUMNS` plus the columns
            :func:`iblnm.task.add_relative_contrast` derives (``side``,
            ``choice_side``, ``relative_contrast``) and the two regressors no
            persession formula reads (``signed_contrast``, ``movement_time``).
            Also assigned to ``self.response_magnitudes``.
        """
        regressors = self._trial_regressors()
        magnitudes = _concat_frames(
            [self._recording_magnitudes(region, hemisphere, target_nm, events)
             for region, hemisphere, target_nm
             in zip(self.brain_region, self.hemisphere, self.target_NM)
             if region in self.photometry_responses],
            _RECORDING_MAGNITUDE_COLUMNS)
        merged = magnitudes.merge(regressors, on='trial', how='left')
        self.response_magnitudes = task.add_relative_contrast(merged)
        return self.response_magnitudes

    def _select_modeling_trials(self, events: Sequence[str] | None,
                                response_col: str | None) -> pd.DataFrame:
        """The uncoded modelling rows, both preparations' shared prefix.

        Assembles the frame and applies the trial exclusions
        (:func:`iblnm.analysis.select_modeling_trials`); coding happens in the
        caller, on whatever row set it ends up with. ``events`` selects which
        of the two frames is assembled — the focal one, one row per recording
        x event x trial, or the donor one, one row per trial and no photometry
        touched.

        Parameters
        ----------
        events : Sequence[str] or None
            Events to cut and merge magnitudes for. ``None`` is the donor
            case: the trial regressors alone, tagged with the first
            recording's hemisphere so the hemisphere-relative ``side`` and
            ``choice_side`` can be derived. Which hemisphere does not matter
            downstream — the other one negates both columns and every
            interaction they enter, which spans the same design space and so
            leaves R² unchanged.
        response_col : str or None
            Response magnitude column whose null rows are dropped. ``None``
            alongside ``events=None``: a donor frame carries no response.

        Returns
        -------
        pandas.DataFrame
            The retained trials, uncoded. In the focal case the merged frame
            is also left on ``self.response_magnitudes``, unselected.
        """
        if events is not None:
            frame = self._merge_response_magnitudes(events)
        else:
            frame = self._trial_regressors().assign(
                hemisphere=next(iter(self.hemisphere), None))
            frame = task.add_relative_contrast(frame)
        return analysis.select_modeling_trials(frame, response_col)

    def _prepare_model_frames(self, formulas: dict,
                              events: Sequence[str] = RESPONSE_EVENTS,
                              response_col: str = 'response',
                              min_trials: int = MIN_TRIALS_PERSESSION,
                              contrast_coding: str = 'log2',
                              ) -> dict[tuple[str, str], pd.DataFrame]:
        """Build this session's fit-ready frames, one per (region, event) cell.

        Four steps: merge the trial regressors with each region's magnitudes
        (:meth:`_merge_response_magnitudes`, whose uncoded result is left on
        ``self.response_magnitudes``), apply the response-independent trial
        exclusions to the whole session, then code and centre each cell on its
        own surviving rows. Coding runs per cell rather than per session
        because a cell's row set is not final until its null responses are
        dropped, and centring computed earlier would not be centring on the
        fitted rows.

        Parameters
        ----------
        formulas : dict
            Drop-one family, flat or event-keyed
            (:func:`resolve_event_family`); every event in ``events`` must have
            a family, which is resolved up front so a missing one raises
            ``MissingFormula`` before any coding work.
        events : Sequence[str]
            Events to cut and code. An event the stored responses do not carry
            yields no cell.
        response_col : str
            Per-trial response magnitude column the formulas model.
        min_trials : int
            A cell with fewer complete-case rows is not scorable and is omitted.
        contrast_coding : str
            Passed to :func:`iblnm.analysis.code_predictors`.

        Returns
        -------
        dict[tuple[str, str], pandas.DataFrame]
            ``(brain_region, event)`` -> that cell's coded, complete-case trial
            frame. Also assigned to ``self.model_frames``.
        """
        families = {event: resolve_event_family(formulas, event)
                    for event in events}
        selected = self._select_modeling_trials(events, response_col)

        self.model_frames = {}
        for (region, event), rows in selected.groupby(
                ['brain_region', 'event'], sort=False):
            coded = analysis.code_predictors(rows, contrast_coding)
            coded = coded.dropna(subset=analysis.formula_union_columns(
                families[event].values(), coded.columns))
            if len(coded) >= min_trials:
                self.model_frames[(region, event)] = coded
        return self.model_frames

    def prepare_donor_frame(self, contrast_coding: str = 'log2') -> DonorFrame:
        """Build this session's contribution to other sessions' swap nulls.

        :meth:`_prepare_model_frames` stopped after its response-independent
        selection step, coding on those rows: the trial regressors with the
        no-go, false-start and negative-reaction-time exclusions applied, then
        coded and centred. No photometry is loaded and no response-null rows
        are dropped, so a donor frame is one frame per session — not one per
        recording-event — and is typically longer than the focal frames it
        donates to. `iblnm.analysis.permutation_null_delta_r2` truncates the
        pair to the shorter length at swap time.

        Trial order is preserved, which is the whole point of the swap: the
        donor's regressor keeps its own serial structure while losing any
        relationship to the focal session's responses.

        Parameters
        ----------
        contrast_coding : str
            Passed to :func:`iblnm.analysis.code_predictors`. Must match the
            focal frames' coding, or the swapped column is on another scale.

        Returns
        -------
        DonorFrame
            This session's identity and its coded trial frame, carrying every
            `config.PERSESSION_REGRESSORS` column. Also assigned to
            ``self.donor_frame``.
        """
        self.load_trials()
        self.load_peak_velocity()
        selected = self._select_modeling_trials(None, None)
        self.donor_frame = DonorFrame(
            self.eid, self.subject, tuple(self.target_NM),
            analysis.code_predictors(selected, contrast_coding))
        return self.donor_frame

    def select_donors(self, donors: dict[str, DonorFrame],
                      donor_scope: str = 'exclude_subject',
                      ) -> list[pd.DataFrame]:
        """Narrow a donor pool to the sessions this one may swap columns with.

        The focal side of the comparison is ``self``, read straight off the
        session — its ``eid``, ``subject`` and ``target_NM``. There is no event
        to match on: a donor frame is built from trials alone, so one frame
        serves every event of every recording this session holds.

        Parameters
        ----------
        donors : dict[str, DonorFrame]
            Every prepared donor of the pass, keyed by eid, this session's own
            included; every scope excludes it.
        donor_scope : {'exclude_session', 'exclude_subject', 'same_target'}
            Which sessions may donate. See ``_SESSION_DONOR_SCOPES``.

        Returns
        -------
        list[pandas.DataFrame]
            The admitted donors' frames, in ``donors`` order; empty when the
            pool admits none. Only the swapped predictor column of each is
            read downstream.

        Raises
        ------
        ValueError
            ``donor_scope`` names no known scope.
        """
        if donor_scope not in _SESSION_DONOR_SCOPES:
            raise ValueError(f"Unrecognized donor_scope {donor_scope!r}; "
                             f"expected one of "
                             f"{sorted(_SESSION_DONOR_SCOPES)}")
        admits = _SESSION_DONOR_SCOPES[donor_scope]
        return [donor.frame for donor in donors.values() if admits(self, donor)]

    def fit_responses(self, formulas: dict, donors: dict[str, DonorFrame],
                      events: Sequence[str] = RESPONSE_EVENTS,
                      reference: str = 'full',
                      response_col: str = 'response',
                      donor_scope: str = 'exclude_subject',
                      n_bootstrap: int = PERSESSION_PVAL_N_BOOTSTRAP,
                      random_state: int = PERSESSION_PVAL_SEED,
                      ) -> pd.DataFrame:
        """This session's complete drop-one OLS result, group-free.

        Everything the per-session modelling pass produces for one session, from
        formulas and a donor mapping alone: the drop-one ΔR², the reference
        model's weights, the cross-session swap null and the p-value scored
        against it. The store is read for this session's own products and
        nothing else — no group is constructed and no other session is opened.

        Per scorable cell (:meth:`_prepare_model_frames`, one per region ×
        event): the family is fitted and differenced off ``reference``
        (:func:`iblnm.analysis.dropone_delta_r2`), the reference model's weights
        are read off the same fits (:func:`_coefficient_rows`), and the null is
        built by swapping each dropped predictor's column in from every admitted
        donor (:func:`iblnm.analysis.permutation_null_delta_r2`). The dropped
        ``predictor`` and the weight's ``regressor`` are the same six names, so
        the two join to one row per predictor.

        Parameters
        ----------
        formulas : dict
            Drop-one family, flat or event-keyed
            (:func:`resolve_event_family`); ``reference`` names the full model
            and every other key is a dropped predictor.
        donors : dict[str, DonorFrame]
            Every prepared donor of the pass, keyed by eid, this session's own
            included; ``donor_scope`` excludes it (:meth:`select_donors`). An
            empty mapping fits without scoring.
        events : Sequence[str]
            Events to cut and fit.
        reference : str
            Full-model key each reduced model's ΔR² is measured against.
        response_col : str
            Per-trial response magnitude column the formulas model.
        donor_scope : {'exclude_subject', 'exclude_session', 'same_target'}
            Which sessions may donate a swapped predictor column.
        n_bootstrap : int
            Length of each cell's null vector.
        random_state : int
            Seed for the swap rng, created once per session. One rng per
            session rather than one per population changes which donors the
            bootstrap resamples, not the statistic.

        Returns
        -------
        pandas.DataFrame
            ``config.OLS_PERSESSION_COLUMNS`` less ``q_value``, one row per
            (region, event, dropped predictor); the FDR correction spans
            sessions, so the group adds that column after collecting these.
            Also assigned to ``self.response_ols``. A cell whose design is
            degenerate for any family member contributes no rows; a cell with
            no scorable donor keeps its fit and carries an empty ``null`` and a
            NaN ``p_value``.
        """
        self.load_trials()
        self.load_peak_velocity()
        self.load_responses('photometry')
        frames = self._prepare_model_frames(formulas, events, response_col)
        donor_frames = self.select_donors(donors, donor_scope)
        target_by_region = dict(zip(self.brain_region, self.target_NM))
        rng = np.random.default_rng(random_state)

        cells = []
        for (region, event), frame in frames.items():
            family = resolve_event_family(formulas, event)
            fits = {name: self.fit_response_model(frame, formula, response_col)
                    for name, formula in family.items()}
            if any(fit is None for fit in fits.values()):
                continue
            nulls = analysis.permutation_null_delta_r2(
                frame, donor_frames, family[reference],
                {name: formula for name, formula in family.items()
                 if name != reference},
                response_col, rng=rng, n_bootstrap=n_bootstrap)
            rows = _dropone_rows(fits, len(frame), reference)
            cells.append(_score_against_null(rows, nulls, len(donor_frames))
                         .assign(eid=self.eid, subject=self.subject,
                                 target_NM=target_by_region[region],
                                 brain_region=region, event=event))

        self.response_ols = _concat_frames(cells, _SESSION_OLS_COLUMNS)
        return self.response_ols

    @staticmethod
    def fit_response_model(df: pd.DataFrame, formula: str,
                           response_col: str = 'response'):
        """Fit one OLS response model on a prepared trial frame.

        Thin, event/region-agnostic wrapper over ``analysis.fit_ols``: the
        ``{response}`` placeholder in ``formula`` is filled with ``response_col``
        before fitting, so the caller owns which magnitude column is the
        response. Static because it reads nothing off the session — the
        group-level drop-one calls it once per formula on a frame it already
        holds.

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

    def delta_r_squared(self, fit, cv: int = None) -> pd.Series:
        """Leave-one-regressor-out drop in R² for each block of an encoding fit.

        For every block in ``fit.slices`` all of its columns are dropped from
        the (already z-scored) design and the ridge model is refit at the same
        ``fit.alpha``; the block's contribution is ΔR² = full R² − reduced R².

        Parameters
        ----------
        fit : iblnm.analysis.EncodingFit
            A fitted encoding model (see
            :func:`iblnm.analysis.fit_encoding_model`).
        cv : int, optional
            ``None`` (default) scores in sample, using ``fit.r2`` as the full
            reference; an int scores the pooled out-of-fold R² over that many
            contiguous KFold splits, both for the full and the reduced models.

        Returns
        -------
        pd.Series
            ΔR² indexed by block name, sorted descending (largest contribution
            first).
        """
        full_r2 = fit.r2 if cv is None else analysis.ridge_r2(
            fit.design, fit.target, fit.alpha, cv)
        deltas = {}
        for name, span in fit.slices.items():
            keep = np.ones(fit.design.shape[1], dtype=bool)
            keep[span] = False
            reduced_r2 = analysis.ridge_r2(
                fit.design[:, keep], fit.target, fit.alpha, cv)
            deltas[name] = full_r2 - reduced_r2
        return pd.Series(deltas, name='delta_r2').sort_values(ascending=False)


def _session_for_processing(h5_path, row, one):
    """Build the session `process` hands to `fn`.

    Parameters
    ----------
    h5_path : Path
        The session's file in the group's store. Read when it exists, so `fn`
        sees what was already built; otherwise the session starts from `row`.
        Adopted as `ps.filepath` either way, so a session whose file does not
        exist yet still writes what `fn` builds into the group's store rather
        than into the default one.
    row : pd.Series or dict
        The catalog row, as a dict when it has crossed a pickle boundary.
    one : one.api.One
        The connection the session queries Alyx through.
    """
    if h5_path.exists():
        return PhotometrySession.from_h5(h5_path, one=one)
    ps = PhotometrySession(pd.Series(row), one=one, load_data=False)
    ps.filepath = h5_path
    return ps


def _process_one(ps, h5_path, fn, kwargs):
    """Run `fn` on one session and flush its errors; return `fn`'s result.

    A `BlockingIOError` propagates instead of being logged, and the flush is
    skipped: it means another process holds the file, which `process` retries
    at the end of the pass. Recording it would mark the session permanently
    failed under the absent-data + present-error rule, for what is a transient
    collision. Any other exception is logged against the session and the
    result is None.
    """
    try:
        result = fn(ps, **kwargs)
    except BlockingIOError:
        raise
    except Exception as e:
        ps.log_error(e)
        result = None
    if h5_path.exists() or ps.errors:
        ps.save_h5(h5_path, groups=['errors'])
    return result


def _process_worker(eid, row_dict, h5_dir, fn, kwargs):
    """Worker function for parallel process(). Runs in a subprocess.

    Creates its own ONE connection, builds a PhotometrySession, calls
    fn(ps, **kwargs), and flushes errors to H5.
    """
    from iblnm.io import _get_default_connection

    one = _get_default_connection()
    h5_path = Path(h5_dir) / f'{eid}.h5'
    ps = _session_for_processing(h5_path, row_dict, one)
    return _process_one(ps, h5_path, fn, kwargs)


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


# Comparison each PHOTOMETRY_QC_THRESHOLDS entry names, kept as strings in
# config.py so the thresholds stay a plain serializable mapping.
_QC_COMPARISONS = {'>=': operator.ge, '>': operator.gt,
                   '<=': operator.le, '<': operator.lt}


def _explode_recordings(sessions: pd.DataFrame) -> pd.DataFrame:
    """Explode session rows to one row per recording, numbering the fibers.

    The `PARALLEL_COLS` list columns are exploded together, so each row keeps
    its region with the hemisphere and target NM recorded alongside it.
    `fiber_idx` numbers a session's recordings in the order they are listed.
    """
    df = sessions.explode(PARALLEL_COLS).copy()
    df['fiber_idx'] = df.groupby('eid').cumcount()
    return df.reset_index(drop=True)


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
    one : one.api.One, optional
        ONE connection instance. Only needed when a load method has to fetch
        from Alyx; sessions read back from H5 need none.
    h5_dir : Path, optional
        Directory containing {eid}.h5 files.
    """

    def __init__(self, sessions, one=None, h5_dir=None):
        self._catalog = sessions.reset_index(drop=True)
        self._filter_mask = pd.Series(True, index=self._catalog.index)
        self._dedup_mask = pd.Series(True, index=self._catalog.index)
        self._recordings_targetnms = False
        self._recordings_photometry_qc = False
        self.one = one
        self.h5_dir = h5_dir if h5_dir is not None else SESSIONS_H5_DIR
        self._sessions = {}  # eid → PhotometrySession cache
        self.response_features = None
        self.performance = None
        # Per-recording raw photometry QC, scanned by complete_catalog.
        self.photometry_qc = None
        self.psychometric_features = None
        self.similarity_matrix = None
        self.decoder = None
        self.persession_ols_features = None
        self.cca_result = None
        self.cohort_cca_results = None
        self.cohort_cca_data = None
        self.cohort_cca_cross_projections = None
        self.cohort_cca_weight_similarities = None
        self.lmm_fits = {}
        self._lmm_group_by = None

    @classmethod
    def from_catalog(cls, catalog, one=None, h5_dir=SESSIONS_H5_DIR,
                     scan_h5=True):
        """Build a group from a session catalog DataFrame.

        Validates parallel list columns and, when ``h5_dir`` is given and
        ``scan_h5`` is True, completes the catalog with everything the filters
        read (``complete_catalog``, one open per catalogued file, so it can take
        a moment). Call ``filter_sessions`` separately.

        Parameters
        ----------
        catalog : pd.DataFrame
            Session catalog (one row per session, with list columns for
            brain_region, hemisphere, target_NM).
        one : one.api.One, optional
            ONE connection instance. Omit it for a group that reads only the
            store: every product is loaded from ``h5_dir``, and a session whose
            file lacks one raises rather than fetching it from Alyx.
        h5_dir : Path, optional
            Directory containing {eid}.h5 files, retained on the group for
            loading and processing. When provided and ``scan_h5`` is True, the
            stored errors, performance and raw photometry QC are scanned out of
            it.
        scan_h5 : bool, optional
            When True (default), run ``complete_catalog``. Set False to reuse
            the filter columns already present on ``catalog`` and skip the scan
            -- useful when rebuilding a group from another group's ``sessions``
            (each filter column is defaulted empty only if it is missing). A
            group that skips the scan over a catalog carrying none of those
            columns keeps nothing when the filters reading them are on.
        """
        from iblnm.config import SESSION_SCHEMA

        df = enforce_schema(catalog.copy(), SESSION_SCHEMA)

        df = validate_parallel_lists(df, PARALLEL_COLS)

        # Default the columns the scan fills, so a group that skips it fails
        # the filters reading them rather than passing them by absence.
        for column in ('logged_errors', 'contrasts'):
            if column not in df.columns:
                df[column] = [[] for _ in range(len(df))]
        if 'fraction_correct' not in df.columns:
            df['fraction_correct'] = np.nan

        group = cls(df, one=one, h5_dir=h5_dir)
        if h5_dir is not None and scan_h5:
            group.complete_catalog()
        return group

    @classmethod
    def from_h5_dir(cls, h5_dir, one=None, scan_h5=True):
        """Build a group from the `metadata` groups of a directory of H5 files.

        The store's own inverse: where `from_catalog` starts from
        `sessions.pqt`, this reconstructs the catalog from what was written,
        for a script that has just built those files and holds no catalog of
        its own. Files with no `metadata` group are skipped.

        The per-subject rankings `day_n` and `session_n` are derived here rather
        than in `from_catalog`: they need every one of the subject's sessions at
        once, which only this method has, and deriving them in `from_catalog`
        would recompute them for every script reading the written catalog.
        Neither column is in `config.SESSION_SCHEMA`, so `enforce_schema`
        carries them along untouched.

        Parameters
        ----------
        h5_dir : Path or str
            Directory of `{eid}.h5` files, adopted as the group's `h5_dir`.
        one : one.api.One, optional
            ONE connection, needed only if the group later has to fetch.
        scan_h5 : bool, optional
            Forwarded to `from_catalog`. When True (default) every file is
            opened a second time by `complete_catalog`; pass False to skip that,
            which leaves `logged_errors` empty, `fraction_correct` NaN and
            `contrasts` empty, and so keeps nothing if the filters reading them
            are on.

        Returns
        -------
        PhotometrySessionGroup
            Group over every session found, with all `SESSION_SCHEMA` columns,
            `day_n` (days since the subject's first session) and `session_n`
            (that session's rank among the subject's days).
        """
        rows = []
        for fpath in sorted(Path(h5_dir).glob('*.h5')):
            with h5py.File(fpath, 'r') as h5:
                row = _read_metadata(h5)
            if row:
                rows.append(row)
        catalog = pd.DataFrame(rows)
        if rows:
            dates = pd.to_datetime(catalog['start_time'],
                                   format='ISO8601').dt.date
            by_subject = dates.groupby(catalog['subject'])
            catalog['day_n'] = by_subject.transform(
                lambda subject_dates: [(date - subject_dates.min()).days
                                       for date in subject_dates])
            catalog['session_n'] = by_subject.rank(method='dense')
        return cls.from_catalog(catalog, one=one, h5_dir=h5_dir,
                                scan_h5=scan_h5)

    def fix_catalog(self) -> None:
        """Apply the Alyx metadata fixups to the catalog the group holds.

        `sessions` returns a copy, so a caller cannot repair the table the group
        filters and builds from without going through the group. It matters
        because `PhotometrySession` reads `brain_region` off its catalog row to
        name the photometry columns, so the fixed regions have to be the ones
        `process` iterates over.

        The frame is replaced rather than mutated: `util.fix_catalog` returns a
        new one, and it keeps its index through the fixups, which the positional
        `_filter_mask` and `_dedup_mask` need.
        """
        self._catalog = fix_catalog(self._catalog)

    @property
    def sessions(self):
        """Session-level view: _catalog rows passing both dedup and filter masks."""
        combined = self._dedup_mask & self._filter_mask
        return self._catalog[combined].copy().reset_index(drop=True)

    @property
    def recordings(self):
        """Recording-level view: sessions exploded to one row per region.

        Reflects the current filter and dedup masks. Filters to
        _recordings_targetnms and to _recordings_photometry_qc, the
        (eid, brain_region) pairs clearing the QC thresholds, both set by
        filter_sessions and both skipped when False.
        """
        df = _explode_recordings(self.sessions)
        if self._recordings_targetnms is not False:
            df = df[df['target_NM'].isin(self._recordings_targetnms)]
        if self._recordings_photometry_qc is not False:
            keys = pd.Series(list(zip(df['eid'], df['brain_region'])),
                             index=df.index, dtype=object)
            df = df[keys.isin(self._recordings_photometry_qc)]
        return df.reset_index(drop=True)

    def _passing_recordings(self, thresholds: dict) -> set:
        """(eid, brain_region) pairs whose stored raw QC clears `thresholds`.

        Reads the QC `complete_catalog` scanned, so it costs nothing on a group
        built with the scan and one pass over the store on a group without it.

        Parameters
        ----------
        thresholds : dict
            `collect_qc` column -> (comparison, cutoff), as
            `config.PHOTOMETRY_QC_THRESHOLDS`. Every entry must pass for the
            recording to survive. A recording with no stored value for a
            metric — its QC group absent, or its build failed — scores NaN and
            fails, as does one whose metric was never stored by any session.

        Returns
        -------
        set of tuple
            The (eid, brain_region) pairs that passed.
        """
        qc = self.collect_qc().reindex(
            columns=['eid', 'brain_region', *thresholds])
        passing = pd.Series(True, index=qc.index)
        for column, (comparison, cutoff) in thresholds.items():
            passing &= _QC_COMPARISONS[comparison](qc[column], cutoff)
        return set(map(tuple, qc.loc[passing, ['eid', 'brain_region']].to_numpy()))

    def filter_sessions(self, session_types=SESSION_TYPES_TO_ANALYZE,
                        exclude_subjects=SUBJECTS_TO_EXCLUDE,
                        exclude_eids=EIDS_TO_DROP,
                        qc_blockers=ANALYSIS_QC_BLOCKERS,
                        targetnms=TARGETNMS_TO_ANALYZE,
                        photometry_qc=PHOTOMETRY_QC_THRESHOLDS,
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
        photometry_qc : dict or False
            Raw-photometry QC thresholds, `collect_qc` column ->
            (comparison, cutoff). Defaults to
            config.PHOTOMETRY_QC_THRESHOLDS. Applies to ``recordings`` and not
            to ``sessions``, as the target-NM filter does: a recording failing
            any threshold is dropped and the rest of its session stays in
            scope. Reads the QC ``complete_catalog`` scanned.
        min_performance : float, dict, or False
            Minimum fraction_correct. Defaults to config.MIN_PERFORMANCE. A
            session with no stored performance scores NaN and is dropped.
        required_contrasts : frozenset of float or False
            Required contrast set. Defaults to config.REQUIRED_CONTRASTS. A
            session with no stored performance has no contrasts and is dropped.
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

        if min_performance is not False:
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

        if required_contrasts is not False:
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
        # Set False first so `recordings` counts what is in scope for the QC
        # filter, before that filter is the one applied.
        self._recordings_photometry_qc = False
        n_recordings = len(self.recordings)
        if photometry_qc is not False:
            self._recordings_photometry_qc = self._passing_recordings(photometry_qc)

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
        # The QC filter drops recordings rather than sessions, so its count is
        # over recordings and carries no entry in the session mask.
        qc_removed = n_recordings - len(self.recordings)
        if qc_removed:
            lines.append(f"  -{qc_removed:4d} photometry_qc (recordings)")
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
        """Get or create a PhotometrySession for a recording row.

        The session reads and writes the group's ``h5_dir``, not the default
        store, so a group pointed at another directory keeps its sessions there.
        """
        eid = rec['eid']
        if eid not in self._sessions:
            ps = PhotometrySession(rec, one=self.one, load_data=False)
            ps.filepath = Path(self.h5_dir) / f'{eid}.h5'
            self._sessions[eid] = ps
        return self._sessions[eid]

    def __iter__(self):
        for _, rec in self.recordings.iterrows():
            yield rec, self._get_session(rec)

    @contextmanager
    def _open_h5(self, eid: str):
        """Open one session's stored H5 read-only, yielding None when absent.

        The collectors read products straight out of the file rather than
        through a `PhotometrySession`, whose load methods would build and write
        what they found missing. A session with no file in `self.h5_dir` yields
        None so a caller can treat it as it treats an empty file.
        """
        fpath = Path(self.h5_dir) / f'{eid}.h5'
        if not fpath.exists():
            yield None
            return
        with h5py.File(fpath, 'r') as h5:
            yield h5

    def _error_log(self, eids) -> pd.DataFrame:
        """Read the `errors/` tree of each named session's H5 into one table.

        Parameters
        ----------
        eids : iterable of str
            Sessions to read, in the order the caller wants them. An eid with
            no file in `self.h5_dir` contributes no rows.

        Returns
        -------
        pandas.DataFrame
            Error log with the `util.LOG_COLUMNS` schema.
        """
        rows = []
        for eid in eids:
            with self._open_h5(eid) as h5:
                if h5 is not None:
                    rows.extend(read_error_tree(h5))
        return pd.DataFrame(rows, columns=LOG_COLUMNS)

    def collect_errors(self) -> pd.DataFrame:
        """Aggregate the filtered sessions' logged errors into one table.

        Only sessions surviving the filter and dedup masks are read, so a
        rollup written from this reports on the cohort the group defines rather
        than on whatever files happen to sit in `self.h5_dir`.

        Returns
        -------
        pandas.DataFrame
            One row per logged error, with the `util.LOG_COLUMNS` schema.
        """
        return self._error_log(self.sessions['eid'])

    def complete_catalog(self) -> None:
        """Read every catalogued session's H5 once, filling in what filters read.

        `filter_sessions` reads three things a catalog does not carry: the
        error types blocking a session, its behavioral performance, and its
        recordings' raw photometry QC. All three are stored per session, so one
        walk scans them together — one open per catalogued file — rather than a
        walk apiece. It reads `self._catalog` and not the filtered view,
        because the mask it feeds does not exist while it runs.

        `logged_errors`, `fraction_correct` and `contrasts` are joined onto
        `_catalog`; the QC is per recording rather than per session, so it
        lands on `self.photometry_qc` instead, where `_passing_recordings`
        reads it. Duplicate (eid, error_type, error_message) entries are
        dropped, so an error logged by two build attempts counts once.

        A session with no file, or without one of the products, gets an empty
        error list, a NaN `fraction_correct`, an empty `contrasts` list and NaN
        for every QC metric — each of which fails the filter reading it, rather
        than passing it by absence.
        """
        regions_by_eid = dict(iter(
            _explode_recordings(self._catalog).groupby('eid')['brain_region']))
        errors, performance, qc = [], [], []
        for eid in self._catalog['eid']:
            stored_qc = {}
            with self._open_h5(eid) as h5:
                if h5 is not None:
                    errors.extend(read_error_tree(h5))
                    if 'trials/performance' in h5:
                        performance.append({'eid': eid} | _load_performance(
                            h5['trials/performance']))
                    if 'photometry' in h5:
                        stored_qc = _read_photometry_qc(h5['photometry'])
            qc.extend({'eid': eid, 'brain_region': region} | stored_qc.get(region, {})
                      for region in regions_by_eid.get(eid, []))

        self.photometry_qc = (pd.DataFrame(qc) if qc else
                              pd.DataFrame(columns=['eid', 'brain_region']))
        error_types = (deduplicate_log(pd.DataFrame(errors, columns=LOG_COLUMNS))
                       .groupby('eid')['error_type'].apply(list).to_dict())
        scanned = pd.DataFrame(performance,
                               columns=['eid', 'fraction_correct', 'contrasts'])
        catalog = self._catalog.drop(
            columns=['logged_errors', 'fraction_correct', 'contrasts'],
            errors='ignore')
        catalog['logged_errors'] = [error_types.get(eid, [])
                                    for eid in catalog['eid']]
        catalog = catalog.merge(scanned, on='eid', how='left')
        catalog['contrasts'] = [contrasts if isinstance(contrasts, list) else []
                                for contrasts in catalog['contrasts']]
        self._catalog = catalog

    def collect_qc(self) -> pd.DataFrame:
        """Every catalogued recording's stored raw photometry QC.

        Returns what `complete_catalog` scanned, running that scan first if the
        group was built without one.

        Returns
        -------
        pandas.DataFrame
            One row per catalogued recording, with `eid`, `brain_region` and
            one column per metric stored in `photometry/{region}/raw/qc`. The
            band is suffixed into the metric name (`n_unique_samples_GCaMP`),
            because QC is stored per region but not per band. A recording whose
            QC group is absent — never built, or its build failed — gets NaN
            for every metric, which fails any threshold compared against it.
        """
        if self.photometry_qc is None:
            self.complete_catalog()
        return self.photometry_qc

    def collect_pose(self, video_qc: dict | None = None) -> pd.DataFrame:
        """Roll the filtered sessions' ``video`` groups up into the pose table.

        One row per session holding a ``video`` group — sessions without LP
        included, their trace-derived columns NaN — plus a bare row for each
        session whose errors record ``MissingVideoTimestamps`` and which
        therefore has no group at all. The movement deltas are recomputed on
        every run rather than read back as stored scalars, so
        `MOVEMENT_RESPONSE_WINDOW` and `BASELINE_WINDOW` stay adjustable
        without re-extracting traces.

        Parameters
        ----------
        video_qc : dict, optional
            eid -> the eight `VIDEO_QC_COLS` labels fetched from Alyx. Passed
            in rather than read from the H5: the labels change when IBL re-runs
            its QC, so nothing here could tell a stored copy had gone stale.
            Sessions absent from it contribute no QC columns and score NaN.

        Returns
        -------
        pandas.DataFrame
            Columns: ``eid``, ``session_type``, ``lp_exists``, one movement
            delta per channel, ``drift``, ``peak_lag_early/mid/late``,
            ``peak_val_early/mid/late``, the `LP_QC_LABELS` manual verdicts,
            ``mean_rt``, the eight `VIDEO_QC_COLS`, ``length_discrepancy``,
            ``framerate_from_tpts``, ``video_qc_score`` and
            ``fraction_correct``. Rows are in catalog order, not sorted.
        """
        video_qc = video_qc or {}
        errors_by_eid = (self.collect_errors().groupby('eid')['error_type']
                         .agg(set).to_dict())
        if self.performance is None:
            self.load_performance()
        performance = self.performance.reindex(columns=['eid', 'fraction_correct'])

        rows = []
        for _, session in self.sessions.iterrows():
            eid = session['eid']
            error_types = errors_by_eid.get(eid, set())
            row = {'eid': eid, 'session_type': session['session_type']}
            with self._open_h5(eid) as h5:
                if h5 is None or 'video' not in h5:
                    # No group to read: the session is reported only when the
                    # camera clock is what it was missing.
                    if 'MissingVideoTimestamps' in error_types:
                        rows.append(row | {'lp_exists': False,
                                           'video_qc_score': -1.0})
                    continue
                row['mean_rt'] = _read_mean_rt(h5)
                row |= _pose_row(h5['video'], video_qc.get(eid, {}), error_types)
            rows.append(row)

        df_pose = pd.DataFrame(rows)
        for col in POSE_TRACE_COLUMNS:
            if col not in df_pose.columns:
                df_pose[col] = np.nan
        return df_pose.merge(performance, on='eid', how='left')

    def collect_session_errors(self) -> pd.DataFrame:
        """Every catalogued session's error types, as `filter_sessions` reads them.

        Rescans the store through `complete_catalog` and returns the column it
        joined on, for a script that wants that table before the filters run —
        to append a synthetic blocker of its own, say.

        Returns
        -------
        pandas.DataFrame
            Columns ['eid', 'logged_errors'], one row per catalogued session in
            catalog order; a session with no H5 file, or none logged, gets an
            empty list.
        """
        self.complete_catalog()
        return self._catalog[['eid', 'logged_errors']].copy()

    def collect_donor_frames(self, frames: list[DonorFrame | None],
                             ) -> dict[str, DonorFrame]:
        """Assemble the first pass's returns into the swap null's donor pool.

        Takes what `process` returned — one item per filtered session, in
        `self.sessions` order — and keys it by the session that produced it.
        The pool is what `PhotometrySession.select_donors` narrows per focal
        session; it is held in memory for the fitting pass and never written.

        Parameters
        ----------
        frames : list[DonorFrame or None]
            Per-session returns in `self.sessions` order. `process` logs a
            session whose function raised and returns None for it; such a
            session is omitted rather than stored as a null donor, which would
            fail at swap time instead of here.

        Returns
        -------
        dict[str, DonorFrame]
            eid -> that session's prepared donor frame, in session order.
        """
        return {eid: frame
                for eid, frame in zip(self.sessions['eid'], frames)
                if frame is not None}

    def collect_fits(self, fits: list[pd.DataFrame | None],
                     n_bootstrap: int = PERSESSION_PVAL_N_BOOTSTRAP,
                     random_state: int = PERSESSION_PVAL_SEED,
                     ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Assemble the fitting pass's returns into the population tables.

        A session scores its own rows against its own null, but the
        false-discovery-rate correction and the per-mouse pooling both span
        sessions, so they are the group's work: this is where `q_value` is
        filled and where a mouse's sessions are pooled into one p-value.

        Parameters
        ----------
        fits : list[pandas.DataFrame or None]
            Per-session returns of `PhotometrySession.fit_responses`
            (`config.OLS_PERSESSION_COLUMNS` less `q_value`), in
            `self.sessions` order. `process` returns None for a session whose
            function raised; it contributes no rows.
        n_bootstrap : int
            Pooled-null draws per per-mouse cell. It does not set the p-value
            floor; the sessions' donor counts do.
        random_state : int
            Seed for the pooling bootstrap.

        Returns
        -------
        ols : pandas.DataFrame
            `config.OLS_PERSESSION_COLUMNS`, one row per (recording, event,
            dropped predictor), `q_value` corrected within each
            `config.PERSESSION_FDR_GROUP_COLS` family.
        mouse : pandas.DataFrame
            `RESPONSE_OLS_MOUSE_PVAL_COLUMNS`, one row per (target_NM, event,
            predictor, subject), pooled from those rows' own `null` vectors
            (:func:`assemble_mouse_pvalue_table`) and corrected over the same
            families at that coarser grain.
        """
        ols = analysis.add_fdr_qvalues(
            _concat_frames([fit for fit in fits if fit is not None],
                           _SESSION_OLS_COLUMNS),
            group_cols=PERSESSION_FDR_GROUP_COLS)
        mouse = analysis.add_fdr_qvalues(
            assemble_mouse_pvalue_table(ols, n_bootstrap=n_bootstrap,
                                        random_state=random_state),
            group_cols=PERSESSION_FDR_GROUP_COLS)
        return ols[OLS_PERSESSION_COLUMNS], mouse

    def filter_to_recordings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Narrow a saved frame to the group's current recordings.

        The read-side counterpart of the filters: a parquet file written by an
        earlier run covers whatever sessions that run analysed, and this keeps
        the rows the group's own `filter_sessions` and `deduplicate` masks
        admit. No I/O — the caller reads the file.

        Matching is on whichever of `eid` and `brain_region` the frame carries,
        so a per-recording frame loses a region dropped by the photometry-QC or
        target filter while its session's other regions stay, and a
        per-session frame is narrowed on the eid alone. A frame keyed by
        neither — the per-mouse table, at its coarser grain — is returned
        whole rather than emptied.

        Parameters
        ----------
        df : pandas.DataFrame
            Any frame; only its identifier columns are read.

        Returns
        -------
        pandas.DataFrame
            A copy narrowed to the matching rows, in their original order, or
            `df` itself when it carries no identifier column.
        """
        identifiers = [column for column in ('eid', 'brain_region')
                       if column in df.columns]
        if not identifiers:
            return df
        keys = pd.MultiIndex.from_frame(df[identifiers])
        kept = pd.MultiIndex.from_frame(self.recordings[identifiers])
        return df[keys.isin(kept)].copy()

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

        Sessions that hit an HDF5 file lock are collected and re-run once at
        the end of the pass, rather than retried in place: the lock is held
        only for the length of one read or write, so by the time the pass ends
        the process that held it is almost certainly done. No sleeps, no
        per-open backoff.

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
            Results from fn, one per unique session, in `self.sessions` order.
            None for failed sessions, including one still locked on the retry.
        """
        sessions = self.sessions
        results, blocked = self._process_pass(sessions, fn, workers, **kwargs)
        if blocked:
            retried, _ = self._process_pass(
                sessions[sessions['eid'].isin(blocked)], fn, workers, **kwargs)
            results.update(retried)
        return [results[eid] for eid in sessions['eid']]

    def _process_pass(self, sessions, fn, workers, **kwargs):
        """Run one pass of `process` over `sessions`, sequentially or pooled.

        Returns ``(results, blocked)``: an eid-keyed dict of results, and the
        set of eids whose run raised `BlockingIOError` and so is worth
        repeating.
        """
        if workers > 1:
            return self._process_parallel(sessions, fn, workers, **kwargs)
        return self._process_sequential(sessions, fn, **kwargs)

    def _process_sequential(self, sessions, fn, **kwargs):
        """Single-process implementation of one `process` pass."""

        results, blocked = {}, set()
        for _, row in tqdm(sessions.iterrows(), total=len(sessions),
                           desc="Processing"):
            eid = row['eid']
            h5_path = Path(self.h5_dir) / f'{eid}.h5'
            ps = _session_for_processing(h5_path, row, self.one)
            try:
                results[eid] = _process_one(ps, h5_path, fn, kwargs)
            except BlockingIOError:
                results[eid] = None
                blocked.add(eid)
        return results, blocked

    def _process_parallel(self, sessions, fn, workers, **kwargs):
        """Parallel implementation of one `process` pass.

        Each worker creates its own ONE connection and PhotometrySession.
        fn must be a picklable top-level function (not a lambda or closure).
        Only the row dict crosses the pickle boundary; bulk data is read and
        written by each worker from its own H5 file.
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed

        # Serialize rows as dicts for pickling
        tasks = {row['eid']: row.to_dict() for _, row in sessions.iterrows()}

        results = dict.fromkeys(tasks)
        blocked = set()
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
                    results[eid] = future.result()
                except BlockingIOError:
                    blocked.add(eid)
                except Exception as e:
                    print(f"\n  FATAL: {eid}: {type(e).__name__}: {e}")

        return results, blocked

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
                                 alternative='two-sided', seed=42,
                                 eids_to_process=None):
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

        view = self.recordings if unit == 'recordings' else self.sessions
        rng = np.random.default_rng(seed)

        if eids_to_process is not None:
            eid_inds = np.where(view['eid'].isin(eids_to_process))[0]
        else:
            eid_inds = np.arange(len(view))

        rows = []
        for pos in tqdm(eid_inds, desc="Permutation test"):
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

    def response_varcomp(self, coefficients, *, mcmc, tau_prior, min_mice,
                         min_sessions_per_mouse, grid_size, hdi_prob):
        """Per-cell mouse-vs-session variance components from the coefficients table.

        Groups the per-session coefficients by ``(target_NM, event, regressor)``
        and, for each cell, applies the inclusion rule (drop mice with fewer than
        ``min_sessions_per_mouse`` sessions; keep the cell only if at least
        ``min_mice`` mice survive), fits the measurement-error variance-components
        model on the survivors, and reduces each posterior to summary stats and a
        KDE-on-grid. Omitted cells appear in neither output. Reads no config —
        the caller passes every analysis choice in.

        Parameters
        ----------
        coefficients : pandas.DataFrame
            Per-session coefficients at grain ``(eid, subject, target_NM,
            brain_region, event, regressor)`` with ``coef`` and ``coef_se``
            columns; ``subject`` is the mouse id. Each row is one session.
        mcmc : dict
            Sampler settings forwarded to
            :func:`iblnm.analysis.fit_measurement_error_varcomp` (``draws``,
            ``tune``, ``chains``, ``target_accept``, ``random_seed``).
        tau_prior : tuple of (str, float)
            tau prior family and scale for the fit.
        min_mice : int
            Minimum surviving mice required to fit a cell.
        min_sessions_per_mouse : int
            Mice with fewer sessions than this in a cell are dropped before the
            mouse-count check.
        grid_size : int
            KDE grid length for each violin shape.
        hdi_prob : float
            Highest-density-interval mass for the summary bounds.

        Returns
        -------
        summary_df : pandas.DataFrame
            One row per (included cell, component) with columns
            ``RESPONSE_VARCOMP_SUMMARY_COLUMNS``; ``n_mice``/``n_sessions`` count
            survivors.
        violin_df : pandas.DataFrame
            Long-form KDE outlines, ``grid_size`` rows per (cell, component),
            columns ``RESPONSE_VARCOMP_VIOLIN_COLUMNS``.
        """
        summary_rows, violin_frames = [], []
        cell_keys = ['target_NM', 'event', 'regressor']
        for (target_nm, event, regressor), cell in coefficients.groupby(cell_keys):
            counts = cell['subject'].value_counts()
            survivors = cell[cell['subject'].isin(
                counts.index[counts >= min_sessions_per_mouse])]
            if survivors['subject'].nunique() < min_mice:
                continue
            v_mouse, v_session = fit_measurement_error_varcomp(
                survivors['coef'].to_numpy(), survivors['coef_se'].to_numpy(),
                survivors['subject'].to_numpy(), tau_prior=tau_prior, **mcmc)
            n_mice, n_sessions = survivors['subject'].nunique(), len(survivors)
            for component, samples in (('V_mouse', v_mouse),
                                       ('V_session', v_session)):
                mean, hdi_low, hdi_high, x_grid, density = summarize_posterior(
                    samples, grid_size=grid_size, hdi_prob=hdi_prob)
                summary_rows.append({
                    'target_NM': target_nm, 'event': event,
                    'regressor': regressor, 'component': component,
                    'mean': mean, 'hdi_low': hdi_low, 'hdi_high': hdi_high,
                    'n_mice': n_mice, 'n_sessions': n_sessions})
                violin_frames.append(pd.DataFrame({
                    'target_NM': target_nm, 'event': event,
                    'regressor': regressor, 'component': component,
                    'x': x_grid, 'density': density}))

        summary_df = pd.DataFrame(
            summary_rows, columns=RESPONSE_VARCOMP_SUMMARY_COLUMNS)
        violin_df = (pd.concat(violin_frames, ignore_index=True)
                     if violin_frames
                     else pd.DataFrame(columns=RESPONSE_VARCOMP_VIOLIN_COLUMNS))
        return summary_df, violin_df

    def load_performance(self) -> pd.DataFrame:
        """Read every catalogued session's `trials/performance` and join it on.

        Each session's product is read from its H5 in `self.h5_dir`; a session
        with no file, or no product in it, contributes nothing. Nothing here
        fetches, so the sessions are built without a ONE connection — this walks
        the whole catalog, and resolving a session path per row would dominate.
        This is the full per-session metric table; the two columns the filters
        read (`fraction_correct` and `contrasts`) reach `_catalog` through
        `complete_catalog`, not through here.

        Returns
        -------
        pandas.DataFrame
            One row per session that had the product, with an `eid` column and
            one column per metric. Also assigned to `self.performance`.
        """
        rows = []
        for _, row in self._catalog.iterrows():
            ps = PhotometrySession(row, one=None, load_data=False)
            ps.filepath = Path(self.h5_dir) / f'{ps.eid}.h5'
            if not ps.stored_product_exists('trials/performance'):
                continue
            rows.append({'eid': ps.eid} | ps.load_performance())
        self.performance = pd.DataFrame(rows)
        return self.performance

    # -----------------------------------------------------------------
    # Trace loading and extraction
    # -----------------------------------------------------------------


    def _code_lmm_predictors(
        self, df: pd.DataFrame, contrast_coding: str = 'log2'
    ) -> pd.DataFrame:
        """Code the trial frame for LMM fitting; do not mutate the input.

        Returns a copy with ``contrast`` transformed (``contrast_coding``),
        ``side`` / ``reward`` deviation-coded to ±0.5 (``side``: contra = +0.5,
        ipsi = −0.5; ``reward``: ``feedbackType`` 1 = +0.5, −1 = −0.5), and
        every ``config.CONTINUOUS_PREDICTORS`` column present mean-centered
        within ``df``. Coding a column a given formula does not use is
        harmless.

        Parameters
        ----------
        df : pd.DataFrame
            Trial-level frame with columns ``contrast``, ``side``, and
            ``feedbackType``.
        contrast_coding : str
            Coding passed to :func:`iblnm.util.get_contrast_coding`.
        """
        return analysis.code_predictors(df, contrast_coding)

    def response_lmm_fit(self, trials, formulas, group_by,
                         response_col='response', reml=True, re_formula='1',
                         min_subjects=2, events=None):
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
        trials : pandas.DataFrame
            The uncoded merged magnitude frame
            (``config.RESPONSE_MAGNITUDE_COLUMNS``), one row per recording x
            event x trial. :func:`iblnm.analysis.select_modeling_trials` runs
            on it here, so every fit shares one trial selection.
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
        df = analysis.select_modeling_trials(trials, response_col)
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

    def response_lmm_crossval(self, trials, formulas, group_by,
                              response_col='response', reference='full',
                              fold_col='subject', min_subjects=3, min_test=5,
                              min_trials=0, events=None):
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

        return self._response_lmm_resample(trials, procedure, formulas,
                                           group_by, response_col,
                                           min_trials=min_trials,
                                           events=events)

    def response_lmm_jackknife(self, trials, formulas, group_by,
                               response_col='response', reference='full',
                               fold_col='subject', min_subjects=3,
                               min_trials=0, events=None):
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

        return self._response_lmm_resample(trials, procedure, formulas,
                                           group_by, response_col,
                                           min_trials=min_trials,
                                           events=events)

    def _response_lmm_resample(self, trials, procedure, formulas, group_by,
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
        df = analysis.select_modeling_trials(trials, response_col)
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

    def response_lmm_effects(self, trials, name, kind, variables=None,
                             response_col='response'):
        """Extract a tidy effect frame from the cached fits of one named model.

        Reads the ``LMMResult``s cached by :meth:`response_lmm_fit` under
        ``(response_col, name, *group_values)``, computes the requested effect
        per group, and tags each row with the group identity (the ``group_by``
        columns from the originating fit call). Names no variable itself.

        Parameters
        ----------
        trials : pandas.DataFrame
            The uncoded merged magnitude frame
            (``config.RESPONSE_MAGNITUDE_COLUMNS``), one row per recording x
            event x trial. :func:`iblnm.analysis.select_modeling_trials` runs
            on it here, so every fit shares one trial selection.
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
        df = analysis.select_modeling_trials(trials, response_col)

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
            if not (hasattr(ps, 'photometry_responses') and hasattr(ps, 'trials')):
                h5_path = Path(self.h5_dir) / f'{eid}.h5'
                if not h5_path.exists():
                    print(f"  H5 file not found: {h5_path}")
                    continue
                ps.load_h5(h5_path, groups=['trials', 'photometry'])

            if brain_region not in getattr(ps, 'photometry_responses', {}):
                continue

            vec = ps.get_response_vector(
                brain_region=brain_region, hemisphere=hemisphere, **kwargs,
            )
            rows[(eid, target_nm, fiber_idx)] = vec

            # Discard raw data to free memory
            del ps.photometry_responses, ps.trials

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

    def get_persession_ols_features(self, trials, formula,
                                    event_name=STIM_ONSET_EVENT,
                                    weight_by_se=False,
                                    contrast_coding='log2',
                                    min_trials=MIN_TRIALS_PERSESSION):
        """Fit a caller-supplied response model per recording, return coefficients.

        Selects the modeling trials of ``trials``, restricts them to
        ``event_name``, then for each recording codes the predictors, drops
        complete-case rows over the formula's columns, and fits ``formula``
        through :func:`iblnm.analysis.fit_ols`.
        The fitted coefficients (or t-statistics) become that recording's feature
        vector. The formula is the caller's, per the layering rules; this method
        does no ``config.LMM_FORMULAS`` lookup.

        Parameters
        ----------
        trials : pandas.DataFrame
            The uncoded merged magnitude frame
            (``config.RESPONSE_MAGNITUDE_COLUMNS``), one row per recording x
            event x trial. :func:`iblnm.analysis.select_modeling_trials` runs
            on it here, so every fit shares one trial selection.
        formula : str
            Wilkinson formula template with a ``{response}`` placeholder, e.g.
            ``LMM_FORMULAS['persession']['full']``. Its coefficient names become
            the output columns.
        event_name : str
            Event to model (default ``config.STIM_ONSET_EVENT``).
        weight_by_se : bool
            If True, store t-statistics (``coef / SE``) instead of raw
            coefficients.
        contrast_coding : str
            Coding passed to :func:`iblnm.analysis.code_predictors`.
        min_trials : int
            A recording with fewer complete-case rows for the event is skipped.

        Returns
        -------
        pd.DataFrame
            ``(n_recordings, n_coefficients)`` indexed by
            ``(eid, target_NM, fiber_idx)``; columns are the fitted model's
            coefficient names (intercept included). Empty when no recording is
            scorable.
        """
        formula = formula.format(response='response')
        df = analysis.select_modeling_trials(trials, 'response')
        df = df[df['event'] == event_name]
        if 'fiber_idx' not in df.columns:
            df = df.assign(fiber_idx=0)

        group_keys = ['eid', 'target_NM', 'brain_region', 'fiber_idx']
        features = {}
        for keys, grp in df.groupby(group_keys):
            coded = analysis.code_predictors(grp, contrast_coding)
            coded = coded.dropna(
                subset=analysis.formula_union_columns([formula], coded.columns))
            if len(coded) < min_trials:
                continue
            fit = analysis.fit_ols(formula, coded)
            if fit is None:
                continue
            features[keys] = fit.tvalues if weight_by_se else fit.params

        if not features:
            self.persession_ols_features = pd.DataFrame()
            return self.persession_ols_features

        result = pd.DataFrame.from_dict(features, orient='index')
        result.index = pd.MultiIndex.from_tuples(result.index, names=group_keys)
        result = result.droplevel('brain_region')
        self.persession_ols_features = result
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

        Reads from ``self.performance`` if already loaded, otherwise from the
        performance parquet at ``performance_path``.

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
            self.performance = self.filter_to_recordings(pd.read_parquet(
                performance_path if performance_path is not None
                else PERFORMANCE_FPATH
            ))
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
                       unit_norm=True, feature_cols=None):
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
        feature_cols : list[str], optional
            Neural-feature columns to fit on. When given, ``X`` is subset to
            these columns after the intercept drop and index alignment (e.g. a
            single category block from :func:`iblnm.analysis.select_block_terms`).
            Default None fits on all columns.

        Returns
        -------
        dict[str, CCAResult]
        """
        from sklearn.preprocessing import StandardScaler
        from iblnm.analysis import fit_cca
        if sparse:
            from iblnm.analysis import fit_sparse_cca

        if self.persession_ols_features is None:
            raise ValueError("persession_ols_features is None")
        if self.psychometric_features is None:
            raise ValueError("psychometric_features is None")

        X = self.persession_ols_features.copy()
        Y = self.psychometric_features.copy()

        if exclude_intercept and 'intercept' in X.columns:
            X = X.drop(columns=['intercept'])

        # Align on shared index
        shared = X.index.intersection(Y.index)
        X = X.loc[shared]
        Y = Y.loc[shared]

        if feature_cols is not None:
            X = X[feature_cols]

        # Group by target_NM
        target_nms = X.index.get_level_values('target_NM')

        results = {}
        data = {}

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

    def response_anovaRM_fit(self, trials, response_col='response',
                             min_subjects=2, min_trials=10):
        """Run repeated-measures ANOVA on subject-mean response magnitudes.

        For each (target_NM, event) group, aggregates trial-level data to
        subject means by (contrast, side, feedbackType), then runs a 3-way
        repeated-measures ANOVA via ``anova_rm``.

        Parameters
        ----------
        trials : pandas.DataFrame
            The uncoded merged magnitude frame
            (``config.RESPONSE_MAGNITUDE_COLUMNS``), one row per recording x
            event x trial. :func:`iblnm.analysis.select_modeling_trials` runs
            on it here, so every fit shares one trial selection.
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

        df = analysis.select_modeling_trials(trials, response_col)

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


