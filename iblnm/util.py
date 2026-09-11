import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from iblnm.config import (
    VALID_TARGETNMS, DATASET_CATEGORIES,
    EXCLUDE_SESSION_TYPES, PROTOCOL_RED_FLAGS, SESSION_TYPES,
    SUBJECTS_TO_EXCLUDE,
    QC_VALUE_ORDER,
    log2_contrast, log2_contrast_inverse,
)
from iblnm.validation import (
    exception_logger,
    InvalidSessionType, InvalidTargetNM, InvalidSessionLength, TrueDuplicateSession,
)


LOG_COLUMNS = ['eid', 'error_type', 'error_message', 'traceback', 'product']


def fill_qc_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Label every unset QC outcome ``NOT_SET`` across the assembled QC table.

    Alyx omits a check entirely when it did not run, so the gap only becomes
    visible once every session is in one table and the column exists. A null
    QC outcome means the same thing as the IBL label ``NOT_SET``, but unlike
    the label it cannot be stored as an HDF5 attr and drops silently out of
    every category-based grouping.

    Outcome columns are identified by content, not by name: a ``qc_`` column
    carrying at least one ``QC_VALUE_ORDER`` label. That keeps the numeric
    ``qc_task_*`` columns — pass fractions, not outcomes — out of it, while
    still covering outcome columns polluted by stray numeric values.

    Parameters
    ----------
    df : pd.DataFrame
        QC table, one row per session, as assembled from ``get_extended_qc``.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with nulls in QC outcome columns replaced by
        ``'NOT_SET'``. A column with no labels at all is left untouched, since
        nothing distinguishes it from a numeric one.
    """
    label_cols = [col for col in df.columns
                  if col.startswith('qc_') and df[col].isin(QC_VALUE_ORDER).any()]
    return df.assign(**{col: df[col].fillna('NOT_SET') for col in label_cols})


def concat_logs(logs):
    """Concatenate log DataFrames, keeping only LOG_COLUMNS. No information loss."""
    dfs = [df for df in logs if df is not None and len(df) > 0]
    if not dfs:
        return pd.DataFrame(columns=LOG_COLUMNS)
    combined = pd.concat(dfs, ignore_index=True)
    return combined[[c for c in LOG_COLUMNS if c in combined.columns]].reindex(
        columns=LOG_COLUMNS
    )


def deduplicate_log(df):
    """Drop rows with duplicate (eid, error_type, error_message) from a log DataFrame."""
    if df is None:
        return None
    if len(df) == 0:
        return df
    return df.drop_duplicates(subset=['eid', 'error_type', 'error_message']).reset_index(drop=True)


def enforce_schema(df, schema):
    """Ensure DataFrame columns match a schema with correct types and defaults.

    Parameters
    ----------
    df : pd.DataFrame
    schema : dict
        Mapping of column_name -> (type, default_value).
        For list columns, NaN values are replaced with a copy of the default list.
        Missing columns are added with the default value.

    Returns
    -------
    pd.DataFrame
        Copy of df with schema enforced.
    """
    df = df.copy()
    for col, (dtype, default) in schema.items():
        if col not in df.columns:
            if isinstance(default, list):
                df[col] = [list(default) for _ in range(len(df))]
            else:
                df[col] = default
        elif isinstance(default, list):
            df[col] = df[col].apply(
                lambda x, d=default: list(x) if isinstance(x, (list, np.ndarray)) else list(d)
            )
    return df


# FIXME: try to remove, can be done inline
def has_dataset(session, dataset):
    """Check if a dataset is available for a session."""
    datasets = session.get('datasets', [])
    if isinstance(datasets, (list, np.ndarray)):
        return dataset in datasets
    return False


def has_dataset_category(session, category):
    """Check if a session has any dataset from a category."""
    category_datasets = DATASET_CATEGORIES.get(category, [])
    return any(has_dataset(session, d) for d in category_datasets)


## FIXME: try to remove, these checks can be done inline/ with has_dataset_category
def add_dataset_flags(df):
    """Add boolean columns for each dataset category."""
    df = df.copy()
    for category in DATASET_CATEGORIES:
        df[f'has_{category}'] = df.apply(
            lambda row: has_dataset_category(row, category), axis=1
        )
    return df


@exception_logger
def get_session_type(session):
    """Maps protocol names onto session types for convenience."""

    protocol = session['task_protocol']

    session_types = np.array(SESSION_TYPES)
    red_flags = np.array(PROTOCOL_RED_FLAGS)

    # Match against typeChoiceWorld pattern (strict)
    type_mask = np.array(
        [t + 'ChoiceWorld' in protocol for t in session_types[:-1]] + ['Histology' in protocol]
    )

    # FIXME: biasedChoiceWorld_ephyssessions currently pass as biased
    # verify these can be treated as normal biased sessions!!

    # FIXME: trainingPhaseChoiceWorld and passiveMockChoiceWorld are not
    # matched — check whether these are normal training/passive sessions

    # Check for red flags in protocol name
    red_flag_mask = np.array([rf in protocol for rf in red_flags])

    if (sum(type_mask) == 1) and not any(red_flag_mask):
        session['session_type'] = str(session_types[type_mask][0])
    elif sum(type_mask) == 0:
        raise InvalidSessionType(
            f"Protocol name {protocol} does not match any recognized session type."
        )
    elif any(red_flag_mask):
        raise InvalidSessionType(
            f"Protocol name {protocol} contains red flags {red_flags[red_flag_mask]}."
        )
    else:
        raise InvalidSessionType(
            f"Protocol name {protocol} matches multiple session types: "
            f"{session_types[type_mask]}."
        )

    return session


@exception_logger
def get_targetNM(session):
    NM = session['NM']
    target_NMs = [
        f"{region.rsplit('-', 1)[0] if region.endswith(('-l', '-r')) else region}-{NM}"
        if region else None
        for region in session['brain_region']
    ]
    # Always set target_NM so it stays parallel with brain_region/hemisphere,
    # even if validation below raises and exception_logger catches it.
    session['target_NM'] = target_NMs
    invalid = [t for t in target_NMs if t is not None and t not in VALID_TARGETNMS]
    if invalid:
        raise InvalidTargetNM(f"Target-NM {invalid} not recognized.")
    return session


@exception_logger
def get_session_length(session):
    t0 = datetime.fromisoformat(session['start_time'])
    end_time = session['end_time']
    if not isinstance(end_time, str):
        raise InvalidSessionLength("Missing end_time")
    t1 = datetime.fromisoformat(end_time)
    if t0.date() != t1.date():
        raise InvalidSessionLength("Session start and end time on different days")
    session['session_length'] = (t1 - t0).total_seconds()
    return session


DEFAULT_DISQUALIFYING_ERRORS = (
    'MissingRawData',
    'InsufficientTrials',
    'TrialsNotInPhotometryTime',
)


@exception_logger
def resolve_duplicate_group(group, disqualifying_errors=DEFAULT_DISQUALIFYING_ERRORS):
    """Return the single row to keep from a group of duplicate sessions.

    Designed for use with df.groupby(...).apply(resolve_duplicate_group, exlog=...).
    Raises TrueDuplicateSession when multiple sessions have no disqualifying errors;
    exception_logger catches this, logs one entry (first eid), and returns the first row.

    Parameters
    ----------
    group : pd.DataFrame
        One group of sessions sharing the same subject/day combination.
    disqualifying_errors : iterable of str
        Error types that mark a session as safely droppable.

    Returns
    -------
    pd.Series — the row to keep.
    """
    if len(group) == 1:
        return group.iloc[0]

    disqualifying_errors = set(disqualifying_errors)
    is_bad = group['logged_errors'].apply(
        lambda errs: any(e in disqualifying_errors for e in errs))
    good = group[~is_bad]

    if len(good) == 0:
        return group.iloc[0]
    if len(good) == 1:
        return good.iloc[0]

    raise TrueDuplicateSession(
        f"Multiple valid sessions: {list(good['eid'])}",
        fallback_row=good.iloc[0],
    )


def clean_sessions(df, exclude_subjects=None, exclude_session_types=None, verbose=True):
    """
    Remove excluded subjects and session types from sessions dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Sessions dataframe
    exclude_subjects : list, optional
        Subjects to exclude. Defaults to SUBJECTS_TO_EXCLUDE from config.
    exclude_session_types : list, optional
        Session types to exclude. Defaults to EXCLUDE_SESSION_TYPES from config.
    verbose : bool
        Print summary of removed data

    Returns
    -------
    pd.DataFrame
        Cleaned sessions dataframe
    """
    if exclude_subjects is None:
        exclude_subjects = SUBJECTS_TO_EXCLUDE
    if exclude_session_types is None:
        exclude_session_types = EXCLUDE_SESSION_TYPES

    n_initial = len(df)
    removed = {}

    # Remove excluded subjects
    mask_subjects = df['subject'].isin(exclude_subjects)
    if mask_subjects.any():
        removed['subjects'] = df.loc[mask_subjects, 'subject'].value_counts().to_dict()
        df = df[~mask_subjects].copy()

    # Remove excluded session types
    mask_types = df['session_type'].isin(exclude_session_types)
    if mask_types.any():
        removed['session_types'] = df.loc[mask_types, 'session_type'].value_counts().to_dict()
        df = df[~mask_types].copy()

    n_final = len(df)

    if verbose:
        print(f"Cleaned sessions: {n_initial} → {n_final} ({n_initial - n_final} removed)")
        if 'subjects' in removed:
            print(f"  Excluded subjects: {removed['subjects']}")
        if 'session_types' in removed:
            print(f"  Excluded session types: {removed['session_types']}")

    return df


def _is_heterogeneous(col):
    """Check if column has mixed types (excluding None/NaN)."""
    non_null = col.dropna()
    if len(non_null) == 0:
        return False
    types = non_null.apply(type).unique()
    numeric_types = {int, float, np.int64, np.int32, np.float64, np.float32}
    if all(t in numeric_types for t in types):
        return False
    return len(types) > 1


def _is_scalar_na(x):
    """Return True iff x is a scalar NA. Arrays/lists are never NA."""
    if isinstance(x, (list, np.ndarray, dict)):
        return False
    try:
        return bool(pd.isna(x))
    except (TypeError, ValueError):
        return False


def _sanitize_for_parquet(df):
    """Convert heterogeneous columns to strings for Parquet compatibility."""
    df = df.copy()
    for col in df.columns:
        if _is_heterogeneous(df[col]):
            df[col] = df[col].apply(lambda x: x if _is_scalar_na(x) else str(x))
    return df


def df2pqt(df, fpath, timestamp=None):
    """Save DataFrame to Parquet, converting heterogeneous columns to strings."""
    fpath = Path(fpath)
    df = _sanitize_for_parquet(df)
    if timestamp is not None:
        timestamp = datetime.now().strftime("%Y-%m-%d-%Hh%M")
        fpath = fpath.with_stem(f"{fpath.stem}_{timestamp}.pqt")
    df.to_parquet(fpath, index=False)


def fill_parallel_lists_from_group(df, columns, group_col='subject'):
    """Fill empty parallel list columns from a consistent source within each group.

    All specified columns are filled together from the same source row, so they
    stay in sync — a column is never filled on its own.

    A source row is valid only if all specified columns are non-empty and the
    same length. A group is consistent if all valid source rows are identical
    across the specified columns.

    A row is filled only where the assignment is forced, never where list
    order would decide it. Filling copies the source's entries onto the empty
    row by position, and position carries no identity of its own: a source
    recording two fibers names the two regions and sides the subject was
    implanted in, never which of them each of the empty row's own ROI channels
    is. Two cases leave no choice, and are the only ones filled:

    - every entry in the column is identical, so which position each goes to
      cannot matter (a subject with two fibers in one region, its sides
      unrecorded, fills its regions and leaves its hemispheres blank)
    - the row already carries one of the parallel columns, and it matches the
      source's exactly, so that column pins the order the rest are filled in
      (a row naming its two regions takes the source's hemispheres against
      them)

    Anything else is left empty. A session with no hemisphere still joins the
    analyses that do not read one, and is dropped by those that do.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with list columns.
    columns : list of str
        Column names that must stay in parallel.
    group_col : str
        Column to group by (default 'subject').

    Returns
    -------
    pd.DataFrame
        Copy with empty lists filled where a consistent source exists.

    Notes
    -----
    Iterates over each group's row indices and writes filled values back by
    label, rather than ``groupby(...).apply``. ``apply`` excludes the grouping
    column from the operation in pandas >=3, which silently dropped
    ``group_col`` from the result.
    """
    df = df.copy()

    def _is_nonempty_list(x):
        return isinstance(x, (list, np.ndarray)) and len(x) > 0

    def _is_empty_list(x):
        return isinstance(x, (list, np.ndarray)) and len(x) == 0

    def _is_populated(row):
        # Every column non-empty and the same length: a row that describes the
        # subject's implant, whatever the fiber count.
        return (all(_is_nonempty_list(row[col]) for col in columns)
                and len({len(row[col]) for col in columns}) == 1)

    def _needs_fill(row):
        return any(_is_empty_list(row[col]) for col in columns)

    for idx in df.groupby(group_col).groups.values():
        group = df.loc[idx, columns]
        # Consistency is judged over every populated row, single-fiber or not:
        # a subject whose sessions disagree about its implant has no source at
        # all, and a multi-fiber row disagreeing with a single-fiber one is
        # exactly that disagreement.
        populated = group[group.apply(_is_populated, axis=1)]
        if populated.empty:
            continue

        first = populated.iloc[0]
        consistent = all(
            np.array_equal(row[col], first[col])
            for _, row in populated.iloc[1:].iterrows()
            for col in columns
        )
        if not consistent:
            continue

        # Fill every column of any row that has at least one empty column,
        # where the source leaves no choice about which entry goes where.
        source_vals = {col: first[col] for col in columns}
        unordered = all(len(set(first[col])) == 1 for col in columns)
        needs_fill = group.apply(_needs_fill, axis=1)
        for i in group.index[needs_fill.to_numpy()]:
            pinned = any(np.array_equal(group.at[i, col], first[col])
                         for col in columns)
            if not (unordered or pinned):
                continue
            for col in columns:
                df.at[i, col] = source_vals[col]

    return df


def validate_parallel_lists(df, columns):
    """Drop rows where parallel list columns have mismatched lengths.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with list columns.
    columns : list of str
        Column names that must have matching lengths per row.

    Returns
    -------
    pd.DataFrame
        Copy with mismatched rows removed.
    """
    def _lengths_match(row):
        lengths = set()
        for col in columns:
            val = row[col]
            if isinstance(val, (list, np.ndarray)):
                lengths.add(len(val))
            else:
                lengths.add(1)
        return len(lengths) <= 1

    if df.empty:
        # An empty apply() returns a frame, and masking with it drops the
        # columns along with the (zero) rows.
        return df.copy()

    mask = df.apply(_lengths_match, axis=1)
    n_dropped = (~mask).sum()
    if n_dropped > 0:
        print(f"  Mismatched parallel lists: -{n_dropped}")
    return df[mask].copy()


def _agg_sliding_metric(series, metric=None, agg_func=np.mean, window=300):
    assert metric is not None
    if series[f'_{metric}_values'] is None:
        return np.nan
    t = series[f'_{metric}_times']
    t_mid = t.min() + (t.max() - t.min()) / 2
    i0, i1 = t.searchsorted([t_mid - window, t_mid + window]).clip(0, len(t) - 1)
    evs = series[f'_{metric}_values'][i0:i1]
    return agg_func(evs)


def _insert_event_times(session, trials):
    """
    Extracts reward_times, cue_times, and movement_times from a trials table
    and enters them as columns in a session Series.

    Parameters
    ----------
    session : pd.Series
    trials : pd.DataFrame

    Returns
    -------
    session : pd.Series
    """
    events = ['goCue_times', 'firstMovement_times', 'feedback_times']
    assert all([event in trials.columns for event in events])

    session['cue_times'] = trials['goCue_times'].values
    session['movement_times'] = trials['firstMovement_times'].values
    session['reward_times'] = trials.query('feedbackType == 1')['feedback_times'].values
    session['omission_times'] = trials.query('feedbackType == -1')['feedback_times'].values

    return session


def sample_recordings(df, metric, percentile_range):
    t0, t1 = np.nanpercentile(df[metric], percentile_range)
    samples = df[(df[metric] >= t0) & (df[metric] <= t1)]
    sample = samples.sample().squeeze()
    return sample


def traj2coord(x, y, z, depth, theta, phi, **kwargs):
    """
    Calculate tip coordinates from insertion trajectory in IBL-Allen coordinate
    system.

    Parameters
    ----------
    x, y, z : float
        Brain surface entry coordinates in µm, relative to bregma
    depth : float
        Insertion depth in µm, measured from brain surface
    theta : float
        Polar angle from vertical in degrees [0-180]
    phi : float
        Azimuth from right, anti-clockwise in degrees [0-360]
        phi=0 means tilted toward right, so tip moves in -ML direction

    Returns
    -------
    tip_ml, tip_ap, tip_dv : float
        Fiber tip coordinates in µm, relative to bregma
    """
    theta_rad = np.radians(theta)
    phi_rad = np.radians(phi)

    delta_ml = -depth * np.sin(theta_rad) * np.cos(phi_rad)
    delta_ap = -depth * np.sin(theta_rad) * np.sin(phi_rad)
    delta_dv = -depth * np.cos(theta_rad)

    tip_ml = x + delta_ml
    tip_ap = y + delta_ap
    tip_dv = z + delta_dv

    return np.array([tip_ml, tip_ap, tip_dv])


def fix_brain_regions(df):
    """Normalize brain_region naming errors from Alyx metadata.

    TEMPFIX: applies REGION_NAME_FIXES (e.g. DRN→DR, SNC→SNc).
    Remove once corrected upstream in Alyx.

    Parameters
    ----------
    df : pd.DataFrame
        Sessions with brain_region list column.

    Returns
    -------
    pd.DataFrame
        Copy with corrected brain_region names.
    """
    from iblnm.config import REGION_NAME_FIXES

    def _fix(regions):
        if not isinstance(regions, (list, np.ndarray)):
            return regions
        fixed = []
        for r in regions:
            bare = r.rsplit('-', 1)[0] if r.endswith(('-l', '-r')) else r
            suffix = r[len(bare):]
            fixed.append(REGION_NAME_FIXES.get(bare, bare) + suffix)
        return fixed

    df = df.copy()
    df['brain_region'] = df['brain_region'].apply(_fix)
    return df


def fill_brain_region_from_fibers(df, fibers_fpath=None):
    """Fill empty brain_region/hemisphere from fiber insertion table.

    For sessions where brain_region is still empty, look up the subject in
    fibers.csv and fill brain_region and hemisphere from fiber insertions.

    Parameters
    ----------
    df : pd.DataFrame
        Sessions with brain_region and hemisphere list columns.
    fibers_fpath : Path, optional
        Path to fibers.csv. Defaults to config.FIBERS_FPATH.

    Returns
    -------
    pd.DataFrame
        Copy with filled brain_region and hemisphere.
    """
    from iblnm.config import FIBERS_FPATH

    if fibers_fpath is None:
        fibers_fpath = FIBERS_FPATH

    df = df.copy()
    fibers = pd.read_csv(fibers_fpath)

    # Derive hemisphere from X-ml_um: >0 → 'l' (left), ≤0 → '' (midline/unknown)
    fibers['hemi'] = fibers['X-ml_um'].apply(
        lambda x: 'l' if x > 0 else ('r' if x < 0 else '')
    )

    # Build per-subject lookup: list of (targeted_region, hemisphere)
    subject_fibers = {}
    for subj, grp in fibers.groupby('subject'):
        pairs = list(zip(grp['targeted_region'], grp['hemi']))
        subject_fibers[subj] = pairs

    # Build per-(subject, region) hemisphere lookup for the fill pass.
    # Returns the hemisphere if unambiguous, '' if bilateral same-target or midline.
    hemi_lookup = {}
    for (subj, region), grp in fibers.groupby(['subject', 'targeted_region']):
        unique_hemis = grp['hemi'].unique()
        if len(unique_hemis) == 1 and unique_hemis[0] != '':
            hemi_lookup[(subj, region)] = unique_hemis[0]
        else:
            hemi_lookup[(subj, region)] = ''

    def _is_empty(x):
        return isinstance(x, (list, np.ndarray)) and len(x) == 0

    # Pass 1: fill brain_region and hemisphere for rows with empty brain_region
    for idx, row in df.iterrows():
        if not _is_empty(row['brain_region']):
            continue

        subj = row['subject']
        if subj not in subject_fibers:
            continue

        pairs = subject_fibers[subj]
        regions = [r for r, _ in pairs]
        hemis = [h for _, h in pairs]

        # Check if bilateral same-target (same region, both hemispheres)
        unique_regions = set(regions)
        if len(unique_regions) == 1 and len(regions) > 1:
            # Bilateral: we know the regions but can't map hemisphere to channel
            df.at[idx, 'brain_region'] = regions
            df.at[idx, 'hemisphere'] = [''] * len(regions)
        else:
            df.at[idx, 'brain_region'] = regions
            df.at[idx, 'hemisphere'] = hemis

    # Pass 2: fill missing hemisphere entries ('' or None) from fiber coordinates
    for idx, row in df.iterrows():
        hemis = row['hemisphere']
        if not isinstance(hemis, (list, np.ndarray)) or len(hemis) == 0:
            continue

        subj = row['subject']
        regions = row['brain_region']
        updated = False
        hemis = list(hemis)
        for i, (region, h) in enumerate(zip(regions, hemis)):
            if h not in ('', None):
                continue
            bare = region.rsplit('-', 1)[0] if region.endswith(('-l', '-r')) else region
            filled = hemi_lookup.get((subj, bare))
            if filled is not None:
                hemis[i] = filled
                updated = True
        # Normalize any remaining None to ''
        hemis = [h if h is not None else '' for h in hemis]
        if updated or None in row['hemisphere']:
            df.at[idx, 'hemisphere'] = hemis

    return df


def contrast_transform(c):
    """Map raw contrast to model scale: log(c + 1)."""
    return np.log(np.asarray(c, dtype=float) + 1)


def contrast_inverse(c_transformed):
    """Inverse of contrast_transform: exp(x) - 1."""
    return np.exp(np.asarray(c_transformed, dtype=float)) - 1


def get_contrast_coding(coding='log'):
    """Return (transform, inverse) functions for the given contrast coding.

    Parameters
    ----------
    coding : str
        One of 'log', 'log2', 'linear', or 'rank'.

    Returns
    -------
    transform : callable
        Maps raw contrast values to model scale.
    inverse : callable
        Maps model-scale values back to raw contrast.
    """
    if coding == 'log':
        return contrast_transform, contrast_inverse

    if coding == 'log2':
        return log2_contrast, log2_contrast_inverse

    if coding == 'linear':
        def _identity(c):
            return np.asarray(c, dtype=float)
        return _identity, _identity

    if coding == 'rank':
        _rank_map = {}

        def _rank_transform(c):
            c = np.asarray(c, dtype=float)
            scalar = c.ndim == 0
            c = np.atleast_1d(c)
            vals = sorted(set(float(v) for v in c) | set(_rank_map.keys()))
            if len(vals) > len(_rank_map):
                _rank_map.clear()
                _rank_map.update({v: float(i) for i, v in enumerate(vals)})
            result = np.array([_rank_map[float(v)] for v in c])
            return float(result[0]) if scalar else result

        _inv_map = {}

        def _rank_inverse(r):
            if not _inv_map and _rank_map:
                _inv_map.update({v: k for k, v in _rank_map.items()})
            r = np.asarray(r, dtype=float)
            scalar = r.ndim == 0
            r = np.atleast_1d(r)
            result = np.array([_inv_map[float(v)] for v in r])
            return float(result[0]) if scalar else result

        return _rank_transform, _rank_inverse

    raise ValueError(f"Unknown contrast coding: {coding!r}. "
                     f"Choose from 'log', 'log2', 'linear', 'rank'.")


def derive_target_nm(df, brain_region_col='brain_region'):
    """Derive target_NM and NM columns from brain_region.

    Uses config.TARGET2NM to map bare region names (without hemisphere suffix)
    to neuromodulator identity. Works on both list columns (sessions shape)
    and scalar columns (recordings shape).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``brain_region_col``.
    brain_region_col : str
        Column name containing brain region(s).

    Returns
    -------
    pd.DataFrame
        Copy with updated 'target_NM' and 'NM' columns.
    """
    from iblnm.config import TARGET2NM

    df = df.copy()

    def _target_nm_from_region(region):
        bare = region.rsplit('-', 1)[0] if region.endswith(('-l', '-r')) else region
        nm = TARGET2NM.get(bare)
        return f'{bare}-{nm}' if nm else None

    first_val = df[brain_region_col].iloc[0] if len(df) > 0 else None
    is_list_col = isinstance(first_val, (list, np.ndarray))

    if is_list_col:
        df['target_NM'] = df[brain_region_col].apply(
            lambda rs: [_target_nm_from_region(r) for r in rs]
            if isinstance(rs, (list, np.ndarray)) else rs
        )
        df['NM'] = df['target_NM'].apply(
            lambda ts: ts[0].split('-')[-1]
            if isinstance(ts, (list, np.ndarray)) and len(ts) > 0 and ts[0]
            else None
        )
    else:
        df['target_NM'] = df[brain_region_col].apply(_target_nm_from_region)
        df['NM'] = df['target_NM'].apply(
            lambda t: t.split('-')[-1] if t else None
        )

    return df


def fix_catalog(sessions: pd.DataFrame) -> pd.DataFrame:
    """Repair a catalog's brain regions and re-derive the columns off them.

    TEMPFIX in whole: every step here compensates for incomplete or wrong Alyx
    session metadata, and the function goes away once that metadata is
    corrected upstream. It cannot live in a per-session load path — filling one
    session's empty brain region from its subject's other sessions needs every
    session at once.

    Parameters
    ----------
    sessions : pd.DataFrame
        One row per session, as read back from the store's `metadata` groups.
        `brain_region`, `hemisphere` and `target_NM` are parallel list columns.

    Returns
    -------
    pd.DataFrame
        Copy carrying the fixed region names and the derived `target_NM`/`NM`,
        conformed to `config.SESSION_SCHEMA`. The schema is enforced on the way
        out because `derive_target_nm` adds columns.
    """
    from iblnm.config import SESSION_SCHEMA

    # `brain_region` and `hemisphere` fill in one call because filling them
    # separately broke their parallelism: subject
    # CQ011's sessions agreed on `hemisphere` but not on `brain_region`, so its
    # empty rows took a length-2 `hemisphere` beside a length-0 `brain_region`
    # and were dropped by `validate_parallel_lists`. `target_NM` is left out —
    # `derive_target_nm` has not run yet, so it still holds whatever the store's
    # metadata carried.
    catalog = fill_parallel_lists_from_group(
        sessions, ['brain_region', 'hemisphere'])
    catalog = fill_brain_region_from_fibers(catalog)
    catalog = fix_brain_regions(catalog)
    catalog = derive_target_nm(catalog)

    return enforce_schema(catalog, SESSION_SCHEMA)


# The predicates a `config.SESSION_GROUPS` criterion can name. Each takes one
# column and the criterion's value and returns a boolean Series. `superset`
# reads a cell holding a collection — `contrasts` is a list or an array
# depending on whether the catalog came from parquet or from the store — and a
# cell holding neither fails rather than raising, so a session with no stored
# contrasts is simply not in the group.
_CRITERIA_OPERATORS = {
    '==': lambda col, value: col == value,
    '>=': lambda col, value: col >= value,
    '<=': lambda col, value: col <= value,
    'in': lambda col, value: col.isin(value),
    'superset': lambda col, value: col.apply(
        lambda cell: set(cell) >= set(value)
        if isinstance(cell, (list, np.ndarray)) else False
    ),
}


def match_criteria(df: pd.DataFrame, criteria: dict) -> pd.Series:
    """Evaluate one group's criteria against a session table.

    Parameters
    ----------
    df : pd.DataFrame
        One row per session. Every column a criterion names must be present.
    criteria : dict
        ``{column: (operator, value)}``, operators as in
        `_CRITERIA_OPERATORS`. All entries are ANDed; an empty dict matches
        every row.

    Returns
    -------
    pd.Series
        Boolean, indexed like `df`.

    Raises
    ------
    KeyError
        If a criterion names a column `df` does not carry.
    """
    mask = pd.Series(True, index=df.index)
    for column, (operator, value) in criteria.items():
        mask &= _CRITERIA_OPERATORS[operator](df[column], value)
    return mask


def label_session_groups(df: pd.DataFrame, groups: dict) -> pd.DataFrame:
    """Label each session with the group it belongs to.

    Groups are evaluated in insertion order and each session takes the **last**
    group whose criteria it satisfies, so overlapping criteria still yield one
    label per session. That single-valuedness is what makes the stacked barplot
    segments sum to the session total.

    Parameters
    ----------
    df : pd.DataFrame
        One row per session, carrying every column the criteria name.
    groups : dict
        `config.SESSION_GROUPS`-shaped: ``{label: {'criteria': ..., 'color':
        ...}}``, ordered low to high. Only `criteria` is read here.

    Returns
    -------
    pd.DataFrame
        Copy carrying a `session_group` column. Sessions matching no group are
        labelled ``'other'``.
    """
    df = df.copy()
    df['session_group'] = 'other'
    for label, group in groups.items():
        df.loc[match_criteria(df, group['criteria']), 'session_group'] = label
    return df



