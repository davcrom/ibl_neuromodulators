import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from one.api import ONE
from brainbox.io.one import SessionLoader

from iblnm.config import (
    VALID_TARGETNMS, DATASET_CATEGORIES, EXCLUDE_SESSION_TYPES,
    PROTOCOL_RED_FLAGS, SESSION_TYPES
)
from iblnm.validation import (
    exception_logger, make_log_entry,
    InvalidSessionType, InvalidTargetNM, InvalidSessionLength, TrueDuplicateSession,
)


LOG_COLUMNS = ['eid', 'error_type', 'error_message', 'traceback']


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


def collect_session_errors(df_sessions: pd.DataFrame, log_sources) -> pd.DataFrame:
    """Merge error logs onto df_sessions, adding a 'logged_errors' column.

    Parameters
    ----------
    df_sessions : pd.DataFrame
        Must contain an 'eid' column.
    log_sources : list of Path, str, or pd.DataFrame
        Paths to parquet error logs or pre-loaded DataFrames (schema: eid, error_type, ...).
        Missing files and empty DataFrames are silently ignored.
        Duplicate (eid, error_type, error_message) rows are removed before grouping.

    Returns
    -------
    pd.DataFrame
        df_sessions with a new 'logged_errors' column: list of error_type strings
        per session (empty list if no errors logged).
    """
    dfs = []
    for source in log_sources:
        if isinstance(source, pd.DataFrame):
            if len(source) > 0:
                dfs.append(source)
        else:
            p = Path(source)
            if p.exists():
                dfs.append(pd.read_parquet(p))

    if dfs:
        all_errors = deduplicate_log(pd.concat(dfs, ignore_index=True))
        errors_by_eid = (
            all_errors.groupby('eid')['error_type']
            .apply(list)
            .reset_index()
            .rename(columns={'error_type': 'logged_errors'})
        )
        df_sessions = df_sessions.merge(errors_by_eid, on='eid', how='left')
        df_sessions['logged_errors'] = df_sessions['logged_errors'].apply(
            lambda x: x if isinstance(x, list) else []
        )
    else:
        df_sessions = df_sessions.copy()
        df_sessions['logged_errors'] = [[] for _ in range(len(df_sessions))]

    return df_sessions


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
    t1 = datetime.fromisoformat(session['end_time'])
    if t0.date()!= t1.date():
        raise InvalidSessionLength(f"Session start and end time on different days")
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


def fill_empty_lists_from_group(df, col, group_col='subject'):
    df = df.copy()

    def fill_group(group):
        # Find non-empty lists
        non_empty = group[group[col].apply(lambda x: isinstance(x, list) and len(x) > 0)]

        if len(non_empty) == 0:
            return group  # No non-empty lists to use

        # Check consistency
        first_list = non_empty[col].iloc[0]
        assert all(x == first_list for x in non_empty[col]), \
            "Inconsistent lists in group"

        # Fill empty lists
        group[col] = group[col].apply(
            lambda x: first_list if isinstance(x, list) and len(x) == 0 else x
        )
        return group

    return df.groupby(group_col, group_keys=False).apply(fill_group)


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


def aggregate_qc_per_session(df_qc: pd.DataFrame, require_all: bool = True) -> pd.DataFrame:
    """
    Aggregate QC metrics per session and compute passes_basic_qc flag.

    Parameters
    ----------
    df_qc : pd.DataFrame
        QC results with columns: eid, n_unique_samples, n_band_inversions
    require_all : bool
        If True, all signals must pass for session to pass.
        If False, any signal passing is sufficient.

    Returns
    -------
    pd.DataFrame
        Columns: eid, passes_basic_qc
    """
    if len(df_qc) == 0:
        return pd.DataFrame(columns=['eid', 'passes_basic_qc'])

    if require_all:
        # All signals must pass: min unique > 0.1, max inversions == 0
        agg = df_qc.groupby('eid').agg({
            'n_unique_samples': 'min',
            'n_band_inversions': 'max',
        }).reset_index()
        agg['passes_basic_qc'] = (
            (agg['n_unique_samples'] > 0.1) &
            (agg['n_band_inversions'] == 0)
        )
    else:
        # Any signal passing is sufficient: max unique > 0.1, min inversions == 0
        agg = df_qc.groupby('eid').agg({
            'n_unique_samples': 'max',
            'n_band_inversions': 'min',
        }).reset_index()
        agg['passes_basic_qc'] = (
            (agg['n_unique_samples'] > 0.1) &
            (agg['n_band_inversions'] == 0)
        )

    return agg[['eid', 'passes_basic_qc']]


