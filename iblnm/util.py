import numpy as np
import pandas as pd
from pathlib import Path
import uuid
from datetime import datetime
from one.api import ONE
from one.alf.exceptions import ALFObjectNotFound
from brainbox.io.one import SessionLoader

from iblnm.config import *


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


def add_dataset_flags(df):
    """Add boolean columns for each dataset category."""
    df = df.copy()
    for category in DATASET_CATEGORIES:
        df[f'has_{category}'] = df.apply(
            lambda row: has_dataset_category(row, category), axis=1
        )
    return df


def _strip_hemisphere_suffix(region):
    """Extract base region name without hemisphere suffix (e.g., 'LC-r' -> 'LC')."""
    if pd.isna(region):
        return None
    parts = str(region).split('-')
    if len(parts) > 1 and parts[-1].lower() in ('l', 'r'):
        return parts[0]
    return region


def _extract_hemisphere_from_name(region):
    """Extract hemisphere from region name suffix (e.g., 'LC-r' -> 'R')."""
    if pd.isna(region):
        return None
    parts = str(region).split('-')
    if len(parts) > 1 and parts[-1].lower() in ('l', 'r'):
        return parts[-1].upper()
    return None


def _get_fiber_hemisphere_lookup(df_fibers=None):
    """Build subject+region -> hemisphere lookup from fiber coordinates."""
    if df_fibers is None:
        try:
            df_fibers = pd.read_csv(FIBERS_FPATH)
        except FileNotFoundError:
            return {}

    df_fibers = df_fibers.copy()
    df_fibers['hemi'] = df_fibers['X-ml_um'].apply(
        lambda x: 'L' if x > 0 else ('R' if x < 0 else None)
    )
    return df_fibers.groupby(['subject', 'targeted_region'])['hemi'].first().to_dict()


def process_regions(df, region_col='brain_region', df_fibers=None, add_hemisphere=True,
                    infer_nm=True, filter_valid=True):
    """
    Process brain region column: normalize names, add hemisphere, infer NM, create target_NM.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with region column (e.g., 'brain_region' or 'target').
    region_col : str
        Column containing region names (e.g., 'LC-r', 'VTA', 'DRN').
    df_fibers : pd.DataFrame, optional
        Fibers dataframe for hemisphere lookup. If None, loads from FIBERS_FPATH.
    add_hemisphere : bool
        Whether to add 'hemisphere' column.
    infer_nm : bool
        Whether to infer NM from region when NM='none'.
    filter_valid : bool
        Whether to filter to VALID_TARGETS.

    Returns
    -------
    pd.DataFrame
        Processed DataFrame with columns: region_base, target_NM, and optionally hemisphere.
    """
    df = df.copy()

    # Strip hemisphere suffix and normalize region names
    df['region_base'] = df[region_col].apply(_strip_hemisphere_suffix)
    df['region_base'] = df['region_base'].replace(REGION_NORMALIZE)

    # Add hemisphere column
    if add_hemisphere:
        df['hemisphere'] = df[region_col].apply(_extract_hemisphere_from_name)

        # Fill from fiber coordinates if subject column exists
        if 'subject' in df.columns:
            fiber_lookup = _get_fiber_hemisphere_lookup(df_fibers)
            if fiber_lookup:
                def get_fiber_hemi(row):
                    return fiber_lookup.get((row['subject'], row['region_base']))

                df['_hemi_fiber'] = df.apply(get_fiber_hemi, axis=1)

                # Warn on mismatches
                has_both = df['hemisphere'].notna() & df['_hemi_fiber'].notna()
                mismatches = df[has_both & (df['hemisphere'] != df['_hemi_fiber'])]
                if len(mismatches) > 0:
                    print(f"Warning: {len(mismatches)} hemisphere mismatches (name vs fiber)")

                # Fill missing from fiber
                df['hemisphere'] = df['hemisphere'].combine_first(df['_hemi_fiber'])
                df = df.drop(columns=['_hemi_fiber'])

    # Infer NM from region when missing
    if infer_nm and 'NM' in df.columns:
        df['NM'] = df['NM'].where(df['NM'] != 'none', df['region_base'].map(TARGET2NM))

    # Create target_NM column
    if 'NM' in df.columns:
        df['target_NM'] = df['region_base'] + '-' + df['NM']
    else:
        df['target_NM'] = df['region_base'].map(lambda r: f"{r}-{TARGET2NM.get(r, 'unknown')}")

    # Filter to valid targets
    if filter_valid:
        df = df[df['target_NM'].isin(VALID_TARGETS)].copy()

    return df


def add_target_nm(df):
    """Explode sessions by target and add target_NM column.

    DEPRECATED: Use process_regions() for new code.
    """
    df = df.explode('target').dropna(subset='target')
    return process_regions(df, region_col='target', add_hemisphere=False)


def add_hemisphere(df, region_col='brain_region', df_fibers=None, priority='name'):
    """
    Add hemisphere column from region name suffix or fiber coordinates.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with region column containing names like "LC-r", "VTA", "SNc-l".
    region_col : str
        Column containing region names with optional hemisphere suffix.
    df_fibers : pd.DataFrame, optional
        Fibers dataframe with 'subject', 'targeted_region', 'X-ml_um'.
        If None and FIBERS_FPATH exists, loads from file.
    priority : str, default 'name'
        Which source takes priority: 'name' (from suffix) or 'fiber' (from coords).
    """
    df = df.copy()

    # Extract hemisphere from region name suffix
    df['_hemi_name'] = df[region_col].apply(_extract_hemisphere_from_name)

    # Extract base region (strip -l/-r suffix)
    df['_region'] = df[region_col].apply(_strip_hemisphere_suffix)

    # Get hemisphere from fiber coordinates
    df['_hemi_fiber'] = None

    if 'subject' in df.columns:
        fiber_lookup = _get_fiber_hemisphere_lookup(df_fibers)
        if fiber_lookup:
            df['_hemi_fiber'] = df.apply(
                lambda row: fiber_lookup.get((row['subject'], row['_region'])), axis=1
            )

    # Check for mismatches and warn
    has_both = df['_hemi_name'].notna() & df['_hemi_fiber'].notna()
    mismatches = df[has_both & (df['_hemi_name'] != df['_hemi_fiber'])]
    if len(mismatches) > 0:
        print(f"Warning: {len(mismatches)} hemisphere mismatches (name vs fiber):")
        for _, row in mismatches.head(5).iterrows():
            subj = row.get('subject', '?')
            print(f"  {subj}/{row[region_col]}: name={row['_hemi_name']}, fiber={row['_hemi_fiber']}")
        if len(mismatches) > 5:
            print(f"  ... and {len(mismatches) - 5} more")

    # Fill hemisphere based on priority
    if priority == 'name':
        df['hemisphere'] = df['_hemi_name'].combine_first(df['_hemi_fiber'])
    else:
        df['hemisphere'] = df['_hemi_fiber'].combine_first(df['_hemi_name'])

    df = df.drop(columns=['_hemi_name', '_hemi_fiber', '_region'])
    return df


def drop_junk_duplicates(df, group_cols, verbose=True):
    """
    Keep one session per cell, preferring good over junk. Drop conflicts.
    """
    n_initial = len(df)
    n_conflicts = (df['session_status'] == 'conflict').sum()

    # Drop conflicts
    df = df[df['session_status'] != 'conflict'].copy()

    # Count groups with multiple session types (before deduplication)
    n_multi_type = 0
    for _, group in df.groupby(group_cols):
        if group['session_type'].nunique() > 1:
            n_multi_type += len(group) - 1  # all but one will be dropped

    # Keep one per group: prefer good over junk
    status_order = {'good': 0, 'junk': 1}
    df = (
        df.assign(_sort_key=df['session_status'].map(status_order))
        .sort_values('_sort_key')
        .drop_duplicates(subset=group_cols, keep='first')
        .drop(columns='_sort_key')
    )
    n_final = len(df)
    n_dropped = n_initial - n_final

    if verbose:
        print(f"Dropped duplicates: {n_initial} → {n_final} ({n_dropped} removed)")
        print(f"  Conflicts: {n_conflicts}")
        if n_multi_type > 0:
            print(f"  Different session types on same day: {n_multi_type}")
        n_junk = n_dropped - n_conflicts - n_multi_type
        if n_junk > 0:
            print(f"  Junk duplicates: {n_junk}")

    return df


def clean_sessions(df, exclude_subjects=None, exclude_session_types=None, verbose=True):
    """
    Remove excluded subjects and session types from sessions dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Sessions dataframe
    exclude_subjects : list, optional
        Subjects to exclude. Defaults to EXCLUDE_SUBJECTS from config.
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
        exclude_subjects = EXCLUDE_SUBJECTS
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


def protocol2type(protocol):
    # Define recognized session types
    session_types = np.array(SESSION_TYPES)
    # Define red flags (if found in filename it indicates a non-standard protocol)
    red_flags = PROTOCOL_RED_FLAGS
    # Determine which session types are found in the protocol name
    choiceworld_type_mask = [t + 'ChoiceWorld' in protocol for t in session_types[:-1]] + ['Histology' in protocol]
    type_mask = [t in protocol for t in session_types[:-1]] + ['Histology' in protocol]
    # Determine if any red flags are present in the protocol name
    red_flag_mask = [rf in protocol for rf in red_flags]
    # Decide what session type to return
    if (sum(choiceworld_type_mask or type_mask) == 1) and not any(red_flag_mask):  # only one session type and no red flags
        return str(session_types[type_mask][0])
    elif (sum(choiceworld_type_mask) == 0) or any(red_flag_mask):  # no/multiple session types or red flags
        return 'misc'
    else:
        raise ValueError


def _is_heterogeneous(col):
    """Check if column has mixed types (excluding None/NaN)."""
    # Get types of non-null values
    non_null = col.dropna()
    if len(non_null) == 0:
        return False
    types = non_null.apply(type).unique()
    # Allow int/float mixing (pandas handles this)
    numeric_types = {int, float, np.int64, np.int32, np.float64, np.float32}
    if all(t in numeric_types for t in types):
        return False
    return len(types) > 1


def _sanitize_for_parquet(df):
    """Convert heterogeneous columns to strings for Parquet compatibility."""
    df = df.copy()
    for col in df.columns:
        if _is_heterogeneous(df[col]):
            df[col] = df[col].apply(lambda x: str(x) if pd.notna(x) else x)
    return df


def _recover_column_dtype(col):
    """Try to recover original dtype from string column."""
    from one.alf.spec import QC

    # Skip non-string columns
    if col.dtype != 'object':
        return col

    # Skip columns that are actually lists/dicts
    non_null = col.dropna()
    if len(non_null) == 0:
        return col
    if isinstance(non_null.iloc[0], (list, dict)):
        return col

    # Try numeric conversion first
    try:
        numeric = pd.to_numeric(col, errors='raise')
        # Check if all values are whole numbers -> convert to Int64
        if numeric.dropna().apply(lambda x: x == int(x)).all():
            return numeric.astype('Int64')
        return numeric
    except (ValueError, TypeError):
        pass

    # Try QC enum conversion
    qc_names = {e.name for e in QC}
    non_null_str = non_null.astype(str)
    if non_null_str.isin(qc_names).all():
        return col.apply(lambda x: QC[x] if pd.notna(x) and x in qc_names else x)

    # Keep as string
    return col


def df2pqt(df, fpath, timestamp=None):
    """Save DataFrame to Parquet, converting heterogeneous columns to strings."""
    fpath = Path(fpath)
    df = _sanitize_for_parquet(df)
    if timestamp is not None:
        timestamp = datetime.now().strftime("%Y-%m-%d-%Hh%M")
        fpath = fpath.with_stem(f"{fpath.stem}_{timestamp}.pqt")
    df.to_parquet(fpath, index=False)


def pqt2df(fpath):
    """Load DataFrame from Parquet and recover original dtypes."""
    df = pd.read_parquet(fpath)
    for col in df.columns:
        df[col] = _recover_column_dtype(df[col])
    return df


def get_session_length(session):
    dt = np.nan
    try:
        t0 = datetime.fromisoformat(session['start_time'])
        t1 = datetime.fromisoformat(session['end_time'])
        if t0.date() == t1.date():
            dt = (t1 - t0).total_seconds()
        else:
            print(f"WARNING: {session['eid']} session start and end time on different days!")
    except:
        pass
    return dt



def resolve_session_status(session_group, columns):
    """
    Resolve sessions by flagging them as 'good', 'junk', or 'conflict'.

    Parameters
    ----------
    session_group : pd.DataFrame
        Group of sessions to evaluate
    columns : list of str
        List of boolean column names that must all be True for a session to be considered good

    Returns
    -------
    pd.Series
        Session status for each session in the group ('good', 'junk', or 'conflict')
    """
    # Create a copy to avoid modifying original data
    group = session_group.copy()

    # Initialize all sessions as 'junk'
    group['session_status'] = 'junk'

    # Find sessions where ALL specified columns are True
    good_sessions_mask = group[columns].all(axis=1)
    good_sessions = group[good_sessions_mask]

    if len(good_sessions) == 1:
        # One session has all requirements - mark as 'good', others remain 'junk'
        group.loc[good_sessions.index, 'session_status'] = 'good'

    elif len(good_sessions) > 1:
            group.loc[good_sessions.index, 'session_status'] = 'conflict'

    return group['session_status']


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
            f"Inconsistent lists in group"

        # Fill empty lists
        group[col] = group[col].apply(
            lambda x: first_list if isinstance(x, list) and len(x) == 0 else x
        )
        return group

    return df.groupby(group_col, group_keys=False).apply(fill_group)


def restrict_photometry_to_task(eid, photometry, one=None, buffer=2):
    assert eid is not None
    if one is None:
        one = ONE()
    loader = SessionLoader(one, eid=eid)
    ## FIXME: appropriately handle cases with multiple task collections
    loader.load_trials(collection='alf/task_00')
    timings = [col for col in loader.trials.columns if col.endswith('_times')]
    t0 = loader.trials[timings].min().min()
    t1 = loader.trials[timings].max().max()
    i0 = photometry.index.searchsorted(t0 - buffer)
    i1 = photometry.index.searchsorted(t1 + buffer)
    return photometry.iloc[i0:i1].copy()


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


# def load_kb_recinfo():
#     df = pd.read_csv('metadata/website.csv')
#     # Convert acronym strings into lists of strings
#     df['region'] = df['_acronyms'].apply(eval)
#     # Add additional metadata
#     df_insertions = pd.read_csv('metadata/insertions.csv')
#     def _merge_metadata(row, df=df_insertions):
#         subj = df_insertions[df_insertions['subject'] ==  row['subject']]
#         for col in [v for v in subj.columns if v != 'subject']:
#             row[col] = subj[col].values
#         return row
#     df = df.apply(_merge_metadata, df=df_insertions, axis='columns')
#     return df


def merge_session_metadata(
    df: pd.DataFrame,
    sessions_fpath: Path = None
) -> pd.DataFrame:
    """
    Merge a dataframe with session metadata.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'eid' column to merge on.
    sessions_fpath : Path, optional
        Path to sessions parquet file. Defaults to SESSIONS_FPATH.

    Returns
    -------
    pd.DataFrame
        Merged dataframe with session metadata.
    """
    if sessions_fpath is None:
        sessions_fpath = SESSIONS_FPATH

    df_sessions = pd.read_parquet(sessions_fpath)

    # Merge on eid
    df_merged = df.merge(
        df_sessions,
        on='eid',
        how='left',
        suffixes=('', '_session')
    )

    return df_merged


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


def build_filter_status(df_sessions: pd.DataFrame, qc_agg: pd.DataFrame = None) -> pd.DataFrame:
    """
    Build DataFrame with filter status for each session.

    Parameters
    ----------
    df_sessions : pd.DataFrame
        Sessions with columns: eid, subject, has_raw_task, has_raw_photometry,
        has_trials, has_photometry, trials_in_photometry_time
    qc_agg : pd.DataFrame, optional
        Output from aggregate_qc_per_session with columns: eid, passes_basic_qc

    Returns
    -------
    pd.DataFrame
        Filter status for each session with all boolean columns
    """
    cols = ['eid', 'subject', 'has_raw_task', 'has_raw_photometry',
            'has_trials', 'has_photometry', 'trials_in_photometry_time']
    result = df_sessions[cols].copy()

    if qc_agg is not None and len(qc_agg) > 0:
        result = result.merge(qc_agg[['eid', 'passes_basic_qc']], on='eid', how='left')
        result['passes_basic_qc'] = result['passes_basic_qc'].astype('boolean').fillna(False).astype(bool)
    else:
        result['passes_basic_qc'] = False

    return result


def merge_failure_logs(
    df_failures: pd.DataFrame,
    logs: list[tuple[str, pd.DataFrame]]
) -> pd.DataFrame:
    """
    Merge failure DataFrame with error logs from multiple sources.

    Parameters
    ----------
    df_failures : pd.DataFrame
        Sessions that failed some criterion, with at least 'eid' column
    logs : list of (source_name, df_log) tuples
        Each df_log should have columns: eid, exception_type, exception_message

    Returns
    -------
    pd.DataFrame
        Failures with error info columns added
    """
    result = df_failures.copy()

    # Initialize error columns
    result['exception_type'] = None
    result['exception_message'] = None
    result['source'] = None

    for source_name, df_log in logs:
        if df_log is None or len(df_log) == 0:
            continue

        # Get error info for sessions in this log
        log_cols = ['eid', 'exception_type', 'exception_message']
        available_cols = [c for c in log_cols if c in df_log.columns]
        log_subset = df_log[available_cols].copy()
        log_subset['source'] = source_name

        # Update rows that match
        for _, row in log_subset.iterrows():
            mask = result['eid'] == row['eid']
            if mask.any():
                result.loc[mask, 'exception_type'] = row.get('exception_type')
                result.loc[mask, 'exception_message'] = row.get('exception_message')
                result.loc[mask, 'source'] = row.get('source')

    return result
