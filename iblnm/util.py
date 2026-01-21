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


def add_target_nm(df):
    """Explode sessions by target and add target_NM column."""
    df = df.explode('target')
    df['target_NM'] = df['target'].str.split('-').str[0] + '-' + df['NM']
    df = df.query('target_NM in @VALID_TARGETS').copy()
    return df


def add_hemisphere(df_sessions, df_fibers=None):
    """
    Add hemisphere column to sessions based on fiber coordinates.

    Hemisphere determined from X-ml_um: x<0 is right, x>0 is left.
    If multiple fibers per subject+region, hemisphere is left blank.

    Parameters
    ----------
    df_sessions : pd.DataFrame
        Sessions dataframe with 'subject' and 'target' columns.
        Target values like "VTA", "NBM-l", "SNc-r" are matched by region.
    df_fibers : pd.DataFrame, optional
        Fibers dataframe with 'subject', 'targeted_region', 'X-ml_um'.
        If None, loads from FIBERS_FPATH.
    """
    df = df_sessions.copy()

    if df_fibers is None:
        try:
            df_fibers = pd.read_csv(FIBERS_FPATH)
        except FileNotFoundError:
            df['hemisphere'] = np.nan
            return df

    # Extract region from target (strip -l/-r suffixes: "NBM-l" -> "NBM")
    df['_region'] = df['target'].str.split('-').str[0]

    # Determine hemisphere from X coordinate
    df_fibers = df_fibers.copy()
    df_fibers['hemisphere'] = df_fibers['X-ml_um'].apply(
        lambda x: 'L' if x > 0 else ('R' if x < 0 else 'M')
    )

    # For each subject+region, keep hemisphere only if there's exactly one fiber
    def get_unique_hemisphere(group):
        if len(group) == 1:
            return group.iloc[0]['hemisphere']
        return np.nan

    fiber_lookup = (
        df_fibers.groupby(['subject', 'targeted_region'])
        .apply(get_unique_hemisphere, include_groups=False)
        .reset_index(name='hemisphere')
    )
    fiber_lookup = fiber_lookup.rename(columns={'targeted_region': '_region'})

    df = df.merge(fiber_lookup, on=['subject', '_region'], how='left')
    df = df.drop(columns=['_region'])
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


def df2pqt(df, fpath, timestamp=None):
    fpath = Path(fpath)
    # Map all columns with non-uniform data types to strings
    df = df.apply(
        lambda col: col if col.map(type).nunique() == 1 else col.astype(str)
        )
    if timestamp is not None:
        timestamp = datetime.now().strftime("%Y-%m-%d-%Hh%M")
        fpath = fpath.with_stem(f"{fpath.stem}_{timestamp}.pqt")
    df.to_parquet(fpath, index=False)


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
