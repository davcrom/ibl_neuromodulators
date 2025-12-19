import numpy as np
import pandas as pd
from pathlib import Path
import uuid
from datetime import datetime
from one.api import ONE
from one.alf.exceptions import ALFObjectNotFound
from brainbox.io.one import SessionLoader

from iblnm.config import *


def protocol2type(protocol):
    ## FIXME: check that the biasCW_ephyssession protocol is handled properly (BCW but with a template session?)
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


def check_extracted_data(session):
    # Extraction path may change depending on iblrig version
    trials = any([
        session['alf/_ibl_trials.table.pqt'],
        session['alf/task_00/_ibl_trials.table.pqt']
    ])
    # All photometry data extracted to these files
    photometry = all([
        session['alf/photometry/photometry.signal.pqt'],
        session['alf/photometry/photometryROI.locations.pqt']
    ])
    return trials & photometry


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
