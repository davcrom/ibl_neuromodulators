import numpy as np
import pandas as pd
import uuid
from one.api import ONE
from brainbox.io.one import SessionLoader

STRAIN2NM = {
    'Ai148xSERTCre': '5HT',
    'Ai148xDATCre': 'DA',
    'Ai148xDbhCre': 'NE',
    'Ai148xTHCre': 'NE',  ## TODO: double-check all THCre mice targeted LC-NE
    'B6.Cg': 'none',
    'Ai148xChATCre': 'ACh',
    'Ai148xDbh-Cre': 'NE',
    'B6.129S2': 'none',
    'Ai95xSERTCre': '5HT',
    'C57BL/6J': 'none',
    None: 'none'
}

LINE2NM = {
    'Ai148xSert': '5HT',
    'Ai148xDat': 'DA',
    'Ai148xDbh': 'NE',
    'Ai148-G6f-cdh x Chat-cre': 'ACh',
    'Ai148xTh': 'NE',
    'Ai148xChat': 'ACh',
    'Ai148xChAT': 'ACh',
    'C57BL/6J': 'none',
    'TetOG6s-Cdh23 x Camk-Cdh23': 'none',
    None: 'none'
}

TARGET2NM = {
    'VTA': 'DA',
    'SNc': 'DA',
    'DR': '5HT',
    'MR': '5HT',
    'LC': 'NE',
    'NBM': 'ACh',
    'SI': 'ACh',
    'PPT': 'ACh'
}

QCVAL2NUM = {  
    np.nan: 0.,
    'NOT SET': 0.01,
    'NOT_SET': 0.01,
    'PASS': 1.,
    'WARNING': 0.66,
    'CRITICAL': 0.33,
    'FAIL': 0.1
}

EVENT2COLOR = {
    'cue': 'blue', 
    'movement': 'orange', 
    'reward': 'green', 
    'omission':'red'
}


def protocol2type(protocol):
    types = np.array(['habituation', 'training', 'biased', 'ephys'])
    type_mask = [t + 'ChoiceWorld' in protocol for t in types]
    if sum(type_mask) == 1:
        return str(types[type_mask][0])
    elif sum(type_mask) == 0:
        return 'misc'
    else:
        raise ValueError


def fill_empty_lists_from_group(df, col, group_col='subject'):
    # Ensure column contains actual lists (not strings)
    df = df.copy()
    # Boolean mask for empty lists
    empty_mask = df[col].apply(lambda x: isinstance(x, list) and len(x) == 0)
    for idx in df[empty_mask].index:
        subject = df.at[idx, group_col]
        # Get all rows with same subject, skip empty lists
        group = df[(df[group_col] == subject) & (~empty_mask)]
        non_empty_values = group[col].tolist()
        # Remove rows that don't contain lists
        non_empty_values = [x for x in non_empty_values if isinstance(x, list)]
        if len(non_empty_values) == 0:
            continue  # no replacement possible
        # Assert all lists are equal
        first = non_empty_values[0]
        assert all(x == first for x in non_empty_values), f"Inconsistent non-empty lists for subject {subject}"
        # Replace empty list
        df.at[idx, col] = first
    return df


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
    

def _load_event_times(series, one=None):
    """
    Extracts reward_times, cue_times, and movement_times for a single row.

    Parameters
    ----------
    row : pd.Series
        A single row of the dataframe containing 'eid'.
    one : object
        The object used to load datasets like trials.

    Returns
    -------
    list
        A list containing reward_times, cue_times, and movement_times.
    """
    assert one is not None
    
    if isinstance(series, pd.Series):
        eid = series['eid']
    elif isinstance(series, uuid.UUID):
        eid = series
        series = pd.Series(data={'eid': str(eid)})
    else:
        raise TypeError('series mus be pd.Series of uuid.UUID')
    
    try:
        trials = one.load_dataset(eid, '*trials.table')
    except:
        print(f"WARNING: no trial data found for {eid}")
        return series
    
    series['cue_times'] = trials['goCue_times'].values
    series['movement_times'] = trials['firstMovement_times'].values
    series['reward_times'] = trials.query('feedbackType == 1')['feedback_times'].values
    series['omission_times'] = trials.query('feedbackType == -1')['feedback_times'].values
    
    return series


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
