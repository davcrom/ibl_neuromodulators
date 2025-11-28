import os
import gc
from functools import lru_cache
import numpy as np
import pandas as pd
from tqdm import tqdm

from one.api import ONE
from one.alf.spec import QC
from one.alf.exceptions import ALFObjectNotFound

from iblnm.config import *
from iblnm.util import protocol2type, fill_empty_lists_from_group

import iblphotometry


@lru_cache(maxsize=1)
def _get_default_connection():
    """
    Create and cache the default database connection. Cached connection allows
    repeated function calls without re-creating connection instance.
    """
    return ONE()


def save_as_pqt(df, fpath):
    # Map all columns with non-uniform data types to strings
    df = df.apply(lambda col: col if col.map(type).nunique() == 1 else col.astype(str))
    df.to_parquet(fpath, index=False)


def fetch_sessions(one, extended=True, save=True):
    """
    Query Alyx for sessions tagged in the neuromodulators project and add session
    info to a dataframe. Quality control metadata is unpacked, and a list of key datasets
    is checked.

    Parameters
    ----------
    one : one.api.OneAlyx
        Alyx database connection instance
    save : bool
        If true, the dataframe will be saved in ./metadata/sessions.csv

    Returns
    -------
    df_sessions : pandas.core.frame.DataFrame
        Dataframe containing detailed info for each session returned by the
        query
    """
    # Query for all sessions in the project with the specified task
    print("Querying database...")
    sessions = one.alyx.rest('sessions', 'list', project='ibl_fibrephotometry')
    df_sessions = pd.DataFrame(sessions).rename(columns={'id': 'eid'})
    df_sessions.drop(columns='projects')
    df_sessions['session_type'] = df_sessions['task_protocol'].map(protocol2type)
    print("Adding subject info...")
    df_sessions = df_sessions.progress_apply(_get_subject_info, one=one, axis='columns').copy()
    # Save initial query before adding more detialed info
    if save:
        save_as_pqt(df_sessions, SESSIONS_FPATH)
    if extended:  # Note: .copy() is applied to de-fragment the dataframe after repeated column additions
        print("Unpacking session dicts...")
        df_sessions = df_sessions.progress_apply(unpack_session_dict, one=one, axis='columns').copy()
        print("Checking datasets...")
        df_sessions = df_sessions.progress_apply(check_datasets, one=one, axis='columns').copy()
    # Get photometry ROIs and brain regions targeted
    df_recordings = pd.read_csv(RECORDINGS_FPATH).dropna()
    df_insertions = pd.read_csv(INSERTIONS_FPATH)
    df_sessions = df_sessions.progress_apply(
        _get_target_regions,
        one=one,
        df_recordings=df_recordings,
        df_insertions=df_insertions,
        axis='columns'
    ).copy()
    if save: save_as_pqt(df_sessions, SESSIONS_FPATH)
    return df_sessions


def unpack_session_dict(series, one=None):
    """
    Unpack useful metadata and extended QC from the session dict for a given eid.
    """
    if one is None:
        one = _get_default_connection()
    # Fetch full session dict
    session_dict = one.alyx.rest('sessions', 'read', id=series['eid'])
    for key in SESSIONDICT_KEYS:
        if key in session_dict.keys():
            series[key] = session_dict[key]
    series['qc_session'] = session_dict['qc']  # aggregate session QC value
    # Add QC vals to series
    if session_dict['extended_qc'] is not None:
        for key, val in session_dict['extended_qc'].items():
            # Clean up key names and add 'qc' tag
            if key.endswith('_qc'):  key = key.rstrip('_qc')
            if not key.startswith('_'): key = '_' + key
            key = 'qc' + key
            # Handle possible QC value types
            if type(val) == list:  # pull val out of list
                series[key.lstrip('qc_')] = val[1:]  # store values underlying QC outcome without the qc_ flag
                val = val[0]  # lists have QC outcome as first entry
            if type(val) == int:
                try:
                    series[key] = QC(val).name  # convert 0-100 values to string
                except ValueError:
                    series[key] = val
            elif type(val) == bool:
                series[key] = 'PASS' if val else 'FAIL'  # convert T/F to pass/fail
            elif (type(val) == float) | (type(val) == str):
                series[key] = val  # directly store strings & floats
            elif val is None:
                series[key] = 'NOT_SET'
            else:
                raise ValueError
    # Add list of session datasets
    if 'data_dataset_session_related' in session_dict.keys():
        series['datasets'] = [d['name'] for d in session_dict['data_dataset_session_related']]
    return series


def get_subject_info(session, one=None):
    """
    Get the mouse strain, line, and genotype for a given session. Try to infer
    the neuromodulatory cell type targeted using this info.

    Parameters
    ----------
    session : pd.Series
        A series representing an IBL session containing the mouse 'nickname' in
        the 'subject' column.

    one : one.api.OneAlyx
        Alyx database connection instance.

    Returns
    -------
    session : pd.Series
        The session series with new entries for mouse 'strain', 'line', and
        'genotype'.
    """
    if one is None:
        one = _get_default_connection()
    subjects = one.alyx.rest('subjects', 'list', nickname=session['subject'])
    assert len(subjects) == 1
    subject = subjects[0]
    for key in ['strain', 'line', 'genotype']:
        session[key] = subject[key]
    ## FIXME: this is horrendous...
    sNM = STRAIN2NM[session['strain']]
    lNM = LINE2NM[session['line']]
    if (sNM != 'none') and (lNM != 'none'):
        if sNM == lNM:
            NM = sNM
        else:
            NM = 'conflict'
    elif (sNM == 'none') and (lNM != 'none'):
        NM = lNM
    elif (sNM != 'none') and (lNM == 'none'):
        NM = sNM
    elif (sNM == 'none') and (lNM == 'none'):
        NM = 'none'
    else:
        raise ValueError
    session['NM'] = NM
    return session


def check_datasets(series, one=None):
    """
    Create a boolean entry for each important dataset for the given eid.
    """
    if one is None:
        one = _get_default_connection()
    # Fetch list of datasets listed under the given eid
    datasets = one.list_datasets(series['eid'])
    for dataset in ALYX_DATASETS:
        series[dataset] = dataset in datasets
    return series


def _get_target_regions(session, one=None, df_recordings=None, df_insertions=None):
    try:  # should work for all properly extracted data
        assert one is not None
        locations = one.load_dataset(id=session['eid'], dataset='photometryROI.locations.pqt')
        session['roi'] = locations.index.to_list()
        session['target'] = locations['brain_region'].to_list()
        session['remote_photometry'] = True
    except ALFObjectNotFound:  # default to using lookup tables
        assert df_recordings is not None
        assert df_insertions is not None
        eid = session['eid']
        rois = df_recordings.query('eid == @eid')
        session['roi'] = rois['region'].to_list()
        target = []
        for fiber in rois['fiber']:
            insertion = df_insertions.query('probename == @fiber')
            if len(insertion) < 1:
                print(f"Missing insertion entry for: {session['eid']}, {fiber}")
            else:
                assert len(insertion) == 1
                assert insertion['subject'].iloc[0] == session['subject']
                target.extend(insertion['targeted_regions'].to_list())
        session['target'] = target
        session['remote_photometry'] = False
    return session


def _get_ntrials_from_raw_taskData(session, one):
    """
    Get number of trials from the raw task data.
    """
    n_trials = np.nan
    try:
        raw_trials = one.load_dataset(session['eid'], dataset='_iblrig_taskData.raw.jsonable')
        if raw_trials is not None:
            n_trials = len(raw_trials)
            del raw_trials
            gc.collect()
    except ALFObjectNotFound:
        pass
    return n_trials


def load_photometry_data(session, one, extracted=True):
    photometry_data = {}
    if extracted:
        photometry = one.load_dataset(id=session['eid'], dataset='photometry.signal.pqt')
        locations = one.load_dataset(id=session['eid'], dataset='photometryROI.locations.pqt').reset_index()
        rois = locations['ROI'].to_list()
    else:
        if len(session['roi']) == 0:
            raise ValueError(f"No ROIs for {session['eid']}")
        raw_data_path = one.eid2path(session['eid']) / 'raw_photometry_data' / 'raw_photometry.csv'
        photometry = iblphotometry.io.from_raw_neurophotometrics_file_to_ibl_df(raw_data_path, version='old')
        photometry = photometry.drop(columns='index')
        rois = session['roi']
    return photometry[list(rois) + ['name']].set_index(photometry['times']).dropna()


def _check_local_datasets(series, one=None, local_cache=None):
    if one is None:
        assert local_cache is not None
        # Instantiate database connection
        one = ONE(cache_dir=local_cache)
    session_path = one.eid2path(series['eid'])
    if session_path is None:
        series['local_photometry'] = False
        return series
    pnames = [reg.name for reg in session_path.joinpath('alf').glob('Region*')]
    photometry_pqt_paths = []
    for pname in pnames:
        photometry_pqt_paths.append(session_path / 'alf' / pname / 'raw_photometry.pqt')
    if not photometry_pqt_paths:
        series['local_photometry'] = False
        return series
    series['local_photometry'] = all([os.path.isfile(pqt_path) for pqt_path in photometry_pqt_paths])
    return series
