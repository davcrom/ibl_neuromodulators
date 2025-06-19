import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from one.api import ONE
## TODO: use me!!
from one.alf.exceptions import ALFObjectNotFound

import sys
sys.path.append('/home/crombie/code/ibl_photometry/src')
import iblphotometry.io as io

from iblnm.util import STRAIN2NM, LINE2NM, TARGET2NM
from iblnm.util import protocol2type, fill_empty_lists_from_group 

LOCAL_CACHE = '/home/crombie/mnt/ccu-iblserver'
REGIONS_FPATH = 'metadata/regions.csv'  # file with eid2roi mapping for photometry
INSERTIONS_FPATH = 'metadata/insertions.csv'  # file with subject to brain region mapping
SESSIONS_FPATH = 'metadata/sessions.pqt'

ALYX_PHOTOMETRY_DATASETS = [
    'alf/photometry/photometry.signal.pqt',
    'alf/photometry/photometryROI.locations.pqt',
    'raw_photometry_data/_neurophotometrics_fpData.channels.csv',
    'raw_photometry_data/_neurophotometrics_fpData.raw.pqt',
]
EXTRACTED_PHOTOMETRY_DATASETS = [
    'alf/photometry/photometry.signal.pqt',
    'alf/photometry/photometryROI.locations.pqt'
]


def fetch_sessions(one, qc=False, check_local=True, save=True):
    """
    Query Alyx for sessions tagged in the psychedelics project and add session
    info to a dataframe. Sessions are restricted to those with the 
    passiveChoiceWorld task protocol, quality control metadata is unpacked, and
    a list of key datasets is checked. Sessions are sorted and labelled
    (session_n) by their order.

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
    # Add subject info to the dataframe
    print("Adding subject info...")
    # Note: .copy() is applied to de-fragment the dataframe after repeated column additions
    df_sessions = df_sessions.progress_apply(_get_subject_info, one=one, axis='columns').copy()
    strain_NM = df_sessions['strain'].replace(STRAIN2NM)
    if qc:
        # Unpack the extended qc from the session dict into dataframe columns
        print("Unpacking extended qc data...")
        df_sessions = df_sessions.progress_apply(_unpack_session_dict, one=one, axis='columns').copy()
    # Check if important datasets are present for the session
    print("Checking remote datasets...")
    df_sessions = df_sessions.progress_apply(_check_remote_datasets, one=one, axis='columns').copy()
    # All datasets must be present to set remote photometry to True
    df_sessions['remote_photometry'] = df_sessions.apply(lambda x: all([x[dset] for dset in EXTRACTED_PHOTOMETRY_DATASETS]), axis='columns')
    if check_local:
        print("Checking local datasets...")
        # Instantiate connection to local database
        df_sessions = df_sessions.progress_apply(
            _check_local_datasets, 
            one=ONE(cache_dir=LOCAL_CACHE),  # instantiate connection to local database
            axis='columns'
            ).copy()
        df_sessions['local_photometry'] = df_sessions['local_photometry'].replace({'nan': False}).astype(bool)
    # Get photometry ROIs and brain regions targeted
    df_regions = pd.read_csv(REGIONS_FPATH)
    df_insertions = pd.read_csv(INSERTIONS_FPATH)
    df_sessions = df_sessions.progress_apply(
        _get_target_regions, 
        one=one, 
        df_regions=df_regions, 
        df_insertions=df_insertions, 
        axis='columns'
    ).copy()
    df_sessions = fill_empty_lists_from_group(df_sessions, 'target', 'subject')
    # Guess missing NM values based on brain region targeted
    df_sessions['NM'] = df_sessions.apply(
        lambda x: TARGET2NM[x['target'][0]] 
        if (x['NM'] == 'none') and x['target'] 
        else x['NM'], axis='columns'
    )
    # Map all columns with non-uniform data types to strings
    df_sessions = df_sessions.apply(lambda col: col if col.map(type).nunique() == 1 else col.astype(str))
    # Label and sort by session number for each subject
    df_sessions['session_n'] = df_sessions.groupby('subject')['start_time'].rank(method='dense').astype(int)
    df_sessions = df_sessions.sort_values(by=['start_time', 'subject']).reset_index(drop=True)
    # Save as csv
    if save:
        df_sessions.to_parquet(SESSIONS_FPATH, index=False)
    return df_sessions


def _get_subject_info(series, one=None):
    assert one is not None
    subjects = one.alyx.rest('subjects', 'list', nickname=series['subject'])
    assert len(subjects) == 1
    subject = subjects[0]
    for key in ['strain', 'line', 'genotype']:
        series[key] = subject[key]
    ## FIXME: this is horrendous...
    sNM = STRAIN2NM[series['strain']]
    lNM = LINE2NM[series['line']]
    if (sNM != 'none') and (lNM != 'none'):
        assert sNM == lNM
        NM = sNM
    elif (sNM == 'none') and (lNM != 'none'): 
        NM = lNM
    elif (sNM != 'none') and (lNM == 'none'): 
        NM = sNM
    elif (sNM == 'none') and (lNM == 'none'): 
        NM = 'none'
    else:
        raise ValueError
    series['NM'] = NM
    return series


def _check_remote_datasets(series, one=None):
    """
    Create a boolean entry for each important dataset for the given eid.
    """
    assert one is not None
    # Fetch list of datasets listed under the given eid
    datasets = one.list_datasets(series['eid'])
    for dataset in ALYX_PHOTOMETRY_DATASETS:
        series[dataset] = dataset in datasets
    return series


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


def _unpack_session_dict(series, one=None):
    """
    Unpack the extended QC from the session dict for a given eid.
    """
    assert one is not None
    # Fetch full session dict
    session_dict = one.alyx.rest('sessions', 'read', id=series['eid'])
    series['session_qc'] = session_dict['qc']  # aggregate session QC value
    # Skip if there is no extended QC present
    if session_dict['extended_qc'] is None:
        return series
    # Add QC vals to series
    for key, val in session_dict['extended_qc'].items():
        # Add _qc flag to any keys that don't have it 
        if not key.endswith('_qc'): key += '_qc'
        if type(val) == list:  
            series[key.rstrip('_qc')] = val[1:]  # store underlying values
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
    return series


def _get_target_regions(session, one=None, df_regions=None, df_insertions=None):
    if session['remote_photometry']:
        assert one is not None
        locations = one.load_dataset(id=session['eid'], dataset='photometryROI.locations.pqt')
        session['roi'] = locations.index.to_list()
        session['target'] = locations['brain_region'].to_list()
    else:
        assert df_regions is not None
        assert df_insertions is not None
        eid = session['eid']
        rois = df_regions.query('eid == @eid')
        session['roi'] = rois['ROI'].to_list()
        subject = session['subject']
        insertions = df_insertions.query('subject == @subject')
        if len(insertions) > 1:
            if any(insertions['targeted_regions'].apply(lambda x: not isinstance(x, str))):
                raise ValueError
        session['target'] = insertions['targeted_regions'].to_list()
    return session


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
        photometry = io.from_raw_neurophotometrics_file_to_ibl_df(raw_data_path, version='old')
        photometry = photometry.drop(columns='index')
        rois = session['roi']
    return photometry[list(rois) + ['name']].set_index(photometry['times']).dropna()


# def _get_targets(session, df_insertions=None, one=None):
#     """
#     For the given session, load the brain regions targeted for each fiber/ ROI.

#     Parameters
#     ----------
#     session : pd.Series
#         Series object containing the eid and photometry ROI labels.
#     df_insertions : pd.DataFrame (optional)
#         Dataframe containing subject to brain region mappings, will be used if
#         mapping is not found in Alyx database

#     Returns
#     -------
#     session : pd.Series
#         Series object with a new entry for 'targeted_regions'
#     """
#     try:
#         assert one is not None
#         locations = one.load_dataset(id=session['eid'], dataset='photometryROI.locations.pqt')
#         session['targeted_regions'] = locations.loc[session['ROI']]['brain_region'].to_list()
#     except:
#         assert df_insertions is not None
#         subject = session['subject']
#         insertions = df_insertions.query('subject == @subject')
#         columns = ['fiber_diameter_um', 'fiber_length_mm', 'numerical_aperture', 'targeted_regions', 'expression']
#         for col in columns:
#             session[col] = insertions[col].dropna().tolist()
#     return session