import numpy as np
import sys

LOCAL_CACHE = '/home/crombie/mnt/ccu-iblserver'
# REGIONS_FPATH = 'metadata/regions.csv'  # file with eid2roi mapping for photometry
RECORDINGS_FPATH = 'metadata/recordings.csv'  # file with eid2roi mapping for photometry
INSERTIONS_FPATH = 'metadata/insertions.csv'  # file with subject to brain region mapping
SESSIONS_FPATH = 'metadata/sessions.pqt'

## FIXME: once ibl-photometry is a stand-alone package we won't need this 
IBLPHOTOMETRYPATH = '/home/davide/code/ibl-photometry/src'
sys.path.append(IBLPHOTOMETRYPATH)

# Values to extract from the session dict
SESSIONDICT_KEYS = ['users', 'lab', 'end_time', 'n_trials']

# Key datasets to check for
ALYX_PHOTOMETRY_DATASETS = [
    'raw_behavior_data/_iblrig_taskData.raw.jsonable',
    'raw_behavior_data/_iblrig_taskSettings.raw.json',
    'raw_task_data_00/_iblrig_taskData.raw.jsonable',  # new data has a different collection structure
    'raw_task_data_00/_iblrig_taskSettings.raw.json',
    'raw_video_data/_iblrig_leftCamera.raw.mp4',
    'alf/_ibl_trials.table.pqt',
    'alf/_ibl_wheel.position.npy',
    'alf/photometry/photometry.signal.pqt',
    'alf/photometry/photometryROI.locations.pqt',
    'raw_photometry_data/_neurophotometrics_fpData.channels.csv',
    'raw_photometry_data/_neurophotometrics_fpData.raw.pqt',
]
ALYX_PHOTOMETRY_DATASETS_NAMES = [
    '_iblrig_taskData.raw.jsonable',
    '_iblrig_taskSettings.raw.json',
    '_iblrig_leftCamera.raw.mp4',
    '_ibl_trials.table.pqt',
    '_ibl_wheel.position.npy',
    'photometry.signal.pqt',
    'photometryROI.locations.pqt',
    '_neurophotometrics_fpData.channels.csv',
    '_neurophotometrics_fpData.raw.pqt',
]

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

MIN_NTRIALS = 45
MIN_SESSIONLENGTH = 20 * 60  # seconds

EXCLUDE_SESSION_TYPES = [
    'habituation',
    'passive',
    'misc',
    'histology'
]

EXCLUDE_SUBJECTS = [
    'SP073',
    'SP072',
    'SP066',
    'VIV-47627',
    'VIV-47615', 
    'VIV-45598', 
    'VIV-45585'
]

QCVAL2NUM = {  
    np.nan: 0.,
    'NOT SET': 0.01,
    'NOT_SET': 0.01,
    'PASS': 1.,
    'WARNING': 0.66,
    'CRITICAL': 0.33,
    'FAIL': 0.1
}

SESSIONTYPE2FLOAT = {
    'habituation': 0.01,
    'training': 0.33,
    'biased': 0.66,
    'ephys': 0.99,
    'misc': 1.0
}

SESSIONTYPE2COLOR = {
    'habituation': 'darkgray',
    'training': 'cornflowerblue',
    'biased': 'mediumpurple', 
    'ephys': 'hotpink',
    'misc':  'sandybrown'
}

TARGETNM2POSITION = {
    'VTA-DA': 0,
    'SNc-DA': 1,
    'DR-5HT': 2,
    'MR-5HT': 3,
    'LC-NE': 4,
    'NBM-ACh': 5,
    'SI-ACh': 6,
    'PPT-ACh': 7
}

EVENT2COLOR = {
    'cue': 'blue', 
    'movement': 'orange', 
    'reward': 'green', 
    'omission':'red'
}