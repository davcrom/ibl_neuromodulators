import sys
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

# Paths
PACKAGE_ROOT = Path(__file__).parent
PROJECT_ROOT = PACKAGE_ROOT.parent
SESSIONS_FPATH = PROJECT_ROOT / 'metadata/sessions.pqt'
SESSIONS_CLEAN_FPATH = PROJECT_ROOT / 'metadata/sessions_clean.pqt'
SESSIONS_QC_FPATH = PROJECT_ROOT / 'metadata/sessions_qc.pqt'
SESSIONS_LOG_FPATH = PROJECT_ROOT / 'metadata/sessions_log.pqt'
INSERTIONS_FPATH = PROJECT_ROOT / 'metadata/insertions.csv'  # file with subject to brain region mapping
FIBERS_FPATH = PROJECT_ROOT / 'metadata/fibers.csv'
QCPHOTOMETRY_FPATH = PROJECT_ROOT / 'data/qc_photometry.pqt'
QCPHOTOMETRY_LOG_FPATH = PROJECT_ROOT / 'data/qc_photometry_log.pqt'


# Values to extract from the session dict
SESSIONDICT_KEYS = ['users', 'lab', 'end_time', 'n_trials']

# Key datasets, note: need to check both old and new formats where applicable
ALYX_DATASETS = [
    'raw_behavior_data/_iblrig_taskData.raw.jsonable',  # old data, before v8
    'raw_behavior_data/_iblrig_taskSettings.raw.json',
    'raw_task_data_00/_iblrig_taskData.raw.jsonable',  # new data, >v8
    'raw_task_data_00/_iblrig_taskSettings.raw.json',
    'raw_video_data/_iblrig_leftCamera.raw.mp4',
    'alf/_ibl_trials.table.pqt',  # old format
    'alf/_ibl_wheel.position.npy',
    'alf/task_00/_ibl_trials.table.pqt',  # new format
    'alf/task_00/_ibl_wheel.position.npy',
    'alf/photometry/photometry.signal.pqt',
    'alf/photometry/photometryROI.locations.pqt',
    'raw_photometry_data/_neurophotometrics_fpData.channels.csv',
    'raw_photometry_data/_neurophotometrics_fpData.raw.pqt',
]


STRAIN2NM = {
    'Ai148xSERTCre': '5HT',
    'Ai148xDATCre': 'DA',
    'Ai148xDbhCre': 'NE',
    'Ai148xTHCre': 'NE',  ## TODO: double-check all THCre mice targeted LC-NE
    # ~ 'B6.Cg': 'none',
    'Ai148xChATCre': 'ACh',
    # ~ 'Ai148xDbh-Cre': 'NE',
    # ~ 'B6.129S2': 'none',
    'Ai95xSERTCre': '5HT',
    # ~ 'C57BL/6J': 'none',
    None: 'none'
}

LINE2NM = {
    'Ai148xSert': '5HT',
    'Ai148xDat': 'DA',
    'Ai148xDbh': 'NE',
    # ~ 'Ai148-G6f-cdh x Chat-cre': 'ACh',
    'Ai148xTh': 'NE',
    'Ai148xChat': 'ACh',
    # ~ 'Ai148xChAT': 'ACh',
    # ~ 'C57BL/6J': 'none',
    # ~ 'TetOG6s-Cdh23 x Camk-Cdh23': 'none',
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

SESSION_TYPES = [
    'habituation',
    'training',
    'advanced',
    'neuromodulator',
    'biased',
    'ephys',
    'passive',
    'histology'
]

# QC parameters
MIN_NTRIALS = 90
MIN_SESSIONLENGTH = 20 * 60  # seconds

# Dataset categories for checking data presence
# Each category is a list of datasets; if ANY is present, category is True
DATASET_CATEGORIES = {
    'raw_task': [
        'raw_behavior_data/_iblrig_taskData.raw.jsonable',
        'raw_task_data_00/_iblrig_taskData.raw.jsonable',
    ],
    'raw_video': [
        'raw_video_data/_iblrig_leftCamera.raw.mp4',
    ],
    'raw_photometry': [
        'raw_photometry_data/_neurophotometrics_fpData.channels.csv',
        'raw_photometry_data/_neurophotometrics_fpData.raw.pqt',
    ],
    'extracted_task': [
        'alf/_ibl_trials.table.pqt',
        'alf/task_00/_ibl_trials.table.pqt',
    ],
    'extracted_wheel': [
        'alf/_ibl_wheel.position.npy',
        'alf/task_00/_ibl_wheel.position.npy',
    ],
    'extracted_photometry_signal': [
        'alf/photometry/photometry.signal.pqt',
    ],
    'extracted_photometry_locations': [
        'alf/photometry/photometryROI.locations.pqt',
    ],
}

PROTOCOL_RED_FLAGS = [
    'RPE',
    'DELAY',
    'delay'
]

EXCLUDE_SESSION_TYPES = [
    'habituation',
    'advanced',
    'neuromodulator',
    'passive',
    'misc',
    'histology'
]

EXCLUDE_SUBJECTS = [
    'SP076',
    'SP075',
    'SP074',
    'SP073',
    'SP072',
    'SP066',
    'VIV-47627',
    'VIV-47615',
    'VIV-45598',
    'VIV-45585',
    'photometry_test_subject_A',
    'photometry_test_subject_B',
]

VALID_TARGETS = [
    'VTA-DA',
    'SNc-DA',
    'DR-5HT',
    'MR-5HT',
    'LC-NE',
    'NBM-ACh',
    'SI-ACh',
    'PPT-ACh'
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

EIDS_TO_DROP = [
    'cd9d071e-c798-4900-891f-b65640ec22b1',  # huge photometry artifact (DR)
    '16aa7570-578f-4daa-8244-844716fb1320',  # huge photometry artifact (DR)
    'f4f1d7fe-d7c8-442b-a7d6-e214223febaf',  # huge photometry artifact (VTA)
    'a60531cd-e1e8-4b3b-b4d9-94b76ccc69c2',  # huge photometry artifact (VTA)
    '1c09046e-48d8-47f3-9d07-2241e3f3a136',  # huge photometry artifact (DR)
]
# '4ac35324-a13c-4517-a61f-7183a2f6ff44'  # severe movement artifacts (LC)
# '46fe69ff-d001-4608-a15e-d5e029c14fc3'  # extreme photobleaching (SNc)
# '69544b1b-7788-4b41-8cad-2d56d5958526'  # extreme photobleaching (SNc)
# '26e1b376-61dd-4d64-b0ab-ac4e6b8b9385'  # extreme photobleaching (SNc)
# '99d32415-3e41-468c-a21e-17f30063eb31'  # massive transients (VTA)
# '3cafedfc-b78b-48ba-9bce-0402b71bbe90'  # piece-wise signal (DR)

# Photometry QC parameters
QC_RAW_METRICS = [
    'n_early_samples',
    'n_band_inversions',
]

QC_SLIDING_METRICS = [
    'n_unique_samples',
    'n_edges',
    'median_absolute_deviance',
    'percentile_distance',
    'percentile_asymmetry',
    'n_outliers',
    'n_expmax_violations',
    'expmax_violation',
    'ar_score'
]

QC_PREPROCESSING = [
    'photobleaching_tau',
]

QC_METRICS_KWARGS = {
    'percentile_asymmetry': {'pc_comp': 75}
}

QC_SLIDING_KWARGS = {
    'w_len': 120,
    'step_len': 60,
    'detrend': True
}

PREPROCESSING_PIPELINE = []

# Analysis parameters
# Event-based analyses
RESPONSE_WINDOW = (-1, 1)
BASELINE_WINDOW = (-0.1, 0)
RESPONSE_WINDOWS = {
    'early': (0.1, 0.35),
    'late': (0.35, 0.6)
}


# Plotting parameters
TICKFONTSIZE = 8
LABELFONTSIZE = 12 
plt.rcParams.update({
    'font.size': TICKFONTSIZE,
    'axes.labelsize': LABELFONTSIZE,
    'axes.titlesize': LABELFONTSIZE,
    'xtick.labelsize': TICKFONTSIZE,
    'ytick.labelsize': TICKFONTSIZE,
    'legend.fontsize': LABELFONTSIZE
})

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

EVENT2COLOR = {
    'cue': 'blue',
    'movement': 'orange',
    'reward': 'green',
    'omission':'red'
}

CONTRAST_CMAP = plt.get_cmap("inferno_r", 6)
CONTRAST_COLORS = {
    'contrast_0.0': CONTRAST_CMAP(1),  # skip first step, too light
    'contrast_0.0625': CONTRAST_CMAP(2),
    'contrast_0.125': CONTRAST_CMAP(3),
    'contrast_0.25': CONTRAST_CMAP(4),
    'contrast_1.0': CONTRAST_CMAP(5),
}

NM_CMAPS = {
    'DA': plt.colormaps['Reds'],
    '5HT': plt.colormaps['Purples'],
    'NE': plt.colormaps['Blues'],
    'ACh': plt.colormaps['Greens'],
}
NM_COLORS = {nm: cmap(0.8) for nm, cmap in NM_CMAPS.items()}

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
