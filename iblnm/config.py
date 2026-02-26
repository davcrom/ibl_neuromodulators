import sys
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from iblphotometry import processing

# Paths
PACKAGE_ROOT = Path(__file__).parent
PROJECT_ROOT = PACKAGE_ROOT.parent
SESSIONS_FPATH = PROJECT_ROOT / 'metadata/sessions.pqt'
SESSIONS_QC_FPATH = PROJECT_ROOT / 'metadata/sessions_qc.pqt'
INSERTIONS_FPATH = PROJECT_ROOT / 'metadata/insertions.csv'  # file with subject to brain region mapping
FIBERS_FPATH = PROJECT_ROOT / 'metadata/fibers.csv'
QCPHOTOMETRY_FPATH = PROJECT_ROOT / 'data/qc_photometry.pqt'
PERFORMANCE_FPATH = PROJECT_ROOT / 'data/performance.pqt'
EVENTS_FPATH = PROJECT_ROOT / 'data/events.pqt'
SESSIONS_H5_DIR = PROJECT_ROOT / 'data' / 'sessions'

# Per-script error logs (unified schema: eid, error_type, error_message, traceback)
QUERY_DATABASE_LOG_FPATH = PROJECT_ROOT / 'metadata/query_database_log.pqt'
PHOTOMETRY_LOG_FPATH = PROJECT_ROOT / 'metadata/photometry_log.pqt'
TASK_LOG_FPATH = PROJECT_ROOT / 'metadata/task_log.pqt'
WHEEL_LOG_FPATH = PROJECT_ROOT / 'metadata/wheel_log.pqt'
ERRORS_FPATH = PROJECT_ROOT / 'metadata/errors.pqt'


# Schema for sessions DataFrame: column -> (type, default)
# Used by enforce_schema() to fill missing columns and coerce NaN in list columns
SESSION_SCHEMA = {
    # From Alyx REST API
    'eid': (str, None),
    'subject': (str, None),
    'start_time': (str, None),
    'task_protocol': (str, None),
    'number': (int, None),
    'projects': (list, []),
    # From get_subject_info
    'strain': (str, None),
    'line': (str, None),
    'genotype': (str, None),
    'NM': (str, None),
    # From get_experiment_description
    'brain_region': (list, []),
    # From hemisphere extraction
    'hemisphere': (list, []),
    # From get_session_info
    'users': (list, []),
    'lab': (str, None),
    'end_time': (str, None),
    '_datasets_from_session_dict': (list, []),
    # From get_datasets
    'datasets': (list, []),
    # Convenience columns
    'session_type': (str, None),
    'target_NM': (list, []),
}


# FIXME: To be removed from the project
SUBJECTS_TO_EXCLUDE = [
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


# TODO: Check these names, try to homogenize
VALID_STRAINS = [
    'B6.Cg',
    'B6.129S2',
    'B6J.Cg-Gt',
    'B6.Cg-Igs7',
    'C57BL/6J',
    'B6.129(Cg)-Slc6a4',
    'B6.SJL-Slc6a3t',
    'B6;129S6-Chat',
    'B6.Cg-Dbh',
]

## FIXME: Old mappings, still useful until we can get NM from line/genotype
STRAIN2NM = {
    'Ai148xSERTCre': '5HT',
    'Ai148xDATCre': 'DA',
    'Ai148xDbhCre': 'NE',
    'Ai148xDbh-Cre': 'NE',  # non-standard format, should be Ai148xDbhCre
    'Ai148xTHCre': 'NE',  ## TODO: double-check all THCre mice targeted LC-NE
    'Ai148xChATCre': 'ACh',
    'Ai95xSERTCre': '5HT',
    # Wild-type strains (no NM) - rescued by line if available:
    # 'B6.Cg': None,
    # 'B6.129S2': None,
    # 'C57BL/6J': None,
}

VALID_LINES = [
    'Ai148xSert',
    'Ai148xDat',
    'Ai148xDbh',
    'Ai148xTh',
    'Ai148xChat',
    'Ai148cdhxChat',  # check this is correct
]

LINE2NM = {
    'Ai148xSert': '5HT',
    'Ai148xDat': 'DA',
    'Ai148xDbh': 'NE',
    'Ai148xTh': 'NE',
    'Ai148xChat': 'ACh',
    'Ai148-G6f-cdh x Chat-cre': 'ACh',  # non-standard format, should be Ai148xChat
}

VALID_NEUROMODULATORS = [
    'DA',
    '5HT',
    'NE',
    'ACh'
]


VALID_TARGETS = [
    'VTA',
    'SNc',
    'DR',
    'MR',
    'LC',
    'NBM',
    'SI',
    'PPT'
]

VALID_TARGETNMS = [
    'VTA-DA',
    'SNc-DA',
    'DR-5HT',
    'MR-5HT',
    'LC-NE',
    'NBM-ACh',
    'SI-ACh',
    'PPT-ACh'
]

# TEMPFIX: can be used to infer NM in case missing
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


# Dataset categories for checking data presence
DATASET_CATEGORIES = {
    'raw_task': [
        'raw_behavior_data/_iblrig_taskData.raw.jsonable',
        'raw_task_data_00/_iblrig_taskData.raw.jsonable',
    ],
    'raw_video': [
        'raw_video_data/_iblrig_leftCamera.raw.mp4',
    ],
    'raw_photometry_channels': [
        'raw_photometry_data/_neurophotometrics_fpData.channels.csv',
    ],
    'raw_photometry_signals': [
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


# Recognized session types
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

SESSION_TYPES_TO_ANALYZE = ('training', 'biased', 'ephys')
SESSION_TYPES_TO_EXCLUDE = ('advanced', 'neuromodulator', 'misc')

# Resampling
TARGET_FS = 30    # Hz, target sampling rate for photometry signals
WHEEL_FS = 100    # Hz, interpolation rate for wheel velocity

# Trial columns to store in HDF5 (beyond computed signed_contrast, contrast)
TRIAL_COLUMNS = [
    'stimOn_times', 'response_times',
    'firstMovement_times', 'feedback_times',
    'choice', 'feedbackType', 'probabilityLeft',
]

# Events for response extraction (NOT goCue — too close to stimOn, variable latency)
RESPONSE_EVENTS = ['stimOn_times', 'firstMovement_times', 'feedback_times']

# QC parameters
MIN_NTRIALS = 90
MIN_SESSIONLENGTH = 20 * 60  # seconds

# Task performance parameters
MIN_BLOCK_LENGTH = 10  # minimum trials per bias block (flag sessions with shorter blocks)
EVENT_TIMES = ['goCue_times', 'firstMovement_times', 'feedback_times']
EVENT_COMPLETENESS_THRESHOLD = 0.9

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


QCVAL2NUM = {
    np.nan: 0.,
    'nan': 0.,  # string 'nan' from parquet files
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
    # 'n_edges',
    'median_absolute_deviance',
    'percentile_distance',
    'percentile_asymmetry',
    # 'n_outliers',
    # 'n_expmax_violations',
    # 'expmax_violation',
    'ar_score'
]

QC_PREPROCESSING = [
    'bleaching_tau',
    'iso_correlation',
]

QC_METRICS_KWARGS = {
    'percentile_asymmetry': {'pc_comp': 75}
}

QC_SLIDING_KWARGS = {
    'w_len': 120,
    'step_len': 60,
    'detrend': True
}

PREPROCESSING_PIPELINES = {
    'isosbestic_correction': [
        dict(
            function=processing.lowpass_bleachcorrect,
            parameters=dict(
                correction_method='subtract-divide',
                N=3,
                Wn=0.01,
            ),
            inputs=('signal',),
            output='signal_bleach_corrected',
        ),
        dict(
            function=processing.lowpass_bleachcorrect,
            parameters=dict(
                correction_method='subtract-divide',
                N=3,
                Wn=0.01,
            ),
            inputs=('reference',),
            output='reference_bleach_corrected',
        ),
        dict(
            function=processing.isosbestic_correct,
            parameters=dict(
                regression_method='mse',
                correction_method='subtract',
            ),
            inputs=('signal_bleach_corrected', 'reference_bleach_corrected'),
            output='result',
        ),
        dict(
            function=processing.zscore,
            parameters=dict(mode='classic'),
            inputs=('result',),
            output='result',
        ),
    ]
}

# Fraction of unique samples per window below which a channel is flagged as suspect
N_UNIQUE_SAMPLES_THRESHOLD = 0.05

# Analysis parameters
# Event-based analyses
RESPONSE_WINDOW = (-1, 1)
BASELINE_WINDOW = (-0.1, 0)
RESPONSE_WINDOWS = {
    'early': (0.1, 0.35),
    'late': (0.35, 0.6)
}


# Plotting parameters
FIGURE_DPI = 150
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

# Create colormap for QC grid plots
QCCMAP = colors.LinearSegmentedColormap.from_list(
    'qc_cmap',
    [(0., 'white'), (0.01, 'gray'), (0.1, 'palevioletred'), (0.33, 'violet'), (0.66, 'orange'), (1., 'limegreen')],
    N=256
)

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

# Target-NM colors
TARGETNM_COLORS = {
    'MR-5HT': '#df67faff',
    'DR-5HT': '#b867faff',
    'VTA-DA': '#ff413dff',
    'SNc-DA': '#ff653dff',
    'LC-NE': '#3f88faff',
    'NBM-ACh': '#40afa1ff',
    'SI-ACh': '#40afa1ff',
    'PPT-ACh': '#00974eff',
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

TARGETNM_POSITIONS = {
    'MR-5HT': 0,
    'DR-5HT': 1,
    'VTA-DA': 2,
    'SNc-DA': 3,
    'LC-NE': 4,
    'NBM-ACh': 5,
    'SI-ACh': 6,
    'PPT-ACh': 7,
}




