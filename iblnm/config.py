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
TRAJECTORIES_FPATH = PROJECT_ROOT / 'metadata/trajectories.json'
QCPHOTOMETRY_FPATH = PROJECT_ROOT / 'data/qc_photometry.pqt'
PERFORMANCE_FPATH = PROJECT_ROOT / 'data/performance.pqt'
RESULTS_DIR = PROJECT_ROOT / 'results'
RESPONSES_DIR = RESULTS_DIR / 'responses'
RESPONSES_FPATH = RESPONSES_DIR / 'responses.pqt'
TRIAL_REGRESSORS_FPATH = RESPONSES_DIR / 'trial_regressors.pqt'
RESPONSE_MATRIX_FPATH = RESPONSES_DIR / 'response_matrix.pqt'
RESPONSE_SIMILARITY_FPATH = RESPONSES_DIR / 'response_similarity_matrix.pqt'
MEAN_TRACES_FPATH = RESPONSES_DIR / 'mean_traces.pqt'
RESPONSE_OLS_PERSESSION_FPATH = RESPONSES_DIR / 'response_ols_persession_dropone.csv'
TASK_ENCODING_DIR = RESULTS_DIR / 'task_encoding'
SESSIONS_H5_DIR = PROJECT_ROOT / 'data' / 'sessions'

# Per-script error logs (unified schema: eid, error_type, error_message, traceback)
EVENTS_LOG_FPATH = PROJECT_ROOT / 'metadata/events_log.pqt'
ERRORS_FPATH = PROJECT_ROOT / 'metadata/errors.pqt'
POSE_FPATH = PROJECT_ROOT / 'metadata/pose.pqt'
POSE_LOG_FPATH = PROJECT_ROOT / 'metadata/pose_log.pqt'

# Video QC parameters
LENGTH_MISMATCH_THRESHOLD = 120  # seconds

# Video QC column names (leftCamera extended-QC fields), in source order.
# First 5 are quality metrics (scored), last 3 are problem flags.
VIDEO_QC_COLS = [
    'qc_videoLeft_focus',
    'qc_videoLeft_position',
    'qc_videoLeft_brightness',
    'qc_videoLeft_resolution',
    'qc_videoLeft_wheel_alignment',
    'qc_videoLeft_timestamps',
    'qc_videoLeft_dropped_frames',
    'qc_videoLeft_pin_state',
]
VIDEO_QC_QUALITY_COLS = VIDEO_QC_COLS[:5]  # quality metrics (scored)
VIDEO_QC_PROBLEM_COLS = VIDEO_QC_COLS[5:]  # problem flags (not scored)

# IBL QC outcome categories ordered most to least severe (for category ordering).
QC_VALUE_ORDER = ['NOT_SET', 'CRITICAL', 'FAIL', 'WARNING', 'PASS']

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


# TEMPFIX: normalize brain_region naming errors from Alyx metadata
# Remove once corrected upstream in Alyx
REGION_NAME_FIXES = {'DRN': 'DR', 'SNC': 'SNc'}

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


TARGETNMS_TO_ANALYZE = [
    'VTA-DA',
    'SNc-DA',
    'DR-5HT',
    # ~'MR-5HT',
    'LC-NE',
    'NBM-ACh',
    # ~'SI-ACh',
    # ~'PPT-ACh'
]

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
POSE_FS = 30      # Hz, common resample rate for pose movement traces (majority camera rate)

# Trial columns to store in HDF5 (beyond computed signed_contrast, contrast)
TRIAL_COLUMNS = [
    'stimOn_times', 'response_times',
    'firstMovement_times', 'feedback_times',
    'choice', 'feedbackType', 'probabilityLeft',
    'stim_side',
]

# Events for response extraction (NOT goCue — too close to stimOn, variable latency)
RESPONSE_EVENTS = ['stimOn_times', 'firstMovement_times', 'feedback_times']

# QC parameters
MIN_NTRIALS = 90
MIN_SESSIONLENGTH = 20 * 60  # seconds

# Error types that block a session from analysis
ANALYSIS_QC_BLOCKERS = {
    'MissingExtractedData', 'MissingRawData',
    'InsufficientTrials', 'IncompleteEventTimes',
    'TrialsNotInPhotometryTime', 'QCValidationError',
    'AmbiguousRegionMapping', 'MissingBlockInfo',
}

# Task performance parameters
MIN_TRAINING_PERFORMANCE = 0.70  # minimum fraction_correct for training sessions
MIN_PERFORMANCE = {'training': MIN_TRAINING_PERFORMANCE}
REQUIRED_CONTRASTS = frozenset({0, 6.25, 12.5, 25, 100})  # percent; must match biased/ephys
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
N_UNIQUE_SAMPLES_THRESHOLD = 0.1


# Analysis parameters
# Event-based analyses
RESPONSE_WINDOW = (-1, 1)
BASELINE_WINDOW = (-0.1, 0)
RESPONSE_WINDOWS = {
    'early': (0.1, 0.35),
    'late': (0.35, 0.6)
}

# Movement encoding analyses
TIMING_VARS = ['reaction_time', 'movement_time', 'peak_velocity']
# Predictor column each timing variable enters the LMM as. reaction_time and
# movement_time are heavily right-skewed (raw skew 7.7 / 14.0), so they enter
# log-transformed; peak_velocity is already roughly symmetric (raw skew 0.9) and
# enters raw. _modeling_frame supplies the matching log_<var> columns.
MOVEMENT_PREDICTORS = {
    'reaction_time': 'log_reaction_time',
    'movement_time': 'log_movement_time',
    'peak_velocity': 'peak_velocity',
}
MIN_SUBJECTS_MOVEMENT = 2
MIN_TRIALS_MOVEMENT = 20

# Pose QC (LightningPose output verification)
LIKELIHOOD_THRESHOLD = 0.9          # gate keypoint speed where confidence < this
MOVEMENT_RESPONSE_WINDOW = (0.1, 0.35)  # post-event scalar window (reuse BASELINE_WINDOW for pre)
CROSSCORR_LAG_WINDOW = 5.0          # paw/wheel cross-correlation lag half-width (s)
CROSSCORR_FS = WHEEL_FS             # common resample rate for paw/wheel cross-correlation (Hz)
LP_QC_LABELS = ('qc_lp', 'qc_movement', 'qc_timing')  # manual QC fields; IBL vocab, default 'NOT_SET'

# Bodypart trace label -> (event column, keypoints, reduction)
POSE_MEASURES = {
    'paw': ('firstMovement_times', ['paw_l', 'paw_r'], 'sum_speed'),
    'nose': ('stimOn_times', ['nose_tip'], 'speed'),
    'tongue_speed': ('feedback_times', ['tongue_end_l', 'tongue_end_r'], 'sum_speed'),
    'tongue_likelihood': ('feedback_times', ['tongue_end_l', 'tongue_end_r'], 'max_likelihood'),
}

# Event the motion_energy channel locks to (baseline is also stimOn-locked).
MOTION_ENERGY_EVENT = 'stimOn_times'


# LMM formula templates: the single source of every model formula. Each family
# maps a model name to a Wilkinson formula with `{response}` as the only
# placeholder (filled with the response column at fit time). In a nested
# comparison set, `full` is the reference model; every other key names the
# predictor whose unique contribution is `r2(full) - r2(<key>)`, and its formula
# is the full model with that predictor dropped. Movement families enumerate one
# concrete model per `TIMING_VARS` entry, the formula naming the real
# `log_<timing>` column (no timing placeholder at fit time).
LMM_FORMULAS = {
    # Task reliability: full interaction model is the reference. Each predictor's
    # ΔR² drops it and every interaction it participates in (so the two-way `*`
    # of the remaining pair); `interactions` drops the whole interaction block
    # (additive model), testing whether the coding is interactive at all. Keyed
    # by event because the reward outcome is only known at feedback, so stimOn
    # and firstMovement drop the reward predictor entirely (identical sets); the
    # feedback set keeps it.
    'task_reliability': {
        'stimOn_times': {
            'full': '{response} ~ contrast * side',
            'contrast': '{response} ~ side',
            'side': '{response} ~ contrast',
            'interactions': '{response} ~ contrast + side',
        },
        'firstMovement_times': {
            'full': '{response} ~ contrast * side',
            'contrast': '{response} ~ side',
            'side': '{response} ~ contrast',
            'interactions': '{response} ~ contrast + side',
        },
        'feedback_times': {
            'full': '{response} ~ contrast * side * reward',
            'contrast': '{response} ~ side * reward',
            'side': '{response} ~ contrast * reward',
            'reward': '{response} ~ contrast * side',
            'interactions': '{response} ~ contrast + side + reward',
        },
    },
    # Task ceiling: saturated single reporting model.
    'task_ceiling': {
        'ceiling': '{response} ~ C(contrast) * side * reward',
    },
    # Movement reliability: one consolidated family per timing variable,
    # extending task_reliability with the movement predictor. `full` is the
    # full interaction model; each main-effect key drops that predictor and all
    # its interactions; `movement` drops the timing predictor entirely;
    # `interactions` drops only the timing predictor's interaction terms (keeps
    # the task interactions, makes the predictor additive). The three-bar r2
    # plot reads the `full`/`contrast`/`movement` subset.
    **{
        f'movement_{t}': {
            'full': f'{{response}} ~ contrast * side * reward * {pred}',
            'contrast': f'{{response}} ~ side * reward * {pred}',
            'side': f'{{response}} ~ contrast * reward * {pred}',
            'reward': f'{{response}} ~ contrast * side * {pred}',
            'movement': '{response} ~ contrast * side * reward',
            'interactions': f'{{response}} ~ contrast * side * reward + {pred}',
        }
        for t, pred in MOVEMENT_PREDICTORS.items()
    },
    # Per-session OLS drop-one: full two-way interaction model is the reference.
    # Each non-`full` key drops one regressor and all of its 2-way interactions,
    # expressed as the `**2` of the remaining four regressors.
    'persession': {
        'full': '{response} ~ (contrast + side + reward + log_reaction_time + peak_velocity)**2',
        'contrast': '{response} ~ (side + reward + log_reaction_time + peak_velocity)**2',
        'side': '{response} ~ (contrast + reward + log_reaction_time + peak_velocity)**2',
        'reward': '{response} ~ (contrast + side + log_reaction_time + peak_velocity)**2',
        'log_reaction_time': '{response} ~ (contrast + side + reward + peak_velocity)**2',
        'peak_velocity': '{response} ~ (contrast + side + reward + log_reaction_time)**2',
    },
}

# Per-session OLS drop-one thresholds: minimum trials for a recording to be fit,
# and minimum recordings per mouse (per cell) for that mouse to be plotted.
MIN_TRIALS_PERSESSION = 50
MIN_RECORDINGS_PERMOUSE = 3


# Plotting parameters
FIGURE_DPI = 150
TICKFONTSIZE = 12
LABELFONTSIZE = 14
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

ANALYSIS_CONTRASTS = [0.0, 6.25, 12.5, 25.0, 100.0]

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




