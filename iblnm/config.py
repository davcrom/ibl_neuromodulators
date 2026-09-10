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
TRIALS_DIR = PROJECT_ROOT / 'data/trials'  # one {subject}.csv per mouse
RESULTS_DIR = PROJECT_ROOT / 'results'
RESPONSES_DIR = RESULTS_DIR / 'responses'
RESPONSE_FIGURES_DIR = PROJECT_ROOT / 'figures/responses'
# `scripts/responses.py` runs one RESPONSES window per invocation and writes
# its tables and figures under a subdirectory of each of those two, named for
# the window. This path is the pre-window layout, kept for the scripts that
# read the magnitude table without choosing a window.
RESPONSE_MAGNITUDES_FPATH = RESPONSES_DIR / 'response_magnitudes.parquet'
# One row per recording x event x trial. `trial` is the trials table's own
# trial number, so a session's trial-level values are identifiable across its
# recordings and events.
# `masked_fraction` is the proportion of the measuring entry's own `window`
# that `mask_subsequent_events` removed before the mean was taken — the masking
# diagnostic's per-trial quantity, carried here because it describes exactly
# the samples `response` was averaged over.
# The trial-level columns follow the magnitude they were measured beside, so
# there is no second table and no join at read time: the grouping variables the
# response figures split on, the columns the trial selection reads, and the one
# stored regressor. Trial-level values repeat across a session's recordings and
# events. `signed_contrast` and `movement_time` are not carried — no persession
# formula and no plot reads either. `side` and `choice_side` are: they are the
# hemisphere-relative recodings of `stim_side` and `choice` that every model
# formula and every contrast figure groups on, and `task.add_relative_contrast`
# cannot re-derive them from this set because it also needs `signed_contrast`.
RESPONSE_MAGNITUDE_COLUMNS = ['eid', 'subject', 'session_type', 'NM',
                              'target_NM', 'brain_region', 'hemisphere',
                              'event', 'trial', 'response', 'masked_fraction',
                              'contrast', 'stim_side', 'side', 'feedbackType',
                              'choice', 'choice_side', 'response_time',
                              'reaction_time', 'probabilityLeft',
                              'peak_velocity']
# How much of the response window the event masking removed, one row per
# trial type within a cohort. Reported alongside every contrast-dependent
# result, because masking removes fast trials and does so more often at high
# contrast.
MASKING_DIAGNOSTIC_GROUP_COLS = ['target_NM', 'event', 'contrast',
                                 'feedbackType']
# The statistics, kept apart from the cell keys because the same reduction is
# also run at a coarser grain for the run's printed summary.
MASKING_DIAGNOSTIC_STATISTICS = ['n_trials', 'masked_fraction_mean',
                                 'pct_any_masked', 'pct_fully_masked',
                                 'pct_move_in_window']
MASKING_DIAGNOSTIC_COLUMNS = (MASKING_DIAGNOSTIC_GROUP_COLS
                              + MASKING_DIAGNOSTIC_STATISTICS)
RESPONSE_MATRIX_FPATH = RESPONSES_DIR / 'response_matrix.pqt'
RESPONSE_SIMILARITY_FPATH = RESPONSES_DIR / 'response_similarity_matrix.pqt'
# Per-recording OLS results at one grain: recording x event x dropped
# predictor. The drop-one fits, the reference model's coefficients, the
# permutation significance and the donor-pool size all key on that grain, so
# they are one frame. The reference-model quantities (n_trials, r2_full,
# r2_full_adj) repeat across a recording-event's predictor rows. `null` is the
# cell's whole permutation null vector, one float32 entry per scorable donor in
# an object-dtype column: parquet stores it as a list column, a reader naming
# other columns pays nothing for it, and holding it makes the per-mouse
# pooling recomputable without refitting. Its median — the old
# `delta_r2_null_median` — is derivable from it and so is no longer stored.
OLS_PERSESSION_COLUMNS = ['eid', 'subject', 'target_NM', 'brain_region',
                          'event', 'predictor', 'n_trials', 'r2_full',
                          'r2_full_adj', 'delta_r2', 'delta_r2_adj', 'null',
                          'coef', 'coef_se', 'p_value', 'q_value', 'n_donors']
# Per-mouse permutation significance — a coarser grain, so its own frame.
TASK_ENCODING_DIR = RESULTS_DIR / 'task_encoding'
DISPERSION_FIGURES_DIR = PROJECT_ROOT / 'figures/task_encoding/dispersion'
SESSIONS_H5_DIR = PROJECT_ROOT / 'data' / 'sessions'
DDM_HMM_DIR = PROJECT_ROOT / 'data' / 'ddm-hmm'
DDM_HMM_PARAMS_FPATH = DDM_HMM_DIR / 'all_mice_bestK_params.csv'
DDM_HMM_FIGURES_DIR = PROJECT_ROOT / 'figures' / 'ddm-hmm'

# Per-script error logs (unified schema: eid, error_type, error_message, traceback)
EVENTS_LOG_FPATH = PROJECT_ROOT / 'metadata/events_log.pqt'
ERRORS_FPATH = PROJECT_ROOT / 'metadata/errors.pqt'
POSE_FPATH = PROJECT_ROOT / 'metadata/pose.pqt'
LP_SESSIONS_FPATH = PROJECT_ROOT / 'metadata/LightningPoseSessions.csv'
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
    'test_mouse'
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

# The trials column every stimulus-onset alignment reads: response cuts,
# reaction times, the encoding and LMM event sets, the viewers. Named once here
# so no consumer hardcodes a column of its own.
#
# ONE's `stimOn_times` is the photodiode's report of the stimulus appearing,
# and on the mainenlab behavior rigs from 2025-06 onward that photodiode misses
# about half the screen flips. IBL's extractor takes the first photodiode pulse
# after the stimulus-on trigger with no time bound, so a missed onset silently
# yields the next pulse instead — a wheel-driven redraw, ~150 ms late. That
# happens on 39% of trials in the affected sessions, and 42% of the catalog's
# sessions carry it on more than 10% of their trials. Nothing marks those
# trials: the value is a real screen event, only the wrong one.
#
# `stimOnTrigger_times` is the Bpod state-machine clock instead, so it never
# sees the photodiode. It is present on every trial of every session and is
# early by the monitor's own latency, a constant that shifts every trial alike
# rather than smearing an average.
#
# Refinement worth making: add that latency back per session, as
# `stimOnTrigger_times` plus the median `stimOn_times - stimOnTrigger_times`
# over the session's trials whose onset the photodiode did catch. Checked
# against the photodiode on the 507,457 trials where it worked, that lands
# within 8.9 ms (SD; median residual 0.0 ms), and every session has enough
# caught trials to define its own median. The whole-catalog median latency is
# 59.8 ms, for scale.
STIM_ONSET_EVENT = 'stimOnTrigger_times'

# Events for response extraction (NOT goCue — too close to stimOn, variable latency)
RESPONSE_EVENTS = [STIM_ONSET_EVENT, 'feedback_times']

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

# How each sliding metric is reduced over windows to one value per recording.
# Strings, not callables, so the choice reads as data here and is looked up in
# `data._QC_AGGREGATORS`. 'q10' is the 10th percentile: n_unique_samples flags a
# channel by its worst windows, the rest describe the recording on average.
QC_SLIDING_AGG = {
    'n_unique_samples':         'q10',
    'median_absolute_deviance': 'mean',
    'percentile_distance':      'mean',
    'percentile_asymmetry':     'mean',
    'ar_score':                 'mean',
}

# Sliding metrics scored on un-detrended windows. Detrending subtracts a fitted
# line from each window, which leaves every sample a distinct float and pins
# n_unique_samples at 1.0 for any non-constant window. The remaining metrics
# measure variability, which a bleaching trend inflates, so they keep the
# detrending in QC_SLIDING_KWARGS.
QC_UNDETRENDED_METRICS = ('n_unique_samples',)

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
            function=processing.resample_signal,
            parameters=dict(fs=TARGET_FS, method='pchip'),
            inputs=('result',),
            output='result',
        ),
        # Last, so that what is stored is what the z-score was applied to.
        # Interpolating an already-z-scored signal attenuates its high
        # frequencies and leaves it short of unit variance.
        dict(
            function=processing.zscore,
            parameters=dict(mode='classic'),
            inputs=('result',),
            output='result',
        ),
    ]
}

# Recording-level photometry QC thresholds read by
# `PhotometrySessionGroup.filter_sessions(photometry_qc=...)`. Keys are columns
# of the `collect_qc` table — a metric of QC_SLIDING_METRICS with the band
# suffixed on, as `photometry/{region}/raw/qc` stores it — and values are the
# (comparison, cutoff) each must satisfy. A recording survives only if every
# entry passes, so both bands must clear their cutoff. At 30 Hz over a 120 s
# window (3600 samples), 0.005 is 18 distinct values. Provisional: it sits in
# the largest observed ratio gap (0.0012 to 0.0077, x6.3) on a 50-recording
# pool stratified over the old, detrended metric, so the gap is not evidence of
# population structure. Re-check once q10 exists for all 3776 recordings.
PHOTOMETRY_QC_THRESHOLDS = {
    'n_unique_samples_GCaMP':      ('>=', 0.005),
    'n_unique_samples_Isosbestic': ('>=', 0.005),
}


# Analysis parameters
# Event-based analyses
RESPONSE_WINDOW = (-1, 1)          # the stored peri-event cut
BASELINE_WINDOW = (-0.1, 0)        # pre-event baseline, subtracted per trial
RESPONSE_MAGNITUDE_WINDOW = (0.1, 0.35)  # averaged for a scalar magnitude

# One entry per analysis window, keyed by the analysis unit downstream. `event`
# is the trials-table column the cut is aligned to, `window` the interval
# measured off it, `masking_events` the following events past which samples are
# blanked, `ANOVA` the repeated-measures factors mapped to the levels kept ([]
# keeps all), and `min_trials` / `min_subjects` the cell occupancy a subject and
# a (target_NM, event) cell must reach.
#
# Masking is forward-only: samples are blanked from a following event onward,
# never before the entry's own event. A window opening before its event is
# therefore unmasked against the preceding trial -- `baseline` can contain the
# previous trial's feedback response, and nothing removes it.
# 100% contrast is dropped from every window's ANOVA: mice rarely err on easy
# trials, so crossing it with `feedbackType` leaves structurally empty cells
# and the repeated-measures fit loses every subject that misses one.
RESPONSES = {
    'baseline': {
        'event': STIM_ONSET_EVENT,
        'window': (-0.35, -0.1),
        'baseline_correct': False,
        'masking_events': [],
        'ANOVA': {'contrast': [0, 6.25, 12.5, 25], 'side': [],
                  'feedbackType': [], 'reaction_time_bin': []},
        'min_trials': 5,
        'min_subjects': 2,
    },
    'stimulus': {
        'event': STIM_ONSET_EVENT,
        'window': RESPONSE_MAGNITUDE_WINDOW,
        'baseline_correct': True,
        'masking_events': ['feedback_times'],
        'ANOVA': {'contrast': [0, 6.25, 12.5, 25], 'side': [],
                  'feedbackType': [], 'reaction_time_bin': []},
        'min_trials': 5,
        'min_subjects': 2,
    },
    'feedback': {
        'event': 'feedback_times',
        'window': RESPONSE_MAGNITUDE_WINDOW,
        'baseline_correct': True,
        'masking_events': [],
        'ANOVA': {'contrast': [0, 6.25, 12.5, 25], 'side': [],
                  'feedbackType': [], 'reaction_time_bin': []},
        'min_trials': 5,
        'min_subjects': 2,
    },
}


def validate_responses(responses: dict[str, dict]) -> None:
    """Check every RESPONSES entry against the stored cut it is measured off.

    Raises rather than logging: a bad edit here is a configuration error, not a
    per-session failure, so it must not be swallowed by `@exception_logger`.

    Parameters
    ----------
    responses : dict[str, dict]
        A RESPONSES-shaped table, keyed by analysis window.

    Raises
    ------
    ValueError
        Naming the offending entry and field, when its `event` is not a cut
        event, its `window` falls outside RESPONSE_WINDOW, a masking event
        precedes its own event, or `signed_contrast` is an ANOVA factor.
    """
    for key, entry in responses.items():
        if entry['event'] not in RESPONSE_EVENTS:
            raise ValueError(
                f"RESPONSES['{key}'] event {entry['event']!r} is not in "
                'RESPONSE_EVENTS')
        t0, t1 = entry['window']
        if not RESPONSE_WINDOW[0] <= t0 < t1 <= RESPONSE_WINDOW[1]:
            raise ValueError(
                f"RESPONSES['{key}'] window {entry['window']} is not an "
                f'increasing interval inside RESPONSE_WINDOW {RESPONSE_WINDOW}')
        for event in entry['masking_events']:
            if (event not in RESPONSE_EVENTS
                    or RESPONSE_EVENTS.index(event)
                    <= RESPONSE_EVENTS.index(entry['event'])):
                raise ValueError(
                    f"RESPONSES['{key}'] masking event {event!r} does not "
                    f"follow {entry['event']!r} in RESPONSE_EVENTS")
        if 'signed_contrast' in entry['ANOVA']:
            raise ValueError(
                f"RESPONSES['{key}'] ANOVA factor 'signed_contrast' collapses "
                'its signed-zero levels; use contrast and side')


validate_responses(RESPONSES)   # a bad edit to the table fails at import


# Predictor coding
def log2_contrast(contrast):
    """Code contrast as log2, clamping zero contrast to zero.

    Parameters
    ----------
    contrast : array_like
        Contrast in percent units. Nonzero values below 1 raise: a
        fraction-unit 100% (= 1.0) would land on the same 0 as a blank screen.

    Returns
    -------
    numpy.ndarray
        log2 of each nonzero value, 0.0 where the contrast is 0.
    """
    contrast = np.asarray(contrast, dtype=float)
    nonzero = contrast != 0
    if np.any(contrast[nonzero] < 1):
        raise ValueError(
            'log2 contrast coding expects contrast in percent units '
            '(nonzero values >= 1); got fractional input. A fraction-unit '
            '100% (=1.0) would collide with the 0->0 clamp.')
    return np.where(nonzero, np.log2(np.where(nonzero, contrast, 1)), 0.0)


def log2_contrast_inverse(coded):
    """Map log2-coded contrast back to percent units; 0 stays 0."""
    coded = np.asarray(coded, dtype=float)
    return np.where(coded != 0, 2 ** coded, 0.0)


def center(values):
    """Subtract the mean, ignoring and preserving NaNs.

    Returns the input type: a Series in, a Series out on its own index.
    """
    return values - np.nanmean(values)


# Input trials column -> (transform, model column it is written to). Each
# transform takes that one column and returns the coded values, so no entry
# needs the frame; a column absent from the frame is simply not coded.
# `contrast` is log2-coded (unconditionally: nothing fits another coding) and
# `side`, `choice_side` and `feedbackType` are deviation-coded to +/-0.5, which
# already puts them on their within-frame mean. reaction_time is heavily
# right-skewed (raw skew 7.7) so it enters log-transformed; peak_velocity is
# already roughly symmetric (raw skew 0.9) and enters raw.
# The continuous entries fold in their own mean-centering, over the frame
# `code_predictors` is handed (one recording-event for the per-session fits),
# so every main effect is read at that frame's own mean rather than at a raw
# zero no go trial reaches -- a 1 s reaction time (log10 = 0) or a motionless
# wheel. Centering after the log, never before it.
PREDICTOR_TRANSFORMS = {
    'contrast': (lambda s: center(log2_contrast(s)), 'contrast'),
    'side': (lambda s: np.where(s == 'contra', 0.5, -0.5), 'side'),
    'feedbackType': (lambda s: np.where(s == 1, 0.5, -0.5), 'reward'),
    'choice_side': (lambda s: np.where(s == 'contra', 0.5, -0.5), 'choice_side'),
    'reaction_time': (lambda s: center(np.log10(s.where(s > 0))),
                      'log_reaction_time'),
    'peak_velocity': (lambda s: center(s), 'peak_velocity'),
}

# Pose QC (LightningPose output verification)
LIKELIHOOD_THRESHOLD = 0.9          # gate keypoint speed where confidence < this
MOVEMENT_RESPONSE_WINDOW = (0.1, 0.35)  # post-event scalar window (reuse BASELINE_WINDOW for pre)
CROSSCORR_LAG_WINDOW = 5.0          # paw/wheel cross-correlation lag half-width (s)
CROSSCORR_FS = WHEEL_FS             # common resample rate for paw/wheel cross-correlation (Hz)
LP_QC_LABELS = ('qc_lp', 'qc_movement', 'qc_timing')  # manual QC fields; IBL vocab, default 'NOT_SET'
IBL_QC_VALUES = ('CRITICAL', 'FAIL', 'WARNING', 'PASS')  # settable verdicts ('NOT_SET' is the default, not a choice)

# Bodypart trace label -> (event column, keypoints, reduction)
POSE_MEASURES = {
    'paw': ('firstMovement_times', ['paw_l', 'paw_r'], 'sum_speed'),
    'nose': (STIM_ONSET_EVENT, ['nose_tip'], 'speed'),
    'tongue_speed': ('feedback_times', ['tongue_end_l', 'tongue_end_r'], 'sum_speed'),
    'tongue_likelihood': ('feedback_times', ['tongue_end_l', 'tongue_end_r'], 'max_likelihood'),
}

# Event the motion_energy channel locks to (baseline is also onset-locked).
MOTION_ENERGY_EVENT = STIM_ONSET_EVENT

# Every movement channel is extracted at every event in this union, so any
# (label, event) cell exists for a consumer; each channel's own response event
# is selected at read time via LABEL2EVENT.
MOVEMENT_EVENTS = sorted(
    {event for event, _, _ in POSE_MEASURES.values()} | {MOTION_ENERGY_EVENT})
LABEL2EVENT = ({label: event for label, (event, _, _) in POSE_MEASURES.items()}
               | {'motion_energy': MOTION_ENERGY_EVENT})


# Stored products
# ---------------
# A "product" is one stored result of the pipeline, named '{modality}/{product}'.
# The same key string names the `errors/` group a failed build is logged under
# and the group a load method checks for. Keys omit the label level: region and
# channel names are data, so 'photometry/raw/qc' names a kind of product whose
# H5 path is 'photometry/{region}/raw/qc'.

# Whether the raw products fetched from Alyx ('photometry/raw', 'wheel/raw' and
# video's three datasets) are kept in the session H5. Off by default: the store
# is already 14 GB of derived data and the ONE cache is where raw bytes belong.
# With it off no raw group is written, so the load methods find nothing stored
# and fetch; turning it on makes the files self-contained.
store_raw = False

# The two excitation bands acquired per region: the calcium-dependent GCaMP
# signal and the isosbestic control used to correct it.
PHOTOMETRY_BANDS = ('GCaMP', 'Isosbestic')

# The wheel matrix is cut from stimulus onset to each trial's own choice, so
# the window end names a trials column rather than a fixed offset in seconds.
# Cutting to the choice rather than to feedback delivery keeps peak_velocity —
# a maximum over the window — free of the outcome-dependent feedback lag.
WHEEL_RESPONSE_EVENTS = [STIM_ONSET_EVENT]
WHEEL_RESPONSE_WINDOW = (0.0, 'response_times')


# Single-session photometry encoding model (kernel-based ridge regression).
# The model grid, bases, ridge tuning, and default term spec read by
# scripts/encoding.py and passed down to the analysis.py builders/fit.
ENCODING_DT = 0.1                   # s, uniform model grid step
ENCODING_N_LAGS = 50               # FIR lags per event block (default basis)
ENCODING_N_BASIS = 10              # raised-cosine bumps (alternative basis)
ENCODING_RCOS_DURATION = 2.5       # s, raised-cosine basis span
ENCODING_RCOS_NLOFFSET = 0.2       # raised-cosine log-stretch offset
ENCODING_ALPHAS = np.logspace(-3, 3, 10)  # ridge alpha grid (1e-3–1e3)
ENCODING_CV = 5                    # contiguous K-fold count for alpha tuning / ΔR²
ENCODING_POSE_KEYPOINTS = ['paw_l', 'paw_r', 'nose']  # continuous pose regressors

# Default event term spec consumed by the modulator block builder and the script.
# Per event: `split_by` (categorical column splitting the event into separate
# kernel sets, or None), `modulators` (column -> 'continuous' [mean-centered] or
# 'categorical' [deviation-coded ±0.5 contra/ipsi]), `interactions` (modulator
# tuples coded as the product). Every event also emits its baseline kernel.
ENCODING_TERMS = {
    STIM_ONSET_EVENT: {
        'split_by': None,
        'modulators': {'side': 'categorical', 'contrast': 'continuous'},
        'interactions': [('side', 'contrast')],
    },
    'firstMovement_times': {
        'split_by': None,
        'modulators': {'choice': 'categorical'},
        'interactions': [],
    },
    'response_times': {
        'split_by': None,
        'modulators': {'choice': 'categorical'},
        'interactions': [],
    },
    'feedback_times': {
        'split_by': 'feedbackType',
        'modulators': {'contrast': 'continuous'},
        'interactions': [],
    },
    'goCue_times': {
        'split_by': None,
        'modulators': {},
        'interactions': [],
    },
}


# Per-recording OLS drop-one regressors: the six main effects the response model
# below is written over, and the figure row order the main-effect drops take.
PERSESSION_REGRESSORS = ['contrast', 'side', 'reward', 'choice_side',
                         'log_reaction_time', 'peak_velocity']


# The per-session OLS response model: one Wilkinson formula with `{response}` as
# the only placeholder, filled with the response column at fit time. It carries
# the six PERSESSION_REGRESSORS mains and every two-way except side:reward,
# choice_side:side and choice_side:reward — choice_side enters explicitly so its
# own interactions are visible, but it is collinear with the side and reward
# mains (choice_side ≈ 2·side·reward), and side:reward itself encodes choice.
RESPONSE_MODEL_FORMULA = (
    '{response} ~ contrast + side + reward + choice_side + log_reaction_time'
    ' + peak_velocity + contrast:side + contrast:reward'
    ' + contrast:choice_side + contrast:log_reaction_time'
    ' + contrast:peak_velocity + side:log_reaction_time'
    ' + side:peak_velocity + reward:log_reaction_time'
    ' + reward:peak_velocity + choice_side:log_reaction_time'
    ' + choice_side:peak_velocity + log_reaction_time:peak_velocity')

# Drop-one labels: each maps to the RESPONSE_MODEL_FORMULA terms removed to build
# that label's reduced model, whose ΔR² against the full model is the label's
# unique contribution. The label is the unit of identity downstream — the
# `predictor` value in the output tables, the figure filename and the plot label
# — and nothing derives it from its term list, so a label may name any set of
# terms. Written out rather than generated for that reason. A main-effect label
# drops itself and every two-way it enters; an interaction label drops one term,
# leaving its constituent mains standing.
RESPONSE_DROPPED_TERMS = {
    'contrast': ['contrast', 'contrast:side', 'contrast:reward',
                 'contrast:choice_side', 'contrast:log_reaction_time',
                 'contrast:peak_velocity'],
    'side': ['side', 'contrast:side', 'side:log_reaction_time',
             'side:peak_velocity'],
    'reward': ['reward', 'contrast:reward', 'reward:log_reaction_time',
               'reward:peak_velocity'],
    'choice_side': ['choice_side', 'contrast:choice_side',
                    'choice_side:log_reaction_time',
                    'choice_side:peak_velocity'],
    'log_reaction_time': ['log_reaction_time', 'contrast:log_reaction_time',
                          'side:log_reaction_time', 'reward:log_reaction_time',
                          'choice_side:log_reaction_time',
                          'log_reaction_time:peak_velocity'],
    'peak_velocity': ['peak_velocity', 'contrast:peak_velocity',
                      'side:peak_velocity', 'reward:peak_velocity',
                      'choice_side:peak_velocity',
                      'log_reaction_time:peak_velocity'],
    'contrast:side': ['contrast:side'],
    'contrast:reward': ['contrast:reward'],
    'contrast:choice_side': ['contrast:choice_side'],
    'contrast:log_reaction_time': ['contrast:log_reaction_time'],
    'contrast:peak_velocity': ['contrast:peak_velocity'],
    'side:log_reaction_time': ['side:log_reaction_time'],
    'side:peak_velocity': ['side:peak_velocity'],
    'reward:log_reaction_time': ['reward:log_reaction_time'],
    'reward:peak_velocity': ['reward:peak_velocity'],
    'choice_side:log_reaction_time': ['choice_side:log_reaction_time'],
    'choice_side:peak_velocity': ['choice_side:peak_velocity'],
    'log_reaction_time:peak_velocity': ['log_reaction_time:peak_velocity'],
}


def formula_terms(formula: str) -> list[str]:
    """Right-hand-side terms of a Wilkinson formula, in the order written.

    The bottom-layer parser both the drop-one validation here and the reduced
    formulas built at fit time read, so a term is split the same way wherever it
    is named. Interaction terms keep their ``':'`` form, which is how patsy
    names the design column they build.
    """
    return [term.strip() for term in formula.split('~')[1].split('+')]


def validate_dropped_terms(formula: str,
                           dropped_terms: dict[str, list[str]]) -> None:
    """Check a drop-one term table against the model it drops terms from.

    Raises rather than logging: a term that matches nothing in the formula
    silently reduces nothing, and a formula term no label drops is a
    contribution the analysis never measures. Both are configuration errors.

    Parameters
    ----------
    formula : str
        A RESPONSE_MODEL_FORMULA-shaped Wilkinson formula.
    dropped_terms : dict[str, list[str]]
        A RESPONSE_DROPPED_TERMS-shaped table, label → terms dropped under it.

    Raises
    ------
    ValueError
        Naming the offending label and term, when a listed term is not a term
        of `formula`; or naming the term, when a formula term appears under no
        label.
    """
    terms = formula_terms(formula)
    for label, dropped in dropped_terms.items():
        for term in dropped:
            if term not in terms:
                raise ValueError(
                    f"drop-one label {label!r} drops term {term!r}, which is "
                    'not a term of the response model formula')
    covered = {term for dropped in dropped_terms.values() for term in dropped}
    for term in terms:
        if term not in covered:
            raise ValueError(
                f"response model term {term!r} is dropped under no drop-one "
                'label, so its contribution is never measured')


validate_dropped_terms(RESPONSE_MODEL_FORMULA, RESPONSE_DROPPED_TERMS)

# False-start cutoff: a go trial whose response_time is at or below this is a
# wheel turn already underway at stimulus onset, not a response to the stimulus.
MIN_RESPONSE_TIME = 0.05  # seconds

# Per-session OLS drop-one thresholds: minimum trials for a recording to be fit,
# and minimum recordings per mouse (per cell) for that mouse to be plotted.
MIN_TRIALS_PERSESSION = 50
MIN_RECORDINGS_PERMOUSE = 3
# Drop-one significance: false-discovery-rate threshold on q_value, applied at
# both grains (per-mouse mean dashes, per-session dots). A mark whose q_value is
# at or above this alpha is drawn gray in the per-session ΔR² grid.
PERSESSION_SIGNIFICANCE_ALPHA = 0.05
# Correction families for that threshold: each (event, predictor) cell of the
# grid is its own question, so each is corrected on its own.
PERSESSION_FDR_GROUP_COLS = ('event', 'predictor')
# Rng seed for the drop-one significance pass, its reproducibility contract.
PERSESSION_PVAL_SEED = 0

# Coefficient-dispersion-vs-behavior scatter: a (subject, target_NM) unit is
# plotted only when it has at least this many scorable sessions in both the
# neural and behavioral dispersion sets.
MIN_SESSIONS_DISPERSION = 3

# CCA neural-feature blocks: the main effects defining the task and movement
# categories. select_block_terms uses these to pick each block's coefficient
# columns (mains plus within-block interactions) from the persession model.
CCA_TASK_MAINS = ['contrast', 'side', 'reward']
CCA_MOVEMENT_MAINS = ['choice_side', 'log_reaction_time', 'peak_velocity']


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

# Cohorts whose event-triggered averages carry a panel inset. Their responses
# are small enough that the y-scale shared across cohorts flattens them, so the
# same traces are drawn again on their own scale.
TRACE_INSET_TARGETNMS = ['DR-5HT', 'NBM-ACh', 'LC-NE']

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




