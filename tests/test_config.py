"""Tests for config constants and static config structures."""

import pytest

from iblnm import config
from iblnm.config import LMM_FORMULAS


def test_pose_qc_scalar_constants():
    assert config.LIKELIHOOD_THRESHOLD == 0.9
    assert config.MOVEMENT_RESPONSE_WINDOW == (0.1, 0.35)
    assert config.CROSSCORR_LAG_WINDOW == 5.0
    assert config.CROSSCORR_FS == config.WHEEL_FS
    assert config.POSE_FS == 30


def test_response_windows_are_distinct_constants():
    """The peri-event cut and the measurement window are separate constants.

    ``RESPONSE_WINDOW`` bounds the stored cut; ``RESPONSE_MAGNITUDE_WINDOW`` is
    the post-event interval a magnitude is averaged over, and must lie inside it.
    """
    assert config.RESPONSE_WINDOW == (-1, 1)
    assert config.RESPONSE_MAGNITUDE_WINDOW == (0.1, 0.35)


def test_responses_table():
    """The three shipped analysis windows, pinned as literal data."""
    assert list(config.RESPONSES) == ['baseline', 'stimulus', 'feedback']
    assert config.RESPONSES['baseline'] == {
        'event': config.STIM_ONSET_EVENT,
        'window': (-0.35, -0.1),
        'baseline_correct': False,
        'masking_events': [],
        'ANOVA': {'contrast': [], 'side': []},
        'min_trials': 5,
        'min_subjects': 2,
    }
    assert config.RESPONSES['stimulus'] == {
        'event': config.STIM_ONSET_EVENT,
        'window': config.RESPONSE_MAGNITUDE_WINDOW,
        'baseline_correct': True,
        'masking_events': ['feedback_times'],
        'ANOVA': {'contrast': [], 'side': []},
        'min_trials': 5,
        'min_subjects': 2,
    }
    assert config.RESPONSES['feedback'] == {
        'event': 'feedback_times',
        'window': config.RESPONSE_MAGNITUDE_WINDOW,
        'baseline_correct': True,
        'masking_events': [],
        'ANOVA': {'contrast': [0, 6.25, 12.5, 25], 'side': [], 'feedbackType': []},
        'min_trials': 5,
        'min_subjects': 2,
    }
    # the measurement window is the constant itself, not a copy of its values
    assert config.RESPONSES['stimulus']['window'] is config.RESPONSE_MAGNITUDE_WINDOW
    assert config.RESPONSES['feedback']['window'] is config.RESPONSE_MAGNITUDE_WINDOW


def _responses_entry(**overrides):
    """One-entry RESPONSES table: the shipped `stimulus` entry, fields replaced."""
    return {'stimulus': {**config.RESPONSES['stimulus'], **overrides}}


def test_validate_responses_accepts_shipped_table():
    config.validate_responses(config.RESPONSES)


def test_validate_responses_rejects_uncut_event():
    with pytest.raises(ValueError, match='stimulus.*event'):
        config.validate_responses(_responses_entry(event='goCue_times'))


@pytest.mark.parametrize('window', [(-2, -1), (0.5, 1.5), (0.35, 0.1)])
def test_validate_responses_rejects_window_outside_cut(window):
    """A window must open and close inside RESPONSE_WINDOW, in that order."""
    with pytest.raises(ValueError, match='stimulus.*window'):
        config.validate_responses(_responses_entry(window=window))


@pytest.mark.parametrize('masking_events', [['goCue_times'], [config.STIM_ONSET_EVENT]])
def test_validate_responses_rejects_backward_masking(masking_events):
    """Masking is forward-only, so a masking event must follow the entry's own."""
    entry = _responses_entry(event='feedback_times', masking_events=masking_events)
    with pytest.raises(ValueError, match='stimulus.*mask'):
        config.validate_responses(entry)


def test_validate_responses_rejects_signed_contrast_factor():
    """`signed_contrast` collapses its two zero levels, so it cannot be a factor."""
    entry = _responses_entry(ANOVA={'signed_contrast': [], 'feedbackType': []})
    with pytest.raises(ValueError, match='stimulus.*signed_contrast'):
        config.validate_responses(entry)


def test_pose_qc_path_constants():
    assert config.POSE_FPATH.name == 'pose.pqt'
    assert config.POSE_LOG_FPATH.name == 'pose_log.pqt'


def test_lp_qc_labels():
    assert config.LP_QC_LABELS == ('qc_lp', 'qc_movement', 'qc_timing')


def test_video_qc_cols():
    assert config.VIDEO_QC_COLS == [
        'qc_videoLeft_focus',
        'qc_videoLeft_position',
        'qc_videoLeft_brightness',
        'qc_videoLeft_resolution',
        'qc_videoLeft_wheel_alignment',
        'qc_videoLeft_timestamps',
        'qc_videoLeft_dropped_frames',
        'qc_videoLeft_pin_state',
    ]


def test_qc_value_order():
    assert config.QC_VALUE_ORDER == ['NOT_SET', 'CRITICAL', 'FAIL', 'WARNING', 'PASS']


def test_pose_measures_structure():
    assert config.POSE_MEASURES['paw'] == (
        'firstMovement_times', ['paw_l', 'paw_r'], 'sum_speed')
    assert config.POSE_MEASURES['nose'] == (
        'stimOnTrigger_times', ['nose_tip'], 'speed')
    assert config.POSE_MEASURES['tongue_speed'] == (
        'feedback_times', ['tongue_end_l', 'tongue_end_r'], 'sum_speed')
    assert config.POSE_MEASURES['tongue_likelihood'] == (
        'feedback_times', ['tongue_end_l', 'tongue_end_r'], 'max_likelihood')


def _format(family):
    """Format every formula in a family with the literal response column name."""
    return {name: tpl.format(response='response') for name, tpl in family.items()}


# Per-variable predictor column: choice enters as the fiber-relative choice
# side, reaction_time log-transformed (heavy right skew), peak_velocity raw.
_EXPECTED_PREDICTORS = {
    'choice': 'choice_side',
    'reaction_time': 'log_reaction_time',
    'peak_velocity': 'peak_velocity',
}


def test_movement_predictors():
    assert config.MOVEMENT_PREDICTORS == _EXPECTED_PREDICTORS


def _termsets(formula):
    """Right-hand-side terms of a formula as a list of variable frozensets."""
    rhs = formula.split('~')[1]
    return [frozenset(t.strip().split(':')) for t in rhs.split('+')]


# The per-recording drop-one family, pinned as literal data: `full` carries the
# six mains and every two-way except side:reward, choice_side:side and
# choice_side:reward, and each other key drops its regressor and every term
# containing it.
_EXPECTED_PERSESSION = {
    'full':
        '{response} ~ contrast + side + reward + choice_side + log_reaction_time'
        ' + peak_velocity + contrast:side + contrast:reward'
        ' + contrast:choice_side + contrast:log_reaction_time'
        ' + contrast:peak_velocity + side:log_reaction_time'
        ' + side:peak_velocity + reward:log_reaction_time'
        ' + reward:peak_velocity + choice_side:log_reaction_time'
        ' + choice_side:peak_velocity + log_reaction_time:peak_velocity',
    'contrast':
        '{response} ~ side + reward + choice_side + log_reaction_time'
        ' + peak_velocity + side:log_reaction_time + side:peak_velocity'
        ' + reward:log_reaction_time + reward:peak_velocity'
        ' + choice_side:log_reaction_time + choice_side:peak_velocity'
        ' + log_reaction_time:peak_velocity',
    'side':
        '{response} ~ contrast + reward + choice_side + log_reaction_time'
        ' + peak_velocity + contrast:reward + contrast:choice_side'
        ' + contrast:log_reaction_time + contrast:peak_velocity'
        ' + reward:log_reaction_time + reward:peak_velocity'
        ' + choice_side:log_reaction_time + choice_side:peak_velocity'
        ' + log_reaction_time:peak_velocity',
    'reward':
        '{response} ~ contrast + side + choice_side + log_reaction_time'
        ' + peak_velocity + contrast:side + contrast:choice_side'
        ' + contrast:log_reaction_time + contrast:peak_velocity'
        ' + side:log_reaction_time + side:peak_velocity'
        ' + choice_side:log_reaction_time + choice_side:peak_velocity'
        ' + log_reaction_time:peak_velocity',
    'choice_side':
        '{response} ~ contrast + side + reward + log_reaction_time'
        ' + peak_velocity + contrast:side + contrast:reward'
        ' + contrast:log_reaction_time + contrast:peak_velocity'
        ' + side:log_reaction_time + side:peak_velocity'
        ' + reward:log_reaction_time + reward:peak_velocity'
        ' + log_reaction_time:peak_velocity',
    'log_reaction_time':
        '{response} ~ contrast + side + reward + choice_side + peak_velocity'
        ' + contrast:side + contrast:reward + contrast:choice_side'
        ' + contrast:peak_velocity + side:peak_velocity'
        ' + reward:peak_velocity + choice_side:peak_velocity',
    'peak_velocity':
        '{response} ~ contrast + side + reward + choice_side'
        ' + log_reaction_time + contrast:side + contrast:reward'
        ' + contrast:choice_side + contrast:log_reaction_time'
        ' + side:log_reaction_time + reward:log_reaction_time'
        ' + choice_side:log_reaction_time',
}


def test_persession_formulas():
    assert LMM_FORMULAS['persession'] == _EXPECTED_PERSESSION


def test_persession_family_structure():
    formulas = _format(LMM_FORMULAS['persession'])
    regressors = list(config.PERSESSION_REGRESSORS)
    assert set(formulas) == {'full', *regressors}

    full = _termsets(formulas['full'])
    assert len(full) == 18
    for r in regressors:                       # all six mains present
        assert frozenset({r}) in full
    for pair in ({'side', 'reward'}, {'choice_side', 'side'},
                 {'choice_side', 'reward'}):   # collinear/choice two-ways out
        assert frozenset(pair) not in full
    for pair in ({'contrast', 'choice_side'},  # choice keeps non-task interactions
                 {'choice_side', 'log_reaction_time'},
                 {'choice_side', 'peak_velocity'}):
        assert frozenset(pair) in full
    for reg in regressors:                     # drop-one omits the regressor
        assert all(reg not in tv for tv in _termsets(formulas[reg]))


def test_persession_thresholds_and_path():
    assert config.MIN_TRIALS_PERSESSION == 50
    assert config.MIN_RECORDINGS_PERMOUSE == 3
    assert config.OLS_PERSESSION_FPATH.name == 'ols_persession.parquet'
