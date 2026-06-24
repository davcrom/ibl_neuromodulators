"""Tests for config constants and static config structures."""
from iblnm import config
from iblnm.config import LMM_FORMULAS, MOVEMENT_VARS


def test_pose_qc_scalar_constants():
    assert config.LIKELIHOOD_THRESHOLD == 0.9
    assert config.MOVEMENT_RESPONSE_WINDOW == (0.1, 0.35)
    assert config.CROSSCORR_LAG_WINDOW == 5.0
    assert config.CROSSCORR_FS == config.WHEEL_FS
    assert config.POSE_FS == 30


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
        'stimOn_times', ['nose_tip'], 'speed')
    assert config.POSE_MEASURES['tongue_speed'] == (
        'feedback_times', ['tongue_end_l', 'tongue_end_r'], 'sum_speed')
    assert config.POSE_MEASURES['tongue_likelihood'] == (
        'feedback_times', ['tongue_end_l', 'tongue_end_r'], 'max_likelihood')


def _format(family):
    """Format every formula in a family with the literal response column name."""
    return {name: tpl.format(response='response') for name, tpl in family.items()}


def test_task_reliability_formulas():
    # Per-event sets: reward is only known at feedback, so stimOn and
    # firstMovement drop it (identical, contrast*side only); feedback keeps it.
    event_sets = LMM_FORMULAS['task_reliability']
    no_reward = {
        'full': 'response ~ contrast * side',
        'contrast': 'response ~ side',
        'side': 'response ~ contrast',
        'interactions': 'response ~ contrast + side',
    }
    assert _format(event_sets['stimOn_times']) == no_reward
    assert _format(event_sets['firstMovement_times']) == no_reward
    # 2nd-order only, no side:reward (that interaction encodes choice).
    assert _format(event_sets['feedback_times']) == {
        'full': 'response ~ contrast * side + contrast * reward',
        'contrast': 'response ~ side + reward',
        'side': 'response ~ contrast * reward',
        'reward': 'response ~ contrast * side',
        'interactions': 'response ~ contrast + side + reward',
    }


def test_task_ceiling_formula():
    formulas = _format(LMM_FORMULAS['task_ceiling'])
    assert formulas == {'ceiling': 'response ~ C(contrast) * side * reward'}


# Per-variable predictor column: choice enters as the fiber-relative choice
# side, reaction_time log-transformed (heavy right skew), peak_velocity raw.
_EXPECTED_PREDICTORS = {
    'choice': 'choice_side',
    'reaction_time': 'log_reaction_time',
    'peak_velocity': 'peak_velocity',
}


def test_movement_vars_and_predictors():
    assert config.MOVEMENT_VARS == ['choice', 'reaction_time', 'peak_velocity']
    assert config.MOVEMENT_PREDICTORS == _EXPECTED_PREDICTORS


def test_movement_family_formulas_choice():
    # One family in full: the revised per-event task base extended with the
    # movement predictor at 2nd order. feedback carries reward (no side:reward),
    # stimOn/firstMovement omit reward.
    family = LMM_FORMULAS['movement_choice']
    assert _format(family['feedback_times']) == {
        'full': 'response ~ (contrast + side + reward + choice_side)**2 - side:reward',
        'contrast': 'response ~ (side + reward + choice_side)**2 - side:reward',
        'side': 'response ~ (contrast + reward + choice_side)**2',
        'reward': 'response ~ (contrast + side + choice_side)**2',
        'movement': 'response ~ contrast * side + contrast * reward',
        'interactions': 'response ~ contrast * side + contrast * reward + choice_side',
    }
    no_reward = {
        'full': 'response ~ (contrast + side + choice_side)**2',
        'contrast': 'response ~ (side + choice_side)**2',
        'side': 'response ~ (contrast + choice_side)**2',
        'movement': 'response ~ contrast * side',
        'interactions': 'response ~ contrast * side + choice_side',
    }
    assert _format(family['stimOn_times']) == no_reward
    assert _format(family['firstMovement_times']) == no_reward


def test_movement_families_per_event_reference_and_predictor():
    # Every movement var has an event-keyed set; each event set has a reference
    # `full` naming the predictor column. Pre-feedback events omit reward.
    for var in MOVEMENT_VARS:
        pred = _EXPECTED_PREDICTORS[var]
        family = LMM_FORMULAS[f'movement_{var}']
        assert set(family) == {
            'stimOn_times', 'firstMovement_times', 'feedback_times'}
        for event, event_set in family.items():
            full = event_set['full'].format(response='response')
            assert pred in full
            assert ('reward' in full) == (event == 'feedback_times')


def test_nested_sets_have_reference_key():
    # task_reliability and the movement families are keyed by event; each
    # event's set has the reference.
    families = ['task_reliability'] + [f'movement_{v}' for v in MOVEMENT_VARS]
    for family in families:
        for event_set in LMM_FORMULAS[family].values():
            assert 'full' in event_set


def test_persession_formulas():
    formulas = _format(LMM_FORMULAS['persession'])
    assert set(formulas) == {
        'full', 'contrast', 'side', 'reward',
        'log_reaction_time', 'peak_velocity',
    }
    assert formulas == {
        'full': 'response ~ (contrast + side + reward + log_reaction_time + peak_velocity)**2',
        'contrast': 'response ~ (side + reward + log_reaction_time + peak_velocity)**2',
        'side': 'response ~ (contrast + reward + log_reaction_time + peak_velocity)**2',
        'reward': 'response ~ (contrast + side + log_reaction_time + peak_velocity)**2',
        'log_reaction_time': 'response ~ (contrast + side + reward + peak_velocity)**2',
        'peak_velocity': 'response ~ (contrast + side + reward + log_reaction_time)**2',
    }


def test_persession_thresholds_and_path():
    assert config.MIN_TRIALS_PERSESSION == 50
    assert config.MIN_RECORDINGS_PERMOUSE == 3
    assert config.RESPONSE_OLS_PERSESSION_FPATH.name == 'response_ols_persession_dropone.csv'
