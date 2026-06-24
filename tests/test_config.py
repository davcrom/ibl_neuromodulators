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


def test_movement_ceiling_formula():
    # Saturated 3-way of the movement predictors; no task vars, no reward.
    assert _format(LMM_FORMULAS['movement_ceiling']) == {
        'ceiling': 'response ~ choice_side * log_reaction_time * peak_velocity'}


def test_task_ceiling_formula():
    # Per-event: reward only at feedback; side:reward dropped, 3-way kept.
    ceiling = LMM_FORMULAS['task_ceiling']
    no_reward = {'ceiling': 'response ~ C(contrast) * side'}
    assert _format(ceiling['stimOn_times']) == no_reward
    assert _format(ceiling['firstMovement_times']) == no_reward
    assert _format(ceiling['feedback_times']) == {
        'ceiling': 'response ~ C(contrast) * side * reward - side:reward'}


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
    # choice interacts only with contrast (choice:side / choice:reward are
    # collinear with the reward / side mains), so its family carries no
    # choice:side or choice:reward terms. feedback carries reward (never
    # side:reward); stimOn/firstMovement omit reward.
    family = LMM_FORMULAS['movement_choice']
    assert _format(family['feedback_times']) == {
        'full': 'response ~ contrast + side + reward + contrast:side + contrast:reward + choice_side + contrast:choice_side',
        'contrast': 'response ~ side + reward + choice_side',
        'side': 'response ~ contrast + reward + contrast:reward + choice_side + contrast:choice_side',
        'reward': 'response ~ contrast + side + contrast:side + choice_side + contrast:choice_side',
        'movement': 'response ~ contrast + side + reward + contrast:side + contrast:reward',
        'interactions': 'response ~ contrast + side + reward + contrast:side + contrast:reward + choice_side',
    }
    no_reward = {
        'full': 'response ~ contrast + side + contrast:side + choice_side + contrast:choice_side',
        'contrast': 'response ~ side + choice_side',
        'side': 'response ~ contrast + choice_side + contrast:choice_side',
        'movement': 'response ~ contrast + side + contrast:side',
        'interactions': 'response ~ contrast + side + contrast:side + choice_side',
    }
    assert _format(family['stimOn_times']) == no_reward
    assert _format(family['firstMovement_times']) == no_reward


def test_choice_omits_side_reward_interactions_unlike_continuous():
    # Continuous predictors interact with side and reward; choice does not.
    rt_full = LMM_FORMULAS['movement_reaction_time']['feedback_times']['full']
    assert 'side:log_reaction_time' in rt_full
    assert 'reward:log_reaction_time' in rt_full
    choice_full = LMM_FORMULAS['movement_choice']['feedback_times']['full']
    assert 'side:choice_side' not in choice_full
    assert 'reward:choice_side' not in choice_full
    assert 'contrast:choice_side' in choice_full


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


def _termsets(formula):
    """Right-hand-side terms of a formula as a list of variable frozensets."""
    rhs = formula.split('~')[1]
    return [frozenset(t.strip().split(':')) for t in rhs.split('+')]


def test_persession_formulas():
    formulas = _format(LMM_FORMULAS['persession'])
    regressors = ['contrast', 'side', 'reward', 'choice_side',
                  'log_reaction_time', 'peak_velocity']
    assert set(formulas) == {'full', *regressors}

    full = _termsets(formulas['full'])
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
    assert config.RESPONSE_OLS_PERSESSION_FPATH.name == 'response_ols_persession_dropone.csv'
