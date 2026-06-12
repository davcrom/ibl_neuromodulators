"""Tests for config constants and static config structures."""
from iblnm import config
from iblnm.config import LMM_FORMULAS, TIMING_VARS


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
    assert config.LP_QC_LABELS == ('qc_lp', 'qc_movement')


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


def test_task_interactions_formulas():
    formulas = _format(LMM_FORMULAS['task_interactions'])
    assert formulas == {
        'full': 'response ~ contrast * side * reward',
        'interactions': 'response ~ contrast + side + reward',
    }


def test_task_main_effects_formulas():
    formulas = _format(LMM_FORMULAS['task_main_effects'])
    assert formulas == {
        'full': 'response ~ contrast + side + reward',
        'contrast': 'response ~ side + reward',
        'side': 'response ~ contrast + reward',
        'reward': 'response ~ contrast + side',
    }


def test_task_ceiling_formula():
    formulas = _format(LMM_FORMULAS['task_ceiling'])
    assert formulas == {'ceiling': 'response ~ C(contrast) * side * reward'}


def test_movement_families_expand_per_timing_var():
    for t in TIMING_VARS:
        assert _format(LMM_FORMULAS[f'movement_additive_{t}']) == {
            'full': f'response ~ contrast + log_{t}',
            'contrast': f'response ~ log_{t}',
            'movement': 'response ~ contrast',
        }
        assert _format(LMM_FORMULAS[f'movement_saturated_{t}']) == {
            'full': f'response ~ contrast * reward * side * log_{t}',
            'contrast': f'response ~ reward * side * log_{t}',
            'movement': 'response ~ contrast * reward * side',
        }
        assert _format(LMM_FORMULAS[f'movement_claims_{t}']) == {
            'predicts_response': f'response ~ log_{t}',
            'within_contrast': f'response ~ C(contrast) + log_{t} + side + reward',
        }


def test_nested_sets_have_reference_key():
    nested = ['task_interactions', 'task_main_effects']
    nested += [f'movement_additive_{t}' for t in TIMING_VARS]
    nested += [f'movement_saturated_{t}' for t in TIMING_VARS]
    for family in nested:
        assert 'full' in LMM_FORMULAS[family]
