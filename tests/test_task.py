"""Tests for iblnm.task functions."""
import numpy as np
import pandas as pd
import pytest

from iblnm.task import (
    get_block_lengths,
    validate_block_structure,
    count_sessions_to_stage,
    get_subjects_by_stage,
    compute_fraction_correct,
    compute_nogo_fraction,
    fit_psychometric,
    fit_psychometric_by_block,
    compute_bias_shift,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_trials_training():
    """Mock trials for a training session (no bias blocks, only 0.5 probabilityLeft)."""
    n_trials = 200
    np.random.seed(42)

    # Generate contrasts: mix of easy and hard
    contrasts = np.random.choice([0, 0.0625, 0.125, 0.25, 0.5, 1.0], size=n_trials)
    sides = np.random.choice([-1, 1], size=n_trials)  # -1 = left, 1 = right

    contrast_left = np.where(sides == -1, contrasts, 0).astype(float)
    contrast_right = np.where(sides == 1, contrasts, 0).astype(float)

    # Generate choices: generally correct for high contrast, random for low
    choice = np.where(
        contrasts >= 0.5,
        sides,  # correct for easy trials
        np.random.choice([-1, 1], size=n_trials)  # random for hard
    )

    # Add some no-go trials (choice = 0)
    nogo_idx = np.random.choice(n_trials, size=10, replace=False)
    choice[nogo_idx] = 0

    # Feedback based on choice vs stimulus side
    feedback_type = np.where(choice == sides, 1, -1)
    feedback_type[choice == 0] = -1  # no-go is always incorrect

    return pd.DataFrame({
        'contrastLeft': contrast_left,
        'contrastRight': contrast_right,
        'choice': choice,
        'feedbackType': feedback_type,
        'probabilityLeft': np.full(n_trials, 0.5),
    })


@pytest.fixture
def mock_trials_biased():
    """Mock trials for a biased session (with 20/50/80 blocks)."""
    np.random.seed(42)

    # Create proper block structure: ~40 trials per block, 3 blocks
    block_20 = np.full(50, 0.2)
    block_50 = np.full(50, 0.5)
    block_80 = np.full(50, 0.8)
    probability_left = np.concatenate([block_50, block_20, block_80, block_50])
    n_trials = len(probability_left)

    # Generate contrasts
    contrasts = np.random.choice([0, 0.0625, 0.125, 0.25, 0.5, 1.0], size=n_trials)
    sides = np.random.choice([-1, 1], size=n_trials)

    contrast_left = np.where(sides == -1, contrasts, 0).astype(float)
    contrast_right = np.where(sides == 1, contrasts, 0).astype(float)

    # Generate choices with bias effect
    choice = np.where(
        contrasts >= 0.5,
        sides,
        np.random.choice([-1, 1], size=n_trials)
    )

    # Add some no-go trials
    nogo_idx = np.random.choice(n_trials, size=10, replace=False)
    choice[nogo_idx] = 0

    feedback_type = np.where(choice == sides, 1, -1)
    feedback_type[choice == 0] = -1

    return pd.DataFrame({
        'contrastLeft': contrast_left,
        'contrastRight': contrast_right,
        'choice': choice,
        'feedbackType': feedback_type,
        'probabilityLeft': probability_left,
    })


@pytest.fixture
def mock_trials_invalid_blocks():
    """Mock trials with rapidly flipping blocks (invalid)."""
    np.random.seed(42)
    n_trials = 100

    # Blocks that flip every 2-5 trials (invalid)
    probability_left = []
    current_prob = 0.5
    probs = [0.2, 0.5, 0.8]
    while len(probability_left) < n_trials:
        block_len = np.random.randint(2, 6)  # 2-5 trials per block (too short)
        probability_left.extend([current_prob] * block_len)
        current_prob = np.random.choice([p for p in probs if p != current_prob])
    probability_left = np.array(probability_left[:n_trials])

    contrasts = np.random.choice([0, 0.0625, 0.125, 0.25, 0.5, 1.0], size=n_trials)
    sides = np.random.choice([-1, 1], size=n_trials)

    contrast_left = np.where(sides == -1, contrasts, 0).astype(float)
    contrast_right = np.where(sides == 1, contrasts, 0).astype(float)
    choice = np.random.choice([-1, 1], size=n_trials)
    feedback_type = np.where(choice == sides, 1, -1)

    return pd.DataFrame({
        'contrastLeft': contrast_left,
        'contrastRight': contrast_right,
        'choice': choice,
        'feedbackType': feedback_type,
        'probabilityLeft': probability_left,
    })


@pytest.fixture
def mock_sessions():
    """Mock sessions dataframe for multiple subjects."""
    return pd.DataFrame({
        'subject': ['mouse1', 'mouse1', 'mouse1', 'mouse1', 'mouse1',
                   'mouse2', 'mouse2', 'mouse2',
                   'mouse3', 'mouse3', 'mouse3', 'mouse3', 'mouse3', 'mouse3'],
        'session_n': [1, 2, 3, 4, 5, 1, 2, 3, 1, 2, 3, 4, 5, 6],
        'session_type': ['training', 'training', 'training', 'biased', 'biased',
                        'training', 'training', 'training',
                        'training', 'training', 'biased', 'biased', 'ephys', 'ephys'],
        'target_NM': ['VTA-DA'] * 5 + ['DR-5HT'] * 3 + ['LC-NE'] * 6,
        'date': pd.date_range('2024-01-01', periods=14),
    })


# =============================================================================
# Block Validation Tests
# =============================================================================

class TestGetBlockLengths:
    def test_single_block(self):
        """Single value throughout returns one block."""
        prob_left = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        lengths = get_block_lengths(prob_left)
        assert len(lengths) == 1
        assert lengths[0] == 5

    def test_two_blocks(self):
        """Two different values returns two blocks."""
        prob_left = np.array([0.5, 0.5, 0.5, 0.2, 0.2])
        lengths = get_block_lengths(prob_left)
        assert len(lengths) == 2
        assert lengths[0] == 3
        assert lengths[1] == 2

    def test_multiple_blocks(self):
        """Multiple block transitions."""
        prob_left = np.array([0.5, 0.5, 0.2, 0.2, 0.2, 0.8, 0.8])
        lengths = get_block_lengths(prob_left)
        assert len(lengths) == 3
        assert list(lengths) == [2, 3, 2]

    def test_empty_array(self):
        """Empty array returns empty."""
        lengths = get_block_lengths(np.array([]))
        assert len(lengths) == 0


class TestValidateBlockStructure:
    def test_valid_blocks(self, mock_trials_biased):
        """Valid block structure should pass."""
        result = validate_block_structure(mock_trials_biased)
        assert result['valid'] is True
        assert result['flagged'] is False
        assert result['min_block_length'] >= 10

    def test_invalid_flipping_blocks(self, mock_trials_invalid_blocks):
        """Rapidly flipping blocks should be flagged."""
        result = validate_block_structure(mock_trials_invalid_blocks)
        assert result['flagged'] is True
        assert result['min_block_length'] < 10

    def test_training_session_valid(self, mock_trials_training):
        """Training sessions (single block) should always be valid."""
        result = validate_block_structure(mock_trials_training)
        assert result['valid'] is True
        assert result['n_blocks'] == 1


# =============================================================================
# Session Stage Tests
# =============================================================================

class TestCountSessionsToStage:
    def test_sessions_to_biased(self, mock_sessions):
        """Test counting training sessions before first biased."""
        result = count_sessions_to_stage(mock_sessions)

        # mouse1: 3 training sessions before biased
        mouse1 = result[result['subject'] == 'mouse1'].iloc[0]
        assert mouse1['sessions_to_biased'] == 3
        assert mouse1['n_training'] == 3
        assert mouse1['n_biased'] == 2

        # mouse2: never reached biased
        mouse2 = result[result['subject'] == 'mouse2'].iloc[0]
        assert pd.isna(mouse2['sessions_to_biased'])
        assert mouse2['n_training'] == 3
        assert mouse2['n_biased'] == 0

    def test_sessions_to_ephys(self, mock_sessions):
        """Test counting biased sessions before first ephys."""
        result = count_sessions_to_stage(mock_sessions)

        # mouse3: 2 biased sessions before ephys
        mouse3 = result[result['subject'] == 'mouse3'].iloc[0]
        assert mouse3['biased_sessions_to_ephys'] == 2
        assert mouse3['n_ephys'] == 2

        # mouse1: never reached ephys
        mouse1 = result[result['subject'] == 'mouse1'].iloc[0]
        assert pd.isna(mouse1['biased_sessions_to_ephys'])


class TestGetSubjectsByStage:
    def test_subjects_by_stage(self, mock_sessions):
        """Test getting subjects that reached each stage."""
        result = get_subjects_by_stage(mock_sessions)

        assert 'training' in result
        assert 'biased' in result
        assert 'ephys' in result

        # All mice have training
        assert set(result['training']) == {'mouse1', 'mouse2', 'mouse3'}

        # Only mouse1 and mouse3 reached biased
        assert set(result['biased']) == {'mouse1', 'mouse3'}

        # Only mouse3 reached ephys
        assert set(result['ephys']) == {'mouse3'}


# =============================================================================
# Performance Metrics Tests
# =============================================================================

class TestFractionCorrect:
    def test_excludes_nogo(self, mock_trials_training):
        """Verify no-go trials are excluded by default."""
        # Count no-go trials
        n_nogo = (mock_trials_training['choice'] == 0).sum()
        assert n_nogo > 0, "Test requires some no-go trials"

        # Fraction correct excluding no-go
        frac = compute_fraction_correct(mock_trials_training, exclude_nogo=True)

        # Manually compute
        valid_trials = mock_trials_training[mock_trials_training['choice'] != 0]
        expected = (valid_trials['feedbackType'] == 1).mean()

        assert np.isclose(frac, expected)

    def test_includes_nogo(self, mock_trials_training):
        """Can include no-go trials if specified."""
        frac_excl = compute_fraction_correct(mock_trials_training, exclude_nogo=True)
        frac_incl = compute_fraction_correct(mock_trials_training, exclude_nogo=False)

        # Including no-go should give lower performance (no-go = incorrect)
        assert frac_incl <= frac_excl


class TestNogoFraction:
    def test_returns_float(self, mock_trials_training):
        """Should return float."""
        result = compute_nogo_fraction(mock_trials_training)
        assert isinstance(result, float)
        assert 0 <= result <= 1

    def test_matches_manual(self, mock_trials_training):
        """Should match manual calculation."""
        result = compute_nogo_fraction(mock_trials_training)
        expected = (mock_trials_training['choice'] == 0).mean()
        assert np.isclose(result, expected)


# =============================================================================
# Psychometric Tests
# =============================================================================

class TestFitPsychometric:
    def test_returns_expected_keys(self, mock_trials_training):
        """Verify fit returns all expected parameters."""
        result = fit_psychometric(mock_trials_training)

        expected_keys = ['bias', 'threshold', 'lapse_left', 'lapse_right',
                        'r_squared', 'n_trials']
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_filters_by_probability_left(self, mock_trials_biased):
        """Should filter trials by probabilityLeft."""
        result_50 = fit_psychometric(mock_trials_biased, probability_left=0.5)
        result_20 = fit_psychometric(mock_trials_biased, probability_left=0.2)

        # Different blocks should give different fits
        assert result_50['n_trials'] != result_20['n_trials']

    def test_r_squared_reasonable(self, mock_trials_training):
        """R-squared should be between 0 and 1."""
        result = fit_psychometric(mock_trials_training)
        assert 0 <= result['r_squared'] <= 1


class TestFitPsychometricByBlock:
    def test_returns_dict_of_blocks(self, mock_trials_biased):
        """Should return dict mapping block type to fit parameters."""
        result = fit_psychometric_by_block(mock_trials_biased)

        assert isinstance(result, dict)
        # Should have at least 50 block (always present in biased)
        assert '50' in result or '20' in result or '80' in result

    def test_skips_invalid_blocks(self, mock_trials_invalid_blocks):
        """Should skip blocks that are too short."""
        result = fit_psychometric_by_block(mock_trials_invalid_blocks)
        # With invalid block structure, may return empty or limited results
        # The function should not raise an error
        assert isinstance(result, dict)


class TestBiasShift:
    def test_computes_difference(self):
        """Bias shift is difference between block biases."""
        fit_20 = {'bias': -10, 'threshold': 20, 'lapse_left': 0.1, 'lapse_right': 0.1}
        fit_80 = {'bias': 10, 'threshold': 20, 'lapse_left': 0.1, 'lapse_right': 0.1}

        shift = compute_bias_shift(fit_20, fit_80)
        # bias_80 - bias_20 = 10 - (-10) = 20
        assert shift == 20


# =============================================================================
# Trial Contrast Computation
# =============================================================================

class TestComputeTrialContrasts:
    """Tests for compute_trial_contrasts: stim_side, contrast, signed_contrast."""

    def _make_trials(self, left, right):
        return pd.DataFrame({'contrastLeft': left, 'contrastRight': right})

    def test_right_stimulus(self):
        """Right stimulus: contrastLeft=NaN, contrastRight=0.25."""
        from iblnm.task import compute_trial_contrasts
        trials = self._make_trials([np.nan], [0.25])
        result = compute_trial_contrasts(trials)
        assert result['stim_side'].iloc[0] == 'right'
        assert result['contrast'].iloc[0] == 25.0
        assert result['signed_contrast'].iloc[0] == 25.0

    def test_left_stimulus(self):
        """Left stimulus: contrastLeft=0.5, contrastRight=NaN."""
        from iblnm.task import compute_trial_contrasts
        trials = self._make_trials([0.5], [np.nan])
        result = compute_trial_contrasts(trials)
        assert result['stim_side'].iloc[0] == 'left'
        assert result['contrast'].iloc[0] == 50.0
        assert result['signed_contrast'].iloc[0] == -50.0

    def test_zero_contrast_left(self):
        """Zero contrast on left: contrastLeft=0, contrastRight=NaN."""
        from iblnm.task import compute_trial_contrasts
        trials = self._make_trials([0.0], [np.nan])
        result = compute_trial_contrasts(trials)
        assert result['stim_side'].iloc[0] == 'left'
        assert result['contrast'].iloc[0] == 0.0
        assert result['signed_contrast'].iloc[0] == 0.0  # -1 * 0.0 = 0.0

    def test_zero_contrast_right(self):
        """Zero contrast on right: contrastLeft=NaN, contrastRight=0."""
        from iblnm.task import compute_trial_contrasts
        trials = self._make_trials([np.nan], [0.0])
        result = compute_trial_contrasts(trials)
        assert result['stim_side'].iloc[0] == 'right'
        assert result['contrast'].iloc[0] == 0.0
        assert result['signed_contrast'].iloc[0] == 0.0

    def test_mixed_trials(self):
        """Multiple trials with different sides and contrasts."""
        from iblnm.task import compute_trial_contrasts
        trials = self._make_trials(
            [np.nan, 0.25, 0.0,    np.nan],
            [1.0,    np.nan, np.nan, 0.0],
        )
        result = compute_trial_contrasts(trials)
        np.testing.assert_array_equal(
            result['stim_side'].values, ['right', 'left', 'left', 'right']
        )
        np.testing.assert_allclose(
            result['contrast'].values, [100.0, 25.0, 0.0, 0.0]
        )
        np.testing.assert_allclose(
            result['signed_contrast'].values, [100.0, -25.0, 0.0, 0.0]
        )

    def test_returns_dataframe_with_correct_columns(self):
        from iblnm.task import compute_trial_contrasts
        trials = self._make_trials([np.nan, 0.5], [0.25, np.nan])
        result = compute_trial_contrasts(trials)
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {'stim_side', 'contrast', 'signed_contrast'}

    def test_preserves_index(self):
        from iblnm.task import compute_trial_contrasts
        trials = self._make_trials([np.nan, 0.5], [0.25, np.nan])
        trials.index = [10, 20]
        result = compute_trial_contrasts(trials)
        assert list(result.index) == [10, 20]


# =============================================================================
# process_task Tests
# =============================================================================

class TestProcessTaskContrasts:
    """process_task should store the session's contrast set."""

    def test_contrasts_column_present(self, mock_trials_training):
        from scripts.task import process_task
        from iblnm.task import compute_trial_contrasts
        from unittest.mock import MagicMock

        ps = MagicMock()
        ps.eid = 'test-eid'
        trials = mock_trials_training.copy()
        ct = compute_trial_contrasts(trials)
        trials['contrast'] = ct['contrast']
        ps.trials = trials

        def _load():
            pass
        ps.load_trials = _load
        ps.validate_n_trials = _load
        ps.validate_event_completeness = _load
        ps.basic_performance.return_value = {'fraction_correct': 0.8}
        ps.validate_block_structure = _load
        ps.block_performance.return_value = {}
        ps.get_trial_timings.return_value = pd.DataFrame({
            'eid': 'test-eid', 'trial': range(len(trials)),
            'reaction_time': np.nan, 'movement_time': np.nan,
            'response_time': np.nan,
        })

        result = process_task(ps)
        assert 'contrasts' in result['performance']
        assert isinstance(result['performance']['contrasts'], list)
        assert all(c >= 0 for c in result['performance']['contrasts'])

    def test_contrasts_values_match_trials(self):
        """Contrasts should be the sorted unique percent values from trials."""
        from scripts.task import process_task
        from iblnm.task import compute_trial_contrasts
        from unittest.mock import MagicMock

        n = 60
        np.random.seed(0)
        sides = np.random.choice([-1, 1], size=n)
        contrasts_frac = np.random.choice([0, 0.25, 1.0], size=n)
        trials = pd.DataFrame({
            'contrastLeft': np.where(sides == -1, contrasts_frac, np.nan),
            'contrastRight': np.where(sides == 1, contrasts_frac, np.nan),
            'choice': sides,
            'feedbackType': np.ones(n, dtype=int),
            'probabilityLeft': np.full(n, 0.5),
        })
        ct = compute_trial_contrasts(trials)
        trials['contrast'] = ct['contrast']

        ps = MagicMock()
        ps.eid = 'eid-1'
        ps.trials = trials

        def _load():
            pass
        ps.load_trials = _load
        ps.validate_n_trials = _load
        ps.validate_event_completeness = _load
        ps.basic_performance.return_value = {'fraction_correct': 0.9}
        ps.validate_block_structure = _load
        ps.block_performance.return_value = {}
        ps.get_trial_timings.return_value = pd.DataFrame({
            'eid': 'eid-1', 'trial': range(n),
            'reaction_time': np.nan, 'movement_time': np.nan,
            'response_time': np.nan,
        })

        result = process_task(ps)
        assert result['performance']['contrasts'] == sorted([0.0, 25.0, 100.0])


class TestProcessTaskTrialTiming:

    def test_returns_trial_timing(self):
        from scripts.task import process_task
        from iblnm.task import compute_trial_contrasts
        from unittest.mock import MagicMock

        n = 30
        np.random.seed(0)
        sides = np.random.choice([-1, 1], size=n)
        contrasts_frac = np.random.choice([0, 0.25, 1.0], size=n)
        stim = np.sort(np.random.uniform(100, 500, n))
        move = stim + 0.2
        feedback = move + 0.15

        trials = pd.DataFrame({
            'contrastLeft': np.where(sides == -1, contrasts_frac, np.nan),
            'contrastRight': np.where(sides == 1, contrasts_frac, np.nan),
            'choice': sides,
            'feedbackType': np.ones(n, dtype=int),
            'probabilityLeft': np.full(n, 0.5),
            'stimOn_times': stim,
            'firstMovement_times': move,
            'feedback_times': feedback,
        })
        ct = compute_trial_contrasts(trials)
        trials['contrast'] = ct['contrast']

        expected_timing = pd.DataFrame({
            'eid': 'eid-1',
            'trial': range(n),
            'reaction_time': move - stim,
            'movement_time': feedback - move,
            'response_time': feedback - stim,
        })

        ps = MagicMock()
        ps.eid = 'eid-1'
        ps.trials = trials

        def _load():
            pass
        ps.load_trials = _load
        ps.validate_n_trials = _load
        ps.validate_event_completeness = _load
        ps.basic_performance.return_value = {'fraction_correct': 0.9}
        ps.validate_block_structure = _load
        ps.block_performance.return_value = {}
        ps.get_trial_timings.return_value = expected_timing

        result = process_task(ps)
        df_timing = result['timing']
        assert isinstance(df_timing, pd.DataFrame)
        assert set(df_timing.columns) >= {'eid', 'trial', 'reaction_time',
                                           'movement_time', 'response_time'}
        assert len(df_timing) == n
        assert (df_timing['eid'] == 'eid-1').all()

