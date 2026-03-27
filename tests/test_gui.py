"""Tests for iblnm.gui helper functions."""
import numpy as np
import pandas as pd
import pytest

from iblnm.task import sort_trials_by_type


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def mixed_trials():
    """Trials with known correct/incorrect and contrast values."""
    return pd.DataFrame({
        'feedbackType': [1,    -1,   1,     -1,   1   ],
        'contrast':     [0.5,  0.25, 0.125, 0.5,  0.25],
    })
    # Correct at positions: 0 (0.5), 2 (0.125), 4 (0.25)
    # Incorrect at positions: 1 (0.25), 3 (0.5)


# =========================================================================
# sort_trials_by_type
# =========================================================================

def test_incorrect_before_correct(mixed_trials):
    """All incorrect positions come before all correct positions (origin='lower')."""
    idx = sort_trials_by_type(mixed_trials)
    incorrect_pos = set(np.where(mixed_trials['feedbackType'] == -1)[0])
    correct_pos   = set(np.where(mixed_trials['feedbackType'] ==  1)[0])

    last_incorrect = max(i for i, v in enumerate(idx) if v in incorrect_pos)
    first_correct  = min(i for i, v in enumerate(idx) if v in correct_pos)
    assert last_incorrect < first_correct


def test_correct_sorted_desc_contrast(mixed_trials):
    """Correct trials appear in descending contrast order."""
    idx = sort_trials_by_type(mixed_trials)
    correct_positions = [i for i, v in enumerate(idx)
                         if mixed_trials.loc[v, 'feedbackType'] == 1]
    contrasts = [mixed_trials.loc[idx[i], 'contrast'] for i in correct_positions]
    assert contrasts == sorted(contrasts, reverse=True)


def test_incorrect_sorted_desc_contrast(mixed_trials):
    """Incorrect trials appear in descending contrast order."""
    idx = sort_trials_by_type(mixed_trials)
    incorrect_positions = [i for i, v in enumerate(idx)
                           if mixed_trials.loc[v, 'feedbackType'] == -1]
    contrasts = [mixed_trials.loc[idx[i], 'contrast'] for i in incorrect_positions]
    assert contrasts == sorted(contrasts, reverse=True)


def test_all_trials_included(mixed_trials):
    """Output contains every trial index exactly once."""
    idx = sort_trials_by_type(mixed_trials)
    assert sorted(idx) == list(range(len(mixed_trials)))


def test_all_correct():
    """Works when all trials are correct — no crash, all indices returned."""
    trials = pd.DataFrame({'feedbackType': [1, 1, 1], 'contrast': [0.5, 0.25, 1.0]})
    idx = sort_trials_by_type(trials)
    assert sorted(idx) == [0, 1, 2]
    contrasts = [trials.loc[i, 'contrast'] for i in idx]
    assert contrasts == sorted(contrasts, reverse=True)


def test_all_incorrect():
    """Works when all trials are incorrect — no crash, all indices returned."""
    trials = pd.DataFrame({'feedbackType': [-1, -1, -1], 'contrast': [0.5, 0.25, 1.0]})
    idx = sort_trials_by_type(trials)
    assert sorted(idx) == [0, 1, 2]
    contrasts = [trials.loc[i, 'contrast'] for i in idx]
    assert contrasts == sorted(contrasts, reverse=True)
