"""Tests for scripts/collect_trials.py, on synthetic frames only."""
import numpy as np
import pandas as pd

import scripts.collect_trials as ct


def mock_trials(n_trials=5, **overrides):
    """A stored trials table carrying every raw column plus the derived ones.

    Every trial is complete and answered: a right-side stimulus, feedback
    0.5 s after the go cue. ``overrides`` replaces whole columns, so a test
    states only the values its assertion turns on.
    """
    trial_starts = np.arange(n_trials, dtype=float) * 10
    trials = pd.DataFrame({
        'intervals_0': trial_starts,
        'intervals_1': trial_starts + 9,
        'stimOnTrigger_times': trial_starts + 1,
        'stimOn_times': trial_starts + 1.01,
        'stimOffTrigger_times': trial_starts + 2,
        'stimOff_times': trial_starts + 2.01,
        'goCueTrigger_times': trial_starts + 1,
        'goCue_times': trial_starts + 1.02,
        'firstMovement_times': trial_starts + 1.2,
        'response_times': trial_starts + 1.5,
        'feedback_times': trial_starts + 1.52,
        'feedbackType': np.ones(n_trials),
        'choice': -np.ones(n_trials),
        'contrastLeft': np.full(n_trials, np.nan),
        'contrastRight': np.full(n_trials, 1.0),
        'probabilityLeft': np.full(n_trials, 0.5),
        'rewardVolume': np.full(n_trials, 1.5),
        'quiescencePeriod': np.full(n_trials, 0.4),
        'trial': np.arange(n_trials),
        'stim_side': ['right'] * n_trials,
        'signed_contrast': np.ones(n_trials),
        'contrast': np.ones(n_trials),
    })
    return trials.assign(**overrides)


def mock_session():
    """A catalog row carrying the five identity columns."""
    return pd.Series({
        'subject': 'ZFM-001',
        'eid': 'abc-123',
        'day_n': 7,
        'session_n': 3,
        'session_type': 'biased',
        'target_NM': ['VTA-DA'],
    })


EXPECTED_COLUMNS = (
    ['subject', 'eid', 'day_n', 'session_n', 'session_type', 'trial_n']
    + ct.RAW_COLUMNS
    + ['stim_side', 'reaction_time', 'false_start', 'no_choice', 'incomplete']
)


def test_build_export_columns_in_order():
    """The export's columns are exactly the fixed set, in the fixed order."""
    export = ct.build_export(mock_trials(n_trials=5), mock_session())

    assert list(export.columns) == EXPECTED_COLUMNS
    assert len(export) == 5


def test_build_export_no_choice_trial_has_no_reaction_time():
    """A no-choice trial is flagged, and its timeout is not a reaction time."""
    trials = mock_trials(n_trials=3)
    trials.loc[1, 'choice'] = 0.0
    trials.loc[1, 'feedback_times'] = trials.loc[1, 'goCue_times'] + 60.029

    export = ct.build_export(trials, mock_session())

    assert list(export['no_choice']) == [False, True, False]
    assert np.isnan(export.loc[1, 'reaction_time'])
    assert export.loc[1, 'false_start'] == False  # noqa: E712 — not None/NaN
    assert export.loc[0, 'reaction_time'] == 0.5


def test_build_export_false_start_below_threshold_only():
    """A 20 ms response is a false start; a 200 ms one is not."""
    trials = mock_trials(n_trials=2)
    trials['feedback_times'] = trials['goCue_times'] + [0.02, 0.2]

    export = ct.build_export(trials, mock_session())

    assert list(export['false_start']) == [True, False]
    assert export['false_start'].dtype == bool


def test_build_export_incomplete_scans_only_the_scanned_columns():
    """A NaN flags a trial incomplete unless it is NaN by construction."""
    trials = mock_trials(n_trials=3)
    trials.loc[0, 'rewardVolume'] = np.nan
    trials.loc[1, ['stimOn_times', 'firstMovement_times']] = np.nan

    export = ct.build_export(trials, mock_session())

    assert list(export['incomplete']) == [True, False, False]
    assert export['incomplete'].dtype == bool


def test_build_export_incomplete_requires_exactly_one_contrast():
    """Both contrasts present, or neither, is a malformed trial."""
    trials = mock_trials(n_trials=3)
    trials.loc[0, 'contrastLeft'] = 0.25
    trials.loc[1, 'contrastRight'] = np.nan

    export = ct.build_export(trials, mock_session())

    assert list(export['incomplete']) == [True, True, False]


def test_build_export_rejects_a_table_missing_a_raw_column():
    """A session whose extractor never produced a raw column is dropped whole."""
    trials = mock_trials().drop(columns='stimOff_times')

    assert ct.build_export(trials, mock_session()) is None


def test_build_export_drops_columns_outside_the_exported_set():
    """A column the export does not name never reaches the file."""
    trials = mock_trials().assign(intervals_bpod_0=0.0)

    export = ct.build_export(trials, mock_session())

    assert 'intervals_bpod_0' not in export.columns
    assert 'signed_contrast' not in export.columns


def test_build_export_carries_the_identity_columns_and_trial_index():
    """Identity comes from the catalog row; `trial_n` is the stored index."""
    trials = mock_trials(n_trials=3, trial=[4, 5, 6])

    export = ct.build_export(trials, mock_session())

    assert set(export['subject']) == {'ZFM-001'}
    assert set(export['eid']) == {'abc-123'}
    assert set(export['day_n']) == {7}
    assert set(export['session_n']) == {3}
    assert set(export['session_type']) == {'biased'}
    assert list(export['trial_n']) == [4, 5, 6]
