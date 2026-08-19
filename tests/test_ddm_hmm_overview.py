"""Tests for scripts/ddm_hmm_overview.py assembly functions."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use('Agg')

import scripts.ddm_hmm_overview as ddm


# =========================================================================
# build_mouse_states_frame
# =========================================================================

def test_build_mouse_states_frame_drops_unfit_and_other_subjects(monkeypatch):
    """Only the requested subject's fit sessions survive the concatenation.

    A session belonging to another subject is never visited, and a session the
    mouse ran but the fit excluded (``states is None``) is dropped — leaving one
    concatenated frame carrying the kept session's trials, state columns and eid.
    """
    sessions = pd.DataFrame({'eid': ['e1', 'e2', 'e3'],
                             'subject': ['M', 'M', 'OTHER']})
    group = SimpleNamespace(sessions=sessions)

    trials = {'e1': pd.DataFrame({'choice': [1, -1, 1]}),
              'e2': pd.DataFrame({'choice': [1, 1]})}
    states = {'e1': pd.DataFrame({'map_state': [1.0, 2.0, 1.0],
                                  'state_1': [0.9, 0.2, 0.8]}),
              'e2': None}  # session absent from the fit

    class FakePS:
        def __init__(self, row, one=None):
            self.eid = row['eid']

        def load_h5(self, groups=None):
            self.trials = trials[self.eid]

        def load_states(self):
            self.states = states[self.eid]

    monkeypatch.setattr(ddm, 'PhotometrySession', FakePS)
    frame = ddm.build_mouse_states_frame(group, 'M', one=None)

    assert list(frame['eid'].unique()) == ['e1']
    assert len(frame) == 3
    assert list(frame['map_state']) == [1, 2, 1]
    assert {'choice', 'map_state', 'state_1', 'eid'} <= set(frame.columns)


def test_build_mouse_states_frame_empty_when_no_fit_sessions(monkeypatch):
    """A mouse with no session in the fit yields an empty frame, not an error."""
    sessions = pd.DataFrame({'eid': ['e1'], 'subject': ['M']})
    group = SimpleNamespace(sessions=sessions)

    class FakePS:
        def __init__(self, row, one=None):
            self.eid = row['eid']

        def load_h5(self, groups=None):
            self.trials = pd.DataFrame({'choice': [1]})

        def load_states(self):
            self.states = None

    monkeypatch.setattr(ddm, 'PhotometrySession', FakePS)
    assert ddm.build_mouse_states_frame(group, 'M', one=None).empty


def test_build_mouse_states_frame_skips_sessions_without_trials(monkeypatch):
    """A session whose H5 has no trials group (empty frame) is skipped, not fit.

    ``load_states`` is never called on it (it would raise on absent trials), so
    only the session with real trials survives.
    """
    sessions = pd.DataFrame({'eid': ['e1', 'e2'], 'subject': ['M', 'M']})
    group = SimpleNamespace(sessions=sessions)

    trials = {'e1': pd.DataFrame(), 'e2': pd.DataFrame({'choice': [1, -1]})}
    states = {'e2': pd.DataFrame({'map_state': [1.0, 2.0]})}

    class FakePS:
        def __init__(self, row, one=None):
            self.eid = row['eid']

        def load_h5(self, groups=None):
            self.trials = trials[self.eid]

        def load_states(self):
            self.states = states[self.eid]

    monkeypatch.setattr(ddm, 'PhotometrySession', FakePS)
    frame = ddm.build_mouse_states_frame(group, 'M', one=None)

    assert list(frame['eid'].unique()) == ['e2']


# =========================================================================
# build_state_param_table
# =========================================================================

def _make_state_trials(state, p_choose_right, rt_base, rt_slope, seed):
    """Synthetic trials for one MAP state with a known psychometric bias and RT slope.

    ``p_choose_right`` is the (contrast-independent) probability of choosing right
    (IBL ``choice == -1``), so a value far from 0.5 plants a strong choice bias.
    Reaction time is ``rt_base + rt_slope * |contrast|`` plus small noise, planting
    a known chronometric slope. Covers both stimulus sides at each of the five
    canonical contrasts.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for contrast in [0, 6.25, 12.5, 25, 100]:
        for side in ['left', 'right']:
            n = 40
            signed = contrast if side == 'right' else -contrast
            choice = np.where(rng.random(n) < p_choose_right, -1, 1)
            correct_side = -1 if side == 'right' else 1
            rt = rt_base + rt_slope * contrast + rng.normal(0, 0.02, n)
            rows += [dict(map_state=state, choice=choice[i],
                          feedbackType=1 if choice[i] == correct_side else -1,
                          probabilityLeft=0.5, stim_side=side, contrast=contrast,
                          signed_contrast=signed, rt=rt[i]) for i in range(n)]
    return pd.DataFrame(rows)


def test_build_state_param_table_recovers_per_state_param_signs():
    """Each state gets one row whose fitted psychometric bias matches the plant.

    State 1 chooses right most of the time (rightward bias, negative ``bias``);
    state 2 is the mirror image (leftward bias, positive ``bias``).
    """
    frame = pd.concat([
        _make_state_trials(1, p_choose_right=0.85, rt_base=1.0, rt_slope=-0.004,
                           seed=1),
        _make_state_trials(2, p_choose_right=0.15, rt_base=0.3, rt_slope=0.004,
                           seed=2),
    ], ignore_index=True)

    table = ddm.build_state_param_table(frame)

    assert list(table['state']) == [1, 2]
    assert set(ddm.FEATURE_COLS) <= set(table.columns)
    by_state = table.set_index('state')
    assert by_state.loc[1, 'bias'] < 0 < by_state.loc[2, 'bias']


def test_state_curves_chronometric_median_rt_by_outcome_and_side():
    """Chronometric frame is median RT per (state, outcome, side); zero splits.

    Correct trials (feedbackType 1) resolve fast (0.3 s), incorrect (feedbackType
    -1) slow (0.9 s). Zero contrast appears on both sides (signed -0.0 left, +0.0
    right), so it yields two separate rows rather than one merged point.
    """
    rows = []
    for feedback, rt in [(1, 0.3), (-1, 0.9)]:
        for sc in [-100.0, -25.0, 25.0, 100.0]:
            rows += [{'map_state': 1, 'choice': 1, 'signed_contrast': sc,
                      'contrast': abs(sc), 'rt': rt, 'feedbackType': feedback,
                      'stim_side': 'left' if sc < 0 else 'right'}
                     for _ in range(3)]
        rows += [{'map_state': 1, 'choice': 1, 'signed_contrast': -0.0,
                  'contrast': 0.0, 'rt': rt, 'feedbackType': feedback,
                  'stim_side': 'left'} for _ in range(3)]
        rows += [{'map_state': 1, 'choice': 1, 'signed_contrast': 0.0,
                  'contrast': 0.0, 'rt': rt, 'feedbackType': feedback,
                  'stim_side': 'right'} for _ in range(3)]
    frame = pd.DataFrame(rows)
    param_table = pd.DataFrame({
        'state': [1], 'bias': [0.0], 'threshold': [20.0],
        'lapse_left': [0.05], 'lapse_right': [0.05],
    })

    chrono = ddm._state_curves(frame, param_table)['chronometric']

    assert set(chrono['outcome']) == {'correct', 'incorrect'}
    assert set(chrono['side']) == {'left', 'right'}
    corr = chrono.query("outcome == 'correct' and signed_contrast == -100")
    assert np.isclose(corr['median_rt'].iloc[0], 0.3)
    # Zero contrast keeps a row per side (two dots at 0), not one merged point.
    zero = chrono.query("outcome == 'correct' and signed_contrast == 0")
    assert len(zero) == 2
    assert set(zero['side']) == {'left', 'right'}


def _one_transition_eid(eid):
    """One eid with a single 0.8->0.2 (L->R) switch; state_1 rises 0.2->0.8 at it."""
    return pd.DataFrame({
        'eid': eid,
        'probabilityLeft': [0.8, 0.8, 0.8, 0.2, 0.2, 0.2],
        'map_state': [1, 1, 1, 1, 1, 1],
        'state_1': [0.2, 0.2, 0.2, 0.8, 0.8, 0.8],
        'state_2': [0.8, 0.8, 0.8, 0.2, 0.2, 0.2],
    })


def test_block_transition_traces_are_baseline_deltas_with_sem():
    """Traces are Δ from the pre-transition baseline, with a matching SEM array.

    Two eids each contribute one identical L->R window (window=2, baseline=2):
    state_1 sits at 0.2 before the switch and 0.8 after, so the baseline-subtracted
    mean is 0 at the pre-transition lags and +0.6 at the transition. Identical
    windows give zero SEM. No R->L transition occurs, so only L->R is returned.
    """
    frame = pd.concat([_one_transition_eid('e1'), _one_transition_eid('e2')],
                      ignore_index=True)

    aligned = ddm._block_transition_traces(frame, window=2, baseline=2)

    assert set(aligned) == {'L->R'}
    mean, sem = aligned['L->R']['mean'], aligned['L->R']['sem']
    assert mean.shape == (5, 2) and sem.shape == (5, 2)
    # state_1 (column 0): lags -2,-1 are baseline (Δ 0); lag 0 is +0.6.
    assert np.allclose(mean[:2, 0], 0.0)
    assert np.isclose(mean[2, 0], 0.6)
    assert np.allclose(sem, 0.0)


# =========================================================================
# _assemble_mouse_views
# =========================================================================

def test_assemble_mouse_views_raises_when_no_mouse_is_in_the_fit(monkeypatch):
    """No modeled mouse survives the group filter -> a named error, not concat's.

    Every subject yielding an empty frame leaves no per-state feature table to
    concatenate; the failure must name the cause rather than surface pandas'
    "No objects to concatenate".
    """
    monkeypatch.setattr(ddm, 'build_mouse_states_frame',
                        lambda group, subject, one: pd.DataFrame())

    with pytest.raises(ValueError, match='no modeled mouse'):
        ddm._assemble_mouse_views(group=None, subjects=['M1', 'M2'], one=None)
