"""Tests for scripts/ddm_hmm_overview.py assembly functions."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
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
    """Each state gets one row whose fitted param signs match the planted behavior.

    State 1 chooses right most of the time (rightward bias, negative ``bias``) with
    RT falling as contrast rises (negative slope); state 2 is the mirror image
    (leftward bias, positive ``bias``; RT rising, positive slope).
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
    assert by_state.loc[1, 'rt_slope'] < 0 < by_state.loc[2, 'rt_slope']
