"""Tests for scripts/ddm_hmm_overview.py assembly functions."""
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use('Agg')

from iblnm.data import PhotometrySession

import scripts.ddm_hmm_overview as ddm


# =========================================================================
# build_mouse_states_frame
# =========================================================================

def _step_photometry(step_levels, stim_times, fs=100, duration=20.0):
    """Photometry dict whose LC signal is a known step in each pre-stimulus window.

    The signal is zero everywhere except the 0.5 s before each ``stim_times``
    entry, where it holds that trial's ``step_levels`` value — so the mean over
    ``[-0.4, -0.1]`` s before stimulus onset is that level exactly, with no
    re-implementation of the extraction.
    """
    times = np.arange(0, duration, 1 / fs)
    signal = np.zeros_like(times)
    for stim_time, level in zip(stim_times, step_levels):
        signal[(times >= stim_time - 0.5) & (times < stim_time)] = level
    return {'GCaMP_preprocessed': pd.DataFrame({'LC': signal}, index=times)}


def test_build_mouse_states_frame_attaches_window_mean_baseline(monkeypatch):
    """Each trial's ``baseline`` is the LC mean over [-0.4, -0.1] s pre-stimulus.

    The synthetic signal steps to a distinct known level in each trial's
    pre-stimulus window, so the extracted baseline must reproduce those levels
    trial by trial.
    """
    sessions = pd.DataFrame({'eid': ['e1'], 'subject': ['M']})
    group = SimpleNamespace(sessions=sessions)
    stim_times = [5.0, 10.0, 15.0]
    levels = [1.0, 2.0, 3.0]

    class FakePS:
        extract_responses = PhotometrySession.extract_responses
        mask_subsequent_events = PhotometrySession.mask_subsequent_events
        subtract_baseline = PhotometrySession.subtract_baseline

        def __init__(self, row, one=None):
            self.eid = row['eid']
            self.brain_region = ['LC']

        def load_h5(self, groups=None):
            self.trials = pd.DataFrame({'stimOn_times': stim_times,
                                        'feedback_times': [6.0, 11.0, 16.0],
                                        'choice': [1, -1, 1]})
            self.photometry = _step_photometry(levels, stim_times)

        def load_states(self):
            self.states = pd.DataFrame({'map_state': [1.0, 2.0, 1.0]})

    monkeypatch.setattr(ddm, 'PhotometrySession', FakePS)
    frame = ddm.build_mouse_states_frame(group, 'M', one=None)

    assert np.allclose(frame['baseline'], levels)


def test_build_mouse_states_frame_baseline_stays_aligned_to_its_trial(monkeypatch):
    """Baselines follow their own trials when the states join leaves NaN gaps.

    The fit covers only trials 0 and 2, so trial 1 joins a NaN ``map_state``;
    every trial must still carry the baseline computed from its own window.
    """
    sessions = pd.DataFrame({'eid': ['e1'], 'subject': ['M']})
    group = SimpleNamespace(sessions=sessions)
    stim_times = [5.0, 10.0, 15.0]
    levels = [1.0, 2.0, 3.0]

    class FakePS:
        extract_responses = PhotometrySession.extract_responses
        mask_subsequent_events = PhotometrySession.mask_subsequent_events
        subtract_baseline = PhotometrySession.subtract_baseline

        def __init__(self, row, one=None):
            self.eid = row['eid']
            self.brain_region = ['LC']

        def load_h5(self, groups=None):
            self.trials = pd.DataFrame({'stimOn_times': stim_times,
                                        'feedback_times': [6.0, 11.0, 16.0],
                                        'choice': [1, -1, 1]})
            self.photometry = _step_photometry(levels, stim_times)

        def load_states(self):
            # Fit dropped trial 1, so its map_state joins as NaN.
            self.states = pd.DataFrame({'map_state': [1.0, 2.0]}, index=[0, 2])

    monkeypatch.setattr(ddm, 'PhotometrySession', FakePS)
    frame = ddm.build_mouse_states_frame(group, 'M', one=None)

    assert frame['map_state'].isna().to_list() == [False, True, False]
    assert np.allclose(frame['baseline'], levels)

def _ambiguous_fiber_frame(monkeypatch, columns, brain_region):
    """Run build_mouse_states_frame on a session with the given fiber metadata."""
    sessions = pd.DataFrame({'eid': ['e1'], 'subject': ['M']})
    group = SimpleNamespace(sessions=sessions)
    stim_times = [5.0, 10.0, 15.0]
    levels = [1.0, 2.0, 3.0]

    class FakePS:
        extract_responses = PhotometrySession.extract_responses

        def __init__(self, row, one=None):
            self.eid = row['eid']
            self.brain_region = brain_region

        def load_h5(self, groups=None):
            self.trials = pd.DataFrame({'stimOn_times': stim_times,
                                        'choice': [1, -1, 1]})
            single = _step_photometry(levels, stim_times)['GCaMP_preprocessed']
            self.photometry = {'GCaMP_preprocessed': pd.DataFrame(
                {col: single['LC'].to_numpy() for col in columns},
                index=single.index)}

        def load_states(self):
            self.states = pd.DataFrame({'map_state': [1.0, 2.0, 1.0]})

    monkeypatch.setattr(ddm, 'PhotometrySession', FakePS)
    return ddm.build_mouse_states_frame(group, 'M', one=None)


MEASURE_COLS = ['baseline', 'stimOn_response', 'feedback_response']


def test_bilateral_session_yields_nan_measures(monkeypatch):
    """Two photometry columns -> no fiber to pick, so measures are NaN, not an error.

    The bilateral sessions name their columns ``LC-l``/``LC-r``, which the
    session's ``brain_region`` entry (``'LC'``) does not match. The trials
    themselves must survive, since they still feed the behavioral figures.
    """
    frame = _ambiguous_fiber_frame(
        monkeypatch, columns=['LC-l', 'LC-r'], brain_region=['LC', 'LC'])

    assert len(frame) == 3
    assert frame[MEASURE_COLS].isna().all().all()
    assert frame['map_state'].notna().all()


def test_duplicated_brain_region_yields_nan_measures(monkeypatch):
    """One column but two brain_region entries -> ambiguous, so measures are NaN.

    These sessions carry ``['LC', 'LC']`` against a single data column, so which
    fiber the column came from is unknown even though only one signal exists.
    """
    frame = _ambiguous_fiber_frame(
        monkeypatch, columns=['LC'], brain_region=['LC', 'LC'])

    assert len(frame) == 3
    assert frame[MEASURE_COLS].isna().all().all()


def _evoked_photometry(stim_levels, feedback_levels, stim_times, feedback_times,
                       offset=0.0, fs=100, duration=20.0):
    """Photometry dict whose LC signal is a known step in each response window.

    The signal is ``offset`` everywhere except the (0.1, 0.35) s after each event
    — ``config.RESPONSE_WINDOWS['early']``, pinned here so the test fixes the
    window independently — where it holds that trial's level above ``offset``.
    The pre-event baseline window (-0.1, 0) is therefore flat at ``offset``, so a
    baseline-subtracted magnitude must come out as the level exactly, whatever
    ``offset`` is.
    """
    times = np.arange(0, duration, 1 / fs)
    signal = np.zeros_like(times)
    for event_times, levels in ((stim_times, stim_levels),
                                (feedback_times, feedback_levels)):
        for event_time, level in zip(event_times, levels):
            signal[(times >= event_time + 0.1) & (times < event_time + 0.35)] = level
    return {'GCaMP_preprocessed': pd.DataFrame({'LC': signal + offset},
                                               index=times)}


def _evoked_frame(monkeypatch, photometry, stim_times, feedback_times):
    """Run build_mouse_states_frame on a one-fiber session with the given signal."""
    sessions = pd.DataFrame({'eid': ['e1'], 'subject': ['M']})
    group = SimpleNamespace(sessions=sessions)
    n_trials = len(stim_times)

    class FakePS:
        extract_responses = PhotometrySession.extract_responses
        mask_subsequent_events = PhotometrySession.mask_subsequent_events
        subtract_baseline = PhotometrySession.subtract_baseline

        def __init__(self, row, one=None):
            self.eid = row['eid']
            self.brain_region = ['LC']

        def load_h5(self, groups=None):
            self.trials = pd.DataFrame({'stimOn_times': stim_times,
                                        'feedback_times': feedback_times,
                                        'choice': [1] * n_trials})
            self.photometry = photometry

        def load_states(self):
            self.states = pd.DataFrame({'map_state': [1.0] * n_trials})

    monkeypatch.setattr(ddm, 'PhotometrySession', FakePS)
    return ddm.build_mouse_states_frame(group, 'M', one=None)


def test_build_mouse_states_frame_attaches_evoked_response_magnitudes(monkeypatch):
    """``stimOn_response``/``feedback_response`` are the post-event step levels.

    Each trial steps to a distinct known level in its post-stimulus and its
    post-feedback response window, so the two magnitude columns must reproduce
    those levels trial by trial and independently of each other.
    """
    stim_times = [5.0, 10.0, 15.0]
    feedback_times = [6.0, 11.0, 16.0]
    stim_levels = [1.0, 2.0, 3.0]
    feedback_levels = [-1.0, -2.0, -3.0]

    frame = _evoked_frame(
        monkeypatch,
        _evoked_photometry(stim_levels, feedback_levels, stim_times, feedback_times),
        stim_times, feedback_times)

    assert np.allclose(frame['stimOn_response'], stim_levels)
    assert np.allclose(frame['feedback_response'], feedback_levels)


def test_evoked_magnitudes_are_baseline_subtracted(monkeypatch):
    """Shifting the whole signal by a constant leaves both magnitudes unchanged.

    Only per-trial baseline subtraction can remove a session-wide DC offset, so
    this pins that the raw window mean is not what lands in the columns.
    """
    stim_times = [5.0, 10.0, 15.0]
    feedback_times = [6.0, 11.0, 16.0]
    stim_levels = [1.0, 2.0, 3.0]
    feedback_levels = [-1.0, -2.0, -3.0]

    shifted = _evoked_frame(
        monkeypatch,
        _evoked_photometry(stim_levels, feedback_levels, stim_times,
                           feedback_times, offset=7.5),
        stim_times, feedback_times)

    assert np.allclose(shifted['stimOn_response'], stim_levels)
    assert np.allclose(shifted['feedback_response'], feedback_levels)


def test_stimOn_response_uses_only_samples_before_feedback(monkeypatch):
    """Masking at the next event shortens, or empties, the stimOn early window.

    Trial 0's feedback lands mid-window, so its magnitude must average only the
    samples before feedback — where the signal is 1.0 — not the window-wide mean
    of 0.4. Trial 1's feedback precedes the window, leaving nothing to average:
    NaN, with no ``RuntimeWarning`` escaping to the caller.
    """
    stim_times = [5.0, 10.0]
    feedback_times = [5.195, 10.05]
    times = np.arange(0, 20.0, 0.01)
    signal = np.zeros_like(times)
    # 1.0 over only the first 0.1 s of the (0.1, 0.35) s early window, so the
    # masked mean (1.0) and the unmasked mean (10 of 25 samples, 0.4) differ.
    for stim_time in stim_times:
        signal[(times >= stim_time + 0.1) & (times < stim_time + 0.2)] = 1.0
    photometry = {'GCaMP_preprocessed': pd.DataFrame({'LC': signal}, index=times)}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        frame = _evoked_frame(monkeypatch, photometry, stim_times, feedback_times)

    assert np.isclose(frame['stimOn_response'].iloc[0], 1.0)
    assert np.isnan(frame['stimOn_response'].iloc[1])
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]


def test_build_mouse_states_frame_drops_unfit_and_other_subjects(monkeypatch):
    """Only the requested subject's fit sessions survive the concatenation.

    A session belonging to another subject is never visited, and a session the
    mouse ran but the fit excluded (``states is None``) is dropped — leaving one
    concatenated frame carrying the kept session's trials, state columns and eid.
    """
    sessions = pd.DataFrame({'eid': ['e1', 'e2', 'e3'],
                             'subject': ['M', 'M', 'OTHER']})
    group = SimpleNamespace(sessions=sessions)

    trials = {'e1': pd.DataFrame({'choice': [1, -1, 1],
                                  'stimOn_times': [5.0, 10.0, 15.0],
                                  'feedback_times': [6.0, 11.0, 16.0]}),
              'e2': pd.DataFrame({'choice': [1, 1],
                                  'stimOn_times': [5.0, 10.0],
                                  'feedback_times': [6.0, 11.0]})}
    states = {'e1': pd.DataFrame({'map_state': [1.0, 2.0, 1.0],
                                  'state_1': [0.9, 0.2, 0.8]}),
              'e2': None}  # session absent from the fit

    class FakePS:
        extract_responses = PhotometrySession.extract_responses
        mask_subsequent_events = PhotometrySession.mask_subsequent_events
        subtract_baseline = PhotometrySession.subtract_baseline

        def __init__(self, row, one=None):
            self.eid = row['eid']
            self.brain_region = ['LC']

        def load_h5(self, groups=None):
            self.trials = trials[self.eid]
            self.photometry = _step_photometry(
                [1.0] * len(self.trials), self.trials['stimOn_times'])

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

    trials = {'e1': pd.DataFrame(),
              'e2': pd.DataFrame({'choice': [1, -1],
                                  'stimOn_times': [5.0, 10.0],
                                  'feedback_times': [6.0, 11.0]})}
    states = {'e2': pd.DataFrame({'map_state': [1.0, 2.0]})}

    class FakePS:
        extract_responses = PhotometrySession.extract_responses
        mask_subsequent_events = PhotometrySession.mask_subsequent_events
        subtract_baseline = PhotometrySession.subtract_baseline

        def __init__(self, row, one=None):
            self.eid = row['eid']
            self.brain_region = ['LC']

        def load_h5(self, groups=None):
            self.trials = trials[self.eid]
            self.photometry = _step_photometry(
                [1.0] * len(self.trials), self.trials.get('stimOn_times', []))

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
    canonical contrasts. ``contrastLeft``/``contrastRight`` are carried as
    fractions with NaN off the stimulus side, matching the stored ONE table.
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
                          contrastLeft=contrast / 100 if side == 'left' else np.nan,
                          contrastRight=contrast / 100 if side == 'right' else np.nan,
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


def _one_transition_eid(eid, post=0.8):
    """One eid with a single 0.8->0.2 (L->R) switch; state_1 rises 0.2->``post``."""
    return pd.DataFrame({
        'eid': eid,
        'probabilityLeft': [0.8, 0.8, 0.8, 0.2, 0.2, 0.2],
        'map_state': [1, 1, 1, 1, 1, 1],
        'state_1': [0.2, 0.2, 0.2, post, post, post],
        'state_2': [0.8, 0.8, 0.8, 1 - post, 1 - post, 1 - post],
    })


def test_transition_traces_are_baseline_deltas_with_sem():
    """Traces are Δ from the pre-transition baseline, with a matching SEM array.

    Two eids each contribute one identical L->R window (window=2, baseline=2):
    state_1 sits at 0.2 before the switch and 0.8 after, so the baseline-subtracted
    mean is 0 at the pre-transition lags and +0.6 at the transition. Identical
    windows give zero SEM. No R->L transition occurs, so only L->R is returned.
    """
    frame = pd.concat([_one_transition_eid('e1'), _one_transition_eid('e2')],
                      ignore_index=True)

    aligned = ddm._transition_traces(
        frame, ['state_1', 'state_2'], ddm._block_transition_indexers(),
        window=2, baseline=2)

    assert set(aligned) == {'L->R'}
    mean, sem = aligned['L->R']['mean'], aligned['L->R']['sem']
    assert mean.shape == (5, 2) and sem.shape == (5, 2)
    # state_1 (column 0): lags -2,-1 are baseline (Δ 0); lag 0 is +0.6.
    assert np.allclose(mean[:2, 0], 0.0)
    assert np.isclose(mean[2, 0], 0.6)
    assert np.allclose(sem, 0.0)


def test_transition_traces_omit_a_group_with_no_transition_anywhere():
    """A group whose indexer finds nothing in any eid gets no entry at all.

    The fixture's blocks only ever step 0.8->0.2, so no R->L transition exists.
    The 0.2->0.8 step at the boundary between the two concatenated eids is not one
    either: were detection run over the whole frame instead of per eid it would be
    picked up as an R->L at row 6.
    """
    frame = pd.concat([_one_transition_eid('e1'), _one_transition_eid('e2')],
                      ignore_index=True)

    aligned = ddm._transition_traces(
        frame, ['state_1', 'state_2'], ddm._block_transition_indexers(),
        window=2, baseline=2)

    assert 'R->L' not in aligned


def test_transition_traces_pool_one_window_per_eid():
    """Two eids pool two windows — their own, and nothing from the boundary.

    The eids differ in how far state_1 rises at the switch (Δ 0.6 and 0.4), so the
    pooled mean at lag 0 is their average and the SEM is finite: exactly two
    windows contributed. A third, cross-boundary window would move both.
    """
    frame = pd.concat([_one_transition_eid('e1', post=0.8),
                       _one_transition_eid('e2', post=0.6)], ignore_index=True)

    aligned = ddm._transition_traces(
        frame, ['state_1', 'state_2'], ddm._block_transition_indexers(),
        window=2, baseline=2)

    mean, sem = aligned['L->R']['mean'], aligned['L->R']['sem']
    assert np.isclose(mean[2, 0], 0.5)  # (0.6 + 0.4) / 2
    assert np.isclose(sem[2, 0], 0.1)   # std([0.6, 0.4], ddof=1) / sqrt(2)


# =========================================================================
# _state_switch_traces
# =========================================================================

def _states_frame(eids, map_states, **measures):
    """Fit-trial frame carrying ``map_state`` and the three NM measure columns.

    Any measure left unspecified is filled with zeros, so a test only names the
    columns whose values it asserts on.
    """
    n = len(map_states)
    columns = {measure: measures.get(measure, np.zeros(n))
               for measure in ddm.MEASURE_LABELS}
    return pd.DataFrame({'eid': eids, 'map_state': map_states, **columns})


def test_state_switch_detection_is_per_eid():
    """A state change across a session boundary is not a switch.

    ``map_state`` steps 1 -> 2 exactly once, at the boundary between the two
    eids, so no state is ever entered and the mouse contributes no trace. The
    same sequence inside one eid is one switch into state 2 — and state 1, which
    only ever opens a session, is still not entered.
    """
    across = ddm._state_switch_traces(
        _states_frame(['e1', 'e1', 'e2', 'e2'], [1, 1, 2, 2]),
        window=1, baseline=1)
    within = ddm._state_switch_traces(
        _states_frame(['e1'] * 4, [1, 1, 2, 2]), window=1, baseline=1)

    assert across == {}
    assert set(within) == set(ddm.MEASURE_LABELS)
    assert within['baseline']['mean'].shape == (3, 1)


def test_state_switch_traces_pool_over_the_origin_state():
    """Switches into a state pool into one line whatever they came from.

    ``map_state`` enters state 2 twice in one eid, once from 3 and once from 1.
    Both windows land in state 2's single column: their lag-0 baseline (0.0)
    deltas of 1.0 and 3.0 average to 2.0 with a finite SEM. Were the origin part
    of the key they would occupy two columns and each carry zero SEM.
    """
    frame = _states_frame(['e1'] * 5, [1, 3, 2, 1, 2],
                          baseline=[0.0, 0.0, 1.0, 0.0, 3.0])

    traces = ddm._state_switch_traces(frame, window=1, baseline=1)

    mean, sem = traces['baseline']['mean'], traces['baseline']['sem']
    assert mean.shape == (3, 3)  # states 1, 2 and 3 are each entered
    assert np.isclose(mean[1, 1], 2.0)  # column 1 is state 2
    assert np.isclose(sem[1, 1], 1.0)  # std([1, 3], ddof=1) / sqrt(2)


def test_state_switch_traces_stack_states_in_ascending_order():
    """Each measure gets one array whose columns are the states, low to high.

    Runs of three trials alternate between states 1 and 2, with every measure
    held at ``10 * map_state``. Entering state 2 is then a +10 step and entering
    state 1 a -10 step, so the sign at lag 0 says which column belongs to which
    state — the plotter colors columns positionally, so a swapped order would
    silently mismatch figure 6's state colors.
    """
    map_states = [1, 1, 1, 2, 2, 2] * 2
    frame = _states_frame(['e1'] * len(map_states), map_states,
                          **{measure: 10.0 * np.array(map_states)
                             for measure in ddm.MEASURE_LABELS})

    traces = ddm._state_switch_traces(frame, ddm.SWITCH_WINDOW,
                                      ddm.SWITCH_BASELINE)

    assert set(traces) == set(ddm.MEASURE_LABELS)
    for measure in ddm.MEASURE_LABELS:
        mean = traces[measure]['mean']
        assert mean.shape == (2 * ddm.SWITCH_WINDOW + 1, 2)
        assert np.isclose(mean[ddm.SWITCH_WINDOW, 0], -10.0)  # column 0: state 1
        assert np.isclose(mean[ddm.SWITCH_WINDOW, 1], 10.0)  # column 1: state 2


def test_state_switch_traces_zero_the_two_trials_before_the_switch():
    """Δ is measured from the ``SWITCH_BASELINE`` trials just before the switch.

    ``baseline`` holds 0.0 through state 1 and 1.0 from the switch into state 2
    onward, so every pre-switch lag is 0 and every lag from the switch on is 1.
    """
    map_states = [1] * 6 + [2] * 6
    frame = _states_frame(['e1'] * len(map_states), map_states,
                          baseline=[0.0] * 6 + [1.0] * 6)

    traces = ddm._state_switch_traces(frame, ddm.SWITCH_WINDOW,
                                      ddm.SWITCH_BASELINE)

    mean = traces['baseline']['mean']
    assert mean.shape == (2 * ddm.SWITCH_WINDOW + 1, 1)  # only state 2 is entered
    assert np.allclose(mean[:ddm.SWITCH_WINDOW, 0], 0.0)
    assert np.allclose(mean[ddm.SWITCH_WINDOW:, 0], 1.0)


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


def test_assemble_mouse_views_measures_view_holds_fit_trials_only(monkeypatch):
    """The 'measures' view carries the three measures for fit trials only.

    The trial the fit dropped (NaN ``map_state``) is absent, and ``state`` comes
    back as an integer label rather than the joined float.
    """
    frame = pd.DataFrame({
        'map_state': [1.0, np.nan, 2.0],
        'baseline': [0.5, 9.9, -0.5],
        'stimOn_response': [1.0, 9.9, 2.0],
        'feedback_response': [-1.0, 9.9, -2.0],
        'feedbackType': [1.0, 1.0, -1.0],
        'choice': [1.0, 1.0, -1.0],
        'eid': ['e1', 'e1', 'e2'],
        'stimOn_times': [1.0, 2.0, 3.0],
        'response_times': [1.5, 2.5, 3.5],
    })
    monkeypatch.setattr(ddm, 'build_mouse_states_frame',
                        lambda group, subject, one: frame.copy())
    monkeypatch.setattr(ddm, 'build_state_param_table',
                        lambda mouse_frame: pd.DataFrame({'state': [1, 2]}))
    monkeypatch.setattr(ddm, '_state_curves', lambda mouse_frame, params: {})
    monkeypatch.setattr(
        ddm, '_transition_traces',
        lambda frame, value_cols, groups, window, baseline: {})
    monkeypatch.setattr(ddm, '_state_switch_traces',
                        lambda frame, window, baseline: {})

    views = ddm._assemble_mouse_views(group=None, subjects=['M'], one=None)

    measures = views['measures']['M']
    assert list(measures.columns) == ['state', 'eid', 'outcome', 'baseline',
                                      'stimOn_response', 'feedback_response']
    assert list(measures['state']) == [1, 2]
    assert pd.api.types.is_integer_dtype(measures['state'])
    assert list(measures['baseline']) == [0.5, -0.5]
    assert list(measures['stimOn_response']) == [1.0, 2.0]
    assert list(measures['feedback_response']) == [-1.0, -2.0]
    assert list(measures['eid']) == ['e1', 'e2']


def test_assemble_mouse_views_measures_label_outcome_and_drop_no_go(monkeypatch):
    """``outcome`` comes from ``feedbackType``; no-go and unknown trials go.

    A no-go trial (``choice == 0``) carries ``feedbackType == -1`` like a real
    error, so only the choice filter can separate the two. A ``feedbackType``
    outside ``OUTCOMES`` has no outcome label and is dropped rather than
    plotted under a guessed one.
    """
    frame = pd.DataFrame({
        'map_state': [1.0, 1.0, 2.0, 2.0],
        'baseline': [0.5, -0.5, 0.1, 0.2],
        'stimOn_response': [1.0, 2.0, 3.0, 4.0],
        'feedback_response': [-1.0, -2.0, -3.0, -4.0],
        'feedbackType': [1.0, -1.0, -1.0, 0.0],
        'choice': [1.0, -1.0, 0.0, 1.0],
        'eid': ['e1', 'e1', 'e1', 'e1'],
        'stimOn_times': [1.0, 2.0, 3.0, 4.0],
        'response_times': [1.5, 2.5, 3.5, 4.5],
    })
    monkeypatch.setattr(ddm, 'build_mouse_states_frame',
                        lambda group, subject, one: frame.copy())
    monkeypatch.setattr(ddm, 'build_state_param_table',
                        lambda mouse_frame: pd.DataFrame({'state': [1, 2]}))
    monkeypatch.setattr(ddm, '_state_curves', lambda mouse_frame, params: {})
    monkeypatch.setattr(
        ddm, '_transition_traces',
        lambda frame, value_cols, groups, window, baseline: {})
    monkeypatch.setattr(ddm, '_state_switch_traces',
                        lambda frame, window, baseline: {})

    views = ddm._assemble_mouse_views(group=None, subjects=['M'], one=None)

    measures = views['measures']['M']
    assert list(measures['outcome']) == ['correct', 'incorrect']
    assert list(measures['baseline']) == [0.5, -0.5]


def test_assemble_mouse_views_measures_drop_all_nan_and_omit_empty_mouse(monkeypatch):
    """A trial survives on any one measure; an all-NaN mouse leaves the view.

    ``M1``'s second trial lost its baseline to the response-window mask but kept
    a finite ``stimOn_response``, so it stays. ``M2``'s every session had an
    ambiguous fiber (all three measures NaN), so it is absent from the view
    entirely while still contributing to the behavioral views.
    """
    frames = {
        'M1': pd.DataFrame({
            'map_state': [1.0, 2.0],
            'baseline': [0.5, np.nan],
            'stimOn_response': [1.0, 2.0],
            'feedback_response': [np.nan, np.nan],
            'feedbackType': [1.0, -1.0],
            'choice': [1.0, -1.0],
            'eid': ['e1', 'e2'],
            'stimOn_times': [1.0, 2.0],
            'response_times': [1.5, 2.5],
        }),
        'M2': pd.DataFrame({
            'map_state': [1.0, 2.0],
            'baseline': [np.nan, np.nan],
            'stimOn_response': [np.nan, np.nan],
            'feedback_response': [np.nan, np.nan],
            'feedbackType': [1.0, -1.0],
            'choice': [1.0, -1.0],
            'eid': ['e3', 'e3'],
            'stimOn_times': [1.0, 2.0],
            'response_times': [1.5, 2.5],
        }),
    }
    monkeypatch.setattr(ddm, 'build_mouse_states_frame',
                        lambda group, subject, one: frames[subject].copy())
    monkeypatch.setattr(ddm, 'build_state_param_table',
                        lambda mouse_frame: pd.DataFrame({'state': [1, 2]}))
    monkeypatch.setattr(ddm, '_state_curves', lambda mouse_frame, params: {})
    monkeypatch.setattr(
        ddm, '_transition_traces',
        lambda frame, value_cols, groups, window, baseline: {})
    monkeypatch.setattr(
        ddm, '_state_switch_traces',
        lambda frame, window, baseline: {'n_trials': len(frame)})

    views = ddm._assemble_mouse_views(group=None, subjects=['M1', 'M2'],
                                      one=None)

    measures = views['measures']['M1']
    assert list(measures['stimOn_response']) == [1.0, 2.0]
    assert measures['baseline'].isna().tolist() == [False, True]
    assert 'M2' not in views['measures']
    # The switch traces follow the same all-NaN rule, over the fit trials.
    assert views['switches'] == {'M1': {'n_trials': 2}}
    # M2 still contributes to the behavioral views.
    assert 'M2' in views['states'] and 'M2' in views['dwell']


# =========================================================================
# main
# =========================================================================

def _stub_main_dependencies(monkeypatch):
    """Stub everything ``main`` needs but the figure calls under test.

    Replaces the catalog/parameter reads, the group construction, the view
    assembly, the PCA and every plotter with cheap stand-ins, and returns the
    recorders: the ``_save`` names in call order, the ``plot_transition_traces``
    calls as ``(args, kwargs)``, and the stub views ``main`` renders.
    """
    views = {
        'states': {'M': pd.DataFrame({'map_state': [1.0]})},
        'dwell': {'M': pd.DataFrame({'state': [1], 'dwell': [3]})},
        'curves': {'M': {}},
        'aligned': {'M': {'L->R': {'mean': np.zeros((2 * ddm.BLOCK_WINDOW + 1, 2)),
                                   'sem': np.zeros((2 * ddm.BLOCK_WINDOW + 1, 2))}}},
        'measures': {'M': pd.DataFrame({'state': [1]})},
        'switches': {'M': {measure: {'mean': np.zeros((2 * ddm.SWITCH_WINDOW + 1, 2)),
                                     'sem': np.zeros((2 * ddm.SWITCH_WINDOW + 1, 2))}
                           for measure in ddm.MEASURE_LABELS}},
        'features': pd.DataFrame({'mouse': ['M', 'M'], 'state': [1, 2],
                                  **{col: [0.0, 1.0] for col in ddm.FEATURE_COLS}}),
    }
    monkeypatch.setattr(ddm.pd, 'read_parquet', lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(ddm.pd, 'read_csv',
                        lambda *a, **k: pd.DataFrame({'mouse': ['M']}))
    monkeypatch.setattr(
        ddm, 'PhotometrySessionGroup',
        SimpleNamespace(from_catalog=lambda *a, **k: SimpleNamespace(
            filter_sessions=lambda *a, **k: None)))
    monkeypatch.setattr(ddm, '_assemble_mouse_views',
                        lambda group, subjects, one: views)
    monkeypatch.setattr(ddm, 'pca_2d',
                        lambda features: (np.zeros((2, 2)), np.zeros((4, 2))))

    saved_names, transition_calls = [], []
    monkeypatch.setattr(ddm, '_save', lambda fig, name: saved_names.append(name))
    monkeypatch.setattr(ddm, 'plot_transition_traces',
                        lambda *args, **kwargs: transition_calls.append(
                            (args, kwargs)))
    for plotter in ('plot_state_posterior_dwell',
                    'plot_state_psychometric_chronometric',
                    'plot_state_param_scatter', 'plot_state_pca',
                    'plot_state_measures'):
        monkeypatch.setattr(ddm, plotter, lambda *a, **k: None)
    return saved_names, transition_calls, views


def test_main_renders_the_state_switch_figure(monkeypatch):
    """``main`` writes a seventh figure from the switch view, measures as columns.

    The switch traces reach ``plot_transition_traces`` unchanged, its columns are
    the ``MEASURE_LABELS`` keys in iteration order (fixing the left-to-right panel
    order), and the lag axis is labelled in trials from the state switch.
    """
    saved_names, transition_calls, views = _stub_main_dependencies(monkeypatch)

    ddm.main(one=object())

    assert len(saved_names) == 7
    assert 'state_switch_measures' in saved_names
    switch_args, switch_kwargs = transition_calls[-1]
    assert switch_args[0] is views['switches']
    assert list(switch_args[1]) == list(ddm.MEASURE_LABELS)
    assert switch_args[2] == ddm.SWITCH_WINDOW
    assert switch_kwargs['xlabel'] == 'trial from state switch'
    assert list(switch_kwargs['ylabels']) == list(ddm.MEASURE_LABELS.values())
