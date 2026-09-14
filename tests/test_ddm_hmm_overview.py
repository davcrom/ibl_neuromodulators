"""Tests for scripts/ddm_hmm_overview.py assembly functions."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import scripts.ddm_hmm_overview as ddm


# =========================================================================
# load_session_states
# =========================================================================

class _FitSession:
    """Stand-in session: stored trials plus one K's fit, read through the API."""

    def __init__(self, trials, fit, eid='e1', subject='M1'):
        self.eid, self.subject = eid, subject
        self._trials, self._fit = trials, fit

    def load_h5(self, groups=None):
        self.trials = self._trials

    def load_hmm(self, k):
        if k not in self._fit:
            raise KeyError(k)
        return self._fit[k]


def test_load_session_states_joins_the_fit_on_trial_keeping_trial_columns():
    """Fit columns join on ``trial``; a shared column keeps the stored value.

    The fit's ``choice`` is sign-flipped against the store's, so the stored
    ``choice`` must survive the join. The fit's rows come in reverse order, so
    only a join on ``trial`` puts each state on its own trial.
    """
    trials = pd.DataFrame({'trial': [0, 1, 2], 'choice': [1, -1, 1]})
    fit_trials = pd.DataFrame({'trial': [2, 1, 0], 'choice': [-1, 1, -1],
                               'viterbi_state': [3, 2, 1]})
    attrs = {'state': np.array([1, 2, 3])}
    ps = _FitSession(trials, {4: {'trials': fit_trials, 'attrs': attrs}})

    fit = ddm.load_session_states(ps, k=4)

    frame = fit['trials']
    assert list(frame['choice']) == [1, -1, 1]
    assert list(frame['viterbi_state']) == [1, 2, 3]
    assert set(frame['eid']) == {'e1'} and set(frame['subject']) == {'M1'}
    assert fit['attrs'] is attrs


# =========================================================================
# behavioral_frame
# =========================================================================

def _new_format_fit():
    """K=2 new-format session: states 1, 2 are DDM, state 3 is no-response.

    Trial 2 is the omission, assigned state 3 with posterior 1.0 there.
    ``map_state`` is absent, as in every new-format fit.
    """
    trials = pd.DataFrame({
        'trial': [0, 1, 2, 3],
        'omission': [False, False, True, False],
        'viterbi_state': [1, 2, 3, 2],
        'p_state_1': [0.9, 0.2, 0.0, 0.1],
        'p_state_2': [0.1, 0.8, 0.0, 0.9],
        'p_state_3': [0.0, 0.0, 1.0, 0.0],
    })
    attrs = {'state': np.array([1, 2, 3]),
             'kind': np.array(['ddm', 'ddm', 'omission'])}
    return trials, attrs


def test_behavioral_frame_strips_the_no_response_state_from_a_new_format_fit():
    """Omission rows and the omission posterior go; ``state`` is Viterbi's."""
    trials, attrs = _new_format_fit()

    frame, labels = ddm.behavioral_frame(trials, attrs)

    assert list(frame['trial']) == [0, 1, 3]
    assert list(frame['state']) == [1, 2, 2]
    assert 'p_state_3' not in frame.columns
    assert {'p_state_1', 'p_state_2'} <= set(frame.columns)
    assert labels == [1, 2]


def test_behavioral_frame_keeps_every_row_of_an_old_format_fit():
    """No ``omission``, no ``kind``: nothing dropped, ``state`` is the MAP state."""
    trials = pd.DataFrame({
        'trial': [0, 1, 2],
        'map_state': [2, 1, 2],
        'p_state_1': [0.3, 0.7, 0.4],
        'p_state_2': [0.7, 0.3, 0.6],
    })

    frame, labels = ddm.behavioral_frame(trials, {'state': np.array([1, 2])})

    assert list(frame['trial']) == [0, 1, 2]
    assert list(frame['state']) == [2, 1, 2]
    assert {'p_state_1', 'p_state_2'} <= set(frame.columns)
    assert labels == [1, 2]


# =========================================================================
# state_params
# =========================================================================

def test_state_params_drops_the_no_response_state_of_a_new_format_fit():
    """The `kind == 'omission'` row carries NaN DDM params and must not plot."""
    attrs = {'state': np.array([1, 2, 3]),
             'kind': np.array(['ddm', 'ddm', 'omission']),
             'B': np.array([1.0, 2.0, np.nan]),
             'k': np.array([3.0, 4.0, np.nan]),
             'a0': np.array([5.0, 6.0, np.nan])}

    params = ddm.state_params('M1', attrs)

    assert list(params['state']) == [1, 2]
    assert list(params['mouse']) == ['M1', 'M1']
    assert list(params['B']) == [1.0, 2.0]
    assert list(params['k']) == [3.0, 4.0]
    assert list(params['a0']) == [5.0, 6.0]


def test_state_params_keeps_every_state_of_an_old_format_fit():
    """Old-format attrs carry no `kind`, and so no no-response state."""
    attrs = {'state': np.array([1, 2, 3, 4]),
             'B': np.array([1.0, 2.0, 3.0, 4.0]),
             'k': np.array([1.0, 2.0, 3.0, 4.0]),
             'a0': np.array([1.0, 2.0, 3.0, 4.0])}

    params = ddm.state_params('M2', attrs)

    assert list(params['state']) == [1, 2, 3, 4]
    assert set(params['mouse']) == {'M2'}


# =========================================================================
# build_state_param_table, _state_curves
# =========================================================================

def _make_state_trials(state, p_choose_right, rt_base, rt_slope, seed):
    """Synthetic trials for one state with a known psychometric bias and RT slope.

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
            rows += [dict(state=state, choice=choice[i],
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
            rows += [{'state': 1, 'choice': 1, 'signed_contrast': sc,
                      'contrast': abs(sc), 'rt': rt, 'feedbackType': feedback,
                      'stim_side': 'left' if sc < 0 else 'right'}
                     for _ in range(3)]
        rows += [{'state': 1, 'choice': 1, 'signed_contrast': -0.0,
                  'contrast': 0.0, 'rt': rt, 'feedbackType': feedback,
                  'stim_side': 'left'} for _ in range(3)]
        rows += [{'state': 1, 'choice': 1, 'signed_contrast': 0.0,
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


# =========================================================================
# _transition_traces
# =========================================================================

def _one_transition_eid(eid, post=0.8):
    """One eid with a single 0.8->0.2 (L->R) switch; p_state_1 rises 0.2->``post``."""
    return pd.DataFrame({
        'eid': eid,
        'probabilityLeft': [0.8, 0.8, 0.8, 0.2, 0.2, 0.2],
        'state': [1, 1, 1, 1, 1, 1],
        'p_state_1': [0.2, 0.2, 0.2, post, post, post],
        'p_state_2': [0.8, 0.8, 0.8, 1 - post, 1 - post, 1 - post],
    })


def test_transition_traces_are_baseline_deltas_with_sem():
    """Traces are Δ from the pre-transition baseline, with a matching SEM array.

    Two eids each contribute one identical L->R window (window=2, baseline=2):
    p_state_1 sits at 0.2 before the switch and 0.8 after, so the
    baseline-subtracted mean is 0 at the pre-transition lags and +0.6 at the
    transition. Identical windows give zero SEM. No R->L transition occurs, so
    only L->R is returned.
    """
    frame = pd.concat([_one_transition_eid('e1'), _one_transition_eid('e2')],
                      ignore_index=True)

    aligned = ddm._transition_traces(
        frame, ['p_state_1', 'p_state_2'], ddm._block_transition_indexers(),
        window=2, baseline=2)

    assert set(aligned) == {'L->R'}
    mean, sem = aligned['L->R']['mean'], aligned['L->R']['sem']
    assert mean.shape == (5, 2) and sem.shape == (5, 2)
    # p_state_1 (column 0): lags -2,-1 are baseline (Δ 0); lag 0 is +0.6.
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
        frame, ['p_state_1', 'p_state_2'], ddm._block_transition_indexers(),
        window=2, baseline=2)

    assert 'R->L' not in aligned


def test_transition_traces_pool_one_window_per_eid():
    """Two eids pool two windows — their own, and nothing from the boundary.

    The eids differ in how far p_state_1 rises at the switch (Δ 0.6 and 0.4), so
    the pooled mean at lag 0 is their average and the SEM is finite: exactly two
    windows contributed. A third, cross-boundary window would move both.
    """
    frame = pd.concat([_one_transition_eid('e1', post=0.8),
                       _one_transition_eid('e2', post=0.6)], ignore_index=True)

    aligned = ddm._transition_traces(
        frame, ['p_state_1', 'p_state_2'], ddm._block_transition_indexers(),
        window=2, baseline=2)

    mean, sem = aligned['L->R']['mean'], aligned['L->R']['sem']
    assert np.isclose(mean[2, 0], 0.5)  # (0.6 + 0.4) / 2
    assert np.isclose(sem[2, 0], 0.1)   # std([0.6, 0.4], ddof=1) / sqrt(2)


# =========================================================================
# _state_switch_traces
# =========================================================================

MEASURES = ['baseline', 'stimulus', 'feedback']


def _states_frame(eids, states, **measures):
    """Fit-trial frame carrying ``state`` and the three :data:`MEASURES` columns.

    Any measure left unspecified is filled with zeros, so a test only names the
    columns whose values it asserts on.
    """
    n = len(states)
    columns = {measure: measures.get(measure, np.zeros(n)) for measure in MEASURES}
    return pd.DataFrame({'eid': eids, 'state': states, **columns})


def test_state_switch_detection_is_per_eid():
    """A state change across a session boundary is not a switch.

    ``state`` steps 1 -> 2 exactly once, at the boundary between the two eids,
    so no state is ever entered and the mouse contributes no trace. The same
    sequence inside one eid is one switch into state 2 — and state 1, which
    only ever opens a session, is still not entered.
    """
    across = ddm._state_switch_traces(
        _states_frame(['e1', 'e1', 'e2', 'e2'], [1, 1, 2, 2]), MEASURES,
        window=1, baseline=1)
    within = ddm._state_switch_traces(
        _states_frame(['e1'] * 4, [1, 1, 2, 2]), MEASURES, window=1, baseline=1)

    assert across == {}
    assert set(within) == set(MEASURES)
    assert within['baseline']['mean'].shape == (3, 1)


def test_state_switch_traces_pool_over_the_origin_state():
    """Switches into a state pool into one line whatever they came from.

    ``state`` enters state 2 twice in one eid, once from 3 and once from 1.
    Both windows land in state 2's single column: their lag-0 baseline (0.0)
    deltas of 1.0 and 3.0 average to 2.0 with a finite SEM. Were the origin part
    of the key they would occupy two columns and each carry zero SEM.
    """
    frame = _states_frame(['e1'] * 5, [1, 3, 2, 1, 2],
                          baseline=[0.0, 0.0, 1.0, 0.0, 3.0])

    traces = ddm._state_switch_traces(frame, MEASURES, window=1, baseline=1)

    mean, sem = traces['baseline']['mean'], traces['baseline']['sem']
    assert mean.shape == (3, 3)  # states 1, 2 and 3 are each entered
    assert np.isclose(mean[1, 1], 2.0)  # column 1 is state 2
    assert np.isclose(sem[1, 1], 1.0)  # std([1, 3], ddof=1) / sqrt(2)


def test_state_switch_traces_stack_states_in_ascending_order():
    """Each measure gets one array whose columns are the states, low to high.

    Runs of three trials alternate between states 1 and 2, with every measure
    held at ``10 * state``. Entering state 2 is then a +10 step and entering
    state 1 a -10 step, so the sign at lag 0 says which column belongs to which
    state — the plotter colors columns positionally, so a swapped order would
    silently mismatch the state colors.
    """
    states = [1, 1, 1, 2, 2, 2] * 2
    frame = _states_frame(['e1'] * len(states), states,
                          **{measure: 10.0 * np.array(states)
                             for measure in MEASURES})

    traces = ddm._state_switch_traces(frame, MEASURES, ddm.SWITCH_WINDOW,
                                      ddm.SWITCH_BASELINE)

    assert set(traces) == set(MEASURES)
    for measure in MEASURES:
        mean = traces[measure]['mean']
        assert mean.shape == (2 * ddm.SWITCH_WINDOW + 1, 2)
        assert np.isclose(mean[ddm.SWITCH_WINDOW, 0], -10.0)  # column 0: state 1
        assert np.isclose(mean[ddm.SWITCH_WINDOW, 1], 10.0)  # column 1: state 2


def test_state_switch_traces_zero_the_two_trials_before_the_switch():
    """Δ is measured from the ``SWITCH_BASELINE`` trials just before the switch.

    ``baseline`` holds 0.0 through state 1 and 1.0 from the switch into state 2
    onward, so every pre-switch lag is 0 and every lag from the switch on is 1.
    """
    states = [1] * 6 + [2] * 6
    frame = _states_frame(['e1'] * len(states), states,
                          baseline=[0.0] * 6 + [1.0] * 6)

    traces = ddm._state_switch_traces(frame, MEASURES, ddm.SWITCH_WINDOW,
                                      ddm.SWITCH_BASELINE)

    mean = traces['baseline']['mean']
    assert mean.shape == (2 * ddm.SWITCH_WINDOW + 1, 1)  # only state 2 is entered
    assert np.allclose(mean[:ddm.SWITCH_WINDOW, 0], 0.0)
    assert np.allclose(mean[ddm.SWITCH_WINDOW:, 0], 1.0)


# =========================================================================
# load_session_measures
# =========================================================================

class _MeasureSession(_FitSession):
    """`_FitSession` that also measures one response magnitude per entry.

    Each ``extract_response_magnitudes`` call answers for the entry's own
    event, as the real method does, so the caller's per-measure loop gets a
    different column of values each time. ``responses_loaded`` records that the
    stored cut was read before any measurement.
    """

    def __init__(self, trials, fit, magnitudes, **kwargs):
        super().__init__(trials, fit, **kwargs)
        self._magnitudes = magnitudes
        self.responses_loaded = False

    def load_responses(self, modality):
        self.responses_loaded = True

    def extract_response_magnitudes(self, window, masking_events,
                                    baseline_correct, events=None):
        assert self.responses_loaded
        return self._magnitudes[events[0], window]


def _magnitude_frame(trials, response, region='VTA'):
    """One event's magnitudes: one row per trial of one recording."""
    return pd.DataFrame({
        'trial': trials, 'target_NM': 'VTA-DA', 'brain_region': region,
        'event': 'x', 'response': response, 'masked_fraction': 0.0,
    })


def test_load_session_measures_puts_one_column_per_measure_on_the_trials():
    """The three RESPONSES entries become three columns, aligned on ``trial``.

    The fit holds trials 0-2 and the magnitudes 1-3, so the inner joins keep
    trials 1 and 2 — and the magnitudes arrive in a different row order than
    the trials, so only a join on ``trial`` puts each value on its own trial.
    """
    trials = pd.DataFrame({'trial': [0, 1, 2], 'choice': [1, -1, 1]})
    fit_trials = pd.DataFrame({'trial': [0, 1, 2], 'viterbi_state': [1, 2, 1]})
    entries = {measure: ddm.RESPONSES[measure] for measure in ddm.MEASURE_LABELS}
    magnitudes = {
        (entry['event'], entry['window']):
            _magnitude_frame([3, 2, 1], [30.0 + i, 20.0 + i, 10.0 + i])
        for i, entry in enumerate(entries.values())
    }
    ps = _MeasureSession(trials, {4: {'trials': fit_trials, 'attrs': {}}},
                         magnitudes)

    frame = ddm.load_session_measures(ps, k=4)['trials']

    assert list(frame['trial']) == [1, 2]
    assert set(ddm.MEASURE_LABELS) <= set(frame.columns)
    # Measure i's value on trial 1 is 10 + i, on trial 2 it is 20 + i.
    for i, measure in enumerate(ddm.MEASURE_LABELS):
        assert list(frame[measure]) == [10.0 + i, 20.0 + i]
    assert list(frame['brain_region']) == ['VTA', 'VTA']


def test_load_session_measures_keeps_one_row_per_recording_and_trial():
    """A two-fiber session contributes both fibers' measures, not one merged row."""
    trials = pd.DataFrame({'trial': [0, 1]})
    fit_trials = pd.DataFrame({'trial': [0, 1], 'viterbi_state': [1, 2]})
    entries = {measure: ddm.RESPONSES[measure] for measure in ddm.MEASURE_LABELS}
    magnitudes = {
        (entry['event'], entry['window']): pd.concat(
            [_magnitude_frame([0, 1], [1.0, 2.0], region='VTA'),
             _magnitude_frame([0, 1], [3.0, 4.0], region='SNc')],
            ignore_index=True)
        for entry in entries.values()
    }
    ps = _MeasureSession(trials, {4: {'trials': fit_trials, 'attrs': {}}},
                         magnitudes)

    frame = ddm.load_session_measures(ps, k=4)['trials']

    assert len(frame) == 4
    assert sorted(frame['brain_region']) == ['SNc', 'SNc', 'VTA', 'VTA']


def test_load_session_measures_measure_wins_a_name_shared_with_the_fit():
    """A fit column named like a measure is dropped, not suffixed.

    The old format's posteriors carry a ``stimulus`` column (the collaborator's
    stimulus-side coding, unused here) that collides with the ``stimulus``
    measure; pandas would suffix both to ``stimulus_x``/``stimulus_y`` and the
    per-mouse figure would find neither.
    """
    trials = pd.DataFrame({'trial': [0, 1]})
    fit_trials = pd.DataFrame({'trial': [0, 1], 'map_state': [1, 2],
                               'stimulus': [-1, 1]})
    entries = {measure: ddm.RESPONSES[measure] for measure in ddm.MEASURE_LABELS}
    magnitudes = {
        (entry['event'], entry['window']):
            _magnitude_frame([0, 1], [10.0 + i, 20.0 + i])
        for i, entry in enumerate(entries.values())
    }
    ps = _MeasureSession(trials, {4: {'trials': fit_trials, 'attrs': {}}},
                         magnitudes)

    frame = ddm.load_session_measures(ps, k=4)['trials']

    stimulus = list(ddm.MEASURE_LABELS).index('stimulus')
    assert list(frame['stimulus']) == [10.0 + stimulus, 20.0 + stimulus]
    assert not [column for column in frame.columns if column.endswith('_x')]


def test_load_session_measures_raises_when_nothing_was_measured():
    """A session whose every recording is gone fails loud rather than empty.

    The photometry-QC filter cuts recordings, not sessions, so a session whose
    only fiber fails it keeps its row with an empty ``brain_region`` and
    measures nothing. Returning that empty frame would put a mouse with no
    measured trial into the per-mouse loop.
    """
    trials = pd.DataFrame({'trial': [0, 1]})
    fit_trials = pd.DataFrame({'trial': [0, 1], 'viterbi_state': [1, 2]})
    entries = {measure: ddm.RESPONSES[measure] for measure in ddm.MEASURE_LABELS}
    magnitudes = {(entry['event'], entry['window']): _magnitude_frame([], [])
                  for entry in entries.values()}
    ps = _MeasureSession(trials, {4: {'trials': fit_trials, 'attrs': {}}},
                         magnitudes)

    with pytest.raises(ValueError, match='e1'):
        ddm.load_session_measures(ps, k=4)


# =========================================================================
# neural_panels
# =========================================================================

def test_neural_panels_difference_is_per_session_and_state():
    """Violin values come out one row per (eid, state), traces per measure.

    Each of the two sessions holds ten trials of one state, correct trials at
    2.0 and incorrect at 0.0 in every measure, so each cell's normalized
    difference is 2 / std([2]*5 + [0]*5, ddof=1) = 1.897.
    """
    rows = []
    for eid, state in [('e1', 1), ('e2', 2)]:
        for feedback, value in [(1, 2.0), (-1, 0.0)]:
            rows += [{'eid': eid, 'state': state, 'feedbackType': feedback,
                      **{measure: value for measure in ddm.MEASURE_LABELS}}
                     for _ in range(5)]
    frame = pd.DataFrame(rows)

    panels = ddm.neural_panels(frame)

    assert set(panels['differences']) == set(ddm.MEASURE_LABELS)
    differences = panels['differences']['stimulus']
    assert list(differences['eid']) == ['e1', 'e2']
    assert list(differences['state']) == [1, 2]
    assert np.allclose(differences['stimulus'], 1.8974, atol=1e-4)
    # No state is ever entered within a session, so no mouse trace exists.
    assert panels['traces'] == {}


# =========================================================================
# main
# =========================================================================

def _session_fit(eid, subject, converged):
    """One old-format session fit as `load_session_states` returns it."""
    trials = pd.DataFrame({
        'trial': [0, 1, 2], 'map_state': [1, 2, 1],
        'p_state_1': [0.9, 0.1, 0.8], 'p_state_2': [0.1, 0.9, 0.2],
        'probabilityLeft': [0.5, 0.5, 0.5],
        ddm.STIM_ONSET_EVENT: [1.0, 2.0, 3.0],
        'response_times': [1.5, 2.5, 3.5],
        'eid': eid, 'subject': subject,
    })
    return {'trials': trials,
            'attrs': {'state': np.array([1, 2]), 'converged': converged,
                      'B': np.array([1.0, 2.0]), 'k': np.array([3.0, 4.0]),
                      'a0': np.array([5.0, 6.0])}}


def _measured_session(eid, subject, target_nm='SNc-DA'):
    """One session as `load_session_measures` returns it: a fit plus measures."""
    fit = _session_fit(eid, subject, True)
    trials = fit['trials'].assign(
        target_NM=target_nm, brain_region='SNc', feedbackType=[1, -1, 1],
        **{measure: [0.1, 0.2, 0.3] for measure in ddm.MEASURE_LABELS})
    return {'trials': trials, 'attrs': fit['attrs']}


def _group(results_by_fn, process_calls=None):
    """Stand-in group whose ``process`` answers per loaded function."""
    def process(fn, **kwargs):
        if process_calls is not None:
            process_calls.append((fn, kwargs))
        return results_by_fn.get(fn, [])

    return SimpleNamespace(filter_sessions=lambda **kwargs: None,
                           deduplicate=lambda: None, process=process)


def test_main_saves_one_behavior_figure_per_mouse_and_warns_unconverged(
        monkeypatch, capsys):
    """One ``{subject}_behavior`` save per mouse with a fit; one warning.

    ``M1``'s fit did not converge and spans two sessions, so it must still be
    warned about once. ``M2``'s ``converged`` is blank (an old-format fit) and
    warns nothing. A session ``process`` returned ``None`` for holds no fit and
    adds no mouse.
    """
    results = [_session_fit('e1', 'M1', False), None,
               _session_fit('e2', 'M1', False), _session_fit('e3', 'M2', np.nan)]
    process_calls = []
    group = _group({ddm.load_session_states: results}, process_calls)
    monkeypatch.setattr(ddm.pd, 'read_parquet', lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(ddm, 'PhotometrySessionGroup',
                        SimpleNamespace(from_catalog=lambda *a, **k: group))
    monkeypatch.setattr(ddm, 'build_state_param_table', lambda frame: None)
    monkeypatch.setattr(ddm, '_state_curves', lambda frame, params: {})
    figures = []
    monkeypatch.setattr(ddm, 'plot_state_behavior',
                        lambda subject, *args: figures.append(subject) or subject)
    saved = []
    monkeypatch.setattr(ddm, '_save', lambda fig, name: saved.append(name))

    ddm.main(one=object())

    assert process_calls == [(ddm.load_session_states, {'k': ddm.DDM_HMM_K}),
                             (ddm.load_session_measures, {'k': ddm.DDM_HMM_K})]
    # No session survives the neural scope here, so no mouse gets a neural figure.
    assert saved == ['M1_behavior', 'M2_behavior', 'ddm_param_scatter']
    warnings = [line for line in capsys.readouterr().out.splitlines()
                if 'converge' in line]
    assert len(warnings) == 1 and 'M1' in warnings[0]


def test_main_scatters_one_row_per_mouse_and_state(monkeypatch):
    """The scatter takes each mouse's fit once, not once per session.

    ``M1`` has two sessions of the same K=2 fit, so it must contribute two
    rows, not four.
    """
    results = [_session_fit('e1', 'M1', True), _session_fit('e2', 'M1', True),
               _session_fit('e3', 'M2', np.nan)]
    group = _group({ddm.load_session_states: results})
    monkeypatch.setattr(ddm.pd, 'read_parquet', lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(ddm, 'PhotometrySessionGroup',
                        SimpleNamespace(from_catalog=lambda *a, **k: group))
    monkeypatch.setattr(ddm, 'build_state_param_table', lambda frame: None)
    monkeypatch.setattr(ddm, '_state_curves', lambda frame, params: {})
    monkeypatch.setattr(ddm, 'plot_state_behavior', lambda *args: None)
    scattered = []
    monkeypatch.setattr(ddm, 'plot_state_param_scatter',
                        lambda params, labels: scattered.append((params, labels)))
    monkeypatch.setattr(ddm, '_save', lambda fig, name: None)

    ddm.main(one=object())

    params, labels = scattered[0]
    assert len(scattered) == 1
    assert list(params['mouse']) == ['M1', 'M1', 'M2', 'M2']
    assert list(params['state']) == [1, 2, 1, 2]
    assert labels is ddm.PARAM_LABELS


def test_main_saves_a_neural_figure_only_for_mice_in_the_neural_scope(monkeypatch):
    """``M1`` is measured and gets a neural figure; ``M2`` is behavior-only.

    The neural scope applies the photometry filters the behavioral one skips,
    so a mouse with a fit need not have a measured session. The title names the
    mouse and the target-NM its recordings carry.
    """
    behavior = [_session_fit('e1', 'M1', True), _session_fit('e3', 'M2', True)]
    measured = [_measured_session('e1', 'M1'), None]
    group = _group({ddm.load_session_states: behavior,
                    ddm.load_session_measures: measured})
    monkeypatch.setattr(ddm.pd, 'read_parquet', lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(ddm, 'PhotometrySessionGroup',
                        SimpleNamespace(from_catalog=lambda *a, **k: group))
    monkeypatch.setattr(ddm, 'build_state_param_table', lambda frame: None)
    monkeypatch.setattr(ddm, '_state_curves', lambda frame, params: {})
    monkeypatch.setattr(ddm, 'plot_state_behavior', lambda *args: None)
    monkeypatch.setattr(ddm, 'plot_state_param_scatter', lambda *args: None)
    titles = []
    monkeypatch.setattr(ddm, 'plot_state_neural',
                        lambda title, *args: titles.append(title))
    saved = []
    monkeypatch.setattr(ddm, '_save', lambda fig, name: saved.append(name))

    ddm.main(one=object())

    assert [name for name in saved if name.endswith('_neural')] == ['M1_neural']
    assert titles == ['M1 SNc-DA']
