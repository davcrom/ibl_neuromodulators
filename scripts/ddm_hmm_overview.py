"""DDM-HMM overview.

Reads each mouse's collaborator-fitted drift-diffusion + hidden-Markov model
(DDM-HMM) states from the session store (`hmm/ddm-k{DDM_HMM_K}`) and writes one
behavioral figure per mouse to ``figures/ddm-hmm/{subject}_behavior.svg``, 3x2:

1. Per-state posterior histograms + assigned-state occupancy.
2. Per-state dwell-time distributions.
3. Per-state psychometric curves, empirical points with the fitted overlay.
4. Per-state chronometric curves (median RT by outcome and stimulus side).
5. Per-state posterior traces around L->R block switches.
6. The same around R->L block switches.

It writes a second figure per mouse, ``figures/ddm-hmm/{subject}_neural.svg``,
3x2, one row per ``config.RESPONSES`` window (baseline, stimulus, feedback):

1. The measure's change around entry into each state, one line per state.
2. Per-state violins of each session's correct-minus-incorrect difference in
   that measure, in units of the session's own spread.

It also writes one across-mouse figure, ``ddm_param_scatter.svg``: a 3D scatter
of every mouse's per-state ``B``, ``k`` and ``a0``, one point per (mouse, DDM
state) and one color per mouse.

The ``PhotometrySessionGroup`` is the source of truth for which sessions are in
scope, and the two per-mouse figures do not share one: the behavioral figures
take the fit's scope (``BEHAVIOR_QC_BLOCKERS``, no photometry QC), the neural
figures the responses analysis's, so a mouse whose recordings fail photometry
QC has a behavioral figure and no neural one. Each session's stored trials are
joined to its stored fit through ``group.process``; a session with no fit for
``DDM_HMM_K`` raises there and drops out. The fit's no-response state and
trials are removed here, never in the store.

No tables are persisted — every quantity recomputes at runtime.

Usage:
    python scripts/ddm_hmm_overview.py
"""
import argparse
from collections.abc import Callable, Hashable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from iblnm.config import (
    BEHAVIOR_QC_BLOCKERS, DDM_HMM_FIGURES_DIR, DDM_HMM_K, RESPONSES,
    SESSIONS_FPATH, SESSIONS_H5_DIR, STIM_ONSET_EVENT,
)
from iblnm.analysis import (
    align_traces_at_transitions, normalized_outcome_difference,
    state_dwell_times, transition_delta_stats,
)
from iblnm.data import PhotometrySession, PhotometrySessionGroup
from iblnm.io import _get_default_connection
from iblnm.task import fit_psychometric
from iblnm.vis import (
    plot_state_behavior, plot_state_neural, plot_state_param_scatter,
)

# probabilityLeft (prev, cur) pairs defining each block-transition type.
BLOCK_TRANSITIONS = {'L->R': (0.8, 0.2), 'R->L': (0.2, 0.8)}
BLOCK_WINDOW = 15  # half-window in trials around a transition (spec Decision)
BLOCK_BASELINE = 5  # trials before a transition defining the Δ-posterior baseline
SWITCH_WINDOW = 5  # half-window in trials around a state switch
SWITCH_BASELINE = 2  # trials before a switch defining the Δ-measure baseline
# feedbackType -> outcome label; splits the chronometric curves.
OUTCOMES = {'correct': 1, 'incorrect': -1}
# The scattered DDM parameters, in x/y/z order, with their plain-English names.
PARAM_LABELS = {'B': 'B (bound)', 'k': 'k (drift-rate gain)', 'a0': 'a₀ (bias)'}
# The `config.RESPONSES` entries measured per trial, in figure row order, with
# their plain-English names.
MEASURE_LABELS = {'baseline': 'pre-stimulus baseline',
                  'stimulus': 'stimulus response',
                  'feedback': 'feedback response'}
# Trials of each outcome a session x state cell needs to yield a violin point.
MIN_OUTCOME_TRIALS = 5
# What one measured row is, before the three measures are pivoted apart: one
# fiber's response on one trial.
MEASURE_KEYS = ['trial', 'target_NM', 'brain_region']


def load_session_states(ps: PhotometrySession, k: int) -> dict:
    """One session's stored trials joined to its stored K-state fit.

    For ``group.process``. The fit's per-trial columns join the stored trials
    on ``trial``; a column both carry keeps the trials' value, because the fit
    codes some of them differently (new-format ``choice`` is sign-flipped). A
    trial the fit does not hold is dropped by the join.

    Parameters
    ----------
    ps : PhotometrySession
        The session, addressing its file in the store.
    k : int
        Number of DDM states in the fit to read.

    Returns
    -------
    dict
        ``{'trials': DataFrame, 'attrs': dict}``: the joined per-trial frame,
        carrying ``eid`` and ``subject``, and the fit's run summary and
        parameters.

    Raises
    ------
    KeyError
        The session holds no fit for ``k`` (from ``load_hmm``).
    """
    ps.load_h5(groups=['trials'])
    fit = ps.load_hmm(k)
    fit_columns = ['trial', *fit['trials'].columns.difference(ps.trials.columns)]
    frame = ps.trials.merge(fit['trials'][fit_columns], on='trial')
    return {'trials': frame.assign(eid=ps.eid, subject=ps.subject),
            'attrs': fit['attrs']}


def load_session_measures(ps: PhotometrySession, k: int) -> dict:
    """One session's trials, its stored fit, and its three response measures.

    For ``group.process``. Extends :func:`load_session_states` with one column
    per :data:`MEASURE_LABELS` entry, measured off the stored photometry
    response cut with that ``config.RESPONSES`` entry's window, masking
    chronology and baseline rule — the measurement
    ``scripts/responses.py`` fits its models on. A session recording from two
    fibers contributes each trial twice, once per fiber, since the measures are
    the fiber's and the trial columns are the session's.

    Parameters
    ----------
    ps : PhotometrySession
        The session, addressing its file in the store.
    k : int
        Number of DDM states in the fit to read.

    Returns
    -------
    dict
        ``{'trials': DataFrame, 'attrs': dict}``: the joined per-trial frame,
        one row per fiber x trial and carrying ``brain_region`` and
        ``target_NM``, and the fit's run summary and parameters.

    Raises
    ------
    KeyError
        The session holds no fit for ``k`` (from ``load_hmm``).
    """
    fit = load_session_states(ps, k)
    ps.load_responses('photometry')
    magnitudes = []
    for measure, entry in ((measure, RESPONSES[measure])
                           for measure in MEASURE_LABELS):
        # Each call overwrites the session's magnitudes, so the columns this
        # one needs are taken before the next entry is measured.
        measured = ps.extract_response_magnitudes(
            entry['window'], entry['masking_events'], entry['baseline_correct'],
            events=[entry['event']])
        magnitudes.append(measured[[*MEASURE_KEYS, 'response']]
                          .assign(measure=measure))
    measures = (pd.concat(magnitudes, ignore_index=True)
                .pivot(index=MEASURE_KEYS, columns='measure', values='response')
                .reset_index())
    return {'trials': fit['trials'].merge(measures, on='trial'),
            'attrs': fit['attrs']}


def behavioral_frame(
    fit_trials: pd.DataFrame, attrs: dict
) -> tuple[pd.DataFrame, list[int]]:
    """Strip the no-response state from one session's joined fit frame.

    Parameters
    ----------
    fit_trials : pandas.DataFrame
        One session's trials joined to its fit (``load_session_states``).
        New-format fits carry ``omission`` and ``viterbi_state``; old-format
        fits carry ``map_state`` and no no-response state.
    attrs : dict
        The fit's attrs: a per-state ``state`` array and, new format only, a
        matching ``kind`` array (``'ddm'`` or ``'omission'``).

    Returns
    -------
    frame : pandas.DataFrame
        No-response trials dropped; ``state`` holds the assigned state
        (``viterbi_state`` where the fit carries it, else ``map_state``); the
        no-response state's ``p_state_{i}`` column dropped.
    labels : list of int
        The DDM states, ascending.
    """
    # Old-format fits carry no `kind`, and so no no-response state.
    no_response = [int(state) for state, kind
                   in zip(attrs['state'], attrs.get('kind', []))
                   if kind == 'omission']
    if 'omission' in fit_trials:
        fit_trials = fit_trials[~fit_trials['omission'].astype(bool)]
    assigned = 'viterbi_state' if 'viterbi_state' in fit_trials else 'map_state'
    frame = fit_trials.drop(columns=[f'p_state_{state}' for state in no_response])
    frame = frame.assign(state=frame[assigned].astype(int))
    labels = sorted(int(state) for state in attrs['state']
                    if state not in no_response)
    return frame, labels


def state_params(subject: str, attrs: dict) -> pd.DataFrame:
    """One mouse's per-state DDM parameters, the no-response state removed.

    Parameters
    ----------
    subject : str
        The mouse the fit belongs to; becomes the ``mouse`` column.
    attrs : dict
        The fit's attrs, carrying one array per parameter indexed by state:
        ``state`` and the :data:`PARAM_LABELS` keys, plus, new format only, a
        matching ``kind`` array (``'ddm'`` or ``'omission'``). The no-response
        state's DDM parameters are NaN, so it is dropped; old-format attrs
        carry no ``kind`` and no such state, and keep every row.

    Returns
    -------
    pandas.DataFrame
        One row per DDM state, columns ``['mouse', 'state', *PARAM_LABELS]``.
    """
    frame = pd.DataFrame({'mouse': subject, 'state': attrs['state'],
                          **{param: attrs[param] for param in PARAM_LABELS}})
    kind = attrs.get('kind')
    return frame if kind is None else frame[kind != 'omission'].reset_index(drop=True)


def build_state_param_table(mouse_frame: pd.DataFrame) -> pd.DataFrame:
    """Per-state psychometric fit for one mouse.

    Groups the mouse's trials by assigned state and fits, per state, a
    psychometric function (via :func:`fit_psychometric`, pooling across
    ``probabilityLeft`` blocks). Feeds the fitted overlay of the psychometric
    panel.

    Parameters
    ----------
    mouse_frame : pandas.DataFrame
        One mouse's concatenated :func:`behavioral_frame` output. Must carry
        ``state``, ``choice``, and the ``contrastLeft``/``contrastRight``
        columns ``fit_psychometric`` reads, which the stored ``trials/table``
        holds verbatim.

    Returns
    -------
    pandas.DataFrame
        One row per state, columns ``['state', 'bias', 'threshold', 'lapse_left',
        'lapse_right']``. NaN where a state has too few trials to fit.
    """
    rows = []
    for state, trials in mouse_frame.groupby('state'):
        psych = fit_psychometric(trials)
        rows.append({
            'state': int(state),
            'bias': psych['bias'],
            'threshold': psych['threshold'],
            'lapse_left': psych['lapse_left'],
            'lapse_right': psych['lapse_right'],
        })
    return pd.DataFrame(rows)


def _state_curves(
    mouse_frame: pd.DataFrame, param_table: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Assemble one mouse's per-state psychometric + chronometric plot frames.

    Builds the long frames :func:`iblnm.vis.draw_state_psychometric` and
    :func:`iblnm.vis.draw_state_chronometric` consume: the psychometric frame
    combines empirical P(choose right) per signed contrast with the fitted
    params from ``param_table``; the chronometric frame is the empirical median
    RT per signed contrast, split by trial outcome correct/incorrect and by
    stimulus side (drawn as plain lines, no fit; sides are not connected).

    Returns
    -------
    dict of str to pandas.DataFrame
        ``{'psychometric', 'chronometric'}``; see the drawers' docstrings for
        the column layout.
    """
    params = param_table.set_index('state')
    psych_rows, chrono_rows = [], []
    for state, trials in mouse_frame.groupby('state'):
        state = int(state)
        p = params.loc[state]
        p_right = (trials['choice'] == -1).groupby(
            trials['signed_contrast']).mean()
        psych_rows += [
            {'state': state, 'signed_contrast': sc, 'p_right': pr,
             'bias': p['bias'], 'threshold': p['threshold'],
             'lapse_left': p['lapse_left'], 'lapse_right': p['lapse_right']}
            for sc, pr in p_right.items()
        ]
        for outcome, feedback in OUTCOMES.items():
            outcome_trials = trials[trials['feedbackType'] == feedback]
            # Split by stimulus side so left/right lines are not connected and
            # zero contrast keeps a separate point per side (two dots at 0).
            for side in ('left', 'right'):
                side_trials = outcome_trials[outcome_trials['stim_side'] == side]
                median_rt = side_trials['rt'].groupby(
                    side_trials['signed_contrast']).median()
                chrono_rows += [
                    {'state': state, 'outcome': outcome, 'side': side,
                     'signed_contrast': sc, 'median_rt': rt}
                    for sc, rt in median_rt.items()
                ]
    return {'psychometric': pd.DataFrame(psych_rows),
            'chronometric': pd.DataFrame(chrono_rows)}


def _block_transition_indexers() -> dict[str, Callable[[pd.DataFrame], np.ndarray]]:
    """Block-transition indexers, one per :data:`BLOCK_TRANSITIONS` entry.

    Each indexer takes one eid's sub-frame and returns the positional row indices
    of the trials at which ``probabilityLeft`` steps from ``prev`` to ``cur``. The
    index is that of the first trial of the new block, so the initial 0.5 block
    never triggers a transition.
    """
    def indexer(prev: float, cur: float) -> Callable[[pd.DataFrame], np.ndarray]:
        def find_transitions(eid_df: pd.DataFrame) -> np.ndarray:
            p_left = eid_df['probabilityLeft'].to_numpy()
            return np.flatnonzero((p_left[:-1] == prev) & (p_left[1:] == cur)) + 1
        return find_transitions

    return {label: indexer(prev, cur)
            for label, (prev, cur) in BLOCK_TRANSITIONS.items()}


def _transition_traces(
    frame: pd.DataFrame,
    value_cols: list[str],
    groups: dict[Hashable, Callable[[pd.DataFrame], np.ndarray]],
    window: int,
    baseline: int,
) -> dict[Hashable, dict[str, np.ndarray]]:
    """Δ traces (mean + SEM) around each group's transitions, for one mouse.

    Slices ``±window`` rows of ``value_cols`` around every transition each group's
    indexer reports, pooling windows across the mouse's eids. Each window is
    expressed as a change from its own pre-transition baseline — the mean over the
    ``baseline`` rows just before the transition (lags ``-baseline … -1``) — then
    averaged across transitions, with the standard error of that mean.

    Detection is per eid, so no transition is ever found across a session
    boundary. Rows are used in the order they appear in ``frame``; the caller
    restricts which rows are present (e.g. to fit trials) before calling.

    Parameters
    ----------
    frame : pandas.DataFrame
        One mouse's trials, in trial order within each eid, carrying ``eid``,
        ``value_cols`` and whatever columns the indexers read.
    value_cols : list of str
        Columns sliced into windows; they become the trailing axis of the result.
    groups : dict of hashable to callable
        Group label -> a function taking one eid's sub-frame and returning the
        positional row indices of that group's transitions within it.
    window : int
        Half-window in trials around each transition.
    baseline : int
        Number of pre-transition trials averaged as each window's own zero.

    Returns
    -------
    dict of hashable to dict of str to numpy.ndarray
        ``{group_label: {'mean': arr, 'sem': arr}}`` with each ``arr`` of shape
        ``(2*window+1, len(value_cols))``; group labels with no transition in any
        eid are absent.
    """
    collected = {label: [] for label in groups}
    for _, eid_df in frame.groupby('eid'):
        values = eid_df[value_cols].to_numpy()
        for label, find_transitions in groups.items():
            idx = find_transitions(eid_df)
            if len(idx):
                collected[label].append(
                    align_traces_at_transitions(values, idx, window))

    return {
        label: transition_delta_stats(
            np.concatenate(windows, axis=0),  # (n_transitions, 2w+1, n_cols)
            baseline)
        for label, windows in collected.items() if windows
    }


def _state_switch_indexers(
    states: list[int],
) -> dict[int, Callable[[pd.DataFrame], np.ndarray]]:
    """State-switch indexers, one per entered state.

    Each indexer takes one eid's sub-frame and returns the positional row indices
    of the trials on which ``state`` changes into that state. The first row of
    a sub-frame is never one, so a session boundary is never a switch, and the
    origin state does not enter the key — every switch into a state is pooled
    regardless of where it came from.
    """
    def indexer(state: int) -> Callable[[pd.DataFrame], np.ndarray]:
        def find_switches(eid_df: pd.DataFrame) -> np.ndarray:
            assigned = eid_df['state'].to_numpy()
            return np.flatnonzero(
                (assigned[1:] != assigned[:-1]) & (assigned[1:] == state)) + 1
        return find_switches

    return {state: indexer(state) for state in states}


def _state_switch_traces(
    frame: pd.DataFrame, measures: list[str], window: int, baseline: int
) -> dict[str, dict[str, np.ndarray]]:
    """Δ traces of per-trial measures around entry into each state, for one mouse.

    Runs :func:`_transition_traces` over ``measures`` with one group per state
    present in ``frame``, then transposes its per-state result into a
    per-measure layout: one panel per measure, one line per entered state.

    Parameters
    ----------
    frame : pandas.DataFrame
        One mouse's fit trials, in trial order within each eid, carrying
        ``eid``, ``state`` and the ``measures`` columns.
    measures : list of str
        Measure columns, in the order the result is keyed.
    window : int
        Half-window in trials around each switch.
    baseline : int
        Number of pre-switch trials averaged as each window's own zero.

    Returns
    -------
    dict of str to dict of str to numpy.ndarray
        ``{measure: {'mean': arr, 'sem': arr}}`` with each ``arr`` of shape
        ``(2*window+1, n_entered_states)``, states stacked in ascending order so
        line colors follow state order. States never entered contribute no
        column; a mouse with no switch at all returns an empty dict.
    """
    states = sorted(frame['state'].unique().astype(int))
    traces = _transition_traces(frame, measures,
                                _state_switch_indexers(states), window, baseline)
    entered = [state for state in states if state in traces]
    if not entered:
        return {}
    return {
        measure: {stat: np.stack([traces[state][stat][:, col]
                                  for state in entered], axis=1)
                  for stat in ('mean', 'sem')}
        for col, measure in enumerate(measures)
    }


def _unconverged(attrs: dict) -> bool:
    """Whether a fit reports it did not converge; a blank (NaN) one does not."""
    converged = attrs.get('converged', np.nan)
    return not pd.isna(converged) and not converged


def behavior_panels(frame: pd.DataFrame) -> dict:
    """One mouse's inputs to :func:`iblnm.vis.plot_state_behavior`.

    Parameters
    ----------
    frame : pandas.DataFrame
        The mouse's concatenated :func:`behavioral_frame` output, in trial
        order within each eid.

    Returns
    -------
    dict
        ``states`` (``state`` + posterior columns), ``dwell`` (run lengths per
        eid), ``curves`` (psychometric + chronometric frames) and
        ``block_traces`` (each :data:`BLOCK_TRANSITIONS` label, in order, to its
        Δ-posterior traces, or ``None`` when the mouse never made it).
    """
    # The chronometric curves' RT, on the pipeline's onset clock rather than
    # the fit's own `rt`.
    frame = frame.assign(rt=frame['response_times'] - frame[STIM_ONSET_EVENT])
    posterior_cols = [c for c in frame.columns if c.startswith('p_state_')]
    traces = _transition_traces(frame, posterior_cols,
                                _block_transition_indexers(),
                                BLOCK_WINDOW, BLOCK_BASELINE)
    return {
        'states': frame[['state', *posterior_cols]],
        'dwell': state_dwell_times(frame['state'].to_numpy(),
                                   frame['eid'].to_numpy()),
        'curves': _state_curves(frame, build_state_param_table(frame)),
        'block_traces': {label: traces.get(label) for label in BLOCK_TRANSITIONS},
    }


def neural_panels(frame: pd.DataFrame) -> dict:
    """One mouse's inputs to :func:`iblnm.vis.plot_state_neural`.

    Parameters
    ----------
    frame : pandas.DataFrame
        The mouse's concatenated :func:`behavioral_frame` output, built from
        :func:`load_session_measures`, in trial order within each eid.

    Returns
    -------
    dict
        ``traces`` (each measure's Δ around entry into a state, one line per
        entered state) and ``differences`` (each measure's per-session
        correct-minus-incorrect difference, in units of the session x state
        cell's own SD).
    """
    measures = list(MEASURE_LABELS)
    return {
        'traces': _state_switch_traces(frame, measures, SWITCH_WINDOW,
                                       SWITCH_BASELINE),
        'differences': {
            measure: normalized_outcome_difference(
                frame, measure, ['eid', 'state'], 'feedbackType', 1, -1,
                MIN_OUTCOME_TRIALS)
            for measure in measures},
    }


def _save(fig: plt.Figure, name: str) -> None:
    """Save ``fig`` to ``DDM_HMM_FIGURES_DIR/{name}.svg`` and close it."""
    DDM_HMM_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(DDM_HMM_FIGURES_DIR / f'{name}.svg', bbox_inches='tight')
    plt.close(fig)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args(argv)


def _by_subject(results: list[dict]) -> dict[str, list[dict]]:
    """Group ``process`` results by the mouse their trials belong to.

    Mice come out in the order the store returned them; each mouse's sessions
    keep their relative order, which is the order their trials concatenate in.
    """
    subjects = dict.fromkeys(result['trials']['subject'].iloc[0]
                             for result in results)
    return {subject: [result for result in results
                      if result['trials']['subject'].iloc[0] == subject]
            for subject in subjects}


def main(one=None) -> None:
    """Render each mouse's behavioral and neural K=`DDM_HMM_K` figures.

    Two passes over one group, differing in scope: the behavioral figures take
    every session the fit covers (``BEHAVIOR_QC_BLOCKERS``, no photometry QC),
    the neural figures only those the responses analysis admits, so a mouse's
    two figures need not rest on the same sessions and a mouse may have no
    neural figure at all.

    Parameters
    ----------
    one : ONE, optional
        Connection for offline H5 access; a default read-only connection is
        created when omitted.
    """
    if one is None:
        one = _get_default_connection()
    group = PhotometrySessionGroup.from_catalog(
        pd.read_parquet(SESSIONS_FPATH), one=one, h5_dir=SESSIONS_H5_DIR)
    group.filter_sessions(session_types=('biased', 'ephys'),
                          qc_blockers=BEHAVIOR_QC_BLOCKERS, photometry_qc=False)
    group.deduplicate()
    fits = [fit for fit in group.process(load_session_states, k=DDM_HMM_K)
            if fit is not None]

    params = []
    by_subject = _by_subject(fits)
    for subject, mouse_fits in by_subject.items():
        if any(_unconverged(fit['attrs']) for fit in mouse_fits):
            print(f"WARNING {subject}: K={DDM_HMM_K} fit did not converge")
        frame = pd.concat([behavioral_frame(fit['trials'], fit['attrs'])[0]
                           for fit in mouse_fits], ignore_index=True)
        panels = behavior_panels(frame)
        _save(plot_state_behavior(subject, panels['states'], panels['dwell'],
                                  panels['curves'], panels['block_traces']),
              f'{subject}_behavior')
        # Every session of a mouse carries the same fit's attrs.
        params.append(state_params(subject, mouse_fits[0]['attrs']))
        print(f"  {subject}: {len(mouse_fits)} sessions, {len(frame)} trials")

    _save(plot_state_param_scatter(pd.concat(params, ignore_index=True),
                                   PARAM_LABELS), 'ddm_param_scatter')

    # The responses analysis's scope: the target-NM and photometry-QC filters
    # the behavioral pass switches off, since the measures are read off the
    # recordings those filters admit.
    group.filter_sessions(session_types=('biased', 'ephys'))
    group.deduplicate()
    measured = [result
                for result in group.process(load_session_measures, k=DDM_HMM_K)
                if result is not None]

    measured_by_subject = _by_subject(measured)
    for subject, sessions in measured_by_subject.items():
        frame = pd.concat([behavioral_frame(session['trials'],
                                            session['attrs'])[0]
                           for session in sessions], ignore_index=True)
        panels = neural_panels(frame)
        targets = ', '.join(dict.fromkeys(frame['target_NM']))
        _save(plot_state_neural(f'{subject} {targets}', panels['traces'],
                                panels['differences'], MEASURE_LABELS,
                                SWITCH_WINDOW),
              f'{subject}_neural')
        print(f"  {subject}: {len(sessions)} measured sessions")

    print(f"Wrote {len(by_subject) + len(measured_by_subject) + 1} figures "
          f"to {DDM_HMM_FIGURES_DIR}")


if __name__ == '__main__':
    parse_args()
    main()
