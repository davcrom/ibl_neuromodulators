"""DDM-HMM first-look overview.

Runs the first look at a collaborator's per-mouse drift-diffusion + hidden-Markov
model (DDM-HMM) fit to the choice/RT behavior of the 8 LC-NE mice, and writes six
figures to ``figures/ddm-hmm/``:

1. Per-state posterior histograms + MAP occupancy and state dwell-time
   distributions (per mouse).
2. Per-state psychometric + chronometric curves (per mouse).
3. Per-state DDM-parameter pairwise scatter (all mice, colored by mouse).
4. PCA of per-state behavioral-parameter features (all mice, colored by mouse).
5. Per-state posterior traces around block transitions (per mouse).
6. Per-state NM distributions — pre-stimulus baseline, stimulus-onset response
   and feedback response — split into correct and incorrect trials (per mouse).

The ``PhotometrySessionGroup`` is the source of truth for which sessions are in
scope: each mouse's trial+state frame is assembled by filtering the group to that
subject, loading H5 trials offline, and attaching the fitted per-trial states via
``PhotometrySession.load_states``. Behavioral parameters and empirical curves are
computed here (variable-specific analysis belongs in the script); the reusable
computations live in ``iblnm.analysis``/``iblnm.task``/``iblnm.vis``.

No tables are persisted — every quantity recomputes at runtime.

Usage:
    python scripts/ddm_hmm_overview.py            # all modeled mice
"""
import warnings
from collections.abc import Callable, Hashable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from iblnm.config import (
    SESSIONS_FPATH, SESSIONS_H5_DIR, DDM_HMM_PARAMS_FPATH, DDM_HMM_FIGURES_DIR,
    RESPONSE_EVENTS, RESPONSE_WINDOW, RESPONSE_WINDOWS,
)
from iblnm.analysis import (
    align_traces_at_transitions, compute_response_magnitude, pca_2d,
    transition_delta_stats,
    state_dwell_times,
)
from iblnm.data import PhotometrySession, PhotometrySessionGroup
from iblnm.io import _get_default_connection
from iblnm.task import (
    fit_psychometric, reconstruct_contrast_sides,
)
from iblnm.vis import (
    plot_state_block_transitions, plot_state_measures, plot_state_param_scatter,
    plot_state_pca, plot_state_posterior_dwell,
    plot_state_psychometric_chronometric,
)

# Behavioral-parameter features feeding the figure-4 PCA (one per state).
FEATURE_COLS = ['bias', 'threshold', 'lapse_left', 'lapse_right']
# probabilityLeft (prev, cur) pairs defining each block-transition type (figure 5).
BLOCK_TRANSITIONS = {'L->R': (0.8, 0.2), 'R->L': (0.2, 0.8)}
BLOCK_WINDOW = 15  # half-window in trials around a transition (spec Decision)
BLOCK_BASELINE = 5  # trials before a transition defining the Δ-posterior baseline
# feedbackType -> outcome label; splits the chronometric curves (figure 2).
OUTCOMES = {'correct': 1, 'incorrect': -1}
# Pre-stimulus NM baseline window, s relative to stimOn_times (figure 6). Not
# config.BASELINE_WINDOW, which is (-0.1, 0) and serves evoked-response
# subtraction — a different quantity.
NM_BASELINE_WINDOW = [-0.4, -0.1]
# Per-trial NM measure -> y-axis label (figure 6). Iteration order fixes the
# figure's left-to-right panel order.
MEASURE_LABELS = {
    'baseline': 'pre-stim baseline (session SD)',
    'stimOn_response': 'stimOn response (Δ session SD)',
    'feedback_response': 'feedback response (Δ session SD)',
}


def _evoked_magnitudes(
    ps: PhotometrySession, signals: pd.DataFrame, column: str
) -> pd.DataFrame:
    """Per-trial evoked response magnitudes for one session's single fiber.

    Runs the project's canonical evoked path on ``signals[column]``: peri-event
    matrices over ``RESPONSE_WINDOW``, samples later than the trial's next event
    masked out, per-trial pre-event baseline subtracted, then averaged over
    ``RESPONSE_WINDOWS['early']``. ``mask_subsequent_events`` masks only the
    non-terminal events, so a trial whose feedback lands inside the early window
    averages the surviving samples only, and one whose feedback precedes the
    window start leaves it empty and yields NaN — hence the suppressed
    all-NaN-slice ``RuntimeWarning``.

    Returns
    -------
    pandas.DataFrame
        Columns ``stimOn_response`` and ``feedback_response``, in session-SD
        units, indexed by ``ps.trials.index``.
    """
    responses = ps.extract_responses(
        signals, events=RESPONSE_EVENTS, window=RESPONSE_WINDOW)[column]
    evoked = ps.subtract_baseline(ps.mask_subsequent_events(responses))
    tpts = evoked.coords['time'].values
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        magnitudes = {
            f"{event.removesuffix('_times')}_response": compute_response_magnitude(
                evoked.sel(event=event).values, tpts, RESPONSE_WINDOWS['early'])
            for event in RESPONSE_EVENTS
        }
    return pd.DataFrame(magnitudes, index=evoked.coords['trial'].values)


def build_mouse_states_frame(
    group: PhotometrySessionGroup, subject: str, one
) -> pd.DataFrame:
    """Concatenate one mouse's per-session trials + fitted states into one frame.

    Filters ``group`` to ``subject``, then for each of its sessions instantiates a
    :class:`PhotometrySession`, loads the H5 ``trials`` and ``photometry`` groups
    offline, and attaches the fitted per-trial states via
    :meth:`PhotometrySession.load_states`. Sessions absent from the fit
    (``states is None``) are dropped. The surviving per-session frames — trials
    joined with their state columns and the three per-trial NM measures, tagged
    with ``eid`` — are concatenated in session order.

    One fiber per mouse: a session yields measures only when its data and its
    metadata agree on exactly one fiber — ``GCaMP_preprocessed`` has one column
    and ``brain_region`` one entry. The bilateral sessions name their columns
    ``LC-l``/``LC-r`` (which is why the column, not ``brain_region[0]``, selects
    the signal), and some sessions carry a duplicated ``['LC', 'LC']`` against a
    single column. Both are ambiguous, so all three measures are NaN and the
    trials are kept, since they still feed the behavioral figures.

    Parameters
    ----------
    group : PhotometrySessionGroup
        Filtered group; its ``sessions`` view supplies the subject's session rows.
    subject : str
        Mouse nickname to assemble.
    one : ONE
        Connection passed through to each :class:`PhotometrySession`.

    Returns
    -------
    pandas.DataFrame
        Full trials columns plus ``map_state``/``state_1``…``state_K`` (NaN on
        trials dropped from the fit), three per-trial NM measure columns in
        session-SD units, and an ``eid`` column, one row per trial across the
        mouse's fit sessions. ``baseline`` is the mean of the session's
        preprocessed signal over :data:`NM_BASELINE_WINDOW` before
        ``stimOn_times``; ``stimOn_response`` and ``feedback_response`` are the
        baseline-subtracted evoked magnitudes from :func:`_evoked_magnitudes`.
        All three are NaN where their window runs off the recording or the
        session's fiber was ambiguous. Empty when no session was in the fit.
    """
    rows = group.sessions[group.sessions['subject'] == subject]
    frames = []
    for _, row in rows.iterrows():
        ps = PhotometrySession(row, one=one)
        ps.load_h5(groups=['trials', 'photometry'])
        if ps.trials is None or ps.trials.empty:
            print(f"  {ps.eid}: no stored trials — skipped")
            continue
        ps.load_states()
        if ps.states is None:
            continue
        frame = ps.trials.join(ps.states)
        signals = ps.photometry['GCaMP_preprocessed']
        if len(signals.columns) == 1 and len(ps.brain_region) == 1:
            column = signals.columns[0]
            responses = ps.extract_responses(
                signals, events=['stimOn_times'], window=NM_BASELINE_WINDOW,
            )
            frame['baseline'] = responses[column].sel(
                event='stimOn_times').mean('time').to_series()
            frame = frame.join(_evoked_magnitudes(ps, signals, column))
        else:
            print(f"  {ps.eid}: {len(signals.columns)} photometry columns, "
                  f"{len(ps.brain_region)} brain regions — no measures")
            frame[['baseline', 'stimOn_response', 'feedback_response']] = np.nan
        frame['eid'] = ps.eid
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_state_param_table(mouse_frame: pd.DataFrame) -> pd.DataFrame:
    """Per-state behavioral-parameter table for one mouse.

    Groups the mouse's trials by MAP state and fits, per state, a psychometric
    function (via :func:`fit_psychometric`, pooling across ``probabilityLeft``
    blocks). ``contrastLeft``/``contrastRight`` are reconstructed from
    ``stim_side`` and ``contrast`` (:func:`reconstruct_contrast_sides`) as
    ``fit_psychometric`` requires. Feeds figure 2 (curve overlays) and figure 4
    (PCA features).

    Parameters
    ----------
    mouse_frame : pandas.DataFrame
        One mouse's concatenated trials + states (from
        :func:`build_mouse_states_frame`). Must carry ``map_state``, ``choice``,
        ``stim_side`` and ``contrast``. Trials dropped from the fit
        (``map_state`` NaN) are ignored by the ``groupby``.

    Returns
    -------
    pandas.DataFrame
        One row per state, columns ``['state', 'bias', 'threshold', 'lapse_left',
        'lapse_right']`` — the psychometric fit. NaN where a state has too few
        trials to fit.
    """
    rows = []
    for state, trials in mouse_frame.groupby('map_state'):
        trials = trials.join(reconstruct_contrast_sides(trials))
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

    Builds the long frames :func:`plot_state_psychometric_chronometric` consumes:
    the psychometric frame combines empirical P(choose right) per signed contrast
    with the fitted params from ``param_table``; the chronometric frame is the
    empirical median RT per signed contrast, split by trial outcome
    correct/incorrect and by stimulus side (drawn as plain lines, no fit; sides
    are not connected).

    Returns
    -------
    dict of str to pandas.DataFrame
        ``{'psychometric', 'chronometric'}``; see the plotter's docstring for the
        column layout.
    """
    params = param_table.set_index('state')
    psych_rows, chrono_rows = [], []
    for state, trials in mouse_frame.groupby('map_state'):
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
    """Figure 5's transition indexers, one per :data:`BLOCK_TRANSITIONS` entry.

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


def _save(fig: plt.Figure, name: str) -> None:
    """Save ``fig`` to ``DDM_HMM_FIGURES_DIR/{name}.svg`` and close it."""
    DDM_HMM_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(DDM_HMM_FIGURES_DIR / f'{name}.svg', bbox_inches='tight')
    plt.close(fig)


def _assemble_mouse_views(
    group: PhotometrySessionGroup, subjects: list[str], one
) -> dict:
    """Build every modeled mouse's plot inputs from the filtered group.

    Iterates ``subjects``, assembling each one's trials+states frame and deriving
    the per-mouse inputs for figures 1, 2, 5 and 6 plus its per-state feature rows.
    Mice with no session in the fit are skipped.

    Returns
    -------
    dict
        Keys ``'states'``, ``'dwell'``, ``'curves'``, ``'aligned'``,
        ``'measures'`` each map subject to that figure's plot input;
        ``'measures'`` holds the fit-only, non-no-go trials carrying at least one
        of :data:`MEASURE_LABELS`, as ``['state', 'eid', 'outcome']`` plus one
        column per measure, and omits a mouse whose every session had an
        ambiguous fiber — such a mouse still appears in the other views.
        ``'features'`` is the concatenated per-state behavioral-feature
        table (with a ``mouse`` column) for the PCA.

    Raises
    ------
    ValueError
        If every subject was skipped, leaving nothing to plot.
    """
    views = {key: {}
             for key in ('states', 'dwell', 'curves', 'aligned', 'measures')}
    param_tables = []
    feedback2outcome = {feedback: label for label, feedback in OUTCOMES.items()}
    for subject in subjects:
        frame = build_mouse_states_frame(group, subject, one)
        if frame.empty:
            print(f"  {subject}: no fit sessions in group — skipped")
            continue
        frame['rt'] = frame['response_times'] - frame['stimOn_times']
        posterior_cols = [c for c in frame.columns if c.startswith('state_')]
        views['states'][subject] = frame[['map_state', *posterior_cols]]

        kept = frame[frame['map_state'].notna()]
        views['dwell'][subject] = state_dwell_times(
            kept['map_state'].astype(int).to_numpy(), kept['eid'].to_numpy())
        measured = (
            kept[kept['choice'] != 0]
            .dropna(subset=list(MEASURE_LABELS), how='all')
            .assign(state=lambda df: df['map_state'].astype(int),
                    outcome=lambda df: df['feedbackType'].map(feedback2outcome))
            .dropna(subset=['outcome'])
        )
        if not measured.empty:
            views['measures'][subject] = measured[
                ['state', 'eid', 'outcome', *MEASURE_LABELS]]

        param_table = build_state_param_table(frame)
        param_table['mouse'] = subject
        param_tables.append(param_table)
        views['curves'][subject] = _state_curves(frame, param_table)
        views['aligned'][subject] = _transition_traces(
            kept, posterior_cols, _block_transition_indexers(),
            BLOCK_WINDOW, BLOCK_BASELINE)
        print(f"  {subject}: {len(kept)} fit trials, "
              f"{param_table['state'].nunique()} states")

    if not param_tables:
        raise ValueError(
            f"no modeled mouse survived the group filter (tried {subjects})")
    views['features'] = pd.concat(param_tables, ignore_index=True)
    return views


def main(one=None) -> None:
    """Assemble every mouse's frame and render the six overview figures.

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
    group.filter_sessions()

    ddm_params = pd.read_csv(DDM_HMM_PARAMS_FPATH)
    subjects = list(dict.fromkeys(ddm_params['mouse']))
    views = _assemble_mouse_views(group, subjects, one)

    _save(plot_state_posterior_dwell(views['states'], views['dwell']),
          'posteriors_dwell')
    _save(plot_state_psychometric_chronometric(views['curves']),
          'psychometric_chronometric')
    _save(plot_state_param_scatter(ddm_params), 'ddm_param_scatter')

    features = views['features'].dropna(subset=FEATURE_COLS)
    scores, loadings = pca_2d(features[FEATURE_COLS].to_numpy())
    _save(plot_state_pca(scores, features['mouse'], features['state'],
                         loadings, FEATURE_COLS), 'behavioral_pca')

    _save(plot_state_block_transitions(views['aligned'], BLOCK_TRANSITIONS,
                                       BLOCK_WINDOW), 'block_transitions')
    _save(plot_state_measures(views['measures'], MEASURE_LABELS),
          'state_measures')
    print(f"Wrote 6 figures to {DDM_HMM_FIGURES_DIR}")


if __name__ == '__main__':
    main()
