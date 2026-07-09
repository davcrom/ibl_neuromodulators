"""DDM-HMM first-look overview.

Runs the first look at a collaborator's per-mouse drift-diffusion + hidden-Markov
model (DDM-HMM) fit to the choice/RT behavior of the 8 LC-NE mice, and writes six
figures to ``figures/ddm-hmm/``:

1. Per-state posterior histograms + MAP occupancy (per mouse).
2. State dwell-time distributions (per mouse).
3. Per-state psychometric + chronometric curves (per mouse).
4a. Per-state DDM-parameter pairwise scatter (all mice, colored by mouse).
4b. PCA of per-state behavioral-parameter features (all mice, colored by mouse).
5. Per-state posterior traces around block transitions (per mouse).

The ``PhotometrySessionGroup`` is the source of truth for which sessions are in
scope: each mouse's trial+state frame is assembled by filtering the group to that
subject, loading H5 trials offline, and attaching the fitted per-trial states via
``PhotometrySession.load_states``. Behavioral parameters and empirical curves are
computed here (variable-specific analysis belongs in the script); the reusable
computations live in ``iblnm.analysis``/``iblnm.task``/``iblnm.vis``.

No tables are persisted — every quantity recomputes at runtime.

Usage:
    python scripts/ddm_hmm_overview.py            # all modeled mice
    python scripts/ddm_hmm_overview.py --window 20  # block-transition half-window
"""
import argparse
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from iblnm.config import (
    SESSIONS_FPATH, SESSIONS_H5_DIR, DDM_HMM_PARAMS_FPATH, DDM_HMM_FIGURES_DIR,
    FIGURE_DPI,
)
from iblnm.analysis import align_traces_at_transitions, pca_2d, state_dwell_times
from iblnm.data import PhotometrySession, PhotometrySessionGroup
from iblnm.io import _get_default_connection
from iblnm.task import (
    fit_chronometric, fit_psychometric, reconstruct_contrast_sides,
)
from iblnm.vis import (
    plot_state_block_transitions, plot_state_dwell_times, plot_state_param_scatter,
    plot_state_pca, plot_state_posterior_histograms,
    plot_state_psychometric_chronometric,
)

# Behavioral-parameter features feeding the goal-4b PCA (one per state).
FEATURE_COLS = ['bias', 'threshold', 'lapse_left', 'lapse_right', 'rt_slope']
# probabilityLeft (prev, cur) pairs defining each block-transition type (goal 5).
BLOCK_TRANSITIONS = {'L->R': (0.8, 0.2), 'R->L': (0.2, 0.8)}
BLOCK_WINDOW = 15  # half-window in trials around a transition (spec Decision)


def build_mouse_states_frame(
    group: PhotometrySessionGroup, subject: str, one
) -> pd.DataFrame:
    """Concatenate one mouse's per-session trials + fitted states into one frame.

    Filters ``group`` to ``subject``, then for each of its sessions instantiates a
    :class:`PhotometrySession`, loads the H5 ``trials`` group offline, and attaches
    the fitted per-trial states via :meth:`PhotometrySession.load_states`. Sessions
    absent from the fit (``states is None``) are dropped. The surviving per-session
    frames — trials joined with their state columns, tagged with ``eid`` — are
    concatenated in session order.

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
        trials dropped from the fit) plus an ``eid`` column, one row per trial
        across the mouse's fit sessions. Empty when no session was in the fit.
    """
    rows = group.sessions[group.sessions['subject'] == subject]
    frames = []
    for _, row in rows.iterrows():
        ps = PhotometrySession(row, one=one)
        ps.load_h5(groups=['trials'])
        if ps.trials is None or ps.trials.empty:
            print(f"  {ps.eid}: no stored trials — skipped")
            continue
        ps.load_states()
        if ps.states is None:
            continue
        frame = ps.trials.join(ps.states)
        frame['eid'] = ps.eid
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_state_param_table(mouse_frame: pd.DataFrame) -> pd.DataFrame:
    """Per-state behavioral-parameter table for one mouse.

    Groups the mouse's trials by MAP state and fits, per state, a psychometric
    function (via :func:`fit_psychometric`, pooling across ``probabilityLeft``
    blocks) and a chronometric median-RT slope (via :func:`fit_chronometric`).
    ``contrastLeft``/``contrastRight`` are reconstructed from ``stim_side`` and
    ``contrast`` (:func:`reconstruct_contrast_sides`) as ``fit_psychometric``
    requires. Feeds goals 3 (curve overlays) and 4b (PCA features).

    Parameters
    ----------
    mouse_frame : pandas.DataFrame
        One mouse's concatenated trials + states (from
        :func:`build_mouse_states_frame`). Must carry ``map_state``, ``choice``,
        ``feedbackType``, ``probabilityLeft``, ``stim_side``, ``contrast``, and a
        reaction-time column ``rt`` (seconds). Trials dropped from the fit
        (``map_state`` NaN) are ignored by the ``groupby``.

    Returns
    -------
    pandas.DataFrame
        One row per state, columns ``['state', 'bias', 'threshold', 'lapse_left',
        'lapse_right', 'rt_slope', 'rt_intercept']``. ``bias``/``threshold``/
        ``lapse_*`` are the psychometric fit; ``rt_slope``/``rt_intercept`` the
        chronometric line (seconds per unit ``|contrast|`` and seconds). NaN where
        a state has too few trials to fit.
    """
    rows = []
    for state, trials in mouse_frame.groupby('map_state'):
        trials = trials.join(reconstruct_contrast_sides(trials))
        psych = fit_psychometric(trials)
        chrono = fit_chronometric(trials, rt_col='rt', contrast_col='contrast')
        rows.append({
            'state': int(state),
            'bias': psych['bias'],
            'threshold': psych['threshold'],
            'lapse_left': psych['lapse_left'],
            'lapse_right': psych['lapse_right'],
            'rt_slope': chrono['slope'],
            'rt_intercept': chrono['intercept'],
        })
    return pd.DataFrame(rows)


def _state_curves(
    mouse_frame: pd.DataFrame, param_table: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Assemble one mouse's per-state psychometric + chronometric plot frames.

    Combines empirical points (P(choose right) per signed contrast; median RT per
    ``|contrast|``) with the fitted parameters from ``param_table`` into the long
    frames :func:`plot_state_psychometric_chronometric` consumes.

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
        median_rt = trials['rt'].groupby(trials['contrast'].abs()).median()
        chrono_rows += [
            {'state': state, 'contrast': c, 'median_rt': rt,
             'slope': p['rt_slope'], 'intercept': p['rt_intercept']}
            for c, rt in median_rt.items()
        ]
    return {'psychometric': pd.DataFrame(psych_rows),
            'chronometric': pd.DataFrame(chrono_rows)}


def _block_transition_traces(
    mouse_frame: pd.DataFrame, window: int
) -> dict[str, np.ndarray]:
    """Mean per-state posterior traces around each block transition for one mouse.

    Detects ``probabilityLeft`` block transitions per eid (0.8->0.2 = L->R,
    0.2->0.8 = R->L; the initial 0.5 block never triggers), slices ``±window``
    trials of the per-state posteriors around each with
    :func:`align_traces_at_transitions`, and pools windows across the mouse's eids
    before averaging.

    Returns
    -------
    dict of str to numpy.ndarray
        ``{transition_type: mean_trace}`` with ``mean_trace`` shape
        ``(2*window+1, K)``; only transition types with at least one occurrence are
        present.
    """
    kept = mouse_frame[mouse_frame['map_state'].notna()]
    state_cols = [c for c in kept.columns if c.startswith('state_')]
    collected = {t: [] for t in BLOCK_TRANSITIONS}
    for _, eid_df in kept.groupby('eid'):
        p_left = eid_df['probabilityLeft'].to_numpy()
        values = eid_df[state_cols].to_numpy()
        for transition, (prev, cur) in BLOCK_TRANSITIONS.items():
            idx = np.flatnonzero((p_left[:-1] == prev) & (p_left[1:] == cur)) + 1
            if len(idx):
                windows, _ = align_traces_at_transitions(values, idx, window)
                collected[transition].append(windows)

    aligned = {}
    for transition, windows in collected.items():
        if windows:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                aligned[transition] = np.nanmean(
                    np.concatenate(windows, axis=0), axis=0)
    return aligned


def _save(fig: plt.Figure, name: str) -> None:
    """Save ``fig`` to ``DDM_HMM_FIGURES_DIR/{name}.png`` and close it."""
    DDM_HMM_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(DDM_HMM_FIGURES_DIR / f'{name}.png', dpi=FIGURE_DPI,
                bbox_inches='tight')
    plt.close(fig)


def _assemble_mouse_views(
    group: PhotometrySessionGroup, subjects: list[str], one, window: int
) -> dict:
    """Build every modeled mouse's plot inputs from the filtered group.

    Iterates ``subjects``, assembling each one's trials+states frame and deriving
    the per-mouse inputs for goals 1, 2, 3 and 5 plus its per-state feature rows.
    Mice with no session in the fit are skipped.

    Returns
    -------
    dict
        Keys ``'states'``, ``'dwell'``, ``'curves'``, ``'aligned'`` each map
        subject to that goal's plot input; ``'features'`` is the concatenated
        per-state behavioral-feature table (with a ``mouse`` column) for the PCA.
    """
    views = {key: {} for key in ('states', 'dwell', 'curves', 'aligned')}
    param_tables = []
    for subject in subjects:
        frame = build_mouse_states_frame(group, subject, one)
        if frame.empty:
            print(f"  {subject}: no fit sessions in group — skipped")
            continue
        frame['rt'] = frame['response_times'] - frame['stimOn_times']
        state_cols = ['map_state'] + [c for c in frame.columns
                                      if c.startswith('state_')]
        views['states'][subject] = frame[state_cols]

        kept = frame[frame['map_state'].notna()]
        views['dwell'][subject] = state_dwell_times(
            kept['map_state'].astype(int).to_numpy(), kept['eid'].to_numpy())

        param_table = build_state_param_table(frame)
        param_table['mouse'] = subject
        param_tables.append(param_table)
        views['curves'][subject] = _state_curves(frame, param_table)
        views['aligned'][subject] = _block_transition_traces(frame, window)
        print(f"  {subject}: {len(kept)} fit trials, "
              f"{param_table['state'].nunique()} states")

    views['features'] = pd.concat(param_tables, ignore_index=True)
    return views


def main(one=None, window: int = BLOCK_WINDOW) -> None:
    """Assemble every mouse's frame and render the six overview figures.

    Parameters
    ----------
    one : ONE, optional
        Connection for offline H5 access; a default read-only connection is
        created when omitted.
    window : int, optional
        Half-window in trials for the block-transition traces (default
        ``BLOCK_WINDOW``).
    """
    if one is None:
        one = _get_default_connection()
    group = PhotometrySessionGroup.from_catalog(
        pd.read_parquet(SESSIONS_FPATH), one=one, h5_dir=SESSIONS_H5_DIR)
    group.filter_sessions()

    ddm_params = pd.read_csv(DDM_HMM_PARAMS_FPATH)
    subjects = list(dict.fromkeys(ddm_params['mouse']))
    views = _assemble_mouse_views(group, subjects, one, window)

    _save(plot_state_posterior_histograms(views['states']), 'goal1_posteriors')
    _save(plot_state_dwell_times(views['dwell']), 'goal2_dwell_times')
    _save(plot_state_psychometric_chronometric(views['curves']),
          'goal3_psychometric_chronometric')
    _save(plot_state_param_scatter(ddm_params), 'goal4a_ddm_param_scatter')

    features = views['features'].dropna(subset=FEATURE_COLS)
    scores, _ = pca_2d(features[FEATURE_COLS].to_numpy())
    _save(plot_state_pca(scores, features['mouse']), 'goal4b_behavioral_pca')

    _save(plot_state_block_transitions(views['aligned'], window),
          'goal5_block_transitions')
    print(f"Wrote 6 figures to {DDM_HMM_FIGURES_DIR}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--window', type=int, default=BLOCK_WINDOW,
                        help='block-transition half-window in trials')
    args = parser.parse_args()
    main(window=args.window)
