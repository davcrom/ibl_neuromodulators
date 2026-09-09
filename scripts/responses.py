"""
Response Analysis Pipeline

Three passes over the biased and ephys sessions. The first prepares each
session's donor frame, the pool the cross-session swap null draws from; the
second fits each session's drop-one OLS models against that pool and returns
its trial-level magnitudes with them, twice over — once as the models saw
them and once with no trial filter applied, which is what the masking
diagnostic counts. The group collects both passes' returns into the population
tables — adding the FDR q-values and the per-mouse pooling — fits the per-cell
variance components, and caches every frame. The plotting pass reads those
frames, condition-averages the store's peri-event cuts, and draws the figures.

Output:
    results/responses/             — one parquet per result frame, plus the
                                     repeated-measures ANOVA table as CSV
    figures/responses/             — contrast_curves/, event_triggered_averages/,
                                     diagnostics/, persession/

Usage:
    python scripts/responses.py              # plot from existing parquet files
    python scripts/responses.py --reprocess  # re-extract + re-fit, then plot
"""
import argparse
from typing import Iterable

import matplotlib
import pandas as pd

matplotlib.use('Agg')  # batch figure generation; never open interactive windows
from matplotlib import pyplot as plt

from iblnm.config import (
    PROJECT_ROOT, SESSIONS_FPATH, SESSIONS_H5_DIR,
    RESPONSES_DIR, RESPONSE_MAGNITUDES_FPATH, RESPONSE_MAGNITUDE_COLUMNS,
    OLS_PERSESSION_FPATH,
    RESPONSE_OLS_MOUSE_PVAL_FPATH,
    MASKING_DIAGNOSTICS_FPATH, MASKING_DIAGNOSTIC_GROUP_COLS,
    MASKING_DIAGNOSTIC_STATISTICS, RESPONSE_MAGNITUDE_WINDOW,
    RESPONSE_EVENTS, RESPONSES, FIGURE_DPI, RESPONSE_MODEL_FORMULA,
    RESPONSE_DROPPED_TERMS, TRACE_INSET_TARGETNMS,
    MIN_RESPONSE_TIME,
)
from iblnm import task
from iblnm.data import DonorFrame, PhotometrySession, PhotometrySessionGroup
from iblnm.io import _get_default_connection
from iblnm.vis import (
    DROPONE_TERM_CLASSES,
    plot_masking_diagnostics,
    plot_mean_response_traces,
    plot_relative_contrast,
    plot_ols_dropone,
    plot_ols_dropone_subject,
    plot_ols_dropone_violin,
    plot_ols_total_r2,
    plot_ols_total_r2_subject,
    plot_ols_total_r2_violin,
)
from iblnm.analysis import aggregate_conditions


# =========================================================================
# Condition aggregation
# =========================================================================

# The spec's three aggregation modes as `analysis.aggregate_conditions`
# arguments: what each mean is taken over, and whether the between-subject
# offset is removed first. `pool` weights trials, `subject` weights mice, and
# `subject_centered` weights recordings with each subject's offset removed.
AGGREGATION_MODES = {
    'pool': {'unit_cols': None, 'center_by': None},
    'subject': {'unit_cols': ['subject'], 'center_by': None},
    'subject_centered': {'unit_cols': ['eid', 'brain_region'],
                         'center_by': 'subject'},
}

# Condition keys of the event-triggered averages. No stimulus-side split: the
# traces are grouped by contrast and outcome alone.
TRACE_GROUP_COLS = ['target_NM', 'event', 'contrast', 'feedbackType', 'time']

# Condition keys of the contrast curves, which do split by stimulus side: the
# two panels are the contra and ipsi halves of the curve.
CONTRAST_GROUP_COLS = ['target_NM', 'event', 'side', 'contrast', 'feedbackType']

# Columns identifying the recording a trace sample came from, carried through
# the long frame so the aggregation can average over recordings or subjects.
_TRACE_KEYS = ['eid', 'subject', 'target_NM', 'brain_region']

# The `config.RESPONSES` entry this pass measures and corrects its traces by:
# the window averaged into a magnitude, whether the pre-event baseline is
# subtracted, and the events past which a trial's samples are blanked.
RESPONSE_ENTRY = RESPONSES['stimulus']

# An ANOVA factor named `<column>_bin` is that trials column cut into
# within-session terciles, labelled low to high. Binning is an analysis choice,
# so it lives here rather than in the fit.
BIN_SUFFIX = '_bin'
TERCILE_LABELS = ('low', 'mid', 'high')


def _recording_traces(rec: pd.Series, ps, trials: pd.DataFrame,
                      correct: bool = True) -> pd.DataFrame:
    """One recording's per-trial trace samples, long, restricted to ``trials``.

    Parameters
    ----------
    rec : pandas.Series
        Recording row, supplying the ``_TRACE_KEYS`` identity and the region
        whose cut is read.
    ps : PhotometrySession
        Session with ``photometry_responses`` and ``trials`` loaded.
    trials : pandas.DataFrame
        The already-selected trials, one row per recording x event x trial,
        carrying the condition columns the aggregation groups on.
    correct : bool
        Apply the response definition's two corrections — mask the samples
        past the next event, then subtract the ``config.BASELINE_WINDOW``
        baseline — before flattening. False leaves the stored cut as it is.

    Returns
    -------
    pandas.DataFrame
        One row per event x trial x time sample, columns ``_TRACE_KEYS`` plus
        ``event``, ``time``, ``value`` and the condition columns ``trials``
        carries. The join is an inner one on (event, trial), so a trial the
        selection dropped contributes no samples.
    """
    responses = ps.photometry_responses[rec['brain_region']]
    if correct:
        responses = ps.subtract_baseline(ps.mask_subsequent_events(
            responses, RESPONSE_ENTRY['masking_events']))
    samples = (responses.to_dataframe(name='value').reset_index()
               .astype({'value': 'float32'}))
    keys = trials[trials['eid'] == rec['eid']]
    keys = keys[keys['brain_region'] == rec['brain_region']]
    return samples.merge(keys.drop(columns=_TRACE_KEYS), on=['event', 'trial'],
                         how='inner').assign(**{key: rec[key]
                                                for key in _TRACE_KEYS})


def condition_traces(recordings, trials: pd.DataFrame,
                     mode: str = 'subject_centered', correct: bool = True,
                     group_cols=TRACE_GROUP_COLS,
                     response_col: str = 'response') -> pd.DataFrame:
    """Correct, select and condition-average per-trial traces.

    The plotting pass's whole computation: each recording's stored peri-event
    cut is corrected, restricted to the trials the models are fitted on, and
    reduced to one mean and SEM per condition. No trace frame is written and
    nothing is retained per trial — the caller hands the result straight to
    ``iblnm.vis``.

    Parameters
    ----------
    recordings : iterable of (pandas.Series, PhotometrySession)
        The recordings to average over, as ``PhotometrySessionGroup`` yields
        them, each session carrying its loaded trials and photometry.
    trials : pandas.DataFrame
        The stored magnitude frame (``config.RESPONSE_MAGNITUDE_COLUMNS``),
        one row per recording x event x trial. It carries the fitted
        selection already, so the traces average the trials the models fit
        without re-deriving one here.
    mode : {'pool', 'subject', 'subject_centered'}
        Averaging unit, via ``AGGREGATION_MODES``.
    correct : bool
        Apply the masking and baseline subtraction (see
        :func:`_recording_traces`). False plots the uncorrected trace.
    group_cols : sequence of str
        Condition keys the means are taken within; ``time`` is one of them, so
        each condition comes back as a trace.
    response_col : str
        Magnitude column whose null rows the trial selection drops.

    Returns
    -------
    pandas.DataFrame
        ``group_cols`` plus ``mean``, ``sem`` and ``n``, the
        :func:`iblnm.analysis.aggregate_conditions` output shape.
    """
    traces = pd.concat([_recording_traces(rec, ps, trials, correct)
                        for rec, ps in recordings], ignore_index=True)
    return aggregate_conditions(traces, 'value', group_cols,
                                **AGGREGATION_MODES[mode])


def _cohort_recordings(group, cohort: pd.DataFrame):
    """Yield ``(recording row, session)`` for one cohort, reading each file once.

    A session's stored trials and photometry are loaded when its first
    recording comes up and dropped from the group's session cache once its last
    one has been yielded, so the pass holds one session's signals at a time
    rather than the whole cohort's. A session with no file in the store, and a
    region carrying no stored cut, contribute nothing — the same tolerance the
    modelling pass has, so one unbuilt recording costs its own traces and not
    the cohort's figure.
    """
    for eid, rows in cohort.groupby('eid'):
        session = group._get_session(rows.iloc[0])
        if not session.filepath.exists():
            continue
        session.load_h5(session.filepath, groups=['trials', 'photometry'])
        cuts = getattr(session, 'photometry_responses', {})
        yield from ((rec, session) for _, rec in rows.iterrows()
                    if rec['brain_region'] in cuts)
        group._sessions.pop(eid, None)


def plot_trace_figures(group, trials, figures_dir, mode='subject_centered',
                       correct=True):
    """Save one event-triggered average figure per cohort, from the store.

    The plotting pass: one cohort at a time, each recording's stored peri-event
    cut is read, corrected, condition-averaged (:func:`condition_traces`) and
    handed to the drawer. Nothing is written but the figures — the per-trial
    traces are discarded with the cohort that produced them.

    Parameters
    ----------
    group : PhotometrySessionGroup
        Filtered to the recordings in scope; it supplies the cohorts and opens
        each session's stored cut.
    trials : pandas.DataFrame
        The stored magnitude frame, whose rows name the trials averaged
        (:func:`condition_traces`).
    figures_dir : Path
        Output directory for the SVG figures.
    mode : {'pool', 'subject', 'subject_centered'}
        Averaging unit, via ``AGGREGATION_MODES``.
    correct : bool
        Mask each trace past the next event and subtract its baseline. False
        plots the uncorrected average.
    """
    for target_nm, cohort in group.recordings.groupby('target_NM'):
        cells = condition_traces(_cohort_recordings(group, cohort), trials,
                                 mode=mode, correct=correct)
        fig = plot_mean_response_traces(
            cells, target_nm, inset=target_nm in TRACE_INSET_TARGETNMS,
            count_label=f"{cohort['eid'].nunique()} sessions, "
                        f"{cohort['subject'].nunique()} mice")
        fig.savefig(figures_dir / f'{target_nm}_traces.svg', dpi=FIGURE_DPI,
                    bbox_inches='tight')
        plt.close(fig)


# =========================================================================
# Response magnitude plotting
# =========================================================================

def print_response_summary(df_responses):
    """Print a summary of the response magnitudes DataFrame."""
    n_sessions = df_responses['eid'].nunique()
    n_subjects = df_responses['subject'].nunique()

    print(f"\n{len(df_responses)} rows, {n_sessions} sessions, {n_subjects} subjects")
    print("\nTrials per target-NM:")
    summary = (
        df_responses[df_responses['event'] == RESPONSE_EVENTS[0]]
        .groupby('target_NM')
        .agg(
            n_subjects=('subject', 'nunique'),
            n_sessions=('eid', 'nunique'),
            n_trials=('trial', 'count'),
        )
    )
    print(summary.to_string())


def plot_response_figures(magnitudes, figures_dir, response_col='response',
                          modes=('pool', 'subject')):
    """Plot response magnitude by contrast x feedback x stimulus side.

    Produces one figure per (target_NM, event) and aggregation mode, named
    ``{target_NM}_{event}_{response_col}_{mode}.svg``. The aggregation happens
    here — ``vis`` receives means and SEMs and draws them as given.

    Parameters
    ----------
    magnitudes : pandas.DataFrame
        The stored magnitude frame
        (``config.RESPONSE_MAGNITUDE_COLUMNS``); its rows are the trials
        drawn.
    figures_dir : Path
        Output directory for SVG files.
    response_col : str
        Column name for the response magnitude, also the window label.
    modes : sequence of str
        Aggregation modes to plot, keys of ``AGGREGATION_MODES``. Every mode
        draws the same means where it weights units equally; they differ in
        what the error bars are taken over.
    """
    for mode in modes:
        cells = aggregate_conditions(magnitudes, response_col,
                                     CONTRAST_GROUP_COLS,
                                     **AGGREGATION_MODES[mode])
        for (target_nm, event), df_group in magnitudes.groupby(['target_NM',
                                                                'event']):
            if df_group['subject'].nunique() < 2:
                continue
            agg_df = cells[(cells['target_NM'] == target_nm)
                           & (cells['event'] == event)]
            fig = plot_relative_contrast(
                agg_df, target_nm, event, window_label=response_col,
                count_label=f"{df_group['eid'].nunique()} sessions, "
                            f"{df_group['subject'].nunique()} subjects")
            event_label = event.replace('_times', '')
            fname = f'{target_nm}_{event_label}_{response_col}_{mode}.svg'
            fig.savefig(figures_dir / fname, dpi=FIGURE_DPI,
                        bbox_inches='tight')
            plt.close(fig)


# =========================================================================
# Per-recording OLS drop-one
# =========================================================================

# The trial exclusions this analysis applies, held in one dict so the rows the
# models are fitted on and the rows the stored magnitude table carries cannot
# drift apart: the fitting loop takes it per fiber x event and the link function
# re-applies it without one to read the session-wide view back.
# Every predictor the models read must be usable on every trial that survives,
# donor and focal alike: a trial carrying a blank or an unloggable value is one
# patsy would drop from the design behind the fit's back, leaving the row count
# the adjusted R-squared is charged against wrong and, once a donor column is
# swapped into it, a NaN in the least-squares design. `code_predictors` codes
# `side`, `choice_side` and `reward` through `np.where`, so a blank
# `stim_side` or `feedbackType` would come out as the opposite level rather
# than as a blank — the completeness entry is the only place that can catch
# those. The reaction-time bound is what a completeness entry cannot say in
# turn: a reaction time at or below zero is present but has no log.
PERSESSION_TRIAL_CRITERIA = {
    'exclude_nogo': True,
    'min_response_time': MIN_RESPONSE_TIME,
    'complete': ['contrast', 'stim_side', 'feedbackType', 'choice',
                 'peak_velocity'],
    'bounds': {'reaction_time': ('>', 0)},
}


def prepare_donor(ps) -> DonorFrame:
    """First pass: prepare one session's contribution to the swap null.

    :func:`fit_session`'s measurement sequence with the photometry left out
    and the response-completeness criterion switched off — a donor contributes
    a regressor column and a trial order, never a response. So the surviving
    trials are one frame per session rather than one per fiber x event, and it
    is typically longer than the focal frames it donates to;
    :func:`iblnm.analysis.permutation_null_delta_r2` truncates the pair to the
    shorter length at swap time.

    The frame is coded relative to the session's first fiber's hemisphere,
    because ``side`` and ``choice_side`` are defined only once a hemisphere is
    named. Which one it is does not matter downstream: the other negates both
    columns and every interaction they enter, which spans the same design
    space and so leaves R² unchanged.

    Trial order is preserved, which is the whole point of the swap — the
    donor's regressor keeps its own serial structure while losing any
    relationship to the focal session's responses.

    Parameters
    ----------
    ps : PhotometrySession
        The session to prepare.

    Returns
    -------
    DonorFrame
        This session's identity and its coded trial frame, carrying every
        `config.PERSESSION_REGRESSORS` column. Plain data holding no reference
        to a group, for :meth:`PhotometrySessionGroup.collect_donor_frames` to
        key by eid.
    """
    ps.load_trials()
    ps.extract_trial_timings()
    ps.load_peak_velocity()
    ps.add_trial_columns(ps.wheel_peak_velocity)
    ps.filter_trials(**PERSESSION_TRIAL_CRITERIA)
    hemisphere = next(iter(ps.hemisphere), None)
    frame = PhotometrySession.code_predictors(
        task.add_relative_contrast(ps.trials.assign(hemisphere=hemisphere)))
    return DonorFrame(ps.eid, ps.subject, tuple(ps.target_NM), frame)


def _join_trials(magnitudes: pd.DataFrame,
                 trials: pd.DataFrame) -> pd.DataFrame:
    """Put one session's trial-level columns beside its response magnitudes.

    The measurement frame is one row per fiber x event x trial and the trials
    table one row per trial, so this is an inner join on ``trial``: whichever
    trials the table carries are the rows that come back, which is how the
    trial mask reaches the magnitudes. ``hemisphere`` rides in on the
    magnitudes, one entry per fiber, so ``side`` and ``choice_side`` come out
    relative to the fiber that measured the row.

    Parameters
    ----------
    magnitudes : pandas.DataFrame
        `iblnm.data._RECORDING_MAGNITUDE_COLUMNS`, masked or not.
    trials : pandas.DataFrame
        The trials table, masked or not.

    Returns
    -------
    pandas.DataFrame
        The join plus the three columns
        :func:`iblnm.task.add_relative_contrast` derives.
    """
    return task.add_relative_contrast(magnitudes.merge(trials, on='trial'))


def add_anova_bins(trials: pd.DataFrame,
                   factors: Iterable[str]) -> pd.DataFrame:
    """Cut one session's continuous ANOVA factors into within-session terciles.

    Called with a single session's rows in hand, so the bin edges are that
    session's own quantiles: every session yields all three bins and no subject
    loses a cell to a binned factor. A column of the same name is overwritten,
    so the factor always carries this run's binning.

    Parameters
    ----------
    trials : pandas.DataFrame
        One session's magnitude rows, one per recording x event x trial. The
        binned column repeats across a trial's recordings and events, which
        leaves the quantiles unchanged.
    factors : iterable of str
        The ANOVA factor names, e.g. a ``config.RESPONSES`` entry's ``ANOVA``
        mapping. Names not ending in ``BIN_SUFFIX`` are left alone; the rest
        name the column they are binned from.

    Returns
    -------
    pandas.DataFrame
        A copy carrying one ``TERCILE_LABELS``-valued column per binned factor.
    """
    binned = trials.copy()
    for factor in factors:
        if factor.endswith(BIN_SUFFIX):
            source = factor[:-len(BIN_SUFFIX)]
            binned[factor] = pd.qcut(binned[source], 3,
                                     labels=TERCILE_LABELS).astype(str)
    return binned


def fit_session(ps, formula: str, dropped_terms: dict,
                donors: dict) -> tuple[pd.DataFrame, pd.DataFrame,
                                       pd.DataFrame]:
    """Second pass: fit one session's drop-one models against the donor pool.

    Sequences the loading and the measurement the fit reads — the trials with
    their timings, the wheel regressor onto the trials table, and every fiber x
    event magnitude — then fits, then re-applies
    ``PERSESSION_TRIAL_CRITERIA`` with no fiber or event named so the
    magnitudes read back span every recording the session holds rather than the
    last combination the loop happened to mask.

    Parameters
    ----------
    ps : PhotometrySession
        The session to fit.
    formula : str
        Full-model Wilkinson formula, ``config.RESPONSE_MODEL_FORMULA``.
    dropped_terms : dict
        Drop-one label -> terms reduced out of ``formula`` under it,
        ``config.RESPONSE_DROPPED_TERMS``.
    donors : dict
        The whole pass's donor pool, keyed by eid; the session narrows it to
        the sessions its ``donor_scope`` admits.

    Returns
    -------
    magnitudes : pandas.DataFrame
        The filtered magnitude view with its trials joined on and coded
        relative to each recording's hemisphere — one row per recording x event
        x trial, carrying the rows the models were fitted on, plus the entry's
        binned ANOVA factors cut against this session's own quantiles.
    fits : pandas.DataFrame
        This session's rows of the population OLS table.
    unfiltered : pandas.DataFrame
        The same frame with no mask applied, for
        :func:`compute_masking_diagnostics`. Read before any mask exists,
        because the trials whose response window was masked end to end carry
        no magnitude, are dropped by the completeness criterion every fit
        applies, and are the ones the diagnostic exists to count.
    """
    ps.load_trials()
    ps.extract_trial_timings()
    ps.load_peak_velocity()
    ps.add_trial_columns(ps.wheel_peak_velocity)
    ps.load_responses('photometry')
    ps.extract_response_magnitudes(
        RESPONSE_ENTRY['window'], RESPONSE_ENTRY['masking_events'],
        RESPONSE_ENTRY['baseline_correct'])
    unfiltered = _join_trials(ps.masking_diagnostics(), ps.trials)
    fits = ps.fit_responses(formula, dropped_terms, donors,
                            **PERSESSION_TRIAL_CRITERIA)
    ps.filter_trials(**PERSESSION_TRIAL_CRITERIA)
    magnitudes = add_anova_bins(
        _join_trials(ps.response_magnitudes, ps.trials),
        RESPONSE_ENTRY['ANOVA'])
    return magnitudes, fits, unfiltered


def compute_masking_diagnostics(
    magnitudes: pd.DataFrame,
    window: tuple[float, float] = RESPONSE_MAGNITUDE_WINDOW,
    group_cols: list[str] = MASKING_DIAGNOSTIC_GROUP_COLS,
) -> pd.DataFrame:
    """How much of the response window the event masking removed, per cell.

    Masking removes the samples following the next event, so it takes away
    more of the window on fast trials — and fast trials are more frequent at
    high contrast. That makes every contrast-dependent result partly a
    statement about which trials still had a window to average, which is what
    this frame reports.

    The trials are counted as they are handed in, with no selection of their
    own: the frame to pass is the unfiltered one
    (:meth:`PhotometrySession.masking_diagnostics`), because the trials whose
    window was masked end to end carry no magnitude, are dropped by every fit,
    and are the ones this frame exists to count.

    Parameters
    ----------
    magnitudes : pandas.DataFrame
        The uncoded merged magnitude frame
        (``config.RESPONSE_MAGNITUDE_COLUMNS``) with no trial mask applied,
        carrying ``masked_fraction`` beside the ``contrast``/``feedbackType``/
        ``reaction_time`` the cells are keyed and counted on.
    window : tuple of float
        The response window, in seconds relative to the event, that
        ``masked_fraction`` was measured over. Only ``pct_move_in_window``
        reads it; the fractions were computed upstream.
    group_cols : sequence of str
        Cell keys. One output row per observed combination.

    Returns
    -------
    pandas.DataFrame
        The cell keys in ``group_cols``, then
        ``config.MASKING_DIAGNOSTIC_STATISTICS``: the trial count,
        the mean masked fraction, and the percentage of trials with any
        masking, with the window masked end to end, and with first movement
        inside the window. A trial with no ``reaction_time`` counts as one
        whose movement was not in the window.
    """
    trials = magnitudes.assign(
        any_masked=magnitudes['masked_fraction'] > 0,
        fully_masked=magnitudes['masked_fraction'] == 1,
        move_in_window=magnitudes['reaction_time'].between(*window),
    )
    cells = aggregate_conditions(trials, 'masked_fraction', group_cols)
    diagnostics = cells[group_cols].assign(
        n_trials=cells['n'], masked_fraction_mean=cells['mean'])
    for source, column in (('any_masked', 'pct_any_masked'),
                           ('fully_masked', 'pct_fully_masked'),
                           ('move_in_window', 'pct_move_in_window')):
        proportions = aggregate_conditions(trials, source, group_cols)
        diagnostics = diagnostics.merge(
            proportions[group_cols + ['mean']].rename(
                columns={'mean': column}), on=group_cols)
        diagnostics[column] *= 100
    return diagnostics[list(group_cols) + MASKING_DIAGNOSTIC_STATISTICS]


def plot_masking_figures(diagnostics: pd.DataFrame, figures_dir) -> None:
    """Save one masking diagnostics figure per (target_NM, event).

    Parameters
    ----------
    diagnostics : pandas.DataFrame
        The cell frame from :func:`compute_masking_diagnostics`.
    figures_dir : Path
        Output directory for SVG files.
    """
    for (target_nm, event), cells in diagnostics.groupby(['target_NM',
                                                          'event']):
        fig = plot_masking_diagnostics(cells, target_nm, event)
        fname = f"{target_nm}_{event.replace('_times', '')}_masking.svg"
        fig.savefig(figures_dir / fname, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close(fig)


# display mode → (drop-one figure fn, full-model R² figure fn)
_PERSESSION_DISPLAY_FNS = {
    'session': (plot_ols_dropone, plot_ols_total_r2),
    'subject': (plot_ols_dropone_subject, plot_ols_total_r2_subject),
    'target': (plot_ols_dropone_violin, plot_ols_total_r2_violin),
}


def dropone_ylim(results: pd.DataFrame, terms: list[str],
                 margin: float = 0.05) -> tuple[float, float] | None:
    """Y-limits spanning one term class's per-session ΔR², padded by ``margin``.

    Every figure of a class carries this range, so a term contributing nothing
    reads as nothing instead of being autoscaled to fill its own panel. The
    padding, a fraction of the values' span, keeps the extreme marks off the
    spines. A class with no values, or one value repeated, has no range to
    share and returns None, leaving the panels to autoscale.

    Parameters
    ----------
    results : pandas.DataFrame
        The merged per-recording OLS frame; ``predictor`` and ``delta_r2_adj``
        are the columns read. Every display mode reduces these same values, so
        one range computed here covers all three.
    terms : list of str
        The class's drop-one labels — a ``vis.DROPONE_TERM_CLASSES`` entry.
    margin : float
        Padding at each end, as a fraction of the span.

    Returns
    -------
    tuple[float, float] or None
    """
    values = results.loc[results['predictor'].isin(terms), 'delta_r2_adj']
    low, high = (values.min(), values.max()) if len(values) else (0.0, 0.0)
    if low == high:
        return None
    pad = margin * (high - low)
    return low - pad, high + pad


def plot_persession_figures(results: pd.DataFrame,
                            mouse_pvalues: pd.DataFrame | None, figures_dir,
                            display: str = 'session') -> None:
    """Save one drop-one ΔR² figure per dropped term, plus full-model R².

    Each ``config.RESPONSE_DROPPED_TERMS`` label gets its own figure — event
    columns, no predictor axis — written as ``{predictor}.svg`` with the
    interaction colon replaced by a hyphen, which a shell and a non-Linux
    filesystem both prefer. Y-limits are shared within a term class and differ
    between the two (``vis.DROPONE_TERM_CLASSES``): main-effect ΔR² runs an
    order of magnitude above interaction ΔR², so one common axis flattens the
    interactions, and autoscaling each figure to itself removes the comparison
    across terms. The full-model R² figure has no predictor axis and stays one
    per display mode. ``display`` selects how each session's values are drawn —
    per-session dots (``session``), per-subject median+IQR (``subject``), or a
    per-target violin (``target``) — via ``_PERSESSION_DISPLAY_FNS``.

    Parameters
    ----------
    results : pandas.DataFrame
        The merged per-recording OLS frame
        (``config.OLS_PERSESSION_COLUMNS``), one row per recording x event x
        dropped predictor. It carries each recording's own q-value, so the
        ``session`` mode figure colors its dots from it directly.
    mouse_pvalues : pandas.DataFrame or None
        The per-mouse p-value table, the coarser grain that colors the subject
        mean dashes. Read in ``session`` mode alone; the other two modes take
        none, so None is fine there.
    figures_dir : Path
        Output directory for the SVG figures.
    display : {'session', 'subject', 'target'}
        Per-session value display mode.
    """
    dropone_fn, total_r2_fn = _PERSESSION_DISPLAY_FNS[display]
    dropone_kwargs = ({'mouse_pvalues': mouse_pvalues}
                      if display == 'session' else {})

    for terms in DROPONE_TERM_CLASSES.values():
        ylim = dropone_ylim(results, terms)
        for term in terms:
            fig = dropone_fn(
                results,
                title=f'Per-session OLS drop-one ΔR²: {term}'
                      '\nevery session is a point',
                predictor=term, ylim=ylim, **dropone_kwargs)
            fig.savefig(figures_dir / f"{term.replace(':', '-')}.svg",
                        dpi=FIGURE_DPI, bbox_inches='tight')
            plt.close(fig)

    fig = total_r2_fn(
        results,
        title='Per-session full-model R²\nevery session is a point')
    fig.savefig(figures_dir / 'response_ols_persession_total_r2.svg',
                dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  {len(RESPONSE_DROPPED_TERMS)} per-session OLS drop-one figures "
          "and the full-model R² figure saved")


# The frames --reprocess writes and the no-flag branch reads back, each under
# the name the plotting steps take it by.
RESULT_FPATHS = {
    'magnitudes': RESPONSE_MAGNITUDES_FPATH,
    'ols': OLS_PERSESSION_FPATH,
    'ols_mouse': RESPONSE_OLS_MOUSE_PVAL_FPATH,
    # Cached like the rest, because the frame it is reduced from is the
    # unfiltered one only the fitting pass holds.
    'masking_diagnostics': MASKING_DIAGNOSTICS_FPATH,
}


def read_result_frames(group, paths: dict = RESULT_FPATHS,
                       ) -> dict[str, pd.DataFrame]:
    """Read the cached result frames, each narrowed to the group's recordings.

    The no-flag branch's whole data step: an earlier ``--reprocess`` run wrote
    these files over whatever sessions it analysed, and this run's own filters
    decide which of those rows are in scope.
    :meth:`PhotometrySessionGroup.filter_to_recordings` does the narrowing at
    whichever grain each frame is keyed on, so the per-mouse table — keyed by
    cell, with no ``eid`` — comes back whole.

    Parameters
    ----------
    group : PhotometrySessionGroup
        Already filtered and deduplicated; only its recordings are read.
    paths : dict[str, pathlib.Path]
        Frame name to parquet path. Defaults to ``RESULT_FPATHS``, every file
        ``--reprocess`` writes.

    Returns
    -------
    dict[str, pandas.DataFrame]
        One narrowed frame per entry in ``paths``, under the same names.
    """
    return {name: group.filter_to_recordings(pd.read_parquet(path))
            for name, path in paths.items()}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--reprocess', action='store_true',
                        help='re-extract responses and re-fit per-session models; '
                             'default plots from existing parquet files')
    parser.add_argument('--persession-display', choices=('session', 'subject',
                                                         'target'),
                        default='session',
                        help='per-session OLS figure display mode: per-session '
                             'dots (session), per-subject median+IQR (subject), '
                             'or per-target violin (target)')
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_args()

    # Create output directories
    data_dir = RESPONSES_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    fig_base = PROJECT_ROOT / 'figures/responses'
    fig_dirs = {
        'contrast_curves': fig_base / 'contrast_curves',
        'diagnostics': fig_base / 'diagnostics',
        'event_triggered_averages': fig_base / 'event_triggered_averages',
        'persession': fig_base / 'persession',
    }
    for d in fig_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # =====================================================================
    # Load sessions and create group
    # =====================================================================
    print(f"Loading sessions from {SESSIONS_FPATH}")
    df = pd.read_parquet(SESSIONS_FPATH)

    one = _get_default_connection()
    group = PhotometrySessionGroup.from_catalog(df, one=one, h5_dir=SESSIONS_H5_DIR)
    group.filter_sessions(
        session_types=('biased', 'ephys')
    )
    dup_log = group.deduplicate()
    print(f"  Deduplicated ({len(dup_log)} true-duplicate groups resolved)")
    recordings = group.recordings
    print(f"  Recordings (session x region): {len(recordings)}")
    print(f"  Sessions: {recordings['eid'].nunique()}, "
          f"mice: {recordings['subject'].nunique()}")
    print("  Recordings per target-NM:")
    print(recordings['target_NM'].value_counts().to_string())

    if args.reprocess:
        # =================================================================
        # Full pipeline: two sequential passes over the store's sessions
        # =================================================================

        # --- Pass 1: the donor pool the swap null draws from ---
        # One prepared frame per session, held in memory and never written.
        # Both passes run at the default `workers=1`: `process` pickles its
        # kwargs once per session, so a parallel pass would serialize the whole
        # pool once for every session in it.
        print("\nPreparing donor frames...")
        donors = group.collect_donor_frames(group.process(prepare_donor))
        print(f"  Donor frames: {len(donors)}")

        # --- Pass 2: each session's drop-one fits, scored against that pool ---
        print("Fitting per-session drop-one OLS models...")
        returns = [frames for frames in
                   group.process(fit_session,
                                 formula=RESPONSE_MODEL_FORMULA,
                                 dropped_terms=RESPONSE_DROPPED_TERMS,
                                 donors=donors)
                   if frames is not None]

        magnitudes = (pd.concat([frames[0] for frames in returns],
                                ignore_index=True) if returns
                      else pd.DataFrame(columns=RESPONSE_MAGNITUDE_COLUMNS))
        if len(magnitudes) == 0:
            print(f"No response magnitudes: all {len(group.sessions)} "
                  "sessions returned nothing. Check the logged errors.")
            raise SystemExit(1)
        # Factors the entry names but the stored schema does not carry — the
        # binned ones — ride along, so the ANOVA finds what this pass derived
        # whether it runs now or off the parquet.
        derived_factors = [factor for factor in RESPONSE_ENTRY['ANOVA']
                           if factor not in RESPONSE_MAGNITUDE_COLUMNS]
        magnitudes = magnitudes[RESPONSE_MAGNITUDE_COLUMNS + derived_factors]

        # The FDR correction and the per-mouse pooling both span sessions, so
        # they happen here rather than in either pass.
        ols, ols_mouse = group.collect_fits([frames[1] for frames in returns])

        magnitudes.to_parquet(RESPONSE_MAGNITUDES_FPATH, index=False)
        ols.to_parquet(OLS_PERSESSION_FPATH, index=False)
        ols_mouse.to_parquet(RESPONSE_OLS_MOUSE_PVAL_FPATH, index=False)
        print(f"Saved response magnitudes to {RESPONSE_MAGNITUDES_FPATH}")
        print(f"Saved per-recording OLS results to {OLS_PERSESSION_FPATH} "
              f"and per-mouse p-values to {RESPONSE_OLS_MOUSE_PVAL_FPATH}")

        # --- Masking diagnostics: how much window each trial type kept ---
        # From the third frame, the one no mask was applied to: the trials the
        # models never saw are the ones this counts, so it is computed here
        # rather than from the stored table, and cached for the no-flag branch.
        print("\nComputing masking diagnostics...")
        unfiltered = pd.concat([frames[2] for frames in returns],
                               ignore_index=True)
        diagnostics = compute_masking_diagnostics(unfiltered)
        diagnostics.to_parquet(MASKING_DIAGNOSTICS_FPATH, index=False)
        # The same statistics at cohort grain, small enough to read in the log.
        print(compute_masking_diagnostics(
            unfiltered, group_cols=['target_NM', 'event']).to_string(
                index=False))
        print(f"Saved masking diagnostics to {MASKING_DIAGNOSTICS_FPATH}")

    else:
        # =================================================================
        # Default: load pre-existing parquet files
        # =================================================================
        for fpath in RESULT_FPATHS.values():
            if not fpath.exists():
                print(f"Error: {fpath} not found. Run with --reprocess first.")
                raise SystemExit(1)

        frames = read_result_frames(group)
        magnitudes = frames['magnitudes']
        ols, ols_mouse = frames['ols'], frames['ols_mouse']
        diagnostics = frames['masking_diagnostics']

    # =====================================================================
    # Response magnitude plots
    # =====================================================================
    print_response_summary(magnitudes)

    print("\nGenerating response magnitude plots...")
    plot_response_figures(magnitudes, fig_dirs['contrast_curves'])
    print(f"Response magnitude figures saved to {fig_dirs['contrast_curves']}")

    # =====================================================================
    # Event-triggered averages — one pass over the store's per-trial traces
    # =====================================================================
    print("\nGenerating event-triggered averages (reading the store)...")
    plot_trace_figures(group, magnitudes,
                       fig_dirs['event_triggered_averages'])
    print("Event-triggered averages saved to "
          f"{fig_dirs['event_triggered_averages']}")

    # =====================================================================
    # Masking diagnostics — how much window each trial type kept
    # =====================================================================
    print("\nGenerating masking diagnostic figures...")
    plot_masking_figures(diagnostics, fig_dirs['diagnostics'])
    print(f"Masking diagnostic figures saved to {fig_dirs['diagnostics']}")

    # =====================================================================
    # Repeated-measures ANOVA on subject means
    # =====================================================================
    print("\nRunning repeated-measures ANOVA on subject means...")
    print(f"  Factors (level filter, [] keeps all): {RESPONSE_ENTRY['ANOVA']}")
    anova_results = group.response_anovaRM_fit(
        magnitudes, RESPONSE_ENTRY['ANOVA'],
        min_trials=RESPONSE_ENTRY['min_trials'],
        min_subjects=RESPONSE_ENTRY['min_subjects'])
    if anova_results:
        all_tables = []
        for (tnm, ev), table in anova_results.items():
            print(f"\n  {tnm} x {ev} "
                  f"({table['n_subjects'].iloc[0]} subjects, "
                  f"{table['n_subjects_dropped'].iloc[0]} incomplete "
                  f"and dropped):")
            for _, row in table.iterrows():
                sig = '*' if row['Pr(>F)'] < 0.05 else ''
                print(f"    {row['Source']:40s} F={row['F']:.3f}  "
                      f"p={row['Pr(>F)']:.4f} {sig}")
            tagged = table.copy()
            tagged.insert(0, 'target_NM', tnm)
            tagged.insert(1, 'event', ev)
            all_tables.append(tagged)
        anova_df = pd.concat(all_tables, ignore_index=True)
        anova_path = data_dir / 'anova_subject_means.csv'
        anova_df.to_csv(anova_path, index=False)
        print(f"\n  ANOVA results saved to {anova_path}")
    else:
        print("  No groups with sufficient data for ANOVA.")

    # =====================================================================
    # Per-session OLS drop-one
    # =====================================================================
    print("\nGenerating per-session OLS drop-one figure...")
    plot_persession_figures(ols, ols_mouse, fig_dirs['persession'],
                            display=args.persession_display)
    print(f"Per-session OLS figures saved to {fig_dirs['persession']}")
