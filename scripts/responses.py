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

import matplotlib
import pandas as pd

matplotlib.use('Agg')  # batch figure generation; never open interactive windows
from matplotlib import pyplot as plt

from iblnm.config import (
    PROJECT_ROOT, SESSIONS_FPATH, SESSIONS_H5_DIR,
    RESPONSES_DIR, RESPONSE_MAGNITUDES_FPATH, RESPONSE_MAGNITUDE_COLUMNS,
    OLS_PERSESSION_FPATH,
    RESPONSE_OLS_MOUSE_PVAL_FPATH, RESPONSE_OLS_COEFS_COLUMNS,
    RESPONSE_VARCOMP_SUMMARY_FPATH, RESPONSE_VARCOMP_VIOLIN_FPATH,
    VARCOMP_MCMC, VARCOMP_TAU_PRIOR, VARCOMP_MIN_MICE,
    VARCOMP_MIN_SESSIONS_PER_MOUSE, VARCOMP_KDE_GRID, VARCOMP_HDI_PROB,
    MASKING_DIAGNOSTICS_FPATH, MASKING_DIAGNOSTIC_GROUP_COLS,
    MASKING_DIAGNOSTIC_STATISTICS, RESPONSE_WINDOWS,
    RESPONSE_EVENTS, FIGURE_DPI, LMM_FORMULAS, TRACE_INSET_TARGETNMS,
    MOVEMENT_VARS, MIN_SUBJECTS_MOVEMENT, MIN_TRIALS_MOVEMENT,
    MIN_RESPONSE_TIME,
)
from iblnm import task
from iblnm.data import DonorFrame, PhotometrySession, PhotometrySessionGroup
from iblnm.io import _get_default_connection
from iblnm.vis import (
    plot_masking_diagnostics,
    plot_mean_response_traces,
    plot_relative_contrast,
    plot_lmm_summary,
    plot_lmm_ceiling,
    plot_lmm_reliability,
    plot_movement_r2_bars,
    plot_ols_dropone,
    plot_ols_dropone_subject,
    plot_ols_dropone_violin,
    plot_ols_total_r2,
    plot_ols_total_r2_subject,
    plot_ols_total_r2_violin,
    plot_varcomp_violins,
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
        responses = ps.subtract_baseline(ps.mask_subsequent_events(responses))
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
# LMM statistical analysis
# =========================================================================

def _save_lmm_frames(frames, data_dir):
    """Write each named LMM result frame to ``data_dir/{name}.csv``.

    Pure save step factored out of :func:`plot_lmm_figures` for testability.
    Writes one CSV per entry, keyed by the dict's names (which follow the
    ``response_lmm_{family}_{output}[_{qualifier}]`` convention), and no other
    files.

    Parameters
    ----------
    frames : dict[str, pandas.DataFrame]
        Mapping of output base-name (no extension) to the frame to save.
    data_dir : pathlib.Path
        Directory the CSVs are written to.
    """
    for name, frame in frames.items():
        frame.to_csv(data_dir / f'{name}.csv', index=False)


def plot_lmm_figures(group, magnitudes, figures_dir, data_dir,
                     response_col='response'):
    """Run the task-LMM suite via the formula-driven data-class methods, save
    each result as a CSV, and plot the labelled summaries.

    Reads model formulas from ``config.LMM_FORMULAS`` and passes flat
    ``{name: formula}`` dicts to the data class — no LMM logic here.
    ``task_reliability`` is keyed by event (reward only enters at feedback), so
    the base fit and reliability comparison run once per event with that event's
    set, scoped via ``events=[event]``. Produces:

    - per-event base-model summary (each event's ``task_reliability[event]
      ['full']``, cached under the shared key ``task_full``), annotated with its
      formula;
    - ceiling R² (``task_ceiling``);
    - out-of-sample (CV) and in-sample (jackknife) reliability ΔR² from the
      per-event ``task_reliability`` comparison sets: per-variable total
      contribution (main effect plus its interactions) and the interaction
      block.

    Parameters
    ----------
    group : PhotometrySessionGroup
        Holds the fit registry the effect frames are read back from.
    magnitudes : pandas.DataFrame
        The uncoded merged magnitude frame every fit's trial selection is
        taken from.
    figures_dir : Path
        Output directory for SVG files.
    data_dir : Path
        Output directory for CSV files.
    response_col : str
        Column name for the response magnitude.
    """
    group_by = ['target_NM', 'event']
    event_formulas = LMM_FORMULAS['task_reliability']

    # Base reporting model + reliability comparison run per event, because the
    # formula set is event-specific (reward only at feedback). Each event is
    # restricted via ``events=[event]`` so its per-event 'full' base fit caches
    # under the shared 'task_full' name without later events overwriting it.
    base_frames, cv_frames, jk_frames = [], [], []
    for event, formulas in event_formulas.items():
        base_frames.append(group.response_lmm_fit(
            magnitudes, {'task_full': formulas['full']}, group_by,
            events=[event]))
        cv_frames.append(group.response_lmm_crossval(
            magnitudes, formulas, group_by, events=[event]))
        jk_frames.append(group.response_lmm_jackknife(
            magnitudes, formulas, group_by, events=[event]))
    r2_base = pd.concat(base_frames, ignore_index=True)
    if r2_base.empty:
        print("  No LMM results.")
        return
    coefficients = group.response_lmm_effects(magnitudes, 'task_full',
                                              'coefficients')
    # Bottom-row panels: main-effect EMMs (predicted mean ± CI) per factor.
    # Events whose model omits reward yield flat reward EMMs (the panel is
    # blank, not an error).
    emm_frames = {f: group.response_lmm_effects(magnitudes, 'task_full',
                                                'emm', [f])
                  for f in ('reward', 'side', 'contrast')}

    # Ceiling: per-event saturated reporting model (reward only at feedback),
    # run per event like the reliability set. plot_lmm_ceiling reads
    # marginal/conditional R², so rename the fit frame's R² columns.
    ceiling = pd.concat(
        [group.response_lmm_fit(magnitudes, cset, group_by, events=[event])
         for event, cset in LMM_FORMULAS['task_ceiling'].items()],
        ignore_index=True)
    ceiling = ceiling.rename(
        columns={'marginal_r2': 'marginal', 'conditional_r2': 'conditional'})

    # Reliability: per-event comparison against that event's full model. The
    # contrast/side/reward predictors are each variable's total ΔR² (main effect
    # plus every interaction it joins); `interactions` is the interaction block.
    reliability_cv = pd.concat(cv_frames, ignore_index=True)
    reliability_jackknife = pd.concat(jk_frames, ignore_index=True)

    _save_lmm_frames({
        'response_lmm_task_coefficients': coefficients,
        'response_lmm_task_ceiling': ceiling,
        'response_lmm_task_reliability_cv': reliability_cv,
        'response_lmm_task_reliability_jackknife': reliability_jackknife,
    }, data_dir)
    print(f"  LMM suite CSVs saved to {data_dir}")

    # Per-event base-model summary, annotated with that event's formula.
    for event in sorted(r2_base['event'].unique()):
        base_formula = event_formulas[event]['full'].format(response=response_col)
        fig = plot_lmm_summary(r2_base, coefficients, emm_frames, event,
                               formula=base_formula)
        fig.savefig(figures_dir / f'response_lmm_task_summary_{event}.svg',
                    dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close(fig)

    fig = plot_lmm_ceiling(ceiling)
    fig.savefig(figures_dir / 'response_lmm_task_ceiling.svg',
                dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)

    # Reliability grids: out-of-sample (CV) and in-sample (jackknife) ΔR² per
    # predictor; both are leave-one-subject-out, named by procedure not "loso".
    # ΔR² is scaled to a proportion of the full model's in-sample marginal R²
    # (the base reporting fit), which is also annotated on each panel.
    full_r2 = r2_base[['target_NM', 'event', 'marginal_r2']]
    for df_reliability, qualifier, label in [
        (reliability_cv, 'cv', 'out-of-sample (cross-validated)'),
        (reliability_jackknife, 'jackknife', 'in-sample (jackknife)'),
    ]:
        if df_reliability.empty:
            continue
        fig = plot_lmm_reliability(
            df_reliability, full_r2,
            title=f'Task LMM reliability — {label}\nfolds are subjects')
        fig.savefig(
            figures_dir / f'response_lmm_task_reliability_{qualifier}.svg',
            dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close(fig)
    print("  LMM summary plots saved")


# =========================================================================
# Movement encoding
# =========================================================================

def _movement_reliability(group, magnitudes, group_by):
    """Stack cv and jackknife ΔR² across the per-movement-variable
    ``movement_<var>`` families. Each family is keyed by event (the revised task
    base is per-event), so each event runs its own set scoped via
    ``events=[event]``, tagged with the movement variable."""
    cv, jk = [], []
    for var in MOVEMENT_VARS:
        for event, formulas in LMM_FORMULAS[f'movement_{var}'].items():
            cv.append(group.response_lmm_crossval(
                magnitudes, formulas, group_by, events=[event],
                min_subjects=MIN_SUBJECTS_MOVEMENT,
                min_trials=MIN_TRIALS_MOVEMENT).assign(movement_var=var))
            jk.append(group.response_lmm_jackknife(
                magnitudes, formulas, group_by, events=[event],
                min_subjects=MIN_SUBJECTS_MOVEMENT,
                min_trials=MIN_TRIALS_MOVEMENT).assign(movement_var=var))
    return (pd.concat(cv, ignore_index=True),
            pd.concat(jk, ignore_index=True))


def _movement_r2(group, magnitudes, group_by):
    """Per-model in-sample marginal R² of the ``movement_<var>`` families.

    Each family is keyed by event; keys are renamed ``<name>_<var>`` so cached
    fits don't collide across movement variables, then stripped back to the
    family keys; :func:`plot_movement_r2_bars` reads the
    ``full``/``contrast``/``movement`` subset.
    """
    rows = []
    for var in MOVEMENT_VARS:
        for event, family in LMM_FORMULAS[f'movement_{var}'].items():
            formulas = {f'{name}_{var}': formula
                        for name, formula in family.items()}
            r2 = group.response_lmm_fit(magnitudes, formulas, group_by,
                                        events=[event],
                                        min_subjects=MIN_SUBJECTS_MOVEMENT)
            r2['name'] = r2['name'].str.replace(f'_{var}$', '', regex=True)
            rows.append(r2.assign(movement_var=var))
    return pd.concat(rows, ignore_index=True)


def plot_movement_figures(group, magnitudes, fig_dirs, data_dir):
    """Movement-encoding analyses over the response events (``RESPONSE_EVENTS``):
    cv/jackknife reliability ΔR² per movement variable (analogous to the task
    reliability plots), the three-bar in-sample R² comparison, and the movement
    ceiling (saturated 3-way of the movement predictors, per event)."""
    group_by = ['target_NM', 'event']

    reliability_cv, reliability_jk = _movement_reliability(
        group, magnitudes, group_by)
    r2 = _movement_r2(group, magnitudes, group_by)

    # Movement ceiling: saturated 3-way of the movement predictors, fit per
    # (target_NM, event). Renamed to marginal/conditional for plot_lmm_ceiling.
    ceiling = group.response_lmm_fit(
        magnitudes, LMM_FORMULAS['movement_ceiling'], group_by,
        min_subjects=MIN_SUBJECTS_MOVEMENT).rename(
        columns={'marginal_r2': 'marginal', 'conditional_r2': 'conditional'})

    _save_lmm_frames({
        'response_lmm_movement_reliability_cv': reliability_cv,
        'response_lmm_movement_reliability_jackknife': reliability_jk,
        'response_lmm_movement_r2': r2,
        'response_lmm_movement_ceiling': ceiling,
    }, data_dir)
    print(f"  Movement LMM CSVs saved to {data_dir}")

    # Reliability ΔR²: one figure per (procedure, movement variable), mirroring
    # the task reliability figure (target_NM × event grid). ΔR² is scaled to a
    # proportion of that variable's full-model in-sample marginal R² (the
    # `full` rows of the r2 frame), which is also annotated on each panel.
    for proc, df_rel in (('cv', reliability_cv), ('jackknife', reliability_jk)):
        for var in MOVEMENT_VARS:
            sub = df_rel[df_rel['movement_var'] == var]
            if sub.empty:
                continue
            full_r2 = r2[(r2['name'] == 'full') & (r2['movement_var'] == var)][
                ['target_NM', 'event', 'marginal_r2']]
            fig = plot_lmm_reliability(
                sub, full_r2,
                title=f'Movement LMM reliability ({var}) — {proc}\n'
                      'folds are subjects')
            fig.savefig(
                fig_dirs['movement_model_comparison']
                / f'response_lmm_movement_reliability_{proc}_{var}.svg',
                dpi=FIGURE_DPI, bbox_inches='tight')
            plt.close(fig)

    # Three-bar in-sample R² comparison: one figure per movement event.
    r2_mv = r2[r2['event'].isin(RESPONSE_EVENTS)]
    for event, df_ev in r2_mv.groupby('event'):
        fig = plot_movement_r2_bars(df_ev)
        fig.savefig(
            fig_dirs['movement_model_comparison']
            / f'response_lmm_movement_r2_{event}.svg',
            dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close(fig)

    # Movement ceiling figure (per-event panels), mirroring the task ceiling.
    fig = plot_lmm_ceiling(
        ceiling, title='Movement ceiling R²\n'
                       'choice × reaction_time × peak_velocity')
    fig.savefig(
        fig_dirs['movement_model_comparison']
        / 'response_lmm_movement_ceiling.svg',
        dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)


# =========================================================================
# Per-recording OLS drop-one
# =========================================================================

# The trial exclusions this analysis applies, held in one dict so the rows the
# models are fitted on and the rows the stored magnitude table carries cannot
# drift apart: the fitting loop takes it per fiber x event and the link function
# re-applies it without one to read the session-wide view back.
PERSESSION_TRIAL_CRITERIA = {
    'exclude_nogo': True,
    'min_response_time': MIN_RESPONSE_TIME,
    'exclude_negative_reaction_time': True,
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


def fit_session(ps, formulas: dict, donors: dict) -> tuple[pd.DataFrame,
                                                           pd.DataFrame,
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
    formulas : dict
        Drop-one family, ``config.LMM_FORMULAS['persession']``.
    donors : dict
        The whole pass's donor pool, keyed by eid; the session narrows it to
        the sessions its ``donor_scope`` admits.

    Returns
    -------
    magnitudes : pandas.DataFrame
        The filtered magnitude view with its trials joined on and coded
        relative to each recording's hemisphere — one row per recording x event
        x trial, carrying the rows the models were fitted on.
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
    ps.extract_response_magnitudes()
    unfiltered = _join_trials(ps.masking_diagnostics(), ps.trials)
    fits = ps.fit_responses(formulas, donors, **PERSESSION_TRIAL_CRITERIA)
    ps.filter_trials(**PERSESSION_TRIAL_CRITERIA)
    magnitudes = _join_trials(ps.response_magnitudes, ps.trials)
    return magnitudes, fits, unfiltered


def varcomp_coefficients(ols_persession: pd.DataFrame) -> pd.DataFrame:
    """Per-session coefficients view of the merged per-recording OLS frame.

    The variance-components stage models one weight per session, which the
    merged frame already carries: the drop-one grain is one row per dropped
    predictor, and the reference model's weight for that same regressor sits on
    it. So the view is a rename and a column subset, no aggregation — the
    dropped ``predictor`` is the ``regressor`` whose weight the row holds.

    Parameters
    ----------
    ols_persession : pandas.DataFrame
        The merged per-recording OLS results, ``config.OLS_PERSESSION_COLUMNS``.

    Returns
    -------
    pandas.DataFrame
        ``config.RESPONSE_OLS_COEFS_COLUMNS``, the grain and schema
        :meth:`PhotometrySessionGroup.response_varcomp` consumes.
    """
    return (ols_persession.rename(columns={'predictor': 'regressor'})
            [RESPONSE_OLS_COEFS_COLUMNS])


def compute_masking_diagnostics(
    magnitudes: pd.DataFrame,
    window: tuple[float, float] = RESPONSE_WINDOWS['early'],
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


def plot_persession_figures(results: pd.DataFrame,
                            mouse_pvalues: pd.DataFrame | None, figures_dir,
                            display: str = 'session') -> None:
    """Save the per-session drop-one ΔR² and full-model R² figures.

    Scopes the merged per-recording OLS frame to ``RESPONSE_EVENTS`` (a cached
    frame may carry events since dropped from the analysis) and saves two
    figures from it: a drop-one ΔR² grid (dropped-regressor rows × event
    columns) and a full-model R² figure (its own y-axis). ``display`` selects
    how each session's values are drawn — per-session dots (``session``),
    per-subject median+IQR (``subject``), or a per-target violin (``target``) —
    via ``_PERSESSION_DISPLAY_FNS``; the SVG filenames are the same in every
    mode.

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
    results = results[results['event'].isin(RESPONSE_EVENTS)]

    dropone_kwargs = ({'mouse_pvalues': mouse_pvalues}
                      if display == 'session' else {})
    fig = dropone_fn(
        results,
        title='Per-session OLS drop-one ΔR²\nevery session is a point',
        **dropone_kwargs)
    fig.savefig(figures_dir / 'response_ols_persession_dropone.svg',
                dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)

    fig = total_r2_fn(
        results,
        title='Per-session full-model R²\nevery session is a point')
    fig.savefig(figures_dir / 'response_ols_persession_total_r2.svg',
                dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)
    print("  Per-session OLS drop-one and full-model R² figures saved")


# The frames --reprocess writes and the no-flag branch reads back, each under
# the name the plotting steps take it by.
RESULT_FPATHS = {
    'magnitudes': RESPONSE_MAGNITUDES_FPATH,
    'ols': OLS_PERSESSION_FPATH,
    'ols_mouse': RESPONSE_OLS_MOUSE_PVAL_FPATH,
    'varcomp_summary': RESPONSE_VARCOMP_SUMMARY_FPATH,
    'varcomp_violin': RESPONSE_VARCOMP_VIOLIN_FPATH,
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
    whichever grain each frame is keyed on, so the per-mouse and
    variance-components tables — keyed by cell, with no ``eid`` — come back
    whole.

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
                                 formulas=LMM_FORMULAS['persession'],
                                 donors=donors)
                   if frames is not None]

        magnitudes = (pd.concat([frames[0] for frames in returns],
                                ignore_index=True) if returns
                      else pd.DataFrame(columns=RESPONSE_MAGNITUDE_COLUMNS))
        if len(magnitudes) == 0:
            print(f"No response magnitudes: all {len(group.sessions)} "
                  "sessions returned nothing. Check the logged errors.")
            raise SystemExit(1)
        magnitudes = magnitudes[RESPONSE_MAGNITUDE_COLUMNS]

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

        # --- Per-cell variance components (mouse vs session) ---
        print("Fitting per-cell variance-components model (PyMC sampling)...")
        varcomp_summary, varcomp_violin = group.response_varcomp(
            varcomp_coefficients(ols),
            mcmc=VARCOMP_MCMC, tau_prior=VARCOMP_TAU_PRIOR,
            min_mice=VARCOMP_MIN_MICE,
            min_sessions_per_mouse=VARCOMP_MIN_SESSIONS_PER_MOUSE,
            grid_size=VARCOMP_KDE_GRID, hdi_prob=VARCOMP_HDI_PROB)
        varcomp_summary.to_parquet(RESPONSE_VARCOMP_SUMMARY_FPATH, index=False)
        varcomp_violin.to_parquet(RESPONSE_VARCOMP_VIOLIN_FPATH, index=False)
        print(f"Saved variance components to {RESPONSE_VARCOMP_SUMMARY_FPATH} "
              f"and {RESPONSE_VARCOMP_VIOLIN_FPATH}")

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
        varcomp_violin = frames['varcomp_violin']
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
    anova_results = group.response_anovaRM_fit(magnitudes)
    if anova_results:
        all_tables = []
        for (tnm, ev), table in anova_results.items():
            print(f"\n  {tnm} x {ev} (method: {table['method'].iloc[0]}):")
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

    # =====================================================================
    # Variance components: mouse vs session posterior violins
    # =====================================================================
    print("\nGenerating variance-components violin figure...")
    fig = plot_varcomp_violins(
        varcomp_violin,
        title='Per-cell variance components\nmouse (left) vs session (right)')
    fig.savefig(fig_dirs['persession'] / 'response_varcomp_violins.svg',
                dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"Variance-components figure saved to {fig_dirs['persession']}")
