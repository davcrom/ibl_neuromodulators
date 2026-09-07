"""
Response Analysis Pipeline

Extracts trial-level response magnitudes and recording-level response
vectors, then produces similarity, decoding, and contrast-based figures.

Includes biased, ephys, and qualifying training sessions (>70% performance
with the full contrast set).

Output:
    data/responses/                — all parquet and CSV data files
    figures/responses/             — all figures, organized by analysis

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
    RESPONSES_DIR, RESPONSE_MAGNITUDES_FPATH, TRIAL_REGRESSORS_FPATH,
    OLS_PERSESSION_FPATH, OLS_PERSESSION_COLUMNS,
    RESPONSE_OLS_MOUSE_PVAL_FPATH, RESPONSE_OLS_COEFS_COLUMNS,
    PERSESSION_FDR_GROUP_COLS,
    RESPONSE_VARCOMP_SUMMARY_FPATH, RESPONSE_VARCOMP_VIOLIN_FPATH,
    VARCOMP_MCMC, VARCOMP_TAU_PRIOR, VARCOMP_MIN_MICE,
    VARCOMP_MIN_SESSIONS_PER_MOUSE, VARCOMP_KDE_GRID, VARCOMP_HDI_PROB,
    RESPONSE_EVENTS, FIGURE_DPI, LMM_FORMULAS,
    MOVEMENT_VARS, MIN_SUBJECTS_MOVEMENT, MIN_TRIALS_MOVEMENT,
    PERSESSION_PVAL_N_BOOTSTRAP, PERSESSION_PVAL_SEED,
)
from iblnm.data import PhotometrySessionGroup
from iblnm.io import _get_default_connection
from iblnm.vis import (
    plot_relative_contrast,
    plot_mean_response_vectors, plot_lmm_summary,
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
from iblnm.analysis import (
    add_fdr_qvalues,
    split_features_by_event,
)


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


def plot_response_figures(group, figures_dir, response_col='response'):
    """Plot response magnitude by contrast x feedback x hemisphere.

    Produces two sets of plots per (target_NM, event):
      - ``_pool.svg``: grand mean over all trials ± SEM
      - ``_subject.svg``: mean of subject means ± SEM of subject means

    Parameters
    ----------
    group : PhotometrySessionGroup
        Must have group.response_magnitudes populated.
    figures_dir : Path
        Output directory for SVG files.
    response_col : str
        Column name for the response magnitude.
    """
    window_label = response_col
    df = group._modeling_frame(response_col)

    for (target_nm, event), df_group in df.groupby(['target_NM', 'event']):
        n_subjects = df_group['subject'].nunique()
        if n_subjects < 2:
            continue

        event_label = event.replace('_times', '')
        for aggregation, suffix in [('pool', '_pool'), ('subject', '_subject')]:
            fig = plot_relative_contrast(df_group, response_col, target_nm, event,
                                         window_label=window_label,
                                         aggregation=aggregation)
            fname = f'{target_nm}_{event_label}_{window_label}{suffix}.svg'
            fig.savefig(figures_dir / fname, dpi=FIGURE_DPI, bbox_inches='tight')
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


def plot_lmm_figures(group, figures_dir, data_dir, response_col='response'):
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
        Must have ``response_magnitudes`` and ``trial_regressors`` populated.
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
            {'task_full': formulas['full']}, group_by, events=[event]))
        cv_frames.append(group.response_lmm_crossval(
            formulas, group_by, events=[event]))
        jk_frames.append(group.response_lmm_jackknife(
            formulas, group_by, events=[event]))
    r2_base = pd.concat(base_frames, ignore_index=True)
    if r2_base.empty:
        print("  No LMM results.")
        return
    coefficients = group.response_lmm_effects('task_full', 'coefficients')
    # Bottom-row panels: main-effect EMMs (predicted mean ± CI) per factor.
    # Events whose model omits reward yield flat reward EMMs (the panel is
    # blank, not an error).
    emm_frames = {f: group.response_lmm_effects('task_full', 'emm', [f])
                  for f in ('reward', 'side', 'contrast')}

    # Ceiling: per-event saturated reporting model (reward only at feedback),
    # run per event like the reliability set. plot_lmm_ceiling reads
    # marginal/conditional R², so rename the fit frame's R² columns.
    ceiling = pd.concat(
        [group.response_lmm_fit(cset, group_by, events=[event])
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
# Response vectors plotting (per-event)
# =========================================================================

def plot_similarity_figures(group, similarity_dir, data_dir):
    """Plot per-event response vector similarity figures.

    For each event, produces:
    1. Mean response vectors (raw + normalized)
    2. Full recording × recording cosine similarity matrix
    3. Reduced target × target summary matrices (all pairs + cross-subject)
    4. Within-target similarity barplot

    Parameters
    ----------
    group : PhotometrySessionGroup
        Must have response_features populated.
    similarity_dir : Path
        Output directory for SVG files.
    data_dir : Path
        Output directory for parquet files.
    """
    features = group.response_features
    per_event = split_features_by_event(features)

    for event_stem, event_features in per_event.items():
        print(f"\n  [{event_stem}] {len(event_features.columns)} features, "
              f"{len(event_features)} recordings")

        # Mean response vectors
        fig = plot_mean_response_vectors(event_features)
        fig.savefig(
            similarity_dir / f'mean_response_vectors_{event_stem}.svg',
            dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close(fig)


# =========================================================================
# Movement encoding
# =========================================================================

def _movement_reliability(group, group_by):
    """Stack cv and jackknife ΔR² across the per-movement-variable
    ``movement_<var>`` families. Each family is keyed by event (the revised task
    base is per-event), so each event runs its own set scoped via
    ``events=[event]``, tagged with the movement variable."""
    cv, jk = [], []
    for var in MOVEMENT_VARS:
        for event, formulas in LMM_FORMULAS[f'movement_{var}'].items():
            cv.append(group.response_lmm_crossval(
                formulas, group_by, events=[event],
                min_subjects=MIN_SUBJECTS_MOVEMENT,
                min_trials=MIN_TRIALS_MOVEMENT).assign(movement_var=var))
            jk.append(group.response_lmm_jackknife(
                formulas, group_by, events=[event],
                min_subjects=MIN_SUBJECTS_MOVEMENT,
                min_trials=MIN_TRIALS_MOVEMENT).assign(movement_var=var))
    return (pd.concat(cv, ignore_index=True),
            pd.concat(jk, ignore_index=True))


def _movement_r2(group, group_by):
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
            r2 = group.response_lmm_fit(formulas, group_by, events=[event],
                                        min_subjects=MIN_SUBJECTS_MOVEMENT)
            r2['name'] = r2['name'].str.replace(f'_{var}$', '', regex=True)
            rows.append(r2.assign(movement_var=var))
    return pd.concat(rows, ignore_index=True)


def plot_movement_figures(group, fig_dirs, data_dir):
    """Movement-encoding analyses over the response events (``RESPONSE_EVENTS``):
    cv/jackknife reliability ΔR² per movement variable (analogous to the task
    reliability plots), the three-bar in-sample R² comparison, and the movement
    ceiling (saturated 3-way of the movement predictors, per event)."""
    group_by = ['target_NM', 'event']

    reliability_cv, reliability_jk = _movement_reliability(group, group_by)
    r2 = _movement_r2(group, group_by)

    # Movement ceiling: saturated 3-way of the movement predictors, fit per
    # (target_NM, event). Renamed to marginal/conditional for plot_lmm_ceiling.
    ceiling = group.response_lmm_fit(
        LMM_FORMULAS['movement_ceiling'], group_by,
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

# Recording-event-predictor key the three per-recording OLS results join on.
# brain_region is part of it: a bilateral session records two regions under one
# eid, and each is fitted separately.
_OLS_PERSESSION_KEYS = ['eid', 'brain_region', 'event', 'predictor']


def assemble_ols_persession(dropone, coefficients, session_pvalues):
    """Merge the per-recording OLS results into one frame, one row per fit.

    The drop-one fits set the grain and the row count; the coefficients and the
    permutation significance are joined onto them on
    ``(eid, brain_region, event, predictor)``, matching the coefficient table's
    ``regressor`` to the dropped ``predictor`` — the drop-one keys are the six
    regressor names. A recording-event-predictor with no scorable donor null
    keeps its fit and carries NaN significance rather than dropping out.

    Parameters
    ----------
    dropone : pandas.DataFrame
        Drop-one fits (``data.RESPONSE_OLS_DROPONE_COLUMNS``), whose ``r2`` and
        ``r2_adj`` are the reference model's and repeat across a
        recording-event's predictor rows; renamed ``r2_full`` / ``r2_full_adj``
        here to say so.
    coefficients : pandas.DataFrame
        Reference-model main-effect weights
        (``config.RESPONSE_OLS_COEFS_COLUMNS``), keyed by ``regressor``.
    session_pvalues : pandas.DataFrame
        Per-recording permutation significance
        (``data.RESPONSE_OLS_SESSION_PVAL_COLUMNS``) with ``q_value`` filled.
        Its own ``delta_r2`` copy is dropped in favour of the fitted one.

    Returns
    -------
    pandas.DataFrame
        ``config.OLS_PERSESSION_COLUMNS``, in that order.
    """
    coefs = (coefficients.rename(columns={'regressor': 'predictor'})
             [_OLS_PERSESSION_KEYS + ['coef', 'coef_se']])
    pvalues = session_pvalues[
        _OLS_PERSESSION_KEYS
        + ['delta_r2_null_median', 'p_value', 'q_value', 'n_donors']]
    merged = (dropone
              .rename(columns={'r2': 'r2_full', 'r2_adj': 'r2_full_adj'})
              .merge(coefs, on=_OLS_PERSESSION_KEYS, how='left')
              .merge(pvalues, on=_OLS_PERSESSION_KEYS, how='left'))
    return merged[OLS_PERSESSION_COLUMNS]


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


# display mode → (drop-one figure fn, full-model R² figure fn)
_PERSESSION_DISPLAY_FNS = {
    'session': (plot_ols_dropone, plot_ols_total_r2),
    'subject': (plot_ols_dropone_subject, plot_ols_total_r2_subject),
    'target': (plot_ols_dropone_violin, plot_ols_total_r2_violin),
}


def plot_persession_figures(group, figures_dir, display='session'):
    """Save the per-session drop-one ΔR² and full-model R² figures.

    Reads ``group.ols_persession`` (assembled under ``--reprocess`` or loaded
    from cache in the default run), scopes it to ``RESPONSE_EVENTS`` (a cached
    frame may carry events since dropped from the analysis), and saves two
    figures from it: a drop-one ΔR² grid (dropped-regressor rows × event
    columns) and a full-model R² figure (its own y-axis). ``display`` selects
    how each session's values are drawn — per-session dots (``session``),
    per-subject median+IQR (``subject``), or a per-target violin (``target``) —
    via ``_PERSESSION_DISPLAY_FNS``; the SVG filenames are the same in every
    mode.

    The frame carries each recording's own q-value, so the ``session`` mode
    figure colors its dots from it directly; ``group.ols_persession_mouse``, the
    coarser per-mouse grain, is threaded in beside it to color the subject mean
    dashes.

    Parameters
    ----------
    group : PhotometrySessionGroup
        Must have ``ols_persession`` populated, and ``ols_persession_mouse``
        when ``display='session'``.
    figures_dir : Path
        Output directory for the SVG figures.
    display : {'session', 'subject', 'target'}
        Per-session value display mode.
    """
    dropone_fn, total_r2_fn = _PERSESSION_DISPLAY_FNS[display]
    results = group.ols_persession
    results = results[results['event'].isin(RESPONSE_EVENTS)]

    dropone_kwargs = ({'mouse_pvalues': group.ols_persession_mouse}
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
        'lmm': fig_base / 'lmm',
        'similarity': fig_base / 'similarity',
        'target_decoding': fig_base / 'target_decoding',
        'traces': fig_base / 'traces',
        'movement_descriptive': fig_base / 'movement/descriptive',
        'movement_model_comparison': fig_base / 'movement/model_comparison',
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
        # Full pipeline: extract from H5 files and re-fit per-session models
        # =================================================================

        # --- Magnitudes and regressors, one pass over the store ---
        print("\nCollecting responses...")
        group.response_magnitudes, group.trial_regressors = (
            group.collect_responses())

        if len(group.response_magnitudes) == 0:
            print("No response magnitudes extracted. Check H5 files exist.")
            raise SystemExit(1)

        group.response_magnitudes.to_parquet(
            RESPONSE_MAGNITUDES_FPATH, index=False)
        group.trial_regressors.to_parquet(TRIAL_REGRESSORS_FPATH, index=False)
        print(f"Saved response magnitudes to {RESPONSE_MAGNITUDES_FPATH}")
        print(f"Saved trial regressors to {TRIAL_REGRESSORS_FPATH}")

        # --- Per-session drop-one fits and full-model coefficients ---
        # One set of coded frames serves the fits and the permutation null, so
        # both score the same rows and neither reopens the store.
        print("Coding per-session model frames...")
        model_frames = group.code_model_frames(LMM_FORMULAS['persession'])
        print(f"  Scorable recording-events: {len(model_frames)}")

        print("Fitting per-session drop-one OLS models...")
        group.response_ols_dropone_results, coefs_df = (
            group.response_ols_dropone(model_frames,
                                       LMM_FORMULAS['persession']))

        # --- Drop-one permutation significance, per session and per mouse ---
        print("Computing drop-one permutation p-values...")
        session_pvalues, mouse_pvalues = group.response_ols_dropone_permutation(
            model_frames, LMM_FORMULAS['persession'],
            n_bootstrap=PERSESSION_PVAL_N_BOOTSTRAP,
            random_state=PERSESSION_PVAL_SEED)
        # Correct each grain within (event, predictor): one family per grid cell.
        # The per-recording grain then merges into the fits and the coefficients
        # it shares a grain with; the per-mouse grain is coarser and stays apart.
        group.ols_persession = assemble_ols_persession(
            group.response_ols_dropone_results, coefs_df,
            add_fdr_qvalues(session_pvalues,
                            group_cols=PERSESSION_FDR_GROUP_COLS))
        group.ols_persession_mouse = add_fdr_qvalues(
            mouse_pvalues, group_cols=PERSESSION_FDR_GROUP_COLS)
        group.ols_persession.to_parquet(OLS_PERSESSION_FPATH, index=False)
        group.ols_persession_mouse.to_parquet(
            RESPONSE_OLS_MOUSE_PVAL_FPATH, index=False)
        print(f"Saved per-recording OLS results to {OLS_PERSESSION_FPATH} "
              f"and per-mouse p-values to {RESPONSE_OLS_MOUSE_PVAL_FPATH}")

        # --- Per-cell variance components (mouse vs session) ---
        print("Fitting per-cell variance-components model (PyMC sampling)...")
        group.response_varcomp_summary, group.response_varcomp_violin = (
            group.response_varcomp(
                varcomp_coefficients(group.ols_persession),
                mcmc=VARCOMP_MCMC, tau_prior=VARCOMP_TAU_PRIOR,
                min_mice=VARCOMP_MIN_MICE,
                min_sessions_per_mouse=VARCOMP_MIN_SESSIONS_PER_MOUSE,
                grid_size=VARCOMP_KDE_GRID, hdi_prob=VARCOMP_HDI_PROB))
        group.response_varcomp_summary.to_parquet(
            RESPONSE_VARCOMP_SUMMARY_FPATH, index=False)
        group.response_varcomp_violin.to_parquet(
            RESPONSE_VARCOMP_VIOLIN_FPATH, index=False)
        print(f"Saved variance components to {RESPONSE_VARCOMP_SUMMARY_FPATH} "
              f"and {RESPONSE_VARCOMP_VIOLIN_FPATH}")

    else:
        # =================================================================
        # Default: load pre-existing parquet files
        # =================================================================
        for fpath in (RESPONSE_MAGNITUDES_FPATH, TRIAL_REGRESSORS_FPATH,
                      OLS_PERSESSION_FPATH,
                      RESPONSE_OLS_MOUSE_PVAL_FPATH,
                      RESPONSE_VARCOMP_SUMMARY_FPATH,
                      RESPONSE_VARCOMP_VIOLIN_FPATH):
            if not fpath.exists():
                print(f"Error: {fpath} not found. Run with --reprocess first.")
                raise SystemExit(1)

        group.load_response_magnitudes(RESPONSE_MAGNITUDES_FPATH)
        group.load_trial_regressors(TRIAL_REGRESSORS_FPATH)
        group.load_ols_persession(OLS_PERSESSION_FPATH)
        group.load_ols_persession_mouse(RESPONSE_OLS_MOUSE_PVAL_FPATH)
        group.load_response_varcomp_summary(RESPONSE_VARCOMP_SUMMARY_FPATH)
        group.load_response_varcomp_violin(RESPONSE_VARCOMP_VIOLIN_FPATH)

    # =====================================================================
    # Response magnitude plots
    # =====================================================================
    print_response_summary(group.response_magnitudes)

    print("\nGenerating response magnitude plots...")
    plot_response_figures(group, fig_dirs['contrast_curves'])
    print(f"Response magnitude figures saved to {fig_dirs['contrast_curves']}")

    # =====================================================================
    # Repeated-measures ANOVA on subject means
    # =====================================================================
    print("\nRunning repeated-measures ANOVA on subject means...")
    anova_results = group.response_anovaRM_fit()
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
    plot_persession_figures(group, fig_dirs['persession'],
                            display=args.persession_display)
    print(f"Per-session OLS figures saved to {fig_dirs['persession']}")

    # =====================================================================
    # Variance components: mouse vs session posterior violins
    # =====================================================================
    print("\nGenerating variance-components violin figure...")
    fig = plot_varcomp_violins(
        group.response_varcomp_violin,
        title='Per-cell variance components\nmouse (left) vs session (right)')
    fig.savefig(fig_dirs['persession'] / 'response_varcomp_violins.svg',
                dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"Variance-components figure saved to {fig_dirs['persession']}")

    # =====================================================================
    # Response vectors: per-event similarity
    # =====================================================================
    # ~ print("\nComputing per-event response vector similarity...")
    # ~ plot_similarity_figures(group, fig_dirs['similarity'], data_dir)
    # ~ print(f"Similarity figures saved to {fig_dirs['similarity']}")
