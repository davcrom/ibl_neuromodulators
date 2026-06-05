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
    python scripts/responses.py          # full pipeline: extract + plot
    python scripts/responses.py --plot   # plot only from existing parquet files
"""
import argparse

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from iblnm.config import (
    PROJECT_ROOT, SESSIONS_FPATH, PERFORMANCE_FPATH,
    QUERY_DATABASE_LOG_FPATH, PHOTOMETRY_LOG_FPATH, TASK_LOG_FPATH,
    RESPONSES_DIR, RESPONSES_FPATH, TRIAL_REGRESSORS_FPATH,
    RESPONSE_MATRIX_FPATH, MEAN_TRACES_FPATH,
    RESPONSE_EVENTS, FIGURE_DPI,
    ANALYSIS_QC_BLOCKERS, TARGETNMS_TO_ANALYZE,
    TIMING_VARS, MIN_SUBJECTS_MOVEMENT, MIN_TRIALS_MOVEMENT,
)
from iblnm.data import PhotometrySessionGroup
from iblnm.io import _get_default_connection
from iblnm.util import collect_session_errors
from iblnm.vis import (
    plot_relative_contrast,
    plot_mean_response_vectors, plot_lmm_summary,
    plot_mean_response_traces,
    plot_movement_response, plot_movement_lmm_summary,
    plot_movement_r2_bars, plot_movement_slope_summary,
)
from iblnm.analysis import (
    split_features_by_event,
    fit_movement_lmm_r2, jackknife_movement_lmm,
    fit_movement_vs_contrast, fit_movement_predicts_response,
    fit_movement_within_contrast, within_contrast_variation,
)

plt.ion()


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

def plot_lmm_figures(group, figures_dir, data_dir,
                     response_col='response'):
    """Generate per-target LMM response plots and consolidated summaries.

    Parameters
    ----------
    group : PhotometrySessionGroup
        Must have lmm_results populated.
    figures_dir : Path
        Output directory for SVG files.
    data_dir : Path
        Output directory for CSV files.
    response_col : str
        Column name for the response magnitude.
    """
    window_label = response_col

    if not group.lmm_results:
        print("  No LMM results to plot.")
        return

    # FIXME: saving doesn't belong in a plot function
    # Save coefficients to data dir
    if len(group.lmm_coefficients) > 0:
        csv_path = data_dir / f'lmm_coefficients_{window_label}.csv'
        group.lmm_coefficients.to_csv(csv_path, index=False)
        print(f"  LMM coefficients saved to {csv_path}")

    # Consolidated summary — one figure per event
    events_seen = sorted(set(ev for _, ev in group.lmm_results.keys()))
    for event_label in events_seen:
        fig = plot_lmm_summary(group, event_label)
        fig.savefig(
            figures_dir / f'lmm_summary_{event_label}_{window_label}.svg',
            dpi=FIGURE_DPI, bbox_inches='tight')
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

# Response DV set for the movement claims: pre-stimulus baseline, stimulus
# onset, and first-movement aligned magnitudes (feedback excluded).
MOVEMENT_EVENTS = ('baseline', 'stimOn_times', 'firstMovement_times')


def build_movement_df(group):
    """Merge response magnitudes with trial regressors for movement modeling.

    Keeps unbiased-block go trials with ``response_time > 0.05`` and a
    non-null response (via :meth:`_modeling_frame`), restricts the response DV
    set to the ``baseline``, ``stimOn``, and ``firstMovement`` events (feedback
    excluded), adds hemisphere-relative contrast/side, and appends a
    ``log_<var>`` column per timing variable (NaN where the value is ≤ 0).
    """
    df = group._modeling_frame()
    df = df[df['event'].isin(MOVEMENT_EVENTS)].copy()
    for var in TIMING_VARS:
        df[f'log_{var}'] = np.where(df[var] > 0, np.log10(df[var]), np.nan)
    return df


def _movement_descriptive_figures(df_resp, figures_dir):
    """Raw-data within-contrast check: NM response vs log(timing) per
    (target_NM, event, predictor), with contrast as color."""
    for (target_nm, event), df_group in df_resp.groupby(['target_NM', 'event']):
        if df_group['subject'].nunique() < MIN_SUBJECTS_MOVEMENT:
            continue
        for var in TIMING_VARS:
            df_valid = df_group.dropna(subset=[f'log_{var}'])
            if len(df_valid) < MIN_TRIALS_MOVEMENT:
                continue
            fig = plot_movement_response(
                df_valid, 'response', f'log_{var}', target_nm, event=event)
            fname = f'{target_nm}_{event}_{var}.svg'
            fig.savefig(figures_dir / fname, dpi=FIGURE_DPI, bbox_inches='tight')
            plt.close(fig)


def _fit_claim_over_grid(df, group_cols, fit_fn):
    """Apply a per-group movement fit ``fit_fn(df_group, timing_col)`` across
    every (group × timing variable) cell, tag each tidy result with its group
    identifiers, and concatenate. Empty fits (too few subjects / non-convergence)
    contribute their columns but no rows."""
    frames = []
    for keys, df_group in df.groupby(group_cols):
        keys = keys if isinstance(keys, tuple) else (keys,)
        for var in TIMING_VARS:
            res = fit_fn(df_group, f'log_{var}')
            for col, val in zip(reversed(group_cols), reversed(keys)):
                res.insert(0, col, val)
            frames.append(res)
    # Drop empty fits before concat (avoids the all-NA concat dtype warning);
    # fall back to the first (schema-carrying) frame if every fit was empty.
    nonempty = [f for f in frames if len(f)]
    if nonempty:
        return pd.concat(nonempty, ignore_index=True)
    return frames[0] if frames else pd.DataFrame()


def _fit_movement_claims(df_resp, figures_dir, data_dir):
    """Fit the three movement claims plus the within-contrast variation check off
    the shared frame, save one tidy CSV per claim, and plot the slope summaries.

    Behavioral claims (movement vs contrast; within-contrast variation) use one
    row per trial (the stimOn event); the response claims (unadjusted and
    within-contrast) run per (target_NM, event, timing variable).
    """
    behavioral = df_resp[df_resp['event'] == 'stimOn_times']
    vs_contrast = _fit_claim_over_grid(
        behavioral, ['target_NM'], fit_movement_vs_contrast)
    variation = _fit_claim_over_grid(
        behavioral, ['target_NM'], within_contrast_variation)
    predicts = _fit_claim_over_grid(
        df_resp, ['target_NM', 'event'],
        lambda df, col: fit_movement_predicts_response(df, 'response', col))
    within = _fit_claim_over_grid(
        df_resp, ['target_NM', 'event'],
        lambda df, col: fit_movement_within_contrast(df, 'response', col))

    vs_contrast.to_csv(data_dir / 'movement_vs_contrast.csv', index=False)
    variation.to_csv(data_dir / 'within_contrast_variation.csv', index=False)
    predicts.to_csv(data_dir / 'movement_predicts_response.csv', index=False)
    within.to_csv(data_dir / 'movement_within_contrast.csv', index=False)

    for tidy, title, formula, fname in [
        (vs_contrast, 'Movement vs contrast', 'timing ~ contrast',
         'movement_vs_contrast.svg'),
        (predicts, 'Movement predicts response (unadjusted)', 'response ~ timing',
         'movement_predicts_response.svg'),
        (within, 'Movement predicts response within contrast',
         'response ~ C(contrast) + timing + side + reward',
         'movement_within_contrast.svg'),
    ]:
        fig = plot_movement_slope_summary(tidy, title, formula=formula)
        fig.savefig(figures_dir / fname, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close(fig)


def _movement_model_comparison(df_resp, figures_dir, data_dir):
    """Jackknife ΔR² (dots) and full-dataset marginal R² (bars) per
    (target_NM, movement variable)."""
    jackknife_frames = []
    bar_rows = []
    for target_nm, df_tnm in df_resp.groupby('target_NM'):
        for var in TIMING_VARS:
            df_valid = df_tnm.dropna(subset=[f'log_{var}'])
            if len(df_valid) < MIN_TRIALS_MOVEMENT:
                continue
            df_jk = jackknife_movement_lmm(df_valid, 'response', f'log_{var}')
            if not df_jk.empty:
                df_jk['target_NM'] = target_nm
                jackknife_frames.append(df_jk)
            full = fit_movement_lmm_r2(df_valid, 'response', f'log_{var}')
            if full is not None:
                bar_rows.append({'target_NM': target_nm,
                                 'timing_col': f'log_{var}', **full})

    if not jackknife_frames:
        return
    df_jk_all = pd.concat(jackknife_frames, ignore_index=True)
    df_jk_all.to_csv(data_dir / 'jackknife_model_comparison.csv', index=False)
    df_bars = pd.DataFrame(bar_rows)
    df_bars.to_csv(data_dir / 'movement_marginal_r2.csv', index=False)

    fig = plot_movement_lmm_summary(df_jk_all)
    fig.savefig(figures_dir / 'model_comparison.svg',
                dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)
    fig = plot_movement_r2_bars(df_bars)
    fig.savefig(figures_dir / 'r2_model_comparison.svg',
                dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)


def plot_movement_figures(group, fig_dirs, data_dir):
    """Run the movement-encoding analyses off the merged modeling frame: the
    ordered claims, the raw-data within-contrast check, and the ΔR²
    unique-contribution comparison (stimOn only, unchanged)."""
    df_resp = build_movement_df(group)
    _fit_movement_claims(df_resp, fig_dirs['movement_slopes'], data_dir)
    _movement_descriptive_figures(df_resp, fig_dirs['movement_descriptive'])
    _movement_model_comparison(
        df_resp.query("event == 'stimOn_times'"),
        fig_dirs['movement_model_comparison'], data_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--plot', action='store_true',
                        help='skip extraction; plot from existing parquet files')
    parser.add_argument('--contrast-coding',
                        choices=['log', 'log2', 'linear', 'rank'],
                        default='log2',
                        help='contrast transform for LMM (default: log2)')
    args = parser.parse_args()

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
        'movement_slopes': fig_base / 'movement/slopes',
    }
    for d in fig_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # =====================================================================
    # Load sessions and create group
    # =====================================================================
    print(f"Loading sessions from {SESSIONS_FPATH}")
    df = pd.read_parquet(SESSIONS_FPATH)
    df = collect_session_errors(
        df, [QUERY_DATABASE_LOG_FPATH, PHOTOMETRY_LOG_FPATH, TASK_LOG_FPATH])
    if PERFORMANCE_FPATH.exists():
        perf = pd.read_parquet(
            PERFORMANCE_FPATH, columns=['eid', 'fraction_correct', 'contrasts'])
        df = df.merge(perf, on='eid', how='left')

    one = _get_default_connection()
    group = PhotometrySessionGroup.from_catalog(df, one=one)
    group.filter_sessions(
        session_types=('biased', 'ephys'),
        qc_blockers=ANALYSIS_QC_BLOCKERS,
        targetnms=TARGETNMS_TO_ANALYZE,
    )
    dup_log = group.deduplicate()
    print(f"  Deduplicated ({len(dup_log)} true-duplicate groups resolved)")
    print(f"  Recordings (session x region): {len(group)}")

    if args.plot:
        # =================================================================
        # Plot-only mode: load pre-existing parquet files
        # =================================================================
        if not RESPONSES_FPATH.exists():
            print(f"Error: {RESPONSES_FPATH} not found. Run without --plot first.")
            raise SystemExit(1)
        if not RESPONSE_MATRIX_FPATH.exists():
            print(f"Error: {RESPONSE_MATRIX_FPATH} not found. Run without --plot first.")
            raise SystemExit(1)
        if not TRIAL_REGRESSORS_FPATH.exists():
            print(f"Error: {TRIAL_REGRESSORS_FPATH} not found. Run without --plot first.")
            raise SystemExit(1)

        group.load_response_magnitudes(RESPONSES_FPATH)
        group.load_trial_regressors(TRIAL_REGRESSORS_FPATH)
        group.load_response_features(RESPONSE_MATRIX_FPATH)
        group.load_mean_traces(MEAN_TRACES_FPATH)

    else:
        # =================================================================
        # Full pipeline: extract from H5 files
        # =================================================================

        # --- Load traces cache ---
        print("\nLoading response traces...")
        group.load_response_traces()

        # --- Response magnitudes ---
        print("Computing trial-level response magnitudes...")
        group.get_response_magnitudes()

        if len(group.response_magnitudes) == 0:
            print("No response magnitudes extracted. Check H5 files exist.")
            raise SystemExit(1)

        group.response_magnitudes.to_parquet(RESPONSES_FPATH, index=False)
        print(f"Saved response magnitudes to {RESPONSES_FPATH}")

        # --- Trial regressors ---
        print("Collecting trial regressors...")
        group.get_trial_regressors()
        group.trial_regressors.to_parquet(TRIAL_REGRESSORS_FPATH, index=False)
        print(f"Saved trial regressors to {TRIAL_REGRESSORS_FPATH}")

        # --- Mean traces ---
        print("Computing mean traces...")
        group.get_mean_traces()
        group.mean_traces.to_parquet(MEAN_TRACES_FPATH, index=False)
        print(f"Saved mean traces to {MEAN_TRACES_FPATH}")

        # --- Response vectors ---
        print("\nBuilding response features...")
        group.get_response_features(nan_handling='drop_features')

        if len(group.response_features) == 0:
            print("No response vectors extracted. Check H5 files.")
            raise SystemExit(1)

        group.response_features.to_parquet(RESPONSE_MATRIX_FPATH)
        print(f"Saved {len(group.response_features)} response vectors "
              f"to {RESPONSE_MATRIX_FPATH}")

    # =====================================================================
    # Mean response traces per target-NM (first figures)
    # =====================================================================
    if group.mean_traces is not None and len(group.mean_traces) > 0:
        print("\nGenerating mean response trace plots...")
        traces_df = group.mean_traces
        targets = sorted(traces_df['target_NM'].unique())
        for target in targets:
            fig = plot_mean_response_traces(traces_df, target)
            fname = f'mean_traces_{target.replace("-", "_")}.svg'
            fig.savefig(fig_dirs['traces'] / fname,
                        dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Trace figures saved to {fig_dirs['traces']}")

    # Free trace cache
    group.flush_response_traces()

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
    anova_results = group.anova_response_magnitudes()
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
    # LMM statistical analysis
    # =====================================================================
    print(f"\nFitting linear mixed-effects models (contrast coding: {args.contrast_coding})...")
    group.fit_lmm(contrast_coding=args.contrast_coding)
    for (tnm, ev), result in group.lmm_results.items():
        ve = result.variance_explained
        print(f"  {tnm} x {ev}: R2 marginal={ve['marginal']:.3f}, "
              f"conditional={ve['conditional']:.3f}")
    plot_lmm_figures(group, fig_dirs['lmm'], data_dir)

    # =====================================================================
    # Movement encoding (ordered claims, raw-data check, ΔR² comparison)
    # =====================================================================
    print("\nRunning movement-variable encoding analyses...")
    plot_movement_figures(group, fig_dirs, data_dir)
    print(f"Movement figures saved under {fig_base / 'movement'}")

    # =====================================================================
    # Response vectors: per-event similarity
    # =====================================================================
    print("\nComputing per-event response vector similarity...")
    plot_similarity_figures(group, fig_dirs['similarity'], data_dir)
    print(f"Similarity figures saved to {fig_dirs['similarity']}")
