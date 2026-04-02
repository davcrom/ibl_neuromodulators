"""
Response Analysis Pipeline

Extracts trial-level response magnitudes and recording-level response
vectors, then produces similarity, decoding, contrast-based, and wheel
kinematics figures.

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

import pandas as pd
from matplotlib import pyplot as plt

from iblnm.config import (
    PROJECT_ROOT, SESSIONS_FPATH, PERFORMANCE_FPATH,
    QUERY_DATABASE_LOG_FPATH, PHOTOMETRY_LOG_FPATH, TASK_LOG_FPATH,
    RESPONSES_DIR, RESPONSES_FPATH, TRIAL_TIMING_FPATH, PEAK_VELOCITY_FPATH,
    RESPONSE_MATRIX_FPATH, MEAN_TRACES_FPATH,
    RESPONSE_EVENTS, FIGURE_DPI,
    ANALYSIS_QC_BLOCKERS, SESSION_TYPES_TO_ANALYZE, TARGETNMS_TO_ANALYZE,
)
from iblnm.data import PhotometrySessionGroup
from iblnm.io import _get_default_connection
from iblnm.task import add_relative_contrast
from iblnm.util import collect_session_errors
from iblnm.vis import (
    plot_relative_contrast,
    plot_similarity_matrix,
    plot_mean_response_vectors, plot_empirical_similarity,
    plot_lmm_response, plot_lmm_summary,
    plot_within_target_similarity,
    plot_mean_response_traces,
    plot_wheel_lmm_summary,
    plot_glm_pca_summary,
)
from iblnm.analysis import (
    split_features_by_event,
    cosine_similarity_matrix, within_between_similarity,
    mean_similarity_by_target, pca_score_stats,
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


def plot_response_figures(group, figures_dir, response_col='response_early'):
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
    df_responses = add_relative_contrast(group.response_magnitudes.copy())
    if group.trial_timing is not None:
        df_responses = df_responses.merge(
            group.trial_timing[['eid', 'trial', 'response_time']],
            on=['eid', 'trial'], how='left',
        )

    window_label = response_col.replace('response_', '')
    df = df_responses.query('probabilityLeft == 0.5').dropna(subset=[response_col]).copy()
    df = df.query('choice != 0')
    if 'response_time' in df.columns:
        df = df.query('response_time > 0.05')

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
                     response_col='response_early'):
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
        Column name for the response magnitude (used for raw data overlay).
    """
    window_label = response_col.replace('response_', '')

    if not group.lmm_results:
        print("  No LMM results to plot.")
        return

    # Prepare raw data for overlay
    df_raw = add_relative_contrast(group.response_magnitudes.copy())
    if group.trial_timing is not None:
        df_raw = df_raw.merge(
            group.trial_timing[['eid', 'trial', 'reaction_time']],
            on=['eid', 'trial'], how='left',
        )
    df_raw = df_raw.query('probabilityLeft == 0.5')
    df_raw = df_raw.dropna(subset=[response_col])
    if 'reaction_time' in df_raw.columns:
        df_raw = df_raw.query('choice != 0 and reaction_time > 0.05')
    else:
        df_raw = df_raw.query('choice != 0')

    # Per-target modeled response plots (save and close)
    # ~ for (target_nm, event_label), result in group.lmm_results.items():
        # ~ event = event_label + '_times'
        # ~ df_group = df_raw[
            # ~ (df_raw['target_NM'] == target_nm) & (df_raw['event'] == event)
        # ~ ]
        # ~ fig = plot_lmm_response(
            # ~ result.predictions, target_nm, event,
            # ~ window_label=window_label,
            # ~ df_raw=df_group, response_col=response_col,
            # ~ contrast_coding=result.contrast_coding,
        # ~ )
        # ~ fname = f'{target_nm}_{event_label}_{window_label}_lmm.svg'
        # ~ fig.savefig(figures_dir / fname, dpi=FIGURE_DPI, bbox_inches='tight')
        # ~ plt.close(fig)

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
# Wheel kinematics LMM analysis
# =========================================================================

def plot_wheel_lmm_figures(group, figures_dir, data_dir):
    """Generate wheel kinematics LMM summary plot and CSV.

    Parameters
    ----------
    group : PhotometrySessionGroup
        Must have wheel_lmm_summary populated.
    figures_dir : Path
        Output directory for SVG files.
    data_dir : Path
        Output directory for CSV files.
    """
    summary = group.wheel_lmm_summary
    if summary is None or len(summary) == 0:
        print("  No wheel LMM results to plot.")
        return

    fig = plot_wheel_lmm_summary(summary)
    fig.savefig(figures_dir / 'wheel_lmm_delta_r2.svg',
                dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)

    csv_path = data_dir / 'wheel_lmm_summary.csv'
    summary.to_csv(csv_path, index=False)
    print(f"  Wheel LMM summary saved to {csv_path}")


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

    # Build subject lookup from recordings
    recs = group.recordings.copy()
    if 'fiber_idx' not in recs.columns:
        recs['fiber_idx'] = 0
    idx_cols = ['eid', 'target_NM', 'fiber_idx']
    rec_indexed = (
        recs[idx_cols + ['subject']]
        .drop_duplicates(subset=idx_cols)
        .set_index(idx_cols)
    )

    for event_stem, event_features in per_event.items():
        print(f"\n  [{event_stem}] {len(event_features.columns)} features, "
              f"{len(event_features)} recordings")

        # Mean response vectors
        fig = plot_mean_response_vectors(event_features)
        fig.savefig(
            similarity_dir / f'mean_response_vectors_{event_stem}.svg',
            dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close(fig)

        # # Similarity analysis (suppressed)
        # sim = cosine_similarity_matrix(event_features)
        # if len(sim) < 2:
        #     print(f"  [{event_stem}] Too few valid recordings, skipping")
        #     continue
        #
        # labels = pd.Series(
        #     sim.index.get_level_values('target_NM').values,
        #     index=sim.index,
        # )
        # subjects = rec_indexed['subject'].reindex(sim.index)
        #
        # # Full similarity matrix
        # fig = plot_similarity_matrix(sim, labels, subjects=subjects)
        # fig.savefig(similarity_dir / f'similarity_matrix_{event_stem}.svg',
        #             dpi=FIGURE_DPI, bbox_inches='tight')
        # plt.close(fig)
        #
        # # Within / between summary
        # wb = within_between_similarity(sim, labels)
        # within_mean = wb[wb['comparison'] == 'within']['similarity'].mean()
        # between_mean = wb[wb['comparison'] == 'between']['similarity'].mean()
        # print(f"  [{event_stem}] Within: {within_mean:.3f}, "
        #       f"Between: {between_mean:.3f}")
        #
        # # Reduced target × target summary matrices
        # target_sim = mean_similarity_by_target(sim, labels)
        # target_sim_loso = mean_similarity_by_target(
        #     sim, labels, subjects=subjects)
        # fig = plot_empirical_similarity(
        #     target_sim, loso_matrix=target_sim_loso)
        # fig.savefig(
        #     similarity_dir / f'empirical_similarity_{event_stem}.svg',
        #     dpi=FIGURE_DPI, bbox_inches='tight')
        # plt.close(fig)
        #
        # # Within-target barplot
        # fig = plot_within_target_similarity(sim, labels, subjects)
        # fig.savefig(
        #     similarity_dir / f'within_target_similarity_{event_stem}.svg',
        #     dpi=FIGURE_DPI, bbox_inches='tight')
        # plt.close(fig)
        #
        # # Save similarity matrix
        # sim.to_parquet(data_dir / f'similarity_matrix_{event_stem}.pqt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--plot', action='store_true',
                        help='skip extraction; plot from existing parquet files')
    parser.add_argument('--contrast-coding', choices=['log', 'linear', 'rank'],
                        default='rank',
                        help='contrast transform for LMM (default: log)')
    parser.add_argument('--cohort-weighted', action='store_true',
                        help='weight PCA by 1/n_k so each target contributes equally')
    parser.add_argument('--ica', action='store_true',
                        help='use ICA instead of PCA for GLM coefficient decomposition')
    args = parser.parse_args()

    # Create output directories
    data_dir = RESPONSES_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    fig_base = PROJECT_ROOT / 'figures/responses'
    fig_dirs = {
        'contrast_curves': fig_base / 'contrast_curves',
        'lmm': fig_base / 'lmm',
        'wheel_lmm': fig_base / 'wheel_lmm',
        'similarity': fig_base / 'similarity',
        'target_decoding': fig_base / 'target_decoding',
        'traces': fig_base / 'traces',
        'glm_pca': fig_base / 'glm_pca',
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
        session_types=SESSION_TYPES_TO_ANALYZE,
        qc_blockers=ANALYSIS_QC_BLOCKERS,
        targetnms=TARGETNMS_TO_ANALYZE,
    )
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

        group.load_response_magnitudes(RESPONSES_FPATH)
        group.load_trial_timing(TRIAL_TIMING_FPATH)
        group.load_peak_velocity(PEAK_VELOCITY_FPATH)
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

        if group.trial_timing is not None:
            group.trial_timing.to_parquet(TRIAL_TIMING_FPATH, index=False)
            print(f"Saved trial timing to {TRIAL_TIMING_FPATH}")

        if len(group.response_magnitudes) == 0:
            print("No response magnitudes extracted. Check H5 files exist.")
            raise SystemExit(1)

        group.response_magnitudes.to_parquet(RESPONSES_FPATH, index=False)
        print(f"Saved response magnitudes to {RESPONSES_FPATH}")

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
    # Response vectors: per-event similarity
    # =====================================================================
    print("\nComputing per-event response vector similarity...")
    plot_similarity_figures(group, fig_dirs['similarity'], data_dir)
    print(f"Similarity figures saved to {fig_dirs['similarity']}")

    # =====================================================================
    # Per-event GLM coefficient PCA
    # =====================================================================
    _decomp = 'ICA' if args.ica else 'PCA'
    _w_label = 'cohort-weighted' if args.cohort_weighted else 'unweighted'
    _decomp_label = f'{_decomp} ({_w_label})'
    print(f"\nPer-event GLM coefficient {_decomp_label}...")
    for event in RESPONSE_EVENTS:
        event_stem = event.replace('_times', '')
        print(f"\n  [{event_stem}] Fitting per-session GLMs...")
        if args.ica:
            decomp_result = group.ica_glm_coefficients(
                event_name=event,
                contrast_coding=args.contrast_coding,
                n_components=3,
                cohort_weighted=args.cohort_weighted,
            )
        else:
            decomp_result = group.pca_glm_coefficients(
                event_name=event,
                contrast_coding=args.contrast_coding,
                n_components=3,
                cohort_weighted=args.cohort_weighted,
            )
        if decomp_result is None:
            print(f"  [{event_stem}] No valid recordings, skipping")
            continue

        comp_label = 'IC' if args.ica else 'PC'
        n_recs = decomp_result.scores.shape[0]
        targets = sorted(set(decomp_result.target_labels))
        print(f"  [{event_stem}] {n_recs} recordings, {len(targets)} targets")
        for i, pct in enumerate(decomp_result.explained_variance_ratio):
            print(f"    {comp_label}{i+1}: {pct:.1%}")

        # Statistical tests on score distributions
        stats_df = pca_score_stats(decomp_result)
        stats_df.to_csv(
            data_dir / f'{_decomp.lower()}_stats_{event_stem}.csv', index=False)
        for pc_i in range(decomp_result.scores.shape[1]):
            kw = stats_df[(stats_df['pc'] == pc_i + 1)
                          & stats_df['target_a'].isna()]
            if len(kw):
                print(f"    {comp_label}{pc_i+1} KW: "
                      f"H={kw['kruskal_h'].iloc[0]:.2f}, "
                      f"p={kw['kruskal_p'].iloc[0]:.4f}")
            pw = stats_df[(stats_df['pc'] == pc_i + 1)
                          & stats_df['target_a'].notna()
                          & (stats_df['mwu_p'] < 0.05)]
            for _, row in pw.iterrows():
                print(f"      {row['target_a']} vs {row['target_b']}: "
                      f"U={row['mwu_u']:.0f}, p={row['mwu_p']:.4f}")

        fig = plot_glm_pca_summary(decomp_result, group.recordings,
                                    stats=stats_df,
                                    comp_label=comp_label)
        fig.suptitle(f'GLM coefficient {_decomp} — {event_stem}', fontsize=12)
        fig.savefig(
            fig_dirs['glm_pca'] / f'{_decomp.lower()}_summary_{event_stem}.svg',
            dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close(fig)

    print(f"GLM {_decomp} figures saved to {fig_dirs['glm_pca']}")

    # =====================================================================
    # Wheel kinematics LMM
    # =====================================================================
    """
    if group.peak_velocity is None:
        print("\nEnriching with peak velocity...")
        group.enrich_peak_velocity()
        group.peak_velocity.to_parquet(PEAK_VELOCITY_FPATH, index=False)
        print(f"Saved peak velocity to {PEAK_VELOCITY_FPATH}")

    n_with_wheel = group.peak_velocity['peak_velocity'].notna().sum()
    print(f"  {n_with_wheel} trials with wheel data")

    print("Fitting wheel kinematics LMMs...")
    group.fit_wheel_lmm()
    if len(group.wheel_lmm_summary) > 0:
        n_sig = (group.wheel_lmm_summary['lrt_pvalue'] < 0.05).sum()
        print(f"  {len(group.wheel_lmm_summary)} model comparisons, "
              f"{n_sig} significant (p < 0.05)")
        plot_wheel_lmm_figures(group, fig_dirs['wheel_lmm'], data_dir)
    else:
        print("  No wheel LMM results (insufficient data).")
    """
