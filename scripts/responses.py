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

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from iblnm.config import (
    PROJECT_ROOT, SESSIONS_FPATH,
    RESPONSES_DIR, RESPONSES_FPATH, TRIAL_TIMING_FPATH, PEAK_VELOCITY_FPATH,
    RESPONSE_MATRIX_FPATH, MEAN_TRACES_FPATH,
    RESPONSE_EVENTS, FIGURE_DPI, ANALYSIS_CONTRASTS,
    MIN_TRAINING_PERFORMANCE, REQUIRED_CONTRASTS,
)
from iblnm.data import PhotometrySessionGroup
from iblnm.io import _get_default_connection
from iblnm.task import add_relative_contrast
from iblnm.util import derive_target_nm
from iblnm.vis import (
    plot_relative_contrast, plot_similarity_matrix,
    plot_mean_response_vectors, plot_empirical_similarity,
    plot_lmm_response, plot_lmm_summary,
    plot_within_target_similarity,
    plot_mean_response_traces,
    plot_wheel_lmm_summary,
    plot_glm_pca_weights, plot_glm_pca_scores,
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


def plot_response_figures(group, figures_dir, response_col='response_early',
                        aggregation='pool'):
    """Plot response magnitude by contrast x feedback x hemisphere.

    Parameters
    ----------
    group : PhotometrySessionGroup
        Must have group.response_magnitudes populated.
    figures_dir : Path
        Output directory for SVG files.
    response_col : str
        Column name for the response magnitude.
    aggregation : str
        'pool' or 'subject'.
    """
    df_responses = add_relative_contrast(group.response_magnitudes.copy())
    if group.trial_timing is not None:
        df_responses = df_responses.merge(
            group.trial_timing[['eid', 'trial', 'reaction_time']],
            on=['eid', 'trial'], how='left',
        )
    df_unbiased = df_responses.query('probabilityLeft == 0.5')

    window_label = response_col.replace('response_', '')
    df = df_unbiased.dropna(subset=[response_col]).copy()
    if 'reaction_time' in df.columns:
        df = df.query('choice != 0 and reaction_time > 0.05')
    else:
        df = df.query('choice != 0')

    for (target_nm, event), df_group in df.groupby(['target_NM', 'event']):
        n_subjects = df_group['subject'].nunique()
        if n_subjects < 2:
            continue

        event_label = event.replace('_times', '')
        fig = plot_relative_contrast(df_group, response_col, target_nm, event,
                                     window_label=window_label,
                                     aggregation=aggregation)
        fname = f'{target_nm}_{event_label}_{window_label}.svg'
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
    for (target_nm, event_label), result in group.lmm_results.items():
        event = event_label + '_times'
        df_group = df_raw[
            (df_raw['target_NM'] == target_nm) & (df_raw['event'] == event)
        ]
        fig = plot_lmm_response(
            result.predictions, target_nm, event,
            window_label=window_label,
            df_raw=df_group, response_col=response_col,
            contrast_coding=result.contrast_coding,
        )
        fname = f'{target_nm}_{event_label}_{window_label}_lmm.svg'
        fig.savefig(figures_dir / fname, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close(fig)

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

        sim = cosine_similarity_matrix(event_features)
        if len(sim) < 2:
            print(f"  [{event_stem}] Too few valid recordings, skipping")
            continue

        labels = pd.Series(
            sim.index.get_level_values('target_NM').values,
            index=sim.index,
        )
        subjects = rec_indexed['subject'].reindex(sim.index)

        # Full similarity matrix
        fig = plot_similarity_matrix(sim, labels, subjects=subjects)
        fig.savefig(similarity_dir / f'similarity_matrix_{event_stem}.svg',
                    dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close(fig)

        # Within / between summary
        wb = within_between_similarity(sim, labels)
        within_mean = wb[wb['comparison'] == 'within']['similarity'].mean()
        between_mean = wb[wb['comparison'] == 'between']['similarity'].mean()
        print(f"  [{event_stem}] Within: {within_mean:.3f}, "
              f"Between: {between_mean:.3f}")

        # Reduced target × target summary matrices
        target_sim = mean_similarity_by_target(sim, labels)
        target_sim_loso = mean_similarity_by_target(
            sim, labels, subjects=subjects)
        fig = plot_empirical_similarity(
            target_sim, loso_matrix=target_sim_loso)
        fig.savefig(
            similarity_dir / f'empirical_similarity_{event_stem}.svg',
            dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close(fig)

        # Within-target barplot
        fig = plot_within_target_similarity(sim, labels, subjects)
        fig.savefig(
            similarity_dir / f'within_target_similarity_{event_stem}.svg',
            dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close(fig)

        # Mean response vectors
        fig = plot_mean_response_vectors(event_features)
        fig.savefig(
            similarity_dir / f'mean_response_vectors_{event_stem}.svg',
            dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close(fig)

        # Save similarity matrix
        sim.to_parquet(data_dir / f'similarity_matrix_{event_stem}.pqt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--plot', action='store_true',
                        help='skip extraction; plot from existing parquet files')
    parser.add_argument('--contrast-coding', choices=['log', 'linear', 'rank'],
                        default='log',
                        help='contrast transform for LMM (default: log)')
    parser.add_argument('--cohort-weighted', action='store_true',
                        help='weight PCA by 1/n_k so each target contributes equally')
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
    # Load sessions
    # =====================================================================
    print(f"Loading sessions from {SESSIONS_FPATH}")
    df_sessions = pd.read_parquet(SESSIONS_FPATH)
    print(f"  Total sessions: {len(df_sessions)}")

    # Derive target_NM and NM from brain_region
    # (brain_region already filled and corrected in query_database.py)
    df_sessions = derive_target_nm(df_sessions)

    # TEMPFIX: drop sessions where brain_region, hemisphere, and target_NM
    # have different numbers of entries. This happens because brain_region
    # and hemisphere are populated independently in query_database.py, so
    # some sessions end up with e.g. brain_region=['VTA'] but hemisphere=[].
    # These cannot be exploded into one row per recording.
    _parallel_cols = ['target_NM', 'brain_region', 'hemisphere']
    _lengths_match = df_sessions[_parallel_cols].apply(
        lambda row: len(set(
            len(v) if isinstance(v, (list, np.ndarray)) else 1
            for v in row
        )) == 1,
        axis=1,
    )
    n_mismatched = (~_lengths_match).sum()
    if n_mismatched > 0:
        _bad = df_sessions.loc[~_lengths_match, ['eid', 'subject', 'brain_region', 'hemisphere']]
        print(f"  Dropping {n_mismatched} sessions with mismatched brain_region/hemisphere lengths:")
        for _, row in _bad.iterrows():
            print(f"    {row['subject']} {row['eid']}: "
                  f"brain_region={row['brain_region']}, hemisphere={row['hemisphere']}")
        df_sessions = df_sessions[_lengths_match].copy()

    # Explode to one row per recording (brain_region x hemisphere x target_NM)
    # Add fiber_idx to distinguish bilateral same-target recordings
    df_recordings = df_sessions.explode(_parallel_cols).copy()
    df_recordings['fiber_idx'] = df_recordings.groupby('eid').cumcount()

    # =====================================================================
    # Create group and apply standard filters
    # =====================================================================
    one = _get_default_connection()
    group = PhotometrySessionGroup(df_recordings, one=one)
    group.filter_recordings(
        session_types=('biased', 'ephys', 'training'),
        min_performance=MIN_TRAINING_PERFORMANCE,
        required_contrasts=REQUIRED_CONTRASTS,
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

        print(f"Loading response magnitudes from {RESPONSES_FPATH}")
        group.response_magnitudes = pd.read_parquet(RESPONSES_FPATH)

        if TRIAL_TIMING_FPATH.exists():
            print(f"Loading trial timing from {TRIAL_TIMING_FPATH}")
            group.trial_timing = pd.read_parquet(TRIAL_TIMING_FPATH)

        if PEAK_VELOCITY_FPATH.exists():
            print(f"Loading peak velocity from {PEAK_VELOCITY_FPATH}")
            group.peak_velocity = pd.read_parquet(PEAK_VELOCITY_FPATH)

        print(f"Loading response matrix from {RESPONSE_MATRIX_FPATH}")
        group.response_features = pd.read_parquet(RESPONSE_MATRIX_FPATH)

        if MEAN_TRACES_FPATH.exists():
            print(f"Loading mean traces from {MEAN_TRACES_FPATH}")
            group.mean_traces = pd.read_parquet(MEAN_TRACES_FPATH)

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

        group.trial_timing.to_parquet(TRIAL_TIMING_FPATH, index=False)
        print(f"Saved trial timing to {TRIAL_TIMING_FPATH}")

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
    # Filter to analysis contrasts (excludes 50%)
    # =====================================================================
    if group.response_magnitudes is not None:
        group.response_magnitudes = group.response_magnitudes[
            group.response_magnitudes['contrast'].isin(ANALYSIS_CONTRASTS)
        ].copy()

    # =====================================================================
    # Mean response traces per target-NM (first figures)
    # =====================================================================
    if group.mean_traces is not None and len(group.mean_traces) > 0:
        print("\nGenerating mean response trace plots...")
        traces_df = group.mean_traces[
            group.mean_traces['contrast'].isin(ANALYSIS_CONTRASTS)
        ]
        targets = sorted(traces_df['target_NM'].unique())
        for target in targets:
            fig = plot_mean_response_traces(traces_df, target)
            fname = f'mean_traces_{target.replace("-", "_")}.svg'
            fig.savefig(fig_dirs['traces'] / fname,
                        dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Trace figures saved to {fig_dirs['traces']}")

    # =====================================================================
    # Response magnitude plots
    # =====================================================================
    print_response_summary(group.response_magnitudes)

    print("\nGenerating response magnitude plots...")
    plot_response_figures(group, fig_dirs['contrast_curves'])
    print(f"Response magnitude figures saved to {fig_dirs['contrast_curves']}")

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
    # Wheel kinematics LMM
    # =====================================================================
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

    # =====================================================================
    # Response vectors: per-event similarity
    # =====================================================================
    print("\nComputing per-event response vector similarity...")
    plot_similarity_figures(group, fig_dirs['similarity'], data_dir)
    print(f"Similarity figures saved to {fig_dirs['similarity']}")

    # =====================================================================
    # Per-event GLM coefficient PCA
    # =====================================================================
    _w_label = 'cohort-weighted' if args.cohort_weighted else 'unweighted'
    print(f"\nPer-event GLM coefficient PCA ({_w_label})...")
    for event in RESPONSE_EVENTS:
        event_stem = event.replace('_times', '')
        print(f"\n  [{event_stem}] Fitting per-session GLMs...")
        pca_result = group.pca_glm_coefficients(
            event_name=event,
            contrast_coding=args.contrast_coding,
            n_components=3,
            cohort_weighted=args.cohort_weighted,
        )
        if pca_result is None:
            print(f"  [{event_stem}] No valid recordings, skipping")
            continue

        n_recs = pca_result.scores.shape[0]
        targets = sorted(set(pca_result.target_labels))
        print(f"  [{event_stem}] {n_recs} recordings, {len(targets)} targets")
        for i, pct in enumerate(pca_result.explained_variance_ratio):
            print(f"    PC{i+1}: {pct:.1%}")

        # Statistical tests on score distributions
        stats_df = pca_score_stats(pca_result)
        stats_df.to_csv(data_dir / f'pca_stats_{event_stem}.csv', index=False)
        for pc_i in range(pca_result.scores.shape[1]):
            kw = stats_df[(stats_df['pc'] == pc_i + 1)
                          & stats_df['target_a'].isna()]
            if len(kw):
                print(f"    PC{pc_i+1} KW: H={kw['kruskal_h'].iloc[0]:.2f}, "
                      f"p={kw['kruskal_p'].iloc[0]:.4f}")
            pw = stats_df[(stats_df['pc'] == pc_i + 1)
                          & stats_df['target_a'].notna()
                          & (stats_df['mwu_p'] < 0.05)]
            for _, row in pw.iterrows():
                print(f"      {row['target_a']} vs {row['target_b']}: "
                      f"U={row['mwu_u']:.0f}, p={row['mwu_p']:.4f}")

        fig = plot_glm_pca_weights(pca_result)
        fig.suptitle(f'GLM coefficient PCA — {event_stem}', fontsize=12)
        fig.savefig(fig_dirs['glm_pca'] / f'pca_weights_{event_stem}.svg',
                    dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close(fig)

        fig = plot_glm_pca_scores(pca_result, stats=stats_df)
        fig.suptitle(f'PC score distributions — {event_stem}', fontsize=12)
        fig.savefig(fig_dirs['glm_pca'] / f'pca_scores_{event_stem}.svg',
                    dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close(fig)

    print(f"GLM PCA figures saved to {fig_dirs['glm_pca']}")

    # Free trace cache
    group.flush_response_traces()
