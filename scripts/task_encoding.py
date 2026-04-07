"""
Task Encoding Analysis

Per-session GLM encoding model decomposed via PCA/ICA across recordings, and
per-cohort CCA between GLM-derived neural features and psychometric parameters.

Requires responses.py to have run first (loads pre-computed response
magnitudes and trial timing from disk).

Output:
    results/task_encoding/             — CSV data files
    figures/task_encoding/pca/         — PCA/ICA summary figures
    figures/task_encoding/cca/scatter/ — per-cohort CCA score scatter plots
    figures/task_encoding/cca/summary/ — CCA summary figures

Usage:
    python scripts/task_encoding.py
    python scripts/task_encoding.py --ica
    python scripts/task_encoding.py --n-permutations 5000
    python scripts/task_encoding.py --events stimOn_times feedback_times
    python scripts/task_encoding.py --plot-only
"""
import argparse

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from iblnm.config import (
    PROJECT_ROOT, SESSIONS_FPATH, PERFORMANCE_FPATH,
    QUERY_DATABASE_LOG_FPATH, PHOTOMETRY_LOG_FPATH, TASK_LOG_FPATH,
    RESPONSES_FPATH, TRIAL_TIMING_FPATH, TASK_ENCODING_DIR,
    RESPONSE_EVENTS, FIGURE_DPI, TARGETNM_COLORS,
    ANALYSIS_QC_BLOCKERS, SESSION_TYPES_TO_ANALYZE, TARGETNMS_TO_ANALYZE,
)
from iblnm.data import PhotometrySessionGroup
from iblnm.io import _get_default_connection
from iblnm.util import collect_session_errors
from iblnm.analysis import pca_score_stats
from iblnm.vis import plot_glm_pca_summary, plot_cohort_cca_summary

plt.ion()


DEFAULT_PARAMS = [
    'psych_50_threshold', 'psych_50_bias',
    'psych_50_lapse_left', 'psych_50_lapse_right',
]

# Grid search defaults for sparse CCA
ALPHA_GRID = [1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
L1_RATIO_GRID = [0.0]


def _event_label(event_name):
    """Strip '_times' suffix for display and filenames."""
    return event_name.replace('_times', '')


# =========================================================================
# PCA/ICA helpers
# =========================================================================

def run_decomposition(group, event, args, data_dir, fig_dir):
    """Fit PCA or ICA on GLM coefficients and save results.

    Parameters
    ----------
    group : PhotometrySessionGroup
    event : str
        Event name (e.g. 'stimOn_times').
    args : argparse.Namespace
        Must have: ica, contrast_coding, cohort_weighted.
    data_dir : Path
        Output directory for CSV files.
    fig_dir : Path
        Output directory for figures.

    Returns
    -------
    GLMPCAResult or None
    """
    event_stem = _event_label(event)
    decomp = 'ICA' if args.ica else 'PCA'
    comp_label = 'IC' if args.ica else 'PC'

    print(f"\n  [{event_stem}] Fitting per-session GLMs + {decomp}...")

    if args.ica:
        result = group.ica_glm_coefficients(
            event_name=event,
            contrast_coding=args.contrast_coding,
            n_components=3,
            cohort_weighted=args.cohort_weighted,
        )
    else:
        result = group.pca_glm_coefficients(
            event_name=event,
            contrast_coding=args.contrast_coding,
            n_components=3,
            cohort_weighted=args.cohort_weighted,
        )

    if result is None:
        print(f"  [{event_stem}] No valid recordings, skipping")
        return None

    n_recs = result.scores.shape[0]
    targets = sorted(set(result.target_labels))
    print(f"  [{event_stem}] {n_recs} recordings, {len(targets)} targets")
    for i, pct in enumerate(result.explained_variance_ratio):
        print(f"    {comp_label}{i+1}: {pct:.1%}")

    # Statistical tests
    stats_df = pca_score_stats(result)
    stats_df.to_csv(
        data_dir / f'{decomp.lower()}_stats_{event_stem}.csv', index=False)

    for pc_i in range(result.scores.shape[1]):
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

    # Plot
    fig = plot_glm_pca_summary(result, group.recordings,
                                stats=stats_df,
                                comp_label=comp_label)
    fig.suptitle(f'GLM coefficient {decomp} — {event_stem}', fontsize=12)
    fig.savefig(
        fig_dir / f'{decomp.lower()}_summary_{event_stem}.svg',
        dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)

    return result


# =========================================================================
# CCA helpers
# =========================================================================

def plot_cohort_scatter(cca_result, cohort_name, event_label):
    """Scatter x_scores vs y_scores for one cohort's CC1."""
    fig, ax = plt.subplots(figsize=(5, 5))
    color = TARGETNM_COLORS.get(cohort_name, 'gray')
    ax.scatter(cca_result.x_scores[:, 0], cca_result.y_scores[:, 0],
               c=color, alpha=0.7, edgecolors='white', s=40)
    r = cca_result.correlations[0]
    p = cca_result.p_values[0] if cca_result.p_values is not None else np.nan
    ax.set_xlabel('Neural canonical variate')
    ax.set_ylabel('Behavioral canonical variate')
    ax.set_title(f'{cohort_name} ({event_label}): r = {r:.3f}, p = {p:.4f}')
    fig.tight_layout()
    return fig


def save_cca_results(results, data_dir, label):
    """Save per-cohort CCAResult data to CSVs.

    Files written:
        {label}_summary.csv  — cohort, correlation, p_value, n, alpha, l1_ratio
        {label}_weights.csv  — cohort, view (neural/behavioral), feature, CC1
        {label}_scores.csv   — cohort, x_score, y_score
    """
    # Summary
    summary_rows = []
    for tnm, r in sorted(results.items()):
        summary_rows.append({
            'event': label,
            'cohort': tnm,
            'correlation': r.correlations[0],
            'p_value': r.p_values[0] if r.p_values is not None else np.nan,
            'n_recordings': r.n_recordings,
            'alpha': r.alpha if r.alpha is not None else np.nan,
            'l1_ratio': r.l1_ratio if r.l1_ratio is not None else np.nan,
        })
    pd.DataFrame(summary_rows).to_csv(
        data_dir / f'{label}_summary.csv', index=False)

    # Weights (neural and behavioral stacked)
    weight_rows = []
    for tnm, r in sorted(results.items()):
        for feat, val in r.x_weights['CC1'].items():
            weight_rows.append({
                'cohort': tnm, 'view': 'neural',
                'feature': feat, 'CC1': val,
            })
        for feat, val in r.y_weights['CC1'].items():
            weight_rows.append({
                'cohort': tnm, 'view': 'behavioral',
                'feature': feat, 'CC1': val,
            })
    pd.DataFrame(weight_rows).to_csv(
        data_dir / f'{label}_weights.csv', index=False)

    # Scores
    score_rows = []
    for tnm, r in sorted(results.items()):
        for i in range(r.x_scores.shape[0]):
            score_rows.append({
                'cohort': tnm,
                'x_score': r.x_scores[i, 0],
                'y_score': r.y_scores[i, 0],
            })
    pd.DataFrame(score_rows).to_csv(
        data_dir / f'{label}_scores.csv', index=False)


def load_cca_results(data_dir, label):
    """Reconstruct per-cohort CCAResult dict from saved CSVs.

    Returns
    -------
    results : dict[str, CCAResult]
    cp : pd.DataFrame
        Cross-projection correlations.
    ws : pd.DataFrame
        Weight cosine similarities.
    """
    from iblnm.analysis import CCAResult

    summary = pd.read_csv(data_dir / f'{label}_summary.csv')
    weights = pd.read_csv(data_dir / f'{label}_weights.csv')
    scores = pd.read_csv(data_dir / f'{label}_scores.csv')
    cp = pd.read_csv(data_dir / f'{label}_cross_projections.csv')
    ws = pd.read_csv(data_dir / f'{label}_weight_similarities.csv')

    results = {}
    for _, row in summary.iterrows():
        tnm = row['cohort']

        w = weights[weights['cohort'] == tnm]
        xw = w[w['view'] == 'neural'].set_index('feature')['CC1']
        yw = w[w['view'] == 'behavioral'].set_index('feature')['CC1']
        x_weights = pd.DataFrame({'CC1': xw})
        y_weights = pd.DataFrame({'CC1': yw})

        s = scores[scores['cohort'] == tnm]
        x_scores = s['x_score'].values[:, np.newaxis]
        y_scores = s['y_score'].values[:, np.newaxis]

        alpha = row['alpha'] if not np.isnan(row['alpha']) else None
        l1_ratio = row['l1_ratio'] if not np.isnan(row['l1_ratio']) else None
        p_val = row['p_value'] if not np.isnan(row['p_value']) else None

        results[tnm] = CCAResult(
            x_weights=x_weights,
            y_weights=y_weights,
            x_scores=x_scores,
            y_scores=y_scores,
            correlations=np.array([row['correlation']]),
            p_values=np.array([p_val]) if p_val is not None else None,
            n_recordings=int(row['n_recordings']),
            n_permutations=0,
            alpha=alpha,
            l1_ratio=l1_ratio,
        )

    return results, cp, ws


def run_cca(group, event, args, data_dir, scatter_dir, summary_dir):
    """Fit per-cohort CCA and save results.

    Parameters
    ----------
    group : PhotometrySessionGroup
    event : str
        Event name (e.g. 'stimOn_times').
    args : argparse.Namespace
        Must have: weight_by_se, contrast_coding, n_permutations, seed,
        sparse, unit_norm, params.
    data_dir : Path
        Output directory for CSV files.
    scatter_dir : Path
        Output directory for scatter plots.
    summary_dir : Path
        Output directory for summary figures.
    """
    label = _event_label(event)
    params = args.params if args.params else DEFAULT_PARAMS

    # If weight_by_se, re-fit GLM with t-statistics for CCA
    if args.weight_by_se:
        print(f"  Re-fitting GLM with t-statistics for CCA...")
        group.get_glm_response_features(
            event_name=event, weight_by_se=True,
            contrast_coding=args.contrast_coding,
        )

    # Align psychometric features to GLM features
    group.response_features = group.glm_response_features
    group.get_psychometric_features(params=params)
    n_valid = group.psychometric_features.notna().all(axis=1).sum()
    print(f"  {n_valid}/{len(group.psychometric_features)} recordings "
          "with complete psychometric data")

    # Fit per-cohort CCA
    cca_type = 'sparse CCA' if args.sparse else 'CCA'
    print(f"  Fitting per-cohort {cca_type} "
          f"({args.n_permutations} permutations)...")
    cca_kwargs = dict(
        n_permutations=args.n_permutations,
        seed=args.seed,
        sparse=args.sparse,
    )
    if args.sparse:
        cca_kwargs['alpha'] = ALPHA_GRID
        cca_kwargs['l1_ratio'] = L1_RATIO_GRID
        cca_kwargs['unit_norm'] = args.unit_norm
    results = group.fit_cohort_cca(**cca_kwargs)

    print(f"\n  Fitted {len(results)} cohorts:")
    for tnm, r in sorted(results.items()):
        alpha_str = (f', a={r.alpha:.4f} (l1={r.l1_ratio})'
                     if r.alpha is not None else '')
        p = r.p_values[0] if r.p_values is not None else np.nan
        sig = '*' if p < 0.05 else ''
        print(f"    {tnm}: r = {r.correlations[0]:.4f}, "
              f"p = {p:.4f}, n = {r.n_recordings}{alpha_str} {sig}")

    # Cross-projection and weight comparison
    print("\n  Cross-projecting...")
    cp = group.cross_project_cca()
    print(cp.to_string(index=False))

    print("\n  Weight cosine similarities...")
    ws = group.compare_cca_weights()
    print(ws.to_string(index=False))

    # Save CSVs
    save_cca_results(results, data_dir, label)
    cp.to_csv(data_dir / f'{label}_cross_projections.csv', index=False)
    ws.to_csv(data_dir / f'{label}_weight_similarities.csv', index=False)

    # Scatter plots
    print("\n  Generating per-cohort scatter plots...")
    for tnm, r in sorted(results.items()):
        fig = plot_cohort_scatter(r, tnm, label)
        fname = f'{label}_{tnm.replace("-", "_")}.svg'
        fig.savefig(scatter_dir / fname, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close(fig)

    # Summary figure
    print("  Generating summary figure...")
    fig = plot_cohort_cca_summary(results, cp, ws)
    fig.suptitle(f'CCA summary — {label}', fontsize=14, y=1.02)
    fig.savefig(summary_dir / f'{label}_summary.svg',
                dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)


def plot_cca_from_saved(data_dir, events, scatter_dir, summary_dir):
    """Reload saved CCA results and regenerate figures."""
    for event in events:
        label = _event_label(event)
        print(f"\n  [{label}] Loading saved CCA results...")

        results, cp, ws = load_cca_results(data_dir, label)
        print(f"  Loaded {len(results)} cohorts")

        for tnm, r in sorted(results.items()):
            fig = plot_cohort_scatter(r, tnm, label)
            fname = f'{label}_{tnm.replace("-", "_")}.svg'
            fig.savefig(scatter_dir / fname,
                        dpi=FIGURE_DPI, bbox_inches='tight')
            plt.close(fig)

        fig = plot_cohort_cca_summary(results, cp, ws)
        fig.suptitle(f'CCA summary — {label}', fontsize=14, y=1.02)
        fig.savefig(summary_dir / f'{label}_summary.svg',
                    dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close(fig)


# =========================================================================
# Main
# =========================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Shared
    parser.add_argument('--plot-only', action='store_true',
                        help='skip fitting; regenerate figures from saved data')
    parser.add_argument('--contrast-coding', choices=['log', 'linear', 'rank'],
                        default='rank',
                        help='contrast transform for GLM (default: rank)')
    parser.add_argument('--events', nargs='+', default=None,
                        help='events to analyze '
                        f'(default: {RESPONSE_EVENTS})')
    # PCA/ICA
    parser.add_argument('--cohort-weighted', action='store_true',
                        help='weight PCA by 1/n_k so each target contributes equally')
    parser.add_argument('--ica', action='store_true',
                        help='use ICA instead of PCA')
    # CCA
    parser.add_argument('--n-permutations', type=int, default=1000,
                        help='CCA permutation test iterations (default: 1000)')
    parser.add_argument('--params', nargs='+', default=None,
                        help='psychometric parameters for CCA behavioral view')
    parser.add_argument('--seed', type=int, default=42,
                        help='RNG seed for CCA permutations')
    parser.add_argument('--weight-by-se', action='store_true',
                        help='use t-statistics instead of raw GLM coefficients for CCA')
    parser.add_argument('--sparse', action='store_true',
                        help='use sparse CCA (ElasticCCA)')
    parser.add_argument('--unit-norm', action='store_true', default=True,
                        help='rescale sparse CCA weights to unit L2 norm (default: True)')
    parser.add_argument('--no-unit-norm', action='store_false', dest='unit_norm',
                        help='keep raw ElasticCCA weight magnitudes')
    args = parser.parse_args()

    events = args.events if args.events else RESPONSE_EVENTS

    # Create output directories
    data_dir = TASK_ENCODING_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    fig_dirs = {
        'pca': PROJECT_ROOT / 'figures/task_encoding/pca',
        'cca_scatter': PROJECT_ROOT / 'figures/task_encoding/cca/scatter',
        'cca_summary': PROJECT_ROOT / 'figures/task_encoding/cca/summary',
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
    print(f"  {len(group)} recordings after filtering")

    # Load pre-computed response magnitudes and trial timing
    if not RESPONSES_FPATH.exists():
        print(f"Error: {RESPONSES_FPATH} not found. "
              "Run scripts/responses.py first.")
        raise SystemExit(1)
    group.load_response_magnitudes(RESPONSES_FPATH)
    group.load_trial_timing(TRIAL_TIMING_FPATH)

    # =====================================================================
    # Plot-only mode
    # =====================================================================
    if args.plot_only:
        # PCA/ICA: re-compute (fast, no permutation test)
        decomp = 'ICA' if args.ica else 'PCA'
        w_label = 'cohort-weighted' if args.cohort_weighted else 'unweighted'
        print(f"\nRe-computing {decomp} ({w_label})...")
        for event in events:
            run_decomposition(group, event, args, data_dir, fig_dirs['pca'])

        # CCA: reload from saved CSVs
        print("\nReloading saved CCA results...")
        plot_cca_from_saved(
            data_dir, events,
            fig_dirs['cca_scatter'], fig_dirs['cca_summary'])

        print(f"\nFigures saved to {PROJECT_ROOT / 'figures/task_encoding'}")
        raise SystemExit(0)

    # =====================================================================
    # Full pipeline: per-event GLM + PCA/ICA + CCA
    # =====================================================================
    decomp = 'ICA' if args.ica else 'PCA'
    w_label = 'cohort-weighted' if args.cohort_weighted else 'unweighted'
    print(f"\n{decomp} ({w_label}) + CCA pipeline")

    for event in events:
        label = _event_label(event)
        print(f"\n{'=' * 60}")
        print(f"Event: {label}")
        print('=' * 60)

        # --- PCA/ICA ---
        # This internally fits the GLM via get_glm_response_features
        run_decomposition(group, event, args, data_dir, fig_dirs['pca'])

        # --- CCA ---
        # Reuses group.glm_response_features from PCA step (unless
        # weight_by_se is set, in which case run_cca re-fits with t-stats)
        run_cca(group, event, args, data_dir,
                fig_dirs['cca_scatter'], fig_dirs['cca_summary'])

    print(f"\nData saved to {data_dir}")
    print(f"Figures saved to {PROJECT_ROOT / 'figures/task_encoding'}")
