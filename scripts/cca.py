"""
Per-Cohort Canonical Correlation Analysis

Fits CCA (k=1) separately per target-NM cohort for each event, then compares
coupling structures via cross-projection and weight cosine similarity.

Output:
    results/cca/                           — CSV data files (per event)
    figures/cca/scatter/                   — per-cohort score scatter plots
    figures/cca/summary/                   — summary figures (per event)

Usage:
    python scripts/cca.py
    python scripts/cca.py --n-permutations 5000
    python scripts/cca.py --events stimOn_times feedback_times firstMovement_times
    python scripts/cca.py --weight-by-se
    python scripts/cca.py --plot-only
"""
import argparse

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from iblnm.config import (
    PROJECT_ROOT, SESSIONS_FPATH, RESPONSES_FPATH,
    TARGETNM_COLORS, FIGURE_DPI,
    MIN_TRAINING_PERFORMANCE, REQUIRED_CONTRASTS,
)
from iblnm.data import PhotometrySessionGroup
from iblnm.io import _get_default_connection
from iblnm.util import derive_target_nm
from iblnm.vis import plot_cohort_cca_summary

plt.ion()


DEFAULT_PARAMS = [
    'psych_50_threshold', 'psych_50_bias',
    'psych_50_lapse_left', 'psych_50_lapse_right',
]

DEFAULT_EVENTS = ['stimOn_times', 'feedback_times']

# Grid search defaults for sparse CCA: favor low regularization / low sparsity
ALPHA_GRID = [1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
L1_RATIO_GRID = [0.0]


def _event_label(event_name):
    """Strip '_times' suffix for display and filenames."""
    return event_name.replace('_times', '')


# =========================================================================
# CSV round-trip for CCAResult
# =========================================================================

def save_cca_results(results, data_dir, label):
    """Save per-cohort CCAResult data to CSVs for later plotting.

    Files written:
        {label}_summary.csv       — cohort, correlation, p_value, n, alpha, l1_ratio
        {label}_weights.csv       — cohort, view (neural/behavioral), feature, CC1
        {label}_scores.csv        — cohort, x_score, y_score
        {label}_cross_projections.csv  — (saved separately by caller)
        {label}_weight_similarities.csv  — (saved separately by caller)
    """
    from iblnm.analysis import CCAResult

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
    ws : pd.DataFrame
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

        # Reconstruct weights
        w = weights[weights['cohort'] == tnm]
        xw = w[w['view'] == 'neural'].set_index('feature')['CC1']
        yw = w[w['view'] == 'behavioral'].set_index('feature')['CC1']
        x_weights = pd.DataFrame({'CC1': xw})
        y_weights = pd.DataFrame({'CC1': yw})

        # Reconstruct scores
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


# =========================================================================
# Plotting helpers (specific to this analysis)
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


# =========================================================================
# Main
# =========================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--n-permutations', type=int, default=1000,
                        help='number of permutations (default: 1000)')
    parser.add_argument('--params', nargs='+', default=None,
                        help='psychometric parameters to include')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--events', nargs='+', default=None,
                        help='events for GLM features '
                        f'(default: {DEFAULT_EVENTS})')
    parser.add_argument('--weight-by-se', action='store_true',
                        help='use t-statistics instead of raw GLM coefficients')
    parser.add_argument('--sparse', action='store_true',
                        help='use sparse CCA (cca-zoo ElasticCCA)')
    parser.add_argument('--unit-norm', action='store_true', default=True,
                        help='rescale sparse CCA weights to unit L2 norm '
                        '(default: True)')
    parser.add_argument('--no-unit-norm', action='store_false', dest='unit_norm',
                        help='keep raw ElasticCCA weight magnitudes')
    parser.add_argument('--plot-only', action='store_true',
                        help='skip fitting, load saved results and regenerate plots')
    args = parser.parse_args()

    params = args.params if args.params else DEFAULT_PARAMS
    events = args.events if args.events else DEFAULT_EVENTS

    # Create output directories
    data_dir = PROJECT_ROOT / 'results/cca'
    data_dir.mkdir(parents=True, exist_ok=True)

    fig_dirs = {
        'scatter': PROJECT_ROOT / 'figures/cca/scatter',
        'summary': PROJECT_ROOT / 'figures/cca/summary',
    }
    for d in fig_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # =====================================================================
    # Plot-only mode: load saved CSVs and regenerate figures
    # =====================================================================
    if args.plot_only:
        for event in events:
            label = _event_label(event)
            print(f"\n{'=' * 60}")
            print(f"Event: {label} (loading saved results)")
            print('=' * 60)

            results, cp, ws = load_cca_results(data_dir, label)
            print(f"  Loaded {len(results)} cohorts")

            # Scatter plots
            for tnm, r in sorted(results.items()):
                fig = plot_cohort_scatter(r, tnm, label)
                fname = f'{label}_{tnm.replace("-", "_")}.svg'
                fig.savefig(fig_dirs['scatter'] / fname,
                            dpi=FIGURE_DPI, bbox_inches='tight')
                plt.close(fig)

            # Summary figure
            fig = plot_cohort_cca_summary(results, cp, ws)
            fig.suptitle(f'CCA summary — {label}', fontsize=14, y=1.02)
            fig.savefig(fig_dirs['summary'] / f'{label}_summary.svg',
                        dpi=FIGURE_DPI, bbox_inches='tight')

        print(f"\nFigures saved to {PROJECT_ROOT / 'figures/cca'}")
        raise SystemExit(0)

    # =====================================================================
    # Load sessions and build group
    # =====================================================================
    print(f"Loading sessions from {SESSIONS_FPATH}")
    df_sessions = pd.read_parquet(SESSIONS_FPATH)
    df_sessions = derive_target_nm(df_sessions)

    # Ensure parallel list columns match before exploding
    _parallel_cols = ['target_NM', 'brain_region', 'hemisphere']
    _lengths_match = df_sessions[_parallel_cols].apply(
        lambda row: len(set(
            len(v) if isinstance(v, (list, np.ndarray)) else 1
            for v in row
        )) == 1,
        axis=1,
    )
    df_sessions = df_sessions[_lengths_match].copy()

    df_recordings = df_sessions.explode(_parallel_cols).copy()
    df_recordings['fiber_idx'] = df_recordings.groupby('eid').cumcount()

    one = _get_default_connection()
    group = PhotometrySessionGroup(df_recordings, one=one)
    group.filter_recordings(
        session_types=('biased', 'ephys', 'training'),
        min_performance=MIN_TRAINING_PERFORMANCE,
        required_contrasts=REQUIRED_CONTRASTS,
    )
    print(f"  {len(group)} recordings after filtering")

    # =====================================================================
    # Load response magnitudes (shared across events)
    # =====================================================================
    if not RESPONSES_FPATH.exists():
        print(f"Error: {RESPONSES_FPATH} not found. "
              "Run scripts/responses.py first.")
        raise SystemExit(1)
    print(f"Loading responses from {RESPONSES_FPATH}")
    group.response_magnitudes = pd.read_parquet(RESPONSES_FPATH)

    # =====================================================================
    # Per-event CCA loop
    # =====================================================================
    for event in events:
        label = _event_label(event)
        print(f"\n{'=' * 60}")
        print(f"Event: {label}")
        print('=' * 60)

        # Compute GLM features for this event
        print(f"Computing GLM features (weight_by_se={args.weight_by_se})...")
        group.get_glm_response_features(
            event_name=event, weight_by_se=args.weight_by_se,
        )
        print(f"  {len(group.glm_response_features)} recordings, "
              f"{group.glm_response_features.shape[1]} features")

        # Load psychometric features (index alignment needs response_features)
        group.response_features = group.glm_response_features
        group.get_psychometric_features(params=params)
        n_valid = group.psychometric_features.notna().all(axis=1).sum()
        print(f"  {n_valid}/{len(group.psychometric_features)} recordings "
              "with complete psychometric data")

        # Fit per-cohort CCA
        cca_type = 'sparse CCA' if args.sparse else 'CCA'
        print(f"Fitting per-cohort {cca_type} "
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
            alpha_str = (f', α={r.alpha:.4f} (l1={r.l1_ratio})'
                        if r.alpha is not None else '')
            p = r.p_values[0] if r.p_values is not None else np.nan
            sig = '*' if p < 0.05 else ''
            print(f"    {tnm}: r = {r.correlations[0]:.4f}, "
                  f"p = {p:.4f}, n = {r.n_recordings}{alpha_str} {sig}")

        # Cross-projection and weight comparison
        print("\nCross-projecting...")
        cp = group.cross_project_cca()
        print(cp.to_string(index=False))

        print("\nWeight cosine similarities...")
        ws = group.compare_cca_weights()
        print(ws.to_string(index=False))

        # -----------------------------------------------------------------
        # Save CSVs (results, cross-projections, weight similarities)
        # -----------------------------------------------------------------
        save_cca_results(results, data_dir, label)
        cp.to_csv(data_dir / f'{label}_cross_projections.csv', index=False)
        ws.to_csv(data_dir / f'{label}_weight_similarities.csv', index=False)

        # -----------------------------------------------------------------
        # Per-cohort scatter plots
        # -----------------------------------------------------------------
        print("\nGenerating per-cohort scatter plots...")
        for tnm, r in sorted(results.items()):
            fig = plot_cohort_scatter(r, tnm, label)
            fname = f'{label}_{tnm.replace("-", "_")}.svg'
            fig.savefig(fig_dirs['scatter'] / fname,
                        dpi=FIGURE_DPI, bbox_inches='tight')
            plt.close(fig)

        # -----------------------------------------------------------------
        # Summary figure (6 panels: bar, cross-proj, Δr, weights, 2× cosine)
        # -----------------------------------------------------------------
        print("Generating summary figure...")
        fig = plot_cohort_cca_summary(results, cp, ws)
        fig.suptitle(f'CCA summary — {label}', fontsize=14, y=1.02)
        fig.savefig(fig_dirs['summary'] / f'{label}_summary.svg',
                    dpi=FIGURE_DPI, bbox_inches='tight')

    print(f"\nData saved to {data_dir}")
    print(f"Figures saved to {PROJECT_ROOT / 'figures/cca'}")
