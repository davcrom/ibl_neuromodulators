"""
Per-Cohort Canonical Correlation Analysis

Fits CCA (k=1) separately per target-NM cohort, then compares coupling
structures via cross-projection and weight cosine similarity.

Output:
    figures/cca/cohort_cca_summary.svg       - 3-panel summary
    figures/cca/cohort_scatter_{target}.svg   - neural vs behavioral score
    figures/cca/cross_projections.csv         - cross-projection correlations
    figures/cca/weight_similarities.csv       - cosine similarities
    figures/cca/cohort_cca_summary.csv        - per-cohort correlations & p-values

Usage:
    python scripts/cca.py
    python scripts/cca.py --n-permutations 5000
    python scripts/cca.py --features glm --event stimOn_times --weight-by-se
"""
import argparse

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from iblnm.config import (
    PROJECT_ROOT, SESSIONS_FPATH, EVENTS_FPATH,
    TARGETNM_COLORS, FIGURE_DPI,
)
from iblnm.data import PhotometrySessionGroup
from iblnm.io import _get_default_connection
from iblnm.util import derive_target_nm
from iblnm.vis import plot_cohort_cca_summary, plot_cca_weight_profiles

plt.ion()


DEFAULT_PARAMS = [
    'psych_50_threshold', 'psych_50_bias',
    'psych_50_lapse_left', 'psych_50_lapse_right',
]


# =========================================================================
# Plotting helpers (specific to this analysis)
# =========================================================================

def plot_cohort_scatter(cca_result, cohort_name):
    """Scatter x_scores vs y_scores for one cohort's CC1."""
    fig, ax = plt.subplots(figsize=(5, 5))
    color = TARGETNM_COLORS.get(cohort_name, 'gray')
    ax.scatter(cca_result.x_scores[:, 0], cca_result.y_scores[:, 0],
               c=color, alpha=0.7, edgecolors='white', s=40)
    r = cca_result.correlations[0]
    p = cca_result.p_values[0] if cca_result.p_values is not None else np.nan
    ax.set_xlabel('Neural canonical variate')
    ax.set_ylabel('Behavioral canonical variate')
    ax.set_title(f'{cohort_name}: r = {r:.3f}, p = {p:.4f}')
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
    parser.add_argument('--features', choices=['glm'],
                        default='glm',
                        help='feature type (default: glm)')
    parser.add_argument('--event', default='stimOn_times',
                        help='event for GLM features (default: stimOn_times)')
    parser.add_argument('--weight-by-se', action='store_true',
                        help='use t-statistics instead of raw GLM coefficients')
    args = parser.parse_args()

    params = args.params if args.params else DEFAULT_PARAMS
    figures_dir = PROJECT_ROOT / 'figures/cca'
    figures_dir.mkdir(parents=True, exist_ok=True)

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
    group.filter_recordings()
    print(f"  {len(group)} recordings after filtering")

    # =====================================================================
    # Load / compute GLM features
    # =====================================================================
    if not EVENTS_FPATH.exists():
        print(f"Error: {EVENTS_FPATH} not found. "
              "Run scripts/responses.py first.")
        raise SystemExit(1)
    print(f"Loading events from {EVENTS_FPATH}")
    group.response_magnitudes = pd.read_parquet(EVENTS_FPATH)
    print(f"Computing GLM features (event={args.event}, "
          f"weight_by_se={args.weight_by_se})...")
    group.get_glm_response_features(
        event_name=args.event, weight_by_se=args.weight_by_se,
    )
    print(f"  {len(group.glm_response_features)} recordings, "
          f"{group.glm_response_features.shape[1]} features")

    # =====================================================================
    # Load psychometric features
    # =====================================================================
    # Psychometric features need response_features for index alignment;
    # temporarily set it so get_psychometric_features works.
    group.response_features = group.glm_response_features
    print(f"Loading psychometric features (params: {params})")
    group.get_psychometric_features(params=params)
    n_valid = group.psychometric_features.notna().all(axis=1).sum()
    print(f"  {n_valid}/{len(group.psychometric_features)} recordings "
          "with complete psychometric data")

    # =====================================================================
    # Fit per-cohort CCA
    # =====================================================================
    print(f"\nFitting per-cohort CCA ({args.n_permutations} permutations)...")
    results = group.fit_cohort_cca(
        n_permutations=args.n_permutations,
        seed=args.seed,
    )

    print(f"\n  Fitted {len(results)} cohorts:")
    for tnm, r in sorted(results.items()):
        p = r.p_values[0] if r.p_values is not None else np.nan
        sig = '*' if p < 0.05 else ''
        print(f"    {tnm}: r = {r.correlations[0]:.4f}, "
              f"p = {p:.4f}, n = {r.n_recordings} {sig}")

    # =====================================================================
    # Cross-projection and weight comparison
    # =====================================================================
    print("\nCross-projecting...")
    cp = group.cross_project_cca()
    print(cp.to_string(index=False))

    print("\nWeight cosine similarities...")
    ws = group.compare_cca_weights()
    print(ws.to_string(index=False))

    # =====================================================================
    # Per-cohort scatter plots
    # =====================================================================
    print("\nGenerating per-cohort scatter plots...")
    for tnm, r in sorted(results.items()):
        fig = plot_cohort_scatter(r, tnm)
        fname = f'cohort_scatter_{tnm.replace("-", "_")}.svg'
        fig.savefig(figures_dir / fname, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close(fig)

    # =====================================================================
    # Summary figure
    # =====================================================================
    print("Generating summary figure...")
    fig = plot_cohort_cca_summary(results, cp, ws)
    fig.savefig(figures_dir / 'cohort_cca_summary.svg',
                dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)

    # =====================================================================
    # Weight profiles (neural + behavioral)
    # =====================================================================
    print("Generating weight profile heatmaps...")
    fig = plot_cca_weight_profiles(results)
    fig.savefig(figures_dir / 'cohort_weight_profiles.svg',
                dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)

    # =====================================================================
    # Save CSVs
    # =====================================================================
    cp.to_csv(figures_dir / 'cross_projections.csv', index=False)
    ws.to_csv(figures_dir / 'weight_similarities.csv', index=False)

    summary_rows = []
    for tnm, r in sorted(results.items()):
        summary_rows.append({
            'cohort': tnm,
            'correlation': r.correlations[0],
            'p_value': r.p_values[0] if r.p_values is not None else np.nan,
            'n_recordings': r.n_recordings,
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(figures_dir / 'cohort_cca_summary.csv', index=False)

    print(f"\nSummary saved to {figures_dir / 'cohort_cca_summary.csv'}")
    print(f"Figures saved to {figures_dir}")
