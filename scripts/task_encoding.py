"""
Task Encoding Analysis

Per-cohort CCA between per-session GLM-derived neural features and psychometric
parameters.

Requires responses.py to have run first (loads pre-computed response
magnitudes and trial timing from disk).

Output:
    results/task_encoding/             — CSV data files
    figures/task_encoding/cca/scatter/ — per-cohort CCA score scatter plots
    figures/task_encoding/cca/summary/ — CCA summary figures

Usage:
    python scripts/task_encoding.py
    python scripts/task_encoding.py --n-permutations 5000
    python scripts/task_encoding.py --events stimOn_times feedback_times
    python scripts/task_encoding.py --plot-only
"""
import argparse

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from iblnm.config import (
    PROJECT_ROOT, SESSIONS_FPATH, SESSIONS_H5_DIR, PERFORMANCE_FPATH,
    RESPONSES_FPATH, TRIAL_REGRESSORS_FPATH, TASK_ENCODING_DIR,
    RESPONSE_EVENTS, FIGURE_DPI, TARGETNM_COLORS,
    ANALYSIS_QC_BLOCKERS, SESSION_TYPES_TO_ANALYZE, TARGETNMS_TO_ANALYZE,
    LMM_FORMULAS, CCA_TASK_MAINS, CCA_MOVEMENT_MAINS,
)
from iblnm.analysis import compute_feature_dispersion, select_block_terms
from iblnm.data import PhotometrySessionGroup
from iblnm.io import _get_default_connection
from iblnm.vis import plot_cohort_cca_summary

plt.ion()


DEFAULT_PARAMS = [
    'psych_50_threshold', 'psych_50_bias',
    'psych_50_lapse_left', 'psych_50_lapse_right',
]

# Neural-feature blocks for the two-block CCA: each block's main effects select
# its coefficient columns (mains + within-block interactions) via
# select_block_terms. Cross-category interactions and the intercept stay in the
# fitted features but are excluded from every block.
CCA_BLOCK_MAINS = {'task': CCA_TASK_MAINS, 'movement': CCA_MOVEMENT_MAINS}

# Grid search defaults for sparse CCA
ALPHA_GRID = [1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
L1_RATIO_GRID = [0.0]


def _event_label(event_name):
    """Strip '_times' suffix for display and filenames."""
    return event_name.replace('_times', '')


def _block_label(event_name, block):
    """Combine event and CCA block into one output label.

    Parameters
    ----------
    event_name : str
        Event name with the ``_times`` suffix (e.g. ``'stimOn_times'``).
    block : str
        Block tag, ``'task'`` or ``'movement'``.

    Returns
    -------
    str
        ``'{event_stem}_{block}'`` (e.g. ``'stimOn_task'``), used to tag the
        per-event, per-block CCA CSVs and figures.
    """
    return f'{_event_label(event_name)}_{block}'


# =========================================================================
# Dispersion scatter
# =========================================================================

def assemble_dispersion_frame(neural_by_event, behavioral_long, block_mains,
                              min_sessions):
    """Build the plot-ready coefficient-dispersion-vs-behavior frame.

    Behavioral dispersion is computed once per subject (sensor-independent) and
    joined to each ``(subject, target_NM)`` unit, so it is identical across the
    event columns. Neural dispersion is computed per event and block over that
    block's coefficient terms, z-scored within ``target_NM``. A unit appears in a
    panel only when both its neural ``n_sessions`` (for that event/block) and its
    behavioral ``n_sessions`` are at least ``min_sessions``.

    Parameters
    ----------
    neural_by_event : dict[str, pandas.DataFrame]
        Maps event name to a long per-session coefficient frame with columns
        ``['subject', 'target_NM', 'eid', 'term', 'coef']``.
    behavioral_long : pandas.DataFrame
        Long per-session behavioral params with columns
        ``['subject', 'eid', 'param', 'value']``.
    block_mains : dict[str, list[str]]
        Maps block label (``'task'``/``'movement'``) to its main-effect factor
        names; :func:`iblnm.analysis.select_block_terms` picks each block's terms.
    min_sessions : int
        Minimum scorable sessions in both the neural and behavioral sets.

    Returns
    -------
    pandas.DataFrame
        One row per surviving ``(subject, target_NM, event, block)``, columns
        ``['subject', 'target_NM', 'event', 'block', 'neural_dispersion',
        'behavioral_dispersion']``.
    """
    behavioral = compute_feature_dispersion(
        behavioral_long, unit_cols=['subject'], session_col='eid',
        feature_col='param', value_col='value', standardize_by=None)
    behavioral = behavioral[behavioral['n_sessions'] >= min_sessions]
    behavioral = behavioral.rename(
        columns={'dispersion': 'behavioral_dispersion'})[
            ['subject', 'behavioral_dispersion']]

    frames = []
    for event, neural_long in neural_by_event.items():
        for block, mains in block_mains.items():
            block_terms = select_block_terms(neural_long['term'].unique(), mains)
            block_long = neural_long[neural_long['term'].isin(block_terms)]
            neural = compute_feature_dispersion(
                block_long, unit_cols=['subject', 'target_NM'],
                session_col='eid', feature_col='term', value_col='coef',
                standardize_by='target_NM')
            neural = neural[neural['n_sessions'] >= min_sessions]
            neural = neural.rename(columns={'dispersion': 'neural_dispersion'})
            merged = neural.merge(behavioral, on='subject')
            merged['event'] = event
            merged['block'] = block
            frames.append(merged[[
                'subject', 'target_NM', 'event', 'block',
                'neural_dispersion', 'behavioral_dispersion']])

    return pd.concat(frames, ignore_index=True)


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
    xlabel = 'Neural canonical variate'
    ylabel = 'Behavioral canonical variate'
    if cca_result.x_variance_explained is not None:
        xlabel += f' (R² = {cca_result.x_variance_explained[0]:.2f})'
    if cca_result.y_variance_explained is not None:
        ylabel += f' (R² = {cca_result.y_variance_explained[0]:.2f})'
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f'{cohort_name} ({event_label}): r = {r:.3f}, p = {p:.4f}')
    fig.tight_layout()
    return fig


def save_cca_results(results, data_dir, label):
    """Save per-cohort CCAResult data to CSVs.

    Files written:
        {label}_summary.csv  — cohort, correlation, p_value, n, alpha, l1_ratio,
                               x/y_variance_explained (CC1 variance extracted)
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
            'x_variance_explained': (r.x_variance_explained[0]
                                     if r.x_variance_explained is not None
                                     else np.nan),
            'y_variance_explained': (r.y_variance_explained[0]
                                     if r.y_variance_explained is not None
                                     else np.nan),
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
        xve = (np.array([row['x_variance_explained']])
               if 'x_variance_explained' in row
               and not np.isnan(row['x_variance_explained']) else None)
        yve = (np.array([row['y_variance_explained']])
               if 'y_variance_explained' in row
               and not np.isnan(row['y_variance_explained']) else None)

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
            x_variance_explained=xve,
            y_variance_explained=yve,
        )

    return results, cp, ws


def _plot_cca_figures(results, cp, ws, label, scatter_dir, summary_dir):
    """Write per-cohort scatter SVGs and the cohort summary SVG for one label.

    Parameters
    ----------
    results : dict[str, CCAResult]
        Per-cohort CCA fits.
    cp : pd.DataFrame
        Cross-projection correlations.
    ws : pd.DataFrame
        Weight cosine similarities.
    label : str
        Output label (``'{event}_{block}'``) used in figure filenames/titles.
    scatter_dir, summary_dir : Path
        Output directories for scatter and summary figures.
    """
    for tnm, r in sorted(results.items()):
        fig = plot_cohort_scatter(r, tnm, label)
        fname = f'{label}_{tnm.replace("-", "_")}.svg'
        fig.savefig(scatter_dir / fname, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close(fig)

    fig = plot_cohort_cca_summary(results, cp, ws)
    fig.suptitle(f'CCA summary — {label}', fontsize=14, y=1.02)
    fig.savefig(summary_dir / f'{label}_summary.svg',
                dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)


def _run_cca_block(group, event, block, feature_cols, cca_kwargs,
                   data_dir, scatter_dir, summary_dir):
    """Fit, cross-project, compare, and save per-cohort CCA for one block.

    The neural view is subset to ``feature_cols`` (one category block); the
    behavioral view and ``cca_kwargs`` are shared across blocks. ``cross_project``
    and ``compare_weights`` read the cohort results populated by the immediately
    preceding ``fit_cohort_cca``, so they belong inside this per-block call. All
    outputs are tagged ``_block_label(event, block)``.

    Parameters
    ----------
    group : PhotometrySessionGroup
    event : str
        Event name (e.g. ``'stimOn_times'``).
    block : str
        Block tag (``'task'`` or ``'movement'``).
    feature_cols : list[str]
        Neural-feature columns for this block.
    cca_kwargs : dict
        Keyword arguments forwarded to ``fit_cohort_cca`` (shared across blocks).
    data_dir, scatter_dir, summary_dir : Path
        Output directories for CSVs, scatter figures, and summary figures.
    """
    label = _block_label(event, block)
    print(f"\n  [{label}] Fitting per-cohort CCA on {len(feature_cols)} "
          "neural features...")
    results = group.fit_cohort_cca(feature_cols=feature_cols, **cca_kwargs)

    print(f"  Fitted {len(results)} cohorts:")
    for tnm, r in sorted(results.items()):
        alpha_str = (f', a={r.alpha:.4f} (l1={r.l1_ratio})'
                     if r.alpha is not None else '')
        p = r.p_values[0] if r.p_values is not None else np.nan
        sig = '*' if p < 0.05 else ''
        print(f"    {tnm}: r = {r.correlations[0]:.4f}, "
              f"p = {p:.4f}, n = {r.n_recordings}{alpha_str} {sig}")

    cp = group.cross_project_cca()
    ws = group.compare_cca_weights()

    save_cca_results(results, data_dir, label)
    cp.to_csv(data_dir / f'{label}_cross_projections.csv', index=False)
    ws.to_csv(data_dir / f'{label}_weight_similarities.csv', index=False)

    _plot_cca_figures(results, cp, ws, label, scatter_dir, summary_dir)


def run_cca(group, event, args, data_dir, scatter_dir, summary_dir):
    """Fit the two-block per-cohort CCA for one event and save results.

    Fits the per-session OLS neural features and the shared psychometric
    (behavioral) features once, then runs CCA twice — once per neural-feature
    block in ``CCA_BLOCK_MAINS`` (task, movement) — writing block-labelled CSVs
    and figures.

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
    params = args.params if args.params else DEFAULT_PARAMS

    # Fit per-session OLS for this event (CCA neural view), shared across blocks
    print("  Fitting per-session OLS for CCA neural features...")
    group.get_persession_ols_features(
        formula=LMM_FORMULAS['persession']['full'],
        event_name=event, weight_by_se=args.weight_by_se,
        contrast_coding=args.contrast_coding,
    )

    # Align psychometric (behavioral) features to the neural features
    group.response_features = group.persession_ols_features
    group.get_psychometric_features(params=params)
    n_valid = group.psychometric_features.notna().all(axis=1).sum()
    print(f"  {n_valid}/{len(group.psychometric_features)} recordings "
          "with complete psychometric data")

    cca_kwargs = dict(
        n_permutations=args.n_permutations,
        seed=args.seed,
        sparse=args.sparse,
    )
    if args.sparse:
        cca_kwargs['alpha'] = ALPHA_GRID
        cca_kwargs['l1_ratio'] = L1_RATIO_GRID
        cca_kwargs['unit_norm'] = args.unit_norm

    columns = group.persession_ols_features.columns
    for block, mains in CCA_BLOCK_MAINS.items():
        feature_cols = select_block_terms(columns, mains)
        _run_cca_block(group, event, block, feature_cols, cca_kwargs,
                       data_dir, scatter_dir, summary_dir)


def plot_cca_from_saved(data_dir, events, scatter_dir, summary_dir):
    """Reload saved per-event, per-block CCA results and regenerate figures."""
    for event in events:
        for block in CCA_BLOCK_MAINS:
            label = _block_label(event, block)
            print(f"\n  [{label}] Loading saved CCA results...")
            results, cp, ws = load_cca_results(data_dir, label)
            print(f"  Loaded {len(results)} cohorts")
            _plot_cca_figures(results, cp, ws, label, scatter_dir, summary_dir)


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
    parser.add_argument('--contrast-coding',
                        choices=['log', 'log2', 'linear', 'rank'],
                        default='log2',
                        help='contrast transform for GLM (default: log2)')
    parser.add_argument('--events', nargs='+', default=None,
                        help='events to analyze '
                        f'(default: {RESPONSE_EVENTS})')
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
    if PERFORMANCE_FPATH.exists():
        perf = pd.read_parquet(
            PERFORMANCE_FPATH, columns=['eid', 'fraction_correct', 'contrasts'])
        df = df.merge(perf, on='eid', how='left')

    one = _get_default_connection()
    group = PhotometrySessionGroup.from_catalog(df, one=one, h5_dir=SESSIONS_H5_DIR)
    group.filter_sessions(
        session_types=SESSION_TYPES_TO_ANALYZE,
        qc_blockers=ANALYSIS_QC_BLOCKERS,
        targetnms=TARGETNMS_TO_ANALYZE,
    )
    print(f"  {len(group)} recordings after filtering")

    # Load pre-computed response magnitudes and trial regressors
    if not RESPONSES_FPATH.exists():
        print(f"Error: {RESPONSES_FPATH} not found. "
              "Run scripts/responses.py first.")
        raise SystemExit(1)
    group.load_response_magnitudes(RESPONSES_FPATH)
    group.load_trial_regressors(TRIAL_REGRESSORS_FPATH)

    # =====================================================================
    # Plot-only mode
    # =====================================================================
    if args.plot_only:
        # CCA: reload from saved CSVs
        print("\nReloading saved CCA results...")
        plot_cca_from_saved(
            data_dir, events,
            fig_dirs['cca_scatter'], fig_dirs['cca_summary'])

        print(f"\nFigures saved to {PROJECT_ROOT / 'figures/task_encoding'}")
        raise SystemExit(0)

    # =====================================================================
    # Full pipeline: per-event GLM + CCA
    # =====================================================================
    print("\nCCA pipeline")

    for event in events:
        label = _event_label(event)
        print(f"\n{'=' * 60}")
        print(f"Event: {label}")
        print('=' * 60)

        # --- CCA ---
        # run_cca fits the per-session GLMs (neural view) at entry.
        run_cca(group, event, args, data_dir,
                fig_dirs['cca_scatter'], fig_dirs['cca_summary'])

    print(f"\nData saved to {data_dir}")
    print(f"Figures saved to {PROJECT_ROOT / 'figures/task_encoding'}")
