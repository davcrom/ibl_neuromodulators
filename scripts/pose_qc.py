"""Pose movement vs. video-QC diagnostic grid.

Reads the self-contained ``metadata/pose.pqt`` and writes a 4x10 grid SVG: per
movement measure (rows), violins split by each video-QC field (columns 1-8) and
Spearman scatters against ``fraction_correct`` and ``mean_rt`` (columns 9-10).

Input:  metadata/pose.pqt
Output: figures/pose_qc/pose_qc_grid.svg
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import spearmanr

from iblnm.config import (
    FIGURE_DPI,
    POSE_FPATH,
    POSE_MEASURES,
    PROJECT_ROOT,
    QC_VALUE_ORDER,
    VIDEO_QC_COLS,
)
from iblnm.vis import violinplot

PERFORMANCE_COLS = ['fraction_correct', 'mean_rt']


def order_qc_categories(values) -> list:
    """Order the QC categories present in ``values`` by severity.

    Parameters
    ----------
    values : array-like
        QC outcome values for one column (strings, possibly with NaN/missing).

    Returns
    -------
    list
        Categories present in ``values``, ordered by ``QC_VALUE_ORDER``. Values
        not in that list (unknown strings) come next in order of appearance, and
        a single ``np.nan`` group is appended last if any value is missing.
    """
    present = pd.unique(pd.Series(values))
    non_null = [v for v in present if not pd.isna(v)]
    known = [c for c in QC_VALUE_ORDER if c in non_null]
    unknown = [v for v in non_null if v not in QC_VALUE_ORDER]
    ordered = known + unknown
    if any(pd.isna(v) for v in present):
        ordered.append(np.nan)
    return ordered


def spearman_finite(x, y):
    """Spearman correlation over rows where both ``x`` and ``y`` are finite.

    Parameters
    ----------
    x, y : array-like
        Paired samples; rows with NaN/inf in either are dropped before ranking.

    Returns
    -------
    scipy.stats._stats_py.SignificanceResult
        The result of ``scipy.stats.spearmanr`` on the finite-pair subset,
        unpackable as ``(correlation, pvalue)``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    return spearmanr(x[mask], y[mask])


def _violin_cell(ax, df: pd.DataFrame, measure: str, qc_col: str) -> None:
    """Draw violins of ``measure`` grouped by the QC categories of ``qc_col``."""
    cats = order_qc_categories(df[qc_col])
    masks = [df[qc_col].isna() if pd.isna(c) else df[qc_col] == c for c in cats]
    groups = [df.loc[m, measure].dropna().values for m in masks]
    positions = range(len(cats))
    violinplot(ax, groups, positions=positions, colors=['black'] * len(cats))
    for pos, group in zip(positions, groups):
        ax.annotate(f'n={len(group)}', (pos, 1), xycoords=('data', 'axes fraction'),
                    ha='center', va='bottom', fontsize=6)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(['NaN' if pd.isna(c) else c for c in cats],
                       rotation=45, ha='right', fontsize=6)


def _scatter_cell(ax, df: pd.DataFrame, measure: str, metric: str) -> None:
    """Scatter ``measure`` (y) vs ``metric`` (x) and annotate Spearman r, p."""
    ax.scatter(df[metric], df[measure], s=5, fc='none', ec='black')
    r, p = spearman_finite(df[metric], df[measure])
    ax.annotate(f'r={r:.2f}\np={p:.1e}', (0.05, 0.95), xycoords='axes fraction',
                ha='left', va='top', fontsize=6)
    ax.set_xlabel(metric, fontsize=6)


def build_grid(df: pd.DataFrame):
    """Build the 4x10 movement-vs-QC diagnostic grid.

    Rows are the ``POSE_MEASURES`` keys; columns 1-8 are ``VIDEO_QC_COLS``
    violins and columns 9-10 are ``fraction_correct`` / ``mean_rt`` scatters.

    Returns
    -------
    matplotlib.figure.Figure
    """
    measures = list(POSE_MEASURES)
    columns = VIDEO_QC_COLS + PERFORMANCE_COLS
    fig, axes = plt.subplots(len(measures), len(columns), sharey='row',
                             figsize=(2 * len(columns), 2 * len(measures)))
    for i, measure in enumerate(measures):
        axes[i, 0].set_ylabel(measure, fontsize=8)
        for j, qc_col in enumerate(VIDEO_QC_COLS):
            _violin_cell(axes[i, j], df, measure, qc_col)
        for k, metric in enumerate(PERFORMANCE_COLS):
            _scatter_cell(axes[i, len(VIDEO_QC_COLS) + k], df, measure, metric)
    fig.tight_layout()
    return fig


def main() -> None:
    df = pd.read_parquet(POSE_FPATH)
    fig = build_grid(df)
    figures_dir = PROJECT_ROOT / 'figures/pose_qc'
    figures_dir.mkdir(parents=True, exist_ok=True)
    fpath = figures_dir / 'pose_qc_grid.svg'
    fig.savefig(fpath, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"Saved {fpath}")


if __name__ == '__main__':
    main()
