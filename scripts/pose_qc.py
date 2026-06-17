"""Pose movement diagnostic grids.

Reads the self-contained ``metadata/pose.pqt`` and writes two SVGs, both with one
row per movement measure:
- ``pose_qc_grid.svg``: violins of each measure split by each video-QC field.
- ``pose_corr_grid.svg``: Spearman scatters of each measure against
  ``fraction_correct``, log reaction time, the max paw-wheel xcorr ``abs_lag``,
  the max xcorr ``peak_value``, and absolute ``drift``.

Input:  metadata/pose.pqt
Output: figures/pose_qc/pose_qc_grid.svg, figures/pose_qc/pose_corr_grid.svg
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
    QCCMAP,
    QCVAL2NUM,
    QC_VALUE_ORDER,
    VIDEO_QC_COLS,
)
from iblnm.vis import violinplot

PEAK_LAG_COLS = ['peak_lag_early', 'peak_lag_mid', 'peak_lag_late']
PEAK_VAL_COLS = ['peak_val_early', 'peak_val_mid', 'peak_val_late']
CORR_COLS = ['fraction_correct', 'log_rt', 'abs_lag', 'peak_value', 'abs_drift']


def max_abs_lag(df: pd.DataFrame) -> pd.Series:
    """Largest paw-wheel peak-lag magnitude across the three session thirds.

    Parameters
    ----------
    df : pd.DataFrame
        Pose table carrying the signed per-third lag columns ``PEAK_LAG_COLS``
        (seconds).

    Returns
    -------
    pd.Series
        ``max(|early|, |mid|, |late|)`` per row, in seconds.
    """
    return df[PEAK_LAG_COLS].abs().max(axis=1)


def max_peak_value(df: pd.DataFrame) -> pd.Series:
    """Largest paw-wheel cross-correlation peak value across the three thirds.

    Parameters
    ----------
    df : pd.DataFrame
        Pose table carrying the per-third peak values ``PEAK_VAL_COLS``.

    Returns
    -------
    pd.Series
        ``max(early, mid, late)`` per row (strongest paw-wheel alignment).
    """
    return df[PEAK_VAL_COLS].max(axis=1)


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
    """Draw violins of ``measure`` split by ``qc_col``, one per QC category.

    All ``QC_VALUE_ORDER`` categories get a tick even when empty, so every cell
    shares the same x-axis. Each violin is colored by ``QCCMAP(QCVAL2NUM[cat])``.
    """
    groups = [df.loc[df[qc_col] == cat, measure].dropna().values
              for cat in QC_VALUE_ORDER]
    cell_colors = [QCCMAP(QCVAL2NUM[cat]) for cat in QC_VALUE_ORDER]
    positions = range(len(QC_VALUE_ORDER))
    violinplot(ax, groups, positions=positions, colors=cell_colors)
    for pos, group in zip(positions, groups):
        ax.annotate(f'n={len(group)}', (pos, 1), xycoords=('data', 'axes fraction'),
                    ha='center', va='bottom', fontsize=6)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(QC_VALUE_ORDER, rotation=45, ha='right', fontsize=6)


def _scatter_cell(ax, df: pd.DataFrame, measure: str, metric: str) -> None:
    """Scatter ``measure`` (y) vs ``metric`` (x) and annotate Spearman r, p."""
    ax.scatter(df[metric], df[measure], s=5, fc='none', ec='black')
    r, p = spearman_finite(df[metric], df[measure])
    ax.annotate(f'r={r:.2f}\np={p:.1e}', (0.05, 0.95), xycoords='axes fraction',
                ha='left', va='top', fontsize=6)
    ax.set_xlabel(metric, fontsize=6)


def build_violin_grid(df: pd.DataFrame):
    """Build the movement-vs-video-QC violin grid (rows x ``VIDEO_QC_COLS``).

    Rows are the ``POSE_MEASURES`` keys; each column is one ``VIDEO_QC_COLS``
    field, with the measure split into violins by that field's QC categories.

    Returns
    -------
    matplotlib.figure.Figure
    """
    measures = list(POSE_MEASURES)
    fig, axes = plt.subplots(len(measures), len(VIDEO_QC_COLS), sharey='row',
                             figsize=(2 * len(VIDEO_QC_COLS), 2 * len(measures)))
    for i, measure in enumerate(measures):
        axes[i, 0].set_ylabel(measure, fontsize=8)
        for j, qc_col in enumerate(VIDEO_QC_COLS):
            _violin_cell(axes[i, j], df, measure, qc_col)
    fig.tight_layout()
    return fig


def build_corr_grid(df: pd.DataFrame):
    """Build the movement-vs-metric Spearman scatter grid (rows x ``CORR_COLS``).

    Rows are the ``POSE_MEASURES`` keys; columns are ``fraction_correct``, log
    reaction time, the max paw-wheel ``abs_lag``, the max xcorr ``peak_value``,
    and the absolute ``drift`` — all derived here from the pose table.

    Returns
    -------
    matplotlib.figure.Figure
    """
    df = df.assign(
        log_rt=np.log(df['mean_rt']),
        abs_lag=max_abs_lag(df),
        peak_value=max_peak_value(df),
        abs_drift=df['drift'].abs(),
    )
    measures = list(POSE_MEASURES)
    fig, axes = plt.subplots(len(measures), len(CORR_COLS), sharey='row',
                             figsize=(2 * len(CORR_COLS), 2 * len(measures)))
    for i, measure in enumerate(measures):
        axes[i, 0].set_ylabel(measure, fontsize=8)
        for j, metric in enumerate(CORR_COLS):
            _scatter_cell(axes[i, j], df, measure, metric)
    fig.tight_layout()
    return fig


def main() -> None:
    df = pd.read_parquet(POSE_FPATH)
    figures_dir = PROJECT_ROOT / 'figures/pose_qc'
    figures_dir.mkdir(parents=True, exist_ok=True)
    for name, builder in [('pose_qc_grid', build_violin_grid),
                          ('pose_corr_grid', build_corr_grid)]:
        fpath = figures_dir / f'{name}.svg'
        builder(df).savefig(fpath, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Saved {fpath}")


if __name__ == '__main__':
    main()
