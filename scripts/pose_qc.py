"""Pose movement diagnostic grids.

Reads the self-contained ``metadata/pose.pqt`` and writes three SVGs:
- ``pose_qc_grid.svg``: violins of each movement measure (rows) split by each
  video-QC field (columns).
- ``pose_timing_grid.svg``: violins of the three derived timing metrics
  (``peak_value``, ``abs_lag``, ``abs_drift``; rows) split by each video-QC
  field (columns).
- ``pose_corr_grid.svg``: Spearman scatters of each movement measure against
  ``fraction_correct``, log reaction time, the max paw-wheel xcorr ``abs_lag``,
  the max xcorr ``peak_value``, and absolute ``drift``. Dots are colored by
  ``--color-by`` (session type, default, or acquisition date).

Input:  metadata/pose.pqt, metadata/sessions.pqt (session_type/start_time for
        dot coloring)
Output: figures/pose_qc/pose_qc_grid.svg,
        figures/pose_qc/pose_timing_grid.svg,
        figures/pose_qc/pose_corr_grid.svg
"""
import argparse

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

from iblnm.config import (
    FIGURE_DPI,
    POSE_FPATH,
    POSE_MEASURES,
    PROJECT_ROOT,
    QCCMAP,
    QCVAL2NUM,
    QC_VALUE_ORDER,
    SESSIONS_FPATH,
    SESSIONTYPE2COLOR,
    VIDEO_QC_COLS,
)
from iblnm.vis import violinplot

PEAK_LAG_COLS = ['peak_lag_early', 'peak_lag_mid', 'peak_lag_late']
PEAK_VAL_COLS = ['peak_val_early', 'peak_val_mid', 'peak_val_late']
CORR_COLS = ['fraction_correct', 'log_rt', 'abs_lag', 'peak_value', 'abs_drift']
TIMING_MEASURES = ['peak_value', 'abs_lag', 'abs_drift']


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Append the per-session timing metrics consumed by the grid builders.

    Parameters
    ----------
    df : pd.DataFrame
        Pose table carrying ``mean_rt`` (seconds), ``drift``, and the per-third
        ``PEAK_LAG_COLS`` and ``PEAK_VAL_COLS`` columns.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with ``log_rt`` (log reaction time), ``abs_lag`` (max
        |paw-wheel lag| across thirds, seconds), ``peak_value`` (max paw-wheel
        xcorr peak across thirds), and ``abs_drift`` (|drift|) added.
    """
    return df.assign(
        log_rt=np.log(df['mean_rt']),
        abs_lag=max_abs_lag(df),
        peak_value=max_peak_value(df),
        abs_drift=df['drift'].abs(),
    )


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


DATE_CMAP = 'viridis'


def add_session_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Left-join ``session_type`` and ``start_time`` from the session catalog.

    Parameters
    ----------
    df : pd.DataFrame
        Pose table carrying an ``eid`` column.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with ``session_type`` (str) and ``start_time`` (ISO
        datetime str) joined from ``SESSIONS_FPATH`` on ``eid``.
    """
    catalog = pd.read_parquet(SESSIONS_FPATH)[['eid', 'session_type', 'start_time']]
    return df.merge(catalog, on='eid', how='left')


def point_colors(df: pd.DataFrame, color_by: str):
    """Per-session scatter-dot colors and an optional colorbar mappable.

    Parameters
    ----------
    df : pd.DataFrame
        Pose table; needs ``session_type`` when ``color_by='session_type'`` or
        ``start_time`` when ``color_by='date'``.
    color_by : {'session_type', 'date'}
        ``'session_type'`` maps each dot through ``SESSIONTYPE2COLOR``;
        ``'date'`` maps each session's acquisition date onto ``DATE_CMAP``,
        normalized across the table.

    Returns
    -------
    colors : np.ndarray
        Per-row color: color-name strings for ``session_type``, RGBA rows for
        ``date``. Length matches ``df``.
    mappable : matplotlib.cm.ScalarMappable or None
        Date colorbar source (carries the norm and cmap); ``None`` for
        ``session_type``.
    """
    if color_by == 'date':
        ordinals = pd.to_datetime(df['start_time']).map(pd.Timestamp.toordinal)
        ordinals = ordinals.to_numpy(dtype=float)
        norm = Normalize(ordinals.min(), ordinals.max())
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=DATE_CMAP)
        return mappable.to_rgba(ordinals), mappable
    return df['session_type'].map(SESSIONTYPE2COLOR).to_numpy(), None


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


def _scatter_cell(ax, df: pd.DataFrame, measure: str, metric: str, colors) -> None:
    """Scatter ``measure`` (y) vs ``metric`` (x) and annotate Spearman r, p.

    ``colors`` is a per-row dot color array (positionally aligned to ``df``)
    from :func:`point_colors`.
    """
    ax.scatter(df[metric], df[measure], s=5, c=colors, edgecolors='none')
    r, p = spearman_finite(df[metric], df[measure])
    ax.annotate(f'r={r:.2f}\np={p:.1e}', (0.05, 0.95), xycoords='axes fraction',
                ha='left', va='top', fontsize=6)
    ax.set_xlabel(metric, fontsize=6)


def _add_color_key(fig, axes, df: pd.DataFrame, color_by: str, mappable) -> None:
    """Attach a date colorbar or a session-type legend to ``fig``.

    For ``color_by='date'`` draws a colorbar off ``mappable`` with date-labeled
    ticks; otherwise draws a legend with one marker per ``session_type`` present
    in ``df``, ordered as in ``SESSIONTYPE2COLOR``.
    """
    if mappable is not None:
        cbar = fig.colorbar(mappable, ax=axes, fraction=0.02, pad=0.01)
        ticks = np.linspace(mappable.norm.vmin, mappable.norm.vmax, 5)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([pd.Timestamp.fromordinal(int(t)).date().isoformat()
                             for t in ticks], fontsize=6)
        return
    present = [t for t in SESSIONTYPE2COLOR if t in set(df['session_type'])]
    handles = [Line2D([0], [0], marker='o', linestyle='', markersize=4,
                      color=SESSIONTYPE2COLOR[t], label=t) for t in present]
    fig.legend(handles=handles, loc='center left', bbox_to_anchor=(1.0, 0.5),
               fontsize=6)


def build_violin_grid(df: pd.DataFrame, measures: list[str]):
    """Build a measure-vs-video-QC violin grid (``measures`` x ``VIDEO_QC_COLS``).

    Each row is one column of ``df`` named in ``measures``; each grid column is
    one ``VIDEO_QC_COLS`` field, with the measure split into violins by that
    field's QC categories.

    Parameters
    ----------
    df : pd.DataFrame
        Pose table holding every ``measures`` column and every ``VIDEO_QC_COLS``
        field.
    measures : list[str]
        Column names to plot as rows (e.g. ``list(POSE_MEASURES)`` or
        ``TIMING_MEASURES``).

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(len(measures), len(VIDEO_QC_COLS), sharey='row',
                             figsize=(2 * len(VIDEO_QC_COLS), 2 * len(measures)))
    for j, qc_col in enumerate(VIDEO_QC_COLS):
        axes[0, j].set_title(qc_col.replace('qc_videoLeft_', ''), fontsize=8)
    for i, measure in enumerate(measures):
        axes[i, 0].set_ylabel(measure, fontsize=8)
        for j, qc_col in enumerate(VIDEO_QC_COLS):
            _violin_cell(axes[i, j], df, measure, qc_col)
    fig.tight_layout()
    return fig


def build_corr_grid(df: pd.DataFrame, color_by: str = 'session_type'):
    """Build the movement-vs-metric Spearman scatter grid (rows x ``CORR_COLS``).

    Rows are the ``POSE_MEASURES`` keys; columns are ``fraction_correct``, log
    reaction time, the max paw-wheel ``abs_lag``, the max xcorr ``peak_value``,
    and the absolute ``drift`` — expected on ``df`` via ``add_derived_metrics``.

    Parameters
    ----------
    df : pd.DataFrame
        Pose table with the derived metric columns plus, for coloring,
        ``session_type`` and ``start_time`` (from :func:`add_session_metadata`).
    color_by : {'session_type', 'date'}
        How to color the per-session dots; see :func:`point_colors`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    colors, mappable = point_colors(df, color_by)
    measures = list(POSE_MEASURES)
    fig, axes = plt.subplots(len(measures), len(CORR_COLS), sharey='row',
                             figsize=(2 * len(CORR_COLS), 2 * len(measures)))
    for i, measure in enumerate(measures):
        axes[i, 0].set_ylabel(measure, fontsize=8)
        for j, metric in enumerate(CORR_COLS):
            _scatter_cell(axes[i, j], df, measure, metric, colors)
    fig.tight_layout()
    _add_color_key(fig, axes, df, color_by, mappable)
    return fig


def main(color_by: str = 'session_type') -> None:
    df = add_session_metadata(add_derived_metrics(pd.read_parquet(POSE_FPATH)))
    figures_dir = PROJECT_ROOT / 'figures/pose_qc'
    figures_dir.mkdir(parents=True, exist_ok=True)
    builders = [
        ('pose_qc_grid', lambda: build_violin_grid(df, list(POSE_MEASURES))),
        ('pose_timing_grid', lambda: build_violin_grid(df, TIMING_MEASURES)),
        ('pose_corr_grid', lambda: build_corr_grid(df, color_by)),
    ]
    for name, builder in builders:
        fpath = figures_dir / f'{name}.svg'
        builder().savefig(fpath, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Saved {fpath}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--color-by', choices=['session_type', 'date'],
                        default='session_type',
                        help='Color pose_corr_grid dots by session type (default) '
                             'or acquisition date.')
    main(color_by=parser.parse_args().color_by)
