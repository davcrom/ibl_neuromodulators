import itertools
import re
import warnings
from collections.abc import Hashable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import quantile_transform

from iblnm.config import (
    ANALYSIS_CONTRASTS, NM_CMAPS, QCCMAP,
    PERSESSION_SIGNIFICANCE_ALPHA, PERSESSION_REGRESSORS,
    RESPONSE_DROPPED_TERMS, RESPONSE_EVENTS, RESPONSE_MAGNITUDE_WINDOW,
    STIM_ONSET_EVENT,
    SESSION_GROUPS, SESSIONTYPE2COLOR, TARGETNM2POSITION,
    TARGETNM_COLORS, TARGETNMS_TO_ANALYZE,
    TICKFONTSIZE, LABELFONTSIZE,
)
from iblnm.analysis import raised_cosine_basis
from iblnm.util import get_contrast_coding


def _coef_label(term):
    """Map a model coefficient name to a display label.

    Strips ``psych_50_`` prefix and converts ``:`` to `` × ``.
    """
    label = term.replace('psych_50_', '')
    return label.replace(':', ' × ')


def set_plotsize(w, h=None, ax=None):
    """
    Set the size of a matplotlib axes object in cm.

    Parameters
    ----------
    w, h : float
        Desired width and height of plot, if height is None, the axis will be
        square.

    ax : matplotlib.axes
        Axes to resize, if None the output of plt.gca() will be re-sized.

    Notes
    -----
    - Use after subplots_adjust (if adjustment is needed)
    - Matplotlib axis size is determined by the figure size and the subplot
      margins (r, l; given as a fraction of the figure size), i.e.
      w_ax = w_fig * (r - l)
    """
    if h is None: # assume square
        h = w
    w /= 2.54 # convert cm to inches
    h /= 2.54
    if not ax: # get current axes
        ax = plt.gca()
    # get margins
    left = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    # set fig dimensions to produce desired ax dimensions
    figw = float(w)/(r-left)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)


def _session_group_colors() -> dict:
    """`config.SESSION_GROUPS` as a label -> color map, in config order."""
    return {label: group_spec['color']
            for label, group_spec in SESSION_GROUPS.items()}


def session_overview_matrix(group, columns='session_n', ax=None,
                            color_by='session_group', split_color_map=None):
    """
    Plot a matrix of sessions per subject, one color layer per session group.

    Each group is painted as its own layer: cells belonging to the group carry
    the group's color, every other cell is NaN and stays transparent, so the
    layers stack without hiding one another. All sessions in group._catalog are
    shown at 50% opacity. Sessions in group.sessions (passing the current
    filter) are overlaid at 100% opacity.

    Parameters
    ----------
    group : PhotometrySessionGroup
        Group object with current filter applied. _catalog provides the
        background; sessions provides the highlighted foreground.
    columns : str
        Column to use for x-axis (e.g., 'day_n', 'session_n').
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    color_by : str
        Column to color cells by. Defaults to 'session_group'.
    split_color_map : dict, optional
        Mapping from color_by values to colors, in the order the layers are
        drawn and the legend is listed. Defaults to the SESSION_GROUPS colors.

    Raises
    ------
    ValueError
        If there is more than one session per (subject, columns) cell in _catalog.
    """
    _color_map = split_color_map or _session_group_colors()

    df_base = group._catalog
    df_overlay = group.sessions

    # Subject order: earliest start_time across all catalog sessions
    first_start = df_base.groupby('subject')['start_time'].min().sort_values()
    subject_order = first_start.index.tolist()

    # Check for duplicates in the catalog
    duplicates = df_base.groupby(['subject', columns]).size()
    if (duplicates > 1).any():
        dup_cells = duplicates[duplicates > 1]
        raise ValueError(
            f"Multiple sessions per cell. Remove duplicates before plotting.\n"
            f"Duplicates:\n{dup_cells}"
        )

    base_matrix = df_base.pivot(index='subject', columns=columns, values=color_by)
    base_matrix = base_matrix.reindex(subject_order)
    # Reindex overlay to the same shape as base; cells it lacks are not painted
    overlay_matrix = df_overlay.pivot(
        index='subject', columns=columns, values=color_by
    ).reindex(index=subject_order, columns=base_matrix.columns)

    groups_present = [label for label in _color_map
                      if (base_matrix == label).any().any()]

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(0.15 * len(base_matrix.columns), 0.15 * len(base_matrix))
        )

    for label in groups_present:
        cmap = colors.ListedColormap([_color_map[label]])
        for matrix, alpha in [(base_matrix, 0.5), (overlay_matrix, 1)]:
            membership = np.where(matrix == label, 1.0, np.nan)
            ax.matshow(membership, cmap=cmap, vmin=0, vmax=1, alpha=alpha)

    ax.legend(handles=[Patch(facecolor=_color_map[label], label=label)
                       for label in groups_present])

    # Format axes
    ax.set_yticks(np.arange(len(base_matrix)))
    ax.set_yticklabels(base_matrix.index)
    ax.set_ylabel('Subject')
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(0, len(base_matrix.columns) + 1, 10))
    ax.tick_params(axis='x', rotation=90)
    ax.xaxis.set_label_position('top')
    ax.set_xlabel(columns)

    # Gridlines
    for xtick in np.arange(len(base_matrix.columns)):
        ax.axvline(xtick - 0.5, color='white')
    for ytick in np.arange(len(base_matrix)):
        ax.axhline(ytick - 0.5, color='white')

    return ax


def target_overview_barplot(df_sessions, ax=None, barwidth=0.8,
                            color_by='session_group', split_color_map=None,
                            horizontal=False):
    """Stacked bar plot of session counts per target region.

    Each bar stacks the categories of ``color_by``, a segment's fill taken from
    ``split_color_map[category]``. The map's key order is the stacking order,
    bottom to top.

    Parameters
    ----------
    df_sessions : pandas.DataFrame
        One row per recording, with ``target_NM``, ``subject``, ``eid``, and the
        ``color_by`` column.
    ax : matplotlib.axes.Axes, optional
    barwidth : float
    color_by : str
        Column whose categories are stacked within each target's bar.
    split_color_map : dict, optional
        Maps ``color_by`` category to fill color, in stacking order. Defaults to
        the SESSION_GROUPS colors.
    horizontal : bool
        If True, draw horizontal bars.
    """
    _color_map = split_color_map or _session_group_colors()

    if len(df_sessions) == 0:
        if ax is None:
            fig, ax = plt.subplots()
        ax.set_title("No data to plot")
        return ax

    # Create a target_NM x color_by matrix with session counts
    df_n = df_sessions.pivot_table(
        columns=color_by,
        index='target_NM',
        aggfunc='size',
        fill_value=0
    )

    if ax is None:
        fig, ax = plt.subplots()

    # Use contiguous positions for targets present in data (sorted by canonical order)
    sorted_targets = sorted(df_n.index, key=lambda x: TARGETNM2POSITION.get(x, 999))
    df_n = df_n.reindex(sorted_targets)
    positions = list(range(len(df_n)))
    cumulative = np.zeros(len(df_n))

    categories = [c for c in _color_map if c in df_n.columns]
    for category in categories:
        ns = df_n[category]
        color = _color_map[category]
        if horizontal:
            ax.barh(positions, ns, left=cumulative, height=barwidth,
                    color=color, label=category)
        else:
            ax.bar(positions, ns, bottom=cumulative, width=barwidth,
                   color=color, label=category)
        _add_bar_labels(ax, positions, ns, horizontal=horizontal,
                        bottoms=cumulative)
        cumulative += ns

    n_mice = df_sessions.groupby('target_NM').apply(
        lambda x: len(x['subject'].unique()),
        include_groups=False
    )
    tick_labels = [
        '%s\n(%d mice)' % (target_NM, n_mice.loc[target_NM])
        for target_NM in df_n.index
    ]

    count_axis = ax.xaxis if horizontal else ax.yaxis
    if horizontal:
        ax.set_yticks(positions)
        ax.set_yticklabels(tick_labels)
        ax.invert_yaxis()
        ax.set_xlabel('N Sessions')
        ax.set_ylabel('Target-NM')
    else:
        ax.set_xticks(positions)
        ax.set_xticklabels(tick_labels)
        ax.set_xlim(right=max(positions) + barwidth)
        ax.tick_params(axis='x', rotation=90)
        ax.set_ylabel('N Sessions')
        ax.set_xlabel('Target-NM')
    # Let matplotlib pick a handful of round count ticks instead of one per 100.
    count_axis.set_major_locator(MaxNLocator(nbins=6, integer=True))

    ax.legend()

    n_recordings = len(df_sessions)
    n_sessions = df_sessions['eid'].nunique()
    n_mice = df_sessions['subject'].nunique()
    ax.set_title(f"{n_recordings} recordings, {n_sessions} sessions, {n_mice} mice")

    return ax


def plot_baseline_propsig(results: pd.DataFrame, ax=None) -> plt.Figure:
    """Paired bars of significant vs non-significant session fractions per target.

    For each ``target_NM`` group, draws two side-by-side bars: the fraction of
    sessions with ``p_value <= 0.05`` (significant, full opacity) and the
    fraction with ``p_value > 0.05`` (non-significant, ``alpha=0.4``). Targets
    are ordered along x by ``TARGETNM2POSITION`` and colored by
    ``TARGETNM_COLORS``.

    Parameters
    ----------
    results : pandas.DataFrame
        Per-session baseline results with ``target_NM`` and ``p_value`` columns.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure/axes is created when omitted.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the barplot.
    """
    prop_sig = results.groupby('target_NM')['p_value'].apply(lambda p: (p <= 0.05).mean())
    prop_nonsig = results.groupby('target_NM')['p_value'].apply(lambda p: (p > 0.05).mean())

    targets = sorted(
        prop_sig.index,
        key=lambda t: TARGETNM2POSITION.get(t, len(TARGETNM2POSITION))
    )
    prop_sig = prop_sig.reindex(targets)
    prop_nonsig = prop_nonsig.reindex(targets)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    width = 0.25
    positions = range(len(targets))
    for i, nm in enumerate(targets):
        color = TARGETNM_COLORS.get(nm, 'gray')
        ax.bar(i - width / 2, prop_sig[nm], width=width, color=color, label=nm)
        ax.bar(i + width / 2, prop_nonsig[nm], width=width, color=color, alpha=0.4)

    ax.set_xticks(positions)
    ax.set_xticklabels(targets, ha='right')
    ax.set_ylabel('Fraction of sessions')

    return ax.figure


def plot_baseline_slope(results: pd.DataFrame, ax=None) -> plt.Figure:
    """Per-session baseline regression slope, one x-slot per mouse.

    Each session is a translucent open dot at its ``observed_slope``. Mice are
    laid out one x-slot apart within their ``target_NM`` group, group width
    scaling with mouse count (via :func:`_group_xslots`); within a group mice
    are ordered left to right by ascending median slope. A session is
    edge-colored by its ``target_NM`` (``TARGETNM_COLORS``) when significant
    (``p_value <= 0.05``) and gray otherwise. A dashed line marks ``slope = 0``.
    Errored sessions (NaN ``observed_slope`` or ``p_value``) are dropped.

    Parameters
    ----------
    results : pandas.DataFrame
        Per-session baseline results with ``subject``, ``target_NM``,
        ``p_value``, and ``observed_slope`` columns.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure/axes is created when omitted.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the slope scatter.
    """
    df = results.dropna(subset=['observed_slope', 'p_value'])
    targets = sorted(
        df['target_NM'].unique(),
        key=lambda t: TARGETNM2POSITION.get(t, len(TARGETNM2POSITION))
    )

    subjects_by_target, slots_by_target, ticks = _group_xslots(df, targets)

    if ax is None:
        # Width tracks the mouse count so dots are not squeezed into a fixed
        # canvas (the source of the wasted margins); height stays compact.
        extent = max((slots.max() for slots in slots_by_target.values()),
                     default=1.0)
        fig, ax = plt.subplots(figsize=(0.45 * extent + 1.5, 5))

    for tnm, subjects in subjects_by_target.items():
        color = TARGETNM_COLORS.get(tnm, 'gray')
        slopes_by_subject = {
            s: df.loc[df['subject'] == s, ['p_value', 'observed_slope']]
            for s in subjects}
        ordered = sorted(
            subjects, key=lambda s: np.median(slopes_by_subject[s]['observed_slope']))
        for subject, x in zip(ordered, slots_by_target[tnm]):
            sub = slopes_by_subject[subject]
            sig = sub['p_value'] <= 0.05
            # Fill encodes significance, hue encodes target-NM: significant
            # sessions are filled in the target colour so they read against the
            # majority; non-significant ones are open in the same colour.
            sig_slopes = sub.loc[sig, 'observed_slope'].to_numpy()
            nonsig_slopes = sub.loc[~sig, 'observed_slope'].to_numpy()
            ax.scatter(np.full(len(nonsig_slopes), x), nonsig_slopes, marker='o',
                       facecolors='none', edgecolors=color, linewidths=0.8,
                       s=_SESSION_MARKER_SIZE, alpha=0.5, zorder=3)
            ax.scatter(np.full(len(sig_slopes), x), sig_slopes, marker='o',
                       facecolors=color, edgecolors=color, linewidths=0.8,
                       s=_SESSION_MARKER_SIZE, alpha=0.85, zorder=4)

    ax.axhline(0, ls='--', color='gray', lw=0.5)
    ax.margins(x=0.01)
    ax.set_xticks([centre for _, centre in ticks])
    ax.set_xticklabels([tnm for tnm, _ in ticks], rotation=30, ha='right',
                       fontsize=TICKFONTSIZE)
    ax.set_ylabel('Slope')
    # Fill legend, drawn in neutral grey since hue carries target-NM, not
    # significance.
    legend_handles = [
        Line2D([], [], marker='o', linestyle='none', markerfacecolor='gray',
               markeredgecolor='gray', label='significant (p ≤ 0.05)'),
        Line2D([], [], marker='o', linestyle='none', markerfacecolor='none',
               markeredgecolor='gray', label='non-significant (p > 0.05)'),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc='upper right',
              fontsize=TICKFONTSIZE)

    return ax.figure


def _color_violin(parts, color, sig: bool) -> None:
    """Style violin bodies: significant filled, non-significant unfilled outline."""
    for body in parts['bodies']:
        body.set_facecolor(color if sig else 'none')
        body.set_edgecolor(color)
        body.set_alpha(0.7 if sig else 1.0)
    for key in ('cbars', 'cmins', 'cmaxes'):
        if key in parts:
            parts[key].set_edgecolor(color)


def plot_baseline_r2(results: pd.DataFrame, ax=None) -> plt.Figure:
    """Violins of observed R² per target, split by significance.

    For each ``target_NM`` group, draws a violin of ``observed_r2`` for the
    significant sessions (``p_value <= 0.05``, filled, ``alpha=0.7``) offset left
    of the target position and one for the non-significant sessions
    (``p_value > 0.05``, unfilled outline) offset right. Empty subsets are
    skipped. Each violin is annotated with its ``n`` above the axis. Targets are
    ordered along x by ``TARGETNM2POSITION`` and colored by ``TARGETNM_COLORS``.

    Parameters
    ----------
    results : pandas.DataFrame
        Per-session baseline results with ``target_NM``, ``p_value``, and
        ``observed_r2`` columns.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure/axes is created when omitted.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the violin plot.
    """
    targets = sorted(
        results['target_NM'].dropna().unique(),
        key=lambda t: TARGETNM2POSITION.get(t, len(TARGETNM2POSITION))
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    for i, target in enumerate(targets):
        group = results[results['target_NM'] == target]
        color = TARGETNM_COLORS.get(target, 'gray')
        sig = group[group['p_value'] <= 0.05]['observed_r2'].dropna().values
        non_sig = group[group['p_value'] > 0.05]['observed_r2'].dropna().values

        if len(sig) > 0:
            parts = ax.violinplot(sig, positions=[i - 0.2], widths=0.35, bw_method=0.2)
            _color_violin(parts, color, sig=True)
        if len(non_sig) > 0:
            parts = ax.violinplot(non_sig, positions=[i + 0.2], widths=0.35, bw_method=0.2)
            _color_violin(parts, color, sig=False)

        ax.text(i - 0.2, ax.get_ylim()[1], f'n={len(sig)}', ha='center', va='bottom', fontsize=10)
        ax.text(i + 0.2, ax.get_ylim()[1], f'n={len(non_sig)}', ha='center', va='bottom', fontsize=10)

    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(targets)
    ax.set_ylabel('Observed R²')

    return ax.figure


# Presentation config per baseline model: the y-axis label for the behavior
# variable and whether it is binary (0/1 outcome fit with a logistic curve) or
# continuous (fit with a line).
_BASELINE_MODEL_DISPLAY = {
    'performance': {'behavior_label': 'correct', 'binary': True,
                    'curve_label': 'P(right)'},
    'reaction_time': {'behavior_label': 'log RT', 'binary': False,
                      'curve_label': 'log RT'},
}


def _lighten(color, amount: float = 0.55) -> tuple:
    """Blend ``color`` toward white by ``amount`` (0 = unchanged, 1 = white)."""
    rgb = np.array(colors.to_rgb(color))
    return tuple(rgb + (1.0 - rgb) * amount)


def _aggregate_tercile_curves(session_curves: list[pd.DataFrame], tercile: str):
    """Cross-session mean±SD of one tercile's per-contrast curve.

    Reindexes each session's ``tercile`` column onto the union of signed-contrast
    levels, then reduces across sessions with ``nanmean``/``nanstd``. Levels that
    are all-NaN (no session contributed a cell) are dropped.

    Parameters
    ----------
    session_curves : list of pandas.DataFrame
        One DataFrame per session, indexed by signed contrast with ``low`` and
        ``high`` columns of mean outcome (NaN where guarded/absent).
    tercile : str
        Column to aggregate, ``'low'`` or ``'high'``.

    Returns
    -------
    levels : numpy.ndarray
        Signed-contrast levels retained (those with a finite mean).
    mean : numpy.ndarray
        Per-level ``nanmean`` of the tercile across sessions.
    sd : numpy.ndarray
        Per-level ``nanstd`` (population, ``ddof=0``) across sessions.
    """
    levels = np.array(sorted(set().union(*(df.index for df in session_curves))))
    stacked = np.vstack([df[tercile].reindex(levels).to_numpy()
                         for df in session_curves])
    with warnings.catch_warnings():
        # All-NaN levels yield NaN here; we drop them below rather than warn.
        warnings.simplefilter('ignore', RuntimeWarning)
        mean = np.nanmean(stacked, axis=0)
        sd = np.nanstd(stacked, axis=0)
    keep = ~np.isnan(mean)
    return levels[keep], mean[keep], sd[keep]


def plot_baseline_tercile_curves(curves_by_target: dict[str, list[pd.DataFrame]],
                                 model: str) -> plt.Figure:
    """Low- vs high-baseline behavioral curves, one axes per target-NM.

    For each ``target_NM`` group, aggregates its significant sessions'
    tercile-split curves into a mean line with a shaded ±SD band per tercile
    (:func:`_aggregate_tercile_curves`). The high tercile is a solid line in the
    target-NM color (``TARGETNM_COLORS``); the low tercile is the same hue
    lightened. Axes are ordered by ``TARGETNM2POSITION`` in a single row.

    Parameters
    ----------
    curves_by_target : dict of str to list of pandas.DataFrame
        Maps each ``target_NM`` to its sessions' curves, each DataFrame indexed
        by signed contrast with ``low`` and ``high`` mean-outcome columns (the
        ticket-01 interchange format).
    model : str
        Baseline model name; selects the y-axis label from
        ``_BASELINE_MODEL_DISPLAY`` (``'P(right)'`` / ``'log RT'``).

    Returns
    -------
    matplotlib.figure.Figure
        The row of tercile-curve axes.
    """
    ylabel = _BASELINE_MODEL_DISPLAY[model]['curve_label']
    targets = sorted(
        curves_by_target,
        key=lambda t: TARGETNM2POSITION.get(t, len(TARGETNM2POSITION))
    )

    fig, axes = plt.subplots(1, len(targets), figsize=(4 * len(targets), 4),
                             squeeze=False, sharey=True)
    for ax, target in zip(axes[0], targets):
        color = TARGETNM_COLORS.get(target, 'gray')
        for tercile, hue in (('high', color), ('low', _lighten(color))):
            levels, mean, sd = _aggregate_tercile_curves(
                curves_by_target[target], tercile)
            ax.plot(levels, mean, color=hue, label=tercile, zorder=3)
            ax.fill_between(levels, mean - sd, mean + sd, color=hue, alpha=0.25)
        ax.set_xlabel('signed contrast (%)')
        ax.set_title(target)

    axes[0][0].set_ylabel(ylabel)
    legend_handles = [
        Line2D([], [], color='gray', label='high'),
        Line2D([], [], color=_lighten('gray'), label='low'),
    ]
    fig.legend(handles=legend_handles, frameon=False, loc='upper right',
               fontsize=TICKFONTSIZE)
    return fig


def plot_baseline_tercile_difference(
        curves_by_target: dict[str, list[pd.DataFrame]], model: str) -> plt.Figure:
    """Per-session high-minus-low tercile difference, one axes per target-NM.

    Makes each session its own control: for every significant session it plots
    the per-contrast ``high - low`` tercile difference as a thin translucent line
    in the target-NM color, overlaid with a bold cross-session mean line
    (:func:`_aggregate_tercile_curves` on the difference) and a dashed zero
    reference. Axes are ordered by ``TARGETNM2POSITION`` in a single row.

    Parameters
    ----------
    curves_by_target : dict of str to list of pandas.DataFrame
        Maps each ``target_NM`` to its sessions' curves, each DataFrame indexed
        by signed contrast with ``low`` and ``high`` mean-outcome columns (the
        ticket-01 interchange format).
    model : str
        Baseline model name; selects the y-axis label from
        ``_BASELINE_MODEL_DISPLAY`` (``'P(right)'`` / ``'log RT'``).

    Returns
    -------
    matplotlib.figure.Figure
        The row of difference axes.
    """
    ylabel = f"Δ {_BASELINE_MODEL_DISPLAY[model]['curve_label']} (high − low)"
    targets = sorted(
        curves_by_target,
        key=lambda t: TARGETNM2POSITION.get(t, len(TARGETNM2POSITION))
    )
    fig, axes = plt.subplots(1, len(targets), figsize=(4 * len(targets), 4),
                             squeeze=False, sharey=True)
    for ax, target in zip(axes[0], targets):
        color = TARGETNM_COLORS.get(target, 'gray')
        diffs = [(df['high'] - df['low']).to_frame('diff')
                 for df in curves_by_target[target]]
        for diff in diffs:
            ax.plot(diff.index, diff['diff'], color=color, lw=0.5, alpha=0.3,
                    label='session', zorder=2)
        levels, mean, _ = _aggregate_tercile_curves(diffs, 'diff')
        ax.plot(levels, mean, color=color, lw=2, label='mean', zorder=3)
        ax.axhline(0, ls='--', color='gray', lw=0.5, zorder=1)
        ax.set_xlabel('signed contrast (%)')
        ax.set_title(target)

    axes[0][0].set_ylabel(ylabel)
    return fig


def plot_baseline_schematic(baseline: np.ndarray, behavior: np.ndarray,
                            model: str, seed: int = 0) -> plt.Figure:
    """Method-schematic strip for the baseline-coding analysis.

    Single row of four panels reading left to right as the analysis narrative:
    (1) the two modelled per-trial variables as traces over trial index,
    (2) their scatter with a display fit curve, (3) a drawn cartoon of the
    donor-swap, and (4) a drawn cartoon of the resulting null distribution with
    the observed statistic marked. Only panels 1-2 use real data; panels 3-4 are
    synthetic illustrations seeded by ``seed``.

    Parameters
    ----------
    baseline : numpy.ndarray
        Per-trial z-scored pre-stimulus baseline, one value per trial.
    behavior : numpy.ndarray
        Per-trial behavior aligned to ``baseline``: 0/1 correctness for the
        ``performance`` model, log reaction time for ``reaction_time``.
    model : str
        ``'performance'`` or ``'reaction_time'``; selects the behavior label and
        whether the scatter fit is logistic (binary) or linear (continuous).
    seed : int, optional
        Seed for the synthetic cartoon panels (default 0), so the figure is
        reproducible.

    Returns
    -------
    matplotlib.figure.Figure
        The four-panel schematic figure.
    """
    display = _BASELINE_MODEL_DISPLAY[model]
    keep = ~(np.isnan(baseline) | np.isnan(behavior))
    baseline, behavior = baseline[keep], behavior[keep]
    color = TARGETNM_COLORS.get('VTA-DA', 'gray')
    rng = np.random.default_rng(seed)

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.2))
    _schematic_traces(axes[0], baseline, behavior, display, color)
    _schematic_scatter(axes[1], baseline, behavior, display, color, rng)
    _schematic_swap(axes[2], rng, color)
    _schematic_null(axes[3], color)
    fig.tight_layout(w_pad=3.0)
    return fig


def _schematic_traces(ax, baseline: np.ndarray, behavior: np.ndarray,
                      display: dict, color: str) -> None:
    """Panel 1: baseline (left axis) and behavior (right axis) over trial index."""
    # Show only the middle 100 trials so individual fluctuations stay legible.
    mid = len(baseline) // 2
    window = slice(max(0, mid - 50), mid + 50)
    trials = np.arange(len(baseline))[window]
    baseline, behavior = baseline[window], behavior[window]
    ax.plot(trials, baseline, color=color, lw=0.8)
    ax.set_xlabel('trial')
    ax.set_ylabel('pre-trial fluorescence (z)', color=color)
    behavior_ax = ax.twinx()
    if display['binary']:
        behavior_ax.plot(trials, behavior, 'o', color='k', ms=3, alpha=0.5)
        behavior_ax.set_yticks([0, 1])
    else:
        behavior_ax.plot(trials, behavior, color='k', lw=0.8, alpha=0.7)
    behavior_ax.set_ylabel(display['behavior_label'], color='k')


def _schematic_scatter(ax, baseline: np.ndarray, behavior: np.ndarray,
                       display: dict, color: str, rng: np.random.Generator) -> None:
    """Panel 2: baseline-vs-behavior points plus a single-predictor display fit.

    The fit ignores the contrast covariate used by the real model; it is a
    logistic curve for the binary outcome and a regression line otherwise.
    """
    y = behavior + rng.uniform(-0.06, 0.06, len(behavior)) if display['binary'] \
        else behavior
    ax.scatter(baseline, y, s=12, color=color, alpha=0.4, edgecolors='none')
    grid = np.linspace(baseline.min(), baseline.max(), 100)
    if display['binary']:
        fit = LogisticRegression().fit(baseline[:, None], behavior)
        curve = fit.predict_proba(grid[:, None])[:, 1]
        ax.set_yticks([0, 1])
    else:
        slope, intercept = np.polyfit(baseline, behavior, 1)
        curve = slope * grid + intercept
    ax.plot(grid, curve, color='k', lw=1.5)
    ax.set_xlabel('pre-trial fluorescence (z)')
    ax.set_ylabel(display['behavior_label'])


def _signal_trace(t: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    """Synthetic photometry-like trace: photobleaching, drift, and calcium transients.

    Overlays a decaying baseline (``exp(-rate * t)``) with a slow random-walk
    drift and several sharp positive transients so the cartoon reads as signal,
    not bleaching alone.
    """
    bleach = np.exp(-rate * t)
    drift = 0.12 * rng.standard_normal(t.size).cumsum() / np.sqrt(t.size)
    transients = sum(
        rng.uniform(0.06, 0.16) * np.exp(-0.5 * ((t - rng.uniform(0, 1)) / 0.02) ** 2)
        for _ in range(rng.integers(5, 9)))
    return bleach + drift + transients + 0.03 * rng.standard_normal(t.size)


def _schematic_swap(ax, rng: np.random.Generator, color: str) -> None:
    """Panel 3: drawn cartoon of the donor-swap keeping behavior fixed.

    The same fake behavior series is drawn above the observed photometry-like
    trace and again above the permuted pool, since the swap replaces the
    fluorescence but not the behavior. A small arrow to an R² sits outside the
    panel beside each photometry trace (observed in the target colour, permuted
    in grey) for the statistic each swap produces. Cartoon only.
    """
    t = np.linspace(0, 1, 200)
    behavior = 0.35 * (rng.integers(0, 2, t.size) - 0.5)
    permuted_y = [-i * 1.1 for i in range(4)]

    ax.plot(t, behavior + 4.6, color='k', lw=0.9)
    ax.annotate('behavior', (0.02, 5.0), color='k', fontsize=TICKFONTSIZE)
    ax.plot(t, _signal_trace(t, 1.2, rng) + 3.2, color=color, lw=1.0)
    ax.annotate('observed photometry', (0.02, 3.7), color=color, fontsize=TICKFONTSIZE)
    ax.plot(t, behavior + 1.4, color='k', lw=0.9)
    ax.annotate('behavior', (0.02, 1.8), color='k', fontsize=TICKFONTSIZE)
    for y in permuted_y:
        ax.plot(t, _signal_trace(t, rng.uniform(0.8, 1.6), rng) + y, color='gray',
                lw=0.8, alpha=0.6)
    ax.annotate('permuted photometry', (0.02, -3.6), color='gray', fontsize=TICKFONTSIZE)

    for y, c in [(3.2, color)] + [(py, 'gray') for py in permuted_y]:
        ax.annotate('', xy=(1.13, y), xytext=(1.02, y), annotation_clip=False,
                    arrowprops=dict(arrowstyle='->', color=c))
        ax.text(1.16, y, r'$R^2$', color=c, va='center', fontsize=TICKFONTSIZE,
                clip_on=False)

    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('swap photometry across sessions')


def _schematic_null(ax, color: str) -> None:
    """Panel 4: drawn cartoon null distribution with the observed statistic marked.

    A stylized Gaussian null with a vertical line for the observed statistic out
    in the right tail. Illustration only — no real permutation values.
    """
    x = np.linspace(-3.5, 4.5, 200)
    null = np.exp(-0.5 * x ** 2)
    ax.fill_between(x, null, color='gray', alpha=0.4)
    observed = 3.2
    ax.axvline(observed, color=color, lw=1.5)
    ax.annotate('observed R²', (observed, 0.9), color=color,
                fontsize=TICKFONTSIZE, ha='center')
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlabel(r'null $R^2$')


def plot_dispersion_scatter(df, events, blocks):
    """Grid of behavioral-vs-neural coefficient-dispersion scatters.

    One panel per ``(block, event)``: rows are ``blocks`` (e.g. task/movement),
    columns are ``events``. Within a panel each ``(subject, target_NM)`` unit is
    one marker at ``(behavioral_dispersion, neural_dispersion)``, colored by its
    ``target_NM`` via ``TARGETNM_COLORS``.

    Parameters
    ----------
    df : pandas.DataFrame
        Plot-ready frame with columns ``['subject', 'target_NM', 'event',
        'block', 'neural_dispersion', 'behavioral_dispersion']``.
    events : list of str
        Event names, in column order.
    blocks : list of str
        Block labels, in row order.

    Returns
    -------
    matplotlib.figure.Figure
        The ``len(blocks)`` × ``len(events)`` grid.
    """
    fig, axes = plt.subplots(
        len(blocks), len(events), figsize=(4 * len(events), 4 * len(blocks)),
        squeeze=False)
    for row, block in enumerate(blocks):
        for col, event in enumerate(events):
            ax = axes[row][col]
            panel = df[(df['block'] == block) & (df['event'] == event)]
            point_colors = [TARGETNM_COLORS.get(t, 'gray')
                            for t in panel['target_NM']]
            ax.scatter(panel['behavioral_dispersion'],
                       panel['neural_dispersion'],
                       c=point_colors, edgecolors='white', s=40)
            if row == len(blocks) - 1:
                ax.set_xlabel('Behavioral dispersion')
            if col == 0:
                ax.set_ylabel(f'{block}\nNeural dispersion')
            if row == 0:
                ax.set_title(event.replace('_times', ''))
    fig.tight_layout()
    return fig


def _add_bar_labels(ax, positions, values, hemisphere_counts=None, color='white',
                    horizontal=False, bottoms=None):
    """Add text labels to bars with optional L/R breakdown.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    positions : sequence of float
        Bar centers along the category axis.
    values : sequence of float
        Bar lengths along the count axis.
    hemisphere_counts : sequence of (int, int), optional
        Per-bar ``(n_left, n_right)``, appended to the count label.
    color : str
    horizontal : bool
    bottoms : sequence of float, optional
        Where each bar starts along the count axis, for stacked segments. The
        label is centered on the segment rather than on the whole bar. Defaults
        to zero, i.e. unstacked bars.
    """
    if bottoms is None:
        bottoms = np.zeros(len(values))
    for i, (pos, n, bottom) in enumerate(zip(positions, values, bottoms)):
        if n > 0:
            if hemisphere_counts is not None:
                n_left, n_right = hemisphere_counts[i]
                label = f'{int(n)}\n{n_left}L/{n_right}R'
            else:
                label = str(int(n))
            x, y = (bottom + n / 2, pos) if horizontal else (pos, bottom + n / 2)
            ax.text(x, y, label, ha='center', va='center', fontweight='bold',
                    color=color, rotation=-90 if horizontal else 90)


def _furthest_group(df_target, color_by, categories, min_sessions):
    """Map each mouse to the furthest group it reaches ``min_sessions`` in.

    Parameters
    ----------
    df_target : pandas.DataFrame
        Recordings of one target, with ``subject``, ``eid`` and ``color_by``.
    color_by : str
        Column holding the group label.
    categories : list of str
        Group labels, ordered low to high. A mouse takes the last of these it
        reaches the threshold in; labels outside the list are ignored.
    min_sessions : int
        Distinct sessions (``eid``) a mouse needs in a group to reach it.

    Returns
    -------
    pandas.Series
        Group label indexed by subject. Mice below the threshold in every group
        are absent, so they are counted nowhere.
    """
    n_sessions = df_target.groupby(['subject', color_by])['eid'].nunique()
    reached = n_sessions[n_sessions >= min_sessions].reset_index()
    reached = reached[reached[color_by].isin(categories)]
    reached = reached.sort_values(color_by, key=lambda s: s.map(categories.index))
    return reached.groupby('subject')[color_by].last()


def mouse_overview_barplot(df_sessions, min_sessions, ax=None, barwidth=0.8,
                           color_by='session_group', split_color_map=None,
                           horizontal=False):
    """Stacked bar plot of mouse counts per target region.

    One bar per target, stacking the groups of ``color_by`` bottom to top in
    the map's key order. Each mouse is counted exactly once, in the furthest
    group where it has at least ``min_sessions`` sessions, so a target's
    segments sum to the mice that reach the threshold anywhere.

    Parameters
    ----------
    df_sessions : pandas.DataFrame
        One row per recording, with ``subject``, ``eid``, ``target_NM``, the
        ``color_by`` column and optionally ``hemisphere``.
    min_sessions : int
        Sessions a mouse needs in a group to be counted in it.
    ax : matplotlib.axes.Axes, optional
    barwidth : float
    color_by : str
        Column holding the session group label.
    split_color_map : dict, optional
        Maps group label to fill color, in stacking and ranking order. Defaults
        to the SESSION_GROUPS colors.
    horizontal : bool
        If True, draw horizontal bars.
    """
    _color_map = split_color_map or _session_group_colors()

    if ax is None:
        fig, ax = plt.subplots()

    if len(df_sessions) == 0:
        ax.set_title("No data to plot")
        return ax

    target_nms = sorted(df_sessions['target_NM'].unique(),
                        key=lambda x: TARGETNM2POSITION.get(x, 999))
    xpos = np.arange(len(target_nms))

    has_hemisphere = 'hemisphere' in df_sessions.columns

    categories = [c for c in _color_map if c in df_sessions[color_by].values]
    by_target = {
        target_nm: _furthest_group(df_sessions[df_sessions['target_NM'] == target_nm],
                                   color_by, categories, min_sessions)
        for target_nm in target_nms
    }

    cumulative = np.zeros(len(target_nms))
    for category in categories:
        counts = []
        hemi_counts = [] if has_hemisphere else None
        for target_nm in target_nms:
            furthest = by_target[target_nm]
            mice = furthest[furthest == category].index
            counts.append(len(mice))
            if has_hemisphere:
                target_df = df_sessions[df_sessions['target_NM'] == target_nm]
                segment_df = target_df[target_df['subject'].isin(mice)]
                n_l = segment_df[segment_df['hemisphere'] == 'l']['subject'].nunique()
                n_r = segment_df[segment_df['hemisphere'] == 'r']['subject'].nunique()
                hemi_counts.append((n_l, n_r))
        if horizontal:
            ax.barh(xpos, counts, barwidth, left=cumulative,
                    color=_color_map[category], label=category)
        else:
            ax.bar(xpos, counts, barwidth, bottom=cumulative,
                   color=_color_map[category], label=category)
        _add_bar_labels(ax, xpos, counts, hemi_counts, horizontal=horizontal,
                        bottoms=cumulative)
        cumulative += counts

    if horizontal:
        ax.set_yticks(xpos)
        ax.set_yticklabels(target_nms)
        ax.invert_yaxis()
        ax.set_xlabel('N Mice')
        ax.set_ylabel('Target-NM')
    else:
        ax.set_xticks(xpos)
        ax.set_xticklabels(target_nms)
        ax.tick_params(axis='x', rotation=90)
        ax.set_ylabel('N Mice')
        ax.set_xlabel('Target-NM')
    ax.legend()
    ax.set_title(f'Mouse training progress by target (≥{min_sessions} sessions)')

    return ax


def qc_grid(df, qc_columns=None, qcval2num=None, ax=None, yticklabels='eid',
            legend=True):
    # Get fresh QCVAL2NUM from config to avoid stale imports
    if qcval2num is None:
        from iblnm.config import QCVAL2NUM
        qcval2num = QCVAL2NUM

    # Ensure qc_columns is a list
    if qc_columns is None:
        qc_columns = list(df.columns)
    else:
        qc_columns = list(qc_columns)

    # Extract and convert QC values to numeric
    df_qc = df[qc_columns].copy()
    for col in qc_columns:
        df_qc[col] = df_qc[col].map(lambda x: qcval2num.get(x, x))
    df_qc = df_qc.astype(float)

    # Create figure if needed
    n_rows, n_cols = len(df_qc), len(qc_columns)
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6, n_cols * 0.5), max(4, n_rows * 0.15)))

    # Plot the matrix
    qcmat = df_qc.values
    ax.matshow(qcmat, cmap=QCCMAP, vmin=0, vmax=1, aspect='auto')

    # Set ticks and labels
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(qc_columns)
    ax.tick_params(axis='x', rotation=90, labelsize=8)
    ax.tick_params(axis='y', labelsize=7)

    # Row labels: default to truncated eid
    if yticklabels == 'eid' and 'eid' in df.columns:
        ax.set_yticklabels(df['eid'].str[:6])
    elif isinstance(yticklabels, str) and yticklabels in df.columns:
        ax.set_yticklabels(df[yticklabels])
    elif isinstance(yticklabels, list):
        labels = df.apply(lambda x: '_'.join(x[yticklabels].astype(str)), axis='columns')
        ax.set_yticklabels(labels)

    # Draw gridlines at cell boundaries
    for i in range(n_cols + 1):
        ax.axvline(i - 0.5, color='white', linewidth=0.5)
    for i in range(n_rows + 1):
        ax.axhline(i - 0.5, color='white', linewidth=0.5)

    # Set axis limits
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)

    # Add legend
    if legend:
        for key, val in qcval2num.items():
            ax.scatter([], [], color=QCCMAP(val), label=key)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Adjust layout to make room for rotated column labels at top
    ax.figure.subplots_adjust(top=0.7, right=0.85)

    return ax


def session_plot(series, pipeline=[], t0=60, t1=120):
    fig = plt.figure(figsize=(12, 6))
    fig.suptitle(f"{series['subject']} - session {series['session_n']} - {series['task_protocol'].split('_')[-1]}, fiber target: {series['brain_region']}")

    psth_axes = [
        fig.add_axes([0.03, 0., 0.3, 0.3]),
        fig.add_axes([0.36, 0., 0.3, 0.3]),
        fig.add_axes([0.69, 0., 0.3, 0.3])
    ]
    signal_axes = [
        fig.add_axes([0.03, 0.7, 0.96, 0.25]),
        fig.add_axes([0.03, 0.4, 0.86, 0.25])
    ]

    from iblphotometry import pipelines as pipe, processing as proc
    tpts = series['GCaMP'].index.values
    signal_raw = series['GCaMP'][series['ROI']].values
    processed = pipe.run_pipeline(pipeline, series['GCaMP'])
    signal_processed = processed[series['ROI']].values

    signal_axes[0].plot(tpts, proc.z(signal_raw), alpha=0.5, color='gray', label='Raw')
    signal_axes[0].plot(tpts, proc.z(signal_processed), alpha=0.5, color='black', label='Processed')
    signal_axes[0].set_xlim([tpts.min(), tpts.max()])
    signal_axes[0].set_xlabel('Time (s)')
    y_min = min(proc.z(signal_raw).min(), proc.z(signal_processed).min())
    y_max = max(proc.z(signal_raw).max(), proc.z(signal_processed).max())
    signal_axes[0].set_ylim([y_min, y_max])
    signal_axes[0].set_ylabel('Signal (z-score)')
    signal_axes[0].legend(loc='upper left', bbox_to_anchor=[.9, -0.2])

    i0, i1 = tpts.searchsorted([t0, t1])
    signal_axes[1].plot(tpts[i0:i1], proc.z(signal_processed)[i0:i1], color='black', label='Processed')
    signal_axes[1].set_xlim([t0, t1])
    signal_axes[1].set_xlabel('Time (s)')
    y_min = proc.z(signal_processed)[i0:i1].min()
    y_max = proc.z(signal_processed)[i0:i1].max()
    signal_axes[1].set_ylim([y_min, y_max])
    signal_axes[1].set_ylabel('Signal (z-score)')

    events_dict = {'cue': psth_axes[0], 'movement': psth_axes[1], 'reward': psth_axes[2], 'omission': psth_axes[2]}
    colors = ['blue', 'orange', 'green', 'red']
    for event, color in zip(['cue', 'reward', 'omission'], ['blue', 'green', 'red']):
        for t in series[f'{event}_times']:
            if (t < t0) or (t > t1):
                continue
            signal_axes[1].axvline(t, color=color)
    from scipy import stats
    from iblnm.analysis import get_responses as psth
    y_max = []
    for (event, ax), color in zip(events_dict.items(), colors):
        responses, tpts = psth(series['GCaMP'], series[f'{event}_times'])
        y_max.append(np.abs(responses.mean(axis=0)).max())
        ax.plot(tpts, responses.mean(axis=0), color=color, label=event)
        ax.plot(tpts, responses.mean(axis=0) - stats.sem(responses, axis=0), ls='--', color=color)
        ax.plot(tpts, responses.mean(axis=0) + stats.sem(responses, axis=0), ls='--', color=color)
        ax.axhline(0, ls='--', color='gray', alpha=0.5)
        ax.set_xlabel('Time (s)')
        ax.legend(loc='upper right')
    for ax in psth_axes:
        ax.set_ylim([-1 * max(y_max), max(y_max)])
        ax.set_yticks([])
    psth_axes[0].set_yticks([-1 * max(y_max), 0, max(y_max)])
    psth_axes[0].ticklabel_format(axis='y', style='sci', scilimits=[-2, 2])
    psth_axes[0].set_ylabel('Response (a.u.)')

    return fig


def violinplot(
    ax, data, positions=None, log_transform=False, remove_outliers=True,
    show_outliers=True, outlier_threshold=1.5, colors=None, alpha=1, **violin_kwargs
):
    """
    Draw violin plots on the given axes with options for log transformation and
    outlier detection. Outliers are defined using the IQR method, and are
    plotted separately as scatter points.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes on which to draw the violins.
    data : list of array-like
        A list where each element is the data (as a 1D array) for one group.
    positions : array-like, optional
        x-axis positions for the violins. If None, defaults to [0, 1, 2, ...].
    log_transform : bool, default False
        If True, apply a natural log transform to each group's data before processing.
        (Your data must be strictly positive, or pre-shifted, when using this.)
    show_outliers : bool, default True
        If True, outliers (as determined by the IQR method) are plotted as separate scatter points.
    outlier_threshold : float, default 1.5
        The multiplier for the IQR to set the outlier boundary.
    colors : list of color, optional
        One color per group. When given, violin bodies are drawn as unfilled
        outlines in these colors and the medians are colored to match.
    alpha : float, default 1
        Opacity of the violin bodies only. Medians and scatter points keep full
        opacity, so a faint violin still shows a crisp median. Since the bodies
        are unfilled, this modulates the outline.
    violin_kwargs : dict
        Other keyword arguments to pass to ax.violinplot().

    Returns
    -------
    violins : matplotlib.collections.PolyCollection
        The object returned by ax.violinplot().
    """
    # Optionally transform each group via log. Ensure the input is a NumPy array.
    if log_transform:
        data = [np.log(x[x > 0]) for x in data]
    else:
        data = [np.array(x) for x in data]

    # If positions are not specified, use sequential positions.
    if positions is None:
        positions = np.arange(len(data))

    violin_data = [d for d in data if len(d) >= 10]
    violin_positions = [p for p, d in zip(positions, data) if len(d) >= 10]
    if colors is not None:
        violin_colors = [c for c, d in zip(colors, data) if len(d) >= 10]
    scatter_data = [d for d in data if len(d) < 10]
    scatter_positions = [p for p, d in zip(positions, data) if len(d) < 10]
    if colors is not None:
        scatter_colors = [c for c, d in zip(colors, data) if len(d) < 10]

    if remove_outliers:
        central_data = []  # Data without outliers, to be plotted in the violins.
        outlier_data = []  # Outlier values to scatter separately.
        # Process each group.
        for vd in violin_data:
            q1, q3 = np.percentile(vd, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - outlier_threshold * iqr
            upper_bound = q3 + outlier_threshold * iqr
            non_outliers = vd[(vd >= lower_bound) & (vd <= upper_bound)]
            outliers = vd[(vd < lower_bound) | (vd > upper_bound)]
            central_data.append(non_outliers)
            outlier_data.append(outliers)
    else:
        central_data = violin_data

    for p, c, sd in zip(scatter_positions, scatter_colors, scatter_data):
        if len(sd) > 0:
            ax.scatter(np.full(sd.shape, p), sd, s=5, fc='none', ec=c)

    # Create the violin plot using only the non-outlier data.
    violins = ax.violinplot(
        central_data,
        violin_positions,
        showmedians=True, showextrema=False, **violin_kwargs
    )
    if colors is not None:
        for pc, color in zip(violins['bodies'], violin_colors):
            pc.set_facecolor('none')
            pc.set_edgecolor(color)
            pc.set_linewidth(1)
            pc.set_alpha(alpha)
        violins['cmedians'].set_color(violin_colors)

    # Optionally, scatter the outlier points.
    if show_outliers:
        ocolors = ['black' for _ in violin_data] if colors is None else violin_colors
        for xpos, color, outliers in zip(violin_positions, ocolors, outlier_data):
            if len(outliers) > 0:
                ax.scatter(np.full(outliers.shape, xpos), outliers, s=10, fc='none', ec=color)

    return violins


def plot_joint_distributions(df, metrics=None, transform=True, bins=30, figsize=(5, 5)):
    """
    Plots joint distributions for each pair of metrics in the upper triangle of a grid.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the metrics as columns.
    metrics : list
        List of column names in `df` representing the metrics.
    figsize : tuple, optional
        Size of the figure. Default is (10, 10).
    bins : int, optional
        Number of bins for the histograms. Default is 30.

    Returns
    -------
    None
    """
    if metrics is None:
        metrics = df.columns
    n_metrics = len(metrics)

    # Replace inf with NaN (can occur in ratio metrics like percentile_asymmetry when denominator is 0)
    df_clean = df[metrics].replace([np.inf, -np.inf], np.nan).dropna()
    X = df_clean.values
    corr = df_clean.corr(method='spearman')
    if transform:
        X = quantile_transform(X, output_distribution='normal', n_quantiles=500)

    fig, axs = plt.subplots(n_metrics, n_metrics, figsize=figsize)

    for i in range(n_metrics):
        for j in range(n_metrics):
            ax = axs[i, j]

            if i < j:  # Upper triangle: plot joint distributions
                x = X[:, j]
                y = X[:, i]
                ax.hist2d(x, y, bins=bins, cmap='YlOrBr', density=True, norm='log')
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)
                ax.set_xlabel(metrics[j] if i == n_metrics - 1 else "")
                ax.set_ylabel(metrics[i] if j == 0 else "")
                ax.text(0.01, 0.8, r'$\rho$' + f'={corr.iloc[i, j]:.2f}', fontsize=TICKFONTSIZE, transform=ax.transAxes)
            elif i == j:  # Diagonal: plot histograms
                data = X[:, i]
                ax.hist(data, bins=bins, color='gray', alpha=0.7)
                ax.set_xticks([])
                ax.set_yticks([])
                lbl = ax.set_ylabel(metrics[i], rotation=25)
                lbl.set_horizontalalignment('right')
                lbl.set_verticalalignment('center')
            else:  # Lower triangle: turn off axes
                ax.axis("off")
    return fig, X


# =============================================================================
# Task Performance Visualization Functions
# =============================================================================

def plot_stage_barplot(
    df_stage_counts: pd.DataFrame,
    ax: plt.Axes = None
) -> plt.Axes:
    """
    Barplot of mice that reached each stage.

    Parameters
    ----------
    df_stage_counts : pd.DataFrame
        DataFrame from count_sessions_to_stage with columns:
        subject, target_NM, n_training, n_biased, n_ephys
    target_nm : str, optional
        Filter to specific target-NM. If None, use all.
    ax : plt.Axes, optional
        Axes to plot on.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))

    df = df_stage_counts.copy()

    # Count mice at each stage
    n_training = (df['n_training'] > 0).sum()
    n_biased = (df['n_biased'] > 0).sum()
    n_ephys = (df['n_ephys'] > 0).sum()

    stages = ['Training', 'Biased', 'Ephys']
    counts = [n_training, n_biased, n_ephys]
    colors = [SESSIONTYPE2COLOR.get('training', 'cornflowerblue'),
              SESSIONTYPE2COLOR.get('biased', 'mediumpurple'),
              SESSIONTYPE2COLOR.get('ephys', 'hotpink')]

    bars = ax.bar(stages, counts, color=colors)

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom', fontweight='bold')

    ax.set_ylabel('N mice')

    return ax


def plot_sessions_to_stage_cdf(
    df_stage_counts: pd.DataFrame,
    stage: str,
    ax: plt.Axes = None,
    color=None
) -> plt.Axes:
    """
    CDF of sessions to reach stage.

    Parameters
    ----------
    df_stage_counts : pd.DataFrame
        DataFrame from count_sessions_to_stage.
    stage : str
        Either 'biased' or 'ephys'.
    ax : plt.Axes, optional
        Axes to plot on.
    color : optional
        Line color.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))

    if stage == 'biased':
        col = 'sessions_to_biased'
        xlabel = 'Training sessions to biased'
    elif stage == 'ephys':
        col = 'biased_sessions_to_ephys'
        xlabel = 'Biased sessions to ephys'
    else:
        raise ValueError(f"stage must be 'biased' or 'ephys', got {stage}")

    # Get values (excluding NaN = mice that didn't reach stage)
    values = df_stage_counts[col].dropna().values

    if len(values) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return ax

    # Sort for CDF
    sorted_vals = np.sort(values)
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)

    # Plot
    if color is None:
        color = 'gray'

    ax.step(sorted_vals, cdf, where='post', color=color, linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Cumulative proportion')
    ax.set_ylim(0, 1.05)

    return ax


def plot_psychometric_parameter_trajectory(
    df_fits: pd.DataFrame,
    parameter: str,
    has_photometry_col: str = 'has_extracted_photometry_signal',
    ax: plt.Axes = None,
    color=None,
    show_mean: bool = True
) -> plt.Axes:
    """
    Plot trajectory of psychometric parameter across training sessions.

    One line per mouse. Thick lines for mice with photometry,
    thin lines for mice without. Optional mean across mice in black.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))

    df = df_fits.copy()

    if parameter not in df.columns:
        ax.text(0.5, 0.5, f'No {parameter} data', ha='center', va='center',
                transform=ax.transAxes)
        return ax

    # Get base color
    base_color = color if color is not None else 'gray'

    # Check if photometry column exists
    has_phot_col = has_photometry_col in df.columns

    # Plot each subject
    for subject, sub_df in df.groupby('subject'):
        sub_df = sub_df.sort_values('session_n')

        # Determine line thickness based on photometry
        if has_phot_col:
            # If any session has photometry, use thick line
            has_phot = sub_df[has_photometry_col].any()
        else:
            has_phot = False

        linewidth = 1.5 if has_phot else 0.5
        alpha = 0.7 if has_phot else 0.3

        ax.plot(sub_df['session_n'], sub_df[parameter],
                color=base_color, linewidth=linewidth, alpha=alpha)

    # Plot mean across mice
    if show_mean:
        mean_df = df.groupby('session_n')[parameter].mean().reset_index()
        ax.plot(mean_df['session_n'], mean_df[parameter],
                color='black', linewidth=2, alpha=1.0, zorder=10)

    ax.set_xlabel('Session number')
    ax.set_ylabel(parameter.replace('_', ' ').title())

    return ax


def plot_performance_trajectory(
    df: pd.DataFrame,
    metric: str = 'fraction_correct',
    has_photometry_col: str = 'has_extracted_photometry_signal',
    ax: plt.Axes = None,
    color=None,
    show_mean: bool = True
) -> plt.Axes:
    """
    Plot trajectory of performance metric across sessions.

    One line per mouse. Thick lines for mice with photometry,
    thin lines for mice without. Optional mean across mice in black.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))

    df = df.copy()

    if metric not in df.columns:
        ax.text(0.5, 0.5, f'No {metric} data', ha='center', va='center',
                transform=ax.transAxes)
        return ax

    # Get base color
    base_color = color if color is not None else 'gray'

    # Check if photometry column exists
    has_phot_col = has_photometry_col in df.columns

    # Plot each subject
    for subject, sub_df in df.groupby('subject'):
        sub_df = sub_df.sort_values('session_n')

        # Determine line thickness based on photometry
        if has_phot_col:
            has_phot = sub_df[has_photometry_col].any()
        else:
            has_phot = False

        linewidth = 1.5 if has_phot else 0.5
        alpha = 0.7 if has_phot else 0.3

        ax.plot(sub_df['session_n'], sub_df[metric],
                color=base_color, linewidth=linewidth, alpha=alpha)

    # Plot mean across mice
    if show_mean:
        mean_df = df.groupby('session_n')[metric].mean().reset_index()
        ax.plot(mean_df['session_n'], mean_df[metric],
                color='black', linewidth=2, alpha=1.0, zorder=10)

    ax.set_xlabel('Session number')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    return ax


def plot_psychometric_curves_50(
    df_fits: pd.DataFrame,
    target_nm: str = None,
    ax: plt.Axes = None,
    contrast_range: tuple = (-100, 100)
) -> plt.Axes:
    """
    Plot psychometric curves for 50-50 block.

    Shows thin line per session + thick grand mean.

    Parameters
    ----------
    df_fits : pd.DataFrame
        Dataframe with psychometric parameters (psych_50_bias, psych_50_threshold, etc.)
    target_nm : str, optional
        Filter to specific target-NM.
    ax : plt.Axes, optional
        Axes to plot on.
    contrast_range : tuple
        Range of contrasts to plot.

    Returns
    -------
    plt.Axes
    """
    import psychofit as psy

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))

    df = df_fits.copy()
    if target_nm is not None and 'target_NM' in df.columns:
        df = df[df['target_NM'] == target_nm]

    # Check for required columns
    required_cols = ['psych_50_bias', 'psych_50_threshold', 'psych_50_lapse_left', 'psych_50_lapse_right']
    if not all(col in df.columns for col in required_cols):
        ax.text(0.5, 0.5, 'No psychometric data', ha='center', va='center',
                transform=ax.transAxes)
        return ax

    # Filter rows with valid fits
    df = df.dropna(subset=required_cols)

    if len(df) == 0:
        ax.text(0.5, 0.5, 'No valid fits', ha='center', va='center',
                transform=ax.transAxes)
        return ax

    # Contrast values for plotting
    contrasts = np.linspace(contrast_range[0], contrast_range[1], 200)

    # Get color
    color = TARGETNM_COLORS.get(target_nm, 'gray') if target_nm else 'gray'

    # Plot individual sessions (thin lines)
    all_curves = []
    for _, row in df.iterrows():
        params = [row['psych_50_bias'], row['psych_50_threshold'],
                  row['psych_50_lapse_right'], row['psych_50_lapse_left']]  # Note: lapse order
        curve = psy.erf_psycho_2gammas(params, contrasts)
        all_curves.append(curve)
        ax.plot(contrasts, curve, color=color, alpha=0.2, linewidth=0.5)

    # Plot grand mean (thick line)
    mean_curve = np.mean(all_curves, axis=0)
    ax.plot(contrasts, mean_curve, color=color, linewidth=2.5, label='Mean')

    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Signed contrast (%)')
    ax.set_ylabel('P(choose right)')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f'{target_nm} (50-50 block)' if target_nm else '50-50 block')

    return ax


def plot_psychometric_curves_by_block(
    df_fits: pd.DataFrame,
    target_nm: str = None,
    ax: plt.Axes = None,
    contrast_range: tuple = (-100, 100)
) -> plt.Axes:
    """
    Plot psychometric grand mean curves by block type.

    Shows one curve per block type (20/50/80) - grand mean only.

    Parameters
    ----------
    df_fits : pd.DataFrame
        Dataframe with psychometric parameters for each block.
    target_nm : str, optional
        Filter to specific target-NM.
    ax : plt.Axes, optional
        Axes to plot on.
    contrast_range : tuple
        Range of contrasts to plot.

    Returns
    -------
    plt.Axes
    """
    import psychofit as psy

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))

    df = df_fits.copy()
    if target_nm is not None and 'target_NM' in df.columns:
        df = df[df['target_NM'] == target_nm]

    contrasts = np.linspace(contrast_range[0], contrast_range[1], 200)

    # Block colors
    block_colors = {'20': '#e74c3c', '50': '#2c3e50', '80': '#3498db'}
    block_labels = {'20': 'p(left)=0.2', '50': 'p(left)=0.5', '80': 'p(left)=0.8'}

    for block in ['20', '50', '80']:
        cols = [f'psych_{block}_bias', f'psych_{block}_threshold',
                f'psych_{block}_lapse_left', f'psych_{block}_lapse_right']

        if not all(col in df.columns for col in cols):
            continue

        block_df = df.dropna(subset=cols)
        if len(block_df) == 0:
            continue

        # Compute curves for all sessions
        all_curves = []
        for _, row in block_df.iterrows():
            params = [row[cols[0]], row[cols[1]], row[cols[3]], row[cols[2]]]  # bias, thresh, lapse_high, lapse_low
            curve = psy.erf_psycho_2gammas(params, contrasts)
            all_curves.append(curve)

        # Plot grand mean
        mean_curve = np.mean(all_curves, axis=0)
        ax.plot(contrasts, mean_curve, color=block_colors[block],
                linewidth=2, label=block_labels[block])

    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Signed contrast (%)')
    ax.set_ylabel('P(choose right)')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='lower right')
    ax.set_title(f'{target_nm}' if target_nm else 'All targets')

    return ax


def plot_psychometric_parameters_boxplot(
    df_fits: pd.DataFrame,
    parameter: str,
    target_nm: str = None,
    ax: plt.Axes = None
) -> plt.Axes:
    """
    Boxplot of psychometric parameter by block type.

    Parameters
    ----------
    df_fits : pd.DataFrame
        Dataframe with psychometric parameters.
    parameter : str
        Which parameter to plot ('bias', 'threshold', 'lapse_left', 'lapse_right').
    target_nm : str, optional
        Filter to specific target-NM.
    ax : plt.Axes, optional
        Axes to plot on.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))

    df = df_fits.copy()
    if target_nm is not None and 'target_NM' in df.columns:
        df = df[df['target_NM'] == target_nm]

    # Block colors
    block_colors = {'20': '#e74c3c', '50': '#2c3e50', '80': '#3498db'}

    data = []
    positions = []
    colors = []

    for i, block in enumerate(['20', '50', '80']):
        col = f'psych_{block}_{parameter}'
        if col not in df.columns:
            continue
        values = df[col].dropna().values
        if len(values) > 0:
            data.append(values)
            positions.append(i)
            colors.append(block_colors[block])

    if len(data) == 0:
        ax.text(0.5, 0.5, f'No {parameter} data', ha='center', va='center',
                transform=ax.transAxes)
        return ax

    bp = ax.boxplot(data, positions=positions, patch_artist=True, widths=0.6)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_xticks(range(3))
    ax.set_xticklabels(['p(L)=0.2', 'p(L)=0.5', 'p(L)=0.8'])
    ax.set_ylabel(parameter.replace('_', ' ').title())
    ax.set_title(f'{target_nm}' if target_nm else 'All targets')

    return ax


def plot_parameter_box(df, param_col, ax, color='black'):
    """Plot boxplot with session dots (left) and subject mean +/- SD (right).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``param_col`` and ``'subject'`` columns.
    param_col : str
        Column name for the parameter values.
    ax : plt.Axes
        Axes to draw on.
    color : str
        Color for the dots and error bars.
    """
    df = df[[param_col, 'subject']].dropna(subset=[param_col]).copy()
    values = df[param_col].values
    if len(values) == 0:
        return ax

    # Box at center
    bp = ax.boxplot(
        [values], positions=[0], widths=0.35, patch_artist=True,
        showfliers=False, zorder=2,
    )
    for element in ('boxes', 'whiskers', 'caps', 'medians'):
        plt.setp(bp[element], color='black', linewidth=1.2)
    bp['boxes'][0].set_facecolor('none')

    # Session dots offset left
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.04, 0.04, size=len(values))
    ax.scatter(
        np.full(len(values), -0.22) + jitter, values,
        s=10, alpha=0.3, color=color, edgecolors='none', zorder=1,
    )

    # Subject mean ± SD offset right, spaced by subject index
    subj = df.groupby('subject')[param_col]
    means = subj.mean()
    sds = subj.std()
    n_subj = len(means)
    x_subj = np.linspace(0.18, 0.45, n_subj) if n_subj > 1 else np.array([0.3])
    ax.errorbar(
        x_subj, means, yerr=sds,
        fmt='o', ms=4, color=color, alpha=0.6, elinewidth=0.8,
        capsize=0, zorder=3,
    )

    ax.set_xticks([])
    ax.set_xlim(-0.5, 0.7)
    return ax


def plot_bias_shift_trajectory(
    df_fits: pd.DataFrame,
    target_nm: str = None,
    ax: plt.Axes = None
) -> plt.Axes:
    """
    Plot bias shift trajectories across biased sessions per mouse.

    Parameters
    ----------
    df_fits : pd.DataFrame
        Dataframe with bias_shift column and session info.
    target_nm : str, optional
        Filter to specific target-NM.
    ax : plt.Axes, optional
        Axes to plot on.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))

    df = df_fits.copy()
    if target_nm is not None and 'target_NM' in df.columns:
        df = df[df['target_NM'] == target_nm]

    if 'bias_shift' not in df.columns:
        ax.text(0.5, 0.5, 'No bias shift data', ha='center', va='center',
                transform=ax.transAxes)
        return ax

    # Filter to biased sessions with valid bias shift
    df = df[df['bias_shift'].notna()]

    if len(df) == 0:
        ax.text(0.5, 0.5, 'No valid bias shifts', ha='center', va='center',
                transform=ax.transAxes)
        return ax

    # Get color
    color = TARGETNM_COLORS.get(target_nm, 'gray') if target_nm else 'gray'

    # Plot each subject
    for subject, sub_df in df.groupby('subject'):
        sub_df = sub_df.sort_values('session_n')

        # Create biased session index (1, 2, 3, ...)
        sub_df = sub_df.reset_index(drop=True)
        sub_df['biased_session_idx'] = sub_df.index + 1

        ax.plot(sub_df['biased_session_idx'], sub_df['bias_shift'],
                color=color, alpha=0.7, linewidth=1, marker='o', markersize=3)

    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Biased session number')
    ax.set_ylabel('Bias shift (80-20)')
    ax.set_title(f'{target_nm}' if target_nm else 'All targets')

    return ax


def create_psychometric_figure(
    df_fits: pd.DataFrame,
    target_nms: list = None
) -> plt.Figure:
    """
    Create complete psychometric analysis figure.

    Layout: target-NM as columns, rows:
    1. Psychometric curves for 50-50 block (thin per session + mean)
    2. Psychometric curves by block (grand mean only)
    3. Boxplots of bias by block
    4. Boxplots of threshold by block
    5. Boxplots of lapse_left by block
    6. Boxplots of lapse_right by block
    7. Bias shift trajectory

    Parameters
    ----------
    df_fits : pd.DataFrame
        Psychometric fits for biased/ephys sessions with target_NM column.
    target_nms : list, optional
        List of target-NMs to include.

    Returns
    -------
    plt.Figure
    """
    from iblnm.config import VALID_TARGETS

    if target_nms is None:
        target_nms = [t for t in VALID_TARGETS if 'target_NM' in df_fits.columns and t in df_fits['target_NM'].values]

    if len(target_nms) == 0:
        target_nms = [None]  # Plot all data together

    n_cols = len(target_nms)
    n_rows = 7

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 2.5 * n_rows))
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    for col, target_nm in enumerate(target_nms):
        # Row 0: Psychometric curves 50-50
        plot_psychometric_curves_50(df_fits, target_nm=target_nm, ax=axes[0, col])

        # Row 1: Psychometric curves by block
        plot_psychometric_curves_by_block(df_fits, target_nm=target_nm, ax=axes[1, col])

        # Rows 2-5: Parameter boxplots
        for row, param in enumerate(['bias', 'threshold', 'lapse_left', 'lapse_right'], start=2):
            plot_psychometric_parameters_boxplot(df_fits, param, target_nm=target_nm, ax=axes[row, col])

        # Row 6: Bias shift trajectory
        plot_bias_shift_trajectory(df_fits, target_nm=target_nm, ax=axes[6, col])

    plt.tight_layout()
    return fig


# Masking diagnostic column → (panel y-label, y-limit).
_MASKING_PANELS = {
    'masked_fraction_mean': ('Window masked', (0, 1)),
    'pct_any_masked': ('% trials any masked', (0, 100)),
    'pct_fully_masked': ('% trials fully masked', (0, 100)),
    'pct_move_in_window': ('% trials moving in window', (0, 100)),
}


def plot_masking_diagnostics(diagnostics, target_nm, event, fig=None):
    """Plot one cohort-event's masking statistics against contrast.

    A pure drawer over the cell frame ``compute_masking_diagnostics`` writes:
    one panel per statistic, contrast on the x-axis, correct and incorrect
    trials as separate lines. Read together with any contrast-dependent
    result, because a statistic rising with contrast means the high-contrast
    responses were averaged over less of the window than the low-contrast
    ones.

    Parameters
    ----------
    diagnostics : pd.DataFrame
        The rows of a single (target_NM x event) cohort-event, in
        ``config.MASKING_DIAGNOSTIC_COLUMNS`` shape.
    target_nm : str
        Target neuromodulator label; used for the title and color lookup.
    event : str
        Raw event name (e.g. 'stimOnTrigger_times'); used for the title.
    fig : plt.Figure or None
        Figure with four existing axes to draw on. If None, a new one-row
        figure is created.

    Returns
    -------
    plt.Figure
    """
    if fig is None:
        fig, _ = plt.subplots(1, len(_MASKING_PANELS), figsize=(12, 3),
                              layout='constrained')

    color = TARGETNM_COLORS.get(target_nm, 'black')
    n_trials = int(diagnostics['n_trials'].sum())
    fig.suptitle(f"{target_nm} — {event.replace('_times', '')}"
                 f"\n{n_trials} trials", fontsize=LABELFONTSIZE)

    contrasts = sorted(diagnostics['contrast'].unique())
    ranks = list(range(len(contrasts)))

    for ax, (column, (label, ylim)) in zip(fig.axes, _MASKING_PANELS.items()):
        for feedback, ls in ((1, '-'), (-1, '--')):
            by_contrast = (
                diagnostics[diagnostics['feedbackType'] == feedback]
                .set_index('contrast').reindex(contrasts))
            ax.plot(ranks, by_contrast[column].values, marker='o', color=color,
                    linestyle=ls,
                    label='correct' if feedback == 1 else 'incorrect')
        ax.set_xticks(ranks)
        ax.set_xticklabels([f'{contrast:g}' for contrast in contrasts])
        ax.set_xlabel('Contrast level')
        ax.set_ylabel(label, fontsize=TICKFONTSIZE)
        ax.set_ylim(*ylim)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.axes[-1].legend(frameon=False, loc='upper left',
                        bbox_to_anchor=(1, 1), fontsize=TICKFONTSIZE)
    return fig


def plot_relative_contrast(agg_df, target_nm, event, fig=None,
                           window_label=None, count_label=None):
    """Plot aggregated response magnitude by contrast, contra and ipsi panels.

    A pure drawer: the means and SEMs are drawn exactly as given, with no
    aggregation, correction or trial selection of its own. Produce ``agg_df``
    with :func:`iblnm.analysis.aggregate_conditions`.

    The contra panel x-axis is inverted so that the highest contrast is on the
    far left; the ipsi panel runs normally left-to-right. Together they read:
    ``100 ← contra % → 0 | 0 ← ipsi % → 100``.

    Parameters
    ----------
    agg_df : pd.DataFrame
        One row per condition of a single (target_NM × event) group, in the
        ``aggregate_conditions`` output shape: group keys ``side``
        ('contra' / 'ipsi'), ``contrast`` (absolute) and ``feedbackType``,
        plus ``mean``, ``sem`` and ``n``.
    target_nm : str
        Target neuromodulator label; used for the title and color lookup.
    event : str
        Raw event name (e.g. 'stimOn_times'); used for the title.
    fig : plt.Figure or None
        Figure with two existing axes to draw on. If None, a new figure is
        created with ``plt.subplots(1, 2, sharey=True)``.
    window_label : str or None
        Label for the response window (e.g. 'early', 'late').
    count_label : str or None
        Second title line describing the data the aggregate was taken over
        (e.g. '42 sessions, 9 subjects'); the caller holds those counts.

    Returns
    -------
    plt.Figure
    """
    if fig is None:
        fig, _ = plt.subplots(1, 2, sharey=True, gridspec_kw={'wspace': 0.05},
                              layout='constrained')

    ax_c, ax_i = fig.axes[0], fig.axes[1]

    event_label = event.replace('_times', '')
    _window = window_label or ''
    color = TARGETNM_COLORS.get(target_nm, 'black')
    fig.suptitle(
        f'{target_nm} — {event_label} ({_window})\n{count_label or ""}',
        fontsize=LABELFONTSIZE,
    )

    # Contrasts from the whole frame so both panels share the same x-axis
    contrasts = sorted(agg_df['contrast'].unique()) if len(agg_df) > 0 else []
    ranks = list(range(len(contrasts)))

    for ax, side in ((ax_c, 'contra'), (ax_i, 'ipsi')):
        df_side = agg_df[agg_df['side'] == side]

        for feedback, ls in ((1, '-'), (-1, '--')):
            df_fb = df_side[df_side['feedbackType'] == feedback]
            if len(df_fb) == 0:
                continue

            by_contrast = df_fb.set_index('contrast').reindex(contrasts)
            label = 'correct' if feedback == 1 else 'incorrect'
            ax.errorbar(ranks, by_contrast['mean'].values,
                        yerr=by_contrast['sem'].values.astype(float),
                        marker='o', color=color, linestyle=ls, label=label)

        ax.set_xticks(ranks)
        ax.set_xticklabels([f'{c:g}' for c in contrasts])
        ax.set_xlabel('Contrast level')
        ax.set_yticks([-1, 0, 1, 2])
        ax.axhline(0, ls='--', color='gray', lw=0.5)

    # Invert contra axis so highest contrast is on the far left
    ax_c.invert_xaxis()

    ax_c.text(0.05, 0.02, 'Contra', ha='left', transform=ax_c.transAxes,
              fontsize=TICKFONTSIZE)
    ax_i.text(0.95, 0.02, 'Ipsi', ha='right', transform=ax_i.transAxes,
              fontsize=TICKFONTSIZE)

    ax_c.set_ylabel(r'$\Delta$ activity (z-score)')
    ax_i.tick_params(left=False)
    ax_i.spines['left'].set_visible(False)
    for ax in (ax_c, ax_i):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    ax_i.legend(frameon=False, loc='upper left', bbox_to_anchor=(1, 1),
                fontsize=TICKFONTSIZE)

    ax_i.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    return fig


def plot_confusion_matrix(confusion, fig=None):
    """Plot a confusion matrix as an annotated heatmap.

    Parameters
    ----------
    confusion : pd.DataFrame
        Square confusion matrix (rows = true, columns = predicted).
    fig : plt.Figure, optional

    Returns
    -------
    plt.Figure
    """
    if fig is None:
        n = len(confusion)
        fig, ax = plt.subplots(1, 1, figsize=(max(4, n * 1.2), max(3.5, n * 1.0)))
    else:
        ax = fig.axes[0]

    im = ax.imshow(confusion.values, cmap='Blues', aspect='equal')
    fig.colorbar(im, ax=ax, label='Count')

    # Annotate cells
    for i in range(len(confusion)):
        for j in range(len(confusion.columns)):
            val = confusion.iloc[i, j]
            ax.text(j, i, str(int(val)), ha='center', va='center',
                    color='white' if val > confusion.values.max() / 2 else 'black')

    ax.set_xticks(range(len(confusion.columns)))
    ax.set_xticklabels(confusion.columns)
    ax.set_yticks(range(len(confusion.index)))
    ax.set_yticklabels(confusion.index)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

    total = confusion.values.sum()
    correct = np.trace(confusion.values)
    accuracy = correct / total if total > 0 else 0
    ax.set_title(f'Confusion matrix (accuracy: {accuracy:.0%})')

    fig.tight_layout()
    return fig


def plot_similarity_matrix(sim_matrix, labels, subjects=None, fig=None):
    """Plot a cosine similarity matrix as a heatmap, sorted by target-NM label.

    Parameters
    ----------
    sim_matrix : pd.DataFrame
        Symmetric similarity matrix from ``cosine_similarity_matrix``.
    labels : pd.Series
        Target-NM label per recording, aligned to sim_matrix index.
    subjects : pd.Series, optional
        Subject per recording. If provided, recordings are sorted by subject
        within each target-NM group.
    fig : plt.Figure, optional
        Existing figure to draw on.

    Returns
    -------
    plt.Figure
    """
    # Sort: primary by target-NM label, secondary by subject if provided
    sort_df = pd.DataFrame({'label': labels})
    if subjects is not None:
        sort_df['subject'] = subjects
        sort_df = sort_df.sort_values(['label', 'subject'])
    else:
        sort_df = sort_df.sort_values('label')
    order = sort_df.index
    sim_sorted = sim_matrix.loc[order, order]

    if fig is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    else:
        ax = fig.axes[0]

    im = ax.imshow(sim_sorted.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    fig.colorbar(im, ax=ax, label='Cosine similarity')

    # Draw group boundaries and label each target
    sorted_labels = labels.loc[order]
    boundaries = np.where(sorted_labels.values[:-1] != sorted_labels.values[1:])[0] + 0.5
    for b in boundaries:
        ax.axhline(b, color='k', linewidth=1.5)
        ax.axvline(b, color='k', linewidth=1.5)

    # Compute group midpoints for tick labels
    group_edges = np.concatenate([[-0.5], boundaries, [len(sorted_labels) - 0.5]])
    group_mids = [(group_edges[i] + group_edges[i + 1]) / 2
                  for i in range(len(group_edges) - 1)]
    unique_labels = sorted_labels.values[
        np.concatenate([[0], (boundaries + 0.5).astype(int)])
    ]

    ax.set_yticks(group_mids)
    ax.set_yticklabels(unique_labels)
    ax.set_xticks(group_mids)
    ax.set_xticklabels(unique_labels, rotation=45, ha='right')

    ax.set_title('Response vector similarity')
    fig.tight_layout()
    return fig


def plot_decoding_coefficients(coefficients, fig=None):
    """Plot L1 logistic regression coefficients as a heatmap.

    Parameters
    ----------
    coefficients : pd.DataFrame
        Shape (n_classes, n_features) from ``decode_target_nm``.
    fig : plt.Figure, optional

    Returns
    -------
    plt.Figure
    """
    if fig is None:
        n_classes, n_features = coefficients.shape
        fig, ax = plt.subplots(1, 1, figsize=(max(8, n_features * 0.3), max(3, n_classes * 0.8)))
    else:
        ax = fig.axes[0]

    vmax = np.abs(coefficients.values).max()
    im = ax.imshow(coefficients.values, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    fig.colorbar(im, ax=ax, label='Coefficient')

    ax.set_yticks(range(len(coefficients.index)))
    ax.set_yticklabels(coefficients.index)
    ax.set_xticks(range(len(coefficients.columns)))
    ax.set_xticklabels(coefficients.columns, rotation=90, fontsize=TICKFONTSIZE)
    ax.set_title('Decoding coefficients (L1 logistic)')
    fig.tight_layout()
    return fig


def plot_feature_contributions(contributions, fig=None):
    """Horizontal bar plot of each feature's unique contribution to decoding.

    Parameters
    ----------
    contributions : pd.DataFrame
        Columns: feature, full_accuracy, reduced_accuracy, delta.
    fig : plt.Figure, optional

    Returns
    -------
    plt.Figure
    """
    df = contributions.sort_values('delta', ascending=True)  # bottom-to-top

    if fig is None:
        n = len(df)
        fig, ax = plt.subplots(1, 1, figsize=(6, max(3, n * 0.3)))
    else:
        ax = fig.axes[0]

    ax.barh(range(len(df)), df['delta'].values)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['feature'].values, fontsize=TICKFONTSIZE)
    ax.set_xlabel(r'$\Delta$ accuracy')
    ax.set_title('Feature unique contribution')
    ax.axvline(0, color='k', linewidth=0.5)
    fig.tight_layout()
    return fig


_SIDE_ORDER = {'contra': 0, 'ipsi': 1}
# Feature labels carry the event with its `_times` suffix stripped (see
# `PhotometrySessionGroup.get_response_vector`), so the onset key follows
# whichever column `STIM_ONSET_EVENT` names.
_EVENT_ORDER = {STIM_ONSET_EVENT.replace('_times', ''): 0,
                'firstMovement': 1, 'feedback': 2}
_FB_ORDER = {'correct': 0, 'incorrect': 1}

def _sort_events(events: Iterable[str]) -> list[str]:
    """Sort ``_times`` event names into trial chronology (``RESPONSE_EVENTS``);
    unknowns sort last by name."""
    order = {e: i for i, e in enumerate(RESPONSE_EVENTS)}
    return sorted(events, key=lambda e: (order.get(e, len(order)), e))


def _scatter_folds(ax, x, df_group, color, value_col='delta_r2'):
    """Plot one group at x-position ``x``: per-fold ``value_col`` as small faint
    markers and the across-fold aggregate as a large black-edged marker."""
    folds = df_group[df_group['fold'] != 'aggregate']
    ax.scatter(np.full(len(folds), x), folds[value_col], color=color, s=20,
               alpha=0.5, zorder=2)
    agg = df_group[df_group['fold'] == 'aggregate']
    if len(agg):
        ax.scatter(x, agg[value_col].iloc[0], color=color, s=90,
                   edgecolor='k', zorder=3)


_FEATURE_RE = re.compile(
    r'^(?P<event>[a-zA-Z]+)_c(?P<contrast>[\d.]+)_(?P<side>contra|ipsi)_(?P<fb>correct|incorrect)$'
)


def feature_sort_key(label):
    """Sort key for feature labels: side > event > feedback > contrast.

    Labels follow the format ``{event}_c{contrast}_{side}_{feedback}``.
    Unparseable labels sort after all valid ones.
    """
    m = _FEATURE_RE.match(label)
    if m is None:
        return (999, 999, 999, 999, label)
    return (
        _SIDE_ORDER.get(m['side'], 99),
        _EVENT_ORDER.get(m['event'], 99),
        _FB_ORDER.get(m['fb'], 99),
        float(m['contrast']),
        label,
    )


def plot_mean_response_vectors(response_matrix, fig=None):
    """Plot mean response vector per target-NM: raw and min-max normalized.

    Creates two stacked axes sharing the x-axis. Top: raw magnitudes.
    Bottom: each recording min-max normalized to [0, 1] independently.

    Parameters
    ----------
    response_matrix : pd.DataFrame
        Rows indexed by (eid, target_NM), columns = feature labels.
    fig : plt.Figure, optional

    Returns
    -------
    plt.Figure
    """
    # Sort features by side > event > feedback > contrast
    col_order = sorted(response_matrix.columns, key=feature_sort_key)
    response_matrix = response_matrix[col_order]

    labels = response_matrix.index.get_level_values('target_NM')
    targets = sorted(labels.unique())

    # Min-max normalize each recording independently
    row_min = response_matrix.min(axis=1)
    row_max = response_matrix.max(axis=1)
    row_range = row_max - row_min
    row_range = row_range.replace(0, np.nan)
    normalized = response_matrix[col_order].sub(row_min, axis=0).div(row_range, axis=0)

    # Identify group boundaries (side × event × feedback, ignoring contrast)
    groups = []
    for col in col_order:
        m = _FEATURE_RE.match(col)
        groups.append((m['side'], m['event'], m['fb']) if m else None)

    if fig is None:
        n_features = response_matrix.shape[1]
        fig, axes = plt.subplots(2, 1, figsize=(max(8, n_features * 0.25), 7),
                                 sharex=True)
    else:
        axes = fig.axes[:2]

    for ax, data, ylabel in zip(
        axes,
        [response_matrix, normalized],
        ['Raw response magnitude', 'Normalized response magnitude'],
    ):
        # Alternating background shading per group
        current_group = None
        shade_idx = 0
        block_start = 0
        for i, g in enumerate(groups + [None]):
            if g != current_group:
                if current_group is not None and shade_idx % 2 == 1:
                    ax.axvspan(block_start - 0.5, i - 0.5,
                               color='0.93', zorder=0)
                current_group = g
                block_start = i
                shade_idx += 1

        for target in targets:
            mask = labels == target
            mean_vec = data.loc[mask].mean(axis=0)
            sem_vec = data.loc[mask].sem(axis=0)
            x = np.arange(len(mean_vec))
            color = TARGETNM_COLORS.get(target, None)
            ax.errorbar(x, mean_vec.values, yerr=sem_vec.values,
                        fmt='o', markersize=3, capsize=2, label=target,
                        color=color)
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False)

    axes[-1].set_xticks(np.arange(len(response_matrix.columns)))
    axes[-1].set_xticklabels(response_matrix.columns, rotation=90, fontsize=TICKFONTSIZE)
    axes[0].set_title('Mean response vectors by target-NM')
    fig.tight_layout()
    return fig


def plot_within_target_similarity(sim_matrix, labels, subjects, fig=None):
    """Barplot of mean within-target similarity with per-subject scatter.

    Parameters
    ----------
    sim_matrix : pd.DataFrame
        Symmetric pairwise cosine similarity matrix.
    labels : pd.Series
        target_NM per recording, aligned to sim_matrix index.
    subjects : pd.Series
        Subject per recording, aligned to sim_matrix index.
    fig : plt.Figure, optional

    Returns
    -------
    plt.Figure
    """
    targets = sorted(labels.unique())
    bar_means = []
    bar_colors = []
    subject_points = []  # list of lists of per-subject means

    for target in targets:
        mask = labels == target
        idx = labels.index[mask]
        # Extract within-target submatrix
        sub = sim_matrix.loc[idx, idx].values
        n = len(idx)
        # All off-diagonal pairs
        triu_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        within_vals = sub[triu_mask]
        bar_means.append(np.nanmean(within_vals))
        bar_colors.append(TARGETNM_COLORS.get(target, 'gray'))

        # Per-subject means
        subj_vals = subjects.loc[idx].values
        pts = []
        for s in sorted(set(subj_vals)):
            s_mask = subj_vals == s
            s_idx = np.where(s_mask)[0]
            if len(s_idx) < 2:
                continue
            s_pairs = []
            for i in range(len(s_idx)):
                for j in range(i + 1, len(s_idx)):
                    s_pairs.append(sub[s_idx[i], s_idx[j]])
            pts.append(np.nanmean(s_pairs))
        subject_points.append(pts)

    if fig is None:
        fig, ax = plt.subplots(figsize=(max(4, len(targets) * 1.2), 4))
    else:
        ax = fig.axes[0]

    x = np.arange(len(targets))
    ax.bar(x, bar_means, color=bar_colors, edgecolor='white', zorder=2)

    rng = np.random.default_rng(0)
    for i, pts in enumerate(subject_points):
        if pts:
            jitter = rng.uniform(-0.15, 0.15, len(pts))
            ax.scatter(x[i] + jitter, pts, color='black', s=15,
                       zorder=3, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(targets, rotation=45, ha='right')
    ax.set_ylabel('Mean within-target similarity')
    ax.set_title('Within-target cosine similarity')
    fig.tight_layout()
    return fig


def _plot_similarity_heatmap(ax, target_sim, title):
    """Render an annotated similarity heatmap on a single axis."""
    im = ax.imshow(target_sim.values, cmap='YlOrRd', aspect='equal',
                   vmin=0, vmax=1)
    for i in range(len(target_sim)):
        for j in range(len(target_sim.columns)):
            val = target_sim.iloc[i, j]
            text = f'{val:.2f}' if np.isfinite(val) else ''
            ax.text(j, i, text, ha='center', va='center',
                    color='white' if val > 0.5 else 'black')
    ax.set_xticks(range(len(target_sim.columns)))
    ax.set_xticklabels(target_sim.columns)
    ax.set_yticks(range(len(target_sim.index)))
    ax.set_yticklabels(target_sim.index)
    ax.set_title(title)
    return im


def plot_empirical_similarity(target_sim, loso_matrix=None, fig=None):
    """Plot target x target mean similarity as an annotated heatmap.

    When ``loso_matrix`` is provided, plots two side-by-side heatmaps:
    all pairs (left) and cross-subject pairs only (right).

    Parameters
    ----------
    target_sim : pd.DataFrame
        Square matrix of mean pairwise similarities (all pairs).
    loso_matrix : pd.DataFrame, optional
        Same structure but computed excluding same-subject pairs.
    fig : plt.Figure, optional

    Returns
    -------
    plt.Figure
    """
    n = len(target_sim)
    if loso_matrix is not None:
        if fig is None:
            fig, axes = plt.subplots(1, 2, figsize=(max(8, n * 2.4), max(3.5, n * 1.0)))
        else:
            axes = fig.axes[:2]
        _plot_similarity_heatmap(axes[0], target_sim, 'All pairs')
        im = _plot_similarity_heatmap(axes[1], loso_matrix, 'Cross-subject pairs')
        fig.colorbar(im, ax=axes, label='Mean cosine similarity', shrink=0.8)
    else:
        if fig is None:
            fig, ax = plt.subplots(1, 1, figsize=(max(4, n * 1.2), max(3.5, n * 1.0)))
        else:
            ax = fig.axes[0]
        im = _plot_similarity_heatmap(ax, target_sim, 'Mean pairwise similarity by target')
        fig.colorbar(im, ax=ax, label='Mean cosine similarity')
    fig.tight_layout()
    return fig


def plot_lmm_variance_explained(r2_df, ax=None):
    """Paired marginal/conditional R² bars per target-NM, for one event.

    Single-panel, ax-injectable. Each target-NM gets two bars in its colour
    (``TARGETNM_COLORS``): marginal R² (alpha 0.5) and conditional R² (opaque).

    Parameters
    ----------
    r2_df : pd.DataFrame
        Columns ``target_NM``, ``marginal_r2``, ``conditional_r2`` (one event).
    ax : plt.Axes, optional
        Axis to draw into; a new figure is created when None.

    Returns
    -------
    plt.Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(4, len(r2_df)), 4),
                               layout='constrained')
    else:
        fig = ax.figure

    targets = sorted(r2_df['target_NM'],
                     key=lambda t: TARGETNM2POSITION.get(t, 999))
    bar_w = 0.35
    for i, tnm in enumerate(targets):
        row = r2_df[r2_df['target_NM'] == tnm].iloc[0]
        color = TARGETNM_COLORS.get(tnm, f'C{i}')
        ax.bar(i - bar_w / 2, row['marginal_r2'], width=bar_w, color=color,
               alpha=0.5, label='Fixed' if i == 0 else '')
        ax.bar(i + bar_w / 2, row['conditional_r2'], width=bar_w, color=color,
               alpha=1.0, label='Fixed + random' if i == 0 else '')

    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(targets, rotation=45, ha='right', fontsize=TICKFONTSIZE)
    ax.set_ylabel('R²')
    ax.set_title('Variance explained')
    ax.legend(frameon=False, fontsize=TICKFONTSIZE, loc='upper left')
    return fig


# Readable x-tick labels for deviation-coded categorical factors. Continuous
# factors (contrast) fall through to their coded level value.
_EMM_LEVEL_LABELS = {
    'reward': {-0.5: 'incorrect', 0.5: 'correct'},
    'side': {0.5: 'contra', -0.5: 'ipsi'},
}


def _decode_contrast_levels(emm_df: pd.DataFrame) -> pd.DataFrame:
    """Replace per-fit-centered coded contrast with true percent, per target-NM.

    The LMM mean-centers log2 contrast on each recording's own trial
    distribution (:meth:`iblnm.data.PhotometrySession.code_predictors`), so one
    percent maps to
    a different coded level for every target-NM. Within a single fit the
    0%-contrast clamp is always the minimum level, so subtracting that fit's own
    minimum cancels its centering constant and the log2 inverse recovers percent.
    Decoding must therefore be per target-NM: a single pooled offset only cancels
    the centering of the target-NM holding the global minimum, leaving the rest
    on incomparable axes. Returns a copy; the input is not mutated.
    """
    inverse = get_contrast_coding('log2')[1]
    emm_df = emm_df.copy()
    emm_df['contrast'] = emm_df.groupby('target_NM')['contrast'].transform(
        lambda c: inverse(c - c.min()))
    return emm_df


def _emm_level_labels(factor: str, levels: list[float]) -> list[str]:
    """X-axis labels for one factor's EMM levels.

    ``reward``/``side`` map deviation codes (±0.5) to words. ``contrast`` levels
    arrive already decoded to percent (see :func:`_decode_contrast_levels`), so
    they — and any other factor — format their numeric level directly.

    Parameters
    ----------
    factor : str
        EMM factor name (the emm frame's level column).
    levels : list of float
        Sorted levels for ``factor`` (percent for ``contrast``).

    Returns
    -------
    list of str
        One tick label per level.
    """
    label_map = _EMM_LEVEL_LABELS.get(factor, {})
    return [str(label_map.get(lvl, f'{lvl:g}')) for lvl in levels]


def plot_marginal_means(emm_df, ax=None):
    """Main-effect estimated marginal means for one factor, per target-NM.

    Single-panel, ax-injectable. The factor is the lone non-value column of
    ``emm_df`` (every column except ``predicted``/``ci_lower``/``ci_upper`` and
    the identity columns ``target_NM``/``event``). Each target-NM is an errorbar
    series across the factor's levels (predicted mean ± 95% CI), slightly offset
    to avoid overlap.

    Parameters
    ----------
    emm_df : pd.DataFrame
        :func:`iblnm.analysis.compute_marginal_means` output tagged with
        ``target_NM`` for one event: a factor column (coded levels),
        ``predicted``, ``ci_lower``, ``ci_upper``.
    ax : plt.Axes, optional
        Axis to draw into; a new figure is created when None.

    Returns
    -------
    plt.Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4), layout='constrained')
    else:
        fig = ax.figure

    value_cols = {'predicted', 'ci_lower', 'ci_upper', 'target_NM', 'event'}
    factor = next(c for c in emm_df.columns if c not in value_cols)
    if factor == 'contrast':
        emm_df = _decode_contrast_levels(emm_df)
    levels = sorted(emm_df[factor].unique())

    targets = sorted(emm_df['target_NM'].unique(),
                     key=lambda t: TARGETNM2POSITION.get(t, 999))
    for i, tnm in enumerate(targets):
        sub = (emm_df[emm_df['target_NM'] == tnm]
               .set_index(factor).reindex(levels))
        means = sub['predicted'].values
        yerr = np.array([means - sub['ci_lower'].values,
                         sub['ci_upper'].values - means])
        x = np.arange(len(levels)) + i * 0.05 - 0.025 * len(targets)
        color = TARGETNM_COLORS.get(tnm, f'C{i}')
        ax.errorbar(x, means, yerr=yerr, fmt='o', capsize=4, color=color,
                    label=tnm, markersize=5)

    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(_emm_level_labels(factor, levels),
                       rotation=45, ha='right', fontsize=TICKFONTSIZE)
    ax.set_ylabel('z-score (EMM)')
    ax.set_title(f'Effect of {factor}')
    ax.axhline(0, ls='--', color='gray', lw=0.5)
    ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(1, 1),
              fontsize=TICKFONTSIZE)
    return fig


def _pval_to_stars(p):
    """Convert p-value to significance stars."""
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return ''


def plot_lmm_coefficient_heatmap(coef_df, ax=None):
    """Heatmap of LMM coefficients for one event: targets × terms.

    Single-panel, ax-injectable. Drops the Intercept term (uninterpretable for
    firstMovement and feedback due to prior-event contamination of the
    baseline). Cell colour is the coefficient (diverging ``RdBu_r``, symmetric
    about zero); asterisks mark significance.

    Parameters
    ----------
    coef_df : pd.DataFrame
        Coefficients for a single event, columns ``term``, ``target_NM``,
        ``Coef.``, ``P>|z|``.
    ax : plt.Axes, optional
        Axis to draw into; a new figure is created when None.

    Returns
    -------
    plt.Figure
    """
    coef_df = coef_df[coef_df['term'] != 'Intercept']

    term_order = [
        'side', 'reward', 'contrast',
        'side:reward', 'contrast:side',
        'contrast:reward', 'contrast:side:reward',
    ]
    targets = sorted(coef_df['target_NM'].unique(),
                     key=lambda x: TARGETNM2POSITION.get(x, 999))
    present_terms = [t for t in term_order if t in coef_df['term'].values]
    present_terms += sorted(set(coef_df['term']) - set(term_order))

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(0.9 * len(present_terms) + 1.5, 0.6 * len(targets) + 1))
    else:
        fig = ax.figure

    coef_matrix = np.full((len(targets), len(present_terms)), np.nan)
    pval_matrix = np.ones((len(targets), len(present_terms)))
    for i, tnm in enumerate(targets):
        for j, term in enumerate(present_terms):
            row = coef_df[(coef_df['target_NM'] == tnm)
                          & (coef_df['term'] == term)]
            if len(row) == 1:
                coef_matrix[i, j] = row['Coef.'].iloc[0]
                pval_matrix[i, j] = row['P>|z|'].iloc[0]

    vmax = np.nanmax(np.abs(coef_matrix))
    im = ax.imshow(coef_matrix, aspect='auto', cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax)

    for i in range(len(targets)):
        for j in range(len(present_terms)):
            stars = _pval_to_stars(pval_matrix[i, j])
            if stars:
                ax.text(j, i, stars, ha='center', va='center',
                        fontsize=TICKFONTSIZE, fontweight='bold',
                        color='k' if abs(coef_matrix[i, j]) < 0.6 * vmax
                        else 'w')

    col_labels = [_coef_label(t) for t in present_terms]
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=TICKFONTSIZE)
    ax.set_yticks(range(len(targets)))
    ax.set_yticklabels(targets, fontsize=TICKFONTSIZE)
    ax.set_title('Coefficients')
    fig.colorbar(im, ax=ax, label='Coefficient', shrink=0.8)
    return fig


def plot_lmm_summary(r2_df, coef_df, emm_frames, event, formula=None,
                     fig=None):
    """5-panel LMM summary for one event, composed from the modular plotters.

    Thin orchestrator: builds the gridspec and delegates each panel to an
    ax-injectable plotter, sourcing data from the precomputed effect frames.
    Each input frame may span several events; all are filtered to ``event``
    before plotting.

    Panels:
    1. Variance explained (R² bars), top-left.
    2. Coefficient heatmap, top-right.
    3+. Main-effect EMM panels (bottom row), one per factor in ``reward``,
       ``side``, ``contrast`` that ``formula`` names — so a reward-free model
       draws no reward panel. With ``formula=None`` all three are drawn.

    Parameters
    ----------
    r2_df : pd.DataFrame
        Per-fit variance explained: ``target_NM``, ``event``, ``marginal_r2``,
        ``conditional_r2``.
    coef_df : pd.DataFrame
        Fixed-effects table: ``term``, ``target_NM``, ``event``, ``Coef.``,
        ``P>|z|``.
    emm_frames : dict[str, pd.DataFrame]
        Maps each bottom-row factor (``'reward'``, ``'side'``, ``'contrast'``)
        to its estimated-marginal-means frame for that factor.
    event : str
        Event to plot; selects rows from each frame.
    formula : str, optional
        Base-model formula; annotated under the title when given.
    fig : plt.Figure, optional

    Returns
    -------
    plt.Figure
    """
    # Bottom-row factors track the model: only EMM panels for factors the
    # formula actually names are drawn, so reward-free events (stimOn,
    # firstMovement) get no reward panel — consistent with the heatmap. Word
    # boundaries keep `side` from matching inside `choice_side`. 6 columns
    # divide evenly for the 1-3 factors a task formula can name.
    bottom_factors = [f for f in ('reward', 'side', 'contrast')
                      if formula is None or re.search(rf'\b{f}\b', formula)]
    span = 6 // len(bottom_factors)

    if fig is None:
        fig = plt.figure(figsize=(16, 9), layout='constrained')
        fig.get_layout_engine().set(w_pad=0.12, h_pad=0.12, wspace=0.06,
                                    hspace=0.10)
    gs = fig.add_gridspec(2, 6)
    ax_r2 = fig.add_subplot(gs[0, :2])
    ax_hm = fig.add_subplot(gs[0, 2:])
    bottom_axes = [fig.add_subplot(gs[1, i * span:(i + 1) * span])
                   for i in range(len(bottom_factors))]

    r2_event = r2_df[r2_df['event'] == event]
    coef_event = coef_df[coef_df['event'] == event]

    plot_lmm_variance_explained(r2_event, ax=ax_r2)
    plot_lmm_coefficient_heatmap(coef_event, ax=ax_hm)

    for ax, factor in zip(bottom_axes, bottom_factors):
        emm = emm_frames[factor]
        plot_marginal_means(emm[emm['event'] == event], ax=ax)

    suptitle = f'LMM summary — {event}'
    if formula is not None:
        suptitle += f'\n{formula}'
    fig.suptitle(suptitle, fontsize=LABELFONTSIZE)
    return fig


_TASK_CEILING_TITLE = ('Ceiling R²\nsaturated C(contrast) × side (× reward at '
                       'feedback), no side:reward')


def plot_lmm_ceiling(ceiling_df, title=_TASK_CEILING_TITLE):
    """Saturated-model ceiling R² (marginal and conditional) per target-NM.

    One panel per event; within each, paired bars per target-NM give the
    fixed-effects (marginal) and fixed+random (conditional) R² of the per-event
    saturated model — the upper bound the parametric models are compared
    against. The default ``title`` describes the task ceiling; the movement
    ceiling passes its own.

    Parameters
    ----------
    ceiling_df : pd.DataFrame
        ``fit_lmm`` ceiling rows: ``target_NM``, ``event``, ``marginal``,
        ``conditional``.
    title : str
        Figure suptitle (the saturated model it depicts).

    Returns
    -------
    plt.Figure
    """
    events = _sort_events(ceiling_df['event'].unique()) if len(ceiling_df) else []
    n_panels = max(len(events), 1)
    fig, axes = plt.subplots(1, n_panels, figsize=(3 * n_panels + 1, 4),
                             sharey=True, layout='constrained')
    axes = np.atleast_1d(axes)
    if len(ceiling_df) == 0:
        fig.suptitle(title, fontsize=LABELFONTSIZE)
        return fig

    bar_w = 0.35
    for ax, event in zip(axes, events):
        df_ev = ceiling_df[ceiling_df['event'] == event]
        targets = sorted(df_ev['target_NM'].unique(),
                         key=lambda x: TARGETNM2POSITION.get(x, 999))
        for i, tnm in enumerate(targets):
            row = df_ev[df_ev['target_NM'] == tnm].iloc[0]
            color = TARGETNM_COLORS.get(tnm, f'C{i}')
            ax.bar(i - bar_w / 2, row['marginal'], width=bar_w, color=color,
                   alpha=0.5, label='Fixed' if i == 0 else '')
            ax.bar(i + bar_w / 2, row['conditional'], width=bar_w, color=color,
                   alpha=1.0, label='Fixed + random' if i == 0 else '')
        ax.set_xticks(range(len(targets)))
        ax.set_xticklabels(targets, rotation=45, ha='right',
                           fontsize=TICKFONTSIZE)
        ax.set_title(event)
    axes[0].set_ylabel('Ceiling R²')
    axes[0].legend(frameon=False, fontsize=TICKFONTSIZE, loc='upper left')
    fig.suptitle(title, fontsize=LABELFONTSIZE)
    return fig


# Order of recognized task terms on the reliability x-axis: main effects, then
# the omnibus interaction block. Unrecognized predictors (e.g. movement timing
# variables) are appended in encounter order.
_RELIABILITY_TERMS = ['contrast', 'side', 'reward', 'interactions']


def _reliability_predictor_order(predictors: Iterable[str]) -> list[str]:
    """Order predictors: known task terms first (``_RELIABILITY_TERMS``), then
    any others in encounter order."""
    present = list(dict.fromkeys(predictors))
    known = [t for t in _RELIABILITY_TERMS if t in present]
    others = [p for p in present if p not in _RELIABILITY_TERMS]
    return known + others


def plot_lmm_reliability(reliability_df, full_r2, title):
    """Out-of-sample ΔR² per predictor — grid of target-NM (rows) × event (cols).

    Each cell shows the raw per-predictor ΔR²: small faint markers are the
    per-fold values (e.g. leave-one-subject-out), the large black-edged marker
    is the across-fold aggregate. Positive = the predictor helps predict
    held-out data. Each target-NM row is drawn in its own color and shares a
    y-axis within the row, so predictors and events are comparable within an NM
    while rows scale independently. The full model's in-sample marginal R² is
    annotated top-left of each panel as an absolute reference. Predictor order
    keeps the recognized task terms (``_RELIABILITY_TERMS``) first and appends
    any others (e.g. ``log_<var>``) in encounter order.

    Parameters
    ----------
    reliability_df : pd.DataFrame
        Long-form ΔR² rows: ``target_NM``, ``event``, ``predictor``, ``fold``,
        ``delta_r2`` (with a per-group ``fold == 'aggregate'`` row).
    full_r2 : pd.DataFrame
        Full model's in-sample marginal R² per panel: columns ``target_NM``,
        ``event``, ``marginal_r2``. Annotated top-left as an absolute reference.
    title : str
        Figure suptitle.

    Returns
    -------
    plt.Figure
    """
    has_data = len(reliability_df) > 0
    targets = (sorted(reliability_df['target_NM'].unique(),
                      key=lambda x: TARGETNM2POSITION.get(x, 999))
               if has_data else [])
    events = _sort_events(reliability_df['event'].unique()) if has_data else []
    n_rows, n_cols = max(len(targets), 1), max(len(events), 1)
    fig, axes = plt.subplots(n_rows, n_cols, squeeze=False, sharex=True,
                             sharey='row', layout='constrained',
                             figsize=(2.6 * n_cols + 1, 2.0 * n_rows + 1))

    if not has_data:
        fig.suptitle(title, fontsize=LABELFONTSIZE)
        return fig

    marginal = full_r2.set_index(['target_NM', 'event'])['marginal_r2']
    terms = _reliability_predictor_order(reliability_df['predictor'])
    for r, target_nm in enumerate(targets):
        color = TARGETNM_COLORS.get(target_nm, 'gray')
        df_t = reliability_df[reliability_df['target_NM'] == target_nm]
        for c, event in enumerate(events):
            ax = axes[r, c]
            df_ev = df_t[df_t['event'] == event]
            for x, term in enumerate(terms):
                _scatter_folds(ax, x, df_ev[df_ev['predictor'] == term], color)
            ax.axhline(0, ls='--', color='gray', lw=0.5)
            m = marginal.get((target_nm, event), np.nan)
            if np.isfinite(m):
                ax.text(0.03, 0.97, f'R²ₘ={m:.2f}', transform=ax.transAxes,
                        va='top', ha='left', fontsize=TICKFONTSIZE)
            if r == 0:
                ax.set_title(event)
        axes[r, 0].set_ylabel(target_nm, fontsize=TICKFONTSIZE)

    for c in range(n_cols):
        axes[-1, c].set_xticks(range(len(terms)))
        axes[-1, c].set_xticklabels(terms, rotation=30, ha='right',
                                    fontsize=TICKFONTSIZE)
    fig.supylabel('Out-of-sample ΔR²')
    fig.suptitle(title, fontsize=LABELFONTSIZE)
    return fig


# x-axis layout (units where one subject occupies a width of 1):
_SUBJECT_SPACING = 0.7       # x between consecutive subjects within a target-NM
_TARGETNM_GAP = 1.0          # blank x between consecutive target-NM groups
_SESSION_MARKER_SIZE = 40    # open-dot marker size for a single session
_MEAN_MARKER_SIZE = 260      # '_' marker size for a subject's mean dash
_MEAN_LINEWIDTH = 3.0        # '_' mean dash thickness
_MEDIAN_MARKER_SIZE = 6      # errorbar median-point diameter (points)


def _group_xslots(df, targets):
    """Lay out one contiguous block of x slots per target-NM group.

    Each target-NM with data gets one slot per subject, ``_SUBJECT_SPACING``
    apart, with ``_TARGETNM_GAP`` blank units between groups — so a target-NM's
    horizontal extent scales with its subject count. Subjects fill slots in the
    name-sorted (alphanumeric) order returned here; callers no longer reorder
    them per panel.

    Parameters
    ----------
    df : pd.DataFrame
        One event's rows; needs ``target_NM`` and ``subject``.
    targets : sequence of str
        Target-NMs in plot order.

    Returns
    -------
    subjects_by_target : dict[str, list[str]]
        Subjects present per target-NM, name-sorted (their plot order).
    slots_by_target : dict[str, np.ndarray]
        The x positions available to each target-NM group.
    ticks : list[tuple[str, float]]
        ``(target_NM, centre_x)`` pairs, one per non-empty target-NM.
    """
    subjects_by_target, slots_by_target, ticks = {}, {}, []
    x = 0.0
    for tnm in targets:
        subjects = sorted(df.loc[df['target_NM'] == tnm, 'subject'].unique())
        if not subjects:
            continue
        xs = x + np.arange(len(subjects)) * _SUBJECT_SPACING
        subjects_by_target[tnm] = subjects
        slots_by_target[tnm] = xs
        ticks.append((tnm, float(xs.mean())))
        x += len(subjects) * _SUBJECT_SPACING + _TARGETNM_GAP
    return subjects_by_target, slots_by_target, ticks


def _scatter_subject(ax, x, deltas, point_colors, summary_color):
    """Plot one subject's sessions and its mean at ``x``.

    Each session is a translucent open dot (no fill) stacked at the subject's x,
    edge-colored by its own entry of ``point_colors``; the subject's mean is a
    single thicker ``'_'`` marker in ``summary_color`` on top. Either color is
    gray where significance routing marks that grain non-significant (see
    ``_significance_color``). The dots stay one ``scatter`` call so a
    single-point collection unambiguously identifies the mean marker.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    x : float
        The subject's x position.
    deltas : np.ndarray
        That subject's per-session ΔR² in one cell.
    point_colors : sequence of color
        One color per session, aligned element-wise to ``deltas``.
    summary_color : color
        Color of the mean dash (the subject's target-NM color, or gray).
    """
    ax.scatter(np.full(len(deltas), x), deltas, marker='o', facecolors='none',
               edgecolors=point_colors, s=_SESSION_MARKER_SIZE, alpha=0.5,
               zorder=3)
    ax.scatter(x, np.mean(deltas), marker='_', color=summary_color,
               s=_MEAN_MARKER_SIZE, linewidths=_MEAN_LINEWIDTH, zorder=4)


def _median_iqr_subject(ax, x, vals, point_colors, summary_color):
    """Draw one subject as a median point with a Q1–Q3 whisker at ``x``.

    The median is a filled point in ``summary_color``; the whisker spans the
    subject's interquartile range (25th–75th percentile of its per-session
    ``vals``). Mirrors ``_scatter_subject``'s zorder conventions; ``point_colors``
    is accepted for that shared draw-mark signature and ignored, since this mode
    draws no per-session mark.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    x : float
        The subject's x position.
    vals : np.ndarray
        That subject's per-session values in one cell.
    point_colors : sequence of color
        Unused.
    summary_color : color
        Marker and whisker color (the subject's target-NM color).
    """
    median, q1, q3 = _median_iqr(vals)
    ax.errorbar(x, median, yerr=[[median - q1], [q3 - median]], fmt='o',
                color=summary_color, markersize=_MEDIAN_MARKER_SIZE, zorder=4)


def _median_iqr(vals):
    """Median and interquartile range of a subject's per-session values.

    Parameters
    ----------
    vals : array-like
        One subject's per-session values in a cell.

    Returns
    -------
    tuple[float, float, float]
        ``(median, q1, q3)`` — the median and the 25th/75th percentiles.
    """
    q1, q3 = np.percentile(vals, [25, 75])
    return float(np.median(vals)), float(q1), float(q3)


def _significance_color(base_color, pvalues, keys, alpha):
    """Resolve a marker color from a permutation q-value at either grain.

    Serves both the per-mouse table (matched on ``event``, ``predictor``,
    ``subject``) and the per-session table (matched on ``eid``, ``event``,
    ``predictor``): the grain is set purely by which columns ``keys`` names.
    Returns ``base_color`` (the target-NM color) when the matched row's
    ``q_value`` is below ``alpha``. Returns ``'gray'`` otherwise — when the
    q-value is at or above ``alpha``, when no row matches, and when ``pvalues``
    is ``None`` (no significance routing, e.g. ``plot_ols_total_r2``).

    Parameters
    ----------
    base_color : color
        The target-NM color, used when significant.
    pvalues : pd.DataFrame or None
        Permutation results carrying ``q_value`` plus every column named in
        ``keys``. ``None`` disables fading.
    keys : dict[str, str]
        Column name -> value the row must equal, jointly identifying one row.
    alpha : float
        False-discovery-rate threshold; ``q_value < alpha`` keeps the color.
    """
    if pvalues is None:
        return base_color
    match = np.logical_and.reduce([(pvalues[col] == value).to_numpy()
                                   for col, value in keys.items()])
    row = pvalues[match]
    if not len(row):
        return 'gray'
    return _qvalue_color(base_color, row['q_value'].iloc[0], alpha)


# The drop-one labels grouped into the two classes whose ΔR² figures share a
# y-scale: the six main effects (`PERSESSION_REGRESSORS` order) and the twelve
# two-way interactions, in the order they are written in the config. Main-effect
# contributions run an order of magnitude above the interactions, so one scale
# across all eighteen flattens the interactions to a line.
DROPONE_TERM_CLASSES = {
    'main': [term for term in RESPONSE_DROPPED_TERMS
             if term in PERSESSION_REGRESSORS],
    'interaction': [term for term in RESPONSE_DROPPED_TERMS
                    if term not in PERSESSION_REGRESSORS],
}


def _dropone_rows(predictor):
    """Grid row + shared y-label for one dropped term's ΔR² figure.

    Parameters
    ----------
    predictor : str
        A ``config.RESPONSE_DROPPED_TERMS`` label. One figure covers one label,
        so the grid it builds is a single row read off that label's frame rows.

    Returns
    -------
    rows : list[tuple[str, str, str]]
        The one ``(row_label, value_column, predictor)`` row, reading
        ``delta_r2_adj`` — the adjusted difference, which charges each model for
        its own parameter count and so is not inflated by the reference model's
        extra terms.
    supylabel : str
    """
    return ([(predictor, 'delta_r2_adj', predictor)],
            'adjusted ΔR² (per-session, in-sample)')


def _total_r2_rows():
    """Grid rows + shared y-label for the per-session full-model R² figure.

    Adjusted rather than raw R², matching ``_dropone_rows``: the full model
    spends 18 parameters and each reduced model 12, so a raw R² would put the
    two figures on different scales — the ΔR² panel charged for its parameters
    and the total panel not.

    Returns
    -------
    rows : list[tuple[str, str, str]]
        A single ``(row_label, value_column, predictor)`` reading full-model
        ``r2_full_adj`` off one predictor (it repeats across predictors per
        session).
    supylabel : str
    """
    return ([('full model R²', 'r2_full_adj', PERSESSION_REGRESSORS[0])],
            'adjusted R² (per-session, in-sample)')


def _pool_by_target(df_cell, value_col, targets):
    """Pool a cell's per-session values by target-NM.

    Collects ``value_col`` across every session (all subjects) of each target-NM
    in ``targets``, dropping targets with no rows in this cell.

    Parameters
    ----------
    df_cell : pd.DataFrame
        One (event, predictor) cell's rows; needs ``target_NM`` and
        ``value_col``. Each row is one session.
    value_col : str
        Column of per-session values to pool (e.g. ``delta_r2`` or ``r2``).
    targets : sequence of str
        Target-NMs in plot order.

    Returns
    -------
    dict[str, np.ndarray]
        ``target_NM -> pooled per-session values``, in ``targets`` order,
        omitting targets with no values in this cell.
    """
    pooled = {tnm: df_cell.loc[df_cell['target_NM'] == tnm, value_col].values
              for tnm in targets}
    return {tnm: vals for tnm, vals in pooled.items() if len(vals)}


def _qvalue_color(base_color, q_value, alpha):
    """Resolve a mark color from one row's own permutation q-value.

    The per-recording grain of :func:`_significance_color`, for a frame that
    carries its own ``q_value`` column rather than a separate table. A row the
    permutation could not score carries NaN, which fails the comparison and so
    grays like a non-significant one.
    """
    return base_color if q_value < alpha else 'gray'


def _population_counts(df):
    """Recordings and mice behind each ``(target_NM, event)`` of a plotted frame.

    A recording is a distinct ``(eid, brain_region)`` pair, not a distinct
    ``eid``: a bilateral session fits one model per region and contributes both.
    Predictor rows repeat a recording, so they are deduplicated first.

    Parameters
    ----------
    df : pd.DataFrame
        Long-form per-recording frame with ``eid``, ``brain_region``,
        ``subject``, ``target_NM`` and ``event``.

    Returns
    -------
    dict[tuple[str, str], tuple[int, int]]
        ``(target_NM, event) -> (n_recordings, n_mice)``.
    """
    counts = (df.drop_duplicates(['eid', 'brain_region', 'target_NM', 'event'])
              .groupby(['target_NM', 'event'])
              .agg(n_recordings=('eid', 'size'),
                   n_mice=('subject', 'nunique')))
    return {key: (row['n_recordings'], row['n_mice'])
            for key, row in counts.iterrows()}


def _target_tick_label(tnm, event, counts_lookup):
    """Target-NM x-tick label, with a ``n=<rec>, m=<mice>`` line if counts given.

    ``counts_lookup`` maps ``(target_NM, event)`` to ``(n_recordings, n_mice)``;
    a missing key (or ``None`` lookup) yields the bare target name.
    """
    if counts_lookup is None or (tnm, event) not in counts_lookup:
        return tnm
    n_recordings, n_mice = counts_lookup[(tnm, event)]
    return f'{tnm}\nn={n_recordings}, m={n_mice}'


def _persession_subject_grid(df, title, rows, supylabel, draw_mark,
                             pvalues=None, alpha=PERSESSION_SIGNIFICANCE_ALPHA,
                             annotate_counts=False, color_by_qvalue=False,
                             ylim=None):
    """Per-subject-slot grid: ``rows`` by event columns, sharing one y-axis.

    Shared layout for the per-session figures that use a subject-slot x-axis.
    Each entry of ``rows`` is one grid row; columns are events (``_sort_events``
    order). Within a panel each subject occupies its own x position, grouped by
    target-NM with a gap so a target-NM's width scales with its subject count
    (see ``_group_xslots``); subjects are placed left to right in the
    alphanumeric order ``_group_xslots`` returns. Each subject's cell values are
    drawn by ``draw_mark`` in the subject's target-NM color, grayed at whichever
    grain is marked non-significant for that cell: ``pvalues`` grays the
    subject's summary mark, ``color_by_qvalue`` grays individual recording marks
    from the frame's own ``q_value``. One x-tick per target-NM is centred on its
    subjects. All panels share one y-axis; the figure size scales with the total
    subject count and the number of rows.

    Parameters
    ----------
    df : pd.DataFrame
        Long-form per-session fits: ``target_NM``, ``event``, ``subject``,
        ``predictor``, and the value columns named in ``rows``.
    title : str
        Figure suptitle.
    rows : list[tuple[str, str, str]]
        ``(row_label, value_column, predictor)`` per grid row. ``predictor``
        selects the frame rows to read (and, since ``r2`` repeats across
        predictors, dedupes a per-session value to one row).
    supylabel : str
        Shared y-axis label.
    draw_mark : callable
        ``draw_mark(ax, x, vals, point_colors, summary_color)`` drawing one
        subject's cell values at ``x`` (e.g. ``_scatter_subject``), with one
        color per value and one for the subject's summary mark.
    pvalues : pd.DataFrame or None
        Per-mouse permutation results. When given, each subject whose cell
        ``q_value >= alpha`` (or has no row) has its summary mark grayed rather
        than drawn in its target-NM color (see ``_significance_color``).
    alpha : float
        False-discovery-rate threshold; a mark keeps its color when
        ``q_value < alpha``.
    annotate_counts : bool
        Append a ``n=<recordings>, m=<mice>`` line to each target's x-tick
        label, counted off ``df`` (``_population_counts``, which needs ``eid``
        and ``brain_region``).
    color_by_qvalue : bool
        Color each recording's mark by its own row's ``q_value`` rather than
        leaving it in the subject's summary color; requires a ``q_value``
        column on ``df``.
    ylim : tuple[float, float] or None
        ``(bottom, top)`` for the shared y-axis. ``None`` autoscales to the
        panels' own data; a range makes figures drawn from different frames
        directly comparable.

    Returns
    -------
    plt.Figure
    """
    counts_lookup = _population_counts(df) if annotate_counts else None
    has_data = len(df) > 0
    events = _sort_events(df['event'].unique()) if has_data else []
    n_rows, n_cols = len(rows), max(len(events), 1)

    if not has_data:
        fig, _ = plt.subplots(n_rows, n_cols, squeeze=False,
                              layout='constrained')
        fig.suptitle(title, fontsize=LABELFONTSIZE)
        return fig

    targets = sorted(df['target_NM'].unique(),
                     key=lambda x: TARGETNM2POSITION.get(x, 999))
    layouts = {event: _group_xslots(df[df['event'] == event], targets)
               for event in events}
    col_extent = max(slots.max() + 1 for _, slots_by_target, _ in layouts.values()
                     for slots in slots_by_target.values())
    fig, axes = plt.subplots(
        n_rows, n_cols, squeeze=False, sharex='col', sharey=True,
        layout='constrained',
        figsize=((0.35 * col_extent * n_cols + 1) * 0.75,
                 (2.0 * n_rows + 1) * 1.25))

    for c, event in enumerate(events):
        subjects_by_target, slots_by_target, ticks = layouts[event]
        for r, (label, value_col, predictor) in enumerate(rows):
            ax = axes[r, c]
            df_cell = df[(df['event'] == event) & (df['predictor'] == predictor)]
            for tnm, subjects in subjects_by_target.items():
                base_color = TARGETNM_COLORS.get(tnm, 'gray')
                for subject, x in zip(subjects, slots_by_target[tnm]):
                    subject_rows = df_cell[df_cell['subject'] == subject]
                    vals = subject_rows[value_col].values
                    if len(vals):
                        summary_color = _significance_color(
                            base_color, pvalues,
                            {'event': event, 'predictor': predictor,
                             'subject': subject}, alpha)
                        point_colors = (
                            [_qvalue_color(base_color, q, alpha)
                             for q in subject_rows['q_value']]
                            if color_by_qvalue
                            else [summary_color] * len(vals))
                        draw_mark(ax, x, vals, point_colors, summary_color)
            ax.axhline(0, ls='--', color='gray', lw=0.5)
            if r == 0:
                ax.set_title(event)
            if c == 0:
                ax.set_ylabel(label, fontsize=TICKFONTSIZE)
            if ylim is not None:
                ax.set_ylim(ylim)
        axes[-1, c].set_xticks([centre for _, centre in ticks])
        axes[-1, c].set_xticklabels(
            [_target_tick_label(tnm, event, counts_lookup) for tnm, _ in ticks],
            rotation=30, ha='right', fontsize=TICKFONTSIZE)
    fig.supylabel(supylabel)
    fig.suptitle(title, fontsize=LABELFONTSIZE)
    return fig


def plot_ols_dropone(df, title, predictor, mouse_pvalues=None,
                     alpha=PERSESSION_SIGNIFICANCE_ALPHA, ylim=None):
    """Per-session drop-one ΔR² for one dropped term — event columns.

    Plots that term's ``delta_r2_adj`` as translucent per-recording dots plus a
    per-subject mean dash, one panel per event. See
    ``_persession_subject_grid``.

    Parameters
    ----------
    df : pd.DataFrame
        The per-recording OLS frame (``config.OLS_PERSESSION_COLUMNS``): one row
        per recording × event × dropped predictor, carrying the fit, that
        recording's own ``q_value`` — which colors its dot — and the identity
        columns the x-tick population counts are taken from.
    title : str
        Figure suptitle.
    predictor : str
        The dropped-term label this figure covers.
    mouse_pvalues : pd.DataFrame or None
        Per-mouse permutation results, the coarser grain that lives in its own
        frame. When given, a subject non-significant for a cell
        (``q_value >= alpha`` or no row) has its mean dash grayed instead of
        drawn in its target-NM color (see ``_significance_color``).
    alpha : float
        False-discovery-rate threshold; a mark keeps its color when
        ``q_value < alpha``.
    ylim : tuple[float, float] or None
        Shared y-axis range, so the figures of one term class are comparable
        (see ``DROPONE_TERM_CLASSES``). ``None`` autoscales to this term's data.
    """
    rows, supylabel = _dropone_rows(predictor)
    return _persession_subject_grid(df, title, rows, supylabel,
                                    draw_mark=_scatter_subject,
                                    pvalues=mouse_pvalues, alpha=alpha,
                                    annotate_counts=True, color_by_qvalue=True,
                                    ylim=ylim)


def plot_ols_total_r2(df, title):
    """Per-session full-model R² — single row × event columns.

    Same format as ``plot_ols_dropone`` but a separate figure (its own y-axis),
    plotting the full-model ``r2_full_adj`` (read off one predictor, since it
    repeats across them). See ``_persession_subject_grid``.
    """
    rows, supylabel = _total_r2_rows()
    return _persession_subject_grid(df, title, rows, supylabel,
                                    draw_mark=_scatter_subject)


def plot_ols_dropone_subject(df, title, predictor, ylim=None):
    """Per-session drop-one ΔR² — one median + Q1–Q3 whisker per subject.

    Same figure as ``plot_ols_dropone`` (one dropped term × event columns) but
    each subject is drawn as its median with an interquartile whisker instead
    of per-session dots. ``ylim`` fixes the shared y-axis. See
    ``_persession_subject_grid``.
    """
    rows, supylabel = _dropone_rows(predictor)
    return _persession_subject_grid(df, title, rows, supylabel,
                                    draw_mark=_median_iqr_subject, ylim=ylim)


def plot_ols_total_r2_subject(df, title):
    """Per-session full-model R² — one median + Q1–Q3 whisker per subject.

    Same figure as ``plot_ols_total_r2`` but each subject is drawn as its
    median with an interquartile whisker. See ``_persession_subject_grid``.
    """
    rows, supylabel = _total_r2_rows()
    return _persession_subject_grid(df, title, rows, supylabel,
                                    draw_mark=_median_iqr_subject)


def _target_violin(ax, slot, vals, color):
    """Draw one target-NM's pooled per-session values as a violin at ``slot``.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    slot : float
        The target-NM's x position (its index in the target order).
    vals : np.ndarray
        Pooled per-session values across the target's subjects in one cell.
    color : color
        Violin face/edge color (the target-NM color).
    """
    parts = ax.violinplot([vals], positions=[slot], showextrema=False)
    for body in parts['bodies']:
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.7)


def _persession_target_grid(df, title, rows, supylabel, ylim=None):
    """Per-target-slot grid: ``rows`` by event columns, sharing one y-axis.

    Shared layout for the per-session figures that use a target-slot x-axis.
    Each entry of ``rows`` is one grid row; columns are events (``_sort_events``
    order). Within a panel each target-NM occupies one x-slot (``TARGETNM2POSITION``
    order) drawn as a violin of the pooled per-session values across all its
    subjects (see ``_pool_by_target``), faced in
    ``TARGETNM_COLORS[target_NM]``. One x-tick per target-NM. All panels share
    one y-axis. An empty frame returns a titled figure.

    Parameters
    ----------
    df : pd.DataFrame
        Long-form per-session fits: ``target_NM``, ``event``, ``subject``,
        ``predictor``, and the value columns named in ``rows``.
    title : str
        Figure suptitle.
    rows : list[tuple[str, str, str]]
        ``(row_label, value_column, predictor)`` per grid row. ``predictor``
        selects the frame rows to read.
    supylabel : str
        Shared y-axis label.
    ylim : tuple[float, float] or None
        ``(bottom, top)`` for the shared y-axis. ``None`` autoscales to the
        panels' own data.

    Returns
    -------
    plt.Figure
    """
    has_data = len(df) > 0
    events = _sort_events(df['event'].unique()) if has_data else []
    n_rows, n_cols = len(rows), max(len(events), 1)

    if not has_data:
        fig, _ = plt.subplots(n_rows, n_cols, squeeze=False,
                              layout='constrained')
        fig.suptitle(title, fontsize=LABELFONTSIZE)
        return fig

    targets = sorted(df['target_NM'].unique(),
                     key=lambda t: TARGETNM2POSITION.get(t, 999))
    fig, axes = plt.subplots(
        n_rows, n_cols, squeeze=False, sharex=True, sharey=True,
        layout='constrained',
        figsize=(1.2 * len(targets) * n_cols + 1, 2.0 * n_rows + 1))

    for c, event in enumerate(events):
        for r, (label, value_col, predictor) in enumerate(rows):
            ax = axes[r, c]
            df_cell = df[(df['event'] == event) & (df['predictor'] == predictor)]
            pooled = _pool_by_target(df_cell, value_col, targets)
            for slot, tnm in enumerate(targets):
                if tnm in pooled:
                    _target_violin(ax, slot, pooled[tnm],
                                   TARGETNM_COLORS.get(tnm, 'gray'))
            ax.axhline(0, ls='--', color='gray', lw=0.5)
            if r == 0:
                ax.set_title(event)
            if c == 0:
                ax.set_ylabel(label, fontsize=TICKFONTSIZE)
            if ylim is not None:
                ax.set_ylim(ylim)
        axes[-1, c].set_xticks(range(len(targets)))
        axes[-1, c].set_xticklabels(targets, rotation=30, ha='right',
                                    fontsize=TICKFONTSIZE)
    fig.supylabel(supylabel)
    fig.suptitle(title, fontsize=LABELFONTSIZE)
    return fig


def plot_ols_dropone_violin(df, title, predictor, ylim=None):
    """Per-session drop-one ΔR² — one violin per target-NM of pooled sessions.

    Same figure as ``plot_ols_dropone`` (one dropped term × event columns) but
    each target-NM is drawn as a violin of the pooled per-session ΔR² across
    all its subjects, instead of per-subject marks. ``ylim`` fixes the shared
    y-axis. See ``_persession_target_grid``.
    """
    rows, supylabel = _dropone_rows(predictor)
    return _persession_target_grid(df, title, rows, supylabel, ylim=ylim)


def plot_ols_total_r2_violin(df, title):
    """Per-session full-model R² — one violin per target-NM of pooled sessions.

    Same figure as ``plot_ols_total_r2`` but each target-NM is drawn as a violin
    of its pooled per-session full-model R². See ``_persession_target_grid``.
    """
    rows, supylabel = _total_r2_rows()
    return _persession_target_grid(df, title, rows, supylabel)


def plot_decoding_summary(coefficients, contributions, fig=None):
    """Coefficients and unique contributions stacked with shared x-axis.

    Features are sorted by descending unique contribution (delta).

    Parameters
    ----------
    coefficients : pd.DataFrame
        Shape (n_classes, n_features).
    contributions : pd.DataFrame
        Columns: feature, delta, etc.
    fig : plt.Figure, optional

    Returns
    -------
    plt.Figure
    """
    # Sort features by side > event > feedback > contrast
    feature_order = sorted(contributions['feature'], key=feature_sort_key)
    contrib_sorted = contributions.set_index('feature').loc[feature_order].reset_index()

    # Reorder coefficients columns
    coefs_sorted = coefficients[feature_order]

    n_classes, n_features = coefs_sorted.shape
    if fig is None:
        fig = plt.figure(figsize=(max(8, n_features * 0.3), 5),
                         layout='constrained')
        gs = fig.add_gridspec(2, 2, height_ratios=[n_classes, 1],
                              width_ratios=[1, 0.02], wspace=0.03)
        ax_coef = fig.add_subplot(gs[0, 0])
        ax_delta = fig.add_subplot(gs[1, 0], sharex=ax_coef)
        ax_cbar = fig.add_subplot(gs[0, 1])
    else:
        ax_coef, ax_delta, ax_cbar = fig.axes[0], fig.axes[1], fig.axes[2]

    # Top: coefficients heatmap
    vmax = np.abs(coefs_sorted.values).max()
    im = ax_coef.imshow(coefs_sorted.values, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                        aspect='auto')
    fig.colorbar(im, cax=ax_cbar, label='Coefficient')
    ax_coef.set_yticks(range(n_classes))
    ax_coef.set_yticklabels(coefs_sorted.index)
    ax_coef.tick_params(axis='x', labelbottom=False)
    ax_coef.set_title('Decoding coefficients (L1 logistic)')

    # Bottom: contribution bars
    x = np.arange(n_features)
    ax_delta.bar(x, contrib_sorted['delta'].values)
    ax_delta.set_xticks(x)
    ax_delta.set_xticklabels(feature_order, rotation=90, fontsize=TICKFONTSIZE)
    ax_delta.set_ylabel(r'$\Delta$ acc.')
    ax_delta.axhline(0, color='k', linewidth=0.5)

    return fig


def plot_response_decoding_summary(response_matrix, coefficients,
                                    contributions, fig=None):
    """Unified figure: normalized response vectors + decoding coefficients + contributions.

    Three stacked axes sharing the x-axis:
    1. Top: min-max normalized response vectors (mean ± SEM per target_NM).
    2. Middle: decoding coefficients heatmap.
    3. Bottom: unique contribution bars.

    Parameters
    ----------
    response_matrix : pd.DataFrame
        Rows indexed by (eid, target_NM, ...), columns = feature labels.
    coefficients : pd.DataFrame
        Shape (n_classes, n_features) L1 logistic weights.
    contributions : pd.DataFrame
        Columns: feature, delta, etc.
    fig : plt.Figure, optional

    Returns
    -------
    plt.Figure
    """
    # Sort features
    feature_order = sorted(contributions['feature'], key=feature_sort_key)
    contrib_sorted = contributions.set_index('feature').loc[feature_order].reset_index()
    coefs_sorted = coefficients[feature_order]
    rm_sorted = response_matrix[feature_order]

    # Min-max normalize each recording
    row_min = rm_sorted.min(axis=1)
    row_max = rm_sorted.max(axis=1)
    row_range = (row_max - row_min).replace(0, np.nan)
    normalized = rm_sorted.sub(row_min, axis=0).div(row_range, axis=0)

    labels = rm_sorted.index.get_level_values('target_NM')
    targets = sorted(labels.unique())
    n_classes, n_features = coefs_sorted.shape

    if fig is None:
        fig = plt.figure(figsize=(max(8, n_features * 0.3), 9),
                         layout='constrained')
        gs = fig.add_gridspec(3, 2, height_ratios=[2, n_classes, 1],
                              width_ratios=[1, 0.02], wspace=0.03)
        ax_resp = fig.add_subplot(gs[0, 0])
        ax_coef = fig.add_subplot(gs[1, 0], sharex=ax_resp)
        ax_delta = fig.add_subplot(gs[2, 0], sharex=ax_resp)
        ax_cbar = fig.add_subplot(gs[1, 1])
    else:
        ax_resp, ax_coef, ax_delta, ax_cbar = (
            fig.axes[0], fig.axes[1], fig.axes[2], fig.axes[3])

    x = np.arange(n_features)

    # --- Top: normalized response vectors ---
    # Alternating background shading
    groups = []
    for col in feature_order:
        m = _FEATURE_RE.match(col)
        groups.append((m['side'], m['event'], m['fb']) if m else None)
    for ax in [ax_resp, ax_coef, ax_delta]:
        current_group = None
        shade_idx = 0
        block_start = 0
        for i, g in enumerate(groups + [None]):
            if g != current_group:
                if current_group is not None and shade_idx % 2 == 1:
                    ax.axvspan(block_start - 0.5, i - 0.5,
                               color='0.93', zorder=0)
                current_group = g
                block_start = i
                shade_idx += 1

    for target in targets:
        mask = labels == target
        mean_vec = normalized.loc[mask].mean(axis=0)
        sem_vec = normalized.loc[mask].sem(axis=0)
        color = TARGETNM_COLORS.get(target, None)
        ax_resp.errorbar(x, mean_vec.values, yerr=sem_vec.values,
                         fmt='o', markersize=3, capsize=2, label=target,
                         color=color)
    ax_resp.set_ylabel('Normalized response')
    ax_resp.legend(frameon=False, fontsize=TICKFONTSIZE)
    ax_resp.tick_params(axis='x', labelbottom=False)

    # --- Middle: coefficients heatmap ---
    vmax = np.abs(coefs_sorted.values).max()
    im = ax_coef.imshow(coefs_sorted.values, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                        aspect='auto')
    fig.colorbar(im, cax=ax_cbar, label='Coefficient')
    ax_coef.set_yticks(range(n_classes))
    ax_coef.set_yticklabels(coefs_sorted.index)
    ax_coef.tick_params(axis='x', labelbottom=False)

    # --- Bottom: contribution bars ---
    ax_delta.bar(x, contrib_sorted['delta'].values)
    ax_delta.set_xticks(x)
    ax_delta.set_xticklabels(feature_order, rotation=90, fontsize=TICKFONTSIZE)
    ax_delta.set_ylabel(r'$\Delta$ acc.')
    ax_delta.axhline(0, color='k', linewidth=0.5)

    return fig


def _draw_traces(ax, df_cell, contrasts, shade_map):
    """Draw one panel's mean traces, one line per contrast, shaded ± SEM.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes drawn on; not styled or scaled here.
    df_cell : pd.DataFrame
        One (event, feedbackType) cell of the aggregate frame, with ``time``,
        ``mean``, ``sem`` and ``contrast``.
    contrasts : list
        Contrast levels drawn, in plotting order.
    shade_map : dict
        Contrast level -> line color.
    """
    for contrast in contrasts:
        df_c = df_cell[df_cell['contrast'] == contrast].sort_values('time')
        if len(df_c) == 0:
            continue
        time_vals = df_c['time'].values
        mean_trace = df_c['mean'].values
        sem_trace = df_c['sem'].values
        color = shade_map.get(contrast, 'gray')
        ax.plot(time_vals, mean_trace, color=color, linewidth=1.5,
                label=f'{contrast}')
        ax.fill_between(time_vals, mean_trace - sem_trace,
                        mean_trace + sem_trace, color=color, alpha=0.2)


def plot_mean_response_traces(agg_df, target_nm, count_label=None,
                              inset=False):
    """Aggregated peri-event response traces for one target-NM.

    A pure drawer: the means and SEMs are drawn exactly as given, with no
    aggregation, correction or trial selection of its own. Produce ``agg_df``
    with :func:`iblnm.analysis.aggregate_conditions`, grouping on
    ``event``, ``contrast``, ``feedbackType`` and ``time``.

    Layout: 2 rows (reward top, omission bottom) × n_events columns.
    Each panel has one line per contrast level, colored by the NM colormap,
    shaded ± SEM.

    Parameters
    ----------
    agg_df : pd.DataFrame
        Aggregated traces for a single target-NM, one row per
        (event, contrast, feedbackType, time), with columns ``mean``, ``sem``
        and ``n``.
    target_nm : str
        Target-NM label; used for the title and the colormap.
    count_label : str or None
        Text annotated on the top-right panel describing the data the
        aggregate was taken over (e.g. '120 trials\\n8 sessions\\n4 mice');
        the caller holds those counts.
    inset : bool
        Add an inset to every panel redrawing the same traces on their own
        y-scale, for cohorts whose responses are too small to read at the
        shared limits.

    Returns
    -------
    plt.Figure
    """
    present = set(agg_df['event'].unique())
    events = [e for e in RESPONSE_EVENTS if e in present]
    # Append any events not in the canonical order
    events += sorted(present - set(RESPONSE_EVENTS))
    n_events = max(len(events), 1)
    feedback_types = [1, -1]
    fb_labels = {1: 'Reward', -1: 'Omission'}
    contrasts = sorted(agg_df['contrast'].unique())

    # Build color map: NM colormap with shades per contrast level
    nm = target_nm.split('-')[-1]
    cmap = NM_CMAPS.get(nm, NM_CMAPS['DA'])
    n_levels = len(ANALYSIS_CONTRASTS)
    shade_map = {c: cmap(0.3 + 0.7 * i / (n_levels - 1))
                 for i, c in enumerate(ANALYSIS_CONTRASTS)}

    fig, axes = plt.subplots(2, n_events,
                             figsize=(4 * n_events, 6),
                             sharey=True, squeeze=False)

    cells = {}
    for col, event in enumerate(events):
        for row, fb in enumerate(feedback_types):
            ax = axes[row, col]
            df_cell = agg_df[
                (agg_df['event'] == event)
                & (agg_df['feedbackType'] == fb)
            ]
            cells[row, col] = df_cell

            _draw_traces(ax, df_cell, contrasts, shade_map)

            ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
            ax.set_ylim(-1.5, 3)

            # Shaded response window
            ax.axvspan(*RESPONSE_MAGNITUDE_WINDOW, alpha=0.12, color='gray',
                       zorder=0)

            if row == 1:
                ax.set_xlabel('Time (s)')
            if col == 0:
                ax.set_ylabel(fb_labels[fb])
            if row == 0:
                event_label = event.replace('_times', '')
                ax.set_title(event_label)

    # Each inset redraws its panel's traces and autoscales, leaving the shared
    # limits to the panel underneath.
    if inset:
        for (row, col), df_cell in cells.items():
            ax_inset = axes[row, col].inset_axes([0.62, 0.58, 0.36, 0.38])
            _draw_traces(ax_inset, df_cell, contrasts, shade_map)
            ax_inset.axvline(0, color='gray', linewidth=0.5, linestyle='--')
            ax_inset.tick_params(labelsize=TICKFONTSIZE * 0.7)
            ax_inset.set_xticklabels([])

    # Legend on first axis
    axes[0, 0].legend(title='Contrast', fontsize=TICKFONTSIZE,
                      title_fontsize=TICKFONTSIZE, loc='upper left')

    if count_label is not None:
        axes[0, -1].annotate(
            count_label,
            xy=(0.95, 0.92), xycoords='axes fraction',
            fontsize=TICKFONTSIZE, ha='right', va='top', color='k',
        )

    fig.suptitle(target_nm, fontsize=LABELFONTSIZE)
    fig.tight_layout()
    return fig



# =============================================================================
# Wheel Kinematics LMM Plots
# =============================================================================

_DV_LABELS = {
    'reaction_time': 'Reaction time',
    'movement_time': 'Movement time',
    'peak_velocity': 'Peak velocity',
}


# Saturated movement models, in bar order: which model each name denotes, its
# display label, and its colour. The config key names the *dropped* predictor,
# so 'movement' (timing dropped) is the contrast-family model and 'contrast'
# (contrast dropped) is the movement-family model.
_MOVEMENT_R2_BARS = [
    ('movement', 'contrast-family', '#888888'),
    ('contrast', 'movement-family', '#1f77b4'),
    ('full', 'full', '#d62728'),
]


def plot_movement_r2_bars(summary_df):
    """In-sample marginal R² of the three nested movement models, per target-NM.

    Reads an in-sample R² frame: for each movement variable, three nested
    models — ``full`` (the per-event task base extended with the movement
    predictor at 2nd order), ``contrast`` (contrast dropped, the movement-family
    model), and ``movement`` (the predictor dropped, the task base). One panel
    per movement
    variable; each target-NM gets three bars. Heights read two ways:
    contrast-family vs. movement-family = which predictor explains more; full
    vs. either = added value.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Long-form, one row per (target_NM, movement var, model). Required
        columns: ``target_NM``, ``movement_var``, ``name`` (``full``/
        ``contrast``/``movement``), ``marginal_r2``. Any ``event`` column is
        ignored; the script passes one event's rows per figure.

    Returns
    -------
    plt.Figure
    """
    movement_vars = sorted(summary_df['movement_var'].unique()) if len(summary_df) else []
    n_panels = max(len(movement_vars), 1)

    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels + 1, 4),
                             sharey=True, layout='constrained')
    if n_panels == 1:
        axes = [axes]

    _title = ('In-sample marginal R² (full-data fit)\n'
              'full: per-event task base + <movement> (2nd-order)')
    if len(summary_df) == 0:
        fig.suptitle(_title, fontsize=LABELFONTSIZE)
        return fig

    targets = sorted(summary_df['target_NM'].unique(),
                     key=lambda x: TARGETNM2POSITION.get(x, 999))
    bar_w = 0.8 / len(_MOVEMENT_R2_BARS)

    for ax, mvar in zip(axes, movement_vars):
        df_mv = summary_df[summary_df['movement_var'] == mvar]
        for i, tnm in enumerate(targets):
            df_tnm = df_mv[df_mv['target_NM'] == tnm].set_index('name')
            for k, (name, label, color) in enumerate(_MOVEMENT_R2_BARS):
                if name not in df_tnm.index:
                    continue
                offset = (k - (len(_MOVEMENT_R2_BARS) - 1) / 2) * bar_w
                ax.bar(i + offset, df_tnm.loc[name, 'marginal_r2'],
                       width=bar_w, color=color,
                       label=label if i == 0 else '')
        ax.set_xticks(range(len(targets)))
        ax.set_xticklabels(targets, rotation=30, ha='right', fontsize=TICKFONTSIZE)
        ax.axhline(0, ls='--', color='gray', lw=0.5)
        ax.set_title(mvar)

    axes[0].set_ylabel('Marginal R²')
    axes[-1].legend(frameon=False, fontsize=TICKFONTSIZE,
                    loc='upper left', bbox_to_anchor=(1, 1))
    fig.suptitle(_title, fontsize=LABELFONTSIZE)
    return fig


def plot_cohort_cca_summary(cohort_results, cross_projections, weight_sims,
                            fig=None):
    """Three-panel summary of per-cohort CCA results.

    Parameters
    ----------
    cohort_results : dict[str, CCAResult]
        Per-cohort CCA fits.
    cross_projections : pd.DataFrame
        Columns: ``data_cohort``, ``weight_cohort``, ``correlation``.
    weight_sims : pd.DataFrame
        Columns: ``cohort_a``, ``cohort_b``, ``neural_cosine``,
        ``behavioral_cosine``.
    fig : matplotlib.figure.Figure, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    targets = sorted(cohort_results.keys())

    if fig is None:
        fig, axes = plt.subplots(1, 6, figsize=(28, 4))
    else:
        axes = fig.subplots(1, 6)

    # Panel 1: per-cohort canonical correlations
    ax = axes[0]
    colors = [TARGETNM_COLORS.get(t, 'gray') for t in targets]
    corrs = [cohort_results[t].correlations[0] for t in targets]
    ax.bar(range(len(targets)), corrs, color=colors)
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(targets, rotation=45, ha='right')
    ax.set_ylim(0, 1)
    ax.set_ylabel('Canonical correlation')
    ax.set_title('Per-cohort CC1')

    # Annotate significance stars
    for i, t in enumerate(targets):
        pv = cohort_results[t].p_values
        if pv is not None:
            stars = _pval_to_stars(pv[0])
            if stars:
                ax.text(i, corrs[i] + 0.01, stars,
                        ha='center', va='bottom', fontsize=TICKFONTSIZE)

    # Panel 2: cross-projection heatmap
    ax = axes[1]
    matrix = cross_projections.pivot(
        index='data_cohort', columns='weight_cohort', values='correlation')
    matrix = matrix.reindex(index=targets, columns=targets)
    im = ax.imshow(matrix.values, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(targets, rotation=45, ha='right')
    ax.set_yticks(range(len(targets)))
    ax.set_yticklabels(targets)
    ax.set_xlabel('Weight source')
    ax.set_ylabel('Data source')
    ax.set_title('Cross-projection')
    fig.colorbar(im, ax=ax, shrink=0.8)

    # Panel 3: delta-r cross-projection (relative to within-cohort diagonal)
    ax = axes[2]
    diag = np.diag(matrix.values)  # within-cohort correlations
    delta_matrix = matrix.values - diag[:, np.newaxis]  # subtract row baseline (own data, different weights)
    im2 = ax.imshow(delta_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(targets, rotation=45, ha='right')
    ax.set_yticks(range(len(targets)))
    ax.set_yticklabels(targets)
    ax.set_xlabel('Weight source')
    ax.set_ylabel('Data source')
    ax.set_title(r'$\Delta r$ (vs within-cohort)')
    fig.colorbar(im2, ax=ax, shrink=0.8)

    # Panel 4: weight profiles (neural + behavioral combined)
    _plot_weight_heatmap_pair(cohort_results, targets, axes[3])

    # Panels 5-6: cosine similarity heatmaps (neural, behavioral)
    cohorts = sorted(weight_sims['cohort_a'].unique())
    for ax, col, title in [
        (axes[4], 'neural_cosine', 'Neural cosine similarity'),
        (axes[5], 'behavioral_cosine', 'Behavioral cosine similarity'),
    ]:
        sim_matrix = weight_sims.pivot(
            index='cohort_a', columns='cohort_b', values=col)
        sim_matrix = sim_matrix.reindex(index=cohorts, columns=cohorts)
        im = ax.imshow(sim_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xticks(range(len(cohorts)))
        ax.set_xticklabels(cohorts, rotation=45, ha='right')
        ax.set_yticks(range(len(cohorts)))
        ax.set_yticklabels(cohorts)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.tight_layout()
    return fig


def _plot_weight_heatmap_pair(cohort_results, targets, ax):
    """Draw a split heatmap of neural|behavioral CC1 weights on a single axis.

    Features on y-axis (neural on top, behavioral on bottom, separated by a
    line), cohorts on x-axis. Shared diverging colormap.
    """
    neural_names = cohort_results[targets[0]].x_weights.index.tolist()
    behav_names = cohort_results[targets[0]].y_weights.index.tolist()

    neural_mat = np.column_stack(
        [cohort_results[t].x_weights['CC1'].values for t in targets])
    behav_mat = np.column_stack(
        [cohort_results[t].y_weights['CC1'].values for t in targets])

    # Stack: neural rows on top, behavioral below
    combined = np.vstack([neural_mat, behav_mat])
    all_names = neural_names + behav_names
    n_neural = len(neural_names)

    vmax = np.max(np.abs(combined))
    im = ax.imshow(combined, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                   aspect='auto')

    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(targets, rotation=45, ha='right')
    ax.set_yticks(range(len(all_names)))
    ax.set_yticklabels([_coef_label(n) for n in all_names])

    # Separator between neural and behavioral
    ax.axhline(n_neural - 0.5, color='black', linewidth=1.5)

    # Label the two sections
    ax.text(-0.7, (n_neural - 1) / 2, 'Neural', ha='right', va='center',
            fontsize=TICKFONTSIZE, fontweight='bold', transform=ax.get_yaxis_transform())
    ax.text(-0.7, n_neural + (len(behav_names) - 1) / 2, 'Behav.',
            ha='right', va='center', fontsize=TICKFONTSIZE, fontweight='bold',
            transform=ax.get_yaxis_transform())

    ax.set_title('CC1 weight profiles')
    ax.figure.colorbar(im, ax=ax, shrink=0.8, label='Weight')


def _draw_rt_violins(df, ax=None, rt_range=None):
    """Horizontal violin plots of response time by contrast, one offset per target-NM.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``response_time``, ``contrast``, and ``target_NM`` columns.
        Contrasts should be absolute (unsigned).
    ax : plt.Axes, optional
        Axes to draw on. Created if not provided.
    rt_range : tuple of float, optional
        ``(min, max)`` response times (seconds) used to fix the log-spaced
        x-axis ticks and limits. When given, the limits are pinned to these
        values rather than autoscaled from ``df`` — pass a shared range across
        panels to make their x-axes consistent. When None, the range is taken
        from ``df``.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    if df.empty:
        ax.set_xlabel('Response time (s)')
        ax.set_ylabel('Contrast (%)')
        return ax

    contrasts = sorted(df['contrast'].unique())
    target_nms = [t for t in TARGETNM_COLORS if t in df['target_NM'].unique()]
    n_targets = len(target_nms)

    # Spread offsets symmetrically around each integer contrast tick
    box_height = 0.6 / max(n_targets, 1)
    offsets = np.linspace(-0.25, 0.25, n_targets) if n_targets > 1 else np.array([0.0])

    for i, target_nm in enumerate(target_nms):
        color = TARGETNM_COLORS.get(target_nm, 'gray')
        df_t = df[df['target_NM'] == target_nm]

        for j, contrast in enumerate(contrasts):
            rt_vals = df_t.loc[df_t['contrast'] == contrast, 'response_time'].dropna().values
            if len(rt_vals) == 0:
                continue
            # Log-transform before KDE so violin shape is correct in log space
            log_vals = np.log10(rt_vals[rt_vals > 0])
            if len(log_vals) == 0:
                continue
            y_pos = j + offsets[i]
            vp = ax.violinplot(
                log_vals,
                positions=[y_pos],
                orientation='horizontal',
                widths=box_height * 0.8,
                showmedians=True,
                showextrema=False,
            )
            for violin in vp['bodies']:
                violin.set_color(color)
                violin.set_linewidth(0.8)
            vp['cmedians'].set_color(color)

    # Linear axis with log-formatted tick labels
    if rt_range is not None:
        rt_lo, rt_hi = rt_range
    else:
        positive = df.loc[df['response_time'] > 0, 'response_time'].dropna()
        rt_lo, rt_hi = (positive.min(), positive.max()) if len(positive) else (None, None)
    if rt_lo is not None:
        log_min = np.floor(np.log10(rt_lo))
        log_max = np.ceil(np.log10(rt_hi))
        tick_powers = np.arange(log_min, log_max + 1)
        ax.set_xticks(tick_powers)
        ax.set_xticklabels([str(10 ** int(p)) if p >= 0 else str(round(10 ** p, 3))
                            for p in tick_powers])
        if rt_range is not None:
            ax.set_xlim(log_min, log_max)

    ax.set_yticks(range(len(contrasts)))
    ax.set_yticklabels([str(c) for c in contrasts])
    ax.set_ylim(-0.6, len(contrasts) - 0.4)
    ax.set_xlabel('Response time (s)')
    ax.set_ylabel('Contrast (%)')

    # Legend
    handles = [
        plt.Line2D([0], [0], color=TARGETNM_COLORS.get(t, 'gray'), linewidth=1.5, label=t)
        for t in target_nms
    ]
    if handles:
        ax.legend(handles=handles, fontsize=TICKFONTSIZE, loc='upper left')

    return ax


# =============================================================================
# Group-based task performance figures
# =============================================================================

def _assemble_rt_trials(magnitudes: pd.DataFrame | None) -> pd.DataFrame:
    """Build the trial table for RT violins from the magnitude frame.

    Takes one row per (eid, trial, target_NM) — the frame carries the
    trial-level columns beside the magnitude, so there is nothing to join —
    keeps trials with a recorded choice across all pLeft blocks, and restricts
    to the analysed target-NMs. Returns an empty frame (with the columns
    ``_draw_rt_violins`` expects) when the magnitudes are missing.

    Parameters
    ----------
    magnitudes : pd.DataFrame or None
        ``config.RESPONSE_MAGNITUDE_COLUMNS``, one row per recording x event x
        trial. ``None`` when the caller found no magnitudes parquet to read.

    Returns
    -------
    pd.DataFrame
        Columns include ``response_time``, ``contrast``, ``target_NM``.
    """
    if magnitudes is None:
        return pd.DataFrame(columns=['response_time', 'contrast', 'target_NM'])
    df_trial = magnitudes.drop_duplicates(
        subset=['eid', 'trial', 'target_NM']).query('choice != 0').copy()
    return df_trial[df_trial['target_NM'].isin(TARGETNMS_TO_ANALYZE)]


def plot_performance_grid(group, magnitudes=None, axes=None):
    """Grid of task-performance panels, one row per target-NM.

    Column 0 holds the 50-50 block psychometric curves (thin line per session,
    thick grand mean); column 1 holds response-time violins by contrast. Both
    columns reuse the per-target drawing helpers, so styling matches the
    standalone figures.

    Parameters
    ----------
    group : PhotometrySessionGroup
        Must have ``group.performance`` loaded.
    magnitudes : pd.DataFrame, optional
        The response magnitudes the RT column is drawn from. ``None`` draws the
        psychometric column alone.
    axes : np.ndarray of Axes, optional
        Shape (n_targets, 2). Created if None.

    Returns
    -------
    plt.Figure
    """
    rec_meta = (
        group.recordings[['eid', 'subject', 'target_NM']]
        .drop_duplicates()
    )
    df_psych = group.performance.merge(rec_meta, on='eid', how='inner')
    df_rt = _assemble_rt_trials(magnitudes)

    # Shared RT x-axis range so all rows align (None when no RT data)
    rt_positive = df_rt.loc[df_rt['response_time'] > 0, 'response_time']
    rt_range = (rt_positive.min(), rt_positive.max()) if len(rt_positive) else None

    targets = [t for t in TARGETNMS_TO_ANALYZE
               if t in df_psych['target_NM'].values]

    if axes is None:
        fig, axes = plt.subplots(
            len(targets), 2, figsize=(10, 4 * len(targets)), squeeze=False)
    else:
        fig = axes.flat[0].figure

    for i, target_nm in enumerate(targets):
        ax_psych = axes[i, 0]
        df_target = df_psych[df_psych['target_NM'] == target_nm]
        plot_psychometric_curves_50(df_target, target_nm=target_nm, ax=ax_psych)
        n_sessions = len(df_target)
        n_subjects = df_target['subject'].nunique()
        ax_psych.text(0.05, 0.85, f'{n_sessions} sessions\n{n_subjects} mice',
                      transform=ax_psych.transAxes)
        ax_psych.set_title(target_nm)

        _draw_rt_violins(df_rt[df_rt['target_NM'] == target_nm],
                         ax=axes[i, 1], rt_range=rt_range)

    fig.tight_layout()
    return fig


def plot_target_comparison(group, params, labels, axes=None):
    """Boxplots comparing target-NMs on performance parameters.

    Runs Kruskal-Wallis per parameter; draws post-hoc Mann-Whitney brackets
    when significant (p < 0.05, Bonferroni-corrected).

    Parameters
    ----------
    group : PhotometrySessionGroup
        Must have ``group.performance`` loaded.
    params : list of str
        Column names in performance to compare.
    labels : list of str
        Display labels (same length as params).
    axes : np.ndarray of Axes, optional
        Shape (1, len(params)). Created if None.

    Returns
    -------
    plt.Figure
    """
    from iblnm.analysis import kruskal_wallis_groups, pairwise_mannwhitney

    rec_meta = (
        group.recordings[['eid', 'subject', 'target_NM']]
        .drop_duplicates()
    )
    df = group.performance.merge(rec_meta, on='eid', how='inner')
    targets = [t for t in TARGETNMS_TO_ANALYZE if t in df['target_NM'].values]

    if axes is None:
        fig, axes = plt.subplots(
            1, len(params),
            figsize=(3.5 * len(params), 4),
            squeeze=False,
        )
    else:
        fig = axes.flat[0].figure

    for i, (col, label) in enumerate(zip(params, labels)):
        ax = axes[0, i]

        H, p_kw, groups_data = kruskal_wallis_groups(df, 'target_NM', col)

        # Keep only targets in TARGETNMS_TO_ANALYZE order
        target_names = [t for t in targets if t in groups_data]
        if len(target_names) < 2:
            ax.set_title(label)
            continue

        positions = list(range(len(target_names)))
        bp = ax.boxplot(
            [groups_data[t] for t in target_names],
            positions=positions, widths=0.5, patch_artist=True,
            showfliers=False,
        )
        for j, tnm in enumerate(target_names):
            color = TARGETNM_COLORS.get(tnm, 'gray')
            bp['boxes'][j].set_facecolor('none')
            bp['boxes'][j].set_edgecolor(color)
            bp['boxes'][j].set_linewidth(1.5)
            bp['medians'][j].set_color(color)
            bp['medians'][j].set_linewidth(2)
            for k in (2 * j, 2 * j + 1):
                bp['whiskers'][k].set_color(color)
                bp['caps'][k].set_color(color)

        # Per-subject mean ± 95% CI overlaid on each boxplot
        for j, tnm in enumerate(target_names):
            color = TARGETNM_COLORS.get(tnm, 'gray')
            df_tnm = df.loc[
                (df['target_NM'] == tnm) & df[col].notna(),
                ['subject', col],
            ]
            subj_stats = (
                df_tnm.groupby('subject')[col]
                .agg(['mean', 'sem', 'count'])
            )
            subj_stats = subj_stats[subj_stats['count'] > 0]
            n_subj = len(subj_stats)
            if n_subj == 0:
                continue
            offsets = np.linspace(-0.15, 0.15, n_subj) if n_subj > 1 else [0.0]
            for k, (subj, row) in enumerate(subj_stats.iterrows()):
                ci = 1.96 * row['sem'] if row['count'] > 1 else 0
                ax.errorbar(
                    j + offsets[k], row['mean'], yerr=ci,
                    fmt='o', color=color, markersize=3,
                    linewidth=0.8, capsize=0, zorder=5,
                )

        ax.set_xticks(positions)
        ax.set_xticklabels([t.split('-')[0] for t in target_names],
                           rotation=45, ha='right', fontsize=TICKFONTSIZE)
        ax.set_title(label)

        if p_kw >= 0.05:
            y_top = ax.get_ylim()[1]
            y_rng = y_top - ax.get_ylim()[0]
            ax.set_ylim(ax.get_ylim()[0], y_top + y_rng * 0.15)
            ax.text(0.5, y_top + y_rng * 0.05,
                    f'H={H:.1f}, p={p_kw:.3f} n.s.',
                    ha='center', va='bottom', fontsize=TICKFONTSIZE)
            continue

        # Post-hoc pairwise tests
        pairwise = pairwise_mannwhitney(
            {t: groups_data[t] for t in target_names})
        sig_pairs = [(target_names.index(a), target_names.index(b), p_corr)
                     for a, b, _, p_corr in pairwise if p_corr < 0.05]

        y_max = ax.get_ylim()[1]
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        step = y_range * 0.06
        for k, (a, b, p_corr) in enumerate(sig_pairs):
            y = y_max + step * (k + 0.5)
            stars = '***' if p_corr < 0.001 else '**' if p_corr < 0.01 else '*'
            ax.plot([a, a, b, b], [y - step * 0.15, y, y, y - step * 0.15],
                    color='black', linewidth=0.8)
            ax.text((a + b) / 2, y, stars, ha='center', va='bottom',
                    fontsize=TICKFONTSIZE)

        label_y = (y_max + step * (len(sig_pairs) + 0.5)
                   if sig_pairs else y_max)
        ax.set_ylim(ax.get_ylim()[0], label_y + step * 2.5)
        ax.text(0.5, label_y + step * 0.5,
                f'H={H:.1f}, p={p_kw:.1e}',
                ha='center', va='bottom', fontsize=TICKFONTSIZE)

    fig.tight_layout()
    return fig


def plot_encoding_prediction(fit, ax=None):
    """Plot the measured signal and the model prediction over time.

    Both traces are drawn over the valid grid samples (``fit.tvec[fit.valid]``).

    Parameters
    ----------
    fit : iblnm.analysis.EncodingFit
        A fitted encoding model.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; created if None.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the data and model traces.
    """
    if ax is None:
        _, ax = plt.subplots()
    times = fit.tvec[fit.valid]
    ax.plot(times, fit.target, label='data')
    ax.plot(times, fit.prediction, 'r', label='model')
    ax.set_xlabel('time (s)')
    ax.set_title(f'{fit.label}  (R$^2$ = {fit.r2:.3f})')
    ax.legend()
    return ax


def plot_encoding_kernels(fit, names, lags, sharey=True):
    """Plot the fitted lagged kernel for each named event block.

    Assumes the FIR (lagged) basis, where a block's back-transformed
    coefficients are the kernel itself (one value per lag). Sample lags are
    converted to seconds via the grid step ``dt = tvec[1] - tvec[0]``.

    Parameters
    ----------
    fit : iblnm.analysis.EncodingFit
        A fitted encoding model.
    names : list of str
        Event block names to plot (keys in ``fit.slices``).
    lags : np.ndarray
        Sample lags used to build the kernels.
    sharey : bool, optional
        Share the y-axis across panels (default True).

    Returns
    -------
    matplotlib.figure.Figure
        One panel per name, kernel amplitude vs lag in seconds.
    """
    fig, axes = plt.subplots(
        ncols=len(names), sharey=sharey, figsize=(3 * len(names), 3),
        squeeze=False)
    lag_seconds = lags * (fit.tvec[1] - fit.tvec[0])
    for ax, name in zip(axes[0], names):
        ax.plot(lag_seconds, fit.get_kernel(name))
        ax.set_title(name, fontsize='small')
        ax.axhline(0, linestyle=':', color='k', lw=1)
        ax.axvline(0, linestyle=':', color='k', lw=1)
        ax.set_xlabel('time (s)')
    fig.tight_layout()
    return fig


def plot_delta_r_squared(deltas, ax=None):
    """Horizontal bar chart of per-regressor ΔR² (largest contribution at top).

    Parameters
    ----------
    deltas : pd.Series
        ΔR² indexed by block name, sorted descending (as returned by
        ``PhotometrySession.delta_r_squared``).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; created if None.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with one horizontal bar per block.
    """
    if ax is None:
        _, ax = plt.subplots()
    # reverse so the largest drop sits at the top of the horizontal bars
    ax.barh(deltas.index[::-1], deltas.values[::-1])
    ax.axvline(0, linestyle=':', color='k', lw=1)
    ax.set_xlabel('ΔR² (drop when left out)')
    ax.figure.tight_layout()
    return ax


def plot_cosine_basis(n_basis=10, rcos_duration=2.5, rcos_nloffset=0.2,
                      dt=0.1, ax=None):
    """Plot the log-raised-cosine bump basis for the given parameters.

    Parameters
    ----------
    n_basis : int, optional
        Number of bumps.
    rcos_duration : float, optional
        Kernel window in seconds.
    rcos_nloffset : float, optional
        Log-warp offset in seconds.
    dt : float, optional
        Time-grid resolution in seconds.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; created if None.

    Returns
    -------
    matplotlib.axes.Axes
        The axes, with one line per bump.
    """
    if ax is None:
        _, ax = plt.subplots()
    basis = raised_cosine_basis(n_basis, rcos_duration, rcos_nloffset, dt)
    times = np.arange(basis.shape[0]) * dt
    ax.plot(times, basis)
    ax.set_xlabel('time after event (s)')
    ax.set_ylabel('basis weight')
    ax.set_title(
        f'raised-cosine basis (n_basis={n_basis}, dur={rcos_duration}s, '
        f'offset={rcos_nloffset}s)')
    return ax


def _summarize_state_posteriors(
    states: pd.DataFrame, bins: np.ndarray
) -> tuple[pd.Series, dict[int, np.ndarray]]:
    """Occupancy fractions and per-state posterior histograms for one mouse.

    Parameters
    ----------
    states : pandas.DataFrame
        One mouse's per-trial DDM-HMM state frame: integer ``map_state`` (NaN on
        trials dropped from the fit) and float posterior columns
        ``state_1``…``state_K`` (each in [0, 1], summing to ~1 on kept trials).
    bins : numpy.ndarray
        Histogram bin edges spanning [0, 1].

    Returns
    -------
    occupancy : pandas.Series
        Indexed by integer state label (1…K); fraction of MAP-assigned trials
        (non-NaN ``map_state``) in each state. Sums to 1.
    histograms : dict of int to numpy.ndarray
        Maps each state label to the counts of that state's posterior column
        across ``bins`` (dropped-trial NaNs excluded).
    """
    state_cols = sorted((c for c in states.columns if c.startswith('state_')),
                        key=lambda c: int(c.split('_')[1]))
    labels = [int(c.split('_')[1]) for c in state_cols]
    assigned = states['map_state'].dropna().astype(int)
    occupancy = assigned.value_counts().reindex(labels, fill_value=0) / len(assigned)
    histograms = {label: np.histogram(states[col].dropna(), bins=bins)[0]
                  for label, col in zip(labels, state_cols)}
    return occupancy, histograms


# Shared layout for the per-mouse "one row per mouse" goal figures.
ROW_HEIGHT = 2.4   # inches per mouse row, so axis heights match across figures
POINT_ALPHA = 0.5  # alpha for empirical data-point markers
DOT_JITTER = 0.12  # half-width, in x units, of the per-session dot spread
OUTCOME_ALPHA = {'correct': 1.0, 'incorrect': 0.5}  # violin body opacity by outcome
MEASURE_DODGE = 0.18  # x offset of each outcome's violin from its state position


def plot_state_posterior_dwell(
    states_by_mouse: dict[str, pd.DataFrame],
    dwell_by_mouse: dict[str, pd.DataFrame],
    n_bins: int = 20,
) -> plt.Figure:
    """Per-state posterior crispness and dwell times, one mouse per row.

    Each mouse gets two axes. Left: a step histogram of every state's posterior
    column (``state_1``…``state_K``) on [0, 1] — crisp assignments concentrate
    mass near 0 and 1 — with an inset bar of each state's MAP occupancy
    (:func:`_summarize_state_posteriors`). Right: a step histogram of run-lengths
    (dwell times in trial units) per state. States are colored consistently
    within a mouse by ``plt.cm.tab10``; labels are unaligned across mice.

    Parameters
    ----------
    states_by_mouse : dict of str to pandas.DataFrame
        Maps each subject to its per-trial state frame (``map_state`` plus
        ``state_1``…``state_K`` posteriors; NaN on trials dropped from the fit),
        from :meth:`PhotometrySession.load_states`.
    dwell_by_mouse : dict of str to pandas.DataFrame
        Maps each subject to its pooled dwell-time frame with columns
        ``['state', 'length']`` (one row per run), from
        :func:`iblnm.analysis.state_dwell_times` applied per eid.
    n_bins : int, optional
        Histogram bins for each panel (default 20).

    Returns
    -------
    matplotlib.figure.Figure
        ``len(states_by_mouse)`` rows by 2 columns (posterior, dwell).
    """
    bins = np.linspace(0, 1, n_bins + 1)
    mice = list(states_by_mouse)
    fig, axes = plt.subplots(len(mice), 2, figsize=(7, ROW_HEIGHT * len(mice)),
                             squeeze=False, layout='constrained')
    for (ax_post, ax_dwell), mouse in zip(axes, mice):
        occupancy, histograms = _summarize_state_posteriors(
            states_by_mouse[mouse], bins)
        state_colors = plt.cm.tab10(np.arange(len(occupancy)))
        for (label, counts), color in zip(histograms.items(), state_colors):
            ax_post.stairs(counts, bins, color=color, label=f'state {label}')
        inset = ax_post.inset_axes([0.08, 0.62, 0.3, 0.32])
        inset.bar(range(len(occupancy)), occupancy.to_numpy(), color=state_colors)
        inset.set_ylim(0, 1)
        inset.set_xticks([])
        inset.tick_params(labelsize=6)
        inset.set_title('occupancy', fontsize=6)
        ax_post.set_xlabel('posterior probability', fontsize=8)
        ax_post.set_ylabel('trials', fontsize=8)
        ax_post.set_title(mouse, fontsize=9)
        ax_post.tick_params(labelsize=7)
        ax_post.legend(fontsize=6, frameon=False)

        dwell = dwell_by_mouse[mouse]
        dwell_bins = np.linspace(0, dwell['length'].max(), n_bins + 1)
        by_state = dwell.groupby('state')['length']
        dwell_colors = plt.cm.tab10(np.arange(by_state.ngroups))
        for (label, lengths), color in zip(by_state, dwell_colors):
            counts, _ = np.histogram(lengths, bins=dwell_bins)
            ax_dwell.stairs(counts, dwell_bins, color=color, label=f'state {label}')
        ax_dwell.set_xlabel('dwell time (trials)', fontsize=8)
        ax_dwell.set_ylabel('runs', fontsize=8)
        ax_dwell.set_title(mouse, fontsize=9)
        ax_dwell.tick_params(labelsize=7)
    return fig


def _draw_outcome_violins(ax, trials: pd.DataFrame, column: str) -> None:
    """Draw one axis of :func:`plot_state_measures`: a state's two outcomes.

    Each state sits at an integer x position, with its correct trials violined at
    ``position - MEASURE_DODGE`` and its incorrect trials at
    ``position + MEASURE_DODGE``, and that group's per-session medians scattered
    over it. Trials with a NaN in ``column`` are dropped first, so a measure that
    is entirely NaN leaves the axis empty rather than raising.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on. Its x ticks are set to the state labels.
    trials : pandas.DataFrame
        One mouse's fit-only trials, with columns ``['state', 'eid', 'outcome']``
        plus ``column``.
    column : str
        The measure column to plot on y.
    """
    trials = trials.dropna(subset=[column])
    states = sorted(trials['state'].unique())
    positions = np.arange(len(states))
    ax.set_xticks(positions, [str(state) for state in states])
    if not states:
        return
    state_colors = plt.cm.tab10(positions)
    # OUTCOME_ALPHA's order puts correct left of incorrect within each state.
    for (outcome, alpha), dodge in zip(OUTCOME_ALPHA.items(),
                                       (-MEASURE_DODGE, MEASURE_DODGE)):
        # Filling absent states keeps the group list aligned with `positions`
        # when an outcome misses a state entirely; violinplot skips empty groups.
        by_state = dict(tuple(trials[trials['outcome'] == outcome]
                              .groupby('state')))
        groups = [by_state.get(state, trials.iloc[:0]) for state in states]
        violinplot(ax, [group[column].to_numpy() for group in groups],
                   positions=positions + dodge, colors=state_colors,
                   remove_outliers=False, show_outliers=False,
                   alpha=alpha, widths=0.3)
        for position, group, color in zip(positions, groups, state_colors):
            medians = group.groupby('eid')[column].median()
            # Evenly spaced offsets, endpoints excluded: deterministic, and a
            # lone session lands on the violin's center.
            offsets = np.linspace(-DOT_JITTER, DOT_JITTER, len(medians) + 2)[1:-1]
            ax.scatter(position + dodge + offsets, medians.to_numpy(), s=8,
                       color=color, alpha=POINT_ALPHA, zorder=3)


def plot_state_measures(
    measures_by_mouse: dict[str, pd.DataFrame], measures: Mapping[str, str]
) -> plt.Figure:
    """Per-state NM measure distributions split by outcome, one mouse per row.

    Each axis is one mouse and one measure: a violin of every trial's value per
    MAP state, split into correct (full opacity, left) and incorrect (half
    opacity, right) trials, with that group's per-session medians overlaid as
    dots. The dots matter because the measures are not re-centered per session —
    a state's shift can be carried by a single session, and that shows up as one
    outlying dot rather than a wider violin. States are colored consistently
    within a mouse by ``plt.cm.tab10``; each mouse is fit separately, so state
    labels carry no meaning across mice.

    y limits are per-axis: a pre-stimulus baseline is an absolute level in
    session-SD units, while an evoked response is a difference from a pre-event
    baseline, and a shared scale would flatten whichever is smaller. A group with
    fewer than 10 trials is drawn by :func:`violinplot` as an open-circle scatter
    of its raw values instead of a violin — splitting by outcome makes that more
    common than pooling would.

    Parameters
    ----------
    measures_by_mouse : dict of str to pandas.DataFrame
        Maps each subject to its fit-only, non-no-go trials with columns
        ``['state', 'eid', 'outcome']`` plus one column per measure. ``outcome``
        is ``'correct'`` or ``'incorrect'``. NaN values are dropped per measure,
        so a mouse missing one measure entirely still draws the others.
    measures : Mapping of str to str
        Measure column to y-axis label. Iteration order fixes the figure's
        left-to-right column order.

    Returns
    -------
    matplotlib.figure.Figure
        ``len(measures_by_mouse)`` rows by ``len(measures)`` columns, states along
        x in ascending label order, the measure labels titling the top row, the
        mouse names on the first column's y axes, and the outcome legend on the
        top-left axis.
    """
    mice = list(measures_by_mouse)
    fig, axes = plt.subplots(
        len(mice), len(measures),
        figsize=(7 * len(measures), ROW_HEIGHT * len(mice)),
        squeeze=False, layout='constrained')
    for row, mouse in zip(axes, mice):
        for ax, column in zip(row, measures):
            _draw_outcome_violins(ax, measures_by_mouse[mouse], column)
            ax.set_xlabel('state', fontsize=8)
            ax.tick_params(labelsize=7)
        # One label per grid edge: the row is a mouse, the column a measure.
        row[0].set_ylabel(mouse, fontsize=9)
    for ax, label in zip(axes[0], measures.values()):
        ax.set_title(label, fontsize=9)
    axes[0][0].legend(handles=[Line2D([], [], color='gray', alpha=alpha,
                                      label=outcome)
                               for outcome, alpha in OUTCOME_ALPHA.items()],
                      fontsize=6, frameon=False)
    return fig


_CHRONO_OUTCOME_STYLE = {
    'correct': {'linestyle': '-', 'fill': True},
    'incorrect': {'linestyle': '--', 'fill': False},
}


def plot_state_psychometric_chronometric(
    curves_by_mouse: dict[str, dict[str, pd.DataFrame]],
    contrast_range: tuple = (-100, 100),
) -> plt.Figure:
    """Per-state psychometric + chronometric curves, one mouse per row.

    Each mouse gets two axes: a psychometric panel (P(rightward choice) vs
    signed contrast) and a chronometric panel (median RT vs signed contrast).
    Every inferred state is overlaid with a consistent colour within a mouse
    (``plt.cm.tab10``, keyed by sorted state label). The chronometric panel draws
    plain lines through the empirical medians, split by trial outcome — correct
    (filled markers, solid line) vs incorrect (open markers, dashed line) — and by
    stimulus side, so the left and right lines are not connected across zero
    contrast (two points at 0). The
    psychometric panel adds a fitted overlay (``psychofit.erf_psycho_2gammas``)
    where its parameters are finite (low-trial states return NaN fits). State
    labels are unaligned across mice.

    Parameters
    ----------
    curves_by_mouse : dict of str to dict of str to pandas.DataFrame
        Maps each subject to a ``{'psychometric', 'chronometric'}`` pair of long
        frames assembled by the orchestration script. The psychometric frame has
        columns ``['state', 'signed_contrast', 'p_right', 'bias', 'threshold',
        'lapse_left', 'lapse_right']`` (fit params constant within a state); the
        chronometric frame has ``['state', 'outcome', 'side', 'signed_contrast',
        'median_rt']`` (``outcome`` in ``{'correct', 'incorrect'}``, ``side`` in
        ``{'left', 'right'}``).
    contrast_range : tuple of float, optional
        Signed-contrast span (percent) over which the psychometric overlay is
        drawn (default ``(-100, 100)``).

    Returns
    -------
    matplotlib.figure.Figure
        Grid of ``len(curves_by_mouse)`` rows by 2 columns (psychometric,
        chronometric).
    """
    import psychofit as psy

    mice = list(curves_by_mouse)
    fig, axes = plt.subplots(len(mice), 2, figsize=(7, ROW_HEIGHT * len(mice)),
                             squeeze=False, layout='constrained')
    grid = np.linspace(contrast_range[0], contrast_range[1], 200)
    for (ax_psych, ax_chrono), mouse in zip(axes, mice):
        psych = curves_by_mouse[mouse]['psychometric']
        chrono = curves_by_mouse[mouse]['chronometric']
        states = sorted(set(psych['state']) | set(chrono['state']))
        state_colors = dict(zip(states, plt.cm.tab10(np.arange(len(states)))))

        for state, points in psych.groupby('state'):
            color = state_colors[state]
            ax_psych.scatter(points['signed_contrast'], points['p_right'],
                             color=color, alpha=POINT_ALPHA, label=f'state {state}')
            params = points[['bias', 'threshold',
                             'lapse_right', 'lapse_left']].iloc[0].to_numpy()
            if np.all(np.isfinite(params)):
                ax_psych.plot(grid, psy.erf_psycho_2gammas(params, grid),
                              color=color, linewidth=1.5)

        # One plain line per (state, outcome, side); sides are drawn separately so
        # they are not connected across zero contrast (two dots at 0).
        for (state, outcome, _), points in chrono.groupby(
                ['state', 'outcome', 'side']):
            color = state_colors[state]
            style = _CHRONO_OUTCOME_STYLE[outcome]
            points = points.sort_values('signed_contrast')
            ax_chrono.plot(points['signed_contrast'], points['median_rt'],
                           color=color, alpha=POINT_ALPHA, marker='o',
                           markersize=3, linestyle=style['linestyle'],
                           markerfacecolor=color if style['fill'] else 'none')

        ax_psych.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        ax_psych.set_xlabel('signed contrast (%)', fontsize=8)
        ax_psych.set_ylabel('P(choose right)', fontsize=8)
        ax_psych.set_title(mouse, fontsize=9)
        ax_psych.tick_params(labelsize=7)
        ax_psych.legend(fontsize=6, frameon=False)
        ax_chrono.set_xlabel('signed contrast (%)', fontsize=8)
        ax_chrono.set_ylabel('median RT (s)', fontsize=8)
        ax_chrono.set_title(mouse, fontsize=9)
        ax_chrono.tick_params(labelsize=7)

    outcome_handles = [
        Line2D([], [], color='0.4', marker='o', linestyle=style['linestyle'],
               markerfacecolor='0.4' if style['fill'] else 'none', label=outcome)
        for outcome, style in _CHRONO_OUTCOME_STYLE.items()]
    axes[0][1].legend(handles=outcome_handles, fontsize=6, frameon=False)
    return fig


_MOUSE_MARKERS = ('o', 's', '^', 'D', 'v', 'P', 'X', '*')


def _mouse_markers(mice: list) -> dict:
    """Map each mouse to a distinct marker shape (mouse = shape in the scatters)."""
    return dict(zip(mice, itertools.cycle(_MOUSE_MARKERS)))


def _scatter_by_state_and_mouse(ax, x, y, states, mouse_labels, markers) -> None:
    """Scatter (x, y) with color = within-mouse state and marker = mouse.

    Color follows the shared per-state scheme used by every other goal figure
    (state ``s`` in 1…K -> ``plt.cm.tab10(s - 1)``); each mouse is drawn as a
    single scatter call so it gets one distinct marker shape.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw into.
    x, y : array-like, shape (n_states,)
        Point coordinates.
    states : array-like of int, shape (n_states,)
        Within-mouse state label (1…K) per point; sets color.
    mouse_labels : array-like of str, shape (n_states,)
        Mouse per point; sets marker.
    markers : dict of str to str
        Mouse -> marker shape (from :func:`_mouse_markers`).
    """
    x, y = np.asarray(x), np.asarray(y)
    states = np.asarray(states)
    mouse_labels = np.asarray(mouse_labels)
    for mouse, marker in markers.items():
        mask = mouse_labels == mouse
        ax.scatter(x[mask], y[mask], marker=marker,
                   color=plt.cm.tab10(states[mask] - 1))


def _state_mouse_handles(states, markers):
    """Legend handles for the state-color and mouse-marker encodings."""
    state_handles = [Line2D([], [], marker='o', linestyle='none',
                            color=plt.cm.tab10(s - 1), label=f'state {s}')
                     for s in states]
    mouse_handles = [Line2D([], [], marker=marker, linestyle='none', color='0.4',
                            label=mouse) for mouse, marker in markers.items()]
    return state_handles, mouse_handles


def _add_state_mouse_legends(fig, states, markers) -> None:
    """Add two outside legends: state->color and mouse->marker shape."""
    state_handles, mouse_handles = _state_mouse_handles(states, markers)
    fig.legend(handles=state_handles, title='state', frameon=False,
               fontsize=TICKFONTSIZE, loc='outside right upper')
    fig.legend(handles=mouse_handles, title='mouse', frameon=False,
               fontsize=TICKFONTSIZE, loc='outside right lower')


def plot_state_param_scatter(
    params_df: pd.DataFrame, params: tuple = ('B', 'k', 'a0')
) -> plt.Figure:
    """3D scatter of per-state DDM parameters; color = state, marker = mouse.

    One point per (mouse, state) in the space of the three ``params`` (default
    B, k, a0). Within-mouse state sets the color (shared ``plt.cm.tab10`` scheme,
    matching every other goal figure) and mouse sets the marker shape, so a
    recurring state signature reads as a cluster of one color across marker
    shapes. State labels are unaligned across mice.

    Parameters
    ----------
    params_df : pandas.DataFrame
        One row per (mouse, state), with ``mouse`` and ``state`` columns and the
        columns named in ``params`` (per-state DDM parameters from
        ``all_mice_bestK_params.csv``).
    params : tuple of str, optional
        The three parameter columns forming the x/y/z axes (default
        ``('B', 'k', 'a0')``).

    Returns
    -------
    matplotlib.figure.Figure
        One 3D axes, with outside state-color and mouse-marker legends.
    """
    x_param, y_param, z_param = params
    mice = sorted(params_df['mouse'].unique())
    markers = _mouse_markers(mice)
    states = sorted(params_df['state'].unique())

    fig = plt.figure(figsize=(6, 6), layout='constrained')
    ax = fig.add_subplot(projection='3d')
    for mouse, marker in markers.items():
        sub = params_df[params_df['mouse'] == mouse]
        ax.scatter(sub[x_param], sub[y_param], sub[z_param], marker=marker,
                   color=plt.cm.tab10(sub['state'].to_numpy() - 1),
                   alpha=POINT_ALPHA, depthshade=False)
    ax.set_xlabel(x_param)
    ax.set_ylabel(y_param)
    ax.set_zlabel(z_param)

    _add_state_mouse_legends(fig, states, markers)
    return fig


def plot_state_pca(
    scores: np.ndarray, mouse_labels: Iterable, states: Iterable,
    loadings: np.ndarray, feature_names: Iterable,
) -> plt.Figure:
    """PC1xPC2 scatter of per-state behavioral features, with loading heatmaps.

    The central axes scatters one point per state by its first two
    principal-component scores (from :func:`iblnm.analysis.pca_2d`); within-mouse
    state sets the color (shared ``plt.cm.tab10`` scheme) and mouse sets the
    marker shape. Two loading heatmaps sit beside the axis each PC defines: PC1's
    feature weights run horizontally under the x-axis, PC2's run vertically beside
    the y-axis, both on a symmetric diverging scale centered at 0.

    Parameters
    ----------
    scores : numpy.ndarray, shape (n_states, 2)
        PC1/PC2 coordinates per state.
    mouse_labels : iterable of str, length n_states
        Mouse per state; sets marker shape.
    states : iterable of int, length n_states
        Within-mouse state label (1…K) per point; sets color.
    loadings : numpy.ndarray, shape (n_features, 2)
        Feature weights for PC1 (column 0) and PC2 (column 1).
    feature_names : iterable of str, length n_features
        Feature labels, in ``loadings`` row order.

    Returns
    -------
    matplotlib.figure.Figure
        The scatter plus the two aligned loading heatmaps.
    """
    mouse_labels = np.asarray(mouse_labels)
    loadings = np.asarray(loadings)
    feature_names = list(feature_names)
    markers = _mouse_markers(sorted(set(mouse_labels)))
    vlim = np.abs(loadings).max()

    # Explicit square layout (square figure => equal strip thickness in inches):
    # the PC1 heatmap spans the scatter's width beneath it, the PC2 heatmap spans
    # its height to the left, and the colorbar matches that thickness on the
    # right, all aligned to the scatter's edges.
    main, left, bottom, strip, gap, cbar_gap = 0.52, 0.32, 0.26, 0.05, 0.09, 0.02
    fig = plt.figure(figsize=(6.5, 6.5))
    ax = fig.add_axes([left, bottom, main, main])
    ax_pc1 = fig.add_axes([left, bottom - gap - strip, main, strip])
    ax_pc2 = fig.add_axes([left - gap - strip, bottom, strip, main])
    cax = fig.add_axes([left + main + cbar_gap, bottom, strip, main])

    _scatter_by_state_and_mouse(ax, scores[:, 0], scores[:, 1], states,
                                mouse_labels, markers)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')

    ax_pc1.imshow(loadings[:, 0][None, :], cmap='RdBu_r', vmin=-vlim, vmax=vlim,
                  aspect='auto')
    ax_pc1.set_yticks([])
    ax_pc1.set_xticks(range(len(feature_names)))
    ax_pc1.set_xticklabels(feature_names, rotation=90, fontsize=7)

    im = ax_pc2.imshow(loadings[:, 1][:, None], cmap='RdBu_r', vmin=-vlim,
                       vmax=vlim, aspect='auto')
    ax_pc2.set_xticks([])
    ax_pc2.set_yticks(range(len(feature_names)))
    ax_pc2.set_yticklabels(feature_names, fontsize=7)

    fig.colorbar(im, cax=cax, label='loading')
    state_handles, mouse_handles = _state_mouse_handles(sorted(set(states)),
                                                        markers)
    fig.legend(handles=state_handles, title='state', frameon=False,
               fontsize=TICKFONTSIZE, loc='upper left', bbox_to_anchor=(1.02, 0.78))
    fig.legend(handles=mouse_handles, title='mouse', frameon=False,
               fontsize=TICKFONTSIZE, loc='upper left', bbox_to_anchor=(1.02, 0.5))
    return fig


def plot_transition_traces(
    traces_by_mouse: Mapping[str, Mapping[Hashable, Mapping[str, np.ndarray]]],
    columns: Iterable[Hashable],
    window: int,
    ylabels: Sequence[str],
    xlabel: str,
    line_labels: Sequence[str],
) -> plt.Figure:
    """Lag traces with SEM bands on a mice x columns grid.

    One row per mouse, one column per key in ``columns``. Each axes overlays the
    mean trace of every column of ``mean`` — Δ from each window's own
    pre-transition baseline — encoded by color (``plt.cm.tab10``), with a shaded
    ±SEM band. Dashed lines mark the transition (vertical) and zero change
    (horizontal). Each axes is titled ``"{mouse} {column}"``. What the columns
    and the overlaid lines mean is the caller's: figure 5 draws block-transition
    types with one line per state, figure 7 draws NM measures with one line per
    entered state. Line colors are positional, so labels are unaligned across
    mice.

    Parameters
    ----------
    traces_by_mouse : mapping of str to mapping
        ``{mouse: {column_key: {'mean': arr, 'sem': arr}}}``, each ``arr`` of
        shape ``(2*window+1, n_lines)`` — the per-position mean and standard
        error across that mouse's transitions.
    columns : iterable of hashable
        Column keys drawn left to right; a mouse missing one gets a blank cell.
    window : int
        Half-window in trials; the x-axis spans ``[-window, window]``.
    ylabels : sequence of str
        Y-axis label per column, in the order of ``columns``.
    xlabel : str
        X-axis label, shared by every axes.
    line_labels : sequence of str
        Legend entry per overlaid line, drawn once on the top-left axes. Must be
        at least as long as the widest mouse's ``n_lines``.

    Returns
    -------
    matplotlib.figure.Figure
        Grid of mouse (rows) x column axes; cells with no data are hidden.
    """
    mice = list(traces_by_mouse)
    columns = list(columns)
    lag = np.arange(-window, window + 1)

    # Same figure width, height, and per-axes decoration (one-line title, x/y
    # labels) as the other one-row-per-mouse figures, so axis height and vertical
    # spacing match them exactly under constrained layout.
    fig, axes = plt.subplots(
        len(mice), len(columns),
        figsize=(3.5 * len(columns), ROW_HEIGHT * len(mice)),
        squeeze=False, layout='constrained')
    for row, mouse in zip(axes, mice):
        for ax, column, ylabel in zip(row, columns, ylabels):
            stats = traces_by_mouse[mouse].get(column)
            if stats is None:
                ax.axis('off')
                continue
            mean, sem = stats['mean'], stats['sem']
            for line, color in enumerate(plt.cm.tab10(np.arange(mean.shape[1]))):
                ax.fill_between(lag, mean[:, line] - sem[:, line],
                                mean[:, line] + sem[:, line],
                                color=color, alpha=0.2, linewidth=0)
                ax.plot(lag, mean[:, line], color=color,
                        label=line_labels[line])
            ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
            ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.set_title(f'{mouse} {column}', fontsize=9)
            ax.tick_params(labelsize=7)
    axes[0][0].legend(fontsize=6, frameon=False, loc='best')
    return fig

