import re
from collections.abc import Iterable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from scipy.stats import sem as scipy_sem
from sklearn.preprocessing import quantile_transform

from iblnm.config import (
    ANALYSIS_CONTRASTS, NM_CMAPS, QCCMAP,
    RESPONSE_EVENTS, RESPONSE_WINDOWS,
    SESSIONTYPE2COLOR, SESSIONTYPE2FLOAT, TARGETNM2POSITION,
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


def session_overview_matrix(group, columns='session_n', ax=None,
                            color_by='session_type',
                            split_float_map=None, split_color_map=None):
    """
    Plot a matrix of sessions per subject, colored by session type.

    All sessions in group._catalog are shown at 50% opacity. Sessions in
    group.sessions (passing the current filter) are overlaid at 100% opacity.

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
        Column to color cells by. Defaults to 'session_type'.
    split_float_map : dict, optional
        Mapping from color_by values to float positions for the colormap.
        Defaults to SESSIONTYPE2FLOAT.
    split_color_map : dict, optional
        Mapping from color_by values to colors. Defaults to SESSIONTYPE2COLOR.

    Raises
    ------
    ValueError
        If there is more than one session per (subject, columns) cell in _catalog.
    """
    _float_map = split_float_map or SESSIONTYPE2FLOAT
    _color_map = split_color_map or SESSIONTYPE2COLOR

    df_base = group._catalog.copy()
    df_overlay = group.sessions.copy()

    df_base['_float'] = df_base[color_by].map(_float_map)
    df_overlay['_float'] = df_overlay[color_by].map(_float_map)

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

    base_matrix = df_base.pivot_table(
        index='subject', columns=columns, values='_float',
        aggfunc='first', fill_value=0,
    )
    overlay_matrix = df_overlay.pivot_table(
        index='subject', columns=columns, values='_float',
        aggfunc='first',
    )

    base_matrix = base_matrix.reindex(subject_order).fillna(0)
    # Reindex overlay to the same shape as base; missing cells stay NaN (not painted)
    overlay_matrix = overlay_matrix.reindex(
        index=subject_order, columns=base_matrix.columns
    )

    # Build colormap from catalog values
    present_types = [st for st in _float_map.keys() if st in df_base[color_by].values]
    color_list = ['white'] + [_color_map[st] for st in present_types]
    cmap = colors.ListedColormap(color_list)
    bounds = [0] + [_float_map[st] for st in present_types] + [1.01]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(0.15 * len(base_matrix.columns), 0.15 * len(base_matrix))
        )

    ax.matshow(base_matrix, cmap=cmap, norm=norm, alpha=0.5)
    ax.matshow(overlay_matrix, cmap=cmap, norm=norm, alpha=1)

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

    # Colorbar
    tick_positions = [(bounds[i] + bounds[i + 1]) / 2 for i in range(1, len(bounds) - 1)]
    cbar = plt.colorbar(ax.images[0], ax=ax, shrink=0.5, boundaries=bounds, ticks=tick_positions)
    cbar.set_ticklabels(present_types)
    cbar.ax.set_ylim(bounds[1], bounds[-1])

    return ax


def target_overview_barplot(df_sessions, ax=None, barwidth=0.8,
                            color_by='session_type', split_color_map=None,
                            bar_color_map=None, split_alpha_map=None,
                            horizontal=False):
    """Stacked bar plot of session counts per target region.

    Each bar stacks the categories of ``color_by``. By default a segment's fill
    comes from ``split_color_map[category]`` at full opacity. Pass
    ``bar_color_map`` (keyed by ``target_NM``) to instead color every segment by
    its target identity, and ``split_alpha_map`` (keyed by category) to fade
    segments by category — e.g. proficient at 1.0, not-proficient at 0.5. When
    both are given the legend shows neutral gray swatches per category, since
    fill color then encodes target, not category.

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
        Maps ``color_by`` category to fill color. Defaults to SESSIONTYPE2COLOR.
        Ignored when ``bar_color_map`` is given.
    bar_color_map : dict, optional
        Maps ``target_NM`` to fill color. When set, segments are colored by
        target identity instead of by category.
    split_alpha_map : dict, optional
        Maps ``color_by`` category to opacity. Defaults to opaque. Also fixes the
        stacking order of categories when set.
    horizontal : bool
        If True, draw horizontal bars.
    """
    _color_map = split_color_map or SESSIONTYPE2COLOR

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

    # Stacking order: alpha map when given, else the color map's canonical order.
    category_order = split_alpha_map or _color_map
    categories = [c for c in category_order if c in df_n.columns]
    for category in categories:
        ns = df_n[category]
        if bar_color_map is not None:
            color = [bar_color_map[target] for target in df_n.index]
        else:
            color = _color_map[category]
        alpha = split_alpha_map.get(category, 1.0) if split_alpha_map else 1.0
        label = None if bar_color_map is not None else category
        if horizontal:
            ax.barh(positions, ns, left=cumulative, height=barwidth,
                    color=color, alpha=alpha, label=label)
            for y, n, x_left in zip(positions, ns, cumulative):
                if n > 0:
                    ax.text(x_left + n/2, y, str(n), ha='center', va='center',
                            fontweight='bold', color='white')
        else:
            ax.bar(positions, ns, bottom=cumulative, width=barwidth,
                   color=color, alpha=alpha, label=label)
            for x, n, y_bottom in zip(positions, ns, cumulative):
                if n > 0:
                    ax.text(x, y_bottom + n/2, str(n), ha='center', va='center',
                            fontweight='bold', color='white')
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

    if bar_color_map is not None and split_alpha_map is not None:
        handles = [Patch(facecolor='gray', alpha=split_alpha_map[c], label=c)
                   for c in categories]
        ax.legend(handles=handles)
    else:
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
                    horizontal=False):
    """Add text labels to bars with optional L/R breakdown."""
    for i, (pos, n) in enumerate(zip(positions, values)):
        if n > 0:
            if hemisphere_counts is not None:
                n_left, n_right = hemisphere_counts[i]
                label = f'{int(n)}\n{n_left}L/{n_right}R'
            else:
                label = str(int(n))
            if horizontal:
                ax.text(n / 2, pos, label, ha='center', va='center',
                        fontweight='bold', color=color)
            else:
                ax.text(pos, n / 2, label, ha='center', va='center',
                        fontweight='bold', color=color, rotation=90)


def mouse_overview_barplot(df_sessions, min_biased_ephys=5, min_ephys=3,
                           min_sessions=None, ax=None, barwidth=0.25,
                           color_by='session_type', split_color_map=None,
                           horizontal=False):
    """
    Barplot showing mouse training progress per target region.

    When min_sessions is provided, uses a simplified single-threshold model:
    one bar per category per target, showing mice with ≥min_sessions in that
    category.

    Legacy mode (min_sessions=None): three bars per target using min_biased_ephys
    and min_ephys thresholds.

    Parameters
    ----------
    df_sessions : pd.DataFrame
        One row per recording, with 'subject', 'target_NM', and color_by columns.
    min_biased_ephys : int
        Legacy: minimum combined biased+ephys sessions.
    min_ephys : int
        Legacy: minimum ephys sessions.
    min_sessions : int, optional
        If set, use simplified threshold: mice with ≥min_sessions in each category.
    ax : matplotlib.axes.Axes, optional
    barwidth : float
    color_by : str
        Column to group bars by. Defaults to 'session_type'.
    split_color_map : dict, optional
        Mapping from color_by values to colors. Defaults to SESSIONTYPE2COLOR.
    horizontal : bool
        If True, draw horizontal bars.
    """
    _color_map = split_color_map or SESSIONTYPE2COLOR

    if ax is None:
        fig, ax = plt.subplots()

    if len(df_sessions) == 0:
        ax.set_title("No data to plot")
        return ax

    target_nms = sorted(df_sessions['target_NM'].unique(),
                        key=lambda x: TARGETNM2POSITION.get(x, 999))
    xpos = np.arange(len(target_nms))

    has_hemisphere = 'hemisphere' in df_sessions.columns

    if min_sessions is not None:
        # Simplified mode: one bar per category per target
        categories = [c for c in _color_map.keys() if c in df_sessions[color_by].values]
        n_cats = len(categories)
        offsets = np.linspace(-(n_cats - 1) / 2, (n_cats - 1) / 2, n_cats) * barwidth

        for offset, category in zip(offsets, categories):
            counts = []
            hemi_counts = [] if has_hemisphere else None
            for target_nm in target_nms:
                target_df = df_sessions[df_sessions['target_NM'] == target_nm]
                cat_df = target_df[target_df[color_by] == category]
                n_per_subject = cat_df.groupby('subject').size()
                qualifying = n_per_subject[n_per_subject >= min_sessions].index
                counts.append(len(qualifying))
                if has_hemisphere:
                    q_df = target_df[target_df['subject'].isin(qualifying)]
                    n_l = q_df[q_df['hemisphere'] == 'l']['subject'].nunique()
                    n_r = q_df[q_df['hemisphere'] == 'r']['subject'].nunique()
                    hemi_counts.append((n_l, n_r))
            if horizontal:
                ax.barh(xpos + offset, counts, barwidth,
                        color=_color_map[category],
                        label=f'≥{min_sessions} {category}')
            else:
                ax.bar(xpos + offset, counts, barwidth,
                       color=_color_map[category],
                       label=f'≥{min_sessions} {category}')
            _add_bar_labels(ax, xpos + offset, counts, hemi_counts,
                            horizontal=horizontal)
    else:
        # Legacy mode
        session_counts = (
            df_sessions.groupby(['target_NM', 'subject', 'session_type'])
            .size().reset_index(name='n_sessions')
        )
        results = []
        for target_nm in target_nms:
            target_data = session_counts[session_counts['target_NM'] == target_nm]
            training_mice = target_data[
                target_data['session_type'] == 'training'
            ]['subject'].unique()
            biased_ephys_counts = target_data[
                target_data['session_type'].isin(['biased', 'ephys'])
            ].groupby('subject')['n_sessions'].sum()
            biased_ephys_mice = biased_ephys_counts[biased_ephys_counts >= min_biased_ephys].index
            ephys_mice = target_data[
                (target_data['session_type'] == 'ephys') &
                (target_data['n_sessions'] >= min_ephys)
            ]['subject'].unique()
            results.append({
                'target_NM': target_nm,
                'n_training': len(training_mice),
                'n_biased_ephys': len(biased_ephys_mice),
                'n_ephys': len(ephys_mice),
            })
        df_results = pd.DataFrame(results)
        _bar = ax.barh if horizontal else ax.bar
        _size_kw = 'height' if horizontal else 'width'
        _stack_kw = 'left' if horizontal else 'bottom'
        _bar(xpos - barwidth, df_results['n_training'].values,
             **{_size_kw: barwidth},
             color=SESSIONTYPE2COLOR['training'], label='training')
        _bar(xpos, df_results['n_biased_ephys'].values,
             **{_size_kw: barwidth},
             color=SESSIONTYPE2COLOR['biased'],
             label=f'≥{min_biased_ephys} biased/ephys')
        _bar(xpos + barwidth, df_results['n_ephys'].values,
             **{_size_kw: barwidth},
             color=SESSIONTYPE2COLOR['ephys'], label=f'≥{min_ephys} ephys')
        _add_bar_labels(ax, xpos - barwidth, df_results['n_training'].values,
                        horizontal=horizontal)
        _add_bar_labels(ax, xpos, df_results['n_biased_ephys'].values,
                        horizontal=horizontal)
        _add_bar_labels(ax, xpos + barwidth, df_results['n_ephys'].values,
                        horizontal=horizontal)

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
    ax.set_title('Mouse training progress by target')

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
    show_outliers=True, outlier_threshold=1.5, colors=None, **violin_kwargs
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
            pc.set_alpha(1)
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


def plot_relative_contrast(df_group, response_col, target_nm, event, fig=None,
                           window_label=None, aggregation='pool', min_trials=10):
    """Plot response magnitude by contrast, split into contra and ipsi panels.

    The contra panel x-axis is inverted so that the highest contrast is on the
    far left; the ipsi panel runs normally left-to-right. Together they read:
    ``100 ← contra % → 0 | 0 ← ipsi % → 100``.

    Parameters
    ----------
    df_group : pd.DataFrame
        Pre-filtered rows for one (target_NM × event) group with columns
        ``side`` ('contra' / 'ipsi'), ``contrast`` (absolute, 0–1),
        ``feedbackType``, ``subject``, and ``<response_col>``.
    response_col : str
        Column name for the response magnitude to plot.
    target_nm : str
        Target neuromodulator label; used for the title and color lookup.
    event : str
        Raw event name (e.g. 'stimOn_times'); used for the title.
    fig : plt.Figure or None
        Figure with two existing axes to draw on. If None, a new figure is
        created with ``plt.subplots(1, 2, sharey=True)``.
    window_label : str or None
        Label for the response window (e.g. 'early', 'late').
    aggregation : str
        'pool' (default): grand mean ± SEM across all trials.
        'subject': mean of subject means ± SEM of subject means.

    Returns
    -------
    plt.Figure
    """
    if aggregation not in ('pool', 'subject'):
        raise ValueError(f"aggregation must be 'pool' or 'subject', got {aggregation!r}")
    if fig is None:
        fig, _ = plt.subplots(1, 2, sharey=True, gridspec_kw={'wspace': 0.05},
                              layout='constrained')

    ax_c, ax_i = fig.axes[0], fig.axes[1]

    event_label = event.replace('_times', '')
    _window = window_label or ''
    color = TARGETNM_COLORS.get(target_nm, 'black')
    n_sessions = df_group['eid'].nunique() if 'eid' in df_group.columns else '?'
    n_subjects = df_group['subject'].nunique() if len(df_group) > 0 else 0
    fig.suptitle(
        f'{target_nm} — {event_label} ({_window})\n'
        f'{n_sessions} sessions, {n_subjects} subjects',
        fontsize=LABELFONTSIZE,
    )

    # Compute contrasts from the full dataset so both panels share the same x-axis
    contrasts = sorted(df_group['contrast'].unique()) if len(df_group) > 0 else []
    ranks = list(range(len(contrasts)))
    rank_map = dict(zip(contrasts, ranks))

    # Subject-mean removal: subtract per-subject mean, add grand mean
    if len(df_group) > 0:
        grand_mean = df_group[response_col].mean()
        subj_means = df_group.groupby('subject')[response_col].transform('mean')
        df_group = df_group.copy()
        df_group[response_col] = df_group[response_col] - subj_means + grand_mean

    # Remove subject × side × feedback × contrast conditions with too few trials
    group_cols = ['subject', 'side', 'feedbackType', 'contrast']
    trial_counts = df_group.groupby(group_cols)[response_col].transform('count')
    df_group = df_group[trial_counts > min_trials]

    for ax, side in ((ax_c, 'contra'), (ax_i, 'ipsi')):
        df_side = df_group[df_group['side'] == side]

        for feedback, ls in ((1, '-'), (-1, '--')):
            df_fb = df_side[df_side['feedbackType'] == feedback]
            if len(df_fb) == 0:
                continue

            means, sems = [], []
            for c in contrasts:
                df_c = df_fb[df_fb['contrast'] == c]
                if aggregation == 'pool':
                    vals = df_c[response_col].dropna()
                    means.append(vals.mean() if len(vals) > 0 else np.nan)
                    sems.append(scipy_sem(vals, nan_policy='omit') if len(vals) > 0 else np.nan)
                else:
                    subj_means = df_c.groupby('subject')[response_col].mean()
                    means.append(subj_means.mean() if len(subj_means) > 0 else np.nan)
                    sems.append(scipy_sem(subj_means, nan_policy='omit') if len(subj_means) > 0 else np.nan)

            label = 'correct' if feedback == 1 else 'incorrect'
            xpos = np.array([rank_map[c] for c in contrasts])
            ax.errorbar(xpos, means, yerr=np.array(sems, dtype=float),
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


def plot_movement_response(df_group, response_col, timing_col, target_nm,
                           event='stimOn_times', fig=None):
    """Scatter response magnitude against a timing variable, colored by contrast.

    The raw-data within-contrast check for the within-contrast model: one point
    per trial (low alpha to show density), with contrast level mapped to a
    continuous color scale, so the response-vs-timing relationship is readable
    within each contrast band. Pools all sides into a single panel.

    Parameters
    ----------
    df_group : pd.DataFrame
        Rows for one (target_NM, event, timing variable) group with columns
        ``subject``, ``contrast``, ``<response_col>``, and ``<timing_col>``
        (already log10-transformed).
    response_col : str
        Column name for the response magnitude.
    timing_col : str
        Column name for the log-transformed timing variable.
    target_nm : str
        Target neuromodulator label; used for the title.
    event : str
        Alignment event label for the response magnitude; used for the title.
    fig : plt.Figure or None
        Figure with one existing axis to draw on. If None, a new figure is
        created.

    Returns
    -------
    plt.Figure
    """
    if fig is None:
        fig, _ = plt.subplots(1, 1, layout='constrained')
    ax = fig.axes[0]

    n_sessions = df_group['eid'].nunique() if 'eid' in df_group.columns else '?'
    n_subjects = df_group['subject'].nunique() if len(df_group) > 0 else 0
    timing_label = timing_col.replace('log_', '')
    fig.suptitle(
        f'{target_nm} — {event} ({timing_label})\n'
        f'{n_sessions} sessions, {n_subjects} subjects',
        fontsize=LABELFONTSIZE,
    )

    if len(df_group) == 0:
        return fig

    sc = ax.scatter(df_group[timing_col], df_group[response_col],
                    c=df_group['contrast'], cmap='viridis', s=8, alpha=0.15,
                    edgecolors='none')
    fig.colorbar(sc, ax=ax, label='Contrast (%)')
    ax.set_xlabel(f'{timing_label} (log₁₀ s)')
    ax.set_ylabel(r'$\Delta$ activity (z-score)')
    ax.axhline(0, ls='--', color='gray', lw=0.5)
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
_EVENT_ORDER = {'stimOn': 0, 'firstMovement': 1, 'feedback': 2}
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
    distribution (:func:`iblnm.analysis.code_predictors`), so one percent maps to
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
        ``response_lmm_fit`` output for one model: ``target_NM``, ``event``,
        ``marginal_r2``, ``conditional_r2``.
    coef_df : pd.DataFrame
        ``response_lmm_effects(name, 'coefficients')``: ``term``,
        ``target_NM``, ``event``, ``Coef.``, ``P>|z|``.
    emm_frames : dict[str, pd.DataFrame]
        Maps each bottom-row factor (``'reward'``, ``'side'``, ``'contrast'``)
        to its ``response_lmm_effects(name, 'emm', [factor])`` frame.
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


# Dropped regressors on the per-recording OLS x-axis, in fixed order: the
# `persession` model family keys minus `full`.
_PERSESSION_DROPONE_PREDICTORS = ['contrast', 'side', 'reward', 'choice_side',
                                  'log_reaction_time', 'peak_velocity']
# x-axis layout (units where one subject occupies a width of 1):
_SUBJECT_SPACING = 0.7       # x between consecutive subjects within a target-NM
_TARGETNM_GAP = 1.0          # blank x between consecutive target-NM groups
_SESSION_MARKER_SIZE = 40    # open-dot marker size for a single session
_MEDIAN_MARKER_SIZE = 260    # '_' marker size for a subject's median dash
_MEDIAN_LINEWIDTH = 3.0      # '_' median dash thickness


def _group_xslots(df, targets):
    """Lay out one contiguous block of x slots per target-NM group.

    Each target-NM with data gets one slot per subject, ``_SUBJECT_SPACING``
    apart, with ``_TARGETNM_GAP`` blank units between groups — so a target-NM's
    horizontal extent scales with its subject count. The slot *positions* are
    fixed (independent of which subject fills which), so the per-panel median
    ordering does not change group widths or tick centres.

    Parameters
    ----------
    df : pd.DataFrame
        One event's rows; needs ``target_NM`` and ``subject``.
    targets : sequence of str
        Target-NMs in plot order.

    Returns
    -------
    subjects_by_target : dict[str, list[str]]
        Subjects present per target-NM (name-sorted; the panel reorders them).
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


def _scatter_subject(ax, x, deltas, color):
    """Plot one subject's sessions and its median at ``x``.

    Each session is a translucent open dot edge-colored by target-NM (no fill)
    stacked at the subject's x; the subject's median is a single thicker ``'_'``
    marker in the same target-NM color on top.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    x : float
        The subject's x position.
    deltas : np.ndarray
        That subject's per-session ΔR² in one cell.
    color : color
        Marker color (the subject's target-NM color).
    """
    ax.scatter(np.full(len(deltas), x), deltas, marker='o', facecolors='none',
               edgecolors=color, s=_SESSION_MARKER_SIZE, alpha=0.5, zorder=3)
    ax.scatter(x, np.median(deltas), marker='_', color=color,
               s=_MEDIAN_MARKER_SIZE, linewidths=_MEDIAN_LINEWIDTH, zorder=4)


def _plot_persession_grid(df, title, rows, supylabel):
    """Per-session scatter grid: ``rows`` by event columns, sharing one y-axis.

    Shared layout for the per-session figures. Each entry of ``rows`` is one
    grid row; columns are events (``_sort_events`` order). Within a panel each
    subject occupies its own x position (sessions as translucent open dots
    edge-colored by target-NM, one thicker target-NM-colored ``'_'`` marker at
    the subject's median), grouped by target-NM with a gap so a target-NM's width
    scales with its subject count (see ``_group_xslots``). Within each group,
    subjects are ordered left to right by ascending median in that panel (so the
    order can differ across panels). One x-tick per target-NM is centred on its
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
                color = TARGETNM_COLORS.get(tnm, 'gray')
                vals_by_subject = {
                    s: df_cell.loc[df_cell['subject'] == s, value_col].values
                    for s in subjects}
                # Order this panel's subjects left to right by ascending median;
                # subjects with no data here sort last (their slot stays empty).
                ordered = sorted(subjects, key=lambda s: (
                    np.median(vals_by_subject[s]) if len(vals_by_subject[s])
                    else np.inf))
                for subject, x in zip(ordered, slots_by_target[tnm]):
                    vals = vals_by_subject[subject]
                    if len(vals):
                        _scatter_subject(ax, x, vals, color)
            ax.axhline(0, ls='--', color='gray', lw=0.5)
            if r == 0:
                ax.set_title(event)
            if c == 0:
                ax.set_ylabel(label, fontsize=TICKFONTSIZE)
        axes[-1, c].set_xticks([centre for _, centre in ticks])
        axes[-1, c].set_xticklabels([tnm for tnm, _ in ticks], rotation=30,
                                    ha='right', fontsize=TICKFONTSIZE)
    fig.supylabel(supylabel)
    fig.suptitle(title, fontsize=LABELFONTSIZE)
    return fig


def plot_ols_dropone(df, title):
    """Per-session drop-one ΔR² — dropped-regressor rows × event columns.

    One row per dropped regressor (``_PERSESSION_DROPONE_PREDICTORS`` order),
    each plotting that regressor's ``delta_r2``. See ``_plot_persession_grid``
    for the shared layout.
    """
    rows = [(p, 'delta_r2', p) for p in _PERSESSION_DROPONE_PREDICTORS]
    return _plot_persession_grid(df, title, rows,
                                 'ΔR² (per-session, in-sample)')


def plot_ols_total_r2(df, title):
    """Per-session full-model R² — single row × event columns.

    Same format as ``plot_ols_dropone`` but a separate figure (its own y-axis),
    plotting the full-model ``r2`` (read off one predictor, since it repeats
    across them). See ``_plot_persession_grid``.
    """
    rows = [('full model R²', 'r2', _PERSESSION_DROPONE_PREDICTORS[0])]
    return _plot_persession_grid(df, title, rows,
                                 'R² (per-session, in-sample)')


_VARCOMP_COLORS = {'V_mouse': '#1f6fb4', 'V_session': '#e08214'}
_VARCOMP_HALF_WIDTH = 0.4  # x half-width of a full-amplitude violin half


def _half_violin(ax, centre, side, x, density, color, label):
    """Fill one half-violin against ``centre`` along the variance axis ``x``.

    The KDE outline ``density`` is normalized to ``_VARCOMP_HALF_WIDTH`` and laid
    out horizontally on one side of ``centre`` (``side`` = -1 left, +1 right), so
    two components share an x-slot as mirrored halves.
    """
    width = _VARCOMP_HALF_WIDTH * density / density.max()
    ax.fill_betweenx(x, centre, centre + side * width, color=color, alpha=0.7, label=label)


def plot_varcomp_violins(violin_df, title):
    """Variance-components posterior violins — regressor rows × event columns.

    Each panel places one x-slot per target-NM (``TARGETNM2POSITION`` order) and
    draws, from the stored ``(x, density)`` KDE outlines, two mirrored half
    violins per slot: ``V_mouse`` (left) and ``V_session`` (right). Panels share
    one y-axis (the standardized variance scale). An empty frame returns a titled
    figure.

    Parameters
    ----------
    violin_df : pd.DataFrame
        Long-form KDE outlines with columns ``target_NM, event, regressor,
        component, x, density`` (``RESPONSE_VARCOMP_VIOLIN_COLUMNS``).
    title : str
        Figure suptitle.

    Returns
    -------
    plt.Figure
    """
    has_data = len(violin_df) > 0
    events = _sort_events(violin_df['event'].unique()) if has_data else []
    present = set(violin_df['regressor']) if has_data else set()
    regressors = [r for r in _PERSESSION_DROPONE_PREDICTORS if r in present]
    n_rows, n_cols = max(len(regressors), 1), max(len(events), 1)

    if not has_data:
        fig, _ = plt.subplots(n_rows, n_cols, squeeze=False,
                              layout='constrained')
        fig.suptitle(title, fontsize=LABELFONTSIZE)
        return fig

    targets = sorted(violin_df['target_NM'].unique(),
                     key=lambda t: TARGETNM2POSITION.get(t, 999))
    fig, axes = plt.subplots(
        n_rows, n_cols, squeeze=False, sharex=True, sharey=True,
        layout='constrained',
        figsize=(1.2 * len(targets) * n_cols + 1, 2.0 * n_rows + 1))

    for c, event in enumerate(events):
        for r, regressor in enumerate(regressors):
            ax = axes[r, c]
            cell = violin_df[(violin_df['event'] == event)
                             & (violin_df['regressor'] == regressor)]
            for slot, tnm in enumerate(targets):
                tcell = cell[cell['target_NM'] == tnm]
                for component, side in (('V_mouse', -1), ('V_session', 1)):
                    body = tcell[tcell['component'] == component]
                    if body.empty:
                        continue
                    _half_violin(ax, slot, side, np.log10(body['x'].values),
                                 body['density'].values,
                                 _VARCOMP_COLORS[component],
                                 component if (slot == 0) else '_nolegend_')
            if r == 0:
                ax.set_title(event)
            if c == 0:
                ax.set_ylabel(regressor, fontsize=TICKFONTSIZE)
        axes[-1, c].set_xticks(range(len(targets)))
        axes[-1, c].set_xticklabels(targets, rotation=30, ha='right',
                                    fontsize=TICKFONTSIZE)
    axes[0, 0].legend()
    fig.supylabel('variance (standardized)')
    fig.suptitle(title, fontsize=LABELFONTSIZE)
    return fig


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


def plot_mean_response_traces(traces_df, target_nm, min_trials=5,
                              baseline_window=(-0.15, 0)):
    """Mean peri-event response traces for one target-NM.

    Layout: 2 rows (reward top, omission bottom) × n_events columns.
    Each panel has one line per contrast level, colored by the NM colormap.

    For each (event, contrast, feedbackType), applies baseline normalization
    (subtract mean in ``baseline_window``), then subject-mean removal before
    computing the grand mean and SEM.

    Conditions where any subject has fewer than ``min_trials`` trials are
    excluded.

    Parameters
    ----------
    traces_df : pd.DataFrame
        Long-form with columns: eid, subject, target_NM, event, contrast,
        feedbackType, time, response, n_trials.
    target_nm : str
        Target-NM label (used for title).
    min_trials : int
        Minimum trials per subject per (event, contrast, feedbackType).
    baseline_window : tuple of float
        (start, end) seconds for baseline normalization.

    Returns
    -------
    plt.Figure
    """
    df = traces_df[traces_df['target_NM'] == target_nm].copy()
    present = set(df['event'].unique())
    events = [e for e in RESPONSE_EVENTS if e in present]
    # Append any events not in the canonical order
    events += sorted(present - set(RESPONSE_EVENTS))
    n_events = max(len(events), 1)
    feedback_types = [1, -1]
    fb_labels = {1: 'Reward', -1: 'Omission'}
    contrasts = sorted(df['contrast'].unique())

    # Build color map: NM colormap with shades per contrast level
    nm = target_nm.split('-')[-1]
    cmap = NM_CMAPS.get(nm, NM_CMAPS['DA'])
    n_levels = len(ANALYSIS_CONTRASTS)
    shade_map = {c: cmap(0.3 + 0.7 * i / (n_levels - 1))
                 for i, c in enumerate(ANALYSIS_CONTRASTS)}

    fig, axes = plt.subplots(2, n_events,
                             figsize=(4 * n_events, 6),
                             sharey=True, squeeze=False)

    for col, event in enumerate(events):
        for row, fb in enumerate(feedback_types):
            ax = axes[row, col]
            df_cell = df[
                (df['event'] == event)
                & (df['feedbackType'] == fb)
            ]

            for contrast in contrasts:
                df_c = df_cell[df_cell['contrast'] == contrast]
                if len(df_c) == 0:
                    continue

                # Filter: drop recordings where the subject has too few trials
                if 'n_trials' in df_c.columns:
                    trials_per_subj = df_c.groupby('subject')['n_trials'].first()
                    bad_subjects = trials_per_subj[trials_per_subj < min_trials].index
                    df_c = df_c[~df_c['subject'].isin(bad_subjects)]
                    if len(df_c) == 0:
                        continue

                # Pivot to (recording, time) matrix — group by
                # (eid, fiber_idx) to distinguish bilateral recordings
                has_fiber = 'fiber_idx' in df_c.columns
                if has_fiber:
                    rec_keys = list(df_c.groupby(['eid', 'fiber_idx']).groups.keys())
                elif 'brain_region' in df_c.columns:
                    rec_keys = list(df_c.groupby(['eid', 'brain_region']).groups.keys())
                else:
                    rec_keys = [(eid,) for eid in df_c['eid'].unique()]
                time_vals = np.array(sorted(df_c['time'].unique()))
                n_recs = len(rec_keys)
                n_time = len(time_vals)

                trace_matrix = np.full((n_recs, n_time), np.nan)
                subjects_arr = []
                for i, key in enumerate(rec_keys):
                    if has_fiber:
                        rec_data = df_c[(df_c['eid'] == key[0]) & (df_c['fiber_idx'] == key[1])]
                    elif 'brain_region' in df_c.columns and len(key) == 2:
                        rec_data = df_c[(df_c['eid'] == key[0]) & (df_c['brain_region'] == key[1])]
                    else:
                        rec_data = df_c[df_c['eid'] == key[0]]
                    rec_data = rec_data.sort_values('time')
                    trace_matrix[i] = rec_data['response'].values
                    subjects_arr.append(rec_data['subject'].iloc[0])
                subjects_arr = np.array(subjects_arr)

                # Baseline normalization: subtract mean in baseline window
                bl_mask = (time_vals >= baseline_window[0]) & (time_vals < baseline_window[1])
                if bl_mask.any():
                    bl_means = np.nanmean(trace_matrix[:, bl_mask], axis=1,
                                          keepdims=True)
                    trace_matrix = trace_matrix - bl_means

                # Subject-mean removal
                grand_mean = np.nanmean(trace_matrix, axis=0)
                adjusted = np.copy(trace_matrix)
                for s in np.unique(subjects_arr):
                    s_mask = subjects_arr == s
                    s_mean = np.nanmean(trace_matrix[s_mask], axis=0)
                    adjusted[s_mask] = (
                        trace_matrix[s_mask] - s_mean + grand_mean
                    )

                mean_trace = np.nanmean(adjusted, axis=0)
                if n_recs > 1:
                    sem_trace = (np.nanstd(adjusted, axis=0, ddof=1)
                                 / np.sqrt(n_recs))
                else:
                    sem_trace = np.zeros(n_time)

                color = shade_map.get(contrast, 'gray')
                ax.plot(time_vals, mean_trace, color=color, linewidth=1.5,
                        label=f'{contrast}')
                if n_recs > 1:
                    ax.fill_between(time_vals, mean_trace - sem_trace,
                                    mean_trace + sem_trace,
                                    color=color, alpha=0.2)

            ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
            ax.set_ylim(-1.5, 3)

            # Shaded response windows (whichever are defined in config)
            for start, end in RESPONSE_WINDOWS.values():
                ax.axvspan(start, end, alpha=0.12, color='gray', zorder=0)

            if row == 1:
                ax.set_xlabel('Time (s)')
            if col == 0:
                ax.set_ylabel(fb_labels[fb])
            if row == 0:
                event_label = event.replace('_times', '')
                ax.set_title(event_label)

    # Legend on first axis
    axes[0, 0].legend(title='Contrast', fontsize=TICKFONTSIZE,
                      title_fontsize=TICKFONTSIZE, loc='upper left')

    # N trials / N sessions / N mice label on last column, top row
    n_sessions = df['eid'].nunique()
    n_mice = df['subject'].nunique()
    if 'n_trials' in df.columns:
        n_trials = int(
            df.groupby(['eid', 'event', 'contrast', 'feedbackType'])
            ['n_trials'].first().sum()
        )
        count_text = f'{n_trials} trials\n{n_sessions} sessions\n{n_mice} mice'
    else:
        count_text = f'{n_sessions} sessions\n{n_mice} mice'
    axes[0, -1].annotate(
        count_text,
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

    Reads the in-sample R² frame from ``response_lmm_fit``: for each movement
    variable, three nested models from the ``movement_<var>`` family — ``full``
    (the revised per-event task base extended with the movement predictor at
    2nd order), ``contrast`` (contrast dropped, the movement-family model), and
    ``movement`` (the predictor dropped, the task base). One panel per movement
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

def _assemble_rt_trials(group):
    """Build the trial table for RT violins from a group.

    Joins ``group.response_magnitudes`` to ``group.trial_regressors`` on
    (eid, trial), keeps trials with a recorded choice across all pLeft blocks,
    and restricts to the analysed target-NMs. Returns an empty frame (with the
    columns ``_draw_rt_violins`` expects) when either source is missing.

    Parameters
    ----------
    group : PhotometrySessionGroup
        May have ``response_magnitudes`` and ``trial_regressors`` loaded.

    Returns
    -------
    pd.DataFrame
        Columns include ``response_time``, ``contrast``, ``target_NM``.
    """
    if group.response_magnitudes is None or group.trial_regressors is None:
        return pd.DataFrame(columns=['response_time', 'contrast', 'target_NM'])
    df_resp = group.response_magnitudes.drop_duplicates(
        subset=['eid', 'trial', 'target_NM'])
    df_trial = df_resp.merge(
        group.trial_regressors[['eid', 'trial', 'response_time',
                                'contrast', 'probabilityLeft', 'choice']],
        on=['eid', 'trial'], how='inner',
    )
    df_trial = df_trial.query('choice != 0').copy()
    return df_trial[df_trial['target_NM'].isin(TARGETNMS_TO_ANALYZE)]


def plot_performance_grid(group, axes=None):
    """Grid of task-performance panels, one row per target-NM.

    Column 0 holds the 50-50 block psychometric curves (thin line per session,
    thick grand mean); column 1 holds response-time violins by contrast. Both
    columns reuse the per-target drawing helpers, so styling matches the
    standalone figures.

    Parameters
    ----------
    group : PhotometrySessionGroup
        Must have ``group.performance`` loaded; ``response_magnitudes`` and
        ``trial_regressors`` are needed for the RT column (empty otherwise).
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
    df_rt = _assemble_rt_trials(group)

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

