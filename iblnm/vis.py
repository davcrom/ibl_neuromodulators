import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import sem as scipy_sem
from sklearn.preprocessing import quantile_transform

from iblnm.config import (
    QCCMAP, SESSIONTYPE2COLOR, SESSIONTYPE2FLOAT,
    TARGETNM2POSITION, TARGETNM_COLORS, TARGETNM_POSITIONS,
    contrast_transform,
)


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


def session_overview_matrix(df, columns='session_n', highlight='good', ax=None):
    """
    Plot a matrix of sessions per subject, colored by session type.

    Parameters
    ----------
    df : pd.DataFrame
        Sessions dataframe with 'subject', 'session_type', and column specified by `columns`
    columns : str
        Column to use for x-axis (e.g., 'day_n', 'session_n')
    highlight : str, callable, or None
        Criteria for highlighting sessions at full opacity (others shown at 50%):
        - 'all': highlight all sessions
        - 'none': no highlighting (all at 50%)
        - callable: function(df) -> boolean mask
    ax : matplotlib.axes.Axes, optional
        Axes to plot on

    Raises
    ------
    ValueError
        If there is more than one session per subject/column cell
    """
    df = df.copy()
    df['_session_type_float'] = df['session_type'].map(SESSIONTYPE2FLOAT)

    # Check for duplicates
    duplicates = df.groupby(['subject', columns]).size()
    if (duplicates > 1).any():
        dup_cells = duplicates[duplicates > 1]
        raise ValueError(
            f"Multiple sessions per cell. Remove duplicates before plotting.\n"
            f"Duplicates:\n{dup_cells}"
        )

    # Build highlight mask
    if highlight == 'all':
        highlight_mask = pd.Series(True, index=df.index)
    elif highlight == 'none' or highlight is None:
        highlight_mask = pd.Series(False, index=df.index)
    elif callable(highlight):
        highlight_mask = highlight(df)
    else:
        raise ValueError(f"Invalid highlight value: {highlight}")

    df['_highlight'] = highlight_mask.astype(int)

    # Create matrices (no duplicates, so 'first' is fine)
    subject_matrix = df.pivot_table(
        index='subject',
        columns=columns,
        values='_session_type_float',
        aggfunc='first',
        fill_value=0
    )

    overlay_matrix = df.pivot_table(
        index='subject',
        columns=columns,
        values='_highlight',
        aggfunc='first',
        fill_value=0
    )

    # Get session types present in the data
    present_session_types = [st for st in SESSIONTYPE2FLOAT.keys() if st in df['session_type'].values]

    # Create categorical colormap with only present session types
    color_list = ['white'] + [SESSIONTYPE2COLOR[st] for st in present_session_types]
    cmap = colors.ListedColormap(color_list)
    bounds = [0] + [SESSIONTYPE2FLOAT[st] for st in present_session_types] + [1.01]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    # Plot base matrix at 50% opacity
    if ax is None:
        fig, ax = plt.subplots(figsize=(0.15 * len(subject_matrix.columns), 0.15 * len(subject_matrix)))
    ax.matshow(subject_matrix, cmap=cmap, norm=norm, alpha=0.5)

    # Overlay highlighted sessions at full opacity
    overlay_subject_matrix = subject_matrix.copy()
    overlay_subject_matrix[~overlay_matrix.astype(bool)] = np.nan
    ax.matshow(overlay_subject_matrix, cmap=cmap, norm=norm, alpha=1)

    # Format axes
    ax.set_yticks(np.arange(len(subject_matrix)))
    ax.set_yticklabels(subject_matrix.index)
    ax.set_ylabel('Subject')
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(0, len(subject_matrix.columns) + 1, 10))
    ax.tick_params(axis='x', rotation=90)
    ax.xaxis.set_label_position('top')
    ax.set_xlabel(columns)

    # Add gridlines
    for xtick in np.arange(len(subject_matrix.columns)):
        ax.axvline(xtick - 0.5, color='white')
    for ytick in np.arange(len(subject_matrix)):
        ax.axhline(ytick - 0.5, color='white')

    # Colorbar
    tick_positions = [(bounds[i] + bounds[i + 1]) / 2 for i in range(1, len(bounds) - 1)]
    cbar = plt.colorbar(ax.images[0], ax=ax, shrink=0.5, boundaries=bounds, ticks=tick_positions)
    cbar.set_ticklabels(present_session_types)
    cbar.ax.set_ylim(bounds[1], bounds[-1])

    return ax


def target_overview_barplot(df_sessions, ax=None, barwidth=0.8):
    """Stacked bar plot of session counts per target region, by session type."""
    if len(df_sessions) == 0:
        if ax is None:
            fig, ax = plt.subplots()
        ax.set_title("No data to plot")
        return ax

    # Create a target_NM x session_type matrix with session counts
    df_n = df_sessions.pivot_table(
        columns='session_type',
        index='target_NM',
        aggfunc='size',
        fill_value=0
    )

    if ax is None:
        fig, ax = plt.subplots()

    # Use contiguous positions for targets present in data (sorted by canonical order)
    sorted_targets = sorted(df_n.index, key=lambda x: TARGETNM2POSITION.get(x, 999))
    df_n = df_n.reindex(sorted_targets)
    xpos = list(range(len(df_n)))
    ypos = np.zeros(len(df_n))

    # Loop over session types present in data
    session_types = [st for st in ['training', 'biased', 'ephys'] if st in df_n.columns]
    for session_type in session_types:
        ns = df_n[session_type]
        color = SESSIONTYPE2COLOR[session_type]
        ax.bar(xpos, ns, bottom=ypos, width=barwidth, color=color, label=session_type)
        for x, n, y_bottom in zip(xpos, ns, ypos):
            if n > 0:
                ax.text(x, y_bottom + n/2, str(n), ha='center', va='center',
                        fontweight='bold', color='white')
        ypos += ns

    ax.set_xticks(list(xpos))
    n_mice = df_sessions.groupby('target_NM').apply(
        lambda x: len(x['subject'].unique()),
        include_groups=False
    )
    ax.set_xticklabels(
        ['%s\n(%d mice)' % (target_NM, n_mice.loc[target_NM]) for target_NM in df_n.index]
    )
    ax.set_xlim(right=max(xpos) + barwidth)
    ax.tick_params(axis='x', rotation=90)
    if max(ypos) > 0:
        ax.set_yticks(np.arange(0, np.ceil(max(ypos) / 100) + 1) * 100)
    ax.set_ylabel('N Sessions')
    ax.set_xlabel('Target-NM')
    ax.legend()

    n_recordings = len(df_sessions)
    n_sessions = df_sessions['eid'].nunique()
    n_mice = df_sessions['subject'].nunique()
    ax.set_title(f"{n_recordings} recordings, {n_sessions} sessions, {n_mice} mice")

    return ax


def _add_bar_labels(ax, xpos, values, hemisphere_counts=None, color='white'):
    """Add vertically-oriented text labels to bars with optional L/R breakdown."""
    for i, (x, n) in enumerate(zip(xpos, values)):
        if n > 0:
            if hemisphere_counts is not None:
                n_left, n_right = hemisphere_counts[i]
                label = f'{int(n)}\n{n_left}L/{n_right}R'
            else:
                label = str(int(n))
            ax.text(x, n / 2, label, ha='center', va='center',
                    fontweight='bold', color=color, rotation=90)


def mouse_overview_barplot(df_sessions, min_biased_ephys=5, min_ephys=3, ax=None, barwidth=0.25):
    """
    Barplot showing mouse training progress per target region.

    Three bars per target:
    - training: mice with any training sessions
    - biased/ephys: mice with ≥min_biased_ephys combined biased+ephys sessions
    - ephys: mice with ≥min_ephys ephys sessions

    If 'hemisphere' column present, bar labels include L/R breakdown.
    """
    if ax is None:
        fig, ax = plt.subplots()

    if len(df_sessions) == 0:
        ax.set_title("No data to plot")
        return ax

    # Count sessions per subject per target_NM per session_type
    session_counts = df_sessions.groupby(['target_NM', 'subject', 'session_type']).size().reset_index(name='n_sessions')

    # Get all target_NMs
    target_nms = sorted(df_sessions['target_NM'].unique(), key=lambda x: TARGETNM_POSITIONS.get(x, 999))

    has_hemisphere = 'hemisphere' in df_sessions.columns

    results = []
    for target_nm in target_nms:
        target_data = session_counts[session_counts['target_NM'] == target_nm]
        target_sessions = df_sessions[df_sessions['target_NM'] == target_nm]

        # Mice with training sessions
        training_mice = target_data[
            target_data['session_type'] == 'training'
        ]['subject'].unique()

        # Mice with sufficient biased + ephys sessions combined
        biased_ephys_counts = target_data[
            target_data['session_type'].isin(['biased', 'ephys'])
        ].groupby('subject')['n_sessions'].sum()
        biased_ephys_mice = biased_ephys_counts[biased_ephys_counts >= min_biased_ephys].index

        # Mice with sufficient ephys sessions
        ephys_mice = target_data[
            (target_data['session_type'] == 'ephys') &
            (target_data['n_sessions'] >= min_ephys)
        ]['subject'].unique()

        result = {
            'target_NM': target_nm,
            'n_training': len(training_mice),
            'n_biased_ephys': len(biased_ephys_mice),
            'n_ephys': len(ephys_mice),
        }

        # Compute per-session-type hemisphere counts
        if has_hemisphere:
            for key, mice in [('training', training_mice),
                              ('biased_ephys', biased_ephys_mice),
                              ('ephys', ephys_mice)]:
                hemi_data = target_sessions[target_sessions['subject'].isin(mice)]
                result[f'n_{key}_L'] = hemi_data[hemi_data['hemisphere'] == 'l']['subject'].nunique()
                result[f'n_{key}_R'] = hemi_data[hemi_data['hemisphere'] == 'r']['subject'].nunique()

        results.append(result)

    df_results = pd.DataFrame(results)

    # Get x positions based on TARGETNM_POSITIONS
    xpos = np.array([TARGETNM_POSITIONS[target_nm] for target_nm in df_results['target_NM']])

    # Plot bars
    ax.bar(xpos - barwidth, df_results['n_training'].values, barwidth,
           color=SESSIONTYPE2COLOR['training'], label='training')
    ax.bar(xpos, df_results['n_biased_ephys'].values, barwidth,
           color=SESSIONTYPE2COLOR['biased'], label=f'≥{min_biased_ephys} biased/ephys')
    ax.bar(xpos + barwidth, df_results['n_ephys'].values, barwidth,
           color=SESSIONTYPE2COLOR['ephys'], label=f'≥{min_ephys} ephys')

    # Add text labels with optional hemisphere breakdown
    if has_hemisphere:
        hemi_training = list(zip(df_results['n_training_L'], df_results['n_training_R']))
        hemi_biased_ephys = list(zip(df_results['n_biased_ephys_L'], df_results['n_biased_ephys_R']))
        hemi_ephys = list(zip(df_results['n_ephys_L'], df_results['n_ephys_R']))
    else:
        hemi_training = hemi_biased_ephys = hemi_ephys = None
    _add_bar_labels(ax, xpos - barwidth, df_results['n_training'].values, hemi_training)
    _add_bar_labels(ax, xpos, df_results['n_biased_ephys'].values, hemi_biased_ephys)
    _add_bar_labels(ax, xpos + barwidth, df_results['n_ephys'].values, hemi_ephys)

    ax.set_xticks(xpos)
    ax.set_xticklabels(df_results['target_NM'].values)
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
                ax.text(0.01, 0.8, r'$\rho$' + f'={corr.iloc[i, j]:.2f}', fontsize=6, transform=ax.transAxes)
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
                           window_label=None, aggregation='pool'):
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
        fontsize=10,
    )

    # Compute contrasts from the full dataset so both panels share the same x-axis
    contrasts = sorted(df_group['contrast'].unique()) if len(df_group) > 0 else []
    xpos = np.array(contrasts, dtype=float)

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
                    means.append(vals.mean())
                    sems.append(scipy_sem(vals, nan_policy='omit') if len(vals) > 1 else np.nan)
                else:
                    subj_means = df_c.groupby('subject')[response_col].mean()
                    means.append(subj_means.mean())
                    sems.append(scipy_sem(subj_means, nan_policy='omit') if len(subj_means) > 1 else np.nan)

            label = 'correct' if feedback == 1 else 'incorrect'
            ax.errorbar(xpos, means, yerr=np.array(sems, dtype=float),
                        marker='o', color=color, linestyle=ls, label=label)

        ax.set_xticks(xpos if len(xpos) else [])
        ax.set_xticklabels([f'{c:g}' for c in contrasts])
        ax.set_xlabel('Contrast (%)')
        ax.axhline(0, ls='--', color='gray', lw=0.5)

    # Invert contra axis so highest contrast is on the far left
    ax_c.invert_xaxis()

    ax_c.text(0.05, 0.02, 'Contra', ha='left', transform=ax_c.transAxes, fontsize=9)
    ax_i.text(0.95, 0.02, 'Ipsi', ha='right', transform=ax_i.transAxes, fontsize=9)

    ax_c.set_ylabel('z-score')
    ax_i.tick_params(left=False)
    ax_i.spines['left'].set_visible(False)
    ax_i.legend(frameon=False, loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)

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
    ax.set_xticklabels(coefficients.columns, rotation=90, fontsize=7)
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
    ax.set_yticklabels(df['feature'].values, fontsize=7)
    ax.set_xlabel(r'$\Delta$ accuracy')
    ax.set_title('Feature unique contribution')
    ax.axvline(0, color='k', linewidth=0.5)
    fig.tight_layout()
    return fig


import re as _re

_SIDE_ORDER = {'contra': 0, 'ipsi': 1}
_EVENT_ORDER = {'stimOn': 0, 'firstMovement': 1, 'feedback': 2}
_FB_ORDER = {'correct': 0, 'incorrect': 1}
_FEATURE_RE = _re.compile(
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
    axes[-1].set_xticklabels(response_matrix.columns, rotation=90, fontsize=7)
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


def plot_lmm_response(predictions, target_nm, event, fig=None,
                      window_label=None, df_raw=None, response_col=None):
    """Plot modeled response curves with 95% CI from an LMM fit.

    Layout mirrors ``plot_relative_contrast``: contra (left) and ipsi (right)
    panels, with correct/incorrect lines from the model predictions.

    Parameters
    ----------
    predictions : pd.DataFrame
        Output from ``fit_events_lmm().predictions`` with columns:
        contrast, side, reward, predicted, ci_lower, ci_upper.
    target_nm : str
        Target neuromodulator label (for title and color).
    event : str
        Event name (e.g. 'stimOn_times').
    fig : plt.Figure, optional
    window_label : str, optional
    df_raw : pd.DataFrame, optional
        Raw trial data for overlay. Must have contrast, side, feedbackType,
        and ``response_col``.
    response_col : str, optional
        Column in df_raw for raw data overlay.

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
    fig.suptitle(f'{target_nm} — {event_label} ({_window}) [LMM]', fontsize=10)

    contrasts = sorted(predictions['contrast'].unique())
    log_contrasts = contrast_transform(contrasts)

    for ax, side in ((ax_c, 'contra'), (ax_i, 'ipsi')):
        df_side = predictions[predictions['side'] == side]

        for reward, ls, label in ((1, '-', 'correct'), (0, '--', 'incorrect')):
            df_r = df_side[df_side['reward'] == reward].sort_values('contrast')
            if len(df_r) == 0:
                continue
            xvals = contrast_transform(df_r['contrast'].values)
            ax.plot(xvals, df_r['predicted'].values,
                    color=color, linestyle=ls, label=label, marker='o',
                    markersize=3)
            ax.fill_between(xvals,
                            df_r['ci_lower'].values, df_r['ci_upper'].values,
                            color=color, alpha=0.15)

        # Raw data overlay
        if df_raw is not None and response_col is not None:
            raw_side = df_raw[df_raw['side'] == side]
            for fb, marker in ((1, 'o'), (-1, 's')):
                raw_fb = raw_side[raw_side['feedbackType'] == fb]
                if len(raw_fb) == 0:
                    continue
                means = raw_fb.groupby('contrast')[response_col].mean()
                ax.scatter(contrast_transform(means.index), means.values,
                           color=color, marker=marker, alpha=0.3, s=15,
                           zorder=0)

        ax.set_xticks(log_contrasts if len(log_contrasts) else [])
        ax.set_xticklabels([f'{c:g}' for c in contrasts])
        ax.set_xlabel('Contrast')
        ax.axhline(0, ls='--', color='gray', lw=0.5)

    ax_c.invert_xaxis()
    ax_c.text(0.05, 0.02, 'Contra', ha='left', transform=ax_c.transAxes, fontsize=9)
    ax_i.text(0.95, 0.02, 'Ipsi', ha='right', transform=ax_i.transAxes, fontsize=9)
    ax_c.set_ylabel('z-score (modeled)')
    ax_i.tick_params(left=False)
    ax_i.spines['left'].set_visible(False)
    ax_i.legend(frameon=False, loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
    ax_i.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    return fig


def plot_lmm_variance_explained(ve_dict, fig=None):
    """Grouped bar chart of marginal and conditional R² per (target, event).

    Parameters
    ----------
    ve_dict : dict
        Keys are (target_nm, event) tuples, values are dicts with
        'marginal' and 'conditional' R² values.
    fig : plt.Figure, optional

    Returns
    -------
    plt.Figure
    """
    if fig is None:
        fig, ax = plt.subplots(1, 1, figsize=(max(4, len(ve_dict) * 1.5), 4),
                               layout='constrained')
    else:
        ax = fig.axes[0]

    if not ve_dict:
        ax.set_title('Variance explained (R²)')
        return fig

    labels = [f'{t}\n{e}' for t, e in ve_dict.keys()]
    marginal = [v['marginal'] for v in ve_dict.values()]
    conditional = [v['conditional'] for v in ve_dict.values()]

    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, marginal, width, label='Marginal R²', color='steelblue')
    ax.bar(x + width / 2, conditional, width, label='Conditional R²', color='coral')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('R²')
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    ax.set_title('Variance explained by LMM')

    return fig


def plot_marginal_means(emm_dict, event, fig=None):
    """Plot estimated marginal means for reward and side, per target-NM.

    Parameters
    ----------
    emm_dict : dict
        Keys are (target_NM, event_label) tuples. Values are dicts with
        'reward' and 'side' keys mapping to DataFrames from
        ``compute_marginal_means`` (columns: level, mean, ci_lower, ci_upper).
    event : str
        Event label to filter from emm_dict (e.g. 'stimOn').
    fig : plt.Figure, optional

    Returns
    -------
    plt.Figure
    """
    # Filter to the requested event
    targets = []
    for (tnm, ev), emms in emm_dict.items():
        if ev == event:
            targets.append((tnm, emms))
    targets.sort(key=lambda x: x[0])

    if fig is None:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4), layout='constrained')
    else:
        axes = fig.axes[:2]

    for ax, factor, level_labels in zip(
        axes,
        ['reward', 'side'],
        [{'0': 'incorrect', '1': 'correct', 0: 'incorrect', 1: 'correct'},
         {'contra': 'contra', 'ipsi': 'ipsi'}],
    ):
        for i, (tnm, emms) in enumerate(targets):
            emm = emms[factor]
            color = TARGETNM_COLORS.get(tnm, f'C{i}')
            levels = emm['level'].values
            means = emm['mean'].values
            ci_lo = emm['ci_lower'].values
            ci_hi = emm['ci_upper'].values
            yerr = np.array([means - ci_lo, ci_hi - means])

            x = np.arange(len(levels))
            ax.errorbar(x + i * 0.05 - 0.025 * len(targets),
                        means, yerr=yerr,
                        fmt='o', capsize=4, color=color, label=tnm,
                        markersize=5)

        # X-axis labels
        if len(targets) > 0:
            sample_emm = targets[0][1][factor]
            levels = sample_emm['level'].values
            x = np.arange(len(levels))
            ax.set_xticks(x)
            ax.set_xticklabels([level_labels.get(l, str(l)) for l in levels])

        ax.set_ylabel('z-score (EMM)')
        ax.set_title(f'Effect of {factor}')
        ax.axhline(0, ls='--', color='gray', lw=0.5)

    axes[-1].legend(frameon=False, loc='upper left', bbox_to_anchor=(1, 1),
                    fontsize=8)
    fig.suptitle(f'Estimated marginal means — {event}', fontsize=10)
    return fig


def plot_lmm_summary(group, event, fig=None):
    """Consolidated 4-panel LMM summary for one event.

    Reads all data from ``group.lmm_results``.

    Panels:
    1. R² (marginal and conditional) as paired dots per target-NM.
    2. Estimated marginal means for reward.
    3. Estimated marginal means for side.
    4. Contrast × reward slopes with subject-level random slopes.

    Parameters
    ----------
    group : PhotometrySessionGroup
        Must have ``lmm_results`` populated (via ``fit_lmm``).
    event : str
        Event label to plot (e.g. 'stimOn').
    fig : plt.Figure, optional

    Returns
    -------
    plt.Figure
    """
    lmm_results = group.lmm_results

    if fig is None:
        fig, axes = plt.subplots(1, 5, figsize=(16, 4), layout='constrained')
    else:
        axes = fig.axes[:5]

    ax_r2, ax_reward, ax_side, ax_contrast, ax_interaction = axes

    # Collect targets for this event
    targets = sorted(set(
        tnm for (tnm, ev) in lmm_results if ev == event
    ))

    # --- Panel 1: R² dots ---
    for i, tnm in enumerate(targets):
        key = (tnm, event)
        if key not in lmm_results:
            continue
        ve = lmm_results[key].variance_explained
        color = TARGETNM_COLORS.get(tnm, f'C{i}')
        ax_r2.scatter(i, ve['marginal'], color=color, marker='o', s=50,
                      zorder=3)
        ax_r2.scatter(i, ve['conditional'], color=color, marker='s', s=50,
                      zorder=3)
        ax_r2.plot([i, i], [ve['marginal'], ve['conditional']],
                   color=color, lw=1.5, zorder=2)

    ax_r2.set_xticks(range(len(targets)))
    ax_r2.set_xticklabels(targets, rotation=45, ha='right', fontsize=7)
    ax_r2.set_ylabel('R²')
    ax_r2.set_ylim(0, min(1, ax_r2.get_ylim()[1] * 1.2))
    ax_r2.set_title('Variance explained')
    from matplotlib.lines import Line2D
    ax_r2.legend(
        [Line2D([0], [0], marker='o', color='gray', ls='none', markersize=6),
         Line2D([0], [0], marker='s', color='gray', ls='none', markersize=6)],
        ['Marginal', 'Conditional'],
        frameon=False, fontsize=7, loc='upper left',
    )

    # Coefficient names for significance lookup
    _factor_coef = {
        'reward': 'C(reward)[T.1]',
        'side': 'C(side)[T.ipsi]',
    }

    # --- Panels 2 & 3: EMMs ---
    for ax, factor, emm_attr, level_labels in [
        (ax_reward, 'reward', 'emm_reward',
         {0: 'incorrect', 1: 'correct'}),
        (ax_side, 'side', 'emm_side',
         {'contra': 'contra', 'ipsi': 'ipsi'}),
    ]:
        for i, tnm in enumerate(targets):
            key = (tnm, event)
            if key not in lmm_results:
                continue
            lmm = lmm_results[key]
            emm = getattr(lmm, emm_attr)
            if emm is None:
                continue
            color = TARGETNM_COLORS.get(tnm, f'C{i}')
            levels = emm['level'].values
            means = emm['mean'].values
            ci_lo = emm['ci_lower'].values
            ci_hi = emm['ci_upper'].values
            yerr = np.array([means - ci_lo, ci_hi - means])
            x = np.arange(len(levels))
            offset = i * 0.06 - 0.03 * len(targets)

            # Filled if significant, open if not
            coef_name = _factor_coef.get(factor)
            sig = False
            if (coef_name is not None
                    and coef_name in lmm.summary_df.index):
                sig = lmm.summary_df.loc[coef_name, 'P>|z|'] < 0.05
            fs = 'full' if sig else 'none'

            ax.errorbar(x + offset, means, yerr=yerr,
                        fmt='o', capsize=3, color=color, label=tnm,
                        markersize=5, fillstyle=fs)

        if targets:
            key0 = (targets[0], event)
            if key0 in lmm_results:
                emm = getattr(lmm_results[key0], emm_attr)
                if emm is not None:
                    levels = emm['level'].values
                    x = np.arange(len(levels))
                    ax.set_xticks(x)
                    ax.set_xticklabels(
                        [level_labels.get(l, str(l)) for l in levels],
                        fontsize=8)

        ax.set_ylabel('z-score (EMM)')
        ax.set_title(f'Effect of {factor}')
        ax.axhline(0, ls='--', color='gray', lw=0.5)

    # --- Panel 4: Contrast EMM (main effect) ---
    for i, tnm in enumerate(targets):
        key = (tnm, event)
        if key not in lmm_results:
            continue
        lmm = lmm_results[key]
        emm_c = getattr(lmm, 'emm_contrast', None)
        if emm_c is None:
            continue
        color = TARGETNM_COLORS.get(tnm, f'C{i}')
        contrasts = emm_c['level'].values
        means = emm_c['mean'].values
        ci_lo = emm_c['ci_lower'].values
        ci_hi = emm_c['ci_upper'].values
        log_c = contrast_transform(contrasts)

        # Filled if log_contrast main effect is significant
        sig = False
        if 'log_contrast' in lmm.summary_df.index:
            sig = lmm.summary_df.loc['log_contrast', 'P>|z|'] < 0.05
        fs = 'full' if sig else 'none'

        ax_contrast.errorbar(log_c, means,
                             yerr=[means - ci_lo, ci_hi - means],
                             fmt='o-', capsize=3, color=color, label=tnm,
                             markersize=4, fillstyle=fs, lw=1)

    contrasts_all = [0.0, 0.0625, 0.125, 0.25, 1.0]
    log_ticks = contrast_transform(contrasts_all)
    ax_contrast.set_xticks(log_ticks)
    ax_contrast.set_xticklabels([f'{c:g}' for c in contrasts_all], fontsize=7)
    ax_contrast.set_xlabel('Contrast')
    ax_contrast.set_ylabel('z-score (EMM)')
    ax_contrast.set_title('Effect of contrast')
    ax_contrast.axhline(0, ls='--', color='gray', lw=0.5)

    # --- Panel 5: Contrast × reward interaction ---
    for i, tnm in enumerate(targets):
        key = (tnm, event)
        if key not in lmm_results:
            continue
        lmm = lmm_results[key]
        slopes = lmm.contrast_slopes
        if slopes is None:
            continue
        color = TARGETNM_COLORS.get(tnm, f'C{i}')
        pop = slopes[slopes['type'] == 'population']
        subj = slopes[slopes['type'] == 'subject']

        # Filled if interaction is significant
        interaction_name = 'log_contrast:C(reward)[T.1]'
        sig = False
        if interaction_name in lmm.summary_df.index:
            sig = lmm.summary_df.loc[interaction_name, 'P>|z|'] < 0.05
        fs = 'full' if sig else 'none'

        for _, row in pop.iterrows():
            reward = row['reward']
            x_pos = reward + i * 0.08 - 0.04 * len(targets)
            ax_interaction.errorbar(
                x_pos, row['slope'],
                yerr=[[row['slope'] - row['ci_lower']],
                      [row['ci_upper'] - row['slope']]],
                fmt='o', capsize=4, color=color, markersize=6, zorder=3,
                fillstyle=fs,
            )

        if len(subj) > 0:
            for _, row in subj.iterrows():
                reward = row['reward']
                x_pos = reward + i * 0.08 - 0.04 * len(targets) + 0.04
                ax_interaction.scatter(
                    x_pos, row['slope'],
                    color=color, alpha=0.4, s=15, zorder=2,
                )

    ax_interaction.set_xticks([0, 1])
    ax_interaction.set_xticklabels(['incorrect', 'correct'], fontsize=8)
    ax_interaction.set_ylabel('Contrast slope')
    ax_interaction.set_title('Contrast × reward')
    ax_interaction.axhline(0, ls='--', color='gray', lw=0.5)

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker='o', color=TARGETNM_COLORS.get(t, f'C{i}'),
                       ls='none', markersize=5) for i, t in enumerate(targets)]
    ax_interaction.legend(handles, targets, frameon=False, fontsize=6,
                          loc='upper left', bbox_to_anchor=(1, 1))

    fig.suptitle(f'LMM summary — {event}', fontsize=11)
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
    ax_delta.set_xticklabels(feature_order, rotation=90, fontsize=7)
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
    ax_resp.legend(frameon=False, fontsize=8)
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
    ax_delta.set_xticklabels(feature_order, rotation=90, fontsize=7)
    ax_delta.set_ylabel(r'$\Delta$ acc.')
    ax_delta.axhline(0, color='k', linewidth=0.5)

    return fig


def plot_mean_response_traces(traces_df, event_name, fig=None):
    """Mean peri-event response traces per target-NM with subject-mean removal.

    For each target_NM, applies subject-mean removal before computing
    the grand mean and SEM:
        adjusted_r(t) = r(t) - mean_s(t) + grand_mean(t)

    Parameters
    ----------
    traces_df : pd.DataFrame
        Long-form with columns: eid, subject, target_NM, brain_region,
        event, time, response.
    event_name : str
        Event name for the title.
    fig : plt.Figure, optional

    Returns
    -------
    plt.Figure
    """
    targets = sorted(traces_df['target_NM'].unique())
    n_targets = len(targets)

    if fig is None:
        fig, axes = plt.subplots(1, n_targets,
                                 figsize=(4 * n_targets, 3.5),
                                 sharey=True, squeeze=False)
        axes = axes[0]
    else:
        axes = fig.axes

    for ax, target in zip(axes, targets):
        df_t = traces_df[traces_df['target_NM'] == target]

        # Pivot to (recording, time) matrix
        recordings = df_t['eid'].unique()
        time_vals = sorted(df_t['time'].unique())
        n_recs = len(recordings)
        n_time = len(time_vals)

        trace_matrix = np.full((n_recs, n_time), np.nan)
        subjects_arr = []
        for i, eid in enumerate(recordings):
            rec_data = df_t[df_t['eid'] == eid].sort_values('time')
            trace_matrix[i] = rec_data['response'].values
            subjects_arr.append(rec_data['subject'].iloc[0])
        subjects_arr = np.array(subjects_arr)

        # Subject-mean removal
        grand_mean = np.nanmean(trace_matrix, axis=0)
        adjusted = np.copy(trace_matrix)
        for s in np.unique(subjects_arr):
            s_mask = subjects_arr == s
            s_mean = np.nanmean(trace_matrix[s_mask], axis=0)
            adjusted[s_mask] = trace_matrix[s_mask] - s_mean + grand_mean

        mean_trace = np.nanmean(adjusted, axis=0)
        if n_recs > 1:
            sem_trace = np.nanstd(adjusted, axis=0, ddof=1) / np.sqrt(n_recs)
        else:
            sem_trace = np.zeros(n_time)

        color = TARGETNM_COLORS.get(target, 'gray')
        ax.plot(time_vals, mean_trace, color=color, linewidth=1.5)
        if n_recs > 1:
            ax.fill_between(time_vals, mean_trace - sem_trace,
                            mean_trace + sem_trace,
                            color=color, alpha=0.3)
        ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
        ax.set_xlabel('Time (s)')
        ax.set_title(target)

    axes[0].set_ylabel('Response (z-scored ΔF/F)')
    fig.suptitle(event_name, fontsize=12)
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


def plot_wheel_lmm_summary(summary_df):
    """Plot delta R² across contrasts for each DV, one line per target_NM.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Output from ``PhotometrySessionGroup.fit_wheel_lmm()``.

    Returns
    -------
    matplotlib.Figure
    """
    dvs = ['reaction_time', 'movement_time', 'peak_velocity']
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

    if len(summary_df) == 0:
        for ax, dv in zip(axes, dvs):
            ax.set_title(_DV_LABELS.get(dv, dv))
            ax.set_xlabel('Contrast (%)')
        axes[0].set_ylabel('ΔR² (marginal)')
        fig.suptitle('NM activity contribution to behavioral vigor', fontsize=12)
        fig.tight_layout()
        return fig

    for ax, dv in zip(axes, dvs):
        df_dv = summary_df[summary_df['dv'] == dv]
        for target_nm, df_tnm in df_dv.groupby('target_NM'):
            df_tnm = df_tnm.sort_values('contrast')
            color = TARGETNM_COLORS.get(target_nm, 'gray')
            ax.plot(df_tnm['contrast'], df_tnm['delta_r2'],
                    'o-', color=color, label=target_nm, markersize=5)
            # Mark significant results (p < 0.05) with filled markers
            sig = df_tnm['lrt_pvalue'] < 0.05
            if sig.any():
                ax.scatter(df_tnm.loc[sig, 'contrast'],
                           df_tnm.loc[sig, 'delta_r2'],
                           color=color, s=40, zorder=5, edgecolors='black',
                           linewidths=0.5)

        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
        ax.set_title(_DV_LABELS.get(dv, dv))
        ax.set_xlabel('Contrast (%)')

    axes[0].set_ylabel('ΔR² (marginal)')
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[-1].legend(handles, labels, fontsize=8)
    fig.suptitle('NM activity contribution to behavioral vigor', fontsize=12)
    fig.tight_layout()
    return fig


def plot_wheel_nm_scatter(df, dv_col, response_col, target_nm,
                           contrast=None):
    """Scatter plot of behavioral DV vs NM response, faceted by contrast.

    Parameters
    ----------
    df : pd.DataFrame
        Trial-level data with ``dv_col``, ``response_col``, ``contrast``,
        and ``subject`` columns.
    dv_col : str
        Dependent variable column name.
    response_col : str
        NM response column name.
    target_nm : str
        Target-NM label for title.
    contrast : float, optional
        If provided, plot only this contrast level. Otherwise facet by all.

    Returns
    -------
    matplotlib.Figure
    """
    df = df.dropna(subset=[dv_col, response_col])
    color = TARGETNM_COLORS.get(target_nm, 'gray')

    if contrast is not None:
        contrasts = [contrast]
        df = df[np.isclose(df['contrast'], contrast)]
    else:
        contrasts = sorted(df['contrast'].unique())

    n_panels = len(contrasts)
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4),
                              sharey=True, squeeze=False)
    axes = axes[0]

    for ax, c in zip(axes, contrasts):
        df_c = df[np.isclose(df['contrast'], c)]
        if len(df_c) == 0:
            ax.set_title(f'{c}%')
            continue

        # Plot each subject
        for subj, df_s in df_c.groupby('subject'):
            ax.scatter(df_s[response_col], df_s[dv_col],
                       alpha=0.3, s=10, color=color)

        # Overall regression line
        if len(df_c) > 2:
            z = np.polyfit(df_c[response_col], df_c[dv_col], 1)
            x_line = np.linspace(df_c[response_col].min(),
                                  df_c[response_col].max(), 50)
            ax.plot(x_line, np.polyval(z, x_line), color='black',
                    linewidth=1.5, linestyle='--')

        cfmt = int(c) if c == int(c) else c
        ax.set_title(f'{cfmt}%')
        ax.set_xlabel(response_col.replace('_', ' '))

    axes[0].set_ylabel(_DV_LABELS.get(dv_col, dv_col))
    fig.suptitle(f'{target_nm}: {_DV_LABELS.get(dv_col, dv_col)} vs NM response',
                 fontsize=12)
    fig.tight_layout()
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
        fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    else:
        axes = fig.subplots(1, 4)

    # Panel 1: per-cohort canonical correlations
    ax = axes[0]
    colors = [TARGETNM_COLORS.get(t, 'gray') for t in targets]
    corrs = [cohort_results[t].correlations[0] for t in targets]
    bars = ax.bar(range(len(targets)), corrs, color=colors)
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(targets, rotation=45, ha='right')
    ax.set_ylabel('Canonical correlation')
    ax.set_title('Per-cohort CC1')

    # Annotate p-values
    for i, t in enumerate(targets):
        pv = cohort_results[t].p_values
        if pv is not None:
            ax.text(i, corrs[i] + 0.01, f'p={pv[0]:.3f}',
                    ha='center', va='bottom', fontsize=8)

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
    for i in range(len(targets)):
        for j in range(len(targets)):
            ax.text(j, i, f'{matrix.values[i, j]:.2f}',
                    ha='center', va='center', fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)

    # Panel 3: delta-r cross-projection (relative to within-cohort diagonal)
    ax = axes[2]
    diag = np.diag(matrix.values)  # within-cohort correlations
    delta_matrix = matrix.values - diag[np.newaxis, :]  # subtract column baseline
    max_abs = max(abs(np.nanmin(delta_matrix)), abs(np.nanmax(delta_matrix)), 0.01)
    im2 = ax.imshow(delta_matrix, cmap='RdBu_r', vmin=-max_abs, vmax=max_abs)
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(targets, rotation=45, ha='right')
    ax.set_yticks(range(len(targets)))
    ax.set_yticklabels(targets)
    ax.set_xlabel('Weight source')
    ax.set_ylabel('Data source')
    ax.set_title(r'$\Delta r$ (vs within-cohort)')
    for i in range(len(targets)):
        for j in range(len(targets)):
            ax.text(j, i, f'{delta_matrix[i, j]:+.2f}',
                    ha='center', va='center', fontsize=9)
    fig.colorbar(im2, ax=ax, shrink=0.8)

    # Panel 4: weight profiles (neural + behavioral combined)
    _plot_weight_heatmap_pair(cohort_results, targets, axes[3])

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
    ax.set_xticklabels(targets, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(all_names)))
    ax.set_yticklabels(all_names, fontsize=7)

    # Separator between neural and behavioral
    ax.axhline(n_neural - 0.5, color='black', linewidth=1.5)

    # Annotate values
    for i in range(combined.shape[0]):
        for j in range(combined.shape[1]):
            ax.text(j, i, f'{combined[i, j]:.2f}',
                    ha='center', va='center', fontsize=7,
                    color='white' if abs(combined[i, j]) > 0.6 * vmax else 'black')

    # Label the two sections
    ax.text(-0.7, (n_neural - 1) / 2, 'Neural', ha='right', va='center',
            fontsize=8, fontweight='bold', transform=ax.get_yaxis_transform())
    ax.text(-0.7, n_neural + (len(behav_names) - 1) / 2, 'Behav.',
            ha='right', va='center', fontsize=8, fontweight='bold',
            transform=ax.get_yaxis_transform())

    ax.set_title('CC1 weight profiles')
    ax.figure.colorbar(im, ax=ax, shrink=0.8, label='Weight')


def plot_cca_weight_profiles(cohort_results, fig=None):
    """Side-by-side heatmaps of neural and behavioral CC1 weights.

    Left panel: neural features (rows) × cohorts (columns).
    Right panel: behavioral features (rows) × cohorts (columns).
    Shared diverging colormap centered at zero.

    Parameters
    ----------
    cohort_results : dict[str, CCAResult]
        Per-cohort CCA fits (sign-aligned).
    fig : matplotlib.figure.Figure, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    targets = sorted(cohort_results.keys())

    neural_names = cohort_results[targets[0]].x_weights.index.tolist()
    behav_names = cohort_results[targets[0]].y_weights.index.tolist()

    neural_mat = np.column_stack(
        [cohort_results[t].x_weights['CC1'].values for t in targets])
    behav_mat = np.column_stack(
        [cohort_results[t].y_weights['CC1'].values for t in targets])

    vmax = max(np.max(np.abs(neural_mat)), np.max(np.abs(behav_mat)))

    if fig is None:
        fig, (ax_n, ax_b) = plt.subplots(
            1, 2, figsize=(4 + len(targets), max(len(neural_names),
                                                  len(behav_names)) * 0.5),
            gridspec_kw={'width_ratios': [len(neural_names),
                                           len(behav_names)]})
    else:
        ax_n, ax_b = fig.subplots(1, 2)

    # Neural weights
    im_n = ax_n.imshow(neural_mat, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                        aspect='auto')
    ax_n.set_xticks(range(len(targets)))
    ax_n.set_xticklabels(targets, rotation=45, ha='right')
    ax_n.set_yticks(range(len(neural_names)))
    ax_n.set_yticklabels(neural_names)
    ax_n.set_title('Neural CC1 weights')
    for i in range(neural_mat.shape[0]):
        for j in range(neural_mat.shape[1]):
            ax_n.text(j, i, f'{neural_mat[i, j]:.2f}',
                      ha='center', va='center', fontsize=8,
                      color='white' if abs(neural_mat[i, j]) > 0.6 * vmax
                      else 'black')

    # Behavioral weights
    im_b = ax_b.imshow(behav_mat, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                        aspect='auto')
    ax_b.set_xticks(range(len(targets)))
    ax_b.set_xticklabels(targets, rotation=45, ha='right')
    ax_b.set_yticks(range(len(behav_names)))
    ax_b.set_yticklabels(behav_names)
    ax_b.set_title('Behavioral CC1 weights')
    for i in range(behav_mat.shape[0]):
        for j in range(behav_mat.shape[1]):
            ax_b.text(j, i, f'{behav_mat[i, j]:.2f}',
                      ha='center', va='center', fontsize=8,
                      color='white' if abs(behav_mat[i, j]) > 0.6 * vmax
                      else 'black')

    fig.colorbar(im_b, ax=[ax_n, ax_b], shrink=0.8, label='CC1 weight')
    fig.tight_layout()
    return fig
