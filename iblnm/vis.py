import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import colors
from sklearn.preprocessing import quantile_transform

from iblnm import config
from iblnm.config import *
from iblnm.config import QCVAL2NUM


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
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    # set fig dimensions to produce desired ax dimensions
    figw = float(w)/(r-l)
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
        - 'good': highlight sessions where session_status == 'good'
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
    if highlight == 'good':
        highlight_mask = df['session_status'] == 'good'
    elif highlight == 'all':
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


def _add_bar_labels(ax, xpos, values, color='white'):
    """Add centered text labels to bars."""
    for x, n in zip(xpos, values):
        if n > 0:
            ax.text(x, n/2, str(int(n)), ha='center', va='center',
                    fontweight='bold', color=color)


def mouse_overview_barplot(df_sessions, min_biased_ephys=5, min_ephys=3, ax=None, barwidth=0.25):
    """
    Barplot showing mouse training progress per target region.

    Three bars per target:
    - training: mice with any training sessions
    - biased/ephys: mice with ≥min_biased_ephys combined biased+ephys sessions
    - ephys: mice with ≥min_ephys ephys sessions

    If 'hemisphere' column present, x-axis labels include hemisphere counts.
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

    results = []
    for target_nm in target_nms:
        target_data = session_counts[session_counts['target_NM'] == target_nm]

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

        results.append({
            'target_NM': target_nm,
            'n_training': len(training_mice),
            'n_biased_ephys': len(biased_ephys_mice),
            'n_ephys': len(ephys_mice),
        })

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

    # Add text labels
    _add_bar_labels(ax, xpos - barwidth, df_results['n_training'].values)
    _add_bar_labels(ax, xpos, df_results['n_biased_ephys'].values)
    _add_bar_labels(ax, xpos + barwidth, df_results['n_ephys'].values)

    # Format axes with hemisphere counts if available
    ax.set_xticks(xpos)
    if 'hemisphere' in df_sessions.columns:
        labels = []
        for target_nm in df_results['target_NM']:
            target_data = df_sessions[df_sessions['target_NM'] == target_nm]
            n_left = target_data[target_data['hemisphere'] == 'L']['subject'].nunique()
            n_right = target_data[target_data['hemisphere'] == 'R']['subject'].nunique()
            labels.append(f'{target_nm}\n({n_left}L, {n_right}R)')
        ax.set_xticklabels(labels)
    else:
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
