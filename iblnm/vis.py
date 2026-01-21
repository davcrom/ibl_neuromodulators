import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import colors

from iblnm import config
from iblnm.config import *
from iblnm.config import QCVAL2NUM

# Create colormap for QC grid plots
QCCMAP = colors.LinearSegmentedColormap.from_list(
    'qc_cmap',
    [(0., 'white'), (0.01, 'gray'), (0.1, 'palevioletred'), (0.33, 'violet'), (0.66, 'orange'), (1., 'limegreen')],
    N=256
)

# Target-NM colors
TARGETNM_COLORS = {
    'MR-5HT': '#df67faff',
    'DR-5HT': '#b867faff',
    'VTA-DA': '#ff413dff',
    'SNc-DA': '#ff653dff',
    'LC-NE': '#3f88faff',
    'NBM-ACh': '#40afa1ff',
    'SI-ACh': '#40afa1ff',
    'PPT-ACh': '#00974eff',
}

TARGETNM_POSITIONS = {
    'MR-5HT': 0,
    'DR-5HT': 1,
    'VTA-DA': 2,
    'SNc-DA': 3,
    'LC-NE': 4,
    'NBM-ACh': 5,
    'SI-ACh': 6,
    'PPT-ACh': 7,
}

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


def session_overview_matrix(df, columns='day_n', highlight='good', ax=None):
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

    xpos = [TARGETNM2POSITION[target_NM] for target_NM in df_n.index]
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


def qc_grid(df, qc_columns=None, qcval2num=None, ax=None, yticklabels=None,
           legend=True):
    if qcval2num is None:
        qcval2num = QCVAL2NUM
    if qc_columns is None:
        qc_columns = df.columns
    df_qc = df[qc_columns].replace(qcval2num)
    if ax is None:
        fig, ax = plt.subplots()
    qcmat = df_qc.values.astype(float)
    ax.matshow(qcmat, cmap=QCCMAP, vmin=0, vmax=1, aspect='auto')
    ax.set_yticks(np.arange(len(df_qc)))
    if type(yticklabels) == str:
        ax.set_xticklabels(df[yticklabels])
    elif type(yticklabels) == list:
        yticklabels = df.apply(lambda x: '_'.join(x[yticklabels].astype(str)), axis='columns')
        ax.set_yticklabels(yticklabels)
    ax.set_xticks(np.arange(len(df_qc.columns)))
    ax.set_xticklabels(qc_columns)
    ax.tick_params(axis='x', rotation=90)
    for xtick in ax.get_xticks():
        ax.axvline(xtick - 0.5, color='white')
    for ytick in ax.get_yticks():
        ax.axhline(ytick - 0.5, color='white')
    if legend:
        for key, val in qcval2num.items():
            ax.scatter(-1, -1, color=QCCMAP(val), label=key)
        ax.set_xlim(left=-0.5)
        ax.set_ylim(top=-0.5)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
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

    X = df[metrics].dropna().values
    corr = df[metrics].dropna().corr(method='spearman')
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
