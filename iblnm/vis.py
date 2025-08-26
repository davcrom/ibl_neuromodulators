import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors

from iblnm import config
from iblnm.config import QCVAL2NUM

LABELFONTSIZE = 6
plt.rcParams['figure.dpi'] = 180
plt.rcParams['axes.labelsize'] = LABELFONTSIZE
plt.rcParams['xtick.labelsize'] = LABELFONTSIZE 
plt.rcParams['ytick.labelsize'] = LABELFONTSIZE 
plt.rcParams['legend.fontsize'] = LABELFONTSIZE 
plt.rcParams['axes.titlesize'] = LABELFONTSIZE 

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


def session_overview_matrix(df, index='subject', columns='day', ax=None):
    df = df.copy()
    df['session_type_float'] = df['session_type'].map(config.SESSIONTYPE2FLOAT)  # convert session type to numerical value

    def _raise_error_on_duplicate(x):
        if len(x) > 1:
            raise ValueError(f"Duplicate entries found: {list(x)}")
        return x.iloc[0]
    
    subject_matrix = df.pivot_table(
        index=index, 
        columns=columns, 
        values='session_type_float',
        aggfunc=_raise_error_on_duplicate,  # all duplicates should be gone
        fill_value=0
    )
    
    # Create categorical colormap
    color_list = ['white'] + list(config.SESSIONTYPE2COLOR.values())
    cmap = colors.ListedColormap(color_list)
    
    # Create boundaries for discrete categories
    bounds = [0] + list(config.SESSIONTYPE2FLOAT.values()) + [1.01]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    
    # Plot the matrix with the custom colormap (figsize is determined by dimensions)
    if ax is None:
        fig, ax = plt.subplots(figsize=(0.15 * len(subject_matrix.columns), 0.15 * len(subject_matrix)))
    im = ax.matshow(subject_matrix, cmap=cmap, norm=norm)
    
    # Format axes
    ax.set_yticks(np.arange(len(subject_matrix)))
    ax.set_yticklabels(subject_matrix.index)
    ax.set_ylabel('Subject')
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(len(subject_matrix.columns)))
    ax.tick_params(axis='x', rotation=90)
    ax.xaxis.set_label_position('top')
    ax.set_xlabel('Days since first training')
    
    # Add custom gridlines
    for xtick in ax.get_xticks():
        ax.axvline(xtick - 0.5, color='white')
    for ytick in ax.get_yticks():
        ax.axhline(ytick - 0.5, color='white')
    
    # Calculate tick positions at the center of each color segment for the colorbar
    tick_positions = [(bounds[i] + bounds[i + 1]) / 2 for i in range(1, len(bounds) - 1)]  # skip the first entry for 0/absent
    tick_labels = list(config.SESSIONTYPE2FLOAT.keys())  # these keys don't include 0/absent
    
    # Plot the colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.5, boundaries=bounds, ticks=tick_positions)
    cbar.set_ticklabels(list(config.SESSIONTYPE2COLOR.keys()))
    cbar.ax.set_ylim(bounds[1], bounds[-1])  # exclude 0/absent

    return fig, ax


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
