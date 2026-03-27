import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import sem as scipy_sem
from sklearn.preprocessing import quantile_transform

from iblnm.config import (
    ANALYSIS_CONTRASTS, NM_CMAPS, QCCMAP, RESPONSE_WINDOWS,
    SESSIONTYPE2COLOR, SESSIONTYPE2FLOAT, TARGETNM2POSITION,
    TARGETNM_COLORS, TARGETNM_POSITIONS,
    get_contrast_coding,
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


def session_overview_matrix(df, columns='session_n', highlight='good', ax=None,
                            subject_order=None):
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

    if subject_order is not None:
        subject_matrix = subject_matrix.reindex(subject_order).fillna(0)
        overlay_matrix = overlay_matrix.reindex(subject_order).fillna(0)

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
        fontsize=10,
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

    ax_c.text(0.05, 0.02, 'Contra', ha='left', transform=ax_c.transAxes, fontsize=9)
    ax_i.text(0.95, 0.02, 'Ipsi', ha='right', transform=ax_i.transAxes, fontsize=9)

    ax_c.set_ylabel('$\Delta$ activity (z-score)')
    ax_i.tick_params(left=False)
    ax_i.spines['left'].set_visible(False)
    ax_i.legend(frameon=False, loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)

    ax_i.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    return fig

## Broken trial number thresholding logic
def plot_relative_contrast_per_subject(df_group, response_col, target_nm, event,
                                        window_label=None, min_trials=10):
    """Per-subject contrast response curves, same layout as plot_relative_contrast.

    Each subject gets its own line (correct=solid, incorrect=dashed).
    Grand mean overlaid in black.
    """
    fig, (ax_c, ax_i) = plt.subplots(1, 2, sharey=True,
                                      gridspec_kw={'wspace': 0.05},
                                      layout='constrained')

    event_label = event.replace('_times', '')
    _window = window_label or ''
    n_sessions = df_group['eid'].nunique() if 'eid' in df_group.columns else '?'
    subjects = sorted(df_group['subject'].unique())
    n_subjects = len(subjects)
    fig.suptitle(
        f'{target_nm} — {event_label} ({_window})\n'
        f'{n_sessions} sessions, {n_subjects} subjects',
        fontsize=10,
    )

    contrasts = sorted(df_group['contrast'].unique()) if len(df_group) > 0 else []
    ranks = list(range(len(contrasts)))
    rank_map = dict(zip(contrasts, ranks))

    cmap = plt.cm.tab20
    subj_colors = {s: cmap(i % 20) for i, s in enumerate(subjects)}

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

            for subj in subjects:
                df_s = df_fb[df_fb['subject'] == subj]
                if len(df_s) == 0:
                    continue
                means = []
                for c in contrasts:
                    vals = df_s.loc[df_s['contrast'] == c, response_col].dropna()
                    means.append(vals.mean() if len(vals) > 0 else np.nan)
                xpos = np.array([rank_map[c] for c in contrasts])
                ax.plot(xpos, means, marker='.', markersize=3,
                        color=subj_colors[subj], linestyle=ls,
                        alpha=0.4, linewidth=0.8)

            # Grand mean (mean of subject means)
            if len(df_fb) > 0:
                grand_means, sems = [], []
                for c in contrasts:
                    subj_means = df_fb[df_fb['contrast'] == c].groupby(
                        'subject')[response_col].apply(
                            lambda x: np.mean(x) if len(x) > 0 else np.nan
                            )
                    grand_means.append(subj_means.mean())
                    sems.append(scipy_sem(subj_means, nan_policy='omit') if len(subj_means) > 0 else np.nan)
                xpos = np.array([rank_map[c] for c in contrasts])
                label = 'correct' if feedback == 1 else 'incorrect'
                # ~ ax.plot(xpos, grand_means, marker='o',
                        # ~ color='black', linestyle=ls, linewidth=2, label=label)
                ax.errorbar(xpos, grand_means, yerr=np.array(sems, dtype=float),
                        marker='o', color='black', linestyle=ls, label=label)

        ax.set_xticks(ranks)
        ax.set_xticklabels([f'{c:g}' for c in contrasts])
        ax.set_xlabel('Contrast level')
        ax.set_yticks([-1, 0, 1, 2])
        ax.axhline(0, ls='--', color='gray', lw=0.5)

    ax_c.invert_xaxis()
    ax_c.text(0.05, 0.02, 'Contra', ha='left', transform=ax_c.transAxes, fontsize=9)
    ax_i.text(0.95, 0.02, 'Ipsi', ha='right', transform=ax_i.transAxes, fontsize=9)
    ax_c.set_ylabel('$\Delta$ activity (z-score)')
    ax_i.tick_params(left=False)
    ax_i.spines['left'].set_visible(False)
    ax_i.legend(frameon=False, loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
    ax_i.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    return fig
    

## From ChatGPT
# ~ def plot_relative_contrast_per_subject(df_group, response_col, target_nm, event,
                                       # ~ window_label=None, min_trials=10):
    # ~ """Per-subject contrast response curves, same layout as plot_relative_contrast.

    # ~ Each subject gets its own line (correct=solid, incorrect=dashed).
    # ~ Grand mean overlaid in black.
    # ~ """
    # ~ fig, (ax_c, ax_i) = plt.subplots(
        # ~ 1, 2, sharey=True,
        # ~ gridspec_kw={'wspace': 0.05},
        # ~ layout='constrained'
    # ~ )

    # ~ event_label = event.replace('_times', '')
    # ~ _window = window_label or ''
    # ~ n_sessions = df_group['eid'].nunique() if 'eid' in df_group.columns else '?'
    # ~ subjects = sorted(df_group['subject'].unique())
    # ~ n_subjects = len(subjects)

    # ~ fig.suptitle(
        # ~ f'{target_nm} — {event_label} ({_window})\n'
        # ~ f'{n_sessions} sessions, {n_subjects} subjects',
        # ~ fontsize=10,
    # ~ )

    # ~ contrasts = sorted(df_group['contrast'].unique()) if len(df_group) > 0 else []
    # ~ ranks = list(range(len(contrasts)))
    # ~ rank_map = dict(zip(contrasts, ranks))

    # ~ cmap = plt.cm.tab20
    # ~ subj_colors = {s: cmap(i % 20) for i, s in enumerate(subjects)}

    # ~ # Subject-mean removal: subtract per-subject mean, add grand mean
    # ~ if len(df_group) > 0:
        # ~ grand_mean = df_group[response_col].mean()
        # ~ subj_means = df_group.groupby('subject')[response_col].transform('mean')
        # ~ df_group = df_group.copy()
        # ~ df_group[response_col] = df_group[response_col] - subj_means + grand_mean

    # ~ for ax, side in ((ax_c, 'contra'), (ax_i, 'ipsi')):
        # ~ df_side = df_group[df_group['side'] == side]

        # ~ for feedback, ls in ((1, '-'), (-1, '--')):
            # ~ df_fb = df_side[df_side['feedbackType'] == feedback]

            # ~ # --- subject lines ---
            # ~ for subj in subjects:
                # ~ df_s = df_fb[df_fb['subject'] == subj]
                # ~ if len(df_s) == 0:
                    # ~ continue

                # ~ means = []
                # ~ for c in contrasts:
                    # ~ vals = df_s.loc[df_s['contrast'] == c, response_col].dropna()
                    # ~ means.append(vals.mean() if len(vals) > min_trials else np.nan)

                # ~ xpos = np.array([rank_map[c] for c in contrasts])

                # ~ ax.plot(
                    # ~ xpos, means,
                    # ~ marker='.',
                    # ~ markersize=3,
                    # ~ color=subj_colors[subj],
                    # ~ linestyle=ls,
                    # ~ alpha=0.4,
                    # ~ linewidth=0.8
                # ~ )

            # ~ # --- grand mean (mean of subject means) ---
            # ~ if len(df_fb) > 0:
                # ~ grand_means = []
                # ~ for c in contrasts:
                    # ~ subj_means = (
                        # ~ df_fb[df_fb['contrast'] == c]
                        # ~ .groupby('subject')[response_col]
                        # ~ .apply(lambda x: np.mean(x) if len(x) > min_trials else np.nan)
                    # ~ )
                    # ~ grand_means.append(subj_means.mean())

                # ~ xpos = np.array([rank_map[c] for c in contrasts])
                # ~ label = 'correct' if feedback == 1 else 'incorrect'

                # ~ ax.plot(
                    # ~ xpos, grand_means,
                    # ~ marker='o',
                    # ~ color='black',
                    # ~ linestyle=ls,
                    # ~ linewidth=2,
                    # ~ label=label
                # ~ )

        # ~ ax.set_xticks(ranks)
        # ~ ax.set_xticklabels([f'{c:g}' for c in contrasts])
        # ~ ax.set_xlabel('Contrast (rank)')
        # ~ ax.axhline(0, ls='--', color='gray', lw=0.5)

    # ~ ax_c.invert_xaxis()

    # ~ ax_c.text(0.05, 0.02, 'Contra', ha='left',
              # ~ transform=ax_c.transAxes, fontsize=9)
    # ~ ax_i.text(0.95, 0.02, 'Ipsi', ha='right',
              # ~ transform=ax_i.transAxes, fontsize=9)

    # ~ ax_c.set_ylabel('z-score')

    # ~ ax_i.tick_params(left=False)
    # ~ ax_i.spines['left'].set_visible(False)

    # ~ ax_i.legend(frameon=False, loc='upper left',
                # ~ bbox_to_anchor=(1, 1), fontsize=8)

    # ~ ax_i.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # ~ return fig
    

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
                      window_label=None, df_raw=None, response_col=None,
                      contrast_coding='log'):
    """Plot modeled response curves with 95% CI from an LMM fit.

    Layout mirrors ``plot_relative_contrast``: contra (left) and ipsi (right)
    panels, with correct/incorrect lines from the model predictions.

    Parameters
    ----------
    predictions : pd.DataFrame
        Output from ``fit_response_lmm().predictions`` with columns:
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

    _transform, _ = get_contrast_coding(contrast_coding)

    contrasts = sorted(predictions['contrast'].unique())
    coded_contrasts = _transform(contrasts)

    for ax, side in ((ax_c, 'contra'), (ax_i, 'ipsi')):
        df_side = predictions[predictions['side'] == side]

        for reward, ls, label in ((1, '-', 'correct'), (0, '--', 'incorrect')):
            df_r = df_side[df_side['reward'] == reward].sort_values('contrast')
            if len(df_r) == 0:
                continue
            xvals = _transform(df_r['contrast'].values)
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
                ax.scatter(_transform(means.index), means.values,
                           color=color, marker=marker, alpha=0.3, s=15,
                           zorder=0)

        ax.set_xticks(coded_contrasts if len(coded_contrasts) else [])
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


def _pval_to_stars(p):
    """Convert p-value to significance stars."""
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return ''


def plot_lmm_coefficient_heatmap(df_coefs):
    """Heatmap of LMM coefficients per event: targets × terms.

    Drops the Intercept term (uninterpretable for firstMovement and feedback
    due to prior-event contamination of the baseline). Colorbar scale is
    shared across all events for comparability.

    Parameters
    ----------
    df_coefs : pd.DataFrame
        LMM coefficients with columns: term, target_NM, event, Coef., P>|z|.

    Returns
    -------
    dict[str, plt.Figure]
        One figure per event, keyed by event name.
    """
    # Canonical term order (Intercept excluded)
    term_order = [
        'side', 'reward', 'log_contrast',
        'side:reward', 'log_contrast:side',
        'log_contrast:reward', 'log_contrast:side:reward',
    ]
    short = {
        'side': 'side',
        'reward': 'reward',
        'log_contrast': 'contrast',
        'side:reward': 'side×reward',
        'log_contrast:side': 'contrast×side',
        'log_contrast:reward': 'contrast×reward',
        'log_contrast:side:reward': 'contrast×side×reward',
    }

    df_coefs = df_coefs[df_coefs['term'] != 'Intercept']
    events = sorted(df_coefs['event'].unique())
    targets = sorted(df_coefs['target_NM'].unique())

    # Global vmax across all events for shared colorbar scale
    vmax = df_coefs['Coef.'].abs().max()

    figs = {}
    for event in events:
        df_ev = df_coefs[df_coefs['event'] == event]
        present_terms = [t for t in term_order if t in df_ev['term'].values]
        present_terms += sorted(set(df_ev['term']) - set(term_order))

        coef_matrix = np.full((len(targets), len(present_terms)), np.nan)
        pval_matrix = np.ones((len(targets), len(present_terms)))
        for i, tnm in enumerate(targets):
            for j, term in enumerate(present_terms):
                row = df_ev[(df_ev['target_NM'] == tnm)
                            & (df_ev['term'] == term)]
                if len(row) == 1:
                    coef_matrix[i, j] = row['Coef.'].iloc[0]
                    pval_matrix[i, j] = row['P>|z|'].iloc[0]

        col_labels = [short.get(t, t) for t in present_terms]

        fig, ax = plt.subplots(
            figsize=(0.9 * len(present_terms) + 1.5, 0.6 * len(targets) + 1))
        im = ax.imshow(coef_matrix, aspect='auto', cmap='RdBu_r',
                        vmin=-vmax, vmax=vmax)

        # Asterisk annotations
        for i in range(len(targets)):
            for j in range(len(present_terms)):
                stars = _pval_to_stars(pval_matrix[i, j])
                if stars:
                    ax.text(j, i, stars, ha='center', va='center',
                            fontsize=9, fontweight='bold',
                            color='k' if abs(coef_matrix[i, j]) < 0.6 * vmax
                            else 'w')

        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(targets)))
        ax.set_yticklabels(targets, fontsize=9)
        ax.set_title(event, fontsize=11)
        fig.colorbar(im, ax=ax, label='Coefficient', shrink=0.8)
        fig.tight_layout()
        figs[event] = fig

    return figs


def plot_lmm_summary(group, event, fig=None):
    """5-panel LMM summary for one event.

    Panels:
    1. R² (fixed effects and fixed+random) as paired dots per target-NM.
    2. Contrast × reward interaction plot.
    3. Contrast × side interaction plot.
    4. Reward × side interaction plot.
    5. Coefficient heatmap (targets × terms, color = coefficient, stars = p).

    Interaction panels: each target-NM gets its own x-position with the two
    levels of the x-factor offset left/right. Dot fill encodes the y-factor
    main effect significance; connecting line style encodes the interaction
    significance (solid = p < 0.05, dashed = not).

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
    from matplotlib.lines import Line2D

    lmm_results = group.lmm_results

    if fig is None:
        fig = plt.figure(figsize=(14, 8), layout='constrained')
        gs = fig.add_gridspec(2, 6)
        axes = [
            fig.add_subplot(gs[0, :2]),   # R²
            fig.add_subplot(gs[0, 2:]),   # coefficient heatmap
            fig.add_subplot(gs[1, :2]),   # contrast × reward
            fig.add_subplot(gs[1, 2:4]),  # contrast × side
            fig.add_subplot(gs[1, 4:]),   # reward × side
        ]
    else:
        axes = fig.axes[:5]

    ax_r2 = axes[0]

    targets = sorted(set(
        tnm for (tnm, ev) in lmm_results if ev == event
    ))

    # --- Panel 1: R² side-by-side bars ---
    bar_w = 0.35
    for i, tnm in enumerate(targets):
        key = (tnm, event)
        if key not in lmm_results:
            continue
        ve = lmm_results[key].variance_explained
        color = TARGETNM_COLORS.get(tnm, f'C{i}')
        ax_r2.bar(i - bar_w / 2, ve['marginal'], width=bar_w,
                  color=color, alpha=0.5, label='Fixed' if i == 0 else '')
        ax_r2.bar(i + bar_w / 2, ve['conditional'], width=bar_w,
                  color=color, alpha=1.0, label='Fixed + random' if i == 0 else '')

    ax_r2.set_xticks(range(len(targets)))
    ax_r2.set_xticklabels(targets, rotation=45, ha='right', fontsize=7)
    ax_r2.set_ylabel('R²')
    ax_r2.set_ylim(0, min(1, ax_r2.get_ylim()[1] * 1.2))
    ax_r2.set_title('Variance explained')
    ax_r2.legend(frameon=False, fontsize=7, loc='upper left')

    # --- Panels 2–4: Interaction plots ---
    _interaction_specs = [
        # (ax_idx, attr, y_label, title, x_labels, y_main_coef)
        (2, 'interaction_contrast_reward', 'Contrast slope',
         'Contrast × reward',
         {'incorrect': 'incorrect', 'correct': 'correct'},
         'log_contrast'),
        (3, 'interaction_contrast_side', 'Contrast slope',
         'Contrast × side',
         {'contra': 'contra', 'ipsi': 'ipsi'},
         'log_contrast'),
        (4, 'interaction_reward_side', 'Reward effect',
         'Reward × side',
         {'contra': 'contra', 'ipsi': 'ipsi'},
         'reward'),
    ]

    for ax_idx, attr, ylabel, title, x_labels, y_coef in \
            _interaction_specs:
        ax = axes[ax_idx]
        for i, tnm in enumerate(targets):
            key = (tnm, event)
            if key not in lmm_results:
                continue
            lmm = lmm_results[key]
            idf = getattr(lmm, attr, None)
            if idf is None:
                continue
            color = TARGETNM_COLORS.get(tnm, f'C{i}')

            # Dot fill: y-factor main effect significance
            y_sig = (y_coef in lmm.summary_df.index
                     and lmm.summary_df.loc[y_coef, 'P>|z|'] < 0.05)
            fs = 'full' if y_sig else 'none'

            # Line style: interaction significance (from interaction DataFrame)
            int_sig = idf['p_interaction'].iloc[0] < 0.05
            ls = '-' if int_sig else '--'

            x_levels = idf['x_level'].values
            effects = idf['effect'].values
            ci_lo = idf['ci_lower'].values
            ci_hi = idf['ci_upper'].values

            # Position: each target at its own x, two levels offset
            x_center = i
            dx = 0.15
            x_pos = [x_center - dx, x_center + dx]

            for j in range(len(x_levels)):
                ax.errorbar(
                    x_pos[j], effects[j],
                    yerr=[[effects[j] - ci_lo[j]], [ci_hi[j] - effects[j]]],
                    fmt='o', capsize=4, color=color, markersize=6,
                    zorder=3, fillstyle=fs,
                )
            ax.plot(x_pos, effects, color=color, ls=ls, lw=1.5, zorder=2)

        # X-axis: label each target position with target name,
        # and add level labels as a secondary indication
        ax.set_xticks(range(len(targets)))
        ax.set_xticklabels(targets, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.axhline(0, ls='--', color='gray', lw=0.5)

        # Add level labels inside the plot, at the bottom
        if targets:
            level_names = list(x_labels.values())
            dx = 0.15
            for j, name in enumerate(level_names):
                x_pos = -dx if j == 0 else dx
                ax.annotate(
                    name, xy=(x_pos, 0.02), xycoords=('data', 'axes fraction'),
                    fontsize=6, ha='center', va='bottom', color='k',
                    rotation=90,
                )

    # Legend on interaction panels
    target_handles = [
        Line2D([0], [0], marker='o', color=TARGETNM_COLORS.get(t, f'C{i}'),
               ls='none', markersize=5)
        for i, t in enumerate(targets)
    ]
    encoding_handles = [
        Line2D([0], [0], marker='o', color='gray', ls='none', markersize=5,
               fillstyle='full'),
        Line2D([0], [0], marker='o', color='gray', ls='none', markersize=5,
               fillstyle='none'),
        Line2D([0], [0], color='gray', ls='-', lw=1.2),
        Line2D([0], [0], color='gray', ls='--', lw=1.2),
    ]
    encoding_labels = ['main effect p<0.05', 'main effect n.s.',
                       'interaction p<0.05', 'interaction n.s.']
    axes[4].legend(
        target_handles + encoding_handles,
        targets + encoding_labels,
        frameon=False, fontsize=6, loc='upper left', bbox_to_anchor=(1, 1),
    )

    # --- Panel 2 (top right): Coefficient heatmap ---
    ax_hm = axes[1]
    _term_order = [
        'log_contrast', 'side', 'reward',
        'log_contrast:side', 'log_contrast:reward',
        'side:reward', 'log_contrast:side:reward',
    ]
    _short = {
        'log_contrast': 'contrast',
        'side': 'side',
        'reward': 'reward',
        'log_contrast:side': 'con×side',
        'log_contrast:reward': 'con×rew',
        'side:reward': 'side×rew',
        'log_contrast:side:reward': 'con×side×rew',
    }

    # Collect coefficients for this event
    present_terms = []
    for term in _term_order:
        for tnm in targets:
            key = (tnm, event)
            if key in lmm_results and term in lmm_results[key].summary_df.index:
                present_terms.append(term)
                break

    if present_terms:
        coef_matrix = np.full((len(targets), len(present_terms)), np.nan)
        pval_matrix = np.ones((len(targets), len(present_terms)))
        for i, tnm in enumerate(targets):
            key = (tnm, event)
            if key not in lmm_results:
                continue
            sdf = lmm_results[key].summary_df
            for j, term in enumerate(present_terms):
                if term in sdf.index:
                    coef_matrix[i, j] = sdf.loc[term, 'Coef.']
                    pval_matrix[i, j] = sdf.loc[term, 'P>|z|']

        vmax = np.nanmax(np.abs(coef_matrix))

        im = ax_hm.imshow(coef_matrix, aspect='auto', cmap='RdBu_r',
                           vmin=-vmax, vmax=vmax)
        for i in range(len(targets)):
            for j in range(len(present_terms)):
                stars = _pval_to_stars(pval_matrix[i, j])
                if stars:
                    ax_hm.text(
                        j, i, stars, ha='center', va='center',
                        fontsize=8, fontweight='bold',
                        color='k' if abs(coef_matrix[i, j]) < 0.6 * vmax
                        else 'w')

        col_labels = [_short.get(t, t) for t in present_terms]
        ax_hm.set_xticks(range(len(col_labels)))
        ax_hm.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=7)
        ax_hm.set_yticks(range(len(targets)))
        ax_hm.set_yticklabels(targets, fontsize=7)
        ax_hm.set_title('Coefficients')
        fig.colorbar(im, ax=ax_hm, label='β', shrink=0.8)

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


def plot_mean_response_traces(traces_df, target_nm, min_trials=10,
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
    _EVENT_ORDER = ['stimOn_times', 'firstMovement_times', 'feedback_times']
    df = traces_df[traces_df['target_NM'] == target_nm].copy()
    present = set(df['event'].unique())
    events = [e for e in _EVENT_ORDER if e in present]
    # Append any events not in the canonical order
    events += sorted(present - set(_EVENT_ORDER))
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

            # Shaded response windows
            early = RESPONSE_WINDOWS['early']
            ax.axvspan(early[0], early[1], alpha=0.12, color='gray',
                       zorder=0)
            if event == 'feedback_times':
                late = RESPONSE_WINDOWS['late']
                ax.axvspan(late[0], late[1], alpha=0.08, color='gray',
                           zorder=0)

            if row == 1:
                ax.set_xlabel('Time (s)')
            if col == 0:
                ax.set_ylabel(fb_labels[fb])
            if row == 0:
                event_label = event.replace('_times', '')
                ax.set_title(event_label)

    # Legend on first axis
    axes[0, 0].legend(title='Contrast', fontsize=7, title_fontsize=8,
                      loc='upper left')

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
        fontsize=8, ha='right', va='top', color='k',
    )

    fig.suptitle(target_nm, fontsize=12)
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


def plot_glm_pca_weights(pca_result, fig=None):
    """Heatmap of PCA component loadings on GLM coefficients.

    Parameters
    ----------
    pca_result : GLMPCAResult
        Output from ``pca_glm_coefficients``.
    fig : plt.Figure, optional

    Returns
    -------
    plt.Figure
    """
    n_pcs = pca_result.components.shape[0]
    if fig is None:
        fig, ax = plt.subplots(figsize=(8, 0.6 * n_pcs + 1.5),
                               layout='constrained')
    else:
        ax = fig.axes[0]

    data = pca_result.components
    vmax = np.abs(data).max()
    im = ax.imshow(data, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)

    # Labels
    short = {
        'log_contrast': 'contrast',
        'log_contrast:side': 'contrast×side',
        'log_contrast:reward': 'contrast×reward',
        'side:reward': 'side×reward',
    }
    xlabels = [short.get(f, f) for f in pca_result.feature_names]
    ylabels = [
        f'PC{i+1} ({pca_result.explained_variance_ratio[i]:.0%})'
        for i in range(n_pcs)
    ]
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=45, ha='right')
    ax.set_yticks(range(n_pcs))
    ax.set_yticklabels(ylabels)

    # Annotate cells
    for i in range(n_pcs):
        for j in range(len(xlabels)):
            val = data[i, j]
            color = 'white' if abs(val) > 0.5 * vmax else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=8, color=color)

    fig.colorbar(im, ax=ax, shrink=0.8, label='Loading')
    return fig


def plot_glm_pca_scores(pca_result, n_pcs=3, stats=None, fig=None):
    """Score distributions per target-NM for top PCs.

    Parameters
    ----------
    pca_result : GLMPCAResult
        Output from ``pca_glm_coefficients``.
    n_pcs : int
        Number of PCs to plot (default 3).
    stats : pd.DataFrame, optional
        Output from ``pca_score_stats``. If provided, annotates
        Kruskal-Wallis p-value on each panel.
    fig : plt.Figure, optional

    Returns
    -------
    plt.Figure
    """
    n_pcs = min(n_pcs, pca_result.scores.shape[1])
    targets = sorted(set(pca_result.target_labels))

    if fig is None:
        fig, axes = plt.subplots(1, n_pcs, figsize=(4 * n_pcs, 4),
                                 sharey=False, layout='constrained')
    else:
        axes = fig.axes
    if n_pcs == 1:
        axes = [axes]

    for pc_idx, ax in enumerate(axes[:n_pcs]):
        scores_pc = pca_result.scores[:, pc_idx]
        data_per_target = []
        positions = list(range(len(targets)))
        colors = [TARGETNM_COLORS.get(t, 'gray') for t in targets]

        for i, target in enumerate(targets):
            mask = pca_result.target_labels == target
            data_per_target.append(scores_pc[mask])

        parts = ax.violinplot(data_per_target, positions=positions,
                              showmedians=True, showextrema=False)
        for i, body in enumerate(parts['bodies']):
            body.set_facecolor(colors[i])
            body.set_alpha(0.6)
        parts['cmedians'].set_color('black')
        parts['cmedians'].set_linewidth(1.5)

        pct = pca_result.explained_variance_ratio[pc_idx]
        ax.set_title(f'PC{pc_idx + 1} ({pct:.0%})')
        ax.set_xticks(positions)
        ax.set_xticklabels([t.split('-')[0] for t in targets],
                           rotation=45, ha='right', fontsize=8)
        ax.axhline(0, ls='--', color='gray', lw=0.5)
        if pc_idx == 0:
            ax.set_ylabel('PC score')

        # Annotate KW p-value
        if stats is not None:
            kw = stats[(stats['pc'] == pc_idx + 1) & stats['target_a'].isna()]
            if len(kw):
                p = kw['kruskal_p'].iloc[0]
                label = f'p={p:.3f}' if p >= 0.001 else f'p={p:.1e}'
                ax.annotate(f'KW {label}', xy=(1, 1),
                            xycoords='axes fraction', fontsize=7,
                            ha='right', va='top', color='k')

    return fig


def plot_glm_pca_summary(pca_result, recordings, n_pcs=3, stats=None,
                         comp_label='PC'):
    """Combined weights heatmap, violin scores, and subject-mean panels.

    Layout (3 rows):
        Row 0: PCA weights heatmap (spans full width).
        Row 1: Violin score distributions per target-NM (one panel per PC).
        Row 2: Subject-mean +/- 95% CI per target-NM (one panel per PC),
                with KW / post-hoc annotation.

    Parameters
    ----------
    pca_result : GLMPCAResult
    recordings : pd.DataFrame
        Must contain 'eid' and 'subject' to map recordings to subjects.
    n_pcs : int
        Number of PCs to plot.
    stats : pd.DataFrame, optional
        Output from ``pca_score_stats`` (recording-level). Used for violin
        KW annotation.

    Returns
    -------
    plt.Figure
    """
    from iblnm.analysis import pca_subject_score_stats

    n_pcs = min(n_pcs, pca_result.scores.shape[1])
    targets = sorted(set(pca_result.target_labels))

    subject_means, subject_stats = pca_subject_score_stats(
        pca_result, recordings)

    fig = plt.figure(figsize=(4 * n_pcs, 12), layout='constrained')
    gs = fig.add_gridspec(3, n_pcs, height_ratios=[1, 1.5, 1.5])

    # --- Row 0: Weights heatmap ---
    ax_hm = fig.add_subplot(gs[0, :])
    data = pca_result.components
    vmax = np.abs(data).max()
    im = ax_hm.imshow(data, aspect='auto', cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax)
    short = {
        'log_contrast': 'contrast',
        'log_contrast:side': 'contrast×side',
        'log_contrast:reward': 'contrast×reward',
        'side:reward': 'side×reward',
    }
    xlabels = [short.get(f, f) for f in pca_result.feature_names]
    ylabels = [
        f'{comp_label}{i+1} ({pca_result.explained_variance_ratio[i]:.0%})'
        for i in range(n_pcs)
    ]
    ax_hm.set_xticks(range(len(xlabels)))
    ax_hm.set_xticklabels(xlabels, rotation=45, ha='right')
    ax_hm.set_yticks(range(n_pcs))
    ax_hm.set_yticklabels(ylabels)
    for i in range(n_pcs):
        for j in range(len(xlabels)):
            val = data[i, j]
            color = 'white' if abs(val) > 0.5 * vmax else 'black'
            ax_hm.text(j, i, f'{val:.2f}', ha='center', va='center',
                       fontsize=8, color=color)
    fig.colorbar(im, ax=ax_hm, shrink=0.8, label='Loading')

    colors = [TARGETNM_COLORS.get(t, 'gray') for t in targets]

    # --- Row 1: Violin plots ---
    for pc_idx in range(n_pcs):
        ax = fig.add_subplot(gs[1, pc_idx])
        scores_pc = pca_result.scores[:, pc_idx]
        data_per_target = []
        positions = list(range(len(targets)))

        for target in targets:
            mask = pca_result.target_labels == target
            data_per_target.append(scores_pc[mask])

        parts = ax.violinplot(data_per_target, positions=positions,
                              showmedians=True, showextrema=False)
        for i, body in enumerate(parts['bodies']):
            body.set_facecolor(colors[i])
            body.set_alpha(0.6)
        parts['cmedians'].set_color('black')
        parts['cmedians'].set_linewidth(1.5)

        pct = pca_result.explained_variance_ratio[pc_idx]
        ax.set_title(f'{comp_label}{pc_idx + 1} ({pct:.0%})')
        ax.set_xticks(positions)
        ax.set_xticklabels([t.split('-')[0] for t in targets],
                           rotation=45, ha='right', fontsize=8)
        ax.axhline(0, ls='--', color='gray', lw=0.5)
        if pc_idx == 0:
            ax.set_ylabel(f'{comp_label} score')

        if stats is not None:
            kw = stats[(stats['pc'] == pc_idx + 1) & stats['target_a'].isna()]
            if len(kw):
                p = kw['kruskal_p'].iloc[0]
                h = kw['kruskal_h'].iloc[0]
                if p >= 0.05:
                    label = f'H={h:.1f}, p={p:.3f} n.s.'
                else:
                    label = f'H={h:.1f}, p={p:.1e}'
                ax.annotate(label, xy=(1, 1),
                            xycoords='axes fraction', fontsize=7,
                            ha='right', va='top', color='k')

    # --- Row 2: Subject means +/- 95% CI ---
    for pc_idx in range(n_pcs):
        ax = fig.add_subplot(gs[2, pc_idx])
        pc_num = pc_idx + 1
        sm = subject_means[subject_means['pc'] == pc_num]

        for i, target in enumerate(targets):
            sm_t = sm[sm['target_NM'] == target]
            n_subj = len(sm_t)
            # Jitter subjects around the target x-position
            jitter = np.linspace(-0.25, 0.25, n_subj) if n_subj > 1 else [0.0]
            for j, (_, row) in enumerate(sm_t.iterrows()):
                ax.errorbar(
                    i + jitter[j], row['mean'],
                    yerr=1.96 * row['sem'],
                    fmt='o', capsize=0, color=colors[i], markersize=4,
                    zorder=3,
                )

        ax.set_xticks(range(len(targets)))
        ax.set_xticklabels([t.split('-')[0] for t in targets],
                           rotation=45, ha='right', fontsize=8)
        ax.axhline(0, ls='--', color='gray', lw=0.5)
        ax.set_title(f'{comp_label}{pc_num} subject means')
        if pc_idx == 0:
            ax.set_ylabel(f'Mean {comp_label} score')

        # Annotate subject-level KW
        kw = subject_stats[
            (subject_stats['pc'] == pc_num) & subject_stats['target_a'].isna()
        ]
        if len(kw):
            p = kw['kruskal_p'].iloc[0]
            h = kw['kruskal_h'].iloc[0]
            if p >= 0.05:
                label = f'H={h:.1f}, p={p:.3f} n.s.'
            else:
                label = f'H={h:.1f}, p={p:.1e}'
            ax.annotate(label, xy=(1, 1),
                        xycoords='axes fraction', fontsize=7,
                        ha='right', va='top', color='k')

        # Draw significance brackets for pairwise comparisons
        pw = subject_stats[
            (subject_stats['pc'] == pc_num)
            & subject_stats['target_a'].notna()
            & (subject_stats['mwu_p'] < 0.05)
        ].sort_values('mwu_p')
        if len(pw):
            target_to_x = {t: i for i, t in enumerate(targets)}
            y_max = ax.get_ylim()[1]
            dy = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.06
            y_bar = y_max + dy
            for _, row in pw.iterrows():
                x0 = target_to_x[row['target_a']]
                x1 = target_to_x[row['target_b']]
                stars = _pval_to_stars(row['mwu_p'])
                # Horizontal bar with small vertical ticks
                ax.plot([x0, x1], [y_bar, y_bar], color='k', lw=0.8)
                ax.plot([x0, x0], [y_bar - dy * 0.3, y_bar], color='k', lw=0.8)
                ax.plot([x1, x1], [y_bar - dy * 0.3, y_bar], color='k', lw=0.8)
                ax.text((x0 + x1) / 2, y_bar, stars,
                        ha='center', va='bottom', fontsize=8)
                y_bar += dy * 1.5
            ax.set_ylim(ax.get_ylim()[0], y_bar + dy * 0.5)

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
    bars = ax.bar(range(len(targets)), corrs, color=colors)
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(targets, rotation=45, ha='right')
    ax.set_ylim(0, 1)
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
    delta_matrix = matrix.values - diag[:, np.newaxis]  # subtract row baseline (own data, different weights)
    im2 = ax.imshow(delta_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
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
        for i in range(len(cohorts)):
            for j in range(len(cohorts)):
                ax.text(j, i, f'{sim_matrix.values[i, j]:.2f}',
                        ha='center', va='center', fontsize=9)
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


def plot_rt_by_contrast(df, ax=None):
    """Horizontal boxplots of response time by contrast, one offset per target-NM.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``response_time``, ``contrast``, and ``target_NM`` columns.
        Contrasts should be absolute (unsigned).
    ax : plt.Axes, optional
        Axes to draw on. Created if not provided.

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
    all_vals = df.loc[df['response_time'] > 0, 'response_time'].dropna()
    if len(all_vals):
        log_min = np.floor(np.log10(all_vals.min()))
        log_max = np.ceil(np.log10(all_vals.max()))
        tick_powers = np.arange(log_min, log_max + 1)
        ax.set_xticks(tick_powers)
        ax.set_xticklabels([str(10 ** int(p)) if p >= 0 else str(round(10 ** p, 3))
                            for p in tick_powers])

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
        ax.legend(handles=handles, fontsize=7, loc='upper left')

    return ax


