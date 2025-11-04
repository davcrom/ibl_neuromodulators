import numpy as np
import pandas as pd
from datetime import datetime
import traceback
from tqdm import tqdm
from scipy import stats
from matplotlib import pyplot as plt

from one.api import ONE
from brainbox.io.one import PhotometrySessionLoader

from iblphotometry import metrics

EVENTS = ['stimOn_times', 'firstMovement_times', 'feedback_times']
TRIAL_TYPES = ['probabilityLeft', 'signed_contrast', 'feedbackType']
N_TRIALS_SESSION = 90
N_TRIALS = 5
contrast_cmap = plt.get_cmap("inferno_r", 5)
COLORS = {
    'contrast_0.0': contrast_cmap(0),
    'contrast_0.0625': contrast_cmap(1),
    'contrast_0.125': contrast_cmap(2),
    'contrast_0.25': contrast_cmap(3),
    'contrast_1.0': contrast_cmap(4),
}

def get_responses(photometry, trials, event, time_window=(-1, 2)):
    """Return peri-event aligned zdFF and time axis."""
    t = photometry.index.values
    SAMPLING_RATE = int(1 / np.mean(np.diff(t)))
    calcium = photometry.values
    t_events = trials[event].dropna().values
    t_events= t_events[
        (t_events + time_window[0] >= t.min()) & (t_events + time_window[1] < t.max())
        ]
    n_trials = len(t_events)
    samples_window = np.arange(time_window[0]*SAMPLING_RATE, time_window[1]*SAMPLING_RATE)
    psth_idx = np.tile(samples_window[:, None], (1, n_trials))
    event_idx = np.searchsorted(t, t_events)
    psth_idx += event_idx
    # ~psth_idx = psth_idx[(psth_idx >= 0) & (psth_idx < len(t))].reshape(-1, n_trials)
    responses = calcium[psth_idx]
    return responses


def get_response_tpts(photometry, time_window=(-1, 2)):
    t = photometry.index.values
    SAMPLING_RATE = int(1 / np.mean(np.diff(t)))
    samples_window = np.arange(time_window[0]*SAMPLING_RATE, time_window[1]*SAMPLING_RATE)
    return np.linspace(time_window[0], time_window[1], samples_window.shape[0])


def normalize_response(trial, bwin=(-0.1, 0), divide=True):
    i0, i1 = trial['tpts'].searchsorted(bwin)
    bval = trial['response'][i0:i1].mean()
    resp_norm = trial['response'] - bval
    if divide:
        resp_norm = resp_norm / bval
    return resp_norm


def resample_response(trial, new_tpts, fill_value=np.nan):
    """
    Resample response data to a new time base using linear interpolation.

    Parameters:
    -----------
    row : pd.Series
        Row containing 'tpts' and 'response' arrays
    new_timebase : np.ndarray
        Target time points for resampling
    fill_value : float, optional
        Value to use for points outside the original time range.
        Default is np.nan. Can also use a tuple (left_fill, right_fill)
        for different values on each side.

    Returns:
    --------
    np.ndarray
        Resampled response values at new_timebase points
    """
    return np.interp(
        new_tpts, trial['tpts'], trial['response'],
        left=fill_value, right=fill_value
        )


def plot_mean_repsonse(trials, color='black', plot_all=False, ax=None, dpi=300, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(dpi=dpi)

    if plot_all:
        for _, trial in trials.iterrows():
            ax.plot(trial['tpts'], trial['response'], color=color, alpha=0.1)

    tpts = trials['tpts'].iloc[0]
    responses = np.stack(trials['response'])
    mean = np.mean(responses, axis=0)
    sem = stats.sem(responses, axis=0)

    ax.plot(tpts, mean, color=color, **kwargs)
    ax.fill_between(tpts, mean - sem, mean + sem, alpha=0.25, color=color)

    return ax

def pval2stars(p, ns='n.s.', na='n/a'):
    if np.isnan(p):
        return na
    elif p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return ns

def cm2in(cm):
    return cm / 2.54

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
    w = cm2in(w) # convert cm to inches
    h = cm2in(h)
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

def clip_axes_to_ticks(ax=None, spines=['left', 'bottom'], ext={}):
    """
    Clip the axis lines to end at the minimum and maximum tick values.

    Parameters
    ----------
    ax : matplotlib.axes
        Axes to resize, if None the output of plt.gca() will be re-sized.

    spines : list
        Axes to keep and clip, axes not included in this list will be removed.
        Valid values include 'left', 'bottom', 'right', 'top'.

    ext : dict
        For each axis in ext.keys() ('left', 'bottom', 'right', 'top'),
        the axis line will be extended beyond the last tick by the value
        specified, e.g. {'left':[0.1, 0.2]} will results in an axis line
        that extends 0.1 units beyond the bottom tick and 0.2 unit beyond
        the top tick.
    """
    if ax is None:
        ax = plt.gca()
    spines2ax = {
        'left': ax.yaxis,
        'top': ax.xaxis,
        'right': ax.yaxis,
        'bottom': ax.xaxis
    }
    all_spines = ['left', 'bottom', 'right', 'top']
    for spine in spines:
        low = min(spines2ax[spine].get_majorticklocs())
        high = max(spines2ax[spine].get_majorticklocs())
        if spine in ext.keys():
            low += ext[spine][0]
            high += ext[spine][1]
        ax.spines[spine].set_bounds(low, high)
    for spine in [spine for spine in all_spines if spine not in spines]:
        ax.spines[spine].set_visible(False)




# Load the sessions
df_sessions = pd.read_parquet('metadata/sessions_2025-10-31-20h08.pqt')

# Restrict the dataframe based on session QC
# ~df_sessions = pd.read_parquet('metadata/sessions_2025-10-31-20h10.pqt')
# ~df_sessions = df_sessions.query('session_status == "good"')
# ~df_sessions['has_photometry'] = df_sessions['alf/photometry/photometry.signal.pqt']
# ~df_sessions = df_sessions.query('has_photometry == True')

# Restrict the dataframe to sessions we're interested in
# ~df_sessions = df_sessions.query('NM == "ACh" and session_type == "biased"')
df_sessions = df_sessions.query('session_type == "biased"')

# Load the insertions table
df_insertions = pd.read_csv('metadata/insertions_all.csv')
df_insertions['hemisphere'] = df_insertions['X-ml_um'].apply(lambda x: 'l' if x > 0 else 'r')

# Connect to the database
one = ONE()

# Loop over sessions
responses = []
exceptions_log = []
for idx, session in tqdm(df_sessions.iterrows(), total=len(df_sessions)):
    try:  # try to get data the new way
        loader = PhotometrySessionLoader(eid=session['eid'], one=one)
        loader.load_photometry()
        loader.load_trials()
    except Exception as e:
        # Capture detailed exception info
        exception_info = {
            'eid': session['eid'],
            'timestamp': datetime.now(),
            'exception_type': type(e).__name__,
            'exception_message': str(e),
            'traceback': traceback.format_exc()
        }
        exceptions_log.append(exception_info)

    # Get the insertion info for this subject
    subject = session['subject']
    insertions = df_insertions.query('subject == @subject')

    # Make sure some basic QC checks are passed
    n_trials = len(loader.trials)
    if n_trials < N_TRIALS_SESSION:
        continue
    full_photometry = pd.concat(
        [loader.photometry['GCaMP'], loader.photometry['Isosbestic']]
        ).sort_index()
    n_early = metrics.n_early_samples(full_photometry)
    n_unique = metrics.n_unique_samples(loader.photometry['GCaMP'])
    if n_early > 0 and n_unique < 500:
        continue

    # Create a signed contrast column
    loader.trials['contrastLeft'] = -1 * loader.trials['contrastLeft']
    loader.trials['signed_contrast'] = loader.trials['contrastRight'].combine_first(
        loader.trials['contrastLeft']
        )

    # Loop over target brain areas for this subject
    for target in loader.photometry['GCaMP'].columns:

        # Check we will be able to say which hemisphere the fiber is in
        if len(insertions) > 1 and len(target.split('-')) == 1:
            continue
        hemisphere = insertions['hemisphere'].iloc[0]

        # Get the time points relative to the event (same for all)
        tpts = get_response_tpts(
            loader.photometry['GCaMP'][target]
            )

        # For each event
        for event in EVENTS:

            # Loop over the different trial types
            for (p, c, fb), trials in loader.trials.groupby(TRIAL_TYPES):

                # Check there are sufficient trials
                if len(trials) < N_TRIALS:
                    continue

                # Collect the info in a dict
                resp_dict = {
                    'subject': session['subject'],
                    'eid': session['eid'],
                    'session_type': session['session_type'],
                    'NM': session['NM'],
                    'target': target.split('-')[0],
                    'hemisphere': hemisphere,
                    'p_left': p,
                    'signed_contrast': c,
                    'feedback': fb,
                    'event': event,
                    'tpts': tpts
                    }

                # Get the responses for each trisl
                resp_dict['response'] = get_responses(
                    loader.photometry['GCaMP'][target],
                    trials,
                    event
                    ).T

                # Append to list
                responses.append(resp_dict)

# Convert list to dataframe, and 'explode' such that each trial gets a row
df_responses = pd.DataFrame(responses).explode('response')

# Save the response dataframe
timestamp = datetime.now().strftime("%Y-%m-%d-%Hh%M")
df_responses.to_parquet(f'responses_{timestamp}.pqt')

# Load the dataframe (in case you already ran the loop)
# ~df_responses = pd.read_parquet('responses_2025-11-03-18h01.pqt')

# Convenience columns
df_responses = df_responses.dropna(subset='response')
df_responses['contrast'] = df_responses['signed_contrast'].apply(np.abs)
df_responses['hemisphere'] = df_responses['hemisphere'].apply(
    lambda x: 1 if x == 'r' else -1
    )

# Normalize the responses
df_responses.loc[:, 'response'] = df_responses.apply(
    normalize_response, axis='columns'
    )

# Resample the repsonses to a common time-base
new_tpts = np.linspace(-0.9, 1.9, 90)
df_responses.loc[:, 'response'] = df_responses.apply(
    lambda x: resample_response(x, new_tpts), axis='columns'
    )
df_responses.loc[:, 'tpts'] = df_responses.apply(lambda x: new_tpts, axis='columns')

# Plot responses by contrast for each target-NM
for (target, NM), df_target in df_responses.groupby(['target', 'NM']):
    for event in EVENTS:
        df_event = df_target.query('event == @event')
        fig, ax = plt.subplots()
        ax.set_title(f'{target}-{NM} - {event}')
        for contrast, trials in df_event.groupby('contrast'):
            plot_mean_repsonse(
                trials,
                ax=ax,
                color=COLORS[f'contrast_{contrast}'],
                label=f'{contrast:.4f}'
                )
        ax.legend(title='Contrast')
