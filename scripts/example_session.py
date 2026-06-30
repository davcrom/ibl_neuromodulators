"""
Example Session Viewer

Selects a high-quality VTA-DA session with lightning pose tracking and plots
a ~30s snippet showing preprocessed photometry, wheel velocity, and pose
estimates aligned to trial events.

Output:
    figures/example_session/   — SVG and PNG figures

Usage:
    python scripts/example_session.py
    python scripts/example_session.py --eid <eid>           # use a specific session
    python scripts/example_session.py --duration 90         # snippet length in seconds
"""
import argparse
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.transforms import blended_transform_factory

from iblnm.config import (
    PROJECT_ROOT, SESSIONS_FPATH, SESSIONS_H5_DIR, PERFORMANCE_FPATH,
    WHEEL_FS, POSE_FS, FIGURE_DPI, TARGETNM_COLORS, ANALYSIS_QC_BLOCKERS,
)
from iblnm.analysis import resample_pose, movement_trace
from iblnm.data import PhotometrySession, PhotometrySessionGroup, _load_pose_xcorr
from iblnm.io import _get_default_connection

plt.ion()

DEFAULT_TARGET_NM = 'VTA-DA'
DEFAULT_DURATION = 30  # seconds

GAP = 0.15            # vertical gap between unit-height normalized trace bands
TRACE_LW = 2.0        # uniform trace linewidth
EVENT_LINE_COLOR = '0.6'  # thin gray for stimulus/feedback event lines
EVENT_LW = 0.8
MARKER_GAP = 0.3      # gap between top band and the marker strip
MARKER_SIZE = 120     # scatter marker area for the event strip circles


def contrast_rank_grays(contrasts, levels):
    """Grayscale value for each contrast by its rank among the session's levels.

    Each ``|contrast|`` is ranked within ``levels`` (the sorted distinct contrast
    magnitudes) and mapped to ``1 - rank / (n_levels - 1)``, so the lowest rank
    is white (1.0) and the highest is black (0.0). Ranking — not raw value — makes
    the mapping unit-agnostic (percent vs fraction). A single level yields all
    0.0 (no divide-by-zero).

    Parameters
    ----------
    contrasts : array_like
        Per-event contrasts (sign ignored).
    levels : array_like
        Sorted distinct contrast magnitudes for the session.

    Returns
    -------
    np.ndarray
        Grayscale values in [0, 1], one per element of ``contrasts``.
    """
    levels = np.asarray(levels)
    n_levels = len(levels)
    if n_levels == 1:
        return np.zeros(len(contrasts))
    ranks = np.searchsorted(levels, np.abs(contrasts))
    return 1 - ranks / (n_levels - 1)


def feedback_colors(feedback_types):
    """Map feedback types to circle colors: green for correct (1), else red."""
    return ['green' if f == 1 else 'red' for f in feedback_types]


# =========================================================================
# Session selection
# =========================================================================

def camera_timing_ok(pose_xcorr):
    """Whether paw–wheel cross-correlation indicates correct camera frame timing.

    Each row of ``pose_xcorr['functions']`` is one session-third's
    cross-correlation curve over lags; a high peak in any third demonstrates
    correct frame timing (a quiet third self-gates to a low peak). Returns True
    when the largest per-third peak exceeds 0.5.

    Parameters
    ----------
    pose_xcorr : dict
        Loaded ``video/crosscorr`` payload; ``functions`` has shape
        (3, n_lags) and may contain NaNs for low-movement thirds.

    Returns
    -------
    bool
        True if the max-across-thirds peak cross-correlation exceeds 0.5.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # all-NaN third
        peak = np.nanmax(np.nanmax(pose_xcorr['functions'], axis=1))
    return bool(peak > 0.5)


def select_example_session(group, target_nm=DEFAULT_TARGET_NM):
    """Pick the highest-performing analysis-ready recording for a target-NM.

    Ranks target-NM recordings by fraction_correct (descending), then returns
    the first whose H5 paw–wheel cross-correlation passes the camera-timing
    gate. A present ``video/crosscorr`` group implies LightningPose was
    extracted, so pose is loadable downstream.

    Parameters
    ----------
    group : PhotometrySessionGroup
        Group already filtered to biased/ephys session type with no blocking
        QC errors.
    target_nm : str
        Target-NM cohort to select from.

    Returns
    -------
    pd.Series
        Recording row for the chosen session.
    """
    recs = group.recordings
    candidates = recs[recs['target_NM'] == target_nm].copy()
    if len(candidates) == 0:
        raise ValueError(f"No {target_nm} recordings after filtering")

    # Merge performance if available
    if PERFORMANCE_FPATH.exists():
        perf = pd.read_parquet(PERFORMANCE_FPATH, columns=['eid', 'fraction_correct'])
        candidates = candidates.merge(perf, on='eid', how='left')
        candidates = candidates.sort_values('fraction_correct', ascending=False)
    else:
        candidates = candidates.sample(frac=1, random_state=42)

    # Pick the first candidate whose camera frame timing is correct
    for _, rec in candidates.iterrows():
        h5_path = Path(SESSIONS_H5_DIR) / f"{rec['eid']}.h5"
        with h5py.File(h5_path, 'r') as f:
            pose_xcorr = _load_pose_xcorr(f['video']) if 'video' in f else None
        if pose_xcorr is not None and camera_timing_ok(pose_xcorr):
            return rec

    raise ValueError(f"No {target_nm} session passed the camera-timing gate")


# =========================================================================
# Data loading
# =========================================================================

def load_continuous_photometry(eid, brain_region, h5_dir=SESSIONS_H5_DIR):
    """Load the full preprocessed photometry signal from H5.

    Returns
    -------
    pd.Series
        Signal indexed by time in seconds.
    """
    h5_path = Path(h5_dir) / f'{eid}.h5'
    with h5py.File(h5_path, 'r') as f:
        pp_grp = f[f'photometry/{brain_region}/preprocessed']
        times = pp_grp['times'][:]
        signal = pp_grp['signal'][:].astype(np.float64)
    return pd.Series(signal, index=times, name=brain_region)


def load_continuous_wheel(one, eid):
    """Load continuous wheel velocity from ONE.

    Returns
    -------
    pd.Series
        Velocity indexed by time in seconds.
    """
    from brainbox.behavior.wheel import interpolate_position, velocity_filtered
    wheel_raw = one.load_object(eid, 'wheel', collection='alf')
    pos, times = interpolate_position(
        wheel_raw['timestamps'], wheel_raw['position'], freq=WHEEL_FS,
    )
    vel, _ = velocity_filtered(pos, fs=WHEEL_FS)
    return pd.Series(vel.astype(np.float32), index=times.astype(np.float32), name='wheel_velocity')


def load_trial_events(eid, h5_dir=SESSIONS_H5_DIR):
    """Load trial event times from H5.

    Returns
    -------
    pd.DataFrame
        Trials table with event time columns.
    """
    h5_path = Path(h5_dir) / f'{eid}.h5'
    with h5py.File(h5_path, 'r') as f:
        data = {}
        for col in f['trials']:
            vals = f[f'trials/{col}'][:]
            if vals.dtype.kind == 'S':
                vals = vals.astype(str)
            data[col] = vals
    return pd.DataFrame(data)


# =========================================================================
# Snippet selection
# =========================================================================

def find_snippet_window(trials, duration=DEFAULT_DURATION, min_trials=8):
    """Find a window with many completed trials (no misses).

    Scans the session in 10s steps and picks the window with the most
    valid trials (choice != 0, feedback present).

    Parameters
    ----------
    trials : pd.DataFrame
    duration : float
        Window length in seconds.
    min_trials : int
        Minimum trials required.

    Returns
    -------
    tuple of float
        (t_start, t_end)
    """
    stim_times = trials['stimOn_times'].dropna().values
    fb_times = trials['feedback_times'].dropna().values
    if len(stim_times) < min_trials:
        raise ValueError("Too few trials to find a good snippet")

    t_min = stim_times.min()
    t_max = stim_times.max()
    step = 10.0

    best_start = t_min
    best_count = 0

    t = t_min
    while t + duration <= t_max + 30:
        mask = (stim_times >= t) & (stim_times <= t + duration)
        n = mask.sum()
        if n > best_count:
            # Check that trials in this window have feedback
            trial_idx = np.where(mask)[0]
            has_fb = sum(
                1 for idx in trial_idx
                if idx < len(fb_times) and not np.isnan(fb_times[idx])
            )
            if has_fb >= min_trials:
                best_count = has_fb
                best_start = t
        t += step

    if best_count < min_trials:
        # Fall back: just use first window with enough stim events
        best_start = stim_times[0]

    return best_start, best_start + duration


# =========================================================================
# Plotting
# =========================================================================

def _normalize_window(values, t, t_start, t_end):
    """Slice a trace to a window and normalize it to [0, 1].

    Parameters
    ----------
    values : 1D array
        Trace samples.
    t : 1D array
        Times (seconds) matching ``values``.
    t_start, t_end : float
        Window bounds (seconds, inclusive).

    Returns
    -------
    1D array
        In-window samples scaled by ``(x - nanmin) / (nanmax - nanmin)``. A
        constant window (``nanmax == nanmin``) returns all-zeros — no
        divide-by-zero. NaNs are preserved as gaps.
    """
    t = np.asarray(t)
    window = np.asarray(values)[(t >= t_start) & (t <= t_end)].astype(float)
    lo, hi = np.nanmin(window), np.nanmax(window)
    if hi == lo:
        return np.zeros_like(window)
    return (window - lo) / (hi - lo)


def build_traces(photometry, wheel, pose_df, pose_times, target_nm):
    """Assemble the six ordered figure traces (top → bottom).

    Parameters
    ----------
    photometry : pd.Series
        Preprocessed photometry signal, time-indexed (seconds).
    wheel : pd.Series
        Wheel velocity, time-indexed (seconds).
    pose_df : pd.DataFrame
        Resampled LightningPose columns (``{part}_x/_y/_likelihood``) on the
        uniform ``pose_times`` grid.
    pose_times : np.ndarray
        Uniform pose time axis (seconds), shared by traces 3–6.
    target_nm : str
        Target-NM cohort; selects the photometry trace color.

    Returns
    -------
    list of dict
        Six entries, each ``{'times', 'values', 'color', 'label'}``: photometry,
        wheel, left-paw speed, right-paw speed, nose speed, tongue likelihood.
        Photometry carries the target-NM color; the five movement traces take
        distinct ``Set1`` colors, skipping ``Set1``'s red (index 0) so they stay
        distinguishable from the red VTA-DA photometry trace.
    """
    movement_colors = plt.cm.Set2.colors[0:5]
    return [
        {'times': photometry.index.values, 'values': photometry.values,
         'color': TARGETNM_COLORS[target_nm], 'label': 'Photometry'},
        {'times': wheel.index.values, 'values': wheel.values,
         'color': movement_colors[0], 'label': 'Wheel'},
        {'times': pose_times,
         'values': movement_trace(pose_df, ['paw_l'], 'speed'),
         'color': movement_colors[1], 'label': 'Left paw'},
        {'times': pose_times,
         'values': movement_trace(pose_df, ['paw_r'], 'speed'),
         'color': movement_colors[2], 'label': 'Right paw'},
        {'times': pose_times,
         'values': movement_trace(pose_df, ['nose_tip'], 'speed'),
         'color': movement_colors[3], 'label': 'Nose'},
        {'times': pose_times,
         'values': movement_trace(pose_df, ['tongue_end_l', 'tongue_end_r'],
                                  'max_likelihood'),
         'color': movement_colors[4], 'label': 'Tongue'},
    ]


def plot_example_session(traces, trials, t_start, t_end):
    """Render the floating six-trace example-session figure.

    Each trace is sliced to ``[t_start, t_end]``, normalized to its in-window
    ``[0, 1]`` range, and stacked top → bottom at non-overlapping unit-height
    offsets, each tagged at the left edge with its label in the trace color.
    Vertical gray lines mark each in-window stimulus onset and
    feedback. A marker strip on a single shared ``y`` just above the top band
    shows, at each event's x-position, a stimulus circle whose grayscale encodes
    the trial's contrast rank (lowest = white, highest = black; black edge) and a
    feedback circle (green = correct, red = incorrect). The single Axes is
    frameless: no spines, ticks, labels, or legend.

    Parameters
    ----------
    traces : list of dict
        Ordered ``{'times', 'values', 'color'}`` entries from ``build_traces``.
    trials : pd.DataFrame
        Trial table with ``stimOn_times``, ``feedback_times`` (seconds),
        ``contrast`` (absolute), and ``feedbackType`` (1 = correct, −1 = wrong).
    t_start, t_end : float
        Snippet window bounds (seconds).

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    label_trans = blended_transform_factory(ax.transAxes, ax.transData)
    n = len(traces)
    step = 1 + GAP
    for k, trace in enumerate(traces):
        t = np.asarray(trace['times'])
        mask = (t >= t_start) & (t <= t_end)
        y = _normalize_window(trace['values'], t, t_start, t_end)
        offset = (n - 1 - k) * step
        ax.plot(t[mask], y + offset, color=trace['color'], linewidth=TRACE_LW)
        ax.text(-0.01, offset + 0.5, trace['label'], transform=label_trans,
                ha='right', va='center', color=trace['color'])

    event_times = np.concatenate([
        trials['stimOn_times'].values, trials['feedback_times'].values])
    event_times = event_times[(event_times >= t_start) & (event_times <= t_end)]
    for t in event_times:
        ax.axvline(t, color=EVENT_LINE_COLOR, linewidth=EVENT_LW, zorder=0)

    # Marker strip on a single shared y just above the top (photometry) band.
    marker_y = (n - 1) * step + 1 + MARKER_GAP

    stim = trials['stimOn_times'].values
    stim_mask = (stim >= t_start) & (stim <= t_end)
    levels = np.unique(np.abs(trials['contrast'].values))
    grays = contrast_rank_grays(trials['contrast'].values[stim_mask], levels)
    ax.scatter(stim[stim_mask], np.full(stim_mask.sum(), marker_y),
               s=MARKER_SIZE, facecolors=np.repeat(grays[:, None], 3, axis=1),
               edgecolors='black', zorder=3)

    fb = trials['feedback_times'].values
    fb_mask = (fb >= t_start) & (fb <= t_end)
    ax.scatter(fb[fb_mask], np.full(fb_mask.sum(), marker_y), s=MARKER_SIZE,
               c=feedback_colors(trials['feedbackType'].values[fb_mask]),
               zorder=3)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig


# =========================================================================
# Main
# =========================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--eid', type=str, default=None,
                        help='use a specific session eid')
    parser.add_argument('--target-nm', type=str, default=DEFAULT_TARGET_NM,
                        help=f'target-NM cohort (default: {DEFAULT_TARGET_NM})')
    parser.add_argument('--duration', type=float, default=DEFAULT_DURATION,
                        help='snippet duration in seconds (default: 60)')
    parser.add_argument('--camera', type=str, default='left',
                        help='camera to use for pose (default: left)')
    args = parser.parse_args()

    fig_dir = PROJECT_ROOT / 'figures/example_session'
    fig_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Load sessions and filter
    # -----------------------------------------------------------------
    print(f"Loading sessions from {SESSIONS_FPATH}")
    one = _get_default_connection()
    group = PhotometrySessionGroup.from_catalog(
        pd.read_parquet(SESSIONS_FPATH), one=one, h5_dir=SESSIONS_H5_DIR)
    group.filter_sessions(session_types=('biased', 'ephys'),
                          qc_blockers=ANALYSIS_QC_BLOCKERS)
    print(f"  {len(group)} recordings after filtering")

    # -----------------------------------------------------------------
    # Select session
    # -----------------------------------------------------------------
    if args.eid:
        rec = group.recordings[group.recordings['eid'] == args.eid].iloc[0]
        print(f"Using specified session: {rec['eid']} ({rec['subject']})")
    else:
        print(f"Selecting best analysis-ready {args.target_nm} session...")
        rec = select_example_session(group, target_nm=args.target_nm)
        print(f"Selected: {rec['eid']} ({rec['subject']}, "
              f"{rec['brain_region']}, {rec['session_type']})")

    eid = rec['eid']
    brain_region = rec['brain_region']
    target_nm = rec['target_NM']

    # -----------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------
    print("Loading photometry...")
    photometry = load_continuous_photometry(eid, brain_region)
    print(f"  {len(photometry)} samples, {photometry.index[-1] - photometry.index[0]:.0f}s")

    print("Loading wheel...")
    wheel = load_continuous_wheel(one, eid)
    print(f"  {len(wheel)} samples")

    print(f"Loading {args.camera} camera pose...")
    ps = PhotometrySession(rec, one=one)
    ps.load_camera_times()
    ps.load_pose()
    pose_df, pose_times = resample_pose(ps.pose, ps.pose_times, POSE_FS)
    print(f"  {len(pose_times)} frames after resampling to {POSE_FS} Hz")

    print("Loading trials...")
    trials = load_trial_events(eid)
    print(f"  {len(trials)} trials")

    # -----------------------------------------------------------------
    # Find snippet window
    # -----------------------------------------------------------------
    t_start, t_end = find_snippet_window(trials, duration=args.duration)
    n_trials_in = ((trials['stimOn_times'] >= t_start)
                   & (trials['stimOn_times'] <= t_end)).sum()
    print(f"Snippet: {t_start:.1f}s – {t_end:.1f}s ({n_trials_in} trials)")

    # -----------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------
    print("Plotting...")
    traces = build_traces(photometry, wheel, pose_df, pose_times, target_nm)
    fig = plot_example_session(traces, trials, t_start, t_end)

    for ext in ('svg', 'png'):
        fpath = fig_dir / f'example_{target_nm.replace("-", "_")}.{ext}'
        fig.savefig(fpath, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"Saved to {fig_dir}")
