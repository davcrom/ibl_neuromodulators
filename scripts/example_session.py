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
    python scripts/example_session.py --body-parts nose_tip paw_l paw_r tongue_end_l tongue_end_r
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from iblnm.config import (
    PROJECT_ROOT, SESSIONS_FPATH, SESSIONS_H5_DIR, PERFORMANCE_FPATH,
    TARGET_FS, WHEEL_FS, FIGURE_DPI, TARGETNM_COLORS,
)
from iblnm.data import PhotometrySessionGroup
from iblnm.io import _get_default_connection
from iblnm.util import derive_target_nm

plt.ion()

DEFAULT_TARGET_NM = 'VTA-DA'
DEFAULT_DURATION = 30  # seconds
DEFAULT_BODY_PARTS = ['nose_tip', 'paw_r', 'tongue_end_l']


# =========================================================================
# Session selection
# =========================================================================

def has_lightning_pose(one, eid):
    """Check whether a session has lightning pose data on ONE."""
    dsets = one.list_datasets(eid)
    return any('lightningPose' in str(d) for d in dsets)


def select_example_session(group, one, target_nm=DEFAULT_TARGET_NM):
    """Pick the best session: highest fraction_correct among those with LP data.

    Parameters
    ----------
    group : PhotometrySessionGroup
        Filtered recordings.
    one : one.api.One
        ONE connection.
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

    # Pick the first session that has lightning pose
    for _, rec in candidates.iterrows():
        if has_lightning_pose(one, rec['eid']):
            return rec

    raise ValueError(f"No {target_nm} session found with lightning pose data")


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
        times = f['times'][:]
        signal = f[f'preprocessed/{brain_region}'][:].astype(np.float64)
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


def load_pose_data(one, eid, body_parts=None, camera='left'):
    """Load lightning pose tracking data from ONE.

    Parameters
    ----------
    one : one.api.One
    eid : str
    body_parts : list of str, optional
        Body part names to extract. If None, returns all.
    camera : str
        Camera name ('left', 'right', 'body').

    Returns
    -------
    times : np.ndarray
        Camera timestamps.
    pose : dict[str, pd.DataFrame]
        Body part name → DataFrame with columns (x, y, likelihood).
    """
    cam_obj = f'{camera}Camera'
    cam_data = one.load_object(eid, cam_obj, collection='alf')

    times = cam_data['times']

    # Lightning pose data is stored under the 'lightningPose' key
    lp = cam_data['lightningPose']

    # lp is a numpy array: (n_frames, n_bodyparts, 3) where 3 = (x, y, likelihood)
    # or a DataFrame. Handle both.
    if isinstance(lp, pd.DataFrame):
        # Columns are multi-level: bodypart_x, bodypart_y, bodypart_likelihood
        all_parts = sorted(set(
            col.rsplit('_', 1)[0] for col in lp.columns
            if col.endswith(('_x', '_y', '_likelihood'))
        ))
        pose = {}
        for part in all_parts:
            if body_parts is not None and part not in body_parts:
                continue
            cols = {
                'x': f'{part}_x',
                'y': f'{part}_y',
                'likelihood': f'{part}_likelihood',
            }
            if all(c in lp.columns for c in cols.values()):
                pose[part] = lp[list(cols.values())].copy()
                pose[part].columns = ['x', 'y', 'likelihood']
    elif isinstance(lp, np.ndarray):
        # Try to get body part names from attributes or column names
        # Typically (n_frames, n_bodyparts, 3)
        n_parts = lp.shape[1] if lp.ndim == 3 else 1
        part_names = [f'part_{i}' for i in range(n_parts)]
        pose = {}
        for i, part in enumerate(part_names):
            if body_parts is not None and part not in body_parts:
                continue
            pose[part] = pd.DataFrame({
                'x': lp[:, i, 0],
                'y': lp[:, i, 1],
                'likelihood': lp[:, i, 2],
            })
    else:
        raise TypeError(f"Unexpected lightningPose type: {type(lp)}")

    return times, pose


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

def _pretty_part_name(part):
    """Convert body part key to readable label."""
    return part.replace('_', ' ').replace(' l', ' (L)').replace(' r', ' (R)')


def plot_example_snippet(photometry, wheel, pose_times, pose, trials,
                         t_start, t_end, brain_region, target_nm,
                         body_parts=None):
    """Plot aligned photometry, wheel, and pose snippet.

    Pose traces are z-scored and stacked with slight vertical offsets in a
    single panel.

    Parameters
    ----------
    photometry : pd.Series
        Preprocessed photometry signal (time-indexed).
    wheel : pd.Series
        Wheel velocity (time-indexed).
    pose_times : np.ndarray
        Camera frame timestamps.
    pose : dict
        Body part name → DataFrame(x, y, likelihood).
    trials : pd.DataFrame
        Trial events table.
    t_start, t_end : float
        Snippet window in seconds.
    brain_region : str
    target_nm : str
    body_parts : list of str, optional
        Which pose body parts to plot. If None, plots all available.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from matplotlib.lines import Line2D

    if body_parts is None:
        body_parts = list(pose.keys())
    n_pose = len(body_parts)
    n_rows = 3  # photometry + wheel + pose (single panel)

    fig, axes = plt.subplots(
        n_rows, 1, figsize=(14, 7), sharex=True,
        gridspec_kw={'hspace': 0.12, 'height_ratios': [1, 0.7, 1.2]},
    )

    color = TARGETNM_COLORS.get(target_nm, 'tab:green')
    lw_signal = 1.3
    lw_event = 1.1

    # --- Trial events in the window ---
    stim_mask = (
        trials['stimOn_times'].between(t_start, t_end)
        & trials['stimOn_times'].notna()
    )
    fb_mask = (
        trials['feedback_times'].between(t_start, t_end)
        & trials['feedback_times'].notna()
    )
    stim_in = trials.loc[stim_mask, 'stimOn_times'].values
    fb_in = trials.loc[fb_mask]

    reward_times = fb_in.loc[fb_in['feedbackType'] == 1, 'feedback_times'].values
    error_times = fb_in.loc[fb_in['feedbackType'] == -1, 'feedback_times'].values

    # --- Panel 0: Photometry ---
    ax = axes[0]
    t_mask = (photometry.index >= t_start) & (photometry.index <= t_end)
    ax.plot(photometry.index[t_mask], photometry.values[t_mask],
            color=color, linewidth=lw_signal)
    ax.set_ylabel('Photometry (z)')
    ax.set_title(f'{target_nm} example session', fontsize=11, loc='left')

    # --- Panel 1: Wheel velocity ---
    ax = axes[1]
    w_mask = (wheel.index >= t_start) & (wheel.index <= t_end)
    ax.plot(wheel.index[w_mask], wheel.values[w_mask],
            color='0.3', linewidth=lw_signal)
    ax.set_ylabel('Wheel velocity (rad/s)')

    # --- Panel 2: Pose (all body parts, z-scored + offset) ---
    ax = axes[2]
    _pose_palette = {
        'nose_tip': '#636EFA',
        'paw_l': '#EF553B',
        'paw_r': '#00CC96',
        'tongue_end_l': '#B6E880',
        'tongue_end_r': '#FF97FF',
    }
    _fallback_colors = ['#4C72B0', '#55A868', '#8172B3', '#C44E52', '#DD8452']
    pose_colors = [
        _pose_palette.get(p, _fallback_colors[i % len(_fallback_colors)])
        for i, p in enumerate(body_parts)
    ]
    p_mask = (pose_times >= t_start) & (pose_times <= t_end)
    t_pose = pose_times[p_mask]
    offset_step = 3.0  # z-score units between traces

    pose_handles = []
    for i, part in enumerate(body_parts):
        if part not in pose:
            continue
        df_part = pose[part]
        y_vals = df_part['y'].values[p_mask].astype(float)
        likelihood = df_part['likelihood'].values[p_mask]

        # Z-score within window
        mu, sigma = np.nanmean(y_vals), np.nanstd(y_vals)
        if sigma > 0:
            y_z = (y_vals - mu) / sigma
        else:
            y_z = y_vals - mu
        y_z += i * offset_step

        # Mask low-confidence to NaN so gaps appear
        y_plot = np.where(likelihood > 0.9, y_z, np.nan)
        line, = ax.plot(t_pose, y_plot, color=pose_colors[i],
                        linewidth=lw_signal)
        pose_handles.append(Line2D(
            [0], [0], color=pose_colors[i], linewidth=lw_signal,
            label=_pretty_part_name(part),
        ))

    ax.set_ylabel('Pose (y, z-scored)')
    ax.set_yticks([])
    if pose_handles:
        ax.legend(handles=pose_handles, loc='upper right', fontsize=8,
                  framealpha=0.7, ncol=min(n_pose, 4))

    # --- Overlay event lines on all panels ---
    for ax in axes:
        for t in stim_in:
            ax.axvline(t, color='steelblue', linewidth=lw_event, alpha=0.6,
                       linestyle='--')
        for t in reward_times:
            ax.axvline(t, color='seagreen', linewidth=lw_event, alpha=0.6)
        for t in error_times:
            ax.axvline(t, color='firebrick', linewidth=lw_event, alpha=0.6)

    # --- Clean up axes ---
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(bottom=False, labelbottom=False)
    # Bottom axis keeps its x-axis
    axes[-1].spines['bottom'].set_visible(True)
    axes[-1].tick_params(bottom=True, labelbottom=True)
    axes[-1].set_xlabel('Time (s)')

    # Event legend on top panel
    event_handles = [
        Line2D([0], [0], color='steelblue', linestyle='--',
               linewidth=lw_event, label='Stimulus'),
        Line2D([0], [0], color='seagreen',
               linewidth=lw_event, label='Reward'),
        Line2D([0], [0], color='firebrick',
               linewidth=lw_event, label='Error'),
    ]
    axes[0].legend(handles=event_handles, loc='upper right', fontsize=8,
                   framealpha=0.7, ncol=3)

    fig.align_ylabels(axes)
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
    parser.add_argument('--body-parts', nargs='+', default=None,
                        help=f'pose body parts to plot (default: {DEFAULT_BODY_PARTS})')
    parser.add_argument('--camera', type=str, default='left',
                        help='camera to use for pose (default: left)')
    args = parser.parse_args()

    body_parts = args.body_parts or DEFAULT_BODY_PARTS

    fig_dir = PROJECT_ROOT / 'figures/example_session'
    fig_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Load sessions and filter
    # -----------------------------------------------------------------
    print(f"Loading sessions from {SESSIONS_FPATH}")
    df_sessions = pd.read_parquet(SESSIONS_FPATH)
    df_sessions = derive_target_nm(df_sessions)

    _parallel_cols = ['target_NM', 'brain_region', 'hemisphere']
    _lengths_match = df_sessions[_parallel_cols].apply(
        lambda row: len(set(
            len(v) if isinstance(v, (list, np.ndarray)) else 1
            for v in row
        )) == 1,
        axis=1,
    )
    df_sessions = df_sessions[_lengths_match].copy()
    df_recordings = df_sessions.explode(_parallel_cols).copy()
    df_recordings['fiber_idx'] = df_recordings.groupby('eid').cumcount()

    one = _get_default_connection()
    group = PhotometrySessionGroup(df_recordings, one=one)
    group.filter_recordings(session_types=('biased', 'ephys'))
    print(f"  {len(group)} recordings after filtering")

    # -----------------------------------------------------------------
    # Select session
    # -----------------------------------------------------------------
    if args.eid:
        rec = group.recordings[group.recordings['eid'] == args.eid].iloc[0]
        print(f"Using specified session: {rec['eid']} ({rec['subject']})")
    else:
        print(f"Selecting best {args.target_nm} session with lightning pose...")
        rec = select_example_session(group, one, target_nm=args.target_nm)
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
    pose_times, pose = load_pose_data(one, eid, body_parts=body_parts,
                                       camera=args.camera)
    print(f"  {len(pose_times)} frames, {len(pose)} body parts: {list(pose.keys())}")

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
    fig = plot_example_snippet(
        photometry, wheel, pose_times, pose, trials,
        t_start, t_end, brain_region, target_nm,
        body_parts=list(pose.keys()),
    )

    for ext in ('svg', 'png'):
        fpath = fig_dir / f'example_{target_nm.replace("-", "_")}.{ext}'
        fig.savefig(fpath, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"Saved to {fig_dir}")
