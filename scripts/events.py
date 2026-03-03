"""
Peri-Event Response Magnitudes

Loads pre-computed H5 session data, extracts trial-level response magnitudes,
and produces a flat DataFrame suitable for linear mixed model analysis in R.

For each recording (session × brain region) and each event (stimOn, firstMovement,
feedback), computes:
  - response_early: mean z-score in the early window (0.1–0.35 s)
  - response_late:  mean z-score in the late window (0.35–0.6 s)
                    (feedback_times only; NaN for stimOn and firstMovement)

Responses are baseline-subtracted and masked at subsequent events before
magnitude computation (e.g. stimOn responses are cut off at firstMovement).

Output: data/events.pqt, figures/events/*.svg
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from iblnm.config import (
    PROJECT_ROOT, SESSIONS_FPATH, SESSIONS_H5_DIR, EVENTS_FPATH,
    QUERY_DATABASE_LOG_FPATH, PHOTOMETRY_LOG_FPATH, TASK_LOG_FPATH,
    SUBJECTS_TO_EXCLUDE, TARGET2NM, TARGETNMS_TO_ANALYZE,
    RESPONSE_EVENTS, RESPONSE_WINDOWS, FIGURE_DPI,
)
from iblnm.vis import plot_relative_contrast
from iblnm.data import PhotometrySession
from iblnm.io import _get_default_connection
from iblnm.util import collect_session_errors, fill_empty_lists_from_group
from iblnm.analysis import compute_response_magnitude

plt.ion()

# Events for which the late window is meaningful (not masked by a subsequent event)
_LATE_WINDOW_EVENTS = {'feedback_times'}


def run_events_pipeline(df_recordings, one=None, h5_dir=None):
    """Extract trial-level response magnitudes from pre-computed H5 files.

    Parameters
    ----------
    df_recordings : pd.DataFrame
        One row per recording (session × brain region), with columns:
        eid, subject, session_type, NM, target_NM, brain_region, hemisphere.
    one : one.api.One
        ONE instance required by PhotometrySession constructor.
    h5_dir : Path, optional
        Directory containing {eid}.h5 files. Defaults to SESSIONS_H5_DIR.

    Returns
    -------
    pd.DataFrame
        One row per (recording × event × trial) with scalar response magnitudes.
    """
    if one is None:
        one = _get_default_connection()
    if h5_dir is None:
        h5_dir = SESSIONS_H5_DIR

    rows = []
    for _, rec in tqdm(df_recordings.iterrows(), total=len(df_recordings),
                       desc="Extracting response magnitudes"):
        eid = rec['eid']
        brain_region = rec['brain_region']
        hemisphere = rec['hemisphere']
        h5_path = h5_dir / f'{eid}.h5'

        if not h5_path.exists():
            continue

        # Load session data from H5
        ps = PhotometrySession(rec, one=one)
        # FIXME: update load_h5 to throw error if requested groups are missing
        ps.load_h5(h5_path, groups=['trials', 'responses'])

        # Check responses and trial data
        if getattr(ps, 'responses', None) is None or getattr(ps, 'trials', None) is None:
            continue

        # Check region exists in H5
        available_regions = list(ps.responses.coords['region'].values)
        if brain_region not in available_regions:
            continue

        # Derive hemisphere from region name
        # ~hemisphere = brain_region[-1] if brain_region.endswith(('-l', '-r')) else None

        # Mask responses at subsequent events, then subtract baseline
        responses = ps.mask_subsequent_events(ps.responses)
        responses = ps.subtract_baseline(responses)

        tpts = responses.coords['time'].values
        n_trials = len(ps.trials)

        # Compute reaction time
        if 'firstMovement_times' in ps.trials.columns and 'stimOn_times' in ps.trials.columns:
            reaction_time = (
                ps.trials['firstMovement_times'].values
                - ps.trials['stimOn_times'].values
            )
        else:
            reaction_time = np.full(n_trials, np.nan)

        # Session-level metadata (constant across trials)
        meta = {
            'eid': eid,
            'subject': rec['subject'],
            'session_type': rec['session_type'],
            'NM': rec['NM'],
            'target_NM': rec['target_NM'],
            'brain_region': brain_region,
            'hemisphere': hemisphere,
        }

        for event in RESPONSE_EVENTS:
            if event not in responses.coords['event'].values:
                continue

            resp = responses.sel(region=brain_region, event=event).values  # (n_trials, n_time)

            early = compute_response_magnitude(resp, tpts, RESPONSE_WINDOWS['early'])

            if event in _LATE_WINDOW_EVENTS:
                late = compute_response_magnitude(resp, tpts, RESPONSE_WINDOWS['late'])
            else:
                late = np.full(n_trials, np.nan)

            for t in range(n_trials):
                rows.append({
                    **meta,
                    'event': event,
                    'trial': t,
                    'signed_contrast': ps.trials['signed_contrast'].iloc[t],
                    'contrast': ps.trials['contrast'].iloc[t],
                    'choice': ps.trials['choice'].iloc[t],
                    'feedbackType': ps.trials['feedbackType'].iloc[t],
                    'probabilityLeft': ps.trials['probabilityLeft'].iloc[t],
                    'reaction_time': reaction_time[t],
                    'response_early': early[t],
                    'response_late': late[t],
                })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def add_relative_contrast(df):
    """Add hemisphere-relative contrast and side columns.

    Signed contrast is right-positive by convention. For right-hemisphere
    recordings, contralateral stimuli are on the left (negative signed_contrast),
    so we flip the sign. For left-hemisphere recordings, contralateral stimuli
    are on the right (positive signed_contrast), so no flip.

    Columns added:
      hemi_sign         : +1 for left hemisphere, -1 for right
      relative_contrast : signed_contrast * hemi_sign (positive = contra)
      side              : 'contra' or 'ipsi'
    """
    df = df.copy()
    # Midline structures (e.g. DR) have hemisphere=None; default hemi_sign=1
    # so right-stimulus is contra, matching the right-hemisphere convention.
    df['hemi_sign'] = df['hemisphere'].map({'l': 1, 'r': -1}).fillna(1)
    df['relative_contrast'] = df['signed_contrast'] * df['hemi_sign']
    # Use np.signbit to correctly assign -0.0 (right hemisphere, zero contrast) to ipsi.
    # np.signbit(-0.0) = True → ~True = False → 'ipsi'; np.signbit(0.0) = False → 'contra'.
    df['side'] = np.where(~np.signbit(df['relative_contrast'].values), 'contra', 'ipsi')
    return df


def plot_response_magnitude_figures(df_events, response_col, figures_dir):
    """Orchestrate response magnitude plots for all target-NM × event groups.

    Filters data, computes subject-demeaned responses, calls
    ``vis.plot_relative_contrast`` for each group, and saves SVG files.

    Parameters
    ----------
    df_events : pd.DataFrame
        Events table with laterality columns (side, contrast, relative_contrast).
    response_col : str
        Column name for the response magnitude ('response_early' or 'response_late').
    figures_dir : Path
        Output directory for SVG files.
    """
    window_label = response_col.replace('response_', '')
    df = df_events.dropna(subset=[response_col]).copy()

    # Drop no-go trials and implausible reaction times
    df = df.query('choice != 0 and reaction_time > 0.05')

    for (target_nm, event), df_group in df.groupby(['target_NM', 'event']):
        n_subjects = df_group['subject'].nunique()
        if n_subjects < 2:
            continue

        df_group = df_group.copy()

        # Subject-demean: subtract each subject's mean, add back the grand mean
        grand_mean = df_group[response_col].mean()
        df_group['centered_mean'] = (
            df_group.groupby('subject')[response_col]
            .transform(lambda x: x - x.mean())
        ) + grand_mean

        event_label = event.replace('_times', '')
        fig = plot_relative_contrast(df_group, 'centered_mean', target_nm, event,
                                      window_label=window_label)
        fname = f'{target_nm}_{event_label}_{window_label}.svg'
        fig.savefig(figures_dir / fname, dpi=FIGURE_DPI, bbox_inches='tight')


if __name__ == '__main__':
    # =========================================================================
    # Load and filter sessions
    # =========================================================================
    print(f"Loading sessions from {SESSIONS_FPATH}")
    df_sessions = pd.read_parquet(SESSIONS_FPATH)
    print(f"  Total sessions: {len(df_sessions)}")

    # Attach upstream error logs for QC-based filtering
    df_sessions = collect_session_errors(
        df_sessions,
        [QUERY_DATABASE_LOG_FPATH, PHOTOMETRY_LOG_FPATH, TASK_LOG_FPATH],
    )

    # Filter to analyzable sessions
    df_sessions = df_sessions[
        df_sessions['session_type'].isin(['biased', 'ephys'])
        & ~df_sessions['subject'].isin(SUBJECTS_TO_EXCLUDE)
    ]
    print(f"  After filtering: {len(df_sessions)}")

    # Derive QC flag
    _qc_blockers = {
        'MissingExtractedData', 'InsufficientTrials', 'TrialsNotInPhotometryTime',
        'QCValidationError', 'FewUniqueSamples', 'AmbiguousRegionMapping'
    }
    df_sessions = df_sessions[
        df_sessions['logged_errors'].apply(
            lambda e: not any(err in _qc_blockers for err in e)
        )
    ].copy()
    print(f"  After QC: {len(df_sessions)}")

    # =========================================================================
    # Explode to one row per recording
    # =========================================================================
    # TEMPFIX: normalize brain_region naming errors from Alyx metadata
    # Fill empty brain_region/hemisphere from other sessions of the same subject
    df_sessions = fill_empty_lists_from_group(df_sessions, 'brain_region')
    df_sessions = fill_empty_lists_from_group(df_sessions, 'hemisphere')
    n_filled = df_sessions['brain_region'].apply(lambda x: len(x) > 0 if isinstance(x, (list, np.ndarray)) else False).sum()
    print(f"  After filling from subject group: {n_filled} sessions with brain_region")

    _REGION_FIXES = {'DRN': 'DR', 'SNC': 'SNc'}

    def _fix_regions(regions):
        if not isinstance(regions, (list, np.ndarray)):
            return regions
        fixed = []
        for r in regions:
            bare = r.rsplit('-', 1)[0] if r.endswith(('-l', '-r')) else r
            suffix = r[len(bare):]
            fixed.append(_REGION_FIXES.get(bare, bare) + suffix)
        return fixed

    df_sessions['brain_region'] = df_sessions['brain_region'].apply(_fix_regions)

    # Rebuild NM and target_NM from corrected brain_region
    def _target_nm_from_region(region):
        bare = region.rsplit('-', 1)[0] if region.endswith(('-l', '-r')) else region
        nm = TARGET2NM.get(bare)
        return f'{bare}-{nm}' if nm else None

    df_sessions['target_NM'] = df_sessions['brain_region'].apply(
        lambda rs: [_target_nm_from_region(r) for r in rs]
        if isinstance(rs, (list, np.ndarray)) else rs
    )
    df_sessions['NM'] = df_sessions['target_NM'].apply(
        lambda ts: ts[0].split('-')[-1]
        if isinstance(ts, (list, np.ndarray)) and len(ts) > 0 and ts[0] else None
    )

    df_recordings = (
        df_sessions
        .explode(['target_NM', 'brain_region', 'hemisphere'])
        .loc[lambda df: df['target_NM'].isin(TARGETNMS_TO_ANALYZE)]
        .copy()
    )
    print(f"  Recordings (session × region): {len(df_recordings)}")

    # =========================================================================
    # Run pipeline
    # =========================================================================
    one = _get_default_connection()
    df_events = run_events_pipeline(df_recordings, one=one)

    if len(df_events) == 0:
        print("No events extracted. Check H5 files exist.")
        raise SystemExit(1)

    # =========================================================================
    # Save
    # =========================================================================
    EVENTS_FPATH.parent.mkdir(parents=True, exist_ok=True)
    df_events.to_parquet(EVENTS_FPATH, index=False)

    # =========================================================================
    # Summary
    # =========================================================================
    n_sessions = df_events['eid'].nunique()
    n_subjects = df_events['subject'].nunique()
    n_trials = len(df_events[df_events['event'] == RESPONSE_EVENTS[0]])

    print(f"\nSaved {len(df_events)} rows to {EVENTS_FPATH}")
    print(f"  Sessions: {n_sessions}, Subjects: {n_subjects}")
    print("\nTrials per target-NM:")
    summary = (
        df_events[df_events['event'] == RESPONSE_EVENTS[0]]
        .groupby('target_NM')
        .agg(
            n_subjects=('subject', 'nunique'),
            n_sessions=('eid', 'nunique'),
            n_trials=('trial', 'count'),
        )
    )
    print(summary.to_string())

    # =========================================================================
    # Plots: response magnitude by contrast × feedback × hemisphere
    # =========================================================================
    figures_dir = PROJECT_ROOT / 'figures/events'
    figures_dir.mkdir(parents=True, exist_ok=True)

    df_events = add_relative_contrast(df_events)

    # Restrict to unbiased block for clean contrast effects
    df_unbiased = df_events.query('probabilityLeft == 0.5')

    print("\nGenerating response magnitude plots...")
    plot_response_magnitude_figures(df_unbiased, 'response_early', figures_dir)

    # Late window only meaningful for feedback_times
    df_feedback = df_unbiased.query('event == "feedback_times"')
    if df_feedback['response_late'].notna().any():
        plot_response_magnitude_figures(df_feedback, 'response_late', figures_dir)

    print(f"Figures saved to {figures_dir}")
