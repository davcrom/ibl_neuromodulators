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
from scipy.stats import sem as scipy_sem
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from tqdm import tqdm

from iblnm.config import (
    PROJECT_ROOT, SESSIONS_FPATH, SESSIONS_H5_DIR, EVENTS_FPATH,
    QUERY_DATABASE_LOG_FPATH, PHOTOMETRY_LOG_FPATH, TASK_LOG_FPATH,
    SESSION_TYPES_TO_ANALYZE, SUBJECTS_TO_EXCLUDE, VALID_TARGETNMS,
    RESPONSE_EVENTS, RESPONSE_WINDOWS, FIGURE_DPI,
    TARGETNM_COLORS,
)
from iblnm.data import PhotometrySession
from iblnm.io import _get_default_connection
from iblnm.util import collect_session_errors
from iblnm.analysis import compute_response_magnitude


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
        h5_path = h5_dir / f'{eid}.h5'

        if not h5_path.exists():
            continue

        # Load session data from H5
        ps = PhotometrySession(rec, one=one)
        ps.load_h5(h5_path, groups=['trials', 'responses'])

        if ps.responses is None or ps.trials is None:
            continue

        # Check region exists in H5
        available_regions = list(ps.responses.coords['region'].values)
        if brain_region not in available_regions:
            continue

        # Derive hemisphere from H5 region name
        hemisphere = brain_region[-1] if brain_region.endswith(('-l', '-r')) else None

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
    df['hemi_sign'] = df['hemisphere'].map({'l': 1, 'r': -1})
    df['relative_contrast'] = df['signed_contrast'] * df['hemi_sign']
    # NaN relative_contrast (missing hemisphere) → NaN side
    df['side'] = np.where(df['relative_contrast'] >= 0, 'contra', 'ipsi')
    df.loc[df['hemi_sign'].isna(), 'side'] = np.nan
    return df


def plot_response_magnitude(df_events, response_col, figures_dir):
    """Plot response magnitude by contrast, feedback, and hemisphere.

    One figure per target-NM × event. Each figure has two panels (contra, ipsi)
    with errorbar plots showing mean-of-subject-means ± SEM across subjects.

    Parameters
    ----------
    df_events : pd.DataFrame
        Events table with laterality columns added.
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

        event_label = event.replace('_times', '')
        color = TARGETNM_COLORS.get(target_nm, 'black')

        fig, axs = plt.subplots(1, 2, sharey=True,
                                gridspec_kw={'wspace': 0.05})
        fig.suptitle(
            f'{target_nm} — {event_label} ({window_label})\n'
            f'{df_group["eid"].nunique()} sessions, {n_subjects} subjects',
            fontsize=10,
        )

        for ax, side in zip(axs, ['contra', 'ipsi']):
            df_side = df_group.query('side == @side')
            if len(df_side) == 0:
                continue
            contrasts = sorted(df_side['contrast'].unique())

            for feedback, ls in [(1, '-'), (-1, '--')]:
                df_fb = df_side.query('feedbackType == @feedback')
                if len(df_fb) == 0:
                    continue

                # Mean-of-subject-means ± SEM across subjects
                means, sems = [], []
                for c in contrasts:
                    df_c = df_fb[df_fb['contrast'] == c]
                    subj_means = df_c.groupby('subject')[response_col].mean()
                    means.append(subj_means.mean())
                    sems.append(scipy_sem(subj_means) if len(subj_means) > 1
                                else np.nan)

                label = 'correct' if feedback == 1 else 'incorrect'
                ax.errorbar(
                    np.arange(len(contrasts)), means, yerr=sems,
                    marker='o', color=color, linestyle=ls, label=label,
                )

            ax.set_xticks(np.arange(len(contrasts)))
            ax.set_xticklabels([f'{c:.0f}' for c in contrasts])
            ax.set_xlabel('Contrast (%)')
            ax.axhline(0, ls='--', color='gray', lw=0.5)
            ax.text(0.5, 0.02, side.capitalize(),
                    ha='center', transform=ax.transAxes, fontsize=9)

            if side == 'contra':
                ax.set_ylabel('Response (z-score)')
            else:
                ax.tick_params(left=False)
                ax.spines['left'].set_visible(False)
                ax.legend(frameon=False, loc='upper left',
                          bbox_to_anchor=(1, 1), fontsize=8)

        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        fig.tight_layout()
        fname = f'{target_nm}_{event_label}_{window_label}.svg'
        fig.savefig(figures_dir / fname, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close(fig)


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
        df_sessions['session_type'].isin(SESSION_TYPES_TO_ANALYZE)
        & ~df_sessions['subject'].isin(SUBJECTS_TO_EXCLUDE)
    ]

    # Derive QC flag (same logic as dataset_overview.py)
    _qc_blockers = {
        'MissingExtractedData', 'InsufficientTrials',
        'TrialsNotInPhotometryTime', 'QCValidationError', 'FewUniqueSamples',
    }
    df_sessions = df_sessions[
        df_sessions['logged_errors'].apply(
            lambda e: not any(err in _qc_blockers for err in e)
        )
    ].copy()
    print(f"  After filtering & QC: {len(df_sessions)}")

    # =========================================================================
    # Explode to one row per recording
    # =========================================================================
    df_recordings = (
        df_sessions
        .explode(['target_NM', 'brain_region', 'hemisphere'])
        .loc[lambda df: df['target_NM'].isin(VALID_TARGETNMS)]
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
    plot_response_magnitude(df_unbiased, 'response_early', figures_dir)

    # Late window only meaningful for feedback_times
    df_feedback = df_unbiased.query('event == "feedback_times"')
    if df_feedback['response_late'].notna().any():
        plot_response_magnitude(df_feedback, 'response_late', figures_dir)

    print(f"Figures saved to {figures_dir}")
