"""
Response Analysis Pipeline

Extracts trial-level response magnitudes and recording-level response
vectors, then produces similarity, decoding, contrast-based, and wheel
kinematics figures.

Includes biased, ephys, and qualifying training sessions (>70% performance
with the full contrast set).

Output:
    data/responses.pqt           — trial-level response magnitudes
    data/trial_timing.pqt        — per-trial reaction and movement times
    data/peak_velocity.pqt       — per-trial peak wheel velocity
    data/response_matrix.pqt     — response feature vectors
    figures/responses/*.svg      — all response analysis figures
    figures/traces/*.svg         — mean response traces

Usage:
    python scripts/responses.py          # full pipeline: extract + plot
    python scripts/responses.py --plot   # plot only from existing parquet files
"""
import argparse

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from iblnm.config import (
    PROJECT_ROOT, SESSIONS_FPATH,
    RESPONSES_FPATH, TRIAL_TIMING_FPATH, PEAK_VELOCITY_FPATH,
    RESPONSE_MATRIX_FPATH, SIMILARITY_MATRIX_FPATH,
    RESPONSE_EVENTS, FIGURE_DPI,
    MIN_TRAINING_PERFORMANCE, REQUIRED_CONTRASTS,
)
from iblnm.data import PhotometrySessionGroup
from iblnm.io import _get_default_connection
from iblnm.task import add_relative_contrast
from iblnm.util import derive_target_nm
from iblnm.vis import (
    plot_relative_contrast, plot_similarity_matrix, plot_confusion_matrix,
    plot_mean_response_vectors, plot_decoding_summary, plot_empirical_similarity,
    plot_lmm_response, plot_lmm_summary,
    plot_within_target_similarity, plot_response_decoding_summary,
    plot_mean_response_traces,
    plot_wheel_lmm_summary,
)
from iblnm.analysis import (
    within_between_similarity, mean_similarity_by_target,
)

plt.ion()


# =========================================================================
# Response magnitude plotting
# =========================================================================

def print_response_summary(df_responses):
    """Print a summary of the response magnitudes DataFrame."""
    n_sessions = df_responses['eid'].nunique()
    n_subjects = df_responses['subject'].nunique()

    print(f"\n{len(df_responses)} rows, {n_sessions} sessions, {n_subjects} subjects")
    print("\nTrials per target-NM:")
    summary = (
        df_responses[df_responses['event'] == RESPONSE_EVENTS[0]]
        .groupby('target_NM')
        .agg(
            n_subjects=('subject', 'nunique'),
            n_sessions=('eid', 'nunique'),
            n_trials=('trial', 'count'),
        )
    )
    print(summary.to_string())


def plot_response_figures(group, figures_dir, response_col='response_early',
                        aggregation='pool'):
    """Plot response magnitude by contrast x feedback x hemisphere.

    Parameters
    ----------
    group : PhotometrySessionGroup
        Must have group.response_magnitudes populated.
    figures_dir : Path
        Output directory for SVG files.
    response_col : str
        Column name for the response magnitude.
    aggregation : str
        'pool' or 'subject'.
    """
    df_responses = add_relative_contrast(group.response_magnitudes.copy())
    if group.trial_timing is not None:
        df_responses = df_responses.merge(
            group.trial_timing[['eid', 'trial', 'event', 'reaction_time']],
            on=['eid', 'trial', 'event'], how='left',
        )
    df_unbiased = df_responses.query('probabilityLeft == 0.5')

    window_label = response_col.replace('response_', '')
    df = df_unbiased.dropna(subset=[response_col]).copy()
    if 'reaction_time' in df.columns:
        df = df.query('choice != 0 and reaction_time > 0.05')
    else:
        df = df.query('choice != 0')

    for (target_nm, event), df_group in df.groupby(['target_NM', 'event']):
        n_subjects = df_group['subject'].nunique()
        if n_subjects < 2:
            continue

        event_label = event.replace('_times', '')
        fig = plot_relative_contrast(df_group, response_col, target_nm, event,
                                     window_label=window_label,
                                     aggregation=aggregation)
        fname = f'{target_nm}_{event_label}_{window_label}.svg'
        fig.savefig(figures_dir / fname, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close(fig)


# =========================================================================
# LMM statistical analysis
# =========================================================================

def plot_lmm_figures(group, figures_dir, response_col='response_early'):
    """Generate per-target LMM response plots and consolidated summaries.

    Requires ``group.fit_lmm()`` to have been called already.

    Parameters
    ----------
    group : PhotometrySessionGroup
        Must have lmm_results populated.
    figures_dir : Path
        Output directory for SVG and CSV files.
    response_col : str
        Column name for the response magnitude (used for raw data overlay).
    """
    window_label = response_col.replace('response_', '')

    if not group.lmm_results:
        print("  No LMM results to plot.")
        return

    # Prepare raw data for overlay
    df_raw = add_relative_contrast(group.response_magnitudes.copy())
    if group.trial_timing is not None:
        df_raw = df_raw.merge(
            group.trial_timing[['eid', 'trial', 'event', 'reaction_time']],
            on=['eid', 'trial', 'event'], how='left',
        )
    df_raw = df_raw.query('probabilityLeft == 0.5')
    df_raw = df_raw.dropna(subset=[response_col])
    if 'reaction_time' in df_raw.columns:
        df_raw = df_raw.query('choice != 0 and reaction_time > 0.05')
    else:
        df_raw = df_raw.query('choice != 0')

    # Per-target modeled response plots (save and close)
    for (target_nm, event_label), result in group.lmm_results.items():
        event = event_label + '_times'
        df_group = df_raw[
            (df_raw['target_NM'] == target_nm) & (df_raw['event'] == event)
        ]
        fig = plot_lmm_response(
            result.predictions, target_nm, event,
            window_label=window_label,
            df_raw=df_group, response_col=response_col,
        )
        fname = f'{target_nm}_{event_label}_{window_label}_lmm.svg'
        fig.savefig(figures_dir / fname, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close(fig)

    # Save coefficients
    if len(group.lmm_coefficients) > 0:
        csv_path = figures_dir / f'lmm_coefficients_{window_label}.csv'
        group.lmm_coefficients.to_csv(csv_path, index=False)
        print(f"  LMM coefficients saved to {csv_path}")

    # Consolidated summary — one figure per event
    events_seen = sorted(set(ev for _, ev in group.lmm_results.keys()))
    for event_label in events_seen:
        fig = plot_lmm_summary(group, event_label)
        fig.savefig(
            figures_dir / f'lmm_summary_{event_label}_{window_label}.svg',
            dpi=FIGURE_DPI, bbox_inches='tight')
    print("  LMM summary plots saved")


# =========================================================================
# Wheel kinematics LMM analysis
# =========================================================================

def plot_wheel_lmm_figures(group, figures_dir, response_col='response_early'):
    """Generate wheel kinematics LMM summary plot and CSV.

    Requires ``group.fit_wheel_lmm()`` to have been called.

    Parameters
    ----------
    group : PhotometrySessionGroup
        Must have wheel_lmm_summary populated.
    figures_dir : Path
        Output directory for SVG and CSV files.
    response_col : str
        Column name for the NM response magnitude.
    """
    summary = group.wheel_lmm_summary
    if summary is None or len(summary) == 0:
        print("  No wheel LMM results to plot.")
        return

    # Summary figure: delta R² across contrasts
    fig = plot_wheel_lmm_summary(summary)
    fig.savefig(figures_dir / 'wheel_lmm_delta_r2.svg',
                dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)

    # CSV export
    csv_path = figures_dir / 'wheel_lmm_summary.csv'
    summary.to_csv(csv_path, index=False)
    print(f"  Wheel LMM summary saved to {csv_path}")


# =========================================================================
# Response vectors plotting
# =========================================================================

def plot_vectors_figures(group, figures_dir):
    """Plot similarity matrix, confusion matrix, and decoding summary.

    Parameters
    ----------
    group : PhotometrySessionGroup
        Must have response_features, similarity_matrix, and decoder populated.
    figures_dir : Path
        Output directory for SVG files.
    """
    sim = group.similarity_matrix
    labels = group.response_features.index.get_level_values('target_NM')
    labels = pd.Series(labels.values, index=group.response_features.index)
    labels_clean = labels.loc[sim.index]

    recs = group.recordings.copy()
    if 'fiber_idx' not in recs.columns:
        recs['fiber_idx'] = 0
    idx_cols = ['eid', 'target_NM', 'fiber_idx']
    rec_indexed = (
        recs[idx_cols + ['subject']]
        .drop_duplicates(subset=idx_cols)
        .set_index(idx_cols)
    )
    subjects_clean = rec_indexed['subject'].reindex(sim.index)

    # Similarity matrix
    fig = plot_similarity_matrix(sim, labels_clean, subjects=subjects_clean)
    fig.savefig(figures_dir / 'similarity_matrix.svg',
                dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"  {len(sim)} recordings in similarity matrix")

    wb = within_between_similarity(sim, labels_clean)
    within_mean = wb[wb['comparison'] == 'within']['similarity'].mean()
    between_mean = wb[wb['comparison'] == 'between']['similarity'].mean()
    print(f"  Within target-NM similarity:  {within_mean:.3f}")
    print(f"  Between target-NM similarity: {between_mean:.3f}")

    # Empirical similarity matrix (target × target mean pairwise similarity)
    target_sim = mean_similarity_by_target(sim, labels_clean)
    target_sim_loso = mean_similarity_by_target(sim, labels_clean,
                                                subjects=subjects_clean)
    fig = plot_empirical_similarity(target_sim, loso_matrix=target_sim_loso)
    fig.savefig(figures_dir / 'empirical_similarity.svg',
                dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"  Empirical similarity (all pairs):\n{target_sim.to_string()}")
    print(f"  Empirical similarity (cross-subject):\n{target_sim_loso.to_string()}")

    # Confusion matrix
    decoder = group.decoder
    print(f"\n  Accuracy (raw):      {decoder.accuracy:.3f}")
    print(f"  Accuracy (balanced): {decoder.balanced_accuracy:.3f}")
    print("  Per-class recall:")
    for name, recall in decoder.per_class_accuracy.items():
        print(f"    {name}: {recall:.3f}")
    print(f"  Confusion matrix:\n{decoder.confusion}")

    fig = plot_confusion_matrix(decoder.confusion)
    fig.savefig(figures_dir / 'confusion_matrix.svg',
                dpi=FIGURE_DPI, bbox_inches='tight')

    # Within-target similarity barplot
    fig = plot_within_target_similarity(sim, labels_clean, subjects_clean)
    fig.savefig(figures_dir / 'within_target_similarity.svg',
                dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)

    # Mean response vectors (legacy — kept for standalone viewing)
    fig = plot_mean_response_vectors(group.response_features)
    fig.savefig(figures_dir / 'mean_response_vectors.svg',
                dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)

    # Coefficients + unique contributions
    print("\nFeature unique contributions:")
    contrib = decoder.contributions.sort_values('delta', ascending=False)
    print("  Top 5 features by delta accuracy:")
    print(contrib.head().to_string(index=False))

    fig = plot_decoding_summary(decoder.coefficients, contrib)
    fig.savefig(figures_dir / 'decoding_summary.svg',
                dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)

    # Unified response vectors + decoding summary
    fig = plot_response_decoding_summary(
        group.response_features, decoder.coefficients, contrib)
    fig.savefig(figures_dir / 'response_decoding_summary.svg',
                dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)

    contrib.to_parquet(figures_dir / 'feature_contributions.pqt', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--plot', action='store_true',
                        help='skip extraction; plot from existing parquet files')
    args = parser.parse_args()

    figures_dir = PROJECT_ROOT / 'figures/responses'
    figures_dir.mkdir(parents=True, exist_ok=True)

    # =====================================================================
    # Load sessions
    # =====================================================================
    print(f"Loading sessions from {SESSIONS_FPATH}")
    df_sessions = pd.read_parquet(SESSIONS_FPATH)
    print(f"  Total sessions: {len(df_sessions)}")

    # Derive target_NM and NM from brain_region
    # (brain_region already filled and corrected in query_database.py)
    df_sessions = derive_target_nm(df_sessions)

    # TEMPFIX: drop sessions where brain_region, hemisphere, and target_NM
    # have different numbers of entries. This happens because brain_region
    # and hemisphere are populated independently in query_database.py, so
    # some sessions end up with e.g. brain_region=['VTA'] but hemisphere=[].
    # These cannot be exploded into one row per recording.
    _parallel_cols = ['target_NM', 'brain_region', 'hemisphere']
    _lengths_match = df_sessions[_parallel_cols].apply(
        lambda row: len(set(
            len(v) if isinstance(v, (list, np.ndarray)) else 1
            for v in row
        )) == 1,
        axis=1,
    )
    n_mismatched = (~_lengths_match).sum()
    if n_mismatched > 0:
        _bad = df_sessions.loc[~_lengths_match, ['eid', 'subject', 'brain_region', 'hemisphere']]
        print(f"  Dropping {n_mismatched} sessions with mismatched brain_region/hemisphere lengths:")
        for _, row in _bad.iterrows():
            print(f"    {row['subject']} {row['eid']}: "
                  f"brain_region={row['brain_region']}, hemisphere={row['hemisphere']}")
        df_sessions = df_sessions[_lengths_match].copy()

    # Explode to one row per recording (brain_region x hemisphere x target_NM)
    # Add fiber_idx to distinguish bilateral same-target recordings
    df_recordings = df_sessions.explode(_parallel_cols).copy()
    df_recordings['fiber_idx'] = df_recordings.groupby('eid').cumcount()

    # =====================================================================
    # Create group and apply standard filters
    # =====================================================================
    one = _get_default_connection()
    group = PhotometrySessionGroup(df_recordings, one=one)
    group.filter_recordings(
        session_types=('biased', 'ephys', 'training'),
        min_performance=MIN_TRAINING_PERFORMANCE,
        required_contrasts=REQUIRED_CONTRASTS,
    )
    print(f"  Recordings (session x region): {len(group)}")

    if args.plot:
        # =================================================================
        # Plot-only mode: load pre-existing parquet files
        # =================================================================
        if not RESPONSES_FPATH.exists():
            print(f"Error: {RESPONSES_FPATH} not found. Run without --plot first.")
            raise SystemExit(1)
        if not RESPONSE_MATRIX_FPATH.exists():
            print(f"Error: {RESPONSE_MATRIX_FPATH} not found. Run without --plot first.")
            raise SystemExit(1)

        print(f"Loading response magnitudes from {RESPONSES_FPATH}")
        group.response_magnitudes = pd.read_parquet(RESPONSES_FPATH)

        if TRIAL_TIMING_FPATH.exists():
            print(f"Loading trial timing from {TRIAL_TIMING_FPATH}")
            group.trial_timing = pd.read_parquet(TRIAL_TIMING_FPATH)

        if PEAK_VELOCITY_FPATH.exists():
            print(f"Loading peak velocity from {PEAK_VELOCITY_FPATH}")
            group.peak_velocity = pd.read_parquet(PEAK_VELOCITY_FPATH)

        print(f"Loading response matrix from {RESPONSE_MATRIX_FPATH}")
        group.response_features = pd.read_parquet(RESPONSE_MATRIX_FPATH)

        mean_traces_fpath = PROJECT_ROOT / 'data/mean_traces.pqt'
        if mean_traces_fpath.exists():
            print(f"Loading mean traces from {mean_traces_fpath}")
            group.mean_traces = pd.read_parquet(mean_traces_fpath)

    else:
        # =================================================================
        # Full pipeline: extract from H5 files
        # =================================================================

        # --- Load traces cache ---
        print("\nLoading response traces...")
        group.load_response_traces()

        # --- Response magnitudes ---
        print("Computing trial-level response magnitudes...")
        group.get_response_magnitudes()

        if len(group.response_magnitudes) == 0:
            print("No response magnitudes extracted. Check H5 files exist.")
            raise SystemExit(1)

        RESPONSES_FPATH.parent.mkdir(parents=True, exist_ok=True)
        group.response_magnitudes.to_parquet(RESPONSES_FPATH, index=False)
        print(f"Saved response magnitudes to {RESPONSES_FPATH}")

        group.trial_timing.to_parquet(TRIAL_TIMING_FPATH, index=False)
        print(f"Saved trial timing to {TRIAL_TIMING_FPATH}")

        # --- Mean traces ---
        print("Computing mean traces...")
        group.get_mean_traces()
        mean_traces_fpath = PROJECT_ROOT / 'data/mean_traces.pqt'
        group.mean_traces.to_parquet(mean_traces_fpath, index=False)
        print(f"Saved mean traces to {mean_traces_fpath}")

        # --- Response vectors ---
        print("\nBuilding response features...")
        group.get_response_features(nan_handling='drop_features')

        if len(group.response_features) == 0:
            print("No response vectors extracted. Check H5 files.")
            raise SystemExit(1)

        RESPONSE_MATRIX_FPATH.parent.mkdir(parents=True, exist_ok=True)
        group.response_features.to_parquet(RESPONSE_MATRIX_FPATH)
        print(f"Saved {len(group.response_features)} response vectors "
              f"to {RESPONSE_MATRIX_FPATH}")

    # =====================================================================
    # Response magnitude plots
    # =====================================================================
    print_response_summary(group.response_magnitudes)

    print("\nGenerating response magnitude plots...")
    plot_response_figures(group, figures_dir)
    print(f"Response magnitude figures saved to {figures_dir}")

    # =====================================================================
    # LMM statistical analysis
    # =====================================================================
    print("\nFitting linear mixed-effects models...")
    group.fit_lmm()
    for (tnm, ev), result in group.lmm_results.items():
        ve = result.variance_explained
        print(f"  {tnm} x {ev}: R2 marginal={ve['marginal']:.3f}, "
              f"conditional={ve['conditional']:.3f}")
    plot_lmm_figures(group, figures_dir)

    # =====================================================================
    # Wheel kinematics LMM
    # =====================================================================
    if group.peak_velocity is None:
        print("\nEnriching with peak velocity...")
        group.enrich_peak_velocity()
        group.peak_velocity.to_parquet(PEAK_VELOCITY_FPATH, index=False)
        print(f"Saved peak velocity to {PEAK_VELOCITY_FPATH}")

    n_with_wheel = group.peak_velocity['peak_velocity'].notna().sum()
    print(f"  {n_with_wheel} trials with wheel data")

    print("Fitting wheel kinematics LMMs...")
    group.fit_wheel_lmm()
    if len(group.wheel_lmm_summary) > 0:
        n_sig = (group.wheel_lmm_summary['lrt_pvalue'] < 0.05).sum()
        print(f"  {len(group.wheel_lmm_summary)} model comparisons, "
              f"{n_sig} significant (p < 0.05)")
        plot_wheel_lmm_figures(group, figures_dir)
    else:
        print("  No wheel LMM results (insufficient data).")

    # =====================================================================
    # Response vectors: similarity + decoding
    # =====================================================================
    print("\nComputing cosine similarity matrix...")
    group.response_similarity_matrix()

    group.similarity_matrix.to_parquet(SIMILARITY_MATRIX_FPATH)
    print(f"Saved similarity matrix to {SIMILARITY_MATRIX_FPATH}")

    print("Decoding target-NM from response vectors...")
    group.decode_target()

    # Save decoding results
    data_dir = PROJECT_ROOT / 'data'
    decoder = group.decoder
    decoder.confusion.to_parquet(data_dir / 'confusion_matrix.pqt')
    decoder.coefficients.to_parquet(data_dir / 'decoding_coefficients.pqt')
    decoder.contributions.to_parquet(
        data_dir / 'feature_contributions.pqt', index=False)
    print(f"Saved decoding results to {data_dir}")

    plot_vectors_figures(group, figures_dir)
    print(f"Response vector figures saved to {figures_dir}")

    # =====================================================================
    # Mean response traces per target-NM
    # =====================================================================
    traces_figures_dir = PROJECT_ROOT / 'figures/traces'
    traces_figures_dir.mkdir(parents=True, exist_ok=True)

    if group.mean_traces is not None and len(group.mean_traces) > 0:
        print("\nGenerating mean response trace plots...")
        for event in RESPONSE_EVENTS:
            df_event = group.mean_traces[group.mean_traces['event'] == event]
            if len(df_event) == 0:
                continue
            event_label = event.replace('_times', '')
            fig = plot_mean_response_traces(df_event, event)
            fig.savefig(traces_figures_dir / f'mean_traces_{event_label}.svg',
                        dpi=FIGURE_DPI, bbox_inches='tight')
            plt.close(fig)
        print(f"Trace figures saved to {traces_figures_dir}")

    # Free trace cache
    group.flush_response_traces()
