"""
Response Analysis Pipeline

Extracts trial-level response magnitudes (events) and recording-level response
vectors, then produces similarity, decoding, and contrast-based figures.

Combines the former events.py and response_vectors.py scripts.

Output:
    data/events.pqt              — trial-level response magnitudes
    data/response_matrix.pqt     — response feature vectors
    figures/events/*.svg         — response magnitude by contrast plots
    figures/events/*_lmm.svg     — LMM modeled response plots
    figures/events/lmm_*.csv     — LMM coefficient tables
    figures/events/lmm_*.svg     — variance explained summary
    figures/response_vectors/*.svg — similarity, confusion, decoding plots

Usage:
    python scripts/responses.py          # full pipeline: extract + plot
    python scripts/responses.py --plot   # plot only from existing parquet files
"""
import argparse

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from iblnm.config import (
    PROJECT_ROOT, SESSIONS_FPATH, EVENTS_FPATH, RESPONSE_MATRIX_FPATH,
    RESPONSE_EVENTS, FIGURE_DPI,
)
from iblnm.data import PhotometrySessionGroup
from iblnm.io import _get_default_connection
from iblnm.task import add_relative_contrast
from iblnm.util import derive_target_nm
from iblnm.vis import (
    plot_relative_contrast, plot_similarity_matrix, plot_confusion_matrix,
    plot_mean_response_vectors, plot_decoding_summary, plot_empirical_similarity,
    plot_lmm_response, plot_lmm_variance_explained,
)
from iblnm.analysis import (
    within_between_similarity, mean_similarity_by_target, fit_events_lmm,
)

plt.ion()


# =========================================================================
# Ad-hoc data fix helpers (remove when corrected upstream)
# =========================================================================

# TEMPFIX: normalize brain_region naming errors from Alyx metadata
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


# =========================================================================
# Events plotting
# =========================================================================

def print_events_summary(df_events):
    """Print a summary of the events DataFrame."""
    n_sessions = df_events['eid'].nunique()
    n_subjects = df_events['subject'].nunique()

    print(f"\n{len(df_events)} rows, {n_sessions} sessions, {n_subjects} subjects")
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


def plot_events_figures(group, figures_dir, response_col='response_early',
                        aggregation='pool'):
    """Plot response magnitude by contrast x feedback x hemisphere.

    Parameters
    ----------
    group : PhotometrySessionGroup
        Must have group.events populated.
    figures_dir : Path
        Output directory for SVG files.
    response_col : str
        Column name for the response magnitude.
    aggregation : str
        'pool' or 'subject'.
    """
    df_events = add_relative_contrast(group.events.copy())
    df_unbiased = df_events.query('probabilityLeft == 0.5')

    window_label = response_col.replace('response_', '')
    df = df_unbiased.dropna(subset=[response_col]).copy()
    df = df.query('choice != 0 and reaction_time > 0.05')

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


# =========================================================================
# LMM statistical analysis
# =========================================================================

def fit_and_plot_lmm(group, figures_dir, response_col='response_early'):
    """Fit LMMs per (target_NM, event) and generate modeled response + R² plots.

    For each group, fits: response ~ log(contrast) * side * reward | subject.
    Saves coefficient tables to CSV and generates modeled response plots and
    a variance explained summary.

    Parameters
    ----------
    group : PhotometrySessionGroup
        Must have group.events populated.
    figures_dir : Path
        Output directory for SVG and CSV files.
    response_col : str
        Column name for the response magnitude.
    """
    df_events = add_relative_contrast(group.events.copy())
    df_unbiased = df_events.query('probabilityLeft == 0.5')

    window_label = response_col.replace('response_', '')
    df = df_unbiased.dropna(subset=[response_col]).copy()
    df = df.query('choice != 0 and reaction_time > 0.05')

    ve_dict = {}
    all_summaries = []

    for (target_nm, event), df_group in df.groupby(['target_NM', 'event']):
        n_subjects = df_group['subject'].nunique()
        if n_subjects < 2:
            continue

        event_label = event.replace('_times', '')
        print(f"  Fitting LMM: {target_nm} × {event_label} "
              f"({len(df_group)} trials, {n_subjects} subjects)...")

        result = fit_events_lmm(df_group, response_col)
        if result is None:
            print(f"    Convergence failed — skipping")
            continue

        ve = result.variance_explained
        ve_dict[(target_nm, event_label)] = ve
        print(f"    R² marginal={ve['marginal']:.3f}, "
              f"conditional={ve['conditional']:.3f}")

        # Save coefficient table
        summary = result.summary_df.copy()
        summary.insert(0, 'target_NM', target_nm)
        summary.insert(1, 'event', event_label)
        summary.index.name = 'term'
        all_summaries.append(summary.reset_index())

        # Modeled response plot
        fig = plot_lmm_response(
            result.predictions, target_nm, event,
            window_label=window_label,
            df_raw=df_group, response_col=response_col,
        )
        fname = f'{target_nm}_{event_label}_{window_label}_lmm.svg'
        fig.savefig(figures_dir / fname, dpi=FIGURE_DPI, bbox_inches='tight')

    # Save all coefficients to one CSV
    if all_summaries:
        coefs_df = pd.concat(all_summaries, ignore_index=True)
        csv_path = figures_dir / f'lmm_coefficients_{window_label}.csv'
        coefs_df.to_csv(csv_path, index=False)
        print(f"  LMM coefficients saved to {csv_path}")

    # Variance explained summary plot
    if ve_dict:
        fig = plot_lmm_variance_explained(ve_dict)
        fig.savefig(figures_dir / f'lmm_variance_explained_{window_label}.svg',
                    dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"  Variance explained plot saved")


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

    rec_indexed = group.recordings.set_index(['eid', 'target_NM'])
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

    # Mean response vectors
    fig = plot_mean_response_vectors(group.response_features)
    fig.savefig(figures_dir / 'mean_response_vectors.svg',
                dpi=FIGURE_DPI, bbox_inches='tight')

    # Coefficients + unique contributions
    print("\nFeature unique contributions:")
    contrib = decoder.contributions.sort_values('delta', ascending=False)
    print("  Top 5 features by delta accuracy:")
    print(contrib.head().to_string(index=False))

    fig = plot_decoding_summary(decoder.coefficients, contrib)
    fig.savefig(figures_dir / 'decoding_summary.svg',
                dpi=FIGURE_DPI, bbox_inches='tight')

    contrib.to_parquet(figures_dir / 'feature_contributions.pqt', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--plot', action='store_true',
                        help='skip extraction; plot from existing parquet files')
    args = parser.parse_args()

    events_figures_dir = PROJECT_ROOT / 'figures/events'
    vectors_figures_dir = PROJECT_ROOT / 'figures/response_vectors'
    events_figures_dir.mkdir(parents=True, exist_ok=True)
    vectors_figures_dir.mkdir(parents=True, exist_ok=True)

    # =====================================================================
    # Load sessions
    # =====================================================================
    print(f"Loading sessions from {SESSIONS_FPATH}")
    df_sessions = pd.read_parquet(SESSIONS_FPATH)
    print(f"  Total sessions: {len(df_sessions)}")

    # =====================================================================
    # Ad-hoc data fixes (remove when corrected upstream)
    # =====================================================================

    # TEMPFIX: normalize brain_region naming errors from Alyx metadata
    df_sessions['brain_region'] = df_sessions['brain_region'].apply(_fix_regions)

    # Derive target_NM and NM from brain_region
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
    df_recordings = df_sessions.explode(_parallel_cols).copy()

    # =====================================================================
    # Create group and apply standard filters
    # =====================================================================
    one = _get_default_connection()
    group = PhotometrySessionGroup(df_recordings, one=one)
    group.filter_recordings()
    print(f"  Recordings (session x region): {len(group)}")

    if args.plot:
        # =================================================================
        # Plot-only mode: load pre-existing parquet files
        # =================================================================
        if not EVENTS_FPATH.exists():
            print(f"Error: {EVENTS_FPATH} not found. Run without --plot first.")
            raise SystemExit(1)
        if not RESPONSE_MATRIX_FPATH.exists():
            print(f"Error: {RESPONSE_MATRIX_FPATH} not found. Run without --plot first.")
            raise SystemExit(1)

        print(f"Loading events from {EVENTS_FPATH}")
        group.events = pd.read_parquet(EVENTS_FPATH)

        print(f"Loading response matrix from {RESPONSE_MATRIX_FPATH}")
        group.response_features = pd.read_parquet(RESPONSE_MATRIX_FPATH)

    else:
        # =================================================================
        # Full pipeline: extract from H5 files
        # =================================================================

        # --- Events ---
        print("\nExtracting trial-level response magnitudes...")
        group.get_events()

        if len(group.events) == 0:
            print("No events extracted. Check H5 files exist.")
            raise SystemExit(1)

        EVENTS_FPATH.parent.mkdir(parents=True, exist_ok=True)
        group.events.to_parquet(EVENTS_FPATH, index=False)
        print(f"Saved events to {EVENTS_FPATH}")

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
    # Events plots
    # =====================================================================
    print_events_summary(group.events)

    print("\nGenerating response magnitude plots...")
    plot_events_figures(group, events_figures_dir)
    print(f"Events figures saved to {events_figures_dir}")

    # =====================================================================
    # LMM statistical analysis
    # =====================================================================
    print("\nFitting linear mixed-effects models...")
    fit_and_plot_lmm(group, events_figures_dir)

    # =====================================================================
    # Response vectors: similarity + decoding + plots
    # =====================================================================
    print("\nComputing cosine similarity matrix...")
    group.response_similarity_matrix()

    print("Decoding target-NM from response vectors...")
    group.decode_target()

    plot_vectors_figures(group, vectors_figures_dir)
    print(f"Response vector figures saved to {vectors_figures_dir}")
