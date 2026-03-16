"""
Response Vector Analysis

Builds a response vector for each recording (mean response per trial-type
condition), computes pairwise cosine similarity, and decodes target
neuromodulator identity via L1 logistic regression.

Output: data/response_matrix.pqt, figures/response_vectors/*.svg

Usage:
    python scripts/response_vectors.py          # full pipeline
    python scripts/response_vectors.py --plot   # plot from existing data
"""
import argparse

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from iblnm.config import (
    PROJECT_ROOT, SESSIONS_FPATH, RESPONSE_MATRIX_FPATH,
    QUERY_DATABASE_LOG_FPATH, PHOTOMETRY_LOG_FPATH, TASK_LOG_FPATH,
    SUBJECTS_TO_EXCLUDE, TARGET2NM, TARGETNMS_TO_ANALYZE, FIGURE_DPI,
)
from iblnm.data import PhotometrySessionGroup
from iblnm.io import _get_default_connection
from iblnm.util import collect_session_errors, fill_empty_lists_from_group
from iblnm.analysis import within_between_similarity
from iblnm.vis import (
    plot_similarity_matrix, plot_confusion_matrix,
    plot_mean_response_vectors, plot_decoding_summary,
)

plt.ion()


def prepare_recordings(df_sessions):
    """Filter sessions and explode to one row per recording."""
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

    # TEMPFIX: normalize brain_region naming errors from Alyx metadata
    df_sessions = fill_empty_lists_from_group(df_sessions, 'brain_region')
    df_sessions = fill_empty_lists_from_group(df_sessions, 'hemisphere')

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
    print(f"  Recordings (session x region): {len(df_recordings)}")
    return df_recordings


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--plot', action='store_true',
                        help='skip extraction; plot from existing data/response_matrix.pqt')
    args = parser.parse_args()

    figures_dir = PROJECT_ROOT / 'figures/response_vectors'
    figures_dir.mkdir(parents=True, exist_ok=True)

    if args.plot:
        # =================================================================
        # Plot-only mode
        # =================================================================
        if not RESPONSE_MATRIX_FPATH.exists():
            print(f"Error: {RESPONSE_MATRIX_FPATH} not found. Run without --plot first.")
            raise SystemExit(1)

        print(f"Loading response matrix from {RESPONSE_MATRIX_FPATH}")
        df_matrix = pd.read_parquet(RESPONSE_MATRIX_FPATH)
        labels = df_matrix.index.get_level_values('target_NM')
        labels = pd.Series(labels.values, index=df_matrix.index)

    else:
        # =================================================================
        # Full pipeline
        # =================================================================
        print(f"Loading sessions from {SESSIONS_FPATH}")
        df_sessions = pd.read_parquet(SESSIONS_FPATH)
        print(f"  Total sessions: {len(df_sessions)}")

        df_recordings = prepare_recordings(df_sessions)

        one = _get_default_connection()
        group = PhotometrySessionGroup(df_recordings, one=one)

        print("Building response features...")
        df_matrix = group.get_response_features(nan_handling='drop_features')

        if len(df_matrix) == 0:
            print("No response vectors extracted. Check H5 files.")
            raise SystemExit(1)

        # Save
        RESPONSE_MATRIX_FPATH.parent.mkdir(parents=True, exist_ok=True)
        df_matrix.to_parquet(RESPONSE_MATRIX_FPATH)
        print(f"Saved {len(df_matrix)} response vectors to {RESPONSE_MATRIX_FPATH}")

        labels = df_matrix.index.get_level_values('target_NM')
        labels = pd.Series(labels.values, index=df_matrix.index)

    # =========================================================================
    # Similarity
    # =========================================================================
    print("\nComputing cosine similarity matrix...")
    if 'group' in dir():
        sim = group.response_similarity_matrix()
    else:
        from iblnm.analysis import cosine_similarity_matrix
        sim = cosine_similarity_matrix(df_matrix)

    labels_clean = labels.loc[sim.index]

    # Recover subjects for the plot
    if 'group' in dir():
        rec_indexed = group.recordings.set_index(['eid', 'target_NM'])
        subjects_clean = rec_indexed['subject'].reindex(sim.index)
    else:
        df_sessions = pd.read_parquet(SESSIONS_FPATH)
        df_recordings = prepare_recordings(df_sessions)
        rec_indexed = df_recordings.set_index(['eid', 'target_NM'])
        subjects_clean = rec_indexed['subject'].reindex(sim.index)

    fig = plot_similarity_matrix(sim, labels_clean, subjects=subjects_clean)
    fig.savefig(figures_dir / 'similarity_matrix.svg', dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"  {len(sim)} recordings in similarity matrix")

    wb = within_between_similarity(sim, labels_clean)
    within_mean = wb[wb['comparison'] == 'within']['similarity'].mean()
    between_mean = wb[wb['comparison'] == 'between']['similarity'].mean()
    print(f"  Within target-NM similarity:  {within_mean:.3f}")
    print(f"  Between target-NM similarity: {between_mean:.3f}")

    # =========================================================================
    # Decoding
    # =========================================================================
    print("\nDecoding target-NM from response vectors...")
    if 'group' in dir():
        group.decode_target()
        decoder = group.decoder
    else:
        from iblnm.analysis import TargetNMDecoder
        df_sessions = pd.read_parquet(SESSIONS_FPATH)
        df_recordings = prepare_recordings(df_sessions)
        rec_indexed = df_recordings.set_index(['eid', 'target_NM'])
        subjects = rec_indexed['subject'].reindex(df_matrix.index)
        decoder = TargetNMDecoder(df_matrix, labels, subjects)
        decoder.fit()
        decoder.unique_contribution()

    print(f"  Accuracy (raw):      {decoder.accuracy:.3f}")
    print(f"  Accuracy (balanced): {decoder.balanced_accuracy:.3f}")
    print("  Per-class recall:")
    for name, recall in decoder.per_class_accuracy.items():
        print(f"    {name}: {recall:.3f}")
    print(f"  Confusion matrix:\n{decoder.confusion}")

    fig = plot_confusion_matrix(decoder.confusion)
    fig.savefig(figures_dir / 'confusion_matrix.svg', dpi=FIGURE_DPI, bbox_inches='tight')

    # --- Mean response vectors ---
    fig = plot_mean_response_vectors(df_matrix)
    fig.savefig(figures_dir / 'mean_response_vectors.svg', dpi=FIGURE_DPI, bbox_inches='tight')

    # --- Coefficients + unique contributions (shared x-axis) ---
    print("\nFeature unique contributions:")
    contrib = decoder.contributions.sort_values('delta', ascending=False)
    print("  Top 5 features by delta accuracy:")
    print(contrib.head().to_string(index=False))

    fig = plot_decoding_summary(decoder.coefficients, contrib)
    fig.savefig(figures_dir / 'decoding_summary.svg', dpi=FIGURE_DPI, bbox_inches='tight')

    contrib.to_parquet(figures_dir / 'feature_contributions.pqt', index=False)

    print(f"\nFigures saved to {figures_dir}")
