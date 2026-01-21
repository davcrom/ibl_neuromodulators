"""
Task Performance Overview Figures

Generates learning progression and psychometric analysis figures from
pre-computed performance data.

Input: metadata/performance.pqt
Output: figures/task_performance/*.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from iblnm.config import (
    SESSIONS_FPATH,
    PERFORMANCE_FPATH,
    PROJECT_ROOT,
    TARGETNM2POSITION,
    NM_COLORS,
)

# Targets to include in overview (exclude MR, SI, PPT)
OVERVIEW_TARGETS = ['VTA-DA', 'SNc-DA', 'DR-5HT', 'LC-NE', 'NBM-ACh']
from iblnm.task import count_sessions_to_stage
from iblnm.util import clean_sessions, drop_junk_duplicates, add_target_nm
from iblnm.vis import (
    plot_stage_barplot,
    plot_sessions_to_stage_cdf,
    plot_psychometric_parameter_trajectory,
    plot_performance_trajectory,
    create_psychometric_figure,
    TARGETNM_COLORS,
)


plt.ion()


def create_learning_figure(df_stage_counts, df_training_fits, target_nms=None):
    """Create learning progression figure with target-NM as columns."""
    if target_nms is None:
        # Use overview targets, sorted by position, filtered to those present in data
        available = set(df_stage_counts['target_NM'].dropna().unique())
        target_nms = sorted(
            [t for t in OVERVIEW_TARGETS if t in available],
            key=lambda x: TARGETNM2POSITION.get(x, 999)
        )

    n_cols = len(target_nms)
    n_rows = 8  # barplot, 2 CDFs, performance trajectory, 4 parameter trajectories

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.5 * n_rows))
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    for col, target_nm in enumerate(target_nms):
        df_target = df_stage_counts[df_stage_counts['target_NM'] == target_nm]
        df_training_target = df_training_fits[df_training_fits['target_NM'] == target_nm]
        color = TARGETNM_COLORS.get(target_nm, 'gray')

        # Row 0: Stage barplot (uses its own color scheme)
        plot_stage_barplot(df_target, ax=axes[0, col])
        axes[0, col].set_title(target_nm)

        # Row 1: CDF sessions to biased
        plot_sessions_to_stage_cdf(df_target, 'biased', ax=axes[1, col], color=color)

        # Row 2: CDF biased sessions to ephys
        plot_sessions_to_stage_cdf(df_target, 'ephys', ax=axes[2, col], color=color)

        # Row 3: Performance trajectory
        plot_performance_trajectory(df_training_target, 'fraction_correct', ax=axes[3, col], color=color)

        # Rows 4-7: Parameter trajectories
        for row, param in enumerate(['bias', 'threshold', 'lapse_left', 'lapse_right'], start=4):
            param_col = f'psych_50_{param}'
            if param_col in df_training_target.columns:
                df_plot = df_training_target.rename(columns={param_col: param})
                plot_psychometric_parameter_trajectory(df_plot, param, ax=axes[row, col], color=color)
            else:
                axes[row, col].text(0.5, 0.5, f'No {param} data', ha='center', va='center',
                                   transform=axes[row, col].transAxes)

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    # Load data
    print(f"Loading sessions from {SESSIONS_FPATH}")
    df_sessions = pd.read_parquet(SESSIONS_FPATH)
    df_sessions = clean_sessions(df_sessions)
    df_sessions = drop_junk_duplicates(df_sessions, ['subject', 'day_n'])

    # Explode by target and add target_NM
    df_sessions = add_target_nm(df_sessions)
    print(f"Loaded {len(df_sessions)} session-target combinations")

    print(f"Loading performance from {PERFORMANCE_FPATH}")
    df_performance = pd.read_parquet(PERFORMANCE_FPATH)

    # Merge target_NM from sessions
    df_performance = df_performance.merge(
        df_sessions[['eid', 'target_NM']].drop_duplicates(),
        on='eid',
        how='left'
    )

    # Compute stage counts and merge target_NM
    df_stage_counts = count_sessions_to_stage(df_sessions)
    subject_to_target = df_sessions.groupby('subject')['target_NM'].first()
    df_stage_counts['target_NM'] = df_stage_counts['subject'].map(subject_to_target)

    # Output directory
    output_dir = PROJECT_ROOT / 'figures/task_performance'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Learning figure (training sessions)
    df_training = df_performance[df_performance['session_type'] == 'training']
    if len(df_training) > 0:
        print("\nGenerating learning figure...")
        fig = create_learning_figure(df_stage_counts, df_training)
        fig.savefig(output_dir / 'learning_progression.svg', dpi=150, bbox_inches='tight')
        print(f"Saved: {output_dir / 'learning_progression.svg'}")

    # Filter out sessions with invalid block structure
    if 'block_structure_valid' in df_performance.columns:
        n_total = len(df_performance)
        df_performance = df_performance[df_performance['block_structure_valid'] == True]
        n_excluded = n_total - len(df_performance)
        if n_excluded > 0:
            print(f"Excluding {n_excluded} sessions with invalid/missing block structure")

    # Psychometric figure (biased/ephys sessions)
    df_biased_ephys = df_performance[df_performance['session_type'].isin(['biased', 'ephys'])]
    if 'has_extracted_photometry_signal' in df_biased_ephys.columns:
        df_biased_ephys = df_biased_ephys[df_biased_ephys['has_extracted_photometry_signal'] == True]

    if len(df_biased_ephys) > 0:
        print("\nGenerating psychometric figure...")
        fig = create_psychometric_figure(df_biased_ephys, target_nms=OVERVIEW_TARGETS)
        fig.savefig(output_dir / 'psychometric_analysis.svg', dpi=150, bbox_inches='tight')
        print(f"Saved: {output_dir / 'psychometric_analysis.svg'}")

    print("\nDone!")
