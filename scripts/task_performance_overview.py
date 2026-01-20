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
    VALID_TARGETS,
)
from iblnm.task import count_sessions_to_stage
from iblnm.util import clean_sessions, drop_junk_duplicates
from iblnm.vis import (
    plot_stage_barplot,
    plot_sessions_to_stage_cdf,
    plot_psychometric_parameter_trajectory,
    create_psychometric_figure,
)


def create_learning_figure(df_stage_counts, df_training_fits, target_nms=None):
    """Create learning progression figure with target-NM as columns."""
    if target_nms is None:
        target_nms = [t for t in VALID_TARGETS if t in df_stage_counts['target_NM'].values]

    n_cols = len(target_nms)
    n_rows = 7  # barplot, 2 CDFs, 4 parameter trajectories

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.5 * n_rows))
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    for col, target_nm in enumerate(target_nms):
        df_target = df_stage_counts[df_stage_counts['target_NM'] == target_nm]
        df_training_target = df_training_fits[df_training_fits['target_NM'] == target_nm]

        # Row 0: Stage barplot
        plot_stage_barplot(df_target, ax=axes[0, col])
        axes[0, col].set_title(target_nm)

        # Row 1: CDF sessions to biased
        plot_sessions_to_stage_cdf(df_target, 'biased', ax=axes[1, col])

        # Row 2: CDF biased sessions to ephys
        plot_sessions_to_stage_cdf(df_target, 'ephys', ax=axes[2, col])

        # Rows 3-6: Parameter trajectories
        for row, param in enumerate(['bias', 'threshold', 'lapse_left', 'lapse_right'], start=3):
            param_col = f'psych_50_{param}'
            if param_col in df_training_target.columns:
                df_plot = df_training_target.rename(columns={param_col: param})
                plot_psychometric_parameter_trajectory(df_plot, param, ax=axes[row, col])
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

    print(f"Loading performance from {PERFORMANCE_FPATH}")
    df_performance = pd.read_parquet(PERFORMANCE_FPATH)

    # Filter out sessions with invalid block structure
    if 'block_structure_valid' in df_performance.columns:
        n_invalid = (~df_performance['block_structure_valid']).sum()
        if n_invalid > 0:
            print(f"Excluding {n_invalid} sessions with invalid block structure")
        df_performance = df_performance[df_performance['block_structure_valid'] == True]

    # Compute stage counts
    df_stage_counts = count_sessions_to_stage(df_sessions)

    # Output directory
    output_dir = PROJECT_ROOT / 'figures/task_performance'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Learning figure (training sessions)
    df_training = df_performance[df_performance['session_type'] == 'training']
    if len(df_training) > 0:
        print("\nGenerating learning figure...")
        fig = create_learning_figure(df_stage_counts, df_training)
        fig.savefig(output_dir / 'learning_progression.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {output_dir / 'learning_progression.png'}")

    # Psychometric figure (biased/ephys sessions)
    df_biased_ephys = df_performance[df_performance['session_type'].isin(['biased', 'ephys'])]
    if 'has_extracted_photometry_signal' in df_biased_ephys.columns:
        df_biased_ephys = df_biased_ephys[df_biased_ephys['has_extracted_photometry_signal'] == True]

    if len(df_biased_ephys) > 0:
        print("\nGenerating psychometric figure...")
        fig = create_psychometric_figure(df_biased_ephys)
        fig.savefig(output_dir / 'psychometric_analysis.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {output_dir / 'psychometric_analysis.png'}")

    print("\nDone!")
