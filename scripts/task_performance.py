"""
Task Performance Psychometric Figures

Generates 50-50 block psychometric curves and parameter distributions from
pre-computed performance data (produced by scripts/task.py).

Includes biased, ephys, and qualifying training sessions (>70% performance
with the full contrast set).

Input:  metadata/sessions.pqt, data/performance.pqt
Output: figures/task_performance/psychometric_50.svg
"""
import pandas as pd
from matplotlib import pyplot as plt

from iblnm.config import (
    PROJECT_ROOT, SESSIONS_FPATH, PERFORMANCE_FPATH, FIGURE_DPI,
    RESPONSES_FPATH, TRIAL_TIMING_FPATH,
    QUERY_DATABASE_LOG_FPATH, PHOTOMETRY_LOG_FPATH, TASK_LOG_FPATH,
    ANALYSIS_QC_BLOCKERS, SESSION_TYPES_TO_ANALYZE, TARGETNMS_TO_ANALYZE,
)
from iblnm.data import PhotometrySessionGroup
from iblnm.io import _get_default_connection
from iblnm.util import collect_session_errors
from iblnm.vis import (
    plot_psychometric_grid, plot_target_comparison, plot_rt_by_contrast,
)

plt.ion()

PSYCH_PARAMS = ['bias', 'threshold', 'lapse_left', 'lapse_right']


if __name__ == '__main__':
    # =====================================================================
    # Load sessions and create group (identical to responses.py)
    # =====================================================================
    print(f"Loading sessions from {SESSIONS_FPATH}")
    df = pd.read_parquet(SESSIONS_FPATH)
    df = collect_session_errors(
        df, [QUERY_DATABASE_LOG_FPATH, PHOTOMETRY_LOG_FPATH, TASK_LOG_FPATH])
    if PERFORMANCE_FPATH.exists():
        perf = pd.read_parquet(
            PERFORMANCE_FPATH, columns=['eid', 'fraction_correct', 'contrasts'])
        df = df.merge(perf, on='eid', how='left')

    one = _get_default_connection()
    group = PhotometrySessionGroup.from_catalog(df, one=one)
    group.filter_sessions(
        session_types=SESSION_TYPES_TO_ANALYZE,
        qc_blockers=ANALYSIS_QC_BLOCKERS,
        targetnms=TARGETNMS_TO_ANALYZE,
    )
    print(f"  Recordings (session x region): {len(group)}")

    # =====================================================================
    # Load data onto group
    # =====================================================================
    if not PERFORMANCE_FPATH.exists():
        print(f"Error: {PERFORMANCE_FPATH} not found. "
              "Run scripts/task.py first.")
        raise SystemExit(1)

    group.load_performance(PERFORMANCE_FPATH)
    if RESPONSES_FPATH.exists() and TRIAL_TIMING_FPATH.exists():
        group.load_response_magnitudes(RESPONSES_FPATH)
        group.load_trial_timing(TRIAL_TIMING_FPATH)

    # =====================================================================
    # Figures
    # =====================================================================
    output_dir = PROJECT_ROOT / 'figures/task_performance'
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Psychometric curves (2x3 grid) ---
    print("\nGenerating 50-50 psychometric figure...")
    fig1 = plot_psychometric_grid(group)
    fig1.savefig(output_dir / 'psychometric_50.svg',
                 dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"Saved: {output_dir / 'psychometric_50.svg'}")

    # --- Target-NM comparison boxplots ---
    summary_params = ['fraction_correct'] + [f'psych_50_{p}' for p in PSYCH_PARAMS]
    summary_labels = ['fraction correct', 'bias', 'threshold',
                      'lapse left', 'lapse right']
    fig2 = plot_target_comparison(group, summary_params, summary_labels)
    fig2.savefig(output_dir / 'target_comparison.svg',
                 dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"Saved: {output_dir / 'target_comparison.svg'}")

    # --- RT distributions by contrast ---
    if group.response_magnitudes is not None and group.trial_timing is not None:
        print("\nGenerating RT-by-contrast figure...")
        fig3 = plot_rt_by_contrast(group)
        fig3.savefig(output_dir / 'rt_by_contrast.svg',
                     dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Saved: {output_dir / 'rt_by_contrast.svg'}")
    else:
        print("\nSkipping RT figure: responses.pqt or trial_timing.pqt not found.")

    # =====================================================================
    # Variance stabilization analysis — at what training performance level
    # does psych parameter variance match biased+ephys?
    # =====================================================================
    """
    import numpy as np
    from scipy.special import logit as _logit

    print("\\n--- Variance stabilization analysis ---")

    # Reload performance unfiltered (need all training sessions)
    df_all = pd.read_parquet(PERFORMANCE_FPATH)
    param_cols = [f'psych_50_{p}' for p in PSYCH_PARAMS]

    # Merge session_type from sessions catalog
    session_types = df_sessions[['eid', 'session_type']].drop_duplicates()
    df_all = df_all.merge(session_types, on='eid', how='left')

    df_biased_ephys = df_all[
        df_all['session_type'].isin(['biased', 'ephys'])
    ]
    df_training = df_all[df_all['session_type'] == 'training'].copy()

    # Variance-stabilizing transforms for bounded parameters
    #   lapses: bounded [0, 0.2] -> logit(x / 0.2), with clipping to avoid +/-inf
    #   threshold: bounded [0, inf) -> log(x)
    #   bias: unbounded -> identity

    def _transform(values, col):
        v = values.copy()
        if 'lapse' in col:
            eps = 0.005
            v = np.clip(v / 0.2, eps, 1 - eps)
            return _logit(v)
        elif 'threshold' in col:
            return np.log(np.clip(v, 1e-3, None))
        return v

    def _var_transformed(series, col):
        vals = series.dropna().values
        if len(vals) <= 2:
            return np.nan
        return np.var(_transform(vals, col), ddof=1)

    ref_var = {col: _var_transformed(df_biased_ephys[col], col)
               for col in param_cols}
    print(f"  Reference variance (biased+ephys, n={len(df_biased_ephys)}):")
    for col, v in ref_var.items():
        print(f"    {col}: {v:.4f}")

    # Sweep performance thresholds
    thresholds = np.arange(0.4, 0.95, 0.01)
    variance_curves = {col: [] for col in param_cols}
    n_sessions = []

    def _mean_transformed(series, col):
        vals = series.dropna().values
        if len(vals) == 0:
            return np.nan
        return np.mean(_transform(vals, col))

    for thr in thresholds:
        df_above = df_training[df_training['fraction_correct'] >= thr]
        n_sessions.append(len(df_above))
        for col in param_cols:
            variance_curves[col].append(
                _var_transformed(df_above[col], col)
            )

    # Compute mean curves (transformed) for training sweep
    ref_mean = {col: _mean_transformed(df_biased_ephys[col], col)
                for col in param_cols}
    mean_curves = {col: [] for col in param_cols}
    for thr in thresholds:
        df_above = df_training[df_training['fraction_correct'] >= thr]
        for col in param_cols:
            mean_curves[col].append(_mean_transformed(df_above[col], col))

    # Find crossovers
    print("\\n  Variance crossovers (training var <= biased+ephys var):")
    var_crossovers = {}
    for col in param_cols:
        curve = np.array(variance_curves[col])
        ref = ref_var[col]
        below = np.where(curve <= ref)[0]
        if len(below) > 0:
            idx = below[0]
            var_crossovers[col] = thresholds[idx]
            print(f"    {col}: {thresholds[idx]:.2f} "
                  f"(n={n_sessions[idx]})")
        else:
            var_crossovers[col] = None
            print(f"    {col}: never crosses")

    # Mean crossover: first threshold where |training mean - ref mean|
    # is within 0.5 SD of the reference (small effect size boundary)
    MEAN_TOL = 0.5
    print(f"\\n  Mean crossovers (|training - ref| < {MEAN_TOL} ref SD):")
    mean_crossovers = {}
    for col in param_cols:
        ref_sd = np.sqrt(ref_var[col])
        curve = np.array(mean_curves[col])
        delta = np.abs(curve - ref_mean[col])
        close = np.where(delta <= MEAN_TOL * ref_sd)[0]
        if len(close) > 0:
            idx = close[0]
            mean_crossovers[col] = thresholds[idx]
            print(f"    {col}: {thresholds[idx]:.2f} "
                  f"(n={n_sessions[idx]})")
        else:
            mean_crossovers[col] = None
            print(f"    {col}: never crosses")

    # Plot: 2 rows (variance, mean) x 4 parameters
    fig_stab, axes_stab = plt.subplots(
        2, len(param_cols),
        figsize=(4 * len(param_cols), 6),
        squeeze=False,
    )

    for i, col in enumerate(param_cols):
        label = col.replace('psych_50_', '')

        # Row 0: variance
        ax = axes_stab[0, i]
        ax.plot(thresholds, variance_curves[col], 'k-', linewidth=1.5)
        ax.axhline(ref_var[col], color='steelblue', linestyle='--',
                   linewidth=1, label='biased+ephys')
        if var_crossovers[col] is not None:
            ax.axvline(var_crossovers[col], color='firebrick', linestyle=':',
                       linewidth=1, alpha=0.7,
                       label=f'{var_crossovers[col]:.2f}')
        ax.set_ylabel('Variance (transformed)')
        ax.set_title(label)
        ax.legend(fontsize=7)

        # Row 1: mean
        ax = axes_stab[1, i]
        ax.plot(thresholds, mean_curves[col], 'k-', linewidth=1.5)
        ax.axhline(ref_mean[col], color='steelblue', linestyle='--',
                   linewidth=1, label='biased+ephys')
        # Shade +/-MEAN_TOL SD around reference mean
        ref_sd = np.sqrt(ref_var[col])
        ax.axhspan(ref_mean[col] - MEAN_TOL * ref_sd,
                    ref_mean[col] + MEAN_TOL * ref_sd,
                    color='steelblue', alpha=0.1)
        if mean_crossovers[col] is not None:
            ax.axvline(mean_crossovers[col], color='firebrick', linestyle=':',
                       linewidth=1, alpha=0.7,
                       label=f'{mean_crossovers[col]:.2f}')
        ax.set_xlabel('Min fraction correct')
        ax.set_ylabel('Mean (transformed)')
        ax.legend(fontsize=7)

    # Secondary x-axis on top row: n sessions
    for i in range(len(param_cols)):
        ax2 = axes_stab[0, i].twiny()
        ax2.plot(thresholds, n_sessions, alpha=0)
        ax2.set_xlabel('n training sessions', fontsize=7)
        ax2.tick_params(labelsize=7)

    fig_stab.tight_layout()
    fig_stab.savefig(output_dir / 'variance_mean_stabilization.svg',
                     dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"Saved: {output_dir / 'variance_mean_stabilization.svg'}")
    """

    print("\nDone!")
