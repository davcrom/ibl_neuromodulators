"""
Task Performance Psychometric Figures

Generates 50-50 block psychometric curves and parameter distributions from
pre-computed performance data (produced by scripts/task.py).

Includes biased, ephys, and qualifying training sessions (>70% performance
with the full contrast set).

Input:  metadata/sessions.pqt, data/performance.pqt
Output: figures/task_performance/psychometric_50.svg
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from iblnm.config import (
    PROJECT_ROOT, SESSIONS_FPATH, PERFORMANCE_FPATH, FIGURE_DPI,
    MIN_TRAINING_PERFORMANCE, REQUIRED_CONTRASTS,
    TARGETNMS_TO_ANALYZE,
)
from iblnm.data import PhotometrySessionGroup
from iblnm.io import _get_default_connection
from iblnm.util import derive_target_nm
from iblnm.vis import plot_psychometric_curves_50, plot_parameter_box, TARGETNM_COLORS

plt.ion()

PSYCH_PARAMS = ['bias', 'threshold', 'lapse_left', 'lapse_right']


if __name__ == '__main__':
    # =====================================================================
    # Load sessions (identical to responses.py)
    # =====================================================================
    print(f"Loading sessions from {SESSIONS_FPATH}")
    df_sessions = pd.read_parquet(SESSIONS_FPATH)
    print(f"  Total sessions: {len(df_sessions)}")

    df_sessions = derive_target_nm(df_sessions)

    # Drop sessions where parallel list columns have mismatched lengths
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
        _bad = df_sessions.loc[
            ~_lengths_match,
            ['eid', 'subject', 'brain_region', 'hemisphere'],
        ]
        print(f"  Dropping {n_mismatched} sessions with mismatched "
              "brain_region/hemisphere lengths:")
        for _, row in _bad.iterrows():
            print(f"    {row['subject']} {row['eid']}: "
                  f"brain_region={row['brain_region']}, "
                  f"hemisphere={row['hemisphere']}")
        df_sessions = df_sessions[_lengths_match].copy()

    # Explode to one row per recording
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

    # =====================================================================
    # Load performance data
    # =====================================================================
    if not PERFORMANCE_FPATH.exists():
        print(f"Error: {PERFORMANCE_FPATH} not found. "
              "Run scripts/task.py first.")
        raise SystemExit(1)

    print(f"Loading performance from {PERFORMANCE_FPATH}")
    df_performance = pd.read_parquet(PERFORMANCE_FPATH)

    # Merge target_NM from group recordings (inner join keeps only
    # sessions that passed filter_recordings)
    rec_meta = (
        group.recordings[['eid', 'subject', 'target_NM']]
        .drop_duplicates()
    )
    df_performance = df_performance.merge(rec_meta, on='eid', how='inner')
    print(f"  {len(df_performance)} session-target rows after merge")

    # =====================================================================
    # Psychometric figure — rows: target-NM, cols: curve + 4 parameters
    # =====================================================================
    output_dir = PROJECT_ROOT / 'figures/task_performance'
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = [t for t in TARGETNMS_TO_ANALYZE
               if t in df_performance['target_NM'].values]

    if len(targets) == 0:
        print("No targets with performance data to plot.")
        raise SystemExit(0)

    n_rows = len(targets)
    n_cols = 1 + len(PSYCH_PARAMS)  # psychometric curve + 4 parameters

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3 * n_cols, 3 * n_rows),
        squeeze=False,
        gridspec_kw={'width_ratios': [2] + [1] * len(PSYCH_PARAMS)},
    )

    print("\nGenerating 50-50 psychometric figure...")
    for row, target_nm in enumerate(targets):
        df_target = df_performance[
            df_performance['target_NM'] == target_nm
        ]
        color = TARGETNM_COLORS.get(target_nm, 'gray')

        # Column 0: psychometric curve
        plot_psychometric_curves_50(df_target, target_nm=target_nm,
                                    ax=axes[row, 0])
        axes[row, 0].set_title(target_nm if row == 0 else '')
        axes[row, 0].set_ylabel(target_nm)

        # Columns 1–4: parameter boxplots
        for col, param in enumerate(PSYCH_PARAMS, start=1):
            param_col = f'psych_50_{param}'
            if param_col in df_target.columns:
                plot_parameter_box(df_target, param_col, axes[row, col],
                                   color=color)
            if row == 0:
                axes[row, col].set_title(param.replace('_', ' '))

    # Unify y-limits per parameter column across all targets;
    # lapse_left and lapse_right share the same scale
    lapse_cols = [i + 1 for i, p in enumerate(PSYCH_PARAMS) if 'lapse' in p]
    other_cols = [i + 1 for i, p in enumerate(PSYCH_PARAMS) if 'lapse' not in p]

    for col_group in [lapse_cols] + [[c] for c in other_cols]:
        ymin = min(axes[r, c].get_ylim()[0]
                   for r in range(n_rows) for c in col_group)
        ymax = max(axes[r, c].get_ylim()[1]
                   for r in range(n_rows) for c in col_group)
        for r in range(n_rows):
            for c in col_group:
                axes[r, c].set_ylim(ymin, ymax)

    fig.tight_layout()
    fig.savefig(output_dir / 'psychometric_50.svg',
                dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"Saved: {output_dir / 'psychometric_50.svg'}")

    # =====================================================================
    # Summary: target-NM comparison with ANOVA + post-hoc
    # =====================================================================
    from scipy.stats import kruskal, mannwhitneyu
    from itertools import combinations

    summary_params = ['fraction_correct'] + [f'psych_50_{p}' for p in PSYCH_PARAMS]
    summary_labels = ['fraction correct', 'bias', 'threshold',
                      'lapse left', 'lapse right']

    fig_sum, axes_sum = plt.subplots(
        1, len(summary_params),
        figsize=(3.5 * len(summary_params), 4),
        squeeze=False,
    )

    print("\n--- Target-NM comparison (ANOVA + post-hoc) ---")
    for i, (col, label) in enumerate(zip(summary_params, summary_labels)):
        ax = axes_sum[0, i]

        # Collect per-target data
        groups_data = {}
        for tnm in targets:
            vals = df_performance.loc[
                df_performance['target_NM'] == tnm, col
            ].dropna().values
            if len(vals) > 0:
                groups_data[tnm] = vals

        if len(groups_data) < 2:
            ax.set_title(label)
            continue

        # Boxplots — unfilled, lines colored by target-NM
        positions = list(range(len(groups_data)))
        target_names = list(groups_data.keys())
        bp = ax.boxplot(
            [groups_data[t] for t in target_names],
            positions=positions, widths=0.5, patch_artist=True,
            showfliers=False,
        )
        for j, tnm in enumerate(target_names):
            color = TARGETNM_COLORS.get(tnm, 'gray')
            bp['boxes'][j].set_facecolor('none')
            bp['boxes'][j].set_edgecolor(color)
            bp['boxes'][j].set_linewidth(1.5)
            bp['medians'][j].set_color(color)
            bp['medians'][j].set_linewidth(2)
            for k in (2 * j, 2 * j + 1):
                bp['whiskers'][k].set_color(color)
                bp['caps'][k].set_color(color)

        ax.set_xticks(positions)
        ax.set_xticklabels([t.split('-')[0] for t in target_names],
                           rotation=45, ha='right', fontsize=8)
        ax.set_title(label)

        # Kruskal-Wallis
        h_stat, p_kw = kruskal(*groups_data.values())
        print(f"  {label}: H={h_stat:.2f}, p={p_kw:.4f}", end='')

        if p_kw >= 0.05:
            print(" (n.s.)")
            y_top = ax.get_ylim()[1]
            y_rng = y_top - ax.get_ylim()[0]
            ax.set_ylim(ax.get_ylim()[0], y_top + y_rng * 0.15)
            ax.text(0.5, y_top + y_rng * 0.05,
                    f'H={h_stat:.1f}, p={p_kw:.3f} n.s.',
                    ha='center', va='bottom', fontsize=7)
            continue

        print(" *")

        # Post-hoc Mann-Whitney U (Bonferroni corrected)
        pairs = list(combinations(range(len(target_names)), 2))
        n_comparisons = len(pairs)
        sig_pairs = []
        for a, b in pairs:
            _, p_mw = mannwhitneyu(groups_data[target_names[a]],
                                   groups_data[target_names[b]],
                                   alternative='two-sided')
            p_corrected = min(p_mw * n_comparisons, 1.0)
            if p_corrected < 0.05:
                sig_pairs.append((a, b, p_corrected))

        # Draw significance brackets
        y_max = ax.get_ylim()[1]
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        step = y_range * 0.06
        for k, (a, b, p_corr) in enumerate(sig_pairs):
            y = y_max + step * (k + 0.5)
            stars = '***' if p_corr < 0.001 else '**' if p_corr < 0.01 else '*'
            ax.plot([a, a, b, b], [y - step * 0.15, y, y, y - step * 0.15],
                    color='black', linewidth=0.8)
            ax.text((a + b) / 2, y, stars, ha='center', va='bottom',
                    fontsize=8)

        # Place ANOVA label above the topmost bracket, expand ylim to fit
        label_y = y_max + step * (len(sig_pairs) + 0.5) if sig_pairs else y_max
        ax.set_ylim(ax.get_ylim()[0],
                     label_y + step * 2.5)
        ax.text(0.5, label_y + step * 0.5,
                f'H={h_stat:.1f}, p={p_kw:.1e}',
                ha='center', va='bottom', fontsize=7)

    fig_sum.tight_layout()
    fig_sum.savefig(output_dir / 'target_comparison.svg',
                    dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"Saved: {output_dir / 'target_comparison.svg'}")

    # =====================================================================
    # Variance stabilization analysis — at what training performance level
    # does psych parameter variance match biased+ephys?
    # =====================================================================
    print("\n--- Variance stabilization analysis ---")

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
    #   lapses: bounded [0, 0.2] → logit(x / 0.2), with clipping to avoid ±inf
    #   threshold: bounded [0, ∞) → log(x)
    #   bias: unbounded → identity
    from scipy.special import logit as _logit

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
    print("\n  Variance crossovers (training var <= biased+ephys var):")
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
    print(f"\n  Mean crossovers (|training - ref| < {MEAN_TOL} ref SD):")
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

    # Plot: 2 rows (variance, mean) × 4 parameters
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
        # Shade ±MEAN_TOL SD around reference mean
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

    print("\nDone!")
