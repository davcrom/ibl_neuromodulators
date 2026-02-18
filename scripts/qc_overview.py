"""
QC Overview Visualizations

Plots:
1. Histograms for basic QC metrics (n_unique_samples, n_band_inversions) with cutoffs
2. Violinplots per target for other metrics (quantile-transformed across all targets)
3. Joint distribution grid for metric pairs
4. PCA scatter plots colored by target, date, session type (z-score normalized)
5. QC failure rates over time
6. Photobleaching tau over time
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, QuantileTransformer

from iblnm.config import (
    PROJECT_ROOT, SESSIONS_FPATH, QCPHOTOMETRY_FPATH, VALID_TARGETS, FIGURE_DPI,
    QC_SLIDING_METRICS, QC_PREPROCESSING, SESSIONTYPE2COLOR,
)
from iblnm.util import process_regions
from iblnm.vis import (
    violinplot, plot_joint_distributions,
    TARGETNM_COLORS, TARGETNM_POSITIONS
)

plt.ion()

# Output directory
figures_dir = PROJECT_ROOT / 'figures/qc_overview'
figures_dir.mkdir(parents=True, exist_ok=True)

# Define metrics
BASIC_QC_METRICS = ['n_unique_samples', 'n_band_inversions']
BASIC_QC_CUTOFFS = {'n_unique_samples': 0.1, 'n_band_inversions': 0}

# Non-basic metrics for violinplots and PCA
OTHER_METRICS = [m for m in QC_SLIDING_METRICS if m not in BASIC_QC_METRICS]
OTHER_METRICS.extend(QC_PREPROCESSING)


# =============================================================================
# Load Data
# =============================================================================

print(f"Loading QC data from {QCPHOTOMETRY_FPATH}")
df_qc = pd.read_parquet(QCPHOTOMETRY_FPATH)

# Filter to GCaMP only (exclude isosbestic reference)
df_qc = df_qc[df_qc['band'] == 'GCaMP'].copy()
print(f"Filtered to GCaMP: {len(df_qc)} signals")

# Merge session metadata (NM, date, session_type, subject)
print(f"Merging session metadata from {SESSIONS_FPATH}")
df_sessions = pd.read_parquet(SESSIONS_FPATH)[['eid', 'subject', 'NM', 'session_type', 'start_time']]
df_sessions['date'] = pd.to_datetime(df_sessions['start_time'], format='ISO8601').dt.date
df_qc = df_qc.merge(df_sessions[['eid', 'subject', 'NM', 'session_type', 'date']], on='eid', how='left')

# Process brain regions: normalize names, add hemisphere, infer NM, create target_NM
df_qc = process_regions(df_qc)
print(f"Valid targets: {len(df_qc)} signals from {df_qc['eid'].nunique()} sessions")

# Clean metric values and report
ALL_METRICS = BASIC_QC_METRICS + OTHER_METRICS
print("\nMetric ranges:")
for metric in ALL_METRICS:
    if metric not in df_qc.columns:
        print(f"  {metric}: MISSING")
        continue
    col = df_qc[metric]
    n_nan = col.isna().sum()
    n_inf = np.isinf(col).sum()
    n_valid = len(col) - n_nan - n_inf
    print(f"  {metric}: [{col.min():.3g}, {col.max():.3g}], n={n_valid}, nan={n_nan}, inf={n_inf}")

# Replace inf with NaN (inf occurs when ratio metrics have zero denominator)
for metric in ALL_METRICS:
    if metric in df_qc.columns:
        df_qc[metric] = df_qc[metric].replace([np.inf, -np.inf], np.nan)

if len(df_qc) == 0:
    print("No valid QC data. Exiting.")
    exit(0)

# Get targets present in data
targets = [t for t in VALID_TARGETS if t in df_qc['target_NM'].unique()]
positions = [TARGETNM_POSITIONS[t] for t in targets]
colors = [TARGETNM_COLORS[t] for t in targets]

# Add basic QC pass/fail flags
df_qc['passes_unique_samples'] = df_qc['n_unique_samples'] > BASIC_QC_CUTOFFS['n_unique_samples']
df_qc['passes_band_inversions'] = df_qc['n_band_inversions'] == BASIC_QC_CUTOFFS['n_band_inversions']
df_qc['passes_basic_qc'] = df_qc['passes_unique_samples'] & df_qc['passes_band_inversions']


# =============================================================================
# 1. Basic QC Histograms
# =============================================================================

print("\nGenerating basic QC histograms...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# n_unique_samples histogram
values = df_qc['n_unique_samples'].dropna().values
cutoff = BASIC_QC_CUTOFFS['n_unique_samples']
n_pass = (values > cutoff).sum()
n_fail = (values <= cutoff).sum()
axes[0].hist(values, bins=50, color='gray', edgecolor='black', alpha=0.7)
axes[0].axvline(cutoff, color='red', linestyle='--', linewidth=2, label=f'cutoff={cutoff}')
axes[0].set_xlabel('n_unique_samples')
axes[0].set_ylabel('Count')
axes[0].set_title(f'n_unique_samples\npass={n_pass}, fail={n_fail}')
axes[0].legend()

# n_band_inversions: bar for pass vs fail
values = df_qc['n_band_inversions'].dropna().values
n_pass_inv = (values == 0).sum()
n_fail_inv = (values > 0).sum()
axes[1].bar(['0 (pass)', '>0 (fail)'], [n_pass_inv, n_fail_inv], color=['green', 'red'], alpha=0.7)
axes[1].set_ylabel('Count')
axes[1].set_title(f'n_band_inversions\npass={n_pass_inv}, fail={n_fail_inv}')

plt.tight_layout()
fig.savefig(figures_dir / '1_basic_qc_histograms.svg', dpi=FIGURE_DPI, bbox_inches='tight')
print(f"Saved {figures_dir / '1_basic_qc_histograms.svg'}")

# =============================================================================
# Filter to signals passing basic QC
# =============================================================================

df_qc_passed = df_qc[df_qc['passes_basic_qc']].copy()
print(f"\nFiltered to {len(df_qc_passed)} signals passing basic QC (from {len(df_qc)})")


# =============================================================================
# 2. Violinplots per Target (quantile-transformed)
# =============================================================================

print("\nGenerating violinplots (quantile-transformed, signals passing basic QC)...")

# Quantile transform across ALL targets for comparable distributions
df_violin = df_qc_passed.copy()
metrics_to_transform = [m for m in OTHER_METRICS if m in df_violin.columns]

for metric in metrics_to_transform:
    valid_mask = df_violin[metric].notna()
    if valid_mask.sum() > 10:
        qt = QuantileTransformer(output_distribution='normal', random_state=42)
        df_violin.loc[valid_mask, f'{metric}_transformed'] = qt.fit_transform(
            df_violin.loc[valid_mask, metric].values.reshape(-1, 1)
        ).flatten()

n_metrics = len(metrics_to_transform)
n_cols = 3
n_rows = int(np.ceil(n_metrics / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
axes = axes.flatten()

for ax, metric in zip(axes, metrics_to_transform):
    col = f'{metric}_transformed'
    if col not in df_violin.columns:
        col = metric

    data = [df_violin[df_violin['target_NM'] == t][col].dropna().values for t in targets]

    violinplot(ax, data, positions=positions, colors=colors,
               remove_outliers=True, show_outliers=True)

    ax.set_xticks(positions)
    ax.set_xticklabels([t.split('-')[0] for t in targets], rotation=45, ha='right')
    ax.set_ylabel(metric)
    ax.set_title(f'{metric}\n(quantile-transformed)')

# Turn off unused axes
for ax in axes[n_metrics:]:
    ax.axis('off')

plt.tight_layout()
fig.savefig(figures_dir / '2_violinplots.svg', dpi=FIGURE_DPI, bbox_inches='tight')
print(f"Saved {figures_dir / '2_violinplots.svg'}")


# =============================================================================
# 3. Joint Distribution Grid
# =============================================================================

print("\nGenerating joint distributions (signals passing basic QC)...")
fig, X = plot_joint_distributions(df_qc_passed, metrics=OTHER_METRICS, transform=True, bins=30,
                                   figsize=(12, 12))
fig.savefig(figures_dir / '3_joint_distributions.svg', dpi=FIGURE_DPI, bbox_inches='tight')
print(f"Saved {figures_dir / '3_joint_distributions.svg'}")


# =============================================================================
# 4. PCA Scatter Plots (z-score normalized)
# =============================================================================

print("\nGenerating PCA scatter plots (z-score normalized, signals passing basic QC)...")

# Exclude preprocessing metrics (bleaching_tau, iso_correlation) from PCA
PCA_METRICS = [m for m in OTHER_METRICS if m not in QC_PREPROCESSING]
df_pca = df_qc_passed.dropna(subset=PCA_METRICS).copy()

if len(df_pca) < 10:
    print("Not enough data for PCA")
else:
    X = df_pca[PCA_METRICS].values

    # Quantile transform to normal distribution
    qt = QuantileTransformer(output_distribution='normal', random_state=42)
    X_transformed = qt.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_transformed)

    df_pca['PC1'] = X_pca[:, 0]
    df_pca['PC2'] = X_pca[:, 1]

    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Color by target
    for target in targets:
        mask = df_pca['target_NM'] == target
        if mask.sum() > 0:
            axes[0].scatter(df_pca.loc[mask, 'PC1'], df_pca.loc[mask, 'PC2'],
                           c=TARGETNM_COLORS[target], label=target, alpha=0.6, s=10)
    axes[0].set_xlabel(f'PC1 ({var1:.1f}%)')
    axes[0].set_ylabel(f'PC2 ({var2:.1f}%)')
    axes[0].set_title('By target (z-score normalized)')
    axes[0].legend(fontsize=8, markerscale=2)

    # 2. Color by date
    if 'date' in df_pca.columns:
        dates = pd.to_datetime(df_pca['date'])
        date_nums = (dates - dates.min()).dt.days
        sc = axes[1].scatter(df_pca['PC1'], df_pca['PC2'], c=date_nums,
                            cmap='viridis', alpha=0.6, s=10)
        plt.colorbar(sc, ax=axes[1], label='Days from start')
    else:
        axes[1].text(0.5, 0.5, 'No date column', ha='center', va='center',
                    transform=axes[1].transAxes)
    axes[1].set_xlabel(f'PC1 ({var1:.1f}%)')
    axes[1].set_ylabel(f'PC2 ({var2:.1f}%)')
    axes[1].set_title('By date')

    # 3. Color by session type
    if 'session_type' in df_pca.columns:
        for stype in df_pca['session_type'].unique():
            mask = df_pca['session_type'] == stype
            color = SESSIONTYPE2COLOR.get(stype, 'gray')
            axes[2].scatter(df_pca.loc[mask, 'PC1'], df_pca.loc[mask, 'PC2'],
                           c=color, label=stype, alpha=0.6, s=10)
        axes[2].legend(fontsize=8, markerscale=2)
    else:
        axes[2].text(0.5, 0.5, 'No session_type column', ha='center', va='center',
                    transform=axes[2].transAxes)
    axes[2].set_xlabel(f'PC1 ({var1:.1f}%)')
    axes[2].set_ylabel(f'PC2 ({var2:.1f}%)')
    axes[2].set_title('By session type')

    plt.tight_layout()
    fig.savefig(figures_dir / '4_pca_scatter.svg', dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"Saved {figures_dir / '4_pca_scatter.svg'}")


# =============================================================================
# 5. QC Failure Rates Over Time
# =============================================================================

print("\nGenerating QC failure rates over time...")

# Group by month
df_qc['month'] = pd.to_datetime(df_qc['date']).dt.to_period('M')

monthly_stats = df_qc.groupby('month').agg(
    n_signals=('eid', 'count'),
    n_fail_unique=('passes_unique_samples', lambda x: (~x).sum()),
    n_fail_inversions=('passes_band_inversions', lambda x: (~x).sum()),
    n_fail_basic=('passes_basic_qc', lambda x: (~x).sum()),
).reset_index()

monthly_stats['fail_rate_unique'] = monthly_stats['n_fail_unique'] / monthly_stats['n_signals'] * 100
monthly_stats['fail_rate_inversions'] = monthly_stats['n_fail_inversions'] / monthly_stats['n_signals'] * 100
monthly_stats['fail_rate_basic'] = monthly_stats['n_fail_basic'] / monthly_stats['n_signals'] * 100

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Top: failure rates
months = monthly_stats['month'].astype(str)
x = range(len(months))

axes[0].plot(x, monthly_stats['fail_rate_unique'], 'o-', label='n_unique_samples ≤ 0.1', color='blue')
axes[0].plot(x, monthly_stats['fail_rate_inversions'], 's-', label='n_band_inversions > 0', color='orange')
axes[0].plot(x, monthly_stats['fail_rate_basic'], '^-', label='Either (basic QC fail)', color='red')
axes[0].set_ylabel('Failure rate (%)')
axes[0].set_title('Basic QC failure rates over time')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Bottom: sample counts
axes[1].bar(x, monthly_stats['n_signals'], color='gray', alpha=0.7)
axes[1].set_ylabel('N signals')
axes[1].set_xlabel('Month')
axes[1].set_xticks(x)
axes[1].set_xticklabels(months, rotation=45, ha='right')

plt.tight_layout()
fig.savefig(figures_dir / '5_qc_failure_over_time.svg', dpi=FIGURE_DPI, bbox_inches='tight')
print(f"Saved {figures_dir / '5_qc_failure_over_time.svg'}")


# =============================================================================
# 6. Photobleaching Tau Over Time
# =============================================================================

print("\nGenerating photobleaching tau over time...")

if 'bleaching_tau' in df_qc_passed.columns:
    df_tau = df_qc_passed[['date', 'bleaching_tau']].dropna()
    df_tau['month'] = pd.to_datetime(df_tau['date']).dt.to_period('M')

    # Average across all sessions per month
    monthly = df_tau.groupby('month')['bleaching_tau'].agg(['mean', 'std', 'count']).reset_index()
    months = monthly['month'].astype(str)
    x = range(len(months))

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.errorbar(x, monthly['mean'], yerr=monthly['std'],
                fmt='o-', color='black', capsize=3)
    ax.set_ylabel('Bleaching tau (s)')
    ax.set_xlabel('Month')
    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=45, ha='right')
    ax.set_title('Photobleaching tau over time (mean ± std, all sessions)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(figures_dir / '6_bleaching_tau_over_time.svg', dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"Saved {figures_dir / '6_bleaching_tau_over_time.svg'}")
else:
    print("bleaching_tau not found in data, skipping plot 6")

print(f"\nFigures saved to {figures_dir}")
