"""
Test qc_signals directly with mock data
"""
import numpy as np
import pandas as pd
from iblphotometry import metrics
from iblphotometry.qc import qc_signals
from iblnm.config import QC_RAW_METRICS, QC_SLIDING_METRICS, QC_METRICS_KWARGS, QC_SLIDING_KWARGS

# Create mock photometry data
# Simulate a dual-fiber recording with GCaMP and Isosbestic bands
n_samples = 10000
timestamps = np.arange(n_samples) / 100.0  # 100 Hz sampling

# Create two brain regions (simulate multi-fiber recording)
vta_signal = np.sin(2 * np.pi * 0.1 * timestamps) + np.random.normal(0, 0.1, n_samples) + 5
snc_signal = np.cos(2 * np.pi * 0.15 * timestamps) + np.random.normal(0, 0.15, n_samples) + 4

photometry = {
    'GCaMP': pd.DataFrame({
        'VTA': vta_signal,
        'SNc': snc_signal
    }, index=timestamps),
    'Isosbestic': pd.DataFrame({
        'VTA': vta_signal * 0.8 + np.random.normal(0, 0.05, n_samples),
        'SNc': snc_signal * 0.7 + np.random.normal(0, 0.08, n_samples)
    }, index=timestamps)
}

print("Mock photometry data structure:")
print(f"Type: {type(photometry)}")
for band, df in photometry.items():
    print(f"  {band}: shape={df.shape}, columns={list(df.columns)}")

# Convert metric names to functions (as done in PhotometrySession.run_qc)
raw_metric_funcs = [getattr(metrics, m) for m in QC_RAW_METRICS]
sliding_metric_funcs = [getattr(metrics, m) for m in QC_SLIDING_METRICS]

print(f"\nRaw metrics: {[m.__name__ for m in raw_metric_funcs]}")
print(f"Sliding metrics: {[m.__name__ for m in sliding_metric_funcs]}")

print("\nRunning QC on raw metrics...")
try:
    qc_raw = qc_signals(
        photometry,
        metrics=raw_metric_funcs,
        metrics_kwargs=QC_METRICS_KWARGS,
        signal_band='GCaMP',
    )
    print(f"Raw QC shape: {qc_raw.shape}")
    print(qc_raw.head(10))
except Exception as e:
    print(f"Error in raw QC: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\nRunning QC on sliding metrics...")
try:
    qc_sliding = qc_signals(
        photometry,
        metrics=sliding_metric_funcs,
        metrics_kwargs=QC_METRICS_KWARGS,
        signal_band='GCaMP',
        sliding_kwargs=QC_SLIDING_KWARGS,
    )
    print(f"Sliding QC shape: {qc_sliding.shape}")
    print(qc_sliding.head(20))

    # Check sliding window data
    sliding_with_windows = qc_sliding[qc_sliding['window'].notna()]
    print(f"\nRows with window info: {len(sliding_with_windows)}")

except Exception as e:
    print(f"Error in sliding QC: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n\nCombining results...")
qc_results = pd.concat([qc_raw, qc_sliding], ignore_index=True)
print(f"Combined QC shape: {qc_results.shape}")
print(f"Unique metrics: {qc_results['metric'].unique()}")
print(f"Unique brain regions: {qc_results['brain_region'].unique()}")
print(f"Unique bands: {qc_results['band'].unique()}")
