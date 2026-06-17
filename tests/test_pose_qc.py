"""Tests for the pure data-transform helpers in scripts/pose_qc.py."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import scripts.pose_qc as pose_qc


def test_order_qc_categories_severity_present_only_nan_last():
    cats = pose_qc.order_qc_categories(['PASS', 'FAIL', 'PASS', np.nan, 'WARNING'])
    assert cats[:3] == ['FAIL', 'WARNING', 'PASS']
    assert len(cats) == 4
    assert isinstance(cats[3], float) and np.isnan(cats[3])


def test_spearman_finite_ignores_nan_rows():
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    y = np.array([2.0, np.nan, 3.0, 1.0, 0.0])
    mask = np.isfinite(x) & np.isfinite(y)
    expected = spearmanr(x[mask], y[mask])
    result = pose_qc.spearman_finite(x, y)
    assert result.correlation == expected.correlation
    assert result.pvalue == expected.pvalue


def test_max_abs_lag_takes_max_magnitude_across_thirds():
    df = pd.DataFrame({
        'peak_lag_early': [-0.65, 0.1],
        'peak_lag_mid': [0.2, -4.9],
        'peak_lag_late': [0.3, 4.8],
    })
    result = pose_qc.max_abs_lag(df)
    np.testing.assert_array_equal(result.values, [0.65, 4.9])
