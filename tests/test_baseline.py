"""Tests for scripts/baseline.py regression models."""
import numpy as np

from scripts.baseline import linear_regression


def test_linear_regression_binary_lpm_recovers_effect_at_ceiling():
    """LPM (OLS on a 0/1 outcome) controls for contrast and stays finite even
    when a contrast level is all-correct — the separation case that breaks a
    logit. Baseline drives accuracy within contrast; the slope recovers the
    positive effect and incremental R^2 exceeds the pure-noise baseline.
    """
    rng = np.random.default_rng(2)
    n = 3000
    contrast = rng.choice([0.0, 0.25, 1.0], size=n)
    baseline = rng.normal(size=n)
    logit_p = -1.0 + 4.0 * contrast + 1.5 * baseline
    correct = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit_p))).astype(float)
    correct[contrast == 1.0] = 1.0  # ceiling level: all-correct, separates a logit

    res = linear_regression(correct, contrast, baseline)
    null = linear_regression(correct, contrast, rng.normal(size=n))

    assert np.isfinite(res['slope']) and np.isfinite(res['r2'])
    assert res['slope'] > 0
    assert res['r2'] > null['r2']


def test_linear_regression_controls_for_contrast():
    """Baseline slope is estimated after C(contrast) absorbs the outcome offset.

    The outcome depends on both contrast and baseline; the returned slope must
    recover the positive baseline effect and the r2 (baseline's incremental
    R^2 over a contrast-only model) must exceed the r2 obtained when baseline
    is pure noise. The contrast effect is large so a spurious baseline signal
    would appear if contrast were not controlled.
    """
    rng = np.random.default_rng(1)
    n = 3000
    contrast = rng.choice([0.0, 0.25, 1.0], size=n)
    baseline = rng.normal(size=n)
    outcome = 5.0 * contrast + 0.5 * baseline + rng.normal(scale=0.1, size=n)

    res = linear_regression(outcome, contrast, baseline)

    noise = rng.normal(size=n)
    null = linear_regression(outcome, contrast, noise)

    assert res['slope'] > 0
    assert res['r2'] > null['r2']
