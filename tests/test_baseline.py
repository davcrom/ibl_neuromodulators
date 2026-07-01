"""Tests for scripts/baseline.py regression models."""
import numpy as np

from scripts.baseline import logistic_regression, linear_regression


def test_logistic_regression_controls_for_contrast():
    """Baseline slope is estimated after C(contrast) absorbs difficulty.

    Accuracy is driven by both contrast and baseline; the returned slope must
    recover the positive baseline effect and the r2 (baseline's incremental
    pseudo-R^2 over a contrast-only model) must exceed the r2 obtained when
    baseline is pure noise independent of the outcome.
    """
    rng = np.random.default_rng(0)
    n = 3000
    contrast = rng.choice([0.0, 0.25, 1.0], size=n)
    baseline = rng.normal(size=n)
    logit_p = -1.0 + 4.0 * contrast + 1.5 * baseline
    correct = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit_p))).astype(float)

    res = logistic_regression(correct, contrast, baseline)

    noise = rng.normal(size=n)
    null = logistic_regression(correct, contrast, noise)

    assert res['slope'] > 0
    assert res['r2'] > null['r2']


def test_linear_regression_controls_for_contrast():
    """Baseline slope is estimated after C(contrast) absorbs its RT offset.

    log RT depends on both contrast and baseline; the returned slope must
    recover the positive baseline effect and the r2 (baseline's incremental
    R^2 over a contrast-only model) must exceed the r2 obtained when baseline
    is pure noise. The contrast effect is large so a spurious baseline signal
    would appear if contrast were not controlled.
    """
    rng = np.random.default_rng(1)
    n = 3000
    contrast = rng.choice([0.0, 0.25, 1.0], size=n)
    baseline = rng.normal(size=n)
    log_rt = 5.0 * contrast + 0.5 * baseline + rng.normal(scale=0.1, size=n)

    res = linear_regression(log_rt, contrast, baseline)

    noise = rng.normal(size=n)
    null = linear_regression(log_rt, contrast, noise)

    assert res['slope'] > 0
    assert res['r2'] > null['r2']
