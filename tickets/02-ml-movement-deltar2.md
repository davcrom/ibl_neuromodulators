# Ticket: ML (not REML) for the nested movement ΔR² comparison

Spec: lmm-framework-revisions
Status: Done
Blocked by: 01

## Goal
`fit_movement_lmm_r2` fits its three nested models with ML so their marginal R²
are on a comparable scale.

## Touches
- `iblnm/analysis.py` — `fit_movement_lmm_r2` (the three `_fit_lmm` calls,
  ~line 2117).

## Approach
- Change `reml=True` → `reml=False` in the three `_fit_lmm` calls. No other
  change. This matches `loso_cv_movement_lmm` and `fit_wheel_lmm`, which already
  use ML. Rationale: REML estimates are not comparable across models differing in
  fixed effects.

## Acceptance
`pytest tests/test_analysis.py` — the existing delta-positivity checks on
synthetic data with a true effect (`test_*` around lines 2522–2546, 2574–2581,
2645–2656) still pass under ML. No new test required.
