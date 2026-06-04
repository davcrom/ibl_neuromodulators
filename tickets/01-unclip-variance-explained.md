# Ticket: Unclip and re-document `_variance_explained`

Spec: lmm-framework-revisions
Status: Done
Blocked by: none

## Goal
`_variance_explained` returns the raw marginal/conditional R² ratios (no [0,1]
clip), and its docstring describes the metric truthfully.

## Touches
- `iblnm/analysis.py` — `_variance_explained` (~line 1509); `fit_movement_lmm_r2`
  docstring (~line 2079).
- `tests/test_analysis.py` — `test_variance_explained_keys` (~lines 888–897).

## Approach
- Remove both `np.clip(..., 0, 1)` calls; return `float(var_fixed / var_y)` and
  `float((var_fixed + var_random) / var_y)`. Keep the `var_y == 0` early return.
- Rewrite the docstring: it is a data-based variance partition with a fixed
  empirical denominator `var(observed y)` (shared across nested models → ΔR² is a
  clean unique/semipartial R²), **not** Nakagawa & Schielzeth; values can fall
  outside [0,1] and that signals misfit.
- In `fit_movement_lmm_r2` docstring, drop the "≥ 0 in-sample" claim; note ΔR²
  can be negative.

## Acceptance
`pytest tests/test_analysis.py::...::test_variance_explained_keys` passes with the
two `0 <= ve[...] <= 1` range assertions replaced by `np.isfinite(...)` checks;
the existing `conditional >= marginal` assertion is kept.
