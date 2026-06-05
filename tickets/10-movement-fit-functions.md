# Ticket: Movement-encoding fit functions (replace per-contrast)

Spec: lmm-framework-revisions
Status: Done
Blocked by: 01, 02, 03, 04, 09

## Goal
The three movement claims (as LMMs) and the within-contrast variation check exist
as pure functions returning tidy result frames; `fit_movement_lmm_per_contrast`
is removed.

## Touches
- `iblnm/analysis.py` — add fit functions; remove `fit_movement_lmm_per_contrast`
  (~2259) and (under ticket 11) its plot.
- `tests/test_analysis.py` — add tests; remove `fit_movement_lmm_per_contrast`
  tests.

## Approach
Reuse `_fit_lmm`, `get_contrast_coding('log2')`, existing deviation coding, and
the `LMMResult.summary_df` tidy pattern. Three models + one descriptive check,
each returning a tidy frame (identifiers + coef, SE, z, p, CI, marginal R²):
- **movement vs contrast** — `timing ~ contrast`, `(1 + contrast | subject)`, per
  `(target_NM, timing_var)`; marginal (no side/reward).
- **movement predicts response (unadjusted)** — `response ~ timing`,
  `(1 + timing | subject)`, per `(target_NM, event, timing_var)`.
- **movement predicts response within contrast** — `response ~ C(contrast) +
  timing + side + reward`, `(1 + timing | subject)`, per
  `(target_NM, event, timing_var)`; report the timing slope + marginal R². No
  `C(contrast):timing` term.
- **within-contrast variation** — descriptive within- vs between-contrast variance
  of each timing variable per `(target_NM, timing_var)`; no fit.
Events: baseline, stimOn, firstMovement (feedback excluded).

## Acceptance
`pytest tests/test_analysis.py`: on synthetic data, each fit function returns the
expected tidy columns; the within-contrast model recovers a known timing slope
sign with `C(contrast)` present; the variation check returns finite within/between
variances. `fit_movement_lmm_per_contrast` and its tests are gone.
