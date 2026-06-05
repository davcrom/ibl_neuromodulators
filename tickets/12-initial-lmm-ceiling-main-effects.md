# Ticket: Initial-LMM suite — ceiling + three main-effect models

Spec: lmm-framework-revisions
Status: Done
Blocked by: 03, 04, 08

## Goal
`fit_lmm` produces, per (target_NM, event): the saturated-categorical ceiling R²
and three single-random-slope main-effect estimates, while the existing
EMM/interaction/slope reporting is preserved unchanged.

## Touches
- `iblnm/data.py` — `fit_lmm` (~2653).
- `iblnm/analysis.py` — ceiling fit (a `_fit_lmm` call with a categorical
  formula; no new machinery).
- `tests/test_data.py` / `tests/test_analysis.py`.

## Approach
- Use `_modeling_frame()` (ticket 08) for trial selection.
- **Ceiling:** `_fit_lmm('response ~ C(contrast) * side * reward', re='1')`;
  report its marginal (and conditional) R².
- **Three main-effect models:** `response ~ contrast * side * reward` fixed, each
  with one random slope — `(1 + contrast | subject)`, `(1 + side | subject)`,
  `(1 + reward | subject)`. Read each main effect only from its own model
  (coef, SE, z, p, CI, marginal R²). Continuous contrast uses `log2` (ticket 04).
- **Unchanged:** `compute_marginal_means`, the full interaction suite
  `compute_interaction_effects` (contrast×reward, contrast×side, reward×side),
  `compute_contrast_slopes`. Variable coding and effect reporting stay as they
  are now.
- Route all fits through `_fit_lmm` (single engine). Reuse the `summary_df` tidy
  pattern for the coefficient outputs.

## Acceptance
`pytest`: on a synthetic group, `fit_lmm` returns per (target, event) a ceiling R²
and three main-effect rows (one per predictor) with the tidy coefficient schema;
the existing EMM/interaction/slope outputs are still produced and unchanged in
shape.
