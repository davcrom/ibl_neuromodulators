# Ticket: Random slope by subject for movement LMMs

Spec: movement_encoding_in_responses
Status: Done
Blocked by: 02

## Goal

Update `fit_movement_lmm_per_contrast` and `loso_cv_movement_lmm` so subject-level random effects include the timing predictor's random slope, not just a random intercept.

## Touches

- `iblnm/analysis.py`:
  - `fit_movement_lmm_per_contrast` (around `analysis.py:2163`): change `re_formula='1'` to `re_formula=f'1 + {timing_col}'` (or equivalent) so the fitted model is `response ~ side + reward + timing + (1 + timing | subject)`. Fixed-effects formula unchanged.
  - `loso_cv_movement_lmm` (`analysis.py:2041`): same update for the Full and Drop-contrast models (both contain `timing_col`). The Drop-timing model keeps `re_formula='1'` (no timing in the formula).
- `tests/test_analysis.py`: add or extend a test that fits the per-contrast LMM on synthetic data with known subject-level slope variability and asserts the random-effects covariance has 2 rows × 2 cols (intercept + slope).

## Approach

- `_fit_lmm` (`analysis.py:1540`) already accepts `re_formula` — pass `f'1 + {timing_col}'` for models that include the timing term.
- statsmodels' `MixedLM` random slopes are parameterised via `re_formula`; the existing helper handles convergence (`None` return on failure). No new convergence handling needed; callers in `responses.py` already skip `None`.
- Update docstrings of both functions to state the random-effects structure.

## Acceptance

1. `tests/test_analysis.py` test that generates synthetic trial data:
   - 6 subjects, 100 trials each, single contrast level, with subject-specific intercepts and subject-specific timing slopes (draw subject slopes from a distribution).
   - Fit with `fit_movement_lmm_per_contrast`.
   - Assert `result['n_subjects'] == 6`, `result['slope']` is finite, and that the LMM result's random-effects covariance is 2 × 2.
2. `pytest tests/test_analysis.py` passes.
3. The existing LOSO test (if any) still passes; if it asserts a specific random-effects structure, update it.
