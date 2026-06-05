# Ticket: Generalize LOSO-CV to the task model; wire into `fit_lmm`

Spec: lmm-framework-revisions
Status: Done
Blocked by: 12

## Goal
A leave-one-subject-out cross-validation compares the full task model vs a
no-interaction model out-of-sample, per (target_NM, event), to test whether the
interaction structure generalizes across animals.

## Touches
- `iblnm/analysis.py` — generalize `loso_cv_movement_lmm` (~2178) so it serves
  the task model (configurable full vs reduced formula), or add a thin
  task-model wrapper reusing its held-out-subject-centering logic.
- `iblnm/data.py` — `fit_lmm` calls it.
- `tests/test_analysis.py`.

## Approach
Reuse the existing centering logic (subtract the held-out subject's mean from
both response and fixed-effects prediction; `_centered_r2`). Full model =
`response ~ contrast * side * reward`; reduced = `response ~ contrast + side +
reward` (no interactions). Both `(1 | subject)`, ML. Report per held-out subject:
out-of-sample R² (full, reduced) and ΔR²; plus an aggregate. Note (in code
comment) that at 6–11 subjects the estimate is noisy/qualitative.

## Acceptance
`pytest tests/test_analysis.py`: on synthetic data with ≥3 subjects, the routine
returns one row per held-out subject with `r2_full`, `r2_reduced`, `delta_r2`,
`subject`, `n_trials`, and a finite aggregate; returns an empty frame below the
minimum subject count.
