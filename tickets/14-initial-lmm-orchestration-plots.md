# Ticket: Initial-LMM orchestration, CSVs, and labelled plots

Spec: lmm-framework-revisions
Status: Done
Blocked by: 12, 13

## Goal
The responses pipeline runs the full initial-LMM suite, saves every result as a
CSV with a consistent schema, and produces labelled summary figures annotated
with the model formula.

## Touches
- `scripts/responses.py` — replace the single-model LMM block + `plot_lmm_figures`
  with the suite orchestration; move all CSV saving out of `plot_*` (#10).
- `iblnm/vis.py` — plots for ceiling R², the three main-effect estimates, and
  LOSO-CV generalizability; reuse the existing `plot_lmm_summary` style for the
  EMM/interaction/CRF figures (unchanged).

## Approach
- Orchestrate ceiling + three main-effect models (ticket 12) + LOSO-CV
  (ticket 13) + the existing EMM/interaction/CRF reporting per (target, event).
- **Consistent CSV (#5):** every result (ceiling R², main-effect coefficients,
  LOSO-CV) saved as a CSV with a consistent tidy schema (identifiers + the
  reported quantities); reuse the `lmm_coefficients` concat pattern.
- **Labels (#7):** all plots labelled (axes, units, target/event); a single-model
  summary figure is annotated with that model's formula.
- Saving lives in the orchestration, plot functions only plot (#10).

## Acceptance
Automated: a `pytest` test that the orchestration, given a synthetic group,
writes the expected CSVs (ceiling, main-effects, LOSO-CV) with consistent tidy
columns and non-empty rows. Manual (not a gate): summary figures render, are
labelled, and show the model formula.
