# Ticket: Delete the orphaned `fit_wheel_lmm` stack

Spec: lmm-framework-revisions
Status: Done
Blocked by: none

## Goal
The unused wheel-LMM code (response → wheel kinematics) is removed; wheel
*velocity* extraction is untouched.

## Touches
- `iblnm/analysis.py` — `fit_wheel_lmm` (~line 2319).
- `iblnm/data.py` — `fit_wheel_lmm` method (~line 2886) and the
  `wheel_lmm_results` / `wheel_lmm_summary` attributes (init ~lines 1439–1440).
- `iblnm/vis.py` — `plot_wheel_lmm_summary` (~line 2997).
- tests — any wheel-LMM tests.

## Approach
Nothing runs `fit_wheel_lmm` (verified 2026-06-04: no caller in `scripts/`). It
is the reverse direction, superseded by the movement-encoding models. Delete the
function, method, attributes, plot, and tests. **Keep** `peak_velocity`
extraction in `trial_regressors` (`get_trial_regressors`, `_peak_velocity`) — it
is separate and still used.

## Acceptance
`grep -rn "fit_wheel_lmm\|wheel_lmm_results\|wheel_lmm_summary\|
plot_wheel_lmm_summary" iblnm/ scripts/ tests/` returns nothing; `pytest` green;
`peak_velocity` still present in `get_trial_regressors` output.
