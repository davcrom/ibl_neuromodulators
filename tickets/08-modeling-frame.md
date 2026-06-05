# Ticket: Single `_modeling_frame()`; rewire all callers (fixes plot/model mismatch)

Spec: lmm-framework-revisions
Status: Done
Blocked by: none

## Goal
One canonical trial-selection method on the group, used by every model and plot,
so plots and models use identical trials (the plotting paths gain the missing
`response_time > 0.05` filter).

## Touches
- `iblnm/data.py` — add `_modeling_frame(response_col='response')` to
  `PhotometrySessionGroup`; `fit_lmm` (~2697) and `anova_response_magnitudes`
  (~2805) call it.
- `scripts/responses.py` — `plot_response_figures` (~94), `plot_lmm_figures`
  (~143), `build_movement_df` (~281) call it.
- `tests/test_data.py` — new test.

## Approach
`_modeling_frame` merges `response_magnitudes` with `trial_regressors` on
`(eid, trial)`, applies `add_relative_contrast`, then filters
`probabilityLeft == 0.5`, drops NaN `response_col`, `choice != 0`,
`response_time > 0.05`; returns the frame. Replace the five inline copies. Honors
the "all filtering flows through `PhotometrySessionGroup`" rule.
`build_movement_df` keeps its movement-specific work (event set, `log_<var>`) on
top of the shared frame.

## Acceptance
`pytest tests/test_data.py`: `_modeling_frame()` on a synthetic group excludes a
trial with `response_time <= 0.05`, a `choice == 0` trial, and a
`probabilityLeft != 0.5` trial; returned columns include `relative_contrast`/
`contrast`/`side`. The two former plotting paths now drop `response_time <= 0.05`
trials (same row count as the modeling frame).
