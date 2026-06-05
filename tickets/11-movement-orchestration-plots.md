# Ticket: Movement-encoding orchestration, CSVs, and plots

Spec: lmm-framework-revisions
Status: Done
Blocked by: 08, 10

## Goal
The responses pipeline runs the new movement claims off the shared modeling
frame, saves each result as a CSV, and produces labelled figures (including the
raw-data within-contrast check).

## Touches
- `scripts/responses.py` — `build_movement_df` (event set), the movement driver
  (replace `_movement_per_contrast_slopes`; keep `_movement_model_comparison`).
- `iblnm/vis.py` — remove `plot_movement_slopes` (~2929); add plots for the new
  claims and the raw-data within-contrast plot.

## Approach
- `build_movement_df`: call `_modeling_frame()` (ticket 08), retain `baseline`,
  `stimOn`, `firstMovement` events (stop filtering to stimOn), add `log_<var>`.
- Driver calls the ticket-10 fit functions per `(target_NM, event, timing_var)`,
  concatenates tidy frames, **saves CSVs** in the responses data dir, and saves
  figures. Saving lives in the driver, not in `plot_*` functions (#10).
- Plots: clearly labelled (axes, units, target/event). Within-contrast model gets
  the accompanying raw-data plot (response vs timing within contrast levels).

## Acceptance
Automated: a `pytest` test that the driver, given a synthetic group, writes the
expected CSV(s) with the tidy schema and non-empty rows; `build_movement_df`
returns all three events. Manual (not a gate): figures render and are labelled.
