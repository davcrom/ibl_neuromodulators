# Ticket: Schema migration — response column rename + drop trial_timing/peak_velocity

Spec: movement_encoding_in_responses
Status: Done
Blocked by: 01

## Goal

Atomic migration of the per-trial schema: `response_early` → `response` everywhere; `self.trial_timing` and `self.peak_velocity` removed; `trial_timing.pqt` retired; `trial_regressors.pqt` is the new persisted file. After this ticket, `pytest` passes and `scripts/responses.py` writes/reads `trial_regressors.pqt`.

The new movement-encoding analyses are not added here — that is ticket 04. This ticket only relocates and renames.

## Touches

- `iblnm/config.py`:
  - Remove `TRIAL_TIMING_FPATH`, `PEAK_VELOCITY_FPATH`, `MOVEMENT_ENCODING_DIR`.
  - Add `TRIAL_REGRESSORS_FPATH = RESPONSES_DIR / 'trial_regressors.pqt'`.
- `iblnm/data.py`:
  - `get_response_magnitudes()`: stop populating `self.trial_timing`; drop the trial-level regressor columns from each row; rename written column `response_early` → `response`. Method becomes "response + recording keys only".
  - Remove `self.trial_timing`, `self.peak_velocity`, `load_trial_timing`, `load_peak_velocity`, `enrich_peak_velocity`, `get_trial_timings` (the per-session helper at `data.py:1171`).
  - Update `fit_lmm`, `anova_response_magnitudes`, `fit_wheel_lmm` (the PSG method, not the analysis function) so they merge from `self.trial_regressors` and use `'response'` instead of `'response_early'`. The `peak_velocity` column now comes from `self.trial_regressors` rather than `self.peak_velocity`.
  - Update default `response_col='response_early'` arguments to `'response'`.
- `iblnm/analysis.py`:
  - `fit_response_lmm`: replace the hardcoded `'response_early'` column references (`dropna`, `y` extraction at lines ~892, ~896) with the function's `response_col` parameter.
  - `fit_wheel_lmm` (the analysis function): default `response_col='response_early'` → `'response'`.
- `iblnm/vis.py`:
  - Update the helper around line 3753 that consumes `group.trial_timing` to consume `group.trial_regressors` (merge keys unchanged).
  - Update docstring `response_early` mentions.
- `scripts/responses.py`:
  - Remove `from iblnm.config import TRIAL_TIMING_FPATH`; add `TRIAL_REGRESSORS_FPATH`.
  - In full mode: call `group.get_trial_regressors()`, write `group.trial_regressors` to `TRIAL_REGRESSORS_FPATH` (in place of the current `group.trial_timing` write).
  - In `--plot` mode: replace `load_trial_timing(TRIAL_TIMING_FPATH)` with `load_trial_regressors(TRIAL_REGRESSORS_FPATH)`; error message updated.
  - Change `response_col='response_early'` defaults in `plot_response_figures` and `plot_lmm_figures` to `'response'`. Update local references.
- `scripts/task.py`:
  - Remove `TRIAL_TIMING_FPATH` import.
  - In `process_task`: stop calling `ps.get_trial_timings()` and stop returning `'timing'`. Return shape becomes `{'performance': result}` directly, or just the performance dict.
  - In `__main__`: remove `timing_frames` collection, `df_trial_timing` construction, the merge-with-existing block, and the write.
- `scripts/task_encoding.py`:
  - Replace `group.load_trial_timing(TRIAL_TIMING_FPATH)` with `group.load_trial_regressors(TRIAL_REGRESSORS_FPATH)` (and update the import).
- `scripts/task_performance.py`:
  - Same swap (`load_trial_timing` → `load_trial_regressors`; `group.trial_timing` → `group.trial_regressors`; error message text). Lines 67, 92, 99.
- `tests/`:
  - Any test that asserts `'response_early'` or `'trial_timing'` updated to `'response'` / `'trial_regressors'`. Run `pytest` and fix breakages.

## Scope note (approved extension)

`get_glm_response_features` (`data.py:2174`, used by `scripts/task_encoding.py`,
covered by `tests/test_data.py::TestGetGLMResponseFeatures`) also consumed the
trial-regressor columns this migration removes from `response_magnitudes`
(`choice`, `contrast`, `signed_contrast`, `stim_side` → `fit_response_glm`) and
read `response_time` from `self.trial_timing`. Not in the original Touches list.
User approved extending the migration: update it to merge `self.trial_regressors`
(predictor columns + `response_time`) into its `events` frame, same pattern as
`fit_lmm`/`anova_response_magnitudes`/`fit_wheel_lmm`, and update its tests.

## Approach

- Single sweep: search for every occurrence of `response_early`, `trial_timing`, `peak_velocity` (as attribute, file path, or method name) in `iblnm/` and `scripts/` and update.
- The new `get_trial_regressors` method from ticket 01 is the canonical writer for `self.trial_regressors`. `responses.py` calls it; downstream methods read from it.
- Keep the existing call sequence in `scripts/responses.py` (load traces → compute response magnitudes → compute mean traces → build response features). Insert `group.get_trial_regressors()` and the parquet write immediately after `group.response_magnitudes.to_parquet(...)`.
- `responses.py` plot-mode block: add `group.load_trial_regressors(TRIAL_REGRESSORS_FPATH)` next to the existing load calls; remove `load_trial_timing`.
- Do not introduce backward-compat aliases — per project rule, rename in place.

## Acceptance

1. `pytest` passes (full suite).
2. `ruff check iblnm scripts tests` passes.
3. `grep -rn "response_early\|trial_timing\|peak_velocity" iblnm/ scripts/ tests/` returns no hits except inside H5 group names (none expected) or the now-deleted scripts (handled by ticket 05).
4. `python scripts/responses.py --plot` runs against an existing `data/responses/trial_regressors.pqt` (generated by a prior full run) without error, producing the same plots as before the migration except for the column rename.
