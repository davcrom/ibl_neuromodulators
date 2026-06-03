# Ticket: PSG get_trial_regressors method

Spec: movement_encoding_in_responses
Status: Done
Blocked by: none

## Goal

Add a `PhotometrySessionGroup.get_trial_regressors()` method that reads each session's H5 file directly and returns a DataFrame of every per-trial predictor used by downstream modeling. Plus the matching `load_trial_regressors(path)` and the `self.trial_regressors` attribute.

## Touches

- `iblnm/data.py`:
  - Add `self.trial_regressors = None` to `PhotometrySessionGroup.__init__` (alongside the existing `self.trial_timing = None` / `self.peak_velocity = None`; the latter two stay in place for now — removed in ticket 02).
  - Add method `get_trial_regressors(self) -> pd.DataFrame`.
  - Add method `load_trial_regressors(self, path)` that delegates to `self._load_parquet(path)` and assigns to `self.trial_regressors`.
- `tests/test_data.py`: add a test for `get_trial_regressors` using a synthetic H5 fixture.

## Approach

- Iterate the unique `eid`s in `self.recordings['eid']`. For each eid, open `SESSIONS_H5_DIR / f'{eid}.h5'` once.
  - From group `trials` read: `stimOn_times`, `firstMovement_times`, `feedback_times`, `signed_contrast`, `contrast`, `stim_side`, `choice`, `feedbackType`, `probabilityLeft`. The H5 trials group is written by `scripts/task.py` (via `ps.save_h5(groups=['trials'])`) — column set matches `ps.trials` after IBL extraction.
  - Compute `reaction_time = firstMovement_times - stimOn_times`, `movement_time = feedback_times - firstMovement_times`, `response_time = feedback_times - stimOn_times`. If any of those time columns is missing, the corresponding derived column is NaN for that session.
  - From group `wheel/responses/velocity` (shape `(n_trials, n_time)`, NaN-padded; written by `scripts/wheel.py`) compute `peak_velocity[t] = max(|velocity[t]|)` over finite samples; NaN if no finite samples or if the `wheel/responses` group is absent.
- Reuse the H5 reading pattern in the existing `enrich_peak_velocity` (`data.py:2960`) for the wheel section.
- Reuse `tqdm` progress bar over eids, matching the existing extraction methods.
- Schema returned (one row per `eid × trial`):
  `eid, trial, signed_contrast, contrast, stim_side, choice, feedbackType, probabilityLeft, reaction_time, movement_time, response_time, peak_velocity`.
- Assign result to `self.trial_regressors` and return it (matches the convention of `get_response_magnitudes` returning the populated frame).

## Acceptance

Unit test in `tests/test_data.py` that:

1. Writes a synthetic H5 file with a `trials` group containing 3 trials (with known values for all required columns) and a `wheel/responses/velocity` dataset with known shape (e.g. `[[0,1,-3], [nan,nan,nan], [2,-5,1]]`).
2. Constructs a minimal `PhotometrySessionGroup` whose `recordings` references that single eid.
3. Calls `group.get_trial_regressors()`.
4. Asserts the returned DataFrame has the exact column set listed above, length 3, and `peak_velocity == [3, NaN, 5]`, `reaction_time == firstMovement_times - stimOn_times` per trial.
5. Asserts a second test that omits the `wheel/responses` group → `peak_velocity` column is all NaN.

`pytest tests/test_data.py -k trial_regressors` passes.
