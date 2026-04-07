# Movement Encoding Script

## Goal

Determine whether trial timing variables (reaction time, movement time, and
later peak wheel velocity) explain NM responses at stimulus onset, and how
that explanatory power compares to task structure (contrast, side, feedback).

Three complementary analyses:

1. **LOSO cross-validated model comparison** — across all contrasts, compare
   the unique out-of-sample variance explained by contrast vs. timing, with
   a per-subject distribution of delta-R².
2. **Per-contrast slope analysis** — at each contrast level, estimate the
   timing-response slope and compare across targets.
3. **Descriptive plots** — NM response vs. timing at each contrast level.

## Inputs

- `data/responses/responses.pqt` — trial-level response magnitudes
  (columns: eid, subject, target_NM, event, trial, contrast, signed_contrast,
  stim_side, feedbackType, choice, probabilityLeft, hemisphere, response_early)
- `data/trial_timing.pqt` — per-trial timing decomposition
  (columns: eid, trial, reaction_time, movement_time, response_time)
- `data/sessions.pqt` + error logs — for session filtering via
  `PhotometrySessionGroup`

## Timing Variables

- `reaction_time` — stimulus onset to first wheel movement (decision latency)
- `movement_time` — first wheel movement to threshold crossing (motor execution)
- (later) `peak_velocity` — maximum absolute wheel velocity during trial

`response_time` is excluded (it's the sum of reaction_time and movement_time
and would be collinear with both).

All timing variables are log10-transformed before entering the models.

## Data Preparation

1. Load sessions and create `PhotometrySessionGroup` with standard filters
   (same as `responses.py`: session types, QC blockers, target NMs).
2. Load pre-existing response magnitudes from parquet.
3. Filter to `event == 'stimOn_times'` (stimulus-onset-aligned responses only).
4. Merge with trial timing on `(eid, trial)`.
5. Apply standard filters: `probabilityLeft == 0.5`, `choice != 0`,
   `response_time > 0.05`.
6. Add relative contrast via `add_relative_contrast()`.
7. Log10-transform timing variables after excluding non-positive values.

## Analysis 1: LOSO Cross-Validated Model Comparison

For each (target_NM, timing_variable), perform leave-one-subject-out
cross-validation to compare three nested LMMs:

- **Full:** `response ~ contrast + side + reward + log_timing`
- **Drop-contrast:** `response ~ side + reward + log_timing`
- **Drop-timing:** `response ~ contrast + side + reward`

Contrast is rank-coded and mean-centered. Side and reward use deviation
coding (±0.5). All terms are additive — no interactions.

### Procedure

For each fold (held-out subject):

1. Fit the three LMMs on the training set (all other subjects) using ML,
   with `(1 | subject)` as the random effect.
2. Predict the held-out subject's trials using fixed effects only (random
   effects set to zero — the held-out subject has no estimated intercept).
3. Compute R² on the held-out trials:
   `R² = 1 - SS_res / SS_tot` where SS_tot uses the held-out trial mean.
   This can be negative if the model predicts worse than the mean.
4. Compute delta-R² for each comparison:
   - `delta_R²_contrast = R²_full - R²_drop_contrast`
   - `delta_R²_timing = R²_full - R²_drop_timing`

Both models being compared use fixed-effects-only prediction on the same
held-out trials, so the missing random intercept cancels out in the delta.

### Outputs per (target_NM, timing_variable)

- Per-subject: R²_full, R²_drop_contrast, R²_drop_timing,
  delta_R²_contrast, delta_R²_timing, n_trials, subject
- Group-level: mean and SEM of each delta-R² across subjects,
  one-sample t-test or Wilcoxon signed-rank test on delta-R² > 0

### Subject counts per target (current data)

| Target | Subjects | Sessions |
|--------|----------|----------|
| VTA-DA | 7 | 210 |
| SNc-DA | 7 | 119 |
| DR-5HT | 11 | 244 |
| LC-NE | 10 | 163 |
| NBM-ACh | 6 | 72 |

With 6-11 subjects per target, LOSO gives 6-11 folds. Each training set
uses 5-10 subjects. This is adequate — the LMMs are estimated from hundreds
to thousands of trials per training fold.

### Summary plot

Dot-and-whisker plot: one row per target_NM, two panels (one per timing
variable). Each dot is one subject's delta-R². Show mean ± SEM as a
horizontal bar. Mark whether the group-level test is significant.

This replaces the single-bar delta-R² plot from the previous spec.

### Why LOSO-CV instead of in-sample R²

- In-sample R² can only increase with more predictors — it cannot reveal
  whether the timing predictor genuinely helps or is fitting noise.
- LOSO-CV delta-R² can be negative for individual subjects, which is
  informative: it means the timing predictor hurts prediction for that
  subject.
- The per-subject distribution enables proper group-level inference
  (N = number of subjects, not number of trials).
- The delta-R² between two models cancels the missing random intercept
  for the held-out subject, so fixed-effects-only prediction is fair.

## Analysis 2: Per-Contrast Slope Analysis

For each (target_NM, contrast_level, timing_variable), fit:

`response ~ side + reward + log_timing + (1 | subject)`

### Outputs per (target_NM, contrast, timing_variable)

- Coefficient (slope) of log_timing ± SE
- p-value for the timing coefficient
- R² marginal

### NBM-ACh interaction test

For NBM-ACh only, at each contrast level, additionally fit:

`response ~ log_timing * side * reward + (1 | subject)`

Compare to the additive model via LRT to test whether the timing-response
relationship differs by side and feedback. This addresses the three-way
interaction visible in the descriptive plots.

### Summary plot

Timing slope (± 95% CI) as a function of contrast, one line per target_NM.
Significant slopes marked. Separate panel per timing variable.

## Descriptive Plots (already implemented)

One figure per (target_NM, contrast, timing_variable):
- Two panels (contra/ipsi), correct/incorrect lines
- Quantile-binned log-transformed timing on x-axis
- NM response on y-axis with subject-mean removal

## Implementation Plan

### Step 1: `loso_cv_movement_lmm` in `analysis.py`

Pure function. Takes a DataFrame (single target_NM, all contrasts) and a
timing column name. Performs LOSO-CV across subjects.

For each fold:
1. Split data into train (N-1 subjects) and test (1 subject).
2. Code predictors (rank contrast, deviation-coded side/reward).
3. Fit three LMMs on training data via `_fit_lmm` with `reml=False`.
4. Build design matrices for the test data using patsy with the training
   model's `design_info`.
5. Predict test trials: `X_test @ fe_params` (fixed effects only).
6. Compute R² on test trials for each model.
7. Compute delta-R² values.

Returns a DataFrame with one row per subject: subject, n_trials,
r2_full, r2_drop_contrast, r2_drop_timing, delta_r2_contrast,
delta_r2_timing, timing_col.

Replaces `fit_movement_lmm`. The old function can be removed.

### Step 2: `fit_movement_lmm_per_contrast` in `analysis.py`

Already implemented. No changes needed.

### Step 3: `plot_movement_lmm_summary` in `vis.py`

Replace the current grouped-bar implementation with a dot-and-whisker
plot showing per-subject delta-R² distributions. One panel per timing
variable. Targets ordered by `TARGETNM2POSITION`.

### Step 4: `plot_movement_slopes` in `vis.py`

Already implemented. No changes needed.

### Step 5: Wire up in `scripts/movement_encoding.py`

Replace the `fit_movement_lmm` call with `loso_cv_movement_lmm`. Collect
per-subject results into a DataFrame, save to CSV, pass to the updated
plot function.

### Testing

`test_loso_cv_movement_lmm` in `test_analysis.py`:
- Synthetic data with known contrast and timing effects.
- Verify returns DataFrame with expected columns.
- Verify one row per subject.
- Verify delta-R² values are positive on average (both predictors have
  real effects in the synthetic data).
- Verify returns empty DataFrame when < 2 subjects.

Updated `TestPlotMovementLMMSummary` in `test_vis.py`:
- Adapt fixture to per-subject format.
- Verify figure returned, correct number of panels.
