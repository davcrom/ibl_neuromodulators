# Movement Encoding Script

## Goal

Determine whether trial timing variables (reaction time, movement time, and
later peak wheel velocity) explain NM responses at stimulus onset, and how
that explanatory power compares to task structure (contrast, side, feedback).

Two complementary analyses:

1. **Pooled model comparison** — across all contrasts, compare the unique
   variance explained by contrast vs. timing.
2. **Per-contrast slope analysis** — at each contrast level, estimate the
   timing-response slope and compare across targets.

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

## Analysis 1: Pooled Model Comparison

For each (target_NM, timing_variable), fit three nested LMMs using ML
(not REML) so likelihood ratio tests are valid:

- **Full:** `response ~ contrast + side + reward + log_timing + (1 | subject)`
- **Drop-contrast:** `response ~ side + reward + log_timing + (1 | subject)`
- **Drop-timing:** `response ~ contrast + side + reward + (1 | subject)`

Contrast is rank-coded (matching `responses.py`). Side and reward use
deviation coding (±0.5). All terms are additive — no interactions.

### Outputs per (target_NM, timing_variable)

- R² marginal for all three models
- delta_R²_contrast = R²(full) − R²(drop-contrast)
- delta_R²_timing = R²(full) − R²(drop-timing)
- LRT chi² and p-value for each comparison (1 df each)
- BIC for all three models → BIC-based Bayes factor:
  BF_contrast = exp((BIC_drop_contrast − BIC_full) / 2)
  BF_timing = exp((BIC_drop_timing − BIC_full) / 2)
  (BF > 1 favors the full model; BF > 10 is strong evidence)

### Summary plot

Grouped bar chart or dot plot: one group per target_NM, comparing
delta_R²_contrast vs delta_R²_timing for each timing variable. Significance
from LRT marked with stars. Log10(BF) shown as secondary annotation or
separate panel.

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

## Changes Already Made to responses.py

- Removed: `TRIAL_TIMING_FPATH`, `PEAK_VELOCITY_FPATH`,
  `plot_wheel_lmm_summary`, `plot_wheel_lmm_figures`
- Removed trial_timing merge/filter from plot functions
- Removed trial_timing/peak_velocity loading/saving from main block
- Removed commented-out wheel LMM block and `'wheel_lmm'` fig dir

## Implementation Plan

### Step 1: `fit_movement_lmm` in `analysis.py`

Pure function. Takes a DataFrame (single target_NM, all contrasts) and a
timing column name. Fits the three nested models. Returns a dict with R²
values, delta-R², LRT stats, and BIC-based Bayes factors.

Reuses `_fit_lmm` for model fitting and `_variance_explained` via the
LMMResult it returns.

### Step 2: `fit_movement_lmm_per_contrast` in `analysis.py`

Pure function. Takes a DataFrame (single target_NM, single contrast) and a
timing column name. Fits the per-contrast model. Returns a dict with the
timing slope, SE, p-value, and R².

### Step 3: `plot_movement_lmm_summary` in `vis.py`

Summary plot for Analysis 1 (pooled comparison).

### Step 4: `plot_movement_slopes` in `vis.py`

Summary plot for Analysis 2 (per-contrast slopes).

### Step 5: Wire up in `scripts/movement_encoding.py`

Orchestrate: load data, iterate over groups, call analysis functions, save
results to CSV, generate plots.

### Testing

Each analysis function gets tests with synthetic data in `test_analysis.py`.
Plot functions get tests in `test_vis.py`. Follow existing patterns (synthetic
fixtures, no Alyx calls).
