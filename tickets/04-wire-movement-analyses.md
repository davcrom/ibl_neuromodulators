# Ticket: Wire movement analyses into responses.py

Spec: movement_encoding_in_responses
Status: Done
Blocked by: 02, 03

## Goal

Add three movement-variable analyses to `scripts/responses.py`, fed by the merged `response_magnitudes` × `trial_regressors` DataFrame, producing CSVs and SVGs under `results/responses/` and `figures/responses/movement/`.

## Touches

- `scripts/responses.py`: add a `# Movement encoding` section after the existing LMM section. Calls the existing `loso_cv_movement_lmm`, `fit_movement_lmm_per_contrast` (now with random slopes — ticket 03), `plot_movement_response`, `plot_movement_lmm_summary`, `plot_movement_slopes`.
- `iblnm/config.py`: no new constants required (`RESPONSES_DIR` already exists; figure subdirs are created inline as `responses.py` already does for `fig_dirs`).

## Approach

Reuse the structure that `movement_encoding.py` currently uses (descriptive → LOSO → per-contrast slopes), but driven off `group.response_magnitudes` and `group.trial_regressors` and committed inside `responses.py`.

Concrete additions in `scripts/responses.py`:

1. Output directories: extend `fig_dirs` with
   `'movement_descriptive': fig_base / 'movement/descriptive'`,
   `'movement_model_comparison': fig_base / 'movement/model_comparison'`,
   `'movement_slopes': fig_base / 'movement/slopes'`.
2. Build the modeling DataFrame (once, reused across the three blocks):
   ```python
   TIMING_VARS = ['reaction_time', 'movement_time', 'peak_velocity']
   df_resp = group.response_magnitudes.query(
       "event == 'stimOn_times' and probabilityLeft == 0.5 and choice != 0"
   ).copy()
   df_resp = add_relative_contrast(df_resp)
   df_resp = df_resp.merge(
       group.trial_regressors[['eid', 'trial'] + TIMING_VARS + ['response_time']],
       on=['eid', 'trial'], how='left',
   )
   df_resp = df_resp.query('response_time > 0.05').dropna(subset=['response'])
   for var in TIMING_VARS:
       df_resp[f'log_{var}'] = np.where(df_resp[var] > 0,
                                        np.log10(df_resp[var]), np.nan)
   ```
3. Descriptive plots block — port the `movement_encoding.py` loop, swapping `response_col='response'` and saving to `fig_dirs['movement_descriptive']`. Filename: `f'{target_nm}_stimOn_{var}_c{contrast:g}.svg'`.
4. LOSO-CV ΔR² block — per `(target_nm, var)`, call `loso_cv_movement_lmm(df_valid, 'response', f'log_{var}')`. Concatenate into a single DataFrame; write to `RESPONSES_DIR / 'loso_cv_model_comparison.csv'`. Render `plot_movement_lmm_summary` to `fig_dirs['movement_model_comparison'] / 'model_comparison.svg'`.
5. Per-contrast slopes block — per `(target_nm, contrast, var)`, call `fit_movement_lmm_per_contrast(df_valid, 'response', f'log_{var}')`. Collect results into a DataFrame; write to `RESPONSES_DIR / 'per_contrast_slopes.csv'`. Render `plot_movement_slopes` to `fig_dirs['movement_slopes'] / 'timing_slopes_by_contrast.svg'`.

Skip thresholds: ≥ 2 subjects per group, ≥ 20 valid trials per (target_NM, contrast, predictor); ≥ 20 valid trials per (target_NM, predictor) for LOSO. Match `movement_encoding.py` exactly.

## Acceptance

1. End-to-end run: `python scripts/responses.py --plot` (against an existing run's parquets) produces the three new figure directories with at least one file each and the two new CSVs under `results/responses/`. Manual check — not a gate.
2. Automated check: extend `tests/test_vis.py` with a smoke test that constructs a synthetic merged DataFrame with two target_NMs, three contrasts, three subjects, 50 trials per cell, calls each of `plot_movement_response`, `plot_movement_lmm_summary`, `plot_movement_slopes` on the appropriate inputs, and asserts each returns a `matplotlib.figure.Figure` (no exception). Synthetic data construction lives in the test, not in source.
3. `pytest` passes.
