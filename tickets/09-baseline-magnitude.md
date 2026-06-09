# Ticket: Pre-stimulus baseline response magnitude

Spec: lmm-framework-revisions
Status: Done
Blocked by: none

## Goal
`response_magnitudes` includes a per-trial `event == 'baseline'` row: the mean of
the **raw** (pre-subtraction) signal in `BASELINE_WINDOW (-0.2, 0)` relative to
stimOn.

## Touches
- `iblnm/data.py` — `load_response_traces` (~1809; `subtract_baseline` at ~1848),
  `get_response_magnitudes` (~1884).
- `tests/test_data.py` — new test.

## Approach
Cached traces are baseline-subtracted (`subtract_baseline`, line 1848), so a
baseline-window magnitude from them is ≈ 0. Capture, **before** subtraction, the
per-trial mean of the raw stimOn-aligned trace over `BASELINE_WINDOW` — the same
quantity `subtract_baseline` computes for the stimOn event — and surface it via
`get_response_magnitudes` as a `baseline` pseudo-event row (one value per trial,
same columns as other magnitude rows with `event='baseline'`). Reuse
`compute_response_magnitude(..., BASELINE_WINDOW)` on the raw trace.

## Acceptance
`pytest tests/test_data.py`: from a synthetic session with a known non-zero
pre-stimulus level, `get_response_magnitudes()` yields `event == 'baseline'` rows
whose values equal the raw baseline-window mean (not ≈ 0), one per trial.
