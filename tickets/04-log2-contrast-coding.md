# Ticket: `log2` contrast coding, used consistently; fix argparse mismatch

Spec: lmm-framework-revisions
Status: Done
Blocked by: none

## Goal
A dataset-independent `log2` contrast coding exists and is the default continuous
coding for the responses analysis; the argparse help/default no longer disagree.

## Touches
- `iblnm/util.py` — `get_contrast_coding` (~line 887; rank closure ~910).
- `iblnm/analysis.py` — `_code_movement_predictors` (~line 2063).
- `scripts/responses.py` — `--contrast-coding` argparse (~lines 389–391).
- `tests/test_util.py` (or `test_analysis.py`) — new test.

## Approach
- Add a `'log2'` branch to `get_contrast_coding` returning `(transform, inverse)`
  where `transform(c) = 0 if c == 0 else log2(c)` (vectorized; contrast in
  percent). Inverse: `2**x` for nonzero, 0 maps back to 0. **Guard:** assert
  nonzero contrasts are ≥ 1 (percent units) and raise (fail loud) otherwise — a
  fraction-unit input would collide 100% (=1.0) with the 0→0 clamp.
- `_code_movement_predictors`: switch its hardcoded `'rank'` to `'log2'`.
- `responses.py`: set `--contrast-coding default='log2'` and fix the help text
  (currently says `(default: log)`). Add `'log2'` to `choices`.

## Approach — verified values (2026-06-04)
For the standard percent set: `0→0, 6.25→2.644, 12.5→3.644, 25→4.644, 100→6.644`
(natural-set gaps `[2.644, 1, 1, 2]`). Confirmed `contrast` is stored in percent
(`compute_trial_contrasts`, `task.py:290`, `* 100`).

## Acceptance
`pytest`: `get_contrast_coding('log2')[0]([0, 6.25, 12.5, 25, 100])` returns the
values above (atol 1e-3); passing a fractional contrast (e.g. `[0, 0.0625, 1.0]`)
raises; `argparse` default parses to `'log2'`.
