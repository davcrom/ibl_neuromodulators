# Ticket: Delete dead/commented blocks in `responses.py`

Spec: lmm-framework-revisions
Status: Done
Blocked by: none

## Goal
The commented-out and empty blocks are gone from `responses.py`.

## Touches
- `scripts/responses.py`.

## Approach
Delete: the commented-out `plot_lmm_response` loop (~lines 151–164), the
commented similarity block (~lines 223–267), and the empty
`# Wheel kinematics LMM analysis` section header (~lines 184–185). Do **not**
touch `loso_cv_movement_lmm` — it is revived elsewhere.

## Acceptance
`ruff check scripts/responses.py` clean; `grep -n "plot_lmm_response\|# sim =\|
Wheel kinematics LMM" scripts/responses.py` returns nothing; `pytest` green;
`python scripts/responses.py --plot` import path still loads (no syntax break).
