# Ticket: Move movement constants to `config.py`

Spec: lmm-framework-revisions
Status: Done
Blocked by: none

## Goal
`TIMING_VARS`, `MIN_SUBJECTS_MOVEMENT`, `MIN_TRIALS_MOVEMENT` live in `config.py`,
not at the top of a script.

## Touches
- `iblnm/config.py` — add the three constants.
- `scripts/responses.py` — import them; delete the local definitions (~lines
  48–50).

## Approach
Project convention (CLAUDE.md): constants/thresholds live in `config.py`. Move
the values verbatim, import in `responses.py`. No behavior change.

## Acceptance
`python -c "from iblnm.config import TIMING_VARS, MIN_SUBJECTS_MOVEMENT,
MIN_TRIALS_MOVEMENT"` succeeds; `ruff check` clean; `responses.py` no longer
defines them locally. (`pytest` still green.)
