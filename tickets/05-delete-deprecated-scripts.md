# Ticket: Delete superseded scripts

Spec: movement_encoding_in_responses
Status: Done
Blocked by: 04

## Goal

Remove the two scripts whose analyses now live in `scripts/responses.py`.

## Touches

- Delete `scripts/movement_encoding.py`.
- Delete `scripts/contrast_coding_comparison.py`.
- `scripts/run_pipeline.py` (if it references either): remove the corresponding stage.

## Approach

- `grep -rn 'movement_encoding\|contrast_coding_comparison' .` before deletion to confirm no remaining import or invocation outside the files themselves.
- Both scripts are top-level entry points (no library imports of their symbols from elsewhere — verified during /software-eng survey: they're not imported anywhere in `iblnm/` or `scripts/`).

## Acceptance

1. Files deleted.
2. `grep -rn 'movement_encoding\|contrast_coding_comparison' iblnm/ scripts/ tests/` returns no hits.
3. `pytest` passes.
4. `python scripts/responses.py --plot` still runs and produces the movement-analysis outputs from ticket 04.
