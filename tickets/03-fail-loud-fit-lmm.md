# Ticket: Fail-loud `_fit_lmm` + drop-visibility warning

Spec: lmm-framework-revisions
Status: Done
Blocked by: none

## Goal
`_fit_lmm` lets unexpected errors propagate, returns None only for genuine
numerical failures, and warns when it drops a fit.

## Touches
- `iblnm/analysis.py` — `_fit_lmm` (the blanket `except Exception` at ~lines
  1589–1590; the specific catch at ~1615).
- `tests/test_analysis.py` — new tests.

## Approach
- Remove `except Exception: return None`. Keep `except (np.linalg.LinAlgError,
  ValueError): return None` (both the fit block and the BLUP/variance block at
  ~1615) so singular/degenerate fits still return None; any other exception
  propagates.
- On every numerical-failure `return None`, `warnings.warn(...)` identifying the
  formula and grouping, so a shrunken result set is never silent.
- Leave the convergence string-match (`'failed to converge' in str(w.message)`)
  unchanged — out of scope.

## Acceptance
New `pytest` tests: (1) a formula referencing a missing column raises (propagates)
rather than returning None; (2) a degenerate/singular fit returns None and emits a
warning (`pytest.warns`).
