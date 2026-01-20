# CLAUDE.md

IBL fiber photometry analysis: data from Alyx → QC → event-based neural responses.

## Workflow

1. **Clarify** → Ask questions before implementing new features
2. **Review** → Understand existing code before changing it
3. **Spec** → Write spec in `specs/` for non-trivial tasks
4. **Test** → Write tests before or alongside code
5. **Implement** → Minimal code that works
6. **Validate** → Verify results make sense

### Key Rules

- **Ask first**: What exactly is needed? Any ambiguities?
- **Scope discipline**: Implement what's requested. Improve code quality, but don't add unrequested features.
- **Review before editing**: Summarize functionality, robustness, potential improvements. Wait for confirmation.
- **Spec when**: new feature/analysis, multi-file changes, ambiguous requirements
- **Skip spec for**: clear bug fixes, simple refactors, adding tests

### Spec Template

```markdown
# Spec: [Task Name]
## Objective - what and why (one sentence)
## Background - current state, problems
## Requirements - inputs, outputs, behavior
## Design - functions/classes, changes to existing code
## Files to Modify
## Verification - how to test
```

## Roles

| Role | Focus | Checklist |
|------|-------|-----------|
| **Scientist** | Simplest analysis that answers the question | Scientific question? Simplest approach? Expected result? |
| **IBL Dev** | Use existing code, ensure test coverage | Exists in brainbox/iblphotometry? Tests? Error handling? |
| **Data Scientist** | Clean, minimal implementation | Valid inputs? Pythonic? Output correct? |

## Project Structure

```
iblnm/           # Core package
  config.py      # Paths, mappings, parameters (check here first)
  data.py        # PhotometrySession class
  io.py          # Alyx queries
  analysis.py    # Response extraction
  util.py        # Helpers
  vis.py         # Plotting
scripts/         # Analysis scripts
tests/           # pytest
specs/           # Planning docs
metadata/        # sessions.pqt, lookup tables
```

## Reference

**Neuromodulators**: DA, 5HT, NE, ACh → brain targets defined in `config.py`

**Environment**: IBL unified (`one`, `brainbox`, `iblphotometry`)
```bash
pip install -e .  # Install
pytest            # Test
ruff check .      # Lint
```

**Docs**: [IBL](https://docs.internationalbrainlab.org/) · [ONE API](https://int-brain-lab.github.io/ONE/)

**Key files**: `config.py` (constants), `metadata/sessions.pqt` (cached data), `notes.txt` (TODOs)
