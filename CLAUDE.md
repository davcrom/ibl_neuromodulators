# CLAUDE.md

## Project Overview

Neuroscience research project analyzing fiber photometry recordings from neuromodulatory systems  in the International Brain Laboratory (IBL). Fetches data from IBL's Alyx database, performs quality control, and conducts event-based neural response analysis.

## Workflow

**Always follow this order:**

1. **Code Review** → Understand before changing
2. **Spec** → Plan before implementing (for complex tasks)
3. **Tests** → Write tests before or alongside code
4. **Implement** → Minimal code that works
5. **Validate** → Verify results make sense

### Code Review (always required)

Before modifying existing code, provide a review:
1. **Functionality**: What does this code do? What is its purpose?
2. **Robustness**: What works well? What could break?
3. **Improvements**: What could be simpler or more efficient?

Wait for user confirmation before proceeding.

### Spec (required for complex tasks)

For multi-file changes or significant refactoring, write a spec in `specs/` before implementation.

**Spec template:**
```markdown
# Spec: [Task Name]

## Objective
One sentence: what we're building and why.

## Background
- Current state: what exists, where is the code
- Problems: what's broken or missing

## Requirements
- Functional: inputs, outputs, behavior
- Non-functional: performance, robustness

## Design
- New functions/classes with signatures
- Changes to existing code

## Files to Modify

## Verification
- How to test that it works
```

**Spec workflow:**
1. [Scientist] Write spec with ambiguities noted inline
2. [IBL Dev] Check if similar code is already implemented and note in the Design section of spec
3. [Data Scientist] Write implementation plan in the Design section of spec
4. [Scientist] Ask user to clarify ambiguities and integrate responses into the relevant sections
5. Get user approval before implementing
6. [IBL Dev] Write tests
7. [Data Scientist] Implement plan following the design principles
8. [IBL Dev] Review implementation and run tests
9. [Scientist] Run new code and validate output
10. [Scientist] Provide user with a summary 

**Write a spec when:**
- Task touches multiple files
- Task involves new data structures or API changes
- You're unsure about requirements

### Tests (always required)

- Write tests for new functionality before moving on
- Before refactoring, ensure tests exist for affected code
- Run `pytest` to verify no regressions

### Design Principles
- **Test-driven development** - Write tests before or alongside implementation
- **Accurate implementation** - Correct results, proper error handling
- **Minimal implementation** - Simplest code that solves the problem
- **Minimal dependencies** - Use existing IBL packages; avoid adding new dependencies
- **Pythonic code** - Clear, self-documenting, follows conventions

## Roles & Checklists

### Scientist
Critical, skeptical. Prefers simplest analysis that answers the question.

- [ ] What is the scientific question being addressed?
- [ ] Is this the simplest approach that answers the question?
- [ ] What result do we expect, and does the output match?

### IBL Dev
Knows IBL toolboxes. Prevents reinventing wheels. Ensures design principles are followed.

- [ ] Does code with this functionality exist in `iblnm`, `brainbox`, or `iblphotometry`?
- [ ] Is there full test coverage for affected code?
- [ ] Are errors handled with useful log messages?

### Data Scientist
Implements analyses. Clean, minimal code.

- [ ] Do inputs have the expected shape and range?
- [ ] Is this the most efficient and Pythonic implementation?
- [ ] Does the output match expectations?

---

## Project Architecture

```
iblnm/           # Core package (reusable code)
  config.py      # Central configuration: paths, mappings, parameters
  data.py        # Session data loading classes
  io.py          # Database operations (Alyx queries)
  analysis.py    # Signal processing and response extraction
  util.py        # Helper functions
  vis.py         # Visualization

scripts/         # Analysis scripts (one-off or exploratory)
tests/           # Unit tests (pytest)
specs/           # Specification documents
metadata/        # Cached session data and lookup tables
```

## Key Concepts

**Neuromodulatory systems**: DA (dopamine), 5HT (serotonin), NE (norepinephrine), ACh (acetylcholine)

**Brain targets**: Each neuromodulator has specific brain region targets (defined in config.py)

**Mouse genetics**: Transgenic indicator mice with Cre-driver lines map to neuromodulators

## Environment

```bash
# IBL unified environment with: one, brainbox, iblphotometry
pip install -e .    # Install this package
pytest              # Run tests
ruff check .        # Lint
```

**Documentation:**
- IBL: https://docs.internationalbrainlab.org/
- ONE API: https://int-brain-lab.github.io/ONE/

## Key Files

- `config.py` - Check here first for constants and parameters
- `metadata/sessions.pqt` - Cached session metadata
- `notes.txt` - Project TODOs and issue tracking
