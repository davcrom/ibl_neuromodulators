# CLAUDE.md

Fiber photometry analysis pipeline for the IBL neuromodulators project.
Ingests session metadata from Alyx, applies QC, preprocesses signals, and
extracts peri-event neural responses. See `README.md` for full API docs,
DataFrame schemas, and HDF5 file structure.

## Pipeline

Scripts run in order. Each produces a parquet error log alongside its outputs.

1. `query_database.py` — fetch session metadata from Alyx → `metadata/sessions.pqt`
2. `photometry.py` — QC, preprocess, extract responses → `data/sessions/{eid}.h5`, `data/qc_photometry.pqt`
3. `task.py` — compute task performance → `data/performance.pqt`
4. `dataset_overview.py` — session coverage figures
5. `qc_overview.py` — QC metric distributions
6. `task_performance_overview.py` — learning curves, psychometrics

## Project Structure

```
iblnm/              # Core package — generic, reusable, no analysis-specific parameters
  config.py          # Paths, schemas, constants, lookup tables, QC thresholds, colors
  data.py            # PhotometrySession: loading, validation, QC, preprocessing, responses
  io.py              # Alyx/ONE queries (subject info, session info, datasets)
  task.py            # Task performance (psychometrics, block validation)
  analysis.py        # Signal processing (response extraction, bleaching tau)
  validation.py      # Custom exceptions, exception_logger decorator, validate_* functions
  util.py            # Logging helpers, pandas utilities, parquet I/O, schema enforcement
  vis.py             # Plotting functions

scripts/             # Analysis-specific: filtering, orchestration, figure layouts
tests/               # pytest (unit tests with synthetic fixtures, no Alyx calls)
specs/               # Design docs for non-trivial features
metadata/            # sessions.pqt, per-script error logs, fibers.csv
data/                # qc_photometry.pqt, performance.pqt, sessions/*.h5
```

## Key Patterns

### Exception Logger (`@exception_logger`)

The central pattern enabling batch processing. Decorated functions accept an
optional `exlog` parameter. When `exlog` is provided, exceptions are logged
(not raised) and the original series is returned so processing continues.
Without `exlog`, exceptions propagate normally (used in tests).

```python
@exception_logger
def get_targetNM(session):
    ...
    raise InvalidTargetNM(...)

# In scripts — errors logged, pipeline continues:
error_log = []
df = df.apply(get_targetNM, axis='columns', exlog=error_log)

# In tests — errors raised normally:
with pytest.raises(InvalidTargetNM):
    get_targetNM(bad_session)
```

Error logs follow a unified schema: `['eid', 'error_type', 'error_message', 'traceback']`.

### Error-Log-Driven Filtering

Downstream scripts don't re-validate; they read upstream error logs and derive
flags from error types. This keeps validation logic in one place and lets each
script decide which errors are fatal for its purpose.

```python
# dataset_overview.py loads all upstream logs
df_sessions = collect_session_errors(df_sessions, [QUERY_DB_LOG, PHOTOMETRY_LOG, TASK_LOG])

# Then derives boolean flags from error types
df_sessions['has_raw_data'] = df_sessions['logged_errors'].apply(
    lambda e: 'MissingRawData' not in e)
```

### PhotometrySession

Data container with convenience methods. Lazy loading by default — data
attributes (`trials`, `photometry`, `responses`, `qc`) start empty and are
populated by explicit method calls. Access data directly, not through getters.

### config.py

Check here first. Contains all paths, the `SESSION_SCHEMA` (column → type/default),
neuromodulator/strain/target lookup tables, QC thresholds, preprocessing
pipeline definitions, visualization constants, and session filtering lists.

## Code Conventions

- **Module vs script**: `iblnm/` has generic functions; `scripts/` has
  analysis-specific orchestration. No generic functions in scripts.
- **Naming**: bare verbs preferred (`fraction_correct`), with `validate_`,
  `get_`, `compute_`, `run_` prefixes where they clarify intent.
- **Docstrings**: numpy-style for functions with 3+ parameters. Omitted when
  code is self-documenting.
- **Parallel list columns**: `brain_region`, `hemisphere`, and `target_NM` are
  parallel lists on each session row. They must always have matching lengths.
- **Parquet for tabular data, HDF5 for session data**: `sessions.pqt` is the
  central catalog; each session's signals, trials, and responses live in
  `data/sessions/{eid}.h5`.

## Environment

```bash
uv pip install -e .   # Install (venv: ~/.venv/ibl)
pytest                # Test
ruff check .          # Lint
```

**Dependencies**: `one-api`, `brainbox`, `iblphotometry` (IBL ecosystem)

**Docs**: [IBL](https://docs.internationalbrainlab.org/) · [ONE API](https://int-brain-lab.github.io/ONE/)
