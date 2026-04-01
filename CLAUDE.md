# CLAUDE.md

Fiber photometry analysis pipeline for the IBL neuromodulators project.
See `README.md` for end-user docs (pipeline usage, PhotometrySession API,
DataFrame schemas, HDF5 structure).

## Where to Find Things

**Start with `config.py`** for any constant, threshold, path, lookup table,
schema definition, or visualization parameter. Everything is centralized there.

| Need | Location |
|---|---|
| File paths, output directories | `config.py` top section |
| Session DataFrame schema | `config.py → SESSION_SCHEMA` |
| NM/strain/target lookups | `config.py → STRAIN2NM, LINE2NM, TARGET2NM` |
| QC thresholds and metrics | `config.py → QC_RAW_METRICS, QC_SLIDING_METRICS, N_UNIQUE_SAMPLES_THRESHOLD` |
| Preprocessing pipeline steps | `config.py → PREPROCESSING_PIPELINES` |
| Analysis windows | `config.py → RESPONSE_WINDOW, BASELINE_WINDOW, RESPONSE_WINDOWS` |
| Colors and plot params | `config.py → NM_COLORS, TARGETNM_COLORS, SESSIONTYPE2COLOR` |
| Valid values for fields | `config.py → VALID_STRAINS, VALID_TARGETS, VALID_TARGETNMS` |
| Session/subject exclusions | `config.py → SUBJECTS_TO_EXCLUDE, EIDS_TO_DROP, EXCLUDE_SESSION_TYPES` |
| Custom exceptions | `validation.py` |
| Validate functions | `validation.py → validate_subject, validate_strain, ...` |
| Alyx/ONE queries | `io.py → get_subject_info, get_brain_region, get_datasets, ...` |
| Session utilities | `util.py → enforce_schema, collect_session_errors, get_session_type, ...` |
| PhotometrySession class | `data.py` |
| Signal processing | `analysis.py → get_responses, resample_signal, compute_bleaching_tau` |
| Psychometric fitting | `task.py → fit_psychometric, fit_psychometric_by_block, compute_fraction_correct` |
| Plotting | `vis.py` (static plots), `gui.py` (interactive viewer) |

## Architecture

```
config.py ← everything depends on this (paths, constants, schemas)
     ↑
validation.py ← defines exceptions + @exception_logger + validate_* functions
     ↑
io.py ← Alyx/ONE queries (uses @exception_logger, config lookups)
     ↑
util.py ← session utilities, schema enforcement, error log aggregation
     ↑
analysis.py ← signal processing (response extraction, resampling, bleaching)
     ↑
task.py ← behavioral performance (psychometrics, block validation)
     ↑
data.py ← PhotometrySession class (composes io, analysis, task, validation)
     ↑
vis.py / gui.py ← plotting
     ↑
scripts/ ← orchestration (filtering, iteration, figure layout)
```

**Module vs script rule**: `iblnm/` contains generic, reusable functions.
`scripts/` contains analysis-specific orchestration. Never put generic
functions in scripts.

**Group object rule**: All data loading and session filtering in scripts MUST
flow through `PhotometrySessionGroup`. Never load parquet/HDF5 files and
filter them independently with ad-hoc merges against session metadata — this
risks silently bypassing the canonical filters (`filter_sessions`,
exclusions, etc.). The group object is the single source of truth for which
sessions are in scope. Scripts that currently do ad-hoc merges against
`rec_meta` are known exceptions (FIXME), not patterns to follow.

**analysis.py vs data.py rule**: `analysis.py` contains pure functions that
take arrays/DataFrames and return results. `data.py` contains the
`PhotometrySessionGroup` class whose methods unpack `self` attributes, call
`analysis.py` functions with the correct arguments for each operation (e.g.,
each pairwise comparison), and package results back onto `self`. Never put
computational logic directly in `data.py` methods — they are orchestrators.

## Key Patterns

### 1. Exception Logger (`@exception_logger`)

The central pattern for batch processing. Understanding this is essential.

```python
# validation.py
@exception_logger
def validate_strain(session, exlog=None):
    if session['strain'] not in VALID_STRAINS:
        raise InvalidStrain(f"Unknown strain: {session['strain']}")
    return session
```

The decorator intercepts the `exlog` keyword argument:
- `exlog=None` (default): exceptions propagate normally. Used in tests.
- `exlog=[]` (list): exceptions are caught, appended as dicts to the list,
  and the original input (Series or DataFrame) is returned unchanged.

The decorated function's signature always includes `exlog=None` as the last
parameter. The decorator handles it transparently — the function body never
references `exlog` directly.

Error log entry schema: `{'eid': str, 'error_type': str, 'error_message': str, 'traceback': str}`.

When adding new validation or processing functions that can fail during batch
processing, decorate them with `@exception_logger` and accept `exlog=None`.

### 2. Error-Log-Driven Filtering

Scripts do not re-validate upstream results. Instead, they read parquet error
logs from previous stages and derive boolean flags from error type strings:

```python
df_sessions = collect_session_errors(df_sessions, [QUERY_DB_LOG, PHOTOMETRY_LOG])
# Adds 'logged_errors' column: list of error_type strings per eid

df_sessions['has_photometry'] = df_sessions['logged_errors'].apply(
    lambda e: 'MissingExtractedData' not in e)
```

Each script decides which error types are fatal for its purpose.

### 3. PhotometrySession Lifecycle

Lazy loading — all data attributes start empty:

```python
ps = PhotometrySession(session_row, one=one)
# ps.trials = None, ps.photometry = {}, ps.responses = None, ps.qc = None

ps.load_trials()       # populates ps.trials
ps.load_photometry()   # populates ps.photometry['GCaMP'], ps.photometry['Isosbestic']
ps.preprocess()        # adds ps.photometry['GCaMP_preprocessed']
ps.extract_responses() # populates ps.responses (xarray DataArray)
```

Access data attributes directly, not through getters. The class extends
`PhotometrySessionLoader` from `brainbox.io.one`.

HDF5 round-trip: `save_h5(mode='w')` writes signal, `save_h5(mode='a')`
appends trials/responses/wheel. `load_h5(fpath)` populates all available
groups.

### 3b. PhotometrySessionGroup Lifecycle

```python
group = PhotometrySessionGroup.from_catalog(SESSIONS_FPATH, one=one)
group.filter_sessions(session_types=('biased', 'ephys'), targetnms=TARGETNMS_TO_ANALYZE)
group.deduplicate()
```

- `group.sessions` — property: session-level rows passing both the filter mask
  and the dedup mask. List columns: `brain_region`, `hemisphere`, `target_NM`.
- `group.recordings` — property: one row per region, scalar columns, plus
  `fiber_idx`. Derived by exploding `sessions` on the parallel list columns.
  Filtered to `_recordings_targetnms` (set by `filter_sessions`). Always
  reflects the current filter and dedup state.

`from_catalog` applies `enforce_schema` and `validate_parallel_lists` before
constructing the object. `filter_sessions(targetnms=TARGETNMS_TO_ANALYZE)` is
the explicit default — pass `targetnms=False` to skip the target filter.

### 4. Parallel List Columns

`brain_region`, `hemisphere`, and `target_NM` are parallel lists on each
session row. They must always have matching lengths. A session recording from
VTA and SNc looks like:

```python
brain_region = ['VTA', 'SNc']
hemisphere   = ['l',   'r']
target_NM    = ['VTA-DA', 'SNc-DA']
```

When exploding sessions to one row per recording, explode all three together:
`df.explode(['brain_region', 'hemisphere', 'target_NM'])`.

### 5. Schema Enforcement

`enforce_schema(df, SESSION_SCHEMA)` fills missing columns with typed defaults
and initializes list columns (replaces NaN with `[]`). Called when loading
`sessions.pqt` to ensure downstream code can assume columns exist.

## Development Principles

- **No backward compatibility shims.** This is an active development project,
  not a library with external consumers. When renaming, delete the old name
  everywhere. No aliases, no re-exports, no deprecation wrappers.
- **Run `ruff check` and the full test suite (`pytest`) only before pushing
  code**, not after every small edit. Trust the code between pushes.
- **Small, modular commits.** Each commit should do one thing. Prefer many
  focused commits over one large commit.

## Code Conventions

- **Naming**: bare nouns preferred for metrics (`fraction_correct`). Prefixes
  `validate_`, `get_`, `compute_`, `run_`, `fit_` clarify intent on verbs.
- **Docstrings**: numpy-style for functions with 3+ parameters. Omitted when
  self-documenting.
- **Parquet for tabular data, HDF5 for session data**: `sessions.pqt` is the
  central catalog; signals, trials, and responses live in `data/sessions/{eid}.h5`.
- **Error logs**: unified schema `['eid', 'error_type', 'error_message', 'traceback']`.
  One parquet log per pipeline stage in `metadata/`.

## Testing

Tests use `pytest` with synthetic fixtures. No Alyx calls.

| Test file | Covers |
|---|---|
| `test_data.py` | PhotometrySession validation, loading, preprocessing |
| `test_validation.py` | `@exception_logger`, all `validate_*` functions |
| `test_task.py` | Block structure, psychometric fitting |
| `test_analysis.py` | Response extraction, bleaching tau |
| `test_util.py` | Session utilities, error log merging, schema enforcement |
| `test_io.py` | Query functions (mocked ONE) |
| `test_vis.py` | Plotting functions |
| `test_dataset_overview.py` | Dataset flag construction |
| `test_wheel.py` | Wheel velocity extraction |

Key fixtures in test files:
- `mock_session_series()` — synthetic session metadata row
- `mock_photometry_data()` — synthetic signal with known bleaching/correlation
- `mock_photometry_session()` — PhotometrySession with injected data

Pattern: test exceptions via `pytest.raises(ExceptionType)` without `exlog`.
Test `@exception_logger` behavior by passing `exlog=[]` and checking the list.

**Visualization tests**: Only test things that affect plot interpretation and
aren't trivially assigned — data transformations, rearrangements, correct
mapping of values to visual encodings. Skip tests for things like panel
count or figure size that are obvious from reading the code.

## Environment

```bash
uv pip install -e .   # Install (venv: ~/.venv/ibl)
pytest                # Test
ruff check .          # Lint
```

**Dependencies**: `one-api`, `brainbox`, `iblphotometry` (IBL ecosystem)

**Docs**: [IBL](https://docs.internationalbrainlab.org/) · [ONE API](https://int-brain-lab.github.io/ONE/)
