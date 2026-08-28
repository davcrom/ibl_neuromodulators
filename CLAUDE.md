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

## Repository Layout

```
iblnm/              # Core package (tracked)
scripts/             # Pipeline stages and analysis scripts (tracked)
tests/               # pytest suite, synthetic fixtures (tracked)
metadata/fibers.csv  # Fiber insertion coordinates (tracked)

# Generated outputs (gitignored)
metadata/            # sessions.pqt, errors.pqt
data/                # qc_photometry.pqt, performance.pqt, sessions/*.h5
results/             # Analysis CSVs (responses/, task_encoding/)
figures/             # Output plots

# Local working files (gitignored)
specs/               # Design specs and implementation plans
tickets/             # Implementation tickets (local working notes, never committed)
notes/               # Personal notes, code snippets, reports, brainstorm docs
histology/           # Histology images (.tif) and fiber track CSVs
```

**Layering rules** (the dependency diagram is the spine; these constrain it):

- `analysis.py` — pure, **variable-agnostic** functions: take arrays/DataFrames
  plus arguments, return results. No model formulas, no predictor coding for
  specific variable sets, no hardcoded column-name logic.
- `data.py` — `PhotometrySessionGroup` methods orchestrate: unpack `self`, code
  and shape this object's data, route it through `analysis.py`, package results
  back onto `self`. Reusable computation belongs in `analysis.py`; the analysis
  to run is the caller's, not chosen here.
- `scripts/` — answer a specific scientific question; every analysis choice is
  made here, read from `config.py` and passed downward as arguments. Never put
  generic functions here.
- Decisions about what to compute flow downward as arguments — which models,
  predictors, and comparisons, in what sequence; no layer reaches up to
  `config.py` to choose its own analysis.
- New model or variable set? Generalize the existing `analysis.py` function and
  have the caller pass the specifics — never add a case-specific variant.

**Group object rule**: All data loading and session filtering in scripts MUST
flow through `PhotometrySessionGroup`. Never load parquet/HDF5 files and
filter them independently with ad-hoc merges against session metadata — this
risks silently bypassing the canonical filters (`filter_sessions`,
exclusions, etc.). The group object is the single source of truth for which
sessions are in scope. Scripts that currently do ad-hoc merges against
`rec_meta` are known exceptions (FIXME), not patterns to follow.

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

Error log entry schema: `{'eid': str, 'error_type': str, 'error_message': str,
'traceback': str, 'product': str | None}`.

When adding new validation or processing functions that can fail during batch
processing, decorate them with `@exception_logger` and accept `exlog=None`.

### 2. Error-Log-Driven Filtering

Scripts do not re-validate upstream results. Each session's errors are the
single source of truth in its H5 `/errors` group (written by every pipeline
stage via `ps.log_error` / the `process` wrapper). There are no per-stage error
log parquets to keep in sync.

`PhotometrySessionGroup.from_catalog(catalog, one, h5_dir=SESSIONS_H5_DIR)`
scans those `/errors` groups and adds a `logged_errors` column (list of
error_type strings per eid). `filter_sessions(qc_blockers=...)` then drops
sessions carrying a blocking error type. Pass `h5_dir=None` (the default, e.g.
in tests) to skip the scan and leave `logged_errors` empty.

```python
group = PhotometrySessionGroup.from_catalog(df, one=one, h5_dir=SESSIONS_H5_DIR)
group.filter_sessions(qc_blockers=ANALYSIS_QC_BLOCKERS)
```

The standalone helper `collect_session_errors(eids, h5_dir)` returns the
`['eid', 'logged_errors']` table directly when a script needs it (e.g. to
augment with synthetic blockers before constructing the group). Each script
decides which error types are fatal for its purpose.

### 3. PhotometrySession Lifecycle

Lazy loading — all data attributes start empty:

```python
ps = PhotometrySession(session_row, one=one)
# ps.trials = None, ps.photometry = {}, ps.photometry_responses = {},
# ps.movement_responses = {}, ps.qc = None

ps.load_trials()          # populates ps.trials
ps.load_raw_photometry()  # ps.photometry['GCaMP'], ps.photometry['Isosbestic']
ps.preprocess()           # adds ps.photometry['GCaMP_preprocessed'], writes it
ps.load_photometry()      # the preprocessed signal, from H5 or built as above
ps.load_responses('photometry')   # ps.photometry_responses, from H5 or cut
```

Load methods are not pure readers. Each attempts its stored product, falls back
to building it, and writes what it built, so a session with an empty H5 fills
itself from Alyx. `load_raw_photometry` and `load_photometry` stay separate —
one method returning either raw or preprocessed is how an analysis silently
runs on the wrong signal. A stored stamp that disagrees with `config.py` raises
`StaleProduct` instead of rebuilding; put the product key in `ps.rebuild` to
force a rebuild.

`load_responses(modality, events, window)` serves every modality, dispatching
through `_RESPONSE_MODALITIES` to that modality's preprocessed-signal loader and
result attribute.

`extract_responses(signals, events=..., window=...)` is signal-source agnostic:
it cuts peri-event matrices out of any `label -> pd.Series` mapping and returns
`dict[label, xr.DataArray]` with dims `(event, trial, time)`, leaving the
caller to assign it. Photometry passes `ps.photometry[PREPROCESSED_BAND]`
(labels are brain regions) and assigns `ps.photometry_responses`; behavior passes
`ps._movement_signals()` with `events=MOVEMENT_EVENTS` (labels are movement
channels) and assigns `ps.movement_responses`. Each movement channel carries
the full event axis; its own response event is selected at read time via
`config.LABEL2EVENT`, and its baseline is the `stimOn_times` cell.

The window is `(t0, t1)`. `t1` is either seconds relative to the event, or the
name of a `ps.trials` column holding each trial's own window end — that second
form is the wheel's cut, `window=(0.0, 'feedback_times')`. Every trial still
shares one time axis spanning to the longest trial and is NaN-padded beyond its
own endpoint. The `trial` coordinate is the `trial` column of `ps.trials`, not
the row position, so responses stay aligned to their trials after filtering.

Truthiness (`if ps.photometry_responses`) checks whether any label has been
extracted; use `region in ps.photometry_responses` to check a specific one.
`subtract_baseline` and `mask_subsequent_events` operate on one label's
DataArray at a time — pass `ps.photometry_responses[region]`.

Access data attributes directly, not through getters. The class extends
`PhotometrySessionLoader` from `brainbox.io.one`.

HDF5 round-trip: `save_h5()` writes all available data groups.
`load_h5(fpath)` populates all available groups. Both dispatch to per-group
handler functions via `_SAVE_HANDLERS` / `_LOAD_HANDLERS` registries keyed
by top-level group name (`metadata`, `errors`, `photometry`, `trials`,
`wheel`, `video`). Adding a new top-level group means writing a handler pair
and registering it in both dicts. See README for the on-disk layout.

Beneath the top-level handlers sit three save/load pairs keyed by the **data
structure** they carry rather than by modality. Each is pure: it takes one
`h5py.Group` plus a payload and never touches the session object. The
orchestrator creates the group (`_replace_group`), loops over regions or
labels, and passes the product's resolved spec, which the save writes as the
group's stamp (`_write_stamp`, read back by `product_status`).

| pair | payload | used for |
|---|---|---|
| `_save_time_series` / `_load_time_series` | time-indexed `pd.Series` (one signal, dataset `signal`) or `pd.DataFrame` (one dataset per column) | preprocessed photometry, wheel |
| `_save_peri_event_matrix` / `_load_peri_event_matrix` | `xr.DataArray(event, trial, time)` | responses, whether the label is a brain region (`photometry/`) or a movement channel (`video/`) |
| `_save_scalars` / `_load_scalars` | flat `dict[str, float]` stored as group attrs | QC metrics, preprocessing diagnostics |

`_read_label_responses(modality_group)` reads every `{label}/responses` subgroup
of one modality in one call, for the loads that want the whole mapping rather
than one label.

`video/pose/qc` is the exception: it holds arrays plus a scalar, so it keeps
its own pair (`_save_pose_xcorr` / `_load_pose_xcorr`).

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

- **Specs go in `specs/`.** Write design specs and implementation plans to
  `specs/` before starting work. Never put spec files in `scripts/` or other
  code directories.
- **Tickets go in `tickets/`, never committed.** `tickets/` is gitignored
  local working notes; never stage or commit ticket files.
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
- **Analysis outputs go in `results/`**: each analysis script writes to its
  own subdirectory (`results/responses/`, `results/task_encoding/`). Output
  paths are defined in `config.py`.
- **Error logs**: unified schema `['eid', 'error_type', 'error_message',
  'traceback', 'product']` (`util.LOG_COLUMNS`). Each session's errors live in
  its H5 `errors/` tree, one group per product, rewritten whole on every build
  attempt of that product.
- **Signed-zero in `signed_contrast`**: zero-contrast trials encode stimulus
  side via IEEE 754 signed zero (`-0.0` = left, `0.0` = right). `unique()`,
  `sorted()`, `set()`, `==`, and pandas `groupby` all treat `-0.0 == 0.0` and
  will collapse the distinction. Use `np.signbit()` when side matters at zero
  contrast, or use the `stim_side` column (the authoritative source).

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

Requires `ibllib` (`photometry-integration` branch) and `ibl-photometry`
(`develop` branch). Core dependencies (`xarray`, `statsmodels`, `cca-zoo`,
etc.) and dev tools (`pytest`, `ruff`) are declared in `pyproject.toml`
with optional `[ibl]` and `[dev]` groups. See `README.md` for full
install instructions.

**Docs**: [IBL](https://docs.internationalbrainlab.org/) · [ONE API](https://int-brain-lab.github.io/ONE/)
