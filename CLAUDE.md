# CLAUDE.md

Fiber photometry analysis pipeline for the IBL neuromodulators project.
See `README.md` for end-user docs (pipeline usage, PhotometrySession API,
DataFrame schemas, HDF5 structure).

## What the Signal Is

**The recorded signal is a genetically encoded calcium indicator (GCaMP), not
a neuromodulator sensor.** Nothing in this dataset uses GRAB, dLight, iAChSnFR,
or any other transmitter-binding sensor, and no analysis here measures
transmitter release. Do not assume otherwise.

Cell-type specificity comes from a double-transgenic cross: a Cre-dependent
GCaMP reporter line (Ai148, and Ai95 in a few subjects) crossed to a Cre driver
for the neuromodulatory population —

| Cre driver | Population | `NM` label |
|---|---|---|
| DAT-Cre | midbrain dopamine (VTA, SNc) | `DA` |
| DbH-Cre (some TH-Cre) | locus coeruleus noradrenaline | `NE` |
| ChAT-Cre | basal forebrain / brainstem acetylcholine | `ACh` |
| SERT-Cre | raphe serotonin | `5HT` |

So `NM`, `target_NM`, `STRAIN2NM`, `LINE2NM`, and `TARGET2NM` all name **which
neuromodulatory cell population is expressing GCaMP** — the identity of the
recorded neurons — never a chemical species being sensed. `photometry['GCaMP']`
is the calcium band and `photometry['Isosbestic']` its control band; a response
is calcium activity in that population, phrased as e.g. "VTA-DA calcium
response", not "dopamine release".

## Where to Find Things

**Start with `config.py`** for any constant, threshold, path, lookup table,
schema definition, or visualization parameter. Everything is centralized there.

| Need | Location |
|---|---|
| File paths, output directories | `config.py` top section |
| Session DataFrame schema | `config.py → SESSION_SCHEMA` |
| NM/strain/target lookups | `config.py → STRAIN2NM, LINE2NM, TARGET2NM` |
| QC thresholds and metrics | `config.py → QC_RAW_METRICS, QC_SLIDING_METRICS, PHOTOMETRY_QC_THRESHOLDS` |
| Preprocessing pipeline steps | `config.py → PREPROCESSING_PIPELINES` |
| Analysis windows | `config.py → RESPONSE_WINDOW, BASELINE_WINDOW, RESPONSE_WINDOWS` |
| Colors and plot params | `config.py → NM_COLORS, TARGETNM_COLORS, SESSIONTYPE2COLOR` |
| Valid values for fields | `config.py → VALID_STRAINS, VALID_TARGETS, VALID_TARGETNMS` |
| Session/subject exclusions | `config.py → SUBJECTS_TO_EXCLUDE, EIDS_TO_DROP, EXCLUDE_SESSION_TYPES` |
| Custom exceptions | `validation.py` |
| Validate functions | `validation.py → validate_subject, validate_strain, ...` |
| Alyx/ONE queries | `io.py → get_subject_info, get_brain_region, get_datasets, ...` |
| Session utilities | `util.py → enforce_schema, get_session_type, ...` |
| Store rollups | `data.py → PhotometrySessionGroup.collect_errors, collect_qc, collect_pose` |
| Session build, one block per modality | `scripts/download.py → build_session, build_trials, build_photometry, build_wheel, build_video` |
| Download CLI (`--workers`, `--session-type`, `--target-NM`) | `scripts/download.py → parse_args, main` |
| Session catalog (`sessions.pqt`) | `scripts/download.py → fetch_catalog` |
| Response re-cut, that product alone | `scripts/rebuild_responses.py → rebuild_responses, main` |
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
single source of truth in its H5 `/errors` group (written by each build block
via `ps.log_error`, and by the `process` wrapper for anything that escapes
them). There are no per-stage error log parquets to keep in sync.

`PhotometrySessionGroup.from_catalog(catalog, one, h5_dir=SESSIONS_H5_DIR)`
opens each catalogued session's file once and reads out everything the filters
need: the `/errors` tree, `trials/performance` and `photometry/{region}/raw/qc`.
That scan is `complete_catalog`, and it leaves `logged_errors` (list of
error_type strings per eid), `fraction_correct` and `contrasts` on `_catalog`
and the per-recording QC on `group.photometry_qc`.
`filter_sessions(qc_blockers=...)` then drops sessions carrying a blocking
error type. Pass `h5_dir=None` (the default, e.g. in tests), or
`scan_h5=False`, to skip the scan; the columns are then defaulted empty, which
fails the filters reading them rather than passing them by absence.

```python
group = PhotometrySessionGroup.from_catalog(df, one=one, h5_dir=SESSIONS_H5_DIR)
group.filter_sessions(qc_blockers=ANALYSIS_QC_BLOCKERS)
```

`group.collect_session_errors()` returns that scan's `['eid', 'logged_errors']`
table on its own, for a script that wants it before the filters run — to append
a synthetic blocker of its own, say. It rescans, so build the group with
`scan_h5=False` and call it explicitly in that case, and the store is read once
rather than twice. Each script decides which error types are fatal for its
purpose.

Every rollup reads through the group and its filters, never by globbing the
store. There is no shared rollup script: a script that wants a flat table
writes it where it reads it, from these three methods. `collect_errors` (the
full log for the filtered sessions), `collect_qc`
(one row per catalogued recording, from `photometry/{region}/raw/qc`, as
`complete_catalog` scanned it) and `collect_pose` (the video table).
`collect_qc` and `collect_session_errors` walk `_catalog` rather than the
filtered view, because what they return feeds `filter_sessions` and the mask
does not exist yet — the same reason `load_performance` does. `PhotometrySessionGroup.from_h5_dir(h5_dir, one, scan_h5=True)`
goes the other way, rebuilding a catalog from the `metadata` groups of files
already written. It also derives the two per-subject rankings — `day_n`, days
since the subject's first session, and `session_n`, that session's dense rank
among the subject's days — because they need every session at once and this is
the method that has them; deriving them in `from_catalog` instead would
recompute them for every analysis script reading `sessions.pqt`. Neither column
is in `SESSION_SCHEMA`, so `enforce_schema` carries them along untouched. Its
`scan_h5` is passed on to `from_catalog`, so `scan_h5=False` reads each file
once instead of twice.

`group.fix_catalog()` is the step after it, and the only one that runs on a
group rather than on a table: it repairs `_catalog` in place with the
`util.fix_catalog` fixups — filling empty regions from the subject's other
sessions, from `metadata/fibers.csv`, correcting region names and deriving
`target_NM`. In place, because `group.sessions` hands back a copy, and
`PhotometrySession` reads `brain_region` off its catalog row to name the
photometry columns; a group built from an unrepaired catalog cannot map its own
signal. The whole function is a TEMPFIX and goes when the upstream Alyx
metadata is corrected. `scripts/download.py → fetch_catalog` is the caller:
read the store, fix, write `sessions.pqt`, then filter — one group through the
whole phase, validated once, after the fixups rather than before them.

### 2b. The Download Build

`scripts/download.py` builds the whole store. Every product is built on every
run: nothing is read back, skipped, or compared against what the file already
holds. `build_session(ps)` is four straight-line blocks — `build_trials`,
`build_photometry`, `build_wheel`, `build_video` — and then
`ps.save_h5(mode='w')`, which rewrites the file whole.

Block order is load-bearing, and it is the only dependency declaration there
is: trials come first because the response cuts read `ps.trials`, and the wheel
precedes the video because `run_pose_qc` reads `ps.wheel_velocity`. A product
finds its input because the block above it left it on the session.

Order matters within a block too, and for a second reason: which error a
failure is logged as. `build_photometry` fetches the extracted bands before the
neurophotometrics table because `fetch_photometry` is the only step that tells
an absent recording from an unextracted one — `MissingRawData` or
`MissingExtractedData`, both in `config.ANALYSIS_QC_BLOCKERS` — where
`fetch_neurophotometrics` raises a bare `ALFObjectNotFound` for the same
session, which blocks nothing. Put the unclassified fetch first and it
abandons the block before the classification is reached, and the session is
analysed with no photometry in it. `validate_qc` still runs before anything is
computed from the bands, which is the invariant that ordering has to keep: a
band inversion means the channels are not the bands they are labelled. Fetching
is not computing.

Each block is one `try`, reproducing the boundary the pipeline had when every
modality was its own script: a fatal step abandons its block, logs one error
against that modality, and the next block still runs. Within a block some checks
are non-fatal — `validate_event_completeness` and `validate_block_structure` in
trials are caught inline and logged, and an incomplete event set in photometry
degrades the response cut to the events that survive rather than abandoning it.
Video's three raw fetches are caught separately, so a session with no
LightningPose still contributes its motion energy and its camera clock.

The CLI narrows which sessions run, never what is built within one:

| flag | effect |
|---|---|
| `--workers N`, `-w N` | parallel worker processes |
| `--session-type` | restrict to the named `config.SESSION_TYPES` |
| `--target-NM` | restrict to sessions carrying a recording from one of the named `config.VALID_TARGETNMS` |

`main` runs `fetch_catalog` first, then `filter_sessions` with every analysis
filter switched off — those are the criteria a session must clear to be
*analysed*, not to be built, and the raw-photometry QC filter especially, since
it reads the QC this pass exists to compute.

### 2c. The Response Rebuild

`scripts/rebuild_responses.py` is the one pass that rebuilds a single product.
It re-cuts `photometry/{region}/responses` and nothing else, for the case a
full download would answer wastefully: `config.RESPONSE_EVENTS`,
`config.RESPONSE_WINDOW` or the cut changed, and every other product in the
store is still current.

It is the mirror image of the download path in what it reads. `main` builds the
group with `from_h5_dir(scan_h5=False)` and `fix_catalog`, so the session list
comes from the store's own `metadata` groups rather than from Alyx or
`sessions.pqt`, and it turns the analysis filters off for the same reason
`download.main` does. `rebuild_responses(ps)` then goes through `load_trials`
and `load_photometry` — the `load_*` tier the download pass never touches,
because here the store is the input rather than the output. Those two loads
read the file; the fallback in `load_photometry` is the only thing that can
reach Alyx, when a session holds no preprocessed band at all.

`ps.complete_events()` is shared with `build_photometry`, and
`save_h5(groups=['photometry'])` replaces each region's `responses` subgroup —
so a renamed event leaves no orphan dataset behind — while round-tripping the
preprocessed band and the QC beside it through the handler pair that read them.
The CLI flags are the download's three, with the same meaning.

### 3. PhotometrySession Lifecycle

Lazy loading — a freshly constructed session carries its metadata and nothing
else. Data attributes are not pre-set to `None` or `{}`: an attribute exists
once something has put it there, so reading one that no step has filled is an
`AttributeError`, not a silent empty value. `hasattr` is therefore the presence
check throughout, and a processing method whose input is missing raises for the
caller to catch and log.

```python
ps = PhotometrySession(session_row, one=one)
# no ps.trials, no ps.photometry, no ps.photometry_responses, ...

ps.load_trials()          # populates ps.trials
ps.load_performance()     # ps.performance, from H5 or scored from the trials
ps.load_raw_photometry()  # ps.photometry['GCaMP'], ps.photometry['Isosbestic']
ps.load_photometry()      # ps.photometry['GCaMP_preprocessed'], from H5 or built
ps.load_responses('photometry')   # ps.photometry_responses, from H5 or cut
ps.load_photometry_qc()           # ps.photometry_qc, from H5 or scored
ps.load_neurophotometrics_qc()    # ps.neurophotometrics_qc, from H5 or scored
ps.load_raw_wheel()       # ps.wheel_position, the irregular encoder samples
ps.load_wheel()           # ps.wheel_velocity at WHEEL_FS, from H5 or built
ps.load_responses('wheel')        # ps.wheel_responses, from H5 or cut
ps.load_peak_velocity()   # ps.wheel_peak_velocity per trial, from H5 or reduced
ps.load_camera_times()    # ps.pose_times, from H5 or Alyx
ps.load_pose()            # ps.pose, from H5 or Alyx
ps.load_motion_energy()   # ps.motion_energy, from H5 or Alyx
ps.load_video_times_qc()          # ps.video_times_qc, from H5 or scored
ps.load_pose_qc()                 # ps.pose_xcorr, from H5 or correlated
ps.load_responses('video')        # ps.movement_responses, from H5 or cut
ps.fetch_video_qc()               # ps.video_qc, always from Alyx, never stored
ps.set_manual_qc(field, value)    # a hand-set verdict, written on its own
```

Three tiers of method sit under this. `fetch_*` takes Alyx in and puts a session
attribute out, nothing else: `fetch_trials`, `fetch_photometry`,
`fetch_neurophotometrics`, `fetch_wheel`, `fetch_camera_times`, `fetch_pose`,
`fetch_motion_energy`. `extract_*` and `run_*_qc` read their inputs off the
session attributes, compute, and assign the result back; they never fetch and
never save. Every derived product has exactly one of them; the fetch-only
products — `trials/table` and the raw groups — have none:

| product | computation |
|---|---|
| `trials/performance` | `extract_performance` |
| `photometry/preprocessed` | `extract_preprocessed_photometry` |
| `wheel/preprocessed` | `extract_wheel_velocity` |
| `wheel/peak_velocity` | `extract_peak_velocity` |
| `video/preprocessed` | `extract_movement_signals` |
| `photometry/responses`, `wheel/responses`, `video/responses` | `extract_responses` |
| `photometry/neurophotometrics/qc` | `run_neurophotometrics_qc` |
| `photometry/raw/qc` | `run_photometry_qc` |
| `video/times/qc` | `run_video_times_qc` |
| `video/pose/qc` | `run_pose_qc` |

`load_*` is the third tier, and the convenience one: it answers "give me this,
and repair it if it is missing". `scripts/download.py` does not use it at all —
it sequences the `fetch_*` and computation calls itself, so each raw dataset is
fetched once per session and each modality's H5 group is written once. That is
the layering rule for the download path: it fetches its raw materials, holds
them on the session, and processes them, never reading the store mid-build.

Load methods are therefore not pure readers. Each reads in order — the session
attribute if it is present, else the stored product, else fetch and process —
and writes what it built, so a session with an empty H5 fills itself from Alyx.
The first tier is a plain `hasattr` check; there is no staleness comparison and
no rebuild set, because a stored product is whatever the last download run
wrote. `load_raw_photometry` and `load_photometry` stay separate — one method
returning either raw or preprocessed is how an analysis silently runs on the
wrong signal.

`config.store_raw` decides whether the raw datasets fetched from Alyx
(`photometry/raw`, `wheel/raw` and video's three datasets) are products at all.
It ships off, because `data/sessions` is already 14 GB of derived data and the
ONE cache is where raw bytes belong. With it off they are fetched by whatever
needs them and then discarded — nothing writes them, so the load methods go to
Alyx every time. With it on they are saved to the H5 like any other product,
each raw load method (`load_raw_photometry`, `load_raw_wheel`,
`_load_raw_video`) reads its stored copy instead of fetching, and the files
become self-contained. QC is unaffected either way: `raw/qc` is computed data in
its own right and is written whether or not the raw it scored was kept, which is
why `photometry/{region}/raw/qc` can exist in a file that stores no
`photometry/raw`.

QC is a product like any other. `run_neurophotometrics_qc` scores the
neurophotometrics source table into `photometry/neurophotometrics/qc`;
`run_photometry_qc` scores the raw bands into `photometry/{region}/raw/qc`. Both store flat `{metric: value}`
attrs, with the band suffixed into the metric name
(`n_unique_samples_GCaMP`) because QC is split per region but not per band.
`run_photometry_qc` issues two `qc_signals` calls over the same windows —
`config.QC_UNDETRENDED_METRICS` without detrending, the rest with it — and
reduces each metric's windows by `config.QC_SLIDING_AGG`.

Video's QC hangs off the product it characterizes rather than off the modality:
`run_video_times_qc` scores the camera clock into `video/times/qc`
(`length_discrepancy`, `framerate_from_tpts`, and it raises `VideoLengthError`
itself when the discrepancy reaches `config.LENGTH_MISMATCH_THRESHOLD`), and
`run_pose_qc` correlates paw speed against wheel speed into `video/pose/qc`. The latter is the
one cross-modal product — it needs the wheel as well as the pose, which is why
the video block runs after the wheel block in the download pass and a session
with good pose but no wheel fails it with the wheel's own missing-data error.

Manual QC is not a product either, but it is stored. `photometry/{region}/
manual_qc` and `video/manual_qc` hold `config.LP_QC_LABELS`-keyed verdicts from
`config.IBL_QC_VALUES`, set by hand in the viewers — per recording for
photometry, per session for video, because there is one fiber per region and one
camera. `set_manual_qc(field, value, region=None)` validates both arguments
and writes that one verdict directly, rather than through `save_h5`, so what is
persisted does not depend on what the session is holding. `_clear_manual_qc
(modality)` drops the verdicts where the raw data is refetched, because they
were passed on samples that have just been replaced.

The eight `config.VIDEO_QC_COLS` leftCamera labels are **not** a product.
`io.get_video_qc(eid, one)` fetches them from Alyx on every use and nothing
stores them: they change when IBL re-runs its QC, independently of anything in
this repo.
`PhotometrySession.fetch_video_qc` is the per-session wrapper; the pose rollup
fetches the whole set itself and passes it to `collect_pose(video_qc=...)`.

`load_responses(modality, events, window)` serves every modality, dispatching
through `_RESPONSE_MODALITIES` to that modality's preprocessed-signal loader,
result attribute, and the extraction arguments to fall back on when the caller
names none — that fallback is how `load_responses('wheel')` alone cuts
stimOn → choice rather than the photometry window.

The wheel's H5 label is `velocity`: one channel, but the label level stays so
every modality's handlers walk labels the same way, and it names the
preprocessed product rather than the raw position stored underneath it.
`fetch_wheel` fetches ONE's `_ibl_wheel.position` + `.timestamps` — the
irregular encoder samples, kept irregular — and `extract_wheel_velocity` hands
them to `analysis.differentiate(series, fs)`, which interpolates onto the
`WHEEL_FS` grid before differentiating, matching
`brainbox.behavior.wheel.velocity_filtered` at its default corner frequency and
order. Only the velocity is gridded; the position stays raw.
`extract_peak_velocity` is the wheel's fourth product: `analysis.peak_velocity`
reducing the stimulus-onset response matrix to one maximum absolute speed per
trial, stored under `wheel/velocity/peak_velocity` through the frame-data
handler pair because it carries no index of its own. It is the one regressor
the store holds, since it is a property of the wheel signal rather than of a
model design; `build_wheel` computes it after the cut it reduces.

The video modality carries three raw products rather than one, because its three
datasets are fetched by separate ONE calls and fail separately:
`video/times` (the per-frame clock), `video/pose` (LightningPose keypoints) and
`video/motion_energy`. `_RAW_VIDEO_DATASETS` maps each to its session attribute,
its ONE dataset and the exception raised when Alyx lacks it, and one private
`_load_raw_video(product)` serves all three. All three are frame-indexed, with
no time axis of their own, which is what keeps `video/pose` and
`video/motion_energy` free of `video/times` as an input. `_movement_signals` is
the `video/preprocessed` load method: it reads the stored `video/{label}/
preprocessed` channels or resamples the raw onto the `POSE_FS` grid via
`extract_movement_signals`, logging a missing pose or motion energy against its
own product and leaving the other's channels intact. Missing camera times
propagate — nothing can be placed on the session clock without them. The
`motion_energy` group is both the raw product and a movement channel, so it
holds its raw frames beside its `preprocessed` and `responses` subgroups.

`extract_responses(signals, events=..., window=...)` is signal-source agnostic:
it cuts peri-event matrices out of any `label -> pd.Series` mapping and returns
`dict[label, xr.DataArray]` with dims `(event, trial, time)`, leaving the
caller to assign it. Photometry passes `ps.photometry[PREPROCESSED_BAND]`
(labels are brain regions) and assigns `ps.photometry_responses`; video passes
`ps._movement_signals()` with `events=MOVEMENT_EVENTS` (labels are movement
channels) and assigns `ps.movement_responses`. Each movement channel carries
the full event axis; its own response event is selected at read time via
`config.LABEL2EVENT`, and its baseline is the `stimOn_times` cell.

The window is `(t0, t1)`. `t1` is either seconds relative to the event, or the
name of a `ps.trials` column holding each trial's own window end — that second
form is the wheel's cut, `window=(0.0, 'response_times')`. Every trial still
shares one time axis spanning to the longest trial and is NaN-padded beyond its
own endpoint. The `trial` coordinate is the `trial` column of `ps.trials`, not
the row position, so responses stay aligned to their trials after filtering.

Truthiness (`if ps.photometry_responses`) checks whether any label has been
extracted; use `region in ps.photometry_responses` to check a specific one.
`subtract_baseline` and `mask_subsequent_events` operate on one label's
DataArray at a time — pass `ps.photometry_responses[region]`.

Access data attributes directly, not through getters. The class extends
`PhotometrySessionLoader` from `brainbox.io.one`.

HDF5 round-trip: `save_h5()` writes all available data groups, appending by
default; `save_h5(mode='w')` truncates first, which is how the download pass
rewrites a file whole. `load_h5(fpath)` populates all available groups and
adopts `fpath` as `self.filepath`, so a later save writes back to the file the
data came from. Both dispatch to per-group
handler functions via `_SAVE_HANDLERS` / `_LOAD_HANDLERS` registries keyed
by top-level group name (`metadata`, `errors`, `photometry`, `trials`,
`wheel`, `video`). Adding a new top-level group means writing a handler pair
and registering it in both dicts. See README for the on-disk layout.

Beneath the top-level handlers sit five save/load pairs keyed by the **data
structure** they carry rather than by modality. Each is pure: it takes one
`h5py.Group` plus a payload and never touches the session object. The
orchestrator creates the group (`_replace_group`) and loops over regions or
labels, handing each pair one group and one payload.

| pair | payload | used for |
|---|---|---|
| `_save_time_series` / `_load_time_series` | time-indexed `pd.Series` (one signal, dataset `signal`) or `pd.DataFrame` (one dataset per column) | preprocessed photometry, raw wheel position, preprocessed wheel velocity |
| `_save_peri_event_matrix` / `_load_peri_event_matrix` | `xr.DataArray(event, trial, time)` | responses, whether the label is a brain region (`photometry/`), the wheel (`wheel/`) or a movement channel (`video/`) |
| `_save_scalars` / `_load_scalars` | flat `dict[str, float]` stored as group attrs | QC metrics, preprocessing diagnostics |
| `_save_frame_data` / `_load_frame_data` | index-free `np.ndarray` (dataset `values`) or `pd.DataFrame` (one dataset per column) | the three raw video datasets, the wheel's per-trial `peak_velocity` |
| `_save_manual_qc` / `_load_manual_qc` | flat `dict[str, str]` of verdicts stored as group attrs | `photometry/{region}/manual_qc`, `video/manual_qc` |

`_save_frame_data` replaces only the group's datasets, not the group, because
`video/motion_energy` holds this product beside the movement channel's
`preprocessed` and `responses` subgroups. `_save_manual_qc` likewise sets attrs
in place rather than replacing its group, so writing one verdict leaves a
session's other verdicts standing.

`_read_label_products(modality_group, product, read)` reads every
`{label}/{product}` subgroup of one modality in one call, for the loads that
want the whole mapping rather than one label; `_read_label_responses` is the
`responses` case of it. Subgroups that do not carry the named product — the raw
and QC groups sitting beside the labels — drop out on their own, so there is no
skip list to keep in sync.

Two products mix structures and so keep their own pairs.
`video/pose/qc` holds arrays plus a scalar (`_save_pose_xcorr` /
`_load_pose_xcorr`). `trials/performance` holds scalars plus the session's
`contrasts` list, which `_save_scalars` could not carry, so
`_save_performance` writes that one entry as a dataset and hands the rest to
`_save_scalars`.

`trials/performance` is the behavioral scoring of `trials/table`:
`load_performance` reads it or scores the table with `extract_performance` —
one method covering the always-computed metrics and, where the session type has
blocks, the per-block psychometrics — and it is the only parameter
`MIN_BLOCK_LENGTH` reaches. `complete_catalog` joins `fraction_correct` and
`contrasts` onto `_catalog` when the group is built, so
`filter_sessions(min_performance=..., required_contrasts=...)` bites without a
prerequisite call. Neither filter skips itself when its column holds nothing: a
session with no stored performance scores NaN for `fraction_correct` and holds
an empty `contrasts`, and is dropped by both.
`PhotometrySessionGroup.load_performance` remains for a script wanting every
performance metric per session rather than the two the filters read.

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
  Filtered to `_recordings_targetnms` and `_recordings_photometry_qc` (both set
  by `filter_sessions`). Always reflects the current filter and dedup state.

`from_catalog` applies `enforce_schema` and `validate_parallel_lists` before
constructing the object. `filter_sessions(targetnms=TARGETNMS_TO_ANALYZE)` is
the explicit default — pass `targetnms=False` to skip the target filter.

Two filters cut recordings rather than sessions, so they leave no entry in
`_filter_mask` and their removal counts print over recordings: the target-NM
filter and `filter_sessions(photometry_qc=PHOTOMETRY_QC_THRESHOLDS)`. The
latter compares each recording's stored raw QC — `collect_qc` columns, a metric
with the band suffixed on — against the `(comparison, cutoff)` pairs in
`config.PHOTOMETRY_QC_THRESHOLDS`, and keeps only recordings clearing every
one, so a session holds on to the regions that pass and loses the rest. A
recording whose `photometry/{region}/raw/qc` group is absent has no value to
compare and fails. Pass `photometry_qc=False` to skip it — which any group over
sessions whose QC has not been built must do, or it keeps nothing.

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
| `test_wheel.py` | Wheel raw, preprocessed and response products |
| `test_download.py` | Catalog fixups, the per-modality build pass, the download CLI |
| `test_rebuild_responses.py` | The response re-cut, its Alyx fallback, its CLI |

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
