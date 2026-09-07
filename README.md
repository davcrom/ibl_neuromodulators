# IBL Neuromodulators

Fiber photometry analysis pipeline for the IBL neuromodulators project. Ingests session metadata and raw signals from Alyx/ONE, applies QC, preprocesses photometry signals, and extracts peri-event neural responses. The signals are genetically encoded calcium indicators (GCaMP), expressed in neuromodulatory populations by crossing a Cre-dependent reporter line to DAT-Cre, DbH-Cre, ChAT-Cre, or SERT-Cre, so each recording measures calcium activity in a genetically defined cell population.

## Setup

This project requires the [IBL unified environment](https://docs.internationalbrainlab.org/02_installation.html#uv-pip-instructions)
with specific development branches of `ibllib` and `ibl-photometry`.

### Fresh install

```bash
# 1. Create and activate a virtual environment
uv venv .venv --prompt ibl --python 3.13
source .venv/bin/activate

# 2. Install IBL packages from the required branches
uv pip install git+https://github.com/int-brain-lab/ibllib@photometry-integration
uv pip install git+https://github.com/int-brain-lab/ibl-photometry@develop

# 3. Install iblnm (editable) with dev tools
uv pip install -e ".[dev]"
```

### Existing IBL environment

If you already have the IBL environment with the correct `ibllib` and
`ibl-photometry` branches:

```bash
uv pip install -e ".[dev]"
```

This installs `iblnm` plus any missing core dependencies (`xarray`,
`statsmodels`, `cca-zoo`, etc.) and dev tools (`pytest`, `ruff`).

### Verify

```bash
pytest                     # run tests
ruff check .               # lint
```

**Docs**: [IBL](https://docs.internationalbrainlab.org/) · [ONE API](https://int-brain-lab.github.io/ONE/)

---

## Pipeline

One script fetches and builds everything. Each session's errors are written
into its H5 `/errors` group, one group per product; the script prints an error
summary rather than writing a separate log file.

```
download.py → data/sessions/{eid}.h5  (every product, rebuilt every run)
     ↓
metadata/sessions.pqt  (the catalog)
```

```bash
python scripts/download.py                          # build the whole store
python scripts/download.py --workers 4              # in parallel
python scripts/download.py --session-type biased    # one session type
python scripts/download.py --target-NM VTA-DA SNc-DA # one target neuromodulator
```

**Phase one — catalog.** Queries the `ibl_fibrephotometry` project on Alyx,
writes each new session's metadata (subject info, brain regions, hemisphere,
dataset availability) into its H5 file, then runs the fixups that need every
session at once: filling empty brain regions from the subject's other sessions
and from `metadata/fibers.csv`, normalizing region names, deriving `target_NM`,
and ranking each session within its subject (`day_n`, `session_n`).

**Phase two — products.** Walks each session one modality at a time —
`trials`, `photometry`, `wheel`, `video` — fetching that modality's raw
datasets once, computing every product made from them while they are still in
memory, and rewriting the session's H5 file whole at the end. Every product is
built on every run: nothing is read back from the file, skipped, or compared
against what is already stored. Re-running the script re-downloads all 5525
sessions from Alyx, which is the price of never having to reason about which
stored product was derived from which version of the code.

The order inside a block is the dependency order, and a failure abandons the
rest of its block, so nothing is ever cut from a signal that was never built.
The order *between* blocks matters for the same reason and is the only
dependency declaration there is: the response cuts read the trials table the
first block fetched, and the video block's pose QC reads the wheel velocity the
block above it computed.

A failure logs one error against its modality and the next block still runs.
Some checks are non-fatal by design: a corrupted block structure is logged and
then repaired, and a session missing some event times still has its responses
cut on the events that remain.

The three flags all narrow which sessions run, never what is built within one.
`--workers` sets the number of parallel worker processes; `--session-type` and
`--target-NM` restrict the pass to the named `config.SESSION_TYPES` and
`config.VALID_TARGETNMS`.

### Re-cutting the responses alone

`scripts/rebuild_responses.py` is the exception to the all-or-nothing build. It
re-cuts `photometry/{region}/responses` for every stored session and touches
nothing else, which is what a change to `config.RESPONSE_EVENTS`,
`config.RESPONSE_WINDOW` or the cut itself calls for — a full `download.py` run
would rebuild every other product and re-download the raw data to no purpose.

```bash
python scripts/rebuild_responses.py                  # every stored session
python scripts/rebuild_responses.py --workers 4      # in parallel
python scripts/rebuild_responses.py --session-type biased
python scripts/rebuild_responses.py --target-NM LC-NE
```

It reads its session list from the store's own `metadata` groups rather than
from Alyx or `sessions.pqt`, and each session's trials table and preprocessed
band from its H5 file. The stale event's dataset goes when the group is
replaced, so a rebuild after an event was renamed leaves no orphan behind. A
session whose file holds no preprocessed band is the one case that reaches
Alyx: `load_photometry` fetches its raw bands and preprocesses them as the
download pass would. The three flags narrow which sessions run, as they do
there.

**Analysis scripts build nothing in bulk.** A session missing a product builds
it through that session's `load_*` when the analysis reaches it. With the
download all-or-nothing, that should not happen: an analysis running over a
freshly built store has its data or has an error explaining why not.

## Rollups

The store is the source for every analysis. A flat view of it — one row per
session or per recording — is a rollup, and each script that wants one writes
its own from the group methods that build them: `collect_errors`, `collect_qc`
and `collect_pose` on `PhotometrySessionGroup`. There is no shared rollup
script, so a script's flat table is written where it is read and answers to the
filters that script set.

`scripts/download.py` writes `metadata/sessions.pqt`, the session catalog, as
part of its catalog phase.

Every rollup reads through `PhotometrySessionGroup`, so what lands in a file is
what the group's filters admit. The pose table is the slow one: the eight
leftCamera QC labels live only on Alyx and nothing here stores them, so they
cost one REST call per session.

### `dataset_overview.py` — Session coverage figures

Joins `sessions.pqt`, `qc_photometry.pqt`, `performance.pqt`, and the errors scanned from the H5 `/errors` groups. Produces session-by-session overview matrices at each processing stage, plus barplots of complete recordings per brain target and per mouse. Writes the unified `metadata/errors.pqt`.

**Output**: `figures/dataset_overview/`

### Analysis scripts

| Script | Purpose |
|---|---|
| `responses.py` | Trial-level response magnitudes, LMM fits, response feature vectors, similarity, decoding, and movement-variable encoding (descriptive, LOSO ΔR², per-contrast timing slopes) |
| `task_encoding.py` | Per-session GLM encoding decomposed via PCA/ICA, per-cohort CCA |
| `task_performance.py` | Learning curves, psychometric trajectories per target |
| `qc_overview.py` | QC metric distributions (histograms, violins, PCA, temporal trends) |
| `video.py` | Video QC pipeline (timestamps, dropped frames, pin state) |
| `session_viewer.py` | Interactive single-session viewer (raw + preprocessed + PSTHs) |
| `example_session.py` | Annotated example of loading and plotting a session |

---

## PhotometrySession

`PhotometrySession` wraps a row from `sessions.pqt` and provides methods for loading, validating, preprocessing, and extracting responses. It extends `PhotometrySessionLoader` from `brainbox.io.one`.

Data attributes are lazy-loaded: `trials`, `photometry`, `photometry_responses`,
`wheel_position`, `wheel_velocity` and the rest do not exist on a freshly
constructed session, and are created by the explicit method call that fills
them. Reading one before then raises `AttributeError` rather than handing back
an empty value that looks like data.

### Loading from ONE

```python
import pandas as pd
from one.api import ONE
from iblnm.config import SESSIONS_FPATH
from iblnm.data import PhotometrySession

one = ONE()
df_sessions = pd.read_parquet(SESSIONS_FPATH)
session_row = df_sessions.iloc[0]

ps = PhotometrySession(session_row, one=one)
ps.load_trials()          # → ps.trials (adds trial, stim_side, contrast, signed_contrast)
ps.load_performance()     # → ps.performance, behavioral scalars from the trials
ps.load_photometry()      # → ps.photometry['GCaMP_preprocessed']
ps.load_raw_photometry()  # → ps.photometry: {'GCaMP': ..., 'Isosbestic': ...}
```

`load_photometry` returns the *preprocessed* signal and `load_raw_photometry`
the raw bands. They are separate methods on purpose: one method returning
either would let an analysis run on raw data without saying so.

### Loading from HDF5

If the pipeline has already run, load preprocessed data from disk:

```python
from iblnm.config import SESSIONS_H5_DIR

ps = PhotometrySession(session_row, one=one)
ps.load_h5(SESSIONS_H5_DIR / f'{ps.eid}.h5')
# → ps.photometry['GCaMP_preprocessed'], ps.trials, ps.photometry_responses,
#   ps.movement_responses, ps.wheel_position, ps.wheel_velocity,
#   ps.wheel_responses, ps.wheel_peak_velocity
```

### Validation

Each method raises a typed exception on failure. In scripts, pass an `exlog` list to log errors instead of raising (see Error Handling below).

```python
ps.validate_trials_in_photometry_time()  # raises TrialsNotInPhotometryTime
ps.validate_n_trials()                   # raises InsufficientTrials
ps.validate_event_completeness()         # raises IncompleteEventTimes
ps.validate_block_structure()            # raises BlockStructureBug
```

### QC

```python
ps.load_neurophotometrics_qc()  # → ps.neurophotometrics_qc, a flat
                                #   {metric: value} for the source table
ps.validate_qc()                # raises QCValidationError on a non-zero
                                #   n_band_inversions or n_early_samples
ps.load_photometry_qc()         # → ps.photometry_qc, sliding-window signal
                                #   quality per region
```

QC is a product like any other: each load method reads the stored group when
there is one and scores the signal when there is not, writing the result.
`run_neurophotometrics_qc` and `run_photometry_qc` are the scoring half,
callable directly when the raw is already on the session and a rescore is what
you want.

`ps.photometry_qc` maps a brain region to a flat `{metric: value}` dict. QC is
split per region but not per band, so the band is suffixed into the metric name:
`n_unique_samples_GCaMP`, `n_unique_samples_Isosbestic`.

`run_photometry_qc` scores each metric over 120 s windows and reduces them with
`config.QC_SLIDING_AGG` — the 10th percentile for `n_unique_samples`, so a
recording is judged by its worst windows rather than its average, and the mean
for the rest. The metrics in `config.QC_UNDETRENDED_METRICS` are scored in their
own pass with detrending off; detrending leaves every sample of a window a
distinct float, which would pin `n_unique_samples` at 1.0 on any signal.

### Preprocessing and response extraction

```python
from iblnm.config import RESPONSE_EVENTS

ps.load_photometry()  # bleach → isosbestic → resample to 30 Hz → zscore
                      # → ps.photometry['GCaMP_preprocessed'], written into
                      #   photometry/{region}/preprocessed

ps.load_responses('photometry', events=RESPONSE_EVENTS)
# → ps.photometry_responses: dict[str, xr.DataArray] keyed by brain region,
#   each DataArray has dims (event, trial, time). Read from H5 when stored,
#   cut and written when not.

ps.save_h5()  # saves all available data groups
```

The load methods are not pure readers. Each reads in order — the attribute the
session already holds, else the stored product, else fetch and build — and
writes what it built, so `load_photometry` on a session with nothing cached
fetches from Alyx, preprocesses, and leaves the result on disk. A second call
costs nothing and never reads back what the first one wrote.

Underneath each `load_*` sit a `fetch_*` that only queries Alyx and an
`extract_*` or `run_*_qc` that only computes — `extract_preprocessed_photometry`
here, `extract_wheel_velocity` for the wheel, `extract_movement_signals` for the
video channels. `scripts/download.py` calls that pair directly and never calls a
`load_*` at all: the build fetches its raw materials, holds them on the session,
and processes them, rather than reading the store halfway through writing it. A
session at a prompt is better served by `load_*`.

The video modality follows the same shape, with three raw products instead of
one because its three datasets are fetched — and fail — independently:

```python
ps.load_camera_times()   # video/times, the per-frame clock; required
ps.load_pose()           # video/pose, the LightningPose keypoints
ps.load_motion_energy()  # video/motion_energy, the per-frame ROI scalar

ps.load_responses('video')
# → dict keyed by movement channel ('paw', 'nose', 'tongue_speed',
#   'tongue_likelihood', 'motion_energy'), same (event, trial, time) dims
```

A missing pose or motion energy is logged against its own product and leaves
the other channels intact; only missing camera times blocks the modality.
Underneath, `extract_responses` is the cutting engine that `load_responses`
calls, and it takes any `label -> pd.Series` mapping — for video that mapping
is the `video/{label}/preprocessed` channels, resampled to `POSE_FS`.

The window end may instead name a trials column, giving each trial its own
endpoint — the cut the wheel needs, from stimulus onset to that trial's
choice:

```python
ps.load_responses('wheel')   # events=['stimOn_times'],
                             # window=(0.0, 'response_times')
# → {'velocity': DataArray}; trials share one time axis spanning to the
#   longest trial, each NaN-padded from its own choice onward

ps.load_peak_velocity()
# → np.ndarray, one maximum absolute speed per trial, reduced from that
#   matrix and stored beside it; NaN where a trial has no wheel samples
```

`peak_velocity` is the only regressor the store holds. It is a property of the
wheel signal, whereas coding, log transforms and trial selection follow a model
design and would go stale when the design changes.

Every channel carries the full event axis. A channel's own response event is
`config.LABEL2EVENT[label]`; its baseline is the `stimOn_times` cell over
`BASELINE_WINDOW`. Both are read-time selections, not separate stored arrays.

### Working with responses

Response transforms operate on a single region's DataArray at a time:

```python
region_responses = ps.photometry_responses['VTA']  # dims: (event, trial, time)

# Baseline subtraction (mean of [-0.1, 0] window)
responses = ps.subtract_baseline(region_responses)

# Mask time points after the next event in a trial sequence
responses = ps.mask_subsequent_events(
    region_responses,
    event_order=['stimOn_times', 'firstMovement_times', 'feedback_times']
)
```

### Task performance

```python
perf = ps.extract_performance()
# {'n_trials': 642, 'contrasts': [0.0, 0.0625, ...], 'fraction_correct': 0.81,
#  'fraction_correct_easy': 0.94, 'nogo_fraction': 0.02, 'psych_50_bias': -1.2,
#  'psych_50_threshold': 8.4, ...}
# biased/ephys sessions also carry psych_20_*, psych_80_* and bias_shift

fit = ps.fit_psychometric()           # {bias, threshold, lapse_left, lapse_right, r_squared, n_trials}
```

---

## PhotometrySessionGroup

`PhotometrySessionGroup` is the central class for all multi-session analyses. It manages session-level filtering and recording-level explosion internally.

### Design principles

- **Constructor takes session-level DataFrames.** List columns (`brain_region`, `hemisphere`, `target_NM`) are kept intact. Explosion to one-row-per-recording happens via `explode_recordings()`.
- **`filter_sessions` filters at the session level** by session type, excluded subjects, QC error types, and target-NM values. Sessions where none of their target_NM entries match are dropped. The target-NM and `photometry_qc` filters additionally cut individual recordings, leaving the rest of their session in scope.
- **`explode_recordings` produces recording-level rows** from the filtered sessions, trimming to only valid target_NM entries and adding `fiber_idx`.
- **`from_catalog` handles the full pipeline**: load parquet, validate parallel lists, filter sessions, explode recordings.
- **Lazy analysis attributes.** `events`, `response_features`, `similarity_matrix`, and `decoder` start as `None` and are populated by explicit method calls.
- **Iterable.** `for rec, ps in group` yields `(recording_row, PhotometrySession)` pairs. Sessions are cached by eid so loading an H5 once serves all regions.

### Usage

```python
from iblnm.config import SESSIONS_FPATH
from iblnm.data import PhotometrySessionGroup
from iblnm.io import _get_default_connection

one = _get_default_connection()

# Load, filter, and explode in one step
group = PhotometrySessionGroup.from_catalog(
    SESSIONS_FPATH, one=one,
    session_types=('biased', 'ephys'),
)
```

### Analysis methods

```python
# Trial-level response magnitudes (one row per recording × event × trial)
group.get_response_magnitudes()
# → group.response_magnitudes (DataFrame)

# Response feature vectors (one row per recording, columns = condition labels)
group.get_response_features(nan_handling='drop_features')
# → group.response_features (DataFrame indexed by (eid, target_NM))

# Pairwise cosine similarity between recordings
group.response_similarity_matrix()
# → group.similarity_matrix (DataFrame)

# Decode target-NM from response vectors (logistic regression with LOSO CV)
group.decode_target()
# → group.decoder (TargetNMDecoder with .accuracy, .confusion, .coefficients, .contributions)
```

### Filtering and subsetting

```python
# fraction_correct and contrasts come from each session's trials/performance
# product; without this call min_performance and required_contrasts have no
# column to read and skip themselves, keeping sessions they should drop.
group.load_performance()

# Standard filters (all parameters optional, default to config values)
group.filter_sessions(
    session_types=('biased', 'ephys'),
    exclude_subjects=['excluded_mouse'],
    qc_blockers={'MissingRawData', 'QCValidationError'},
    targetnms=['VTA-DA', 'DR-5HT'],
    # Recording-level: each recording's stored raw QC must clear every
    # threshold, so a session keeps its passing regions and loses the rest.
    # A recording with no stored QC has nothing to compare and fails.
    photometry_qc={'n_unique_samples_GCaMP': ('>=', 0.005),
                   'n_unique_samples_Isosbestic': ('>=', 0.005)},
)

# Boolean mask
group.filter(group.recordings['NM'] == 'DA')

# Indexing
rec, ps = group[0]  # first recording
```

### Iteration

```python
for rec, ps in group:
    # rec: pd.Series (recording metadata)
    # ps: PhotometrySession (cached by eid, loads H5 on first access)
    ps.load_h5(h5_path)
    ...
```

---

## Error Handling

The `@exception_logger` decorator is the central pattern for batch processing. Functions decorated with it accept an optional `exlog` parameter:

- **Without `exlog`**: exceptions propagate normally (used in tests)
- **With `exlog=[]`**: exceptions are caught, logged as dicts, and the original row is returned so the pipeline continues

```python
from iblnm.validation import exception_logger, InvalidBrainRegion

@exception_logger
def validate_brain_region(session):
    ...
    raise InvalidBrainRegion(...)

# In scripts — errors logged, pipeline continues:
error_log = []
df = df.apply(validate_brain_region, axis='columns', exlog=error_log)

# In tests — errors raised:
with pytest.raises(InvalidBrainRegion):
    validate_brain_region(bad_session)
```

Error log entries follow the schema: `['eid', 'error_type', 'error_message',
'traceback', 'product']`. `product` names the stored product whose
build raised — `ps.log_error(e, product='video/pose')` — and decides which
`errors/{product}` group the entry is saved under. It is None for failures not
attributable to a single product, which land in the `errors/` root.

Downstream scripts read each session's errors from its H5 `/errors` group —
`from_catalog(..., h5_dir=...)` scans them into a `logged_errors` column (via
`group.collect_session_errors()`) — and filter sessions based on which error
types are present, rather than re-validating.

---

## Data Files

### `metadata/sessions.pqt` — one row per session

| Column | Type | Description |
|---|---|---|
| `eid` | str | Alyx session UUID |
| `subject` | str | Mouse name |
| `start_time` | str | ISO 8601 session start |
| `session_type` | str | training / biased / ephys / habituation / histology |
| `NM` | str | Neuromodulator: DA, 5HT, NE, ACh |
| `brain_region` | list[str] | Recording targets, e.g. `['VTA', 'SNc']` |
| `hemisphere` | list[str] | Hemisphere per region, e.g. `['l', 'r']` |
| `target_NM` | list[str] | Combined labels, e.g. `['VTA-DA', 'SNc-DA']` |
| `lab` | str | Recording lab |
| `day_n` | int | Days since subject's first session |
| `session_n` | float | Session index (dense rank within subject) |
| `session_length` | float | Duration in seconds |
| `strain`, `line`, `genotype` | str | Mouse genetics |
| `datasets` | list[str] | ALF dataset paths available on ONE |

`brain_region`, `hemisphere`, and `target_NM` are parallel lists that must always have matching lengths. To get one row per recording, explode all three together: `df.explode(['brain_region', 'hemisphere', 'target_NM'])`.

### `results/responses/response_magnitudes.parquet` — one row per (recording x event x trial)

Recording keys + response magnitude only. Trial-level task and movement
predictors live in `trial_regressors.parquet` (join on `eid`, `trial`).

| Column | Type | Description |
|---|---|---|
| `eid` | str | Session UUID |
| `subject` | str | Mouse name |
| `session_type` | str | biased / ephys |
| `NM` | str | Neuromodulator |
| `target_NM` | str | Target-NM label |
| `brain_region` | str | Recording target |
| `hemisphere` | str | l / r |
| `event` | str | stimOn_times / firstMovement_times / feedback_times |
| `trial` | int | Trial index |
| `response` | float | Mean response in early window (0.1-0.35s) |
| `masked_fraction` | float | Fraction of that window masked at the next event |

### `results/responses/masking_diagnostics.parquet` — one row per (target_NM x event x contrast x feedbackType)

How much of the response window each trial type kept. Masking removes the
samples after the next event, so it takes more of the window on fast trials,
and fast trials are more common at high contrast — read this beside any
contrast-dependent result. Counted over the modeling trials, including the
ones whose window was masked end to end and whose magnitude is therefore NaN.

| Column | Type | Description |
|---|---|---|
| `target_NM` | str | Target-NM label |
| `event` | str | stimOnTrigger_times / feedback_times |
| `contrast` | float | Unsigned stimulus contrast |
| `feedbackType` | float | 1 reward / -1 punishment |
| `n_trials` | int | Trials in the cell |
| `masked_fraction_mean` | float | Mean proportion of the window masked |
| `pct_any_masked` | float | % of trials with any masked sample |
| `pct_fully_masked` | float | % of trials with every sample masked |
| `pct_move_in_window` | float | % of trials whose `reaction_time` is inside the window |

### `results/responses/trial_regressors.parquet` — one row per (eid x trial)

Per-trial task and movement predictors. Join to `response_magnitudes.parquet`
on `eid`, `trial`.

| Column | Type | Description |
|---|---|---|
| `eid` | str | Session UUID |
| `trial` | int | Trial index |
| `signed_contrast` | float | Signed stimulus contrast |
| `contrast` | float | Unsigned stimulus contrast |
| `stim_side` | str | left / right |
| `choice` | float | -1 left / 0 no-go / 1 right |
| `feedbackType` | float | 1 reward / -1 punishment |
| `probabilityLeft` | float | Block probability |
| `reaction_time` | float | firstMovement - stimOn (seconds) |
| `movement_time` | float | feedback - firstMovement (seconds) |
| `response_time` | float | feedback - stimOn (seconds) |
| `peak_velocity` | float | Max abs wheel velocity per trial |

### `data/qc_photometry.pqt` — one row per (session, brain region)

QC is stored per region but not per band, so the band it scored is suffixed
into each metric name: `n_unique_samples_GCaMP`, `n_unique_samples_Isosbestic`.

| Column | Type | Description |
|---|---|---|
| `eid` | str | Session UUID |
| `brain_region` | str | Single recording target |
| `n_unique_samples_{band}` | float | Fraction of unique values |
| `n_band_inversions_{band}` | int | Samples where GCaMP < Isosbestic (> 0 fatal) |
| `n_early_samples_{band}` | int | Samples before recording start (> 0 fatal) |
| `ar_score_{band}` | float | AR(1) autocorrelation coefficient |
| `median_absolute_deviance_{band}` | float | MAD of signal |
| `percentile_asymmetry_{band}` | float | (p75-p50) / (p50-p25) skewness proxy |
| `percentile_distance_{band}` | float | (p75-p25) / median spread proxy |
| `bleaching_tau` | float | Photobleaching time constant in seconds (GCaMP only) |
| `iso_correlation` | float | R² between GCaMP and Isosbestic (GCaMP only) |

### `data/performance.pqt` — one row per session

| Column | Type | Description |
|---|---|---|
| `eid` | str | Session UUID |
| `n_trials` | int | Total trial count |
| `fraction_correct` | float | Overall fraction correct |
| `fraction_correct_easy` | float | Fraction correct on 100% contrast |
| `nogo_fraction` | float | Fraction of no-go trials |
| `psych_50_{param}` | float | Psychometric fit on 50/50 block (bias, threshold, lapse_left, lapse_right, r_squared) |
| `psych_80_{param}` | float | 80% left block fit (biased/ephys only) |
| `psych_20_{param}` | float | 20% left block fit (biased/ephys only) |
| `bias_shift` | float | psych_80_bias - psych_20_bias |

### HDF5: `data/sessions/{eid}.h5`

File is organized into top-level groups (`metadata`, `errors`, `photometry`,
`trials`, `wheel`, `video`) that mirror the `PhotometrySession` attributes.
Each group is read/written by a dedicated handler pair registered in
`_SAVE_HANDLERS` and `_LOAD_HANDLERS` in `iblnm/data.py`. Photometry data is
organized per brain region so single-region loads do not require reading the
full file; video data follows the same `{label}/responses/` layout per
movement channel.

```
{eid}.h5
├── metadata/
│   ├── attrs: eid, subject, start_time, number, task_protocol,
│   │          session_type, lab, NM, strain, line, end_time,
│   │          session_length, day_n, session_n, url
│   └── datasets (list-valued fields):
│       genotype, projects, users, brain_region, hemisphere,
│       target_NM, datasets
│
├── errors/                          # mirrors the product tree; a group holds
│   │                                # the last build attempt of its product,
│   │                                # rewritten whole, never accumulated
│   ├── eid, error_type, error_message, traceback, product
│   │                     str[M]     # failures logged with no product
│   └── {product}/                   # e.g. photometry/raw, video/pose
│       └── eid, error_type, error_message, traceback, product
│                         str[M]
│
├── photometry/
│   └── {brain_region}/
│       ├── raw/                     # the bands as fetched from Alyx, kept only
│       │   ├── {band}/              # when config.store_raw is on
│       │   │   ├── times     float64 (R,)   that band's own sample times, since
│       │   │   └── signal    float64 (R,)   acquisition interleaves the bands
│       │   └── qc/                  # scored from the raw and stored either way
│       │       └── attrs: one per QC metric, the band suffixed into its name
│       │                  (n_unique_samples_GCaMP, ...)
│       │
│       ├── preprocessed/
│       │   ├── times     float64 (N,)    sample times at 30 Hz
│       │   ├── signal    float64 (N,)    z-scored, isosbestic-corrected GCaMP
│       │   └── attrs: bleaching_tau, iso_correlation — diagnostics of this
│       │              preprocessing run, so qc/ depends only on raw/
│       │
│       ├── responses/
│       │   ├── times                float64 (W,)     time relative to event
│       │   ├── trials               int64   (T,)     trial indices
│       │   ├── stimOn_times         float64 (T, W)
│       │   ├── firstMovement_times  float64 (T, W)
│       │   └── feedback_times       float64 (T, W)
│       │
│       └── manual_qc/               # verdicts set by hand, one per recording
│           └── attrs: qc_lp, qc_movement, qc_timing
│
├── trials/
│   └── table/                      # the ONE trials table verbatim, plus the
│       │                           # four columns load_trials derives
│       ├── intervals_0          float64 (T,)
│       ├── intervals_1          float64 (T,)
│       ├── goCue_times          float64 (T,)
│       ├── stimOn_times         float64 (T,)
│       ├── firstMovement_times  float64 (T,)
│       ├── response_times       float64 (T,)
│       ├── feedback_times       float64 (T,)
│       ├── choice               float64 (T,)   -1 left, 0 no-go, 1 right
│       ├── feedbackType         float64 (T,)   1 reward, -1 punishment
│       ├── contrastLeft         float64 (T,)   fraction, NaN if right stimulus
│       ├── contrastRight        float64 (T,)   fraction, NaN if left stimulus
│       ├── rewardVolume         float64 (T,)   µL
│       ├── probabilityLeft      float64 (T,)   0.2, 0.5, or 0.8
│       ├── signed_contrast      float64 (T,)   percent, negative = left
│       ├── contrast             float64 (T,)   percent, unsigned
│       ├── stim_side            str     (T,)   'left' or 'right'
│       └── trial                int64   (T,)   raw ONE index; trial identity
│   └── performance/                # behavioral scalars scored from the table
│       ├── contrasts          float64 (C,)   the sorted levels presented, percent
│       └── attrs: n_trials, fraction_correct, fraction_correct_easy,
│                  nogo_fraction, psych_50_* — and psych_20_*, psych_80_*,
│                  bias_shift where the session has blocks
│
├── wheel/
│   └── velocity/                    the wheel's one label, named for the
│       ├── raw/                     preprocessed signal, not the raw position;
│       │                            kept only when config.store_raw is on
│       │   ├── times     float64 (E,)   irregular encoder timestamps
│       │   └── signal    float64 (E,)   wheel position, radians
│       ├── preprocessed/
│       │   ├── times     float64 (V,)   uniform grid at WHEEL_FS
│       │   └── signal    float64 (V,)   velocity, radians per second
│       ├── responses/
│       │   ├── times          float64 (Wf,)  0 → longest trial's feedback
│       │   ├── trials         int64   (T,)
│       │   └── stimOn_times   float64 (T, Wf)  NaN past each trial's feedback
│       └── peak_velocity/
│           └── values   float64 (T,)   max abs velocity per trial, reduced
│                                       from the responses matrix; NaN where
│                                       the trial has no wheel samples
│
└── video/
    ├── manual_qc/                   verdicts set by hand, one set per session
    │   └── attrs: qc_lp, qc_movement, qc_timing
    │
    ├── times/                       raw, one group per independently
    │   ├── values   float64 (F,)    fetched dataset, each kept only
    │   │                            when config.store_raw is on
    │   └── qc/
    │       └── attrs: length_discrepancy, framerate_from_tpts
    ├── pose/
    │   ├── {keypoint}_x           float64 (F,)   LightningPose columns
    │   ├── {keypoint}_y           float64 (F,)
    │   ├── {keypoint}_likelihood  float64 (F,)
    │   └── qc/                      paw–wheel timing diagnostic; the one
    │       ├── functions   float64 (3, L)   cross-modal product, needing the
    │       ├── lags        float64 (L,)     wheel velocity as well as the pose
    │       ├── peak_lags   float64 (3,)
    │       └── attrs: drift
    │
    └── {movement_channel}/          paw, nose, tongue_speed,
        ├── preprocessed/            tongue_likelihood, motion_energy.
        │   ├── times     float64 (P,)   uniform grid at POSE_FS
        │   └── signal    float64 (P,)
        └── responses/
            ├── times                float64 (W,)
            ├── trials               int64   (T,)
            ├── stimOn_times         float64 (T, W)
            ├── firstMovement_times  float64 (T, W)
            └── feedback_times       float64 (T, W)
                                     The motion_energy channel's group also
                                     holds its raw `values` dataset, since the
                                     raw product and the channel share a name.
```

The `{movement_channel}/` groups are the one part of the tree `download.py` does
not write. The video block stores the keypoints, the motion energy and the two
QC results; the derived movement signals and their responses are built on demand
by `load_responses('video')` and written then.

`config.store_raw` decides whether the raw datasets fetched from Alyx —
`photometry/{region}/raw`, `wheel/velocity/raw`, and video's `times/`, `pose/`
and `motion_energy/` — are products at all. It ships off, because
`data/sessions` is already 14 GB of derived data and the ONE cache is where raw
bytes belong. With it off nothing writes those groups and the load methods fetch
from Alyx every time; turning it on makes the files self-contained and the same
load methods read their stored copy instead. The QC scored from the raw is
stored either way — it is computed data in its own right — which is why
`photometry/{region}/raw/qc` and `video/pose/qc` can sit under a group that
holds no raw.

The `manual_qc/` groups hold verdicts set by hand in the viewers, from the IBL
vocabulary `CRITICAL`/`FAIL`/`WARNING`/`PASS` (`config.IBL_QC_VALUES`), keyed by
`config.LP_QC_LABELS`. Photometry is scored per recording and video per session,
because there is one fiber per region but one camera. Writing
them goes through `PhotometrySession.set_manual_qc(field, value, region=None)`,
which validates both arguments and writes that one verdict straight to the file.
Re-downloading a modality's raw data drops them, since the verdict was passed on
frames or samples that have just been replaced.

The eight leftCamera extended-QC labels from Alyx are deliberately absent: they
change when IBL re-runs its QC, independently of anything in this repo.
`io.get_video_qc(eid, one)` fetches them on demand instead, one REST call per
session.

`N` = samples at 30 Hz, `T` = trial count, `W` = response window samples,
`M` = logged error count, `L` = cross-correlation lag count, `E` = encoder
sample count, `V` = wheel samples at 100 Hz, `Wf` = samples from stimulus onset
to the longest trial's feedback at 100 Hz, `F` = camera frame count, `P` =
movement samples at POSE_FS.

`save_h5(groups=...)` and `load_h5(groups=...)` accept the top-level group
names (`'metadata'`, `'errors'`, `'photometry'`, `'trials'`, `'wheel'`,
`'video'`) to restrict which handlers run. Omit `groups` to process everything
present.

---

## Project Structure

```
iblnm/                      # Core package (generic, reusable)
  config.py                 # Paths, constants, QC thresholds, color mappings
  data.py                   # PhotometrySession, PhotometrySessionGroup
  io.py                     # Alyx/ONE queries
  task.py                   # Task performance (psychometrics, block validation)
  analysis.py               # Signal processing, LMMs, similarity, decoding
  validation.py             # Custom exceptions, @exception_logger, validate_* functions
  util.py                   # Pandas utilities, parquet I/O, schema enforcement
  vis.py                    # Plotting functions
  gui.py                    # Interactive session viewer widget

scripts/                    # Pipeline stages and analysis scripts
tests/                      # pytest (synthetic fixtures, no Alyx calls)

# Generated outputs (gitignored)
metadata/                   # sessions.pqt, errors.pqt (fibers.csv is tracked)
data/                       # qc_photometry.pqt, performance.pqt, sessions/*.h5
results/                    # Analysis outputs (responses/, task_encoding/)
figures/                    # Output plots
specs/                      # Design specs (gitignored, local working docs)
```
