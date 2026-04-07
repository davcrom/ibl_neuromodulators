# IBL Neuromodulators

Fiber photometry analysis pipeline for the IBL neuromodulators project. Ingests session metadata and raw signals from Alyx/ONE, applies QC, preprocesses photometry signals, and extracts peri-event neural responses.

## Setup

```bash
uv pip install -e .        # editable install (venv: ~/.venv/ibl)
pytest                     # run tests
ruff check .               # lint
```

**Dependencies**: `one-api`, `brainbox`, `iblphotometry` (IBL ecosystem); `pandas`, `xarray`, `numpy`, `scipy`, `matplotlib`, `h5py`.

**Docs**: [IBL](https://docs.internationalbrainlab.org/) · [ONE API](https://int-brain-lab.github.io/ONE/)

---

## Pipeline

Scripts run in order. Each produces a parquet error log alongside its outputs.

```
query_database.py → photometry.py → task.py → wheel.py → dataset_overview.py
       ↓                  ↓             ↓          ↓              ↓
  sessions.pqt      {eid}.h5 +    performance  {eid}.h5      figures +
                  qc_photometry      .pqt     (wheel group)  errors.pqt
```

Run the full pipeline or resume from a specific stage:

```bash
python scripts/run_pipeline.py                    # all stages
python scripts/run_pipeline.py --from photometry  # resume from photometry
python scripts/run_pipeline.py --only task        # single stage
python scripts/run_pipeline.py --skip-errors      # continue past failures
```

### Stage 1: `query_database.py` — Session metadata

Queries the `ibl_fibrephotometry` project on Alyx, enriches each session with subject info (strain, line, neuromodulator), brain regions, hemisphere, and dataset availability. Validates all metadata fields.

```bash
python scripts/query_database.py                 # incremental update
python scripts/query_database.py --redownload    # re-download everything
python scripts/query_database.py --extended-qc   # also fetch Alyx extended QC
```

**Output**: `metadata/sessions.pqt`, `metadata/query_database_log.pqt`

### Stage 2: `photometry.py` — QC, preprocessing, response extraction

Processes each session through a tiered pipeline:

1. Load trials and photometry from ONE
2. Validate that trials fall within the photometry recording window (fatal)
3. Raw QC: check for band inversions and early samples (fatal)
4. Sliding QC: compute signal quality metrics (fatal)
5. Preprocess: bleach correction → isosbestic regression → z-score → resample to 30 Hz
6. Extract peri-event responses for `stimOn_times`, `firstMovement_times`, `feedback_times`
7. Save signal and responses to HDF5

**Output**: `data/sessions/{eid}.h5`, `data/qc_photometry.pqt`, `metadata/photometry_log.pqt`

### Stage 3: `task.py` — Task performance

Computes per-session metrics: fraction correct, no-go fraction, psychometric function parameters (bias, threshold, lapses) for the 50/50 block, and per-block psychometrics and bias shift for biased/ephys sessions.

**Output**: `data/performance.pqt`, `metadata/task_log.pqt`

### Stage 4: `wheel.py` — Per-trial wheel velocity

Extracts wheel velocity for each trial (stimOn → feedback), NaN-padded to the longest trial. Appends a `wheel/` group to existing HDF5 files.

**Output**: appended `data/sessions/{eid}.h5`, `metadata/wheel_log.pqt`

### Stage 5: `dataset_overview.py` — Session coverage figures

Joins `sessions.pqt`, `qc_photometry.pqt`, `performance.pqt`, and all error logs. Produces session-by-session overview matrices at each processing stage, plus barplots of complete recordings per brain target and per mouse.

**Output**: `figures/dataset_overview/`

### Analysis scripts

| Script | Purpose |
|---|---|
| `responses.py` | Trial-level response magnitudes, response feature vectors, similarity, and decoding |
| `qc_overview.py` | QC metric distributions (histograms, violins, PCA, temporal trends) |
| `task_performance_overview.py` | Learning curves, psychometric trajectories per target |
| `video.py` | Video QC pipeline (timestamps, dropped frames, pin state) |
| `session_viewer.py` | Interactive single-session viewer (raw + preprocessed + PSTHs) |
| `plot_fiber_locations.py` | Fiber tip coordinates on brain atlas slices |
| `recording_capacity.py` | Estimate recording-ready mice by a target date |
| `sync_lp_progress.py` | Lightning Pose progress sync |
| `demo_load_session.py` | Annotated example of loading and plotting a session |

---

## PhotometrySession

`PhotometrySession` wraps a row from `sessions.pqt` and provides methods for loading, validating, preprocessing, and extracting responses. It extends `PhotometrySessionLoader` from `brainbox.io.one`.

Data attributes are lazy-loaded: `trials`, `photometry`, `responses`, `qc`, and `wheel_velocity` start empty and are populated by explicit method calls.

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
ps.load_trials()      # → ps.trials (DataFrame with signed_contrast, contrast added)
ps.load_photometry()  # → ps.photometry dict: {'GCaMP': ..., 'Isosbestic': ...}
```

### Loading from HDF5

If the pipeline has already run, load preprocessed data from disk:

```python
from iblnm.config import SESSIONS_H5_DIR

ps = PhotometrySession(session_row, one=one)
ps.load_h5(SESSIONS_H5_DIR / f'{ps.eid}.h5')
# → ps.photometry['GCaMP_preprocessed'], ps.trials, ps.responses, ps.wheel_velocity
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
ps.run_raw_qc()                    # n_band_inversions, n_early_samples → ps.qc
ps.validate_qc()                   # raises BandInversion or EarlySamples
ps.run_sliding_qc()                # sliding-window signal quality metrics → ps.qc
ps.validate_few_unique_samples()   # raises FewUniqueSamples (non-fatal)
```

After QC, `ps.qc` is a DataFrame with one row per `(brain_region, band)`.

### Preprocessing and response extraction

```python
from iblnm.config import RESPONSE_EVENTS

ps.preprocess()  # bleach → isosbestic → zscore → resample to 30 Hz
                 # → ps.photometry['GCaMP_preprocessed']

ps.extract_responses(events=RESPONSE_EVENTS)
# → ps.responses: xarray DataArray with dims (region, event, trial, time)

ps.save_h5()  # saves all available data groups
```

### Working with responses

```python
# Baseline subtraction (mean of [-0.1, 0] window)
responses = ps.subtract_baseline(ps.responses)

# Mask time points after the next event in a trial sequence
responses = ps.mask_subsequent_events(
    ps.responses,
    event_order=['stimOn_times', 'firstMovement_times', 'feedback_times']
)
```

### Task performance

```python
perf = ps.basic_performance()
# {'fraction_correct': 0.81, 'fraction_correct_easy': 0.94, 'nogo_fraction': 0.02,
#  'psych_50_bias': -1.2, 'psych_50_threshold': 8.4, ...}

block_perf = ps.block_performance()   # per-block psychometrics (biased/ephys only)
fit = ps.fit_psychometric()           # {bias, threshold, lapse_left, lapse_right, r_squared, n_trials}
```

---

## PhotometrySessionGroup

`PhotometrySessionGroup` is the central class for all multi-session analyses. It manages session-level filtering and recording-level explosion internally.

### Design principles

- **Constructor takes session-level DataFrames.** List columns (`brain_region`, `hemisphere`, `target_NM`) are kept intact. Explosion to one-row-per-recording happens via `explode_recordings()`.
- **`filter_sessions` filters at the session level** by session type, excluded subjects, QC error types, and target-NM values. Sessions where none of their target_NM entries match are dropped.
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
# Standard filters (all parameters optional, default to config values)
group.filter_sessions(
    session_types=('biased', 'ephys'),
    exclude_subjects=['excluded_mouse'],
    qc_blockers={'MissingRawData', 'QCValidationError'},
    targetnms=['VTA-DA', 'DR-5HT'],
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

Error log entries follow the schema: `['eid', 'error_type', 'error_message', 'traceback']`.

Downstream scripts read upstream error logs via `collect_session_errors()` and filter sessions based on which error types are present, rather than re-validating.

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

### `data/events.pqt` — one row per (recording x event x trial)

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
| `signed_contrast` | float | Signed stimulus contrast |
| `contrast` | float | Unsigned stimulus contrast |
| `choice` | float | -1 left / 0 no-go / 1 right |
| `feedbackType` | float | 1 reward / -1 punishment |
| `probabilityLeft` | float | Block probability |
| `stim_side` | str | left / right |
| `reaction_time` | float | firstMovement - stimOn (seconds) |
| `response_early` | float | Mean response in early window (0.1-0.35s) |

### `data/response_matrix.pqt` — one row per recording

Response feature vectors indexed by `(eid, target_NM)`. Each column is a condition label encoding event x contrast x laterality x feedback (e.g. `stimOn_c1_contra_correct`). Values are mean response magnitudes in the early window.

### `data/qc_photometry.pqt` — one row per (session, brain region, band)

| Column | Type | Description |
|---|---|---|
| `eid` | str | Session UUID |
| `brain_region` | str | Single recording target |
| `band` | str | GCaMP or Isosbestic |
| `n_unique_samples` | float | Fraction of unique values (< 0.05 flagged) |
| `n_band_inversions` | int | Samples where GCaMP < Isosbestic (> 0 fatal) |
| `n_early_samples` | int | Samples before recording start (> 0 fatal) |
| `ar_score` | float | AR(1) autocorrelation coefficient |
| `median_absolute_deviance` | float | MAD of signal |
| `percentile_asymmetry` | float | (p75-p50) / (p50-p25) skewness proxy |
| `percentile_distance` | float | (p75-p25) / median spread proxy |
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

```
{eid}.h5
├── attrs
│   ├── eid            str       session UUID
│   ├── subject        str       mouse name
│   ├── session_type   str       training / biased / ephys
│   ├── date           str       YYYY-MM-DD
│   ├── fs             int       30  (Hz, resampled signal rate)
│   └── response_window  float[2]  [-1.0, 1.0]  (seconds)
│
├── times              float64 (N,)    sample times at 30 Hz
│
├── preprocessed/
│   └── {brain_region} float64 (N,)   z-scored, isosbestic-corrected GCaMP
│
├── trials/
│   ├── stimOn_times          float64 (T,)
│   ├── firstMovement_times   float64 (T,)
│   ├── feedback_times        float64 (T,)
│   ├── response_times        float64 (T,)
│   ├── choice                float64 (T,)   -1 left, 0 no-go, 1 right
│   ├── feedbackType          float64 (T,)   1 reward, -1 punishment
│   ├── probabilityLeft       float64 (T,)   0.2, 0.5, or 0.8
│   ├── signed_contrast       float64 (T,)   negative = left stimulus
│   └── contrast              float64 (T,)   unsigned
│
├── responses/
│   ├── time           float64 (W,)    time relative to event
│   └── {brain_region}/
│       ├── stimOn_times          float64 (T, W)
│       ├── firstMovement_times   float64 (T, W)
│       └── feedback_times        float64 (T, W)
│
└── wheel/
    ├── velocity       float32 (T, W)  per-trial wheel velocity; NaN-padded
    └── attrs: fs=1000, t0_event='stimOn_times', t1_event='feedback_times'
```

`N` = samples at 30 Hz, `T` = trial count, `W` = response window samples (60 for [-1, 1] s at 30 Hz).

---

## Project Structure

```
iblnm/                      # Core package (generic, reusable)
  config.py                 # Paths, constants, QC thresholds, color mappings
  data.py                   # PhotometrySession, PhotometrySessionGroup
  io.py                     # Alyx/ONE queries
  task.py                   # Task performance (psychometrics, block validation)
  analysis.py               # Signal processing, similarity, decoding
  validation.py             # Custom exceptions, @exception_logger, validate_* functions
  util.py                   # Pandas utilities, parquet I/O, schema enforcement
  vis.py                    # Plotting functions
  gui.py                    # Interactive session viewer widget

scripts/                    # Analysis scripts
tests/                      # pytest (synthetic fixtures, no Alyx calls)
specs/                      # Design docs
metadata/                   # sessions.pqt, error logs, fibers.csv, trajectories.json
data/                       # qc_photometry.pqt, performance.pqt, events.pqt, sessions/*.h5
figures/                    # Output plots
```
