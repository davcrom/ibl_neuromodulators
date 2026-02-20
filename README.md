# IBL Neuromodulators

Fiber photometry analysis pipeline for the IBL neuromodulators project. Ingests session metadata and raw signals from Alyx/ONE, applies QC, preprocesses photometry signals, and extracts peri-event neural responses.

---

## Project Structure

```
iblnm/                      # Core package
  config.py                 # Paths, constants, QC thresholds, color mappings
  data.py                   # PhotometrySession class
  io.py                     # Alyx/ONE queries (subject info, session info, datasets)
  task.py                   # Task performance computation (psychometrics, block validation)
  analysis.py               # Signal processing (response extraction, preprocessing utilities)
  util.py                   # Validation, logging, pandas utilities, parquet I/O
  vis.py                    # Plotting functions

scripts/                    # Analysis scripts (run in order)
  query_database.py         # 1. Download and cache session metadata
  photometry.py             # 2. QC, preprocess, extract responses
  task.py                   # 3. Compute task performance metrics
  dataset_overview.py       # 4. Session coverage figures
  qc_overview.py            # 5. QC metric distributions
  task_performance_overview.py  # 6. Learning curves and psychometric figures

tests/                      # pytest
specs/                      # Design docs for non-trivial features
metadata/                   # sessions.pqt, error logs, fibers.csv
data/
  qc_photometry.pqt         # QC metrics per (eid, brain_region, band)
  performance.pqt           # Task performance metrics per session
  sessions/                 # HDF5 files, one per eid
figures/                    # Output plots
```

---

## Scripts

### `query_database.py` — Download session metadata

Queries the `ibl_fibrephotometry` project on Alyx and enriches each session with subject info (strain, line, neuromodulator), session info (lab, brain regions, users), and dataset availability. Validates all metadata fields and saves results to `metadata/sessions.pqt`. Errors are captured per session in `metadata/query_database_log.pqt`.

```bash
python scripts/query_database.py                 # incremental update
python scripts/query_database.py --redownload    # re-download everything
python scripts/query_database.py --extended-qc   # also fetch Alyx extended QC
```

### `photometry.py` — QC, preprocessing, response extraction

Processes each session through a tiered pipeline:

1. Load trials and photometry from ONE
2. Validate that trials fall within the photometry recording window (fatal)
3. Raw QC: check for band inversions and early samples (fatal)
4. Sliding QC: compute signal quality metrics (fatal)
5. Preprocess: bleach correction → isosbestic regression → z-score → resample to 30 Hz; save signal to HDF5
6. Extract peri-event responses for `stimOn_times`, `firstMovement_times`, `feedback_times`; save to HDF5

Outputs: `data/sessions/{eid}.h5`, `data/qc_photometry.pqt`, `metadata/photometry_log.pqt`.

### `task.py` — Task performance metrics

Loads trials from ONE and computes per-session performance metrics: fraction correct, no-go fraction, psychometric function parameters (bias, threshold, lapses) for the 50/50 block, and — for biased and ephys sessions — per-block psychometrics and bias shift.

Outputs: `data/performance.pqt`, `metadata/task_log.pqt`.

### `dataset_overview.py` — Session coverage figures

Joins `sessions.pqt`, `qc_photometry.pqt`, `performance.pqt`, and all error logs to produce session-by-session overview matrices at each processing stage (registered → raw data → extracted → passing QC). Also generates barplots of complete recordings per brain target and per mouse.

Outputs: `figures/dataset_overview/`.

### `qc_overview.py` — QC metric distributions

Visualizes photometry signal quality across the dataset: histograms of the binary QC metrics with pass/fail cutoffs, violin plots of continuous metrics per target (quantile-transformed), pairwise joint distributions, PCA scatter (colored by target, date, and session type), QC failure rates over time, and photobleaching tau trajectories.

Outputs: `figures/qc_overview/`.

### `task_performance_overview.py` — Learning curves and psychometrics

Generates figures from pre-computed `performance.pqt`: training stage barplots, CDFs of sessions to reach biased/ephys stage, performance trajectories over training, psychometric parameter trajectories, and grand-mean psychometric curves per block type — organized by neuromodulator target.

Outputs: `figures/task_performance/`.

---

## PhotometrySession

`PhotometrySession` extends `PhotometrySessionLoader` from `brainbox.io.one` and wraps a row from `sessions.pqt`. It provides methods for loading, validating, QC, preprocessing, and response extraction.

### Instantiation

```python
import pandas as pd
from one.api import ONE
from iblnm.data import PhotometrySession

one = ONE()
df_sessions = pd.read_parquet('metadata/sessions.pqt')
session_series = df_sessions.iloc[0]

ps = PhotometrySession(session_series, one=one)
```

### Loading data from ONE

```python
ps.load_trials()      # downloads trials from ONE → ps.trials (DataFrame)
ps.load_photometry()  # downloads GCaMP + Isosbestic signals from ONE → ps.photometry dict
```

### Validation

Each method raises a typed exception on failure, so pipelines can catch and log errors at the right granularity.

```python
ps.validate_trials_in_photometry_time()  # raises TrialsNotInPhotometryTime
ps.validate_n_trials()                   # raises InsufficientTrials
ps.validate_event_completeness()         # raises IncompleteEventTimes
ps.validate_block_structure()            # raises BlockStructureBug
```

### QC

```python
ps.run_raw_qc()              # computes n_band_inversions, n_early_samples → ps.qc
ps.validate_qc()             # raises BandInversion or EarlySamples if checks fail
ps.run_sliding_qc()          # computes sliding-window metrics → ps.qc
ps.validate_few_unique_samples()  # raises FewUniqueSamples (non-fatal, log only)
```

After QC runs, `ps.qc` is a DataFrame with one row per `(brain_region, band)`.

### Preprocessing and saving

```python
ps.preprocess()        # bleach → isosbestic → zscore → resample; adds 'GCaMP_preprocessed' band
ps.save_h5(mode='w')  # write session attributes + preprocessed signal to HDF5
```

### Response extraction

```python
from iblnm.config import RESPONSE_EVENTS  # ('stimOn_times', 'firstMovement_times', 'feedback_times')

ps.extract_responses(events=RESPONSE_EVENTS)
# ps.responses is an xarray DataArray with dims (region, event, trial, time)

ps.save_h5(mode='a')  # append trials + responses to existing HDF5
```

### Working with responses

```python
# Baseline subtraction (subtracts mean of [-0.1, 0] window by default)
responses_bl = ps.subtract_baseline(ps.responses)

# Mask time points that fall after the next event in a sequence
responses_masked = ps.mask_subsequent_events(
    ps.responses,
    event_order=['stimOn_times', 'firstMovement_times', 'feedback_times']
)
```

### Loading preprocessed data from HDF5

```python
from iblnm.config import SESSIONS_H5_DIR

ps.load_h5(SESSIONS_H5_DIR / f'{ps.eid}.h5')
# populates ps.photometry['GCaMP_preprocessed'], ps.trials, ps.responses
```

### Task performance

```python
perf = ps.basic_performance()
# {'fraction_correct': 0.81, 'fraction_correct_easy': 0.94, 'nogo_fraction': 0.02,
#  'psych_50_bias': -1.2, 'psych_50_threshold': 8.4, ...}

block_perf = ps.block_performance()
# {'psych_20_bias': ..., 'psych_80_bias': ..., 'bias_shift': ..., ...}

fit = ps.fit_psychometric()
# {'bias': ..., 'threshold': ..., 'lapse_left': ..., 'lapse_right': ..., 'r_squared': ..., 'n_trials': ...}
```

---

## DataFrames

### `metadata/sessions.pqt` — one row per session

| column | type | description |
|---|---|---|
| `eid` | str | Alyx session UUID |
| `subject` | str | mouse name |
| `start_time` | str | ISO8601 session start |
| `session_type` | str | training / biased / ephys / habituation / histology |
| `NM` | str | neuromodulator: DA, 5HT, NE, ACh |
| `brain_region` | list[str] | recording targets, e.g. `['VTA', 'SNc']` |
| `hemisphere` | list[str] | hemisphere per region, e.g. `['l', 'r']` |
| `target_NM` | list[str] | combined labels, e.g. `['VTA-DA', 'SNc-DA']` |
| `lab` | str | recording lab |
| `day_n` | int | days since subject's first session |
| `session_n` | float | session index (dense rank within subject) |
| `session_length` | float | duration in seconds |
| `strain`, `line`, `genotype` | str | mouse genetics |
| `datasets` | list[str] | ALF dataset paths available on ONE |

### `data/qc_photometry.pqt` — one row per (session, brain region, band)

| column | type | description |
|---|---|---|
| `eid` | str | session UUID |
| `brain_region` | str | single recording target |
| `band` | str | GCaMP or Isosbestic |
| `n_unique_samples` | float | fraction of unique signal values (< 0.1 → flat/clipped) |
| `n_band_inversions` | int | samples where GCaMP < Isosbestic (> 0 → fatal) |
| `n_early_samples` | int | samples before recording officially started (> 0 → fatal) |
| `ar_score` | float | AR(1) autocorrelation coefficient |
| `median_absolute_deviance` | float | MAD of signal |
| `percentile_asymmetry` | float | (p75−p50) / (p50−p25) — skewness proxy |
| `percentile_distance` | float | (p75−p25) / median — spread proxy |
| `bleaching_tau` | float | photobleaching time constant in seconds (GCaMP only) |
| `iso_correlation` | float | Pearson r between GCaMP and Isosbestic (GCaMP only) |

### `data/performance.pqt` — one row per session

| column | type | description |
|---|---|---|
| `eid` | str | session UUID |
| `n_trials` | int | total trial count |
| `fraction_correct` | float | overall fraction correct |
| `fraction_correct_easy` | float | fraction correct on 100% contrast trials |
| `nogo_fraction` | float | fraction of no-go trials |
| `psych_50_{param}` | float | psychometric fit on 50/50 block; params: bias, threshold, lapse_left, lapse_right, r_squared |
| `psych_80_{param}` | float | psychometric fit on 80% left block (biased/ephys only) |
| `psych_20_{param}` | float | psychometric fit on 20% left block (biased/ephys only) |
| `bias_shift` | float | psych_80_bias − psych_20_bias (biased/ephys only) |

---

## HDF5 file structure

One file per session: `data/sessions/{eid}.h5`.

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
│   └── {brain_region} float64 (N,)   z-scored, isosbestic-corrected GCaMP signal
│                                      one dataset per recorded region
│
├── trials/
│   ├── stimOn_times          float64 (T,)
│   ├── firstMovement_times   float64 (T,)
│   ├── feedback_times        float64 (T,)
│   ├── response_times        float64 (T,)
│   ├── choice                float64 (T,)   -1 = CCW, 0 = no-go, 1 = CW
│   ├── feedbackType          float64 (T,)   1 = reward, -1 = punishment
│   ├── probabilityLeft       float64 (T,)   0.2, 0.5, or 0.8
│   ├── signed_contrast       float64 (T,)   negative = right stimulus
│   └── contrast              float64 (T,)   unsigned contrast
│
└── responses/
    ├── time           float64 (W,)    time axis relative to event (e.g. 60 pts for [-1, 1] s)
    └── {brain_region}/
        ├── stimOn_times          float64 (T, W)   attrs: window_t0, window_t1
        ├── firstMovement_times   float64 (T, W)
        └── feedback_times        float64 (T, W)
```

`N` = samples at 30 Hz, `T` = trial count, `W` = response window samples (60 for a [−1, 1] s window at 30 Hz).
