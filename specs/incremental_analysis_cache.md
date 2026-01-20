# Spec: Incremental Analysis Cache

## Objective

Implement incremental data saving for long-running analyses with parameter tracking, allowing scripts to resume from where they left off and detect when config changes invalidate cached results.

## Background

**Current state:**
- `photometry_qc.py` and `task_performance.py` loop over all sessions
- Results saved only at the end — crashes lose all progress
- Config parameters in `config.py` are flat module-level constants
- No tracking of which parameters produced which results

**Problems:**
- Long analyses (~2000 sessions) can crash mid-way, losing hours of work
- New sessions added to database require re-running entire analysis
- Changing parameters (e.g., `QC_SLIDING_KWARGS`) silently produces inconsistent results

## Requirements

### Functional

1. **Incremental saving**: Save results in batches (e.g., every N sessions or every M minutes)
2. **Resume**: On restart, detect completed sessions and skip them
3. **Parameter tracking**: Store config parameters alongside results
4. **Parameter diff**: On restart, compare stored vs current config; prompt user if changed
5. **Print config**: Convenience function to display relevant parameters before analysis starts

### Non-functional

1. **Simple**: No external databases, just files
2. **Modular**: Cache logic separate from analysis logic
3. **Backwards compatible**: Can still run analyses without caching

## Design

### Config restructure (`config.py`)

Replace flat constants with nested dicts per analysis:

```python
# General parameters
GENERAL = {
    'min_ntrials': 400,
    'exclude_subjects': ['...'],
    'exclude_session_types': ['habituation'],
}

# Analysis-specific parameters
PHOTOMETRY_QC = {
    'raw_metrics': ['n_early_samples', 'n_band_inversions'],
    'sliding_metrics': ['median_absolute_deviance', 'percentile_distance', ...],
    'sliding_kwargs': {'w_len': 120, 'step_len': 60, 'detrend': True},
    'metrics_kwargs': {'percentile_asymmetry': {'pc_comp': 75}},
}

TASK_PERFORMANCE = {
    'psych_fit_method': 'plotting',  # 'plotting' or 'training'
}

# Convenience function to get all params for an analysis
def get_analysis_config(analysis_name):
    """Return merged dict of general + analysis-specific params."""
    configs = {
        'photometry_qc': PHOTOMETRY_QC,
        'task_performance': TASK_PERFORMANCE,
    }
    return {**GENERAL, **configs[analysis_name]}
```

### AnalysisCache class (`iblnm/cache.py`)

```python
class AnalysisCache:
    """
    Manages incremental saving and resuming of analysis results.

    Storage format:
    - {name}.pqt: DataFrame with results (one row per session/recording)
    - {name}.json: Metadata including config params and completed eids

    Usage:
        cache = AnalysisCache('photometry_qc', config=PHOTOMETRY_QC)
        cache.check_config()  # Prompts user if config changed

        for eid in eids:
            if cache.is_complete(eid):
                continue
            result = run_analysis(eid)
            cache.add(eid, result)
            cache.save_if_needed()  # Saves every N sessions

        cache.save()  # Final save
    """

    def __init__(self, name, config, data_dir=None, batch_size=50):
        """
        Parameters
        ----------
        name : str
            Analysis name (e.g., 'photometry_qc')
        config : dict
            Parameters used for this analysis
        data_dir : Path, optional
            Directory for cache files. Default: PROJECT_ROOT/data
        batch_size : int
            Save after this many new results
        """

    def check_config(self, force=False):
        """
        Compare current config with cached config.

        If different, prints diff and prompts user:
        - 'recompute': Clear cache and start fresh
        - 'continue': Keep old results, compute new sessions with new config (warning: inconsistent)
        - 'abort': Exit

        If force=True, skips prompt and recomputes.
        """

    def is_complete(self, eid):
        """Return True if eid already has results in cache."""

    def add(self, eid, result):
        """
        Add result for eid to pending results.

        Parameters
        ----------
        eid : str
        result : pd.DataFrame or pd.Series
            Must have 'eid' column or be indexed by eid
        """

    def save_if_needed(self):
        """Save if pending results >= batch_size."""

    def save(self):
        """Save all pending results to disk."""

    def load(self):
        """Load existing results from disk."""

    def get_results(self):
        """Return all results as DataFrame."""

    def print_config(self):
        """Pretty-print current config parameters."""
```

### Metadata JSON structure

```json
{
    "analysis": "photometry_qc",
    "created": "2024-01-20T10:30:00",
    "updated": "2024-01-20T14:45:00",
    "config": {
        "raw_metrics": ["n_early_samples", "n_band_inversions"],
        "sliding_kwargs": {"w_len": 120, "step_len": 60, "detrend": true}
    },
    "completed_eids": ["eid1", "eid2", "..."],
    "n_sessions": 1523,
    "n_errors": 12
}
```

### Updated script pattern (`scripts/photometry_qc.py`)

```python
from iblnm.config import PHOTOMETRY_QC, SESSIONS_CLEAN_FPATH
from iblnm.cache import AnalysisCache

if __name__ == '__main__':
    df_sessions = pd.read_parquet(SESSIONS_CLEAN_FPATH)
    df_sessions = df_sessions.query('session_status == "good"')

    cache = AnalysisCache('photometry_qc', config=PHOTOMETRY_QC)
    cache.print_config()
    cache.check_config()

    one = _get_default_connection()

    for idx, row in tqdm(df_sessions.iterrows(), total=len(df_sessions)):
        if cache.is_complete(row['eid']):
            continue

        try:
            ps = PhotometrySession(row, one=one)
            result = ps.run_qc()
            cache.add(row['eid'], result)
        except Exception as e:
            cache.add_error(row['eid'], e)

        cache.save_if_needed()

    cache.save()
    df_qc = cache.get_results()
```

### Config diff display

```
=== photometry_qc config ===
sliding_kwargs:
  w_len: 120
  step_len: 60
  detrend: True
raw_metrics: ['n_early_samples', 'n_band_inversions']

⚠ Config changed since last run:

  sliding_kwargs.w_len:
    cached: 120
    current: 180

Options:
  [r] Recompute all (clear cache)
  [c] Continue (inconsistent results)
  [a] Abort

Choice:
```

## Files to Modify

| File | Changes |
|------|---------|
| `iblnm/config.py` | Restructure to nested dicts per analysis |
| `iblnm/cache.py` | New file: AnalysisCache class |
| `scripts/photometry_qc.py` | Use AnalysisCache |
| `scripts/task_performance.py` | Use AnalysisCache |

## Verification

1. **Unit tests for AnalysisCache:**
   - `test_save_load_roundtrip`
   - `test_resume_skips_completed`
   - `test_config_diff_detection`
   - `test_batch_saving`

2. **Integration test:**
   - Run photometry_qc on 10 sessions
   - Kill mid-way, verify partial results saved
   - Resume, verify skips completed sessions
   - Change config param, verify diff displayed

3. **Manual verification:**
   - Run on full dataset, confirm incremental progress visible
   - Verify JSON metadata is human-readable and correct
