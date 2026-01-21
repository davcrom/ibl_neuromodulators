# Spec: Task Performance Analysis

## Objective

Analyze behavioral performance across sessions to characterize learning progression, session-level performance metrics, and psychometric function parameters per block type.

## Requirements Summary

### 1. Sessions to Training Stage
- **Computation method**: Option A - Count of training sessions before first biased session
- **Session counting**: All sessions regardless of gaps (use `session_n`)
- **Missing stages**: Subjects who never reached biased/ephys are included in barplots showing stage counts, but excluded from cumulative distributions (CDFs are only for mice that reached that stage)

### 2. Performance Metrics
- Overall fraction correct
- Fraction correct for each contrast level
- Fraction correct on easy trials (≥50% contrast)
- **Trial exclusions**: Exclude no-go trials from performance calculations
- **No-go tracking**: Compute fraction of no-go trials per contrast and total

### 3. Psychometric Fits
- Fit per block type (50%, 20%, 80% probability left) for any block types that exist in the session
- **Parameters**: bias, threshold, lapse_left, lapse_right, r_squared
- **Block validation**: Flag and exclude sessions where bias blocks appear to flip every 1-2 trials (minimum block length: 10 trials)

### 4. Output
- Separate file: `metadata/performance.pqt`
- Convenience function: `merge_session_metadata(df_performance, sessions_fpath=SESSIONS_FPATH)`

### 5. Visualizations

#### Learning Figure (by target-NM columns)
| Row | Content |
|-----|---------|
| 1 | Barplot: mice that reached each stage (training, biased, ephys) |
| 2 | CDF: sessions to biased (for mice that reached biased) |
| 3 | CDF: biased sessions to ephys (for mice that reached ephys) |
| 4 | Bias trajectory across training sessions (one line per mouse: thin=no photometry, thick=has photometry) |
| 5 | Threshold trajectory across training sessions (one line per mouse: thin=no photometry, thick=has photometry) |
| 6 | Lapse low trajectory across training sessions (one line per mouse: thin=no photometry, thick=has photometry) |
| 7 | Lapse high trajectory across training sessions (one line per mouse: thin=no photometry, thick=has photometry) |

**Note**: No grand mean on parameter trajectory plots - individual mouse lines only.

#### Psychometric Figure (biased/ephys sessions, by target-NM columns)
| Row | Content |
|-----|---------|
| 1 | Psychometric curves for 50-50 block: thin line per session + thick grand mean |
| 2 | Psychometric curves by block type: grand mean only (one curve per block type: 20/50/80) |
| 3 | Boxplots of psychometric parameters per block type (4 subplots: bias, threshold, lapse_low, lapse_high) |
| 4 | Bias shift trajectory: difference between 80-20 and 20-80 blocks per mouse across consecutive biased sessions |

**Note**: Exclude sessions missing extracted photometry data from psychometric plots.

## Design

### Module: `iblnm/task.py`

Core computational functions for task performance metrics.

```python
from brainbox.behavior.training import (
    get_signed_contrast,
    compute_psychometric,
    compute_performance,
    compute_performance_easy,
)

# Block validation
MIN_BLOCK_LENGTH = 10  # minimum trials per block

def validate_block_structure(trials: pd.DataFrame) -> dict:
    """
    Check if bias blocks have valid structure (not flipping every trial).

    Parameters
    ----------
    trials : pd.DataFrame
        Trials data with probabilityLeft column

    Returns
    -------
    dict with keys:
        - valid: bool, True if block structure is valid
        - min_block_length: int, shortest block in trials
        - n_blocks: int, number of block transitions
        - flagged: bool, True if blocks flip too frequently
    """

def get_block_lengths(probability_left: np.ndarray) -> np.ndarray:
    """Return array of consecutive block lengths."""

# Session stage counting
def count_sessions_to_stage(df_sessions: pd.DataFrame) -> pd.DataFrame:
    """
    Count training sessions before first biased session for each subject.

    Returns DataFrame with columns:
    - subject, target_NM
    - n_training, n_biased, n_ephys
    - sessions_to_biased (NaN if never reached)
    - biased_sessions_to_ephys (NaN if never reached)
    """

def get_subjects_by_stage(df_sessions: pd.DataFrame) -> dict[str, list[str]]:
    """Return dict with lists of subjects that reached each stage."""

# Performance metrics (wrapper functions that handle no-go exclusion)
def compute_fraction_correct(
    trials: pd.DataFrame,
    exclude_nogo: bool = True
) -> float:
    """Overall fraction correct, excluding no-go trials by default."""

def compute_fraction_correct_by_contrast(
    trials: pd.DataFrame,
    exclude_nogo: bool = True
) -> pd.Series:
    """Fraction correct for each contrast level."""

def compute_fraction_correct_easy(
    trials: pd.DataFrame,
    exclude_nogo: bool = True
) -> float:
    """Fraction correct on easy trials (≥50% contrast)."""

def compute_nogo_fraction(trials: pd.DataFrame) -> dict:
    """
    Compute fraction of no-go trials.
    Returns {'total': float, 'by_contrast': pd.Series}
    """

# Psychometric fitting (wraps brainbox functions)
def fit_psychometric(
    trials: pd.DataFrame,
    probability_left: float | None = None,
    compute_r_squared: bool = True
) -> dict:
    """
    Fit psychometric function for given trials.

    Uses brainbox.behavior.training.compute_psychometric with plotting=True
    for better parameter estimates.

    Parameters
    ----------
    trials : pd.DataFrame
        Trials data with choice, contrastLeft, contrastRight, probabilityLeft
    probability_left : float, optional
        If provided, filter to trials with this probabilityLeft value.
        If None, use all trials.
    compute_r_squared : bool
        If True, compute goodness of fit (pseudo R-squared)

    Returns
    -------
    dict with keys: bias, threshold, lapse_left, lapse_right, r_squared, n_trials
    """

def compute_r_squared(trials: pd.DataFrame, psych_params: np.ndarray, block: float = None) -> float:
    """
    Compute pseudo R-squared for psychometric fit.

    Compares predicted vs actual choice proportions at each contrast.
    """

def fit_psychometric_by_block(trials: pd.DataFrame) -> dict[str, dict]:
    """
    Fit psychometric for each block type present in session.

    Returns dict mapping block_type ('50', '20', '80') to fit parameters.
    Only includes blocks with valid structure (>= MIN_BLOCK_LENGTH).
    """

def compute_bias_shift(fit_20: dict, fit_80: dict) -> float:
    """Compute bias difference between 80-20 and 20-80 blocks."""

# Session-level computation
def compute_session_performance(
    trials: pd.DataFrame
) -> dict:
    """
    Compute all performance metrics for a single session.

    Returns dict with:
    - fraction_correct, fraction_correct_easy, fraction_correct_by_contrast
    - nogo_fraction_total, nogo_fraction_by_contrast
    - block_structure_valid, min_block_length (for flagging)
    - psych_50_*, psych_20_*, psych_80_* (if blocks exist and valid)
    - bias_shift (if both 20 and 80 blocks exist and valid)
    """

# I/O
def merge_session_metadata(
    df_performance: pd.DataFrame,
    sessions_fpath: Path = SESSIONS_FPATH
) -> pd.DataFrame:
    """Merge performance data with session metadata."""
```

### Module additions to `iblnm/vis.py`

Plotting functions for task performance analysis.

```python
# Learning progression plots
def plot_stage_barplot(
    df_stage_counts: pd.DataFrame,
    target_nms: list[str],
    ax: plt.Axes = None
) -> plt.Axes:
    """Barplot of mice that reached each stage, per target-NM."""

def plot_sessions_to_stage_cdf(
    df_sessions_to_stage: pd.DataFrame,
    stage: str,  # 'biased' or 'ephys'
    target_nms: list[str],
    ax: plt.Axes = None
) -> plt.Axes:
    """CDF of sessions to reach stage, per target-NM."""

def plot_psychometric_parameter_trajectory(
    df_fits: pd.DataFrame,
    parameter: str,  # 'bias', 'threshold', 'lapse_left', 'lapse_right'
    target_nm: str,
    has_photometry_col: str = 'has_extracted_photometry',
    ax: plt.Axes = None
) -> plt.Axes:
    """
    Plot trajectory of psychometric parameter across training sessions.

    Parameters
    ----------
    df_fits : pd.DataFrame
        Dataframe with columns: subject, session_n, {parameter}, has_extracted_photometry
    parameter : str
        Which parameter to plot
    target_nm : str
        Target-NM to filter for
    has_photometry_col : str
        Column name indicating if session has extracted photometry
    ax : plt.Axes, optional

    Notes
    -----
    - One line per mouse (no grand mean)
    - Thick lines for mice with photometry data
    - Thin lines for mice without photometry data
    """

# Psychometric plots
def plot_psychometric_curves_50(
    df_fits: pd.DataFrame,
    target_nm: str,
    ax: plt.Axes = None
) -> plt.Axes:
    """
    Plot psychometric curves for 50-50 block.

    Shows thin line per session + thick grand mean.
    """

def plot_psychometric_curves_by_block(
    df_fits: pd.DataFrame,
    target_nm: str,
    ax: plt.Axes = None
) -> plt.Axes:
    """
    Plot psychometric grand mean curves by block type.

    Shows one curve per block type (20/50/80) - grand mean only, no individual sessions.
    """

def plot_psychometric_parameters_boxplot(
    df_fits: pd.DataFrame,
    parameter: str,
    target_nm: str,
    ax: plt.Axes = None
) -> plt.Axes:
    """Boxplot of psychometric parameter by block type."""

def plot_bias_shift_trajectory(
    df_fits: pd.DataFrame,
    target_nm: str,
    ax: plt.Axes = None
) -> plt.Axes:
    """Plot bias shift trajectories across biased sessions per mouse."""

# Composite figure functions
def create_learning_figure(
    df_sessions: pd.DataFrame,
    df_stage_counts: pd.DataFrame,
    df_training_fits: pd.DataFrame,
    target_nms: list[str] = None
) -> plt.Figure:
    """
    Create complete learning progression figure.

    Layout: target-NM as columns, rows as described in spec.
    """

def create_psychometric_figure(
    df_sessions: pd.DataFrame,
    df_fits: pd.DataFrame,
    target_nms: list[str] = None
) -> plt.Figure:
    """
    Create complete psychometric analysis figure.

    Layout: target-NM as columns, rows as described in spec.
    Excludes sessions without extracted photometry.
    """
```

### Module: `scripts/task_performance.py`

Main script for running the analysis pipeline.

```python
def main():
    """
    1. Load sessions
    2. Compute sessions to stage
    3. For each session:
       a. Load trials
       b. Validate block structure (flag if blocks flip too fast)
       c. Compute performance metrics
    4. Save to metadata/performance.pqt
    5. Generate figures (excluding flagged sessions)
    """
```

## Files to Modify/Create

| File | Action |
|------|--------|
| `iblnm/task.py` | Create new module |
| `iblnm/vis.py` | Add plotting functions |
| `iblnm/config.py` | Add PERFORMANCE_FPATH, MIN_BLOCK_LENGTH constant |
| `scripts/task_performance.py` | Refactor existing script |
| `tests/test_task.py` | Create tests for task.py |
| `tests/test_vis_task.py` | Create tests for vis.py task plots |

## Test Plan

### `tests/test_task.py`

```python
# Test fixtures
@pytest.fixture
def mock_trials_training():
    """Mock trials for a training session (no blocks)."""

@pytest.fixture
def mock_trials_biased():
    """Mock trials for a biased session (with 20/50/80 blocks)."""

@pytest.fixture
def mock_trials_invalid_blocks():
    """Mock trials with rapidly flipping blocks (invalid)."""

@pytest.fixture
def mock_sessions():
    """Mock sessions dataframe for multiple subjects."""

# Block validation tests
def test_validate_block_structure_valid():
    """Test that valid block structure passes."""

def test_validate_block_structure_flipping():
    """Test that rapidly flipping blocks are flagged."""

def test_get_block_lengths():
    """Test block length calculation."""

# Performance tests
def test_fraction_correct_excludes_nogo():
    """Verify no-go trials are excluded from performance."""

def test_fraction_correct_by_contrast():
    """Test contrast-level performance calculation."""

def test_fraction_correct_easy():
    """Test easy trials (≥50% contrast) performance."""

def test_nogo_fraction():
    """Test no-go fraction calculation."""

# Psychometric tests
def test_fit_psychometric_returns_expected_keys():
    """Verify fit returns all expected parameters including r_squared."""

def test_fit_psychometric_by_block():
    """Test block-specific psychometric fitting."""

def test_fit_psychometric_skips_invalid_blocks():
    """Verify invalid blocks are skipped in fitting."""

def test_compute_r_squared():
    """Test R-squared calculation."""

def test_bias_shift():
    """Test bias shift calculation."""

# Session stage tests
def test_count_sessions_to_stage():
    """Test session counting logic."""

def test_count_sessions_to_stage_never_reached():
    """Test handling of subjects that never reached stage."""
```

## Dependencies on brainbox

The following functions from `brainbox.behavior.training` should be used:

| Function | Purpose |
|----------|---------|
| `get_signed_contrast()` | Convert contrastLeft/Right to signed contrast |
| `compute_psychometric()` | Fit psychometric with `plotting=True` for better params |
| `compute_performance()` | Get performance by contrast and block |
| `compute_performance_easy()` | Get easy trial performance |

The underlying `psychofit` toolbox is the same, but brainbox provides better parameter initialization when `plotting=True`:
- `parstart = [0., 40., 0.1, 0.1]`
- `parmin = [-50., 10., 0., 0.]`
- `parmax = [50., 50., 0.2, 0.2]`
- `nfits = 10`

## Notes from Notebook Review

The notebooks in `notebooks/behaviour/` contain potentially reusable code, but require careful review:

### Reusable concepts (with modifications needed):
- `proportion_correct()` - basic structure good, but needs no-go exclusion
- `erf_psycho_2gammas` - standard IBL psychometric function
- Plotting patterns for psychometric curves and parameter evolution

### Issues identified:
1. Several functions have hardcoded column names that don't match our schema
2. Syntax error in `fit_by_subject_session()` at line 54
3. Functions don't handle no-go trials
4. Block filtering uses `probabilityLeft` directly rather than block types
5. Missing r-squared computation in psychometric fits
6. Some plotting functions use deprecated seaborn `ci` parameter
7. No validation of block structure (flipping blocks)

### Recommended approach:
- Use `brainbox.behavior.training.compute_psychometric` with `plotting=True`
- Add block structure validation before fitting
- Adapt the visualization patterns but rewrite the functions cleanly
- Add proper r-squared calculation using predicted vs actual choices
