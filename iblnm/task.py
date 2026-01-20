"""
Task Performance Analysis Module

Computes behavioral performance metrics including:
- Sessions to reach training stages
- Performance metrics (fraction correct, by contrast, easy trials)
- Psychometric function parameters per block type
- Block structure validation
"""
import numpy as np
import pandas as pd
import psychofit as psy

from brainbox.behavior.training import (
    get_signed_contrast,
    compute_psychometric,
)

from iblnm.config import MIN_BLOCK_LENGTH


# =============================================================================
# Block Validation
# =============================================================================

def get_block_lengths(probability_left: np.ndarray) -> np.ndarray:
    """
    Return array of consecutive block lengths.

    Parameters
    ----------
    probability_left : np.ndarray
        Array of probabilityLeft values for each trial.

    Returns
    -------
    np.ndarray
        Array of block lengths (number of consecutive trials with same probabilityLeft).
    """
    if len(probability_left) == 0:
        return np.array([])

    # Find where probability changes
    changes = np.diff(probability_left) != 0
    change_indices = np.where(changes)[0] + 1

    # Add start and end indices
    indices = np.concatenate([[0], change_indices, [len(probability_left)]])

    # Compute lengths between consecutive indices
    lengths = np.diff(indices)

    return lengths


def validate_block_structure(trials: pd.DataFrame) -> dict:
    """
    Check if bias blocks have valid structure (not flipping every trial).

    Parameters
    ----------
    trials : pd.DataFrame
        Trials data with probabilityLeft column.

    Returns
    -------
    dict
        Dictionary with keys:
        - valid: bool, True if block structure is valid
        - min_block_length: int, shortest block in trials
        - n_blocks: int, number of block transitions
        - flagged: bool, True if blocks flip too frequently
    """
    if 'probabilityLeft' not in trials.columns:
        return {
            'valid': True,
            'min_block_length': len(trials),
            'n_blocks': 1,
            'flagged': False,
        }

    prob_left = trials['probabilityLeft'].values
    lengths = get_block_lengths(prob_left)

    if len(lengths) == 0:
        return {
            'valid': False,
            'min_block_length': 0,
            'n_blocks': 0,
            'flagged': True,
        }

    min_length = int(lengths.min())
    n_blocks = len(lengths)

    # Flag if any block is shorter than MIN_BLOCK_LENGTH
    flagged = min_length < MIN_BLOCK_LENGTH
    valid = not flagged

    return {
        'valid': valid,
        'min_block_length': min_length,
        'n_blocks': n_blocks,
        'flagged': flagged,
    }


# =============================================================================
# Session Stage Counting
# =============================================================================

def count_sessions_to_stage(df_sessions: pd.DataFrame) -> pd.DataFrame:
    """
    Count training sessions before first biased session for each subject.

    Parameters
    ----------
    df_sessions : pd.DataFrame
        Sessions dataframe with 'subject', 'session_type', 'session_n' columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - subject
        - n_training, n_biased, n_ephys
        - sessions_to_biased (NaN if never reached)
        - biased_sessions_to_ephys (NaN if never reached)
    """
    results = []

    for subject, group in df_sessions.groupby('subject'):
        # Sort by session_n
        group = group.sort_values('session_n')

        # Count sessions of each type
        n_training = (group['session_type'] == 'training').sum()
        n_biased = (group['session_type'] == 'biased').sum()
        n_ephys = (group['session_type'] == 'ephys').sum()

        # Sessions to biased: count training sessions before first biased
        biased_sessions = group[group['session_type'] == 'biased']
        if len(biased_sessions) > 0:
            first_biased_n = biased_sessions['session_n'].min()
            training_before = group[
                (group['session_type'] == 'training') &
                (group['session_n'] < first_biased_n)
            ]
            sessions_to_biased = len(training_before)
        else:
            sessions_to_biased = np.nan

        # Biased sessions to ephys: count biased sessions before first ephys
        ephys_sessions = group[group['session_type'] == 'ephys']
        if len(ephys_sessions) > 0:
            first_ephys_n = ephys_sessions['session_n'].min()
            biased_before = group[
                (group['session_type'] == 'biased') &
                (group['session_n'] < first_ephys_n)
            ]
            biased_sessions_to_ephys = len(biased_before)
        else:
            biased_sessions_to_ephys = np.nan

        results.append({
            'subject': subject,
            'n_training': n_training,
            'n_biased': n_biased,
            'n_ephys': n_ephys,
            'sessions_to_biased': sessions_to_biased,
            'biased_sessions_to_ephys': biased_sessions_to_ephys,
        })

    return pd.DataFrame(results)


def get_subjects_by_stage(df_sessions: pd.DataFrame) -> dict:
    """
    Return dict with lists of subjects that reached each stage.

    Parameters
    ----------
    df_sessions : pd.DataFrame
        Sessions dataframe with 'subject', 'session_type' columns.

    Returns
    -------
    dict
        Dictionary mapping stage name to list of subjects that reached it.
    """
    stages = {}

    for stage in ['training', 'biased', 'ephys']:
        subjects = df_sessions[df_sessions['session_type'] == stage]['subject'].unique()
        stages[stage] = list(subjects)

    return stages


# =============================================================================
# Performance Metrics
# =============================================================================

def _get_signed_contrast(trials: pd.DataFrame) -> np.ndarray:
    """Convert trials DataFrame to signed contrast array."""
    contrast_left = trials['contrastLeft'].fillna(0).values
    contrast_right = trials['contrastRight'].fillna(0).values
    return (contrast_right - contrast_left) * 100  # in percent, right positive


def compute_fraction_correct(
    trials: pd.DataFrame,
    exclude_nogo: bool = True,
) -> float:
    df = trials.copy()
    if exclude_nogo:
        df = df[df['choice'] != 0]
    return (df['feedbackType'] == 1).mean()


def compute_nogo_fraction(trials: pd.DataFrame) -> float:
    return (trials['choice'] == 0).mean()


# =============================================================================
# Psychometric Fitting
# =============================================================================

def _compute_r_squared(
    signed_contrast: np.ndarray,
    choice: np.ndarray,
    psych_params: np.ndarray,
    block_mask: np.ndarray = None
) -> float:
    """
    Compute pseudo R-squared for psychometric fit.

    Parameters
    ----------
    signed_contrast : np.ndarray
        Signed contrast values (in percent).
    choice : np.ndarray
        Choice values (-1 = left, 1 = right).
    psych_params : np.ndarray
        Psychometric parameters [bias, threshold, lapse_low, lapse_high].
    block_mask : np.ndarray, optional
        Boolean mask for trials to include.

    Returns
    -------
    float
        Pseudo R-squared value (0 to 1).
    """
    if block_mask is not None:
        signed_contrast = signed_contrast[block_mask]
        choice = choice[block_mask]

    if len(signed_contrast) == 0:
        return np.nan

    # Get unique contrasts and observed proportions
    contrasts = np.unique(signed_contrast)
    observed = []
    predicted = []

    for c in contrasts:
        mask = signed_contrast == c
        # Proportion choosing right (choice == -1 in IBL convention)
        obs_prop = (choice[mask] == -1).mean()
        observed.append(obs_prop)

        # Predicted proportion from psychometric function
        pred_prop = psy.erf_psycho_2gammas(psych_params, np.array([c]))[0]
        predicted.append(pred_prop)

    observed = np.array(observed)
    predicted = np.array(predicted)

    # Compute R-squared
    ss_res = np.sum((observed - predicted) ** 2)
    ss_tot = np.sum((observed - observed.mean()) ** 2)

    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0

    r_squared = 1 - (ss_res / ss_tot)
    return max(0, r_squared)  # Clamp to 0 if negative


def fit_psychometric(
    trials: pd.DataFrame,
    probability_left: float = None,
    compute_r_squared: bool = True
) -> dict:
    """
    Fit psychometric function for given trials.

    Uses brainbox.behavior.training.compute_psychometric with plotting=True
    for better parameter estimates.

    Parameters
    ----------
    trials : pd.DataFrame
        Trials data with choice, contrastLeft, contrastRight, probabilityLeft.
    probability_left : float, optional
        If provided, filter to trials with this probabilityLeft value.
        If None, use all trials.
    compute_r_squared : bool
        If True, compute goodness of fit.

    Returns
    -------
    dict
        Dictionary with keys: bias, threshold, lapse_left, lapse_right,
        r_squared, n_trials.
    """
    # Convert to format expected by brainbox
    from iblutil.util import Bunch

    df = trials.copy()

    # Filter by probability_left if specified
    if probability_left is not None and 'probabilityLeft' in df.columns:
        df = df[df['probabilityLeft'] == probability_left]

    if len(df) == 0:
        return {
            'bias': np.nan,
            'threshold': np.nan,
            'lapse_left': np.nan,
            'lapse_right': np.nan,
            'r_squared': np.nan,
            'n_trials': 0,
        }

    # Create Bunch object for brainbox function
    trials_bunch = Bunch()
    trials_bunch['contrastLeft'] = df['contrastLeft'].fillna(0).values
    trials_bunch['contrastRight'] = df['contrastRight'].fillna(0).values
    trials_bunch['choice'] = df['choice'].values
    trials_bunch['feedbackType'] = df['feedbackType'].values
    trials_bunch['probabilityLeft'] = df['probabilityLeft'].values if 'probabilityLeft' in df.columns else np.full(len(df), 0.5)

    # Compute psychometric using brainbox with plotting=True for better params
    try:
        psych_params = compute_psychometric(trials_bunch, plotting=True)
    except Exception:
        return {
            'bias': np.nan,
            'threshold': np.nan,
            'lapse_left': np.nan,
            'lapse_right': np.nan,
            'r_squared': np.nan,
            'n_trials': len(df),
        }

    # Extract parameters
    bias, threshold, lapse_high, lapse_low = psych_params

    # Compute R-squared if requested
    r_sq = np.nan
    if compute_r_squared:
        signed_contrast = _get_signed_contrast(df)
        r_sq = _compute_r_squared(signed_contrast, df['choice'].values, psych_params)

    return {
        'bias': bias,
        'threshold': threshold,
        'lapse_left': lapse_low,  # lapse_low is for left choices
        'lapse_right': lapse_high,  # lapse_high is for right choices
        'r_squared': r_sq,
        'n_trials': len(df),
    }


def fit_psychometric_by_block(trials: pd.DataFrame) -> dict:
    """
    Fit psychometric for each block type present in session.

    Parameters
    ----------
    trials : pd.DataFrame
        Trials data with probabilityLeft column.

    Returns
    -------
    dict
        Dictionary mapping block_type ('50', '20', '80') to fit parameters.
        Only includes blocks with valid structure (>= MIN_BLOCK_LENGTH).
    """
    results = {}

    if 'probabilityLeft' not in trials.columns:
        # No blocks, fit all trials
        fit = fit_psychometric(trials)
        results['50'] = fit
        return results

    # Get unique probability values
    prob_values = trials['probabilityLeft'].unique()

    # Map probability to block name
    prob_to_block = {0.5: '50', 0.2: '20', 0.8: '80'}

    for prob in prob_values:
        if prob not in prob_to_block:
            continue

        block_name = prob_to_block[prob]
        block_trials = trials[trials['probabilityLeft'] == prob]

        # Check if block has enough trials
        if len(block_trials) < MIN_BLOCK_LENGTH:
            continue

        fit = fit_psychometric(block_trials)
        results[block_name] = fit

    return results


def compute_bias_shift(fit_20: dict, fit_80: dict) -> float:
    """
    Compute bias difference between 80-20 and 20-80 blocks.

    Parameters
    ----------
    fit_20 : dict
        Psychometric fit for 20% left block.
    fit_80 : dict
        Psychometric fit for 80% left block.

    Returns
    -------
    float
        Bias shift (bias_80 - bias_20).
    """
    if fit_20 is None or fit_80 is None:
        return np.nan
    if np.isnan(fit_20.get('bias', np.nan)) or np.isnan(fit_80.get('bias', np.nan)):
        return np.nan

    return fit_80['bias'] - fit_20['bias']


