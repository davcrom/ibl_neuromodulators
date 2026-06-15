"""Pure data-layer helpers for the LPViewer GUI.

Session filtering, likelihood→alpha mapping, frame↔trial indexing, and label
persistence — all as pure functions, testable headless. No Qt or pyplot imports
belong in this module; the GUI class (PyQt5 / FigureCanvasQTAgg) lives elsewhere.
"""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from iblnm.config import LP_QC_LABELS

# Settable IBL QC verdicts (the default 'NOT_SET' is not a manual choice).
IBL_QC_VALUES = ('CRITICAL', 'FAIL', 'WARNING', 'PASS')


def filter_sessions_table(
    df_pose: pd.DataFrame,
    measure: str,
    value_range: tuple[float, float],
    session_types: tuple[str, ...],
) -> list[str]:
    """Return eids whose `measure` falls in `value_range` and whose
    `session_type` is in `session_types`.

    Drives the histogram-brush → session-dropdown coupling. The range is
    inclusive on both ends; `session_type` is matched against the selected set.
    """
    low, high = value_range
    in_range = df_pose[measure].between(low, high)
    in_types = df_pose['session_type'].isin(session_types)
    return df_pose.loc[in_range & in_types, 'eid'].tolist()


def likelihood_to_alpha(
    likelihood: float | np.ndarray,
) -> float | np.ndarray:
    """Map a per-keypoint likelihood to a marker alpha in [0, 1].

    Identity mapping clipped to the unit interval, so a keypoint's overlay
    opacity tracks its tracking confidence directly (0 → transparent,
    1 → opaque). Accepts a scalar or an array.
    """
    return np.clip(likelihood, 0.0, 1.0)


def frames_in_trial(
    camera_times: np.ndarray, trial_start: float, trial_end: float
) -> np.ndarray:
    """Return the frame indices whose camera time falls in `[trial_start,
    trial_end]` (inclusive), for intra-trial frame stepping.
    """
    return np.flatnonzero(
        (camera_times >= trial_start) & (camera_times <= trial_end))


def apply_label(
    df_pose: pd.DataFrame, eid: str, field: str, value: str
) -> pd.DataFrame:
    """Return a copy of `df_pose` with manual QC `field` set to `value` for
    `eid`. Validates `field` against `LP_QC_LABELS` and `value` against the IBL
    vocabulary; the input frame is left unmodified.
    """
    if field not in LP_QC_LABELS:
        raise ValueError(f"Unknown QC field: {field!r} (expected {LP_QC_LABELS})")
    if value not in IBL_QC_VALUES:
        raise ValueError(f"Invalid QC value: {value!r} (expected {IBL_QC_VALUES})")
    updated = df_pose.copy()
    updated.loc[updated['eid'] == eid, field] = value
    return updated


def persist_labels(h5_path: str | Path, qc_lp: str, qc_movement: str) -> None:
    """Write the two manual QC labels into a session's `video` H5 group.

    Sets the `qc_lp` / `qc_movement` group attrs in place, leaving the
    automatically-extracted traces and cross-correlation subgroups untouched.
    """
    with h5py.File(h5_path, 'a') as f:
        attrs = f['video'].attrs
        for label, value in zip(LP_QC_LABELS, (qc_lp, qc_movement)):
            attrs[label] = value
