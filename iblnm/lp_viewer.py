"""Data layer + Qt GUI for the LightningPose output QC viewer (LPViewer).

The pure data layer — session filtering, likelihood→alpha mapping, frame↔trial
indexing, label persistence, and the cohort/session data model (``LPViewerModel``)
— is testable headless and carries no display state. The PyQt5 ``LPViewer``
window at the bottom is thin wiring over that layer; its Qt imports resolve to
matplotlib's backend binding and are import-safe headless (a display is only
needed to *show* the window, not to import it).
"""
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from iblnm.config import LP_QC_LABELS
from iblnm.data import _load_pose_traces, _load_pose_xcorr

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


@dataclass
class SessionPanels:
    """Trial-averaged panel data for one session, read from its ``video`` H5 group.

    ``traces`` maps each bodypart label to its trial-mean event-locked trace
    (1-D, aligned to ``times``); ``xcorr`` is the per-third cross-correlation
    dict (``functions``, ``lags``, ``peak_lags``, ``drift``).
    """
    times: np.ndarray
    traces: dict[str, np.ndarray]
    xcorr: dict
    fraction_correct: float | None


class LPViewerModel:
    """Cohort + per-session data model behind the LPViewer, free of any Qt.

    Wraps the pose roll-up table (one row per session, scalar measures +
    ``session_type``), the session H5 directory, and an optional performance
    table. Drives the histogram-brush → dropdown filtering and assembles the
    trial-averaged panel data for a selected session.
    """

    def __init__(
        self,
        df_cohort: pd.DataFrame,
        h5_dir: str | Path,
        df_performance: pd.DataFrame | None = None,
    ):
        self.df_cohort = df_cohort
        self.h5_dir = Path(h5_dir)
        self.df_performance = df_performance

    def filter(
        self,
        measure: str,
        value_range: tuple[float, float],
        session_types: tuple[str, ...],
    ) -> list[str]:
        """Return eids whose `measure` is in `value_range` and `session_type`
        is in `session_types` (the brushed-histogram → dropdown coupling)."""
        return filter_sessions_table(
            self.df_cohort, measure, value_range, session_types)

    def session_panels(self, eid: str) -> SessionPanels:
        """Load the `video` H5 group for `eid` and assemble its panel data:
        the trial-mean trace per bodypart, the cross-correlation dict, and the
        session's `fraction_correct` (or None when unavailable)."""
        with h5py.File(self.h5_dir / f'{eid}.h5', 'r') as f:
            traces = _load_pose_traces(f['video'])
            xcorr = _load_pose_xcorr(f['video'])
        trial_means = {
            str(bodypart): traces.sel(bodypart=bodypart).mean('trial').values
            for bodypart in traces.coords['bodypart'].values
        }
        return SessionPanels(
            times=traces.coords['time'].values,
            traces=trial_means,
            xcorr=xcorr,
            fraction_correct=self._fraction_correct(eid),
        )

    def _fraction_correct(self, eid: str) -> float | None:
        """Look up `fraction_correct` for `eid` in the performance table."""
        if self.df_performance is None:
            return None
        match = self.df_performance.loc[
            self.df_performance['eid'] == eid, 'fraction_correct']
        return match.item() if len(match) else None
