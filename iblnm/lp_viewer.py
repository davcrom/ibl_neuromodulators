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

import cv2
import h5py
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.qt_compat import QtCore, QtWidgets
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector

from iblnm.config import (
    DATASET_CATEGORIES,
    LP_QC_LABELS,
    POSE_MEASURES,
)
from iblnm.data import (
    LP_QC_NOT_SET,
    PhotometrySession,
    _load_pose_traces,
    _load_pose_xcorr,
)

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


# ─────────────────────────────────────────────────────────────────────────────
# Frame source — lazy single-frame reader for the GUI overlay
# ─────────────────────────────────────────────────────────────────────────────

RAW_VIDEO_DSET = DATASET_CATEGORIES['raw_video'][0]


class FrameSource:
    """A cv2 video handle paired with the LP pose table for one session.

    Reads one frame at a time from the raw mp4 (no full-video load) and returns
    that frame's keypoint coordinates and likelihoods for overlay. Keypoint
    names are inferred from the pose columns (`{kp}_x`, `{kp}_y`,
    `{kp}_likelihood`).
    """

    def __init__(self, video_path: str | Path, pose: pd.DataFrame,
                 camera_times: np.ndarray):
        self.capture = cv2.VideoCapture(str(video_path))
        self.pose = pose
        self.camera_times = np.asarray(camera_times)
        self.keypoints = sorted({
            col.rsplit('_', 1)[0] for col in pose.columns
            if col.endswith(('_x', '_y', '_likelihood'))
        })

    def read(self, frame_idx: int) -> tuple[np.ndarray | None, dict]:
        """Return `(rgb_frame, keypoints)` for `frame_idx`.

        `keypoints` maps each keypoint name to `(x, y, likelihood)`. Returns
        `(None, {})` when the frame cannot be decoded.
        """
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self.capture.read()
        if not ok:
            return None, {}
        row = self.pose.iloc[frame_idx]
        keypoints = {
            kp: (row[f'{kp}_x'], row[f'{kp}_y'], row[f'{kp}_likelihood'])
            for kp in self.keypoints
        }
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), keypoints

    def close(self) -> None:
        self.capture.release()


# ─────────────────────────────────────────────────────────────────────────────
# Qt GUI — thin wiring over LPViewerModel + the pure helpers above.
# Import-safe headless; a display is needed only to show the window.
# ─────────────────────────────────────────────────────────────────────────────

HISTOGRAM_MEASURES = list(POSE_MEASURES)  # paw, nose, tongue_speed, tongue_likelihood


class LPViewer(QtWidgets.QMainWindow):
    """LightningPose output QC viewer.

    Native Qt controls (session-type multiselect, brushed-histogram → session
    dropdown, two QC label selectors) drive an `LPViewerModel`; embedded
    matplotlib canvases render the cohort histograms, the event-locked panels,
    the paw–wheel cross-correlation, and the keypoint-overlay frame viewer.
    Manual labels are written back to each touched session's H5 on close.
    """

    def __init__(self, model: LPViewerModel, one=None):
        super().__init__()
        self.model = model
        self.one = one
        self.active_brush: tuple[str, tuple[float, float]] | None = None
        self.frame_source: FrameSource | None = None
        self.trial_frames = np.array([], dtype=int)
        self.frame_pos = 0
        self.touched_eids: set[str] = set()
        self.setWindowTitle('LPViewer — LightningPose output QC')

        central = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(central)
        layout.addWidget(self._build_controls(), stretch=1)
        layout.addWidget(self._build_display(), stretch=2)
        self.setCentralWidget(central)
        self._refresh_dropdown()

    # -- construction ---------------------------------------------------------

    def _build_controls(self) -> QtWidgets.QWidget:
        """Session-type multiselect, the four brushable histograms, the session
        dropdown, the fraction-correct readout, and the two QC selectors."""
        panel = QtWidgets.QWidget()
        box = QtWidgets.QVBoxLayout(panel)

        box.addWidget(QtWidgets.QLabel('Session types'))
        self.type_list = QtWidgets.QListWidget()
        for session_type in sorted(self.model.df_cohort['session_type'].dropna().unique()):
            item = QtWidgets.QListWidgetItem(session_type)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked)
            self.type_list.addItem(item)
        self.type_list.itemChanged.connect(lambda _: self._refresh_dropdown())
        box.addWidget(self.type_list)

        self.hist_fig = Figure(figsize=(4, 6))
        self.hist_canvas = FigureCanvasQTAgg(self.hist_fig)
        self._draw_histograms()
        box.addWidget(self.hist_canvas)

        box.addWidget(QtWidgets.QLabel('Sessions in range'))
        self.session_combo = QtWidgets.QComboBox()
        self.session_combo.currentTextChanged.connect(self._on_session_selected)
        box.addWidget(self.session_combo)

        self.performance_label = QtWidgets.QLabel('fraction_correct: —')
        box.addWidget(self.performance_label)

        self.label_combos = {}
        for field in LP_QC_LABELS:
            box.addWidget(QtWidgets.QLabel(field))
            combo = QtWidgets.QComboBox()
            combo.addItems([LP_QC_NOT_SET, *IBL_QC_VALUES])
            combo.currentTextChanged.connect(
                lambda value, f=field: self._on_label(f, value))
            self.label_combos[field] = combo
            box.addWidget(combo)

        return panel

    def _build_display(self) -> QtWidgets.QWidget:
        """The event-locked + cross-correlation panels and the frame viewer."""
        panel = QtWidgets.QWidget()
        box = QtWidgets.QVBoxLayout(panel)

        self.panel_fig = Figure(figsize=(8, 5))
        self.panel_canvas = FigureCanvasQTAgg(self.panel_fig)
        box.addWidget(self.panel_canvas)

        self.frame_fig = Figure(figsize=(5, 4))
        self.frame_canvas = FigureCanvasQTAgg(self.frame_fig)
        self.frame_ax = self.frame_fig.add_subplot(111)
        self.frame_ax.set_axis_off()
        box.addWidget(self.frame_canvas)

        controls = QtWidgets.QHBoxLayout()
        self.trial_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.trial_slider.setEnabled(False)
        self.trial_slider.valueChanged.connect(self._on_trial_changed)
        prev_button = QtWidgets.QPushButton('◀ Prev frame')
        next_button = QtWidgets.QPushButton('Next frame ▶')
        prev_button.clicked.connect(lambda: self._step_frame(-1))
        next_button.clicked.connect(lambda: self._step_frame(1))
        controls.addWidget(QtWidgets.QLabel('Trial'))
        controls.addWidget(self.trial_slider)
        controls.addWidget(prev_button)
        controls.addWidget(next_button)
        box.addLayout(controls)

        return panel

    def _draw_histograms(self) -> None:
        """One brushable histogram per cohort measure; each SpanSelector sets
        the active brush range for that measure and refreshes the dropdown."""
        self.span_selectors = []
        for i, measure in enumerate(HISTOGRAM_MEASURES):
            ax = self.hist_fig.add_subplot(len(HISTOGRAM_MEASURES), 1, i + 1)
            ax.hist(self.model.df_cohort[measure].dropna(), bins=30)
            ax.set_ylabel(measure, fontsize=8)
            selector = SpanSelector(
                ax, lambda lo, hi, m=measure: self._on_brush(m, lo, hi),
                'horizontal', useblit=True)
            self.span_selectors.append(selector)
        self.hist_fig.tight_layout()

    # -- filtering ------------------------------------------------------------

    def _selected_types(self) -> tuple[str, ...]:
        return tuple(
            self.type_list.item(i).text()
            for i in range(self.type_list.count())
            if self.type_list.item(i).checkState() == QtCore.Qt.Checked
        )

    def _on_brush(self, measure: str, low: float, high: float) -> None:
        self.active_brush = (measure, (low, high))
        self._refresh_dropdown()

    def _refresh_dropdown(self) -> None:
        """Repopulate the session dropdown from the active brush + type filter."""
        if self.active_brush is None:
            eids = []
        else:
            measure, value_range = self.active_brush
            eids = self.model.filter(measure, value_range, self._selected_types())
        self.session_combo.blockSignals(True)
        self.session_combo.clear()
        self.session_combo.addItems(eids)
        self.session_combo.blockSignals(False)
        if eids:
            self._on_session_selected(eids[0])

    # -- session display ------------------------------------------------------

    def _on_session_selected(self, eid: str) -> None:
        if not eid:
            return
        self.current_eid = eid
        panels = self.model.session_panels(eid)
        self._draw_panels(panels)
        fraction = panels.fraction_correct
        self.performance_label.setText(
            f'fraction_correct: {fraction:.3f}' if fraction is not None
            else 'fraction_correct: —')
        self._sync_label_combos(eid)
        self._load_frame_source(eid)

    def _draw_panels(self, panels: SessionPanels) -> None:
        """Draw the three event-locked panels and the cross-correlation panel."""
        self.panel_fig.clear()
        ax_paw, ax_nose, ax_tongue, ax_xcorr = self.panel_fig.subplots(2, 2).ravel()

        ax_paw.plot(panels.times, panels.traces['paw'])
        ax_paw.set_title('Paw @ firstMovement')
        ax_nose.plot(panels.times, panels.traces['nose'])
        ax_nose.set_title('Nose @ feedback')
        ax_tongue.plot(panels.times, panels.traces['tongue_speed'], color='C0')
        ax_tongue.set_ylabel('tongue speed', color='C0')
        ax_like = ax_tongue.twinx()
        ax_like.plot(panels.times, panels.traces['tongue_likelihood'], color='C1')
        ax_like.set_ylabel('likelihood', color='C1')
        ax_tongue.set_title('Tongue @ feedback')
        for ax in (ax_paw, ax_nose, ax_tongue):
            ax.axvline(0.0, color='k', lw=0.5, ls='--')

        for label, function in zip(('early', 'mid', 'late'),
                                   panels.xcorr['functions']):
            ax_xcorr.plot(panels.xcorr['lags'], function, label=label)
        ax_xcorr.legend(fontsize=8)
        ax_xcorr.set_title(f"Paw–wheel xcorr (drift={panels.xcorr['drift']:.3f})")
        ax_xcorr.set_xlabel('lag (s)')
        self.panel_fig.tight_layout()
        self.panel_canvas.draw_idle()

    def _sync_label_combos(self, eid: str) -> None:
        """Set each QC selector to the session's stored label without retriggering
        the change handler."""
        row = self.model.df_cohort.loc[self.model.df_cohort['eid'] == eid]
        for field, combo in self.label_combos.items():
            value = row[field].iloc[0] if field in row else LP_QC_NOT_SET
            if pd.isna(value):
                value = LP_QC_NOT_SET
            combo.blockSignals(True)
            combo.setCurrentText(value)
            combo.blockSignals(False)

    def _on_label(self, field: str, value: str) -> None:
        if value == LP_QC_NOT_SET or not getattr(self, 'current_eid', None):
            return
        self.model.df_cohort = apply_label(
            self.model.df_cohort, self.current_eid, field, value)
        self.touched_eids.add(self.current_eid)

    # -- frame viewer ---------------------------------------------------------

    def _load_frame_source(self, eid: str) -> None:
        """Load pose + trials + the raw mp4 for `eid` and arm the trial slider.

        Best-effort: any load failure leaves the frame viewer blank rather than
        crashing the session switch.
        """
        if self.frame_source is not None:
            self.frame_source.close()
            self.frame_source = None
        if self.one is None:
            return
        try:
            row = self.model.df_cohort.loc[
                self.model.df_cohort['eid'] == eid].iloc[0]
            session = PhotometrySession(row, one=self.one, load_data=False)
            session.load_pose()
            session.load_trials()
            video_path = self.one.load_dataset(
                eid, RAW_VIDEO_DSET, download_only=True)
        except Exception as error:  # noqa: BLE001 - viewer must not crash on bad data
            self.frame_ax.clear()
            self.frame_ax.set_axis_off()
            self.frame_ax.text(0.5, 0.5, f'No frames: {error}', ha='center')
            self.frame_canvas.draw_idle()
            return
        self.frame_source = FrameSource(
            video_path, session.pose, session.pose_times)
        self.trials = session.trials
        self.trial_slider.blockSignals(True)
        self.trial_slider.setRange(0, len(self.trials) - 1)
        self.trial_slider.setValue(0)
        self.trial_slider.setEnabled(True)
        self.trial_slider.blockSignals(False)
        self._on_trial_changed(0)

    def _on_trial_changed(self, trial_idx: int) -> None:
        """Compute the frame indices spanning the selected trial and show its
        first frame."""
        if self.frame_source is None:
            return
        trial = self.trials.iloc[trial_idx]
        self.trial_frames = frames_in_trial(
            self.frame_source.camera_times,
            trial['stimOn_times'], trial['feedback_times'])
        self.frame_pos = 0
        self._draw_frame()

    def _step_frame(self, delta: int) -> None:
        if len(self.trial_frames) == 0:
            return
        self.frame_pos = int(np.clip(
            self.frame_pos + delta, 0, len(self.trial_frames) - 1))
        self._draw_frame()

    def _draw_frame(self) -> None:
        """Render the current frame with each keypoint overlaid at an alpha set
        by its likelihood."""
        if len(self.trial_frames) == 0:
            return
        frame_idx = int(self.trial_frames[self.frame_pos])
        image, keypoints = self.frame_source.read(frame_idx)
        self.frame_ax.clear()
        self.frame_ax.set_axis_off()
        if image is None:
            self.frame_canvas.draw_idle()
            return
        self.frame_ax.imshow(image)
        for x, y, likelihood in keypoints.values():
            self.frame_ax.scatter(x, y, s=30, color='r',
                                  alpha=float(likelihood_to_alpha(likelihood)))
        self.frame_ax.set_title(f'frame {frame_idx}')
        self.frame_canvas.draw_idle()

    def keyPressEvent(self, event) -> None:
        if event.key() == QtCore.Qt.Key_Left:
            self._step_frame(-1)
        elif event.key() == QtCore.Qt.Key_Right:
            self._step_frame(1)
        else:
            super().keyPressEvent(event)

    # -- persistence ----------------------------------------------------------

    def closeEvent(self, event) -> None:
        """Write the two manual QC labels for every touched session back to H5."""
        for eid in self.touched_eids:
            row = self.model.df_cohort.loc[
                self.model.df_cohort['eid'] == eid].iloc[0]
            persist_labels(self.model.h5_dir / f'{eid}.h5',
                           row['qc_lp'], row['qc_movement'])
        if self.frame_source is not None:
            self.frame_source.close()
        super().closeEvent(event)
