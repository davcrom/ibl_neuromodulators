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
from matplotlib import colormaps
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.qt_compat import QtCore, QtWidgets
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle
from matplotlib.ticker import FuncFormatter
from matplotlib.widgets import SpanSelector

from iblnm.config import (
    DATASET_CATEGORIES,
    LP_QC_LABELS,
    POSE_MEASURES,
    QC_VALUE_ORDER,
    SESSIONTYPE2COLOR,
    VIDEO_QC_COLS,
)
from iblnm.data import (
    LP_QC_NOT_SET,
    PhotometrySession,
    _load_pose_traces,
    _load_pose_xcorr,
)

# Settable IBL QC verdicts (the default 'NOT_SET' is not a manual choice).
IBL_QC_VALUES = ('CRITICAL', 'FAIL', 'WARNING', 'PASS')


def start_time_to_numeric(values) -> np.ndarray:
    """Convert ISO-8601 `start_time` strings to float nanoseconds since epoch.

    Returns a float array (so `NaT`/missing entries become `NaN`, which integer
    epoch encodings cannot represent) suitable for binning and brushing on a
    numeric datetime axis. `values` is any array-like of ISO-8601 strings.
    """
    times = pd.DatetimeIndex(pd.to_datetime(values, format='ISO8601'))
    numeric = times.asi8.astype(float)
    numeric[times.isna()] = np.nan
    return numeric


def select_population(
    df: pd.DataFrame,
    session_types: tuple[str, ...],
    qc_selections: dict[str, set[str]],
    fc_range: tuple[float, float] | None,
    start_time_range: tuple[float, float] | None,
) -> np.ndarray:
    """Boolean mask over `df` for the population-filter conjunction.

    Combines (AND) every active constraint: `session_type` membership in
    `session_types`; for each `qc_selections` field with a non-empty verdict set,
    `df[field]` membership in that set (so verdicts within a field are OR, fields
    across are AND); `fraction_correct` in `fc_range` (inclusive) when given; and
    the numeric `start_time` (see `start_time_to_numeric`) in `start_time_range`
    (inclusive) when given. A field mapped to an empty set, or a `None` range, is
    dropped from the conjunction. With no active constraints the mask is all-True.
    """
    constraints = [df['session_type'].isin(session_types).to_numpy()]
    constraints += [
        df[field].isin(verdicts).to_numpy()
        for field, verdicts in qc_selections.items() if verdicts
    ]
    if fc_range is not None:
        constraints.append(df['fraction_correct'].between(*fc_range).to_numpy())
    if start_time_range is not None:
        numeric = start_time_to_numeric(df['start_time'])
        low, high = start_time_range
        constraints.append((numeric >= low) & (numeric <= high))
    return np.logical_and.reduce(constraints)


def filter_dropdown(
    df: pd.DataFrame,
    population_mask: np.ndarray,
    metric_ranges: dict[str, tuple[float, float]],
) -> list[str]:
    """Eids in the population that also fall within every metric brush range.

    `population_mask` is the boolean mask from `select_population`. Each entry of
    `metric_ranges` adds an inclusive `between` constraint on that measure (the
    selection-only movement brushes). An empty `metric_ranges` yields the whole
    population, not an empty set.
    """
    mask = population_mask
    if metric_ranges:
        mask = mask & np.logical_and.reduce([
            df[measure].between(low, high).to_numpy()
            for measure, (low, high) in metric_ranges.items()
        ])
    return df.loc[mask, 'eid'].tolist()


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


def format_session_title(
    eid: str,
    subject: str,
    start_time: object,
    session_type: str,
    fraction_correct: float | None,
) -> str:
    """Build the one-line session title shown above the display panels.

    Format: ``{eid[:8]} · {subject} · {YYYY-MM-DD} · {session_type} ·
    performance: {pct}``, where ``pct`` is the whole-percent
    ``fraction_correct`` or ``—`` when it is missing. ``start_time`` may be an
    ISO string or a datetime/Timestamp.
    """
    date = pd.to_datetime(start_time).strftime('%Y-%m-%d')
    performance = (
        '—' if fraction_correct is None or pd.isna(fraction_correct)
        else f'{round(fraction_correct * 100)}%')
    return (f'{eid[:8]} · {subject} · {date} · {session_type} · '
            f'performance: {performance}')


def format_event_timings(
    frame_time: float,
    stimOn: float,
    firstMovement: float,
    feedback: float,
) -> list[str]:
    """Label the current frame's signed time relative to each trial event.

    Each label reads ``{event}: {frame_time - event_time:+.2f} s`` — negative
    before the event, positive after — or ``{event}: —`` when the event time is
    NaN (e.g. `firstMovement` on no-movement trials).
    """
    events = {'stimOn': stimOn, 'firstMovement': firstMovement,
              'feedback': feedback}
    return [
        f'{name}: —' if np.isnan(time) else f'{name}: {frame_time - time:+.2f} s'
        for name, time in events.items()
    ]


def trial_schematic_values(trial: pd.Series) -> dict:
    """Derive the trial-schematic inputs from a trial row.

    Returns `side` (`'left'` when `contrastLeft` is non-NaN else `'right'`),
    `contrast` (the non-NaN of `contrastLeft`/`contrastRight`, in [0, 1]),
    `correct` (`feedbackType == 1`), and `prob_left` (`probabilityLeft`). One
    of `contrastLeft`/`contrastRight` is NaN on every trial.
    """
    left = np.isnan(trial['contrastLeft'])
    return {
        'side': 'right' if left else 'left',
        'contrast': trial['contrastRight'] if left else trial['contrastLeft'],
        'correct': trial['feedbackType'] == 1,
        'prob_left': trial['probabilityLeft'],
    }


def draw_trial_schematic(
    ax,
    side: str,
    contrast: float,
    correct: bool,
    prob_left: float,
) -> None:
    """Render a schematic of the current trial onto `ax`.

    Draws a horizontal "screen" strip with a center tick, a filled stimulus
    disc placed left or right of center per `side`, and a two-segment bias bar
    above the strip. The disc grayscale darkens and its radius grows with
    `contrast` (in [0, 1]), with the contrast percentage printed inside it; the
    disc's ring is green when `correct` else red. The bias bar's left-segment
    width is proportional to `prob_left`. Clears `ax` first; Qt-free so it
    renders headless under the Agg backend.
    """
    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    strip_x, strip_w, strip_y = 0.1, 0.8, 0.4
    center_x = strip_x + strip_w / 2
    ax.add_patch(Rectangle((strip_x, strip_y - 0.03), strip_w, 0.06,
                           facecolor='0.85', edgecolor='0.4', label='strip'))
    ax.plot([center_x, center_x], [strip_y - 0.05, strip_y + 0.05],
            color='0.4', lw=1)

    disc_x = center_x + (0.18 if side == 'right' else -0.18)
    radius = 0.05 + 0.10 * contrast
    ax.add_patch(Circle((disc_x, strip_y), radius, facecolor=str(1 - contrast),
                        edgecolor='green' if correct else 'red', lw=3,
                        label='disc'))
    ax.text(disc_x, strip_y, f'{contrast * 100:.3g}%', ha='center',
            va='center', color='red' if contrast > 0.5 else 'black', fontsize=8)

    bar_y, bar_h = strip_y + 0.2, 0.06
    left_w = strip_w * prob_left
    ax.add_patch(Rectangle((strip_x, bar_y), left_w, bar_h,
                           facecolor='tab:blue', label='bias_left'))
    ax.add_patch(Rectangle((strip_x + left_w, bar_y), strip_w - left_w, bar_h,
                           facecolor='tab:orange', label='bias_right'))


def histogram_by_type(
    df: pd.DataFrame,
    measure: str,
    session_types: tuple[str, ...],
    bins: int = 30,
    edge_source: pd.DataFrame | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Split a cohort measure into per-session-type density histograms.

    Bin edges are fixed to the `edge_source[measure]` range (the full cohort)
    while per-type density counts are taken from `df` (the population-filtered
    subset), so the histograms reshape to the current population without the
    bins shifting as it changes. When `edge_source is None`, edges come from
    `df` itself — identical to taking counts and edges from one frame. For each
    type in `session_types`, returns its density-normalized counts over the
    shared edges; a type with no rows in `df` (e.g. filtered out of the
    population) yields all-zero counts. NaN measure values are dropped. Returns
    `(bin_edges, {session_type: density_counts})`.
    """
    edge_frame = edge_source if edge_source is not None else df
    bin_edges = np.histogram_bin_edges(edge_frame[measure].dropna(), bins=bins)
    per_type = {}
    for session_type in session_types:
        values = df.loc[df['session_type'] == session_type, measure].dropna()
        per_type[session_type] = (
            np.histogram(values, bins=bin_edges, density=True)[0]
            if len(values) else np.zeros(len(bin_edges) - 1))
    return bin_edges, per_type


def keypoint_colors(keypoints: list[str]) -> dict[str, tuple]:
    """Map each keypoint to a distinct `tab10` color, keyed by name.

    Colors are assigned over the sorted keypoint names so the same keypoint
    always gets the same color across frames and sessions. `tab10` holds 10
    distinct colors; with more keypoints the cycle repeats.
    """
    palette = colormaps['tab10'].colors
    return {
        keypoint: palette[i % len(palette)]
        for i, keypoint in enumerate(sorted(keypoints))
    }


def trial_frame_window(
    stimOn_time: float, feedback_time: float
) -> tuple[float, float]:
    """Return the frame-extraction window for a trial: 0.1 s before stimulus
    onset to 0.5 s after feedback, so the pre-stimulus baseline and the
    post-feedback consummatory response are both visible.
    """
    return (stimOn_time - 0.1, feedback_time + 0.5)


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


def update_pose_qc(
    pose_path: str | Path, eid: str, qc_lp: str, qc_movement: str
) -> None:
    """Mirror the two manual QC labels for `eid` into the pose roll-up parquet.

    Read-modify-write of `pose_path`: updates only the `qc_lp`/`qc_movement`
    cells for `eid`, leaving every other row and column untouched, so the
    derived roll-up stays in sync with the per-session H5 without re-running
    `scripts/pose.py --rollup`.
    """
    df = pd.read_parquet(pose_path)
    mask = df['eid'] == eid
    for label, value in zip(LP_QC_LABELS, (qc_lp, qc_movement)):
        df.loc[mask, label] = value
    df.to_parquet(pose_path)


def save_label(
    h5_path: str | Path, pose_path: str | Path, eid: str,
    qc_lp: str, qc_movement: str,
) -> str:
    """Persist both manual QC labels for `eid` and return a status line.

    Writes the values to the per-session `video` H5 attrs (the canonical
    store) and mirrors them into the pose roll-up parquet. Catches write
    failures (missing file/`video` group, unwritable path) so a bad session
    surfaces in the GUI status bar instead of crashing the viewer. Returns a
    confirmation naming the H5 file on success, or an error description.
    """
    try:
        persist_labels(h5_path, qc_lp, qc_movement)
        update_pose_qc(pose_path, eid, qc_lp, qc_movement)
    except (OSError, KeyError) as error:
        return f'Save failed for {eid}: {error}'
    return (f'Saved qc_lp={qc_lp}, qc_movement={qc_movement} '
            f'→ {Path(h5_path).name} + pose.pqt')


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
    ``session_type`` + ``fraction_correct``) and the session H5 directory.
    Drives the histogram-brush → dropdown filtering and assembles the
    trial-averaged panel data for a selected session.
    """

    def __init__(
        self,
        df_cohort: pd.DataFrame,
        h5_dir: str | Path,
        pose_path: str | Path | None = None,
    ):
        self.df_cohort = df_cohort
        self.h5_dir = Path(h5_dir)
        self.pose_path = pose_path

    def population_mask(
        self,
        session_types: tuple[str, ...],
        qc_selections: dict[str, set[str]],
        fc_range: tuple[float, float] | None,
        start_time_range: tuple[float, float] | None,
    ) -> np.ndarray:
        """Boolean mask over `df_cohort` for the population-filter selections
        (session types, video-QC grid, `fraction_correct`, `start_time`)."""
        return select_population(
            self.df_cohort, session_types, qc_selections, fc_range,
            start_time_range)

    def dropdown_eids(
        self,
        metric_ranges: dict[str, tuple[float, float]],
        session_types: tuple[str, ...],
        qc_selections: dict[str, set[str]],
        fc_range: tuple[float, float] | None,
        start_time_range: tuple[float, float] | None,
    ) -> list[str]:
        """Session-dropdown eids: the population mask AND the movement-metric
        brushes in `metric_ranges` (the population predicate refined by the
        selection-only histogram brushes)."""
        mask = self.population_mask(
            session_types, qc_selections, fc_range, start_time_range)
        return filter_dropdown(self.df_cohort, mask, metric_ranges)

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
        match = self.df_cohort.loc[
            self.df_cohort['eid'] == eid, 'fraction_correct']
        fraction_correct = match.item() if len(match) else None
        return SessionPanels(
            times=traces.coords['time'].values,
            traces=trial_means,
            xcorr=xcorr,
            fraction_correct=None if pd.isna(fraction_correct) else fraction_correct,
        )


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
HISTOGRAM_TITLES = {
    'paw': 'paw speed @ firstMovement',
    'nose': 'nose speed @ stimOn',
    'tongue_speed': 'tongue speed @ feedback',
    'tongue_likelihood': 'tongue likelihood @ feedback',
}


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
        self.brush_ranges: dict[str, tuple[float, float]] = {}
        self.fc_range: tuple[float, float] | None = None
        self.start_time_range: tuple[float, float] | None = None
        self.frame_source: FrameSource | None = None
        self.trial_frames = np.array([], dtype=int)
        self.frame_pos = 0
        self.setWindowTitle('LPViewer — LightningPose output QC')

        central = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(central)
        layout.addWidget(self._build_controls(), stretch=1)
        layout.addWidget(self._build_display(), stretch=2)
        self.setCentralWidget(central)
        self.statusBar().showMessage('Ready')
        self._refresh_dropdown()

    # -- construction ---------------------------------------------------------

    def _build_controls(self) -> QtWidgets.QWidget:
        """Static population-filter panel above the reactive metric-distribution
        panel, then the session dropdown and the two QC selectors."""
        panel = QtWidgets.QWidget()
        box = QtWidgets.QVBoxLayout(panel)

        box.addWidget(self._build_population_panel())
        box.addWidget(self._build_metric_panel())

        box.addWidget(QtWidgets.QLabel('Sessions in range'))
        self.session_combo = QtWidgets.QComboBox()
        # NoFocus so the combo never grabs the keyboard; otherwise it consumes
        # Up/Down (to change its item) before they reach keyPressEvent, which
        # is where Up/Down drive trial navigation. Mouse selection still works.
        self.session_combo.setFocusPolicy(QtCore.Qt.NoFocus)
        # Force a bounded scrollable popup (combobox-popup: 0) so a long session
        # list scrolls instead of clipping or spilling off-screen.
        self.session_combo.setStyleSheet('QComboBox { combobox-popup: 0; }')
        self.session_combo.setMaxVisibleItems(20)
        self.session_combo.currentTextChanged.connect(self._on_session_selected)
        box.addWidget(self.session_combo)

        self.label_combos = {}
        for field in LP_QC_LABELS:
            box.addWidget(QtWidgets.QLabel(field))
            combo = QtWidgets.QComboBox()
            combo.setFocusPolicy(QtCore.Qt.NoFocus)
            combo.addItems([LP_QC_NOT_SET, *IBL_QC_VALUES])
            combo.currentTextChanged.connect(
                lambda value, f=field: self._on_label(f, value))
            self.label_combos[field] = combo
            box.addWidget(combo)

        return panel

    def _build_population_panel(self) -> QtWidgets.QWidget:
        """Static population filters: session-type checkboxes, the video-QC grid,
        and the full-cohort fraction_correct / start_time histograms. Toggling
        any control reshapes the reactive metric panel and refilters the
        dropdown; these distributions themselves never reshape."""
        panel = QtWidgets.QWidget()
        box = QtWidgets.QVBoxLayout(panel)
        box.setSpacing(2)
        box.setContentsMargins(4, 4, 4, 4)

        box.addWidget(QtWidgets.QLabel('Session types'))
        type_row = QtWidgets.QHBoxLayout()
        self.type_checks = []
        for session_type in sorted(self.model.df_cohort['session_type'].dropna().unique()):
            check = QtWidgets.QCheckBox(session_type)
            check.setChecked(True)
            check.stateChanged.connect(lambda _: self._on_population_changed())
            self.type_checks.append(check)
            type_row.addWidget(check)
        box.addLayout(type_row)

        box.addWidget(QtWidgets.QLabel('Video QC'))
        box.addLayout(self._build_qc_grid())

        self.population_fig = Figure(figsize=(4, 3.6))
        self.population_canvas = FigureCanvasQTAgg(self.population_fig)
        self.population_canvas.mpl_connect(
            'button_press_event', self._on_population_hist_click)
        self._draw_population_histograms()
        box.addWidget(self.population_canvas)

        return panel

    def _build_qc_grid(self) -> QtWidgets.QGridLayout:
        """Field × verdict grid of independent toggle cells; none checked by
        default. Rows are `VIDEO_QC_COLS` (the `qc_videoLeft_` prefix stripped
        for the label), columns the five `QC_VALUE_ORDER` verdicts. Populates
        `self.qc_grid[field][verdict]` with the cell checkboxes that
        `_qc_selections` reads."""
        grid = QtWidgets.QGridLayout()
        grid.setVerticalSpacing(2)
        for col, verdict in enumerate(QC_VALUE_ORDER, start=1):
            grid.addWidget(QtWidgets.QLabel(verdict), 0, col)
        self.qc_grid = {}
        for row, field in enumerate(VIDEO_QC_COLS, start=1):
            grid.addWidget(
                QtWidgets.QLabel(field.removeprefix('qc_videoLeft_')), row, 0)
            self.qc_grid[field] = {}
            for col, verdict in enumerate(QC_VALUE_ORDER, start=1):
                cell = QtWidgets.QCheckBox()
                cell.stateChanged.connect(lambda _: self._on_population_changed())
                self.qc_grid[field][verdict] = cell
                grid.addWidget(cell, row, col)
        return grid

    def _draw_population_histograms(self) -> None:
        """Draw the full-cohort fraction_correct and start_time distributions,
        each with a brushable SpanSelector. `start_time` is plotted on the
        numeric epoch-nanosecond axis (`start_time_to_numeric`) with date-string
        tick labels, so a brush's extents are directly the predicate's numeric
        `start_time_range`. Stored ranges are re-applied as visible spans."""
        self.population_fig.clear()
        self.population_selectors = {}
        ax_fc, ax_time = self.population_fig.subplots(2, 1)

        _, fc_edges, _ = ax_fc.hist(
            self.model.df_cohort['fraction_correct'].dropna(), bins=30)
        # Pin x-limits to the bin span (also disables autoscale) so the
        # interactive SpanSelector's handle artists, which start at x=0, can't
        # drag the data limits and balloon the axis range.
        ax_fc.set_xlim(fc_edges[0], fc_edges[-1])
        ax_fc.set_ylabel('fraction_correct', fontsize=7)
        ax_fc.set_yticks([])
        ax_fc.tick_params(axis='x', labelsize=6)
        self._add_population_selector('fraction_correct', ax_fc, self.fc_range)

        numeric = start_time_to_numeric(self.model.df_cohort['start_time'])
        _, time_edges, _ = ax_time.hist(numeric[~np.isnan(numeric)], bins=20)
        ax_time.set_xlim(time_edges[0], time_edges[-1])
        ax_time.set_ylabel('start_time', fontsize=7)
        ax_time.set_yticks([])
        ax_time.xaxis.set_major_formatter(FuncFormatter(
            lambda ns, _: pd.Timestamp(int(ns)).strftime('%Y-%m-%d')))
        ax_time.tick_params(axis='x', labelsize=6)
        self._add_population_selector(
            'start_time', ax_time, self.start_time_range)

        self.population_fig.tight_layout()
        self.population_canvas.draw_idle()

    def _add_population_selector(
        self, measure: str, ax, current_range: tuple[float, float] | None,
    ) -> None:
        """Attach a horizontal SpanSelector for `measure` to `ax`, restoring
        `current_range` as a visible span when one is set."""
        selector = SpanSelector(
            ax, lambda lo, hi, m=measure: self._on_population_brush(m, lo, hi),
            'horizontal', useblit=True, interactive=True)
        if current_range is not None:
            selector.extents = current_range
        self.population_selectors[measure] = selector

    def _build_metric_panel(self) -> QtWidgets.QWidget:
        """Reactive panel: the four movement-metric histograms, reshaped to the
        current population and brushable to refine the dropdown."""
        panel = QtWidgets.QWidget()
        box = QtWidgets.QVBoxLayout(panel)
        box.addWidget(QtWidgets.QLabel('Metric distributions'))
        self.hist_fig = Figure(figsize=(4, 5.4))
        self.hist_canvas = FigureCanvasQTAgg(self.hist_fig)
        self.hist_canvas.mpl_connect('button_press_event', self._on_hist_click)
        self._draw_histograms()
        box.addWidget(self.hist_canvas)
        return panel

    def _build_display(self) -> QtWidgets.QWidget:
        """The event-locked + cross-correlation panels and the frame viewer."""
        panel = QtWidgets.QWidget()
        box = QtWidgets.QVBoxLayout(panel)

        self.title_label = QtWidgets.QLabel('—')
        box.addWidget(self.title_label)

        self.panel_fig = Figure(figsize=(8, 5))
        self.panel_canvas = FigureCanvasQTAgg(self.panel_fig)
        box.addWidget(self.panel_canvas)

        self.frame_fig = Figure(figsize=(5, 4))
        self.frame_canvas = FigureCanvasQTAgg(self.frame_fig)
        self.frame_ax = self.frame_fig.add_subplot(111)
        self.frame_ax.set_axis_off()
        frame_row = QtWidgets.QHBoxLayout()
        frame_row.addWidget(self.frame_canvas)
        event_col = QtWidgets.QVBoxLayout()
        self.schematic_fig = Figure(figsize=(3, 2))
        self.schematic_canvas = FigureCanvasQTAgg(self.schematic_fig)
        self.schematic_ax = self.schematic_fig.add_subplot(111)
        self.schematic_ax.set_axis_off()
        event_col.addWidget(self.schematic_canvas)
        self.event_labels = [QtWidgets.QLabel('—') for _ in range(3)]
        for label in self.event_labels:
            event_col.addWidget(label)
        frame_row.addLayout(event_col)
        box.addLayout(frame_row)

        controls = QtWidgets.QHBoxLayout()
        self.trial_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.trial_slider.setEnabled(False)
        # Don't let the slider grab focus, or it consumes the arrow keys before
        # they reach keyPressEvent (Left/Right step frames, Up/Down step trials).
        self.trial_slider.setFocusPolicy(QtCore.Qt.NoFocus)
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
        """Per cohort measure, overlay one density histogram per checked session
        type (colored via SESSIONTYPE2COLOR, alpha < 1). Counts come from the
        population-filtered subset while bin edges stay pinned to the full
        cohort, so the panels reshape to the current population without bins
        shifting. Each axes carries a brushable SpanSelector; stored brush
        ranges are re-applied as visible spans. Rebuilds the figure from scratch
        so it tracks the checkbox state. Double-clicking an axes clears its
        range."""
        self.hist_fig.clear()
        self.span_selectors = {}
        self.hist_axes = {}
        types = self._selected_types()
        mask = self.model.population_mask(
            types, self._qc_selections(), self.fc_range, self.start_time_range)
        population = self.model.df_cohort.loc[mask]
        for i, measure in enumerate(HISTOGRAM_MEASURES):
            ax = self.hist_fig.add_subplot(len(HISTOGRAM_MEASURES), 1, i + 1)
            edges, per_type = histogram_by_type(
                population, measure, types,
                edge_source=self.model.df_cohort)
            for session_type, density in per_type.items():
                ax.stairs(density, edges, fill=True, alpha=0.5,
                          color=SESSIONTYPE2COLOR.get(session_type),
                          label=session_type)
            ax.set_title(HISTOGRAM_TITLES[measure], fontsize=8)
            if i == 0 and types:
                ax.legend(fontsize=6)
            selector = SpanSelector(
                ax, lambda lo, hi, m=measure: self._on_brush(m, lo, hi),
                'horizontal', useblit=True, interactive=True)
            if measure in self.brush_ranges:
                selector.extents = self.brush_ranges[measure]
            self.span_selectors[measure] = selector
            self.hist_axes[ax] = measure
        self.hist_fig.tight_layout()
        self.hist_canvas.draw_idle()

    def _on_population_changed(self) -> None:
        """Any population-filter toggle (session type or video-QC cell): reshape
        the reactive metric panel and refilter the dropdown."""
        self._draw_histograms()
        self._refresh_dropdown()

    def _on_population_brush(self, measure: str, low: float, high: float) -> None:
        """Store a fraction_correct / start_time brush range, then reshape the
        metric panel and refilter the dropdown."""
        if measure == 'fraction_correct':
            self.fc_range = (low, high)
        else:
            self.start_time_range = (low, high)
        self._on_population_changed()

    def _on_population_hist_click(self, event) -> None:
        """Double-clicking a population histogram clears its stored range."""
        if not event.dblclick:
            return
        for measure, selector in self.population_selectors.items():
            if event.inaxes is selector.ax:
                if measure == 'fraction_correct':
                    self.fc_range = None
                else:
                    self.start_time_range = None
                selector.clear()
                self.population_canvas.draw_idle()
                self._on_population_changed()
                return

    # -- filtering ------------------------------------------------------------

    def _selected_types(self) -> tuple[str, ...]:
        return tuple(
            check.text() for check in self.type_checks if check.isChecked())

    def _qc_selections(self) -> dict[str, set[str]]:
        """Per-field set of checked verdicts from the video-QC grid; a field with
        no checked cell maps to an empty set (unconstrained downstream)."""
        return {
            field: {verdict for verdict, cell in cells.items() if cell.isChecked()}
            for field, cells in self.qc_grid.items()
        }

    def _on_brush(self, measure: str, low: float, high: float) -> None:
        self.brush_ranges[measure] = (low, high)
        self._refresh_dropdown()

    def _on_hist_click(self, event) -> None:
        """Double-clicking a histogram clears that measure's range."""
        if not event.dblclick or event.inaxes not in self.hist_axes:
            return
        measure = self.hist_axes[event.inaxes]
        if self.brush_ranges.pop(measure, None) is not None:
            self.span_selectors[measure].clear()
            self.hist_canvas.draw_idle()
            self._refresh_dropdown()

    def _refresh_dropdown(self) -> None:
        """Repopulate the session dropdown from the population predicate (types,
        video-QC grid, fraction_correct, start_time) AND the movement-metric
        brushes. Does not auto-load a session — loading happens only when the
        user picks an entry."""
        eids = self.model.dropdown_eids(
            self.brush_ranges, self._selected_types(), self._qc_selections(),
            self.fc_range, self.start_time_range)
        self.session_combo.blockSignals(True)
        self.session_combo.clear()
        self.session_combo.addItems(eids)
        self.session_combo.blockSignals(False)

    # -- session display ------------------------------------------------------

    def _on_session_selected(self, eid: str) -> None:
        if not eid:
            return
        self.current_eid = eid
        panels = self.model.session_panels(eid)
        self._draw_panels(panels)
        row = self.model.df_cohort.loc[self.model.df_cohort['eid'] == eid].iloc[0]
        self.title_label.setText(format_session_title(
            eid, row['subject'], row['start_time'], row['session_type'],
            panels.fraction_correct))
        self._sync_label_combos(eid)
        self._load_frame_source(eid)

    def _draw_panels(self, panels: SessionPanels) -> None:
        """Draw the three event-locked panels and the cross-correlation panel."""
        self.panel_fig.clear()
        ax_paw, ax_nose, ax_tongue, ax_xcorr = self.panel_fig.subplots(2, 2).ravel()

        ax_paw.plot(panels.times, panels.traces['paw'])
        ax_paw.set_title('Paw @ firstMovement')
        ax_nose.plot(panels.times, panels.traces['nose'])
        ax_nose.set_title('Nose @ stimOn')
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
        """Apply a QC verdict and persist it immediately to H5 + pose.pqt.

        Saving on every set (rather than only on close) keeps a manual verdict
        from being lost if the viewer exits abnormally; the outcome is echoed
        to the status bar.
        """
        if value == LP_QC_NOT_SET or not getattr(self, 'current_eid', None):
            return
        eid = self.current_eid
        self.model.df_cohort = apply_label(
            self.model.df_cohort, eid, field, value)
        row = self.model.df_cohort.loc[
            self.model.df_cohort['eid'] == eid].iloc[0]
        status = save_label(self.model.h5_dir / f'{eid}.h5',
                            self.model.pose_path, eid,
                            row['qc_lp'], row['qc_movement'])
        self.statusBar().showMessage(status)

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
        self.keypoint_colors = keypoint_colors(self.frame_source.keypoints)
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
        self.current_trial_idx = trial_idx
        trial = self.trials.iloc[trial_idx]
        draw_trial_schematic(self.schematic_ax, **trial_schematic_values(trial))
        self.schematic_canvas.draw_idle()
        low, high = trial_frame_window(
            trial['stimOn_times'], trial['feedback_times'])
        self.trial_frames = frames_in_trial(
            self.frame_source.camera_times, low, high)
        self.frame_pos = 0
        self._draw_frame()

    def _step_frame(self, delta: int) -> None:
        if len(self.trial_frames) == 0:
            return
        self.frame_pos = int(np.clip(
            self.frame_pos + delta, 0, len(self.trial_frames) - 1))
        self._draw_frame()

    def _step_trial(self, delta: int) -> None:
        """Move the trial slider by `delta`; its valueChanged drives the frame
        view, so slider and frames stay in sync."""
        self.trial_slider.setValue(self.trial_slider.value() + delta)

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
        for name, (x, y, likelihood) in keypoints.items():
            self.frame_ax.scatter(x, y, s=30, color=self.keypoint_colors[name],
                                  alpha=float(likelihood_to_alpha(likelihood)))
        self.frame_ax.set_title(
            f'trial {self.current_trial_idx} · frame {frame_idx}')
        self._update_event_labels(frame_idx)
        self.frame_canvas.draw_idle()

    def _update_event_labels(self, frame_idx: int) -> None:
        """Set the three event-timing labels for the current frame relative to
        the current trial's stimOn, firstMovement, and feedback times."""
        trial = self.trials.iloc[self.current_trial_idx]
        frame_time = self.frame_source.camera_times[frame_idx]
        timings = format_event_timings(
            frame_time, trial['stimOn_times'], trial['firstMovement_times'],
            trial['feedback_times'])
        for label, text in zip(self.event_labels, timings):
            label.setText(text)

    def keyPressEvent(self, event) -> None:
        if event.key() == QtCore.Qt.Key_Left:
            self._step_frame(-1)
        elif event.key() == QtCore.Qt.Key_Right:
            self._step_frame(1)
        elif event.key() == QtCore.Qt.Key_Up:
            self._step_trial(-1)
        elif event.key() == QtCore.Qt.Key_Down:
            self._step_trial(1)
        else:
            super().keyPressEvent(event)

    # -- persistence ----------------------------------------------------------

    def closeEvent(self, event) -> None:
        """Release the video handle. Labels are already persisted on each set."""
        if self.frame_source is not None:
            self.frame_source.close()
        super().closeEvent(event)
