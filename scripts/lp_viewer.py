"""
LPViewer launcher

Opens the LightningPose output QC viewer for the pose roll-up cohort. Parallels
scripts/session_viewer.py but is Qt-driven (a QApplication event loop, not
plt.show()).

Builds the cohort table from the pose roll-up (metadata/pose.pqt) enriched with
session metadata (metadata/sessions.pqt) and fraction_correct
(data/performance.pqt), then hands it to LPViewer.

Usage:
    python scripts/lp_viewer.py
"""
import sys

import pandas as pd
from matplotlib.backends.qt_compat import QtWidgets

from iblnm.config import (
    PERFORMANCE_FPATH,
    POSE_FPATH,
    SESSIONS_FPATH,
    SESSIONS_H5_DIR,
)
from iblnm.io import _get_default_connection
from iblnm.lp_viewer import LPViewer, LPViewerModel

# Session-metadata columns the viewer needs: session_type for the cohort filter,
# subject/start_time/number to construct a PhotometrySession for the frame viewer.
META_COLS = ['eid', 'subject', 'start_time', 'number', 'session_type']


def build_cohort(df_pose: pd.DataFrame, df_sessions: pd.DataFrame) -> pd.DataFrame:
    """Enrich the pose roll-up with session metadata, keyed by ``eid``.

    Left-joins ``META_COLS`` from the session catalog onto the pose roll-up, so
    every pose-extracted session gains ``session_type`` (for filtering) and the
    fields needed to load frames. Returns one row per ``df_pose`` eid.
    """
    meta = df_sessions[META_COLS].drop_duplicates('eid')
    return df_pose.merge(meta, on='eid', how='left')


if __name__ == '__main__':
    df_cohort = build_cohort(
        pd.read_parquet(POSE_FPATH), pd.read_parquet(SESSIONS_FPATH))
    df_performance = pd.read_parquet(
        PERFORMANCE_FPATH, columns=['eid', 'fraction_correct'])

    model = LPViewerModel(
        df_cohort, SESSIONS_H5_DIR, df_performance, pose_path=POSE_FPATH)
    one = _get_default_connection()

    app = QtWidgets.QApplication(sys.argv)
    viewer = LPViewer(model, one=one)
    viewer.show()
    sys.exit(app.exec_())
