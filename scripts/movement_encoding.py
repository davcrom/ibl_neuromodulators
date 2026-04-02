"""
Movement Encoding Analysis

Plots NM response at stimulus onset as a function of trial timing variables
(reaction time, movement time, response time), for each target-NM and
contrast level. Timing variables are log-transformed and quantile-binned.

Output:
    figures/movement_encoding/       — SVG figures

Usage:
    python scripts/movement_encoding.py
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from iblnm.config import (
    PROJECT_ROOT, SESSIONS_FPATH, PERFORMANCE_FPATH,
    QUERY_DATABASE_LOG_FPATH, PHOTOMETRY_LOG_FPATH, TASK_LOG_FPATH,
    RESPONSES_DIR, RESPONSES_FPATH, TRIAL_TIMING_FPATH,
    FIGURE_DPI,
    ANALYSIS_QC_BLOCKERS, SESSION_TYPES_TO_ANALYZE, TARGETNMS_TO_ANALYZE,
)
from iblnm.data import PhotometrySessionGroup
from iblnm.io import _get_default_connection
from iblnm.task import add_relative_contrast
from iblnm.util import collect_session_errors
from iblnm.vis import plot_movement_response

plt.ion()

TIMING_VARS = ['reaction_time', 'movement_time', 'response_time']


if __name__ == '__main__':

    # =====================================================================
    # Load sessions and create group
    # =====================================================================
    print(f"Loading sessions from {SESSIONS_FPATH}")
    df = pd.read_parquet(SESSIONS_FPATH)
    df = collect_session_errors(
        df, [QUERY_DATABASE_LOG_FPATH, PHOTOMETRY_LOG_FPATH, TASK_LOG_FPATH])
    if PERFORMANCE_FPATH.exists():
        perf = pd.read_parquet(
            PERFORMANCE_FPATH, columns=['eid', 'fraction_correct', 'contrasts'])
        df = df.merge(perf, on='eid', how='left')

    one = _get_default_connection()
    group = PhotometrySessionGroup.from_catalog(df, one=one)
    group.filter_sessions(
        session_types=SESSION_TYPES_TO_ANALYZE,
        qc_blockers=ANALYSIS_QC_BLOCKERS,
        targetnms=TARGETNMS_TO_ANALYZE,
    )
    print(f"  Recordings (session x region): {len(group)}")

    # =====================================================================
    # Load response magnitudes and trial timing
    # =====================================================================
    if not RESPONSES_FPATH.exists():
        print(f"Error: {RESPONSES_FPATH} not found. Run responses.py first.")
        raise SystemExit(1)
    if not TRIAL_TIMING_FPATH.exists():
        print(f"Error: {TRIAL_TIMING_FPATH} not found. Run responses.py first.")
        raise SystemExit(1)

    group.load_response_magnitudes(RESPONSES_FPATH)
    trial_timing = pd.read_parquet(TRIAL_TIMING_FPATH)
    print(f"  {len(group.response_magnitudes)} response rows loaded")
    print(f"  {len(trial_timing)} trial timing rows loaded")

    # =====================================================================
    # Prepare data: filter, merge, log-transform
    # =====================================================================
    df_resp = group.response_magnitudes.query(
        "event == 'stimOn_times' and probabilityLeft == 0.5 and choice != 0"
    ).copy()
    df_resp = add_relative_contrast(df_resp)
    df_resp = df_resp.merge(
        trial_timing[['eid', 'trial'] + TIMING_VARS],
        on=['eid', 'trial'], how='left',
    )
    df_resp = df_resp.query('response_time > 0.05')
    df_resp = df_resp.dropna(subset=['response_early'])

    # Log-transform timing variables (exclude non-positive values)
    for var in TIMING_VARS:
        log_col = f'log_{var}'
        df_resp[log_col] = np.where(
            df_resp[var] > 0, np.log10(df_resp[var]), np.nan)

    print(f"  {len(df_resp)} trials after filtering")

    # =====================================================================
    # Plot: NM response vs timing, per (target_NM, contrast)
    # =====================================================================
    fig_dir = PROJECT_ROOT / 'figures/movement_encoding'
    fig_dir.mkdir(parents=True, exist_ok=True)

    for (target_nm, contrast), df_group in df_resp.groupby(['target_NM', 'contrast']):
        n_subjects = df_group['subject'].nunique()
        if n_subjects < 2:
            continue

        for var in TIMING_VARS:
            log_col = f'log_{var}'
            df_valid = df_group.dropna(subset=[log_col])
            if len(df_valid) < 20:
                continue

            fig = plot_movement_response(
                df_valid, 'response_early', log_col, target_nm, contrast)
            fname = f'{target_nm}_stimOn_{var}_c{contrast:g}.svg'
            fig.savefig(fig_dir / fname, dpi=FIGURE_DPI, bbox_inches='tight')
            plt.close(fig)

    print(f"\nFigures saved to {fig_dir}")
