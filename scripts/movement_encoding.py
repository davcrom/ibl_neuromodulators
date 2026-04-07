"""
Movement Encoding Analysis

Tests whether trial timing variables (reaction time, movement time) explain
NM responses at stimulus onset, and how that compares to contrast.

Three analyses:
  1. LOSO cross-validated model comparison — per-subject delta-R² from
     dropping contrast vs. timing
  2. Per-contrast slope analysis — timing slope at each contrast level
  3. Descriptive plots — NM response vs. timing at each contrast level

Output:
    data/movement_encoding/          — CSV summary tables
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
    RESPONSES_FPATH, TRIAL_TIMING_FPATH,
    FIGURE_DPI,
    ANALYSIS_QC_BLOCKERS, SESSION_TYPES_TO_ANALYZE, TARGETNMS_TO_ANALYZE,
)
from iblnm.data import PhotometrySessionGroup
from iblnm.io import _get_default_connection
from iblnm.task import add_relative_contrast
from iblnm.util import collect_session_errors
from iblnm.analysis import loso_cv_movement_lmm, fit_movement_lmm_per_contrast
from iblnm.vis import (
    plot_movement_response,
    plot_movement_lmm_summary,
    plot_movement_slopes,
)

plt.ion()

TIMING_VARS = ['reaction_time', 'movement_time']


if __name__ == '__main__':

    # =====================================================================
    # Output directories
    # =====================================================================
    data_dir = PROJECT_ROOT / 'data/movement_encoding'
    data_dir.mkdir(parents=True, exist_ok=True)

    fig_dir = PROJECT_ROOT / 'figures/movement_encoding'
    fig_dirs = {
        'descriptive': fig_dir / 'descriptive',
        'model_comparison': fig_dir / 'model_comparison',
        'slopes': fig_dir / 'slopes',
    }
    for d in fig_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

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
        trial_timing[['eid', 'trial'] + TIMING_VARS + ['response_time']],
        on=['eid', 'trial'], how='left',
    )
    df_resp = df_resp.query('response_time > 0.05')
    df_resp = df_resp.dropna(subset=['response_early'])

    for var in TIMING_VARS:
        log_col = f'log_{var}'
        df_resp[log_col] = np.where(
            df_resp[var] > 0, np.log10(df_resp[var]), np.nan)

    print(f"  {len(df_resp)} trials after filtering")

    # =====================================================================
    # Descriptive plots: NM response vs timing, per (target_NM, contrast)
    # =====================================================================
    print("\nGenerating descriptive plots...")
    for (target_nm, contrast), df_group in df_resp.groupby(['target_NM', 'contrast']):
        if df_group['subject'].nunique() < 2:
            continue
        for var in TIMING_VARS:
            log_col = f'log_{var}'
            df_valid = df_group.dropna(subset=[log_col])
            if len(df_valid) < 20:
                continue
            fig = plot_movement_response(
                df_valid, 'response_early', log_col, target_nm, contrast)
            fname = f'{target_nm}_stimOn_{var}_c{contrast:g}.svg'
            fig.savefig(fig_dirs['descriptive'] / fname,
                        dpi=FIGURE_DPI, bbox_inches='tight')
            plt.close(fig)
    print(f"  Descriptive figures saved to {fig_dirs['descriptive']}")

    # =====================================================================
    # Analysis 1: LOSO cross-validated model comparison
    # =====================================================================
    print("\nRunning LOSO-CV model comparisons...")
    cv_frames = []
    for target_nm, df_tnm in df_resp.groupby('target_NM'):
        for var in TIMING_VARS:
            log_col = f'log_{var}'
            df_valid = df_tnm.dropna(subset=[log_col])
            if len(df_valid) < 20:
                continue
            df_cv = loso_cv_movement_lmm(
                df_valid, 'response_early', log_col)
            if df_cv.empty:
                continue
            df_cv['target_NM'] = target_nm
            cv_frames.append(df_cv)
            mean_dc = df_cv['delta_r2_contrast'].mean()
            mean_dt = df_cv['delta_r2_timing'].mean()
            n_subj = len(df_cv)
            print(f"  {target_nm} x {var} ({n_subj} subjects): "
                  f"mean ΔR²_contrast={mean_dc:.4f}, "
                  f"mean ΔR²_timing={mean_dt:.4f}")

    if cv_frames:
        df_cv_all = pd.concat(cv_frames, ignore_index=True)
        df_cv_all.to_csv(
            data_dir / 'loso_cv_model_comparison.csv', index=False)
        print(f"  Saved to {data_dir / 'loso_cv_model_comparison.csv'}")

        fig = plot_movement_lmm_summary(df_cv_all)
        fig.savefig(fig_dirs['model_comparison'] / 'model_comparison.svg',
                    dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close(fig)
        print(f"  Summary figure saved to {fig_dirs['model_comparison']}")
    else:
        print("  No valid model comparisons.")

    # =====================================================================
    # Analysis 2: Per-contrast slope analysis
    # =====================================================================
    print("\nFitting per-contrast slope models...")
    slope_rows = []
    for (target_nm, contrast), df_group in df_resp.groupby(
            ['target_NM', 'contrast']):
        if df_group['subject'].nunique() < 2:
            continue
        for var in TIMING_VARS:
            log_col = f'log_{var}'
            df_valid = df_group.dropna(subset=[log_col])
            if len(df_valid) < 20:
                continue
            result = fit_movement_lmm_per_contrast(
                df_valid, 'response_early', log_col)
            if result is None:
                continue
            result['target_NM'] = target_nm
            result['contrast'] = contrast
            slope_rows.append(result)

    if slope_rows:
        df_slopes = pd.DataFrame(slope_rows)
        df_slopes.to_csv(data_dir / 'per_contrast_slopes.csv', index=False)
        print(f"  Saved {len(df_slopes)} slopes to "
              f"{data_dir / 'per_contrast_slopes.csv'}")

        fig = plot_movement_slopes(df_slopes)
        fig.savefig(fig_dirs['slopes'] / 'timing_slopes_by_contrast.svg',
                    dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close(fig)
        print(f"  Slopes figure saved to {fig_dirs['slopes']}")
    else:
        print("  No valid per-contrast models.")
