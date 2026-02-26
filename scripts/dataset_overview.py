"""
Dataset Overview

Produces four session overview matrices:
1. All registered sessions (post-metadata and session-type filtering)
2. Sessions with raw data (task + photometry)
3. Sessions with complete data (extracted + sufficient trials + TIPT)
4. Sessions passing basic photometry QC

All flags are derived from upstream error logs — no QC parquet is loaded.
No write-back to any upstream file.
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from iblnm.config import (
    PROJECT_ROOT, SESSIONS_FPATH,
    QUERY_DATABASE_LOG_FPATH, PHOTOMETRY_LOG_FPATH,
    TASK_LOG_FPATH,
    ERRORS_FPATH, FIGURE_DPI,
    SUBJECTS_TO_EXCLUDE, SESSION_TYPES_TO_ANALYZE, VALID_TARGETNMS,
    TARGET2NM,
)
from iblnm.util import (
    resolve_duplicate_group,
    concat_logs, deduplicate_log,
    collect_session_errors, fill_empty_lists_from_group, LOG_COLUMNS,
)
from iblnm.vis import (
    session_overview_matrix, target_overview_barplot, mouse_overview_barplot, set_plotsize
)

plt.ion()

# Create output directory
figures_dir = PROJECT_ROOT / 'figures/dataset_overview'
figures_dir.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Load Data
# =============================================================================

print(f"Loading sessions from {SESSIONS_FPATH}")
df_sessions = pd.read_parquet(SESSIONS_FPATH)
n_total = len(df_sessions)

# Attach all upstream error logs (needed for filtering and dedup)
df_sessions = collect_session_errors(
    df_sessions,
    [QUERY_DATABASE_LOG_FPATH, PHOTOMETRY_LOG_FPATH, TASK_LOG_FPATH],
)

# Filter: remove sessions with fatal metadata errors
# ~fatal_errors = {'InvalidNeuromodulator', 'InvalidTarget', 'InvalidTargetNM'}
# ~df_sessions = df_sessions[
    # ~df_sessions['logged_errors'].apply(lambda errs: not any(e in fatal_errors for e in errs))
# ~].copy()
# ~n_after_fatal = len(df_sessions)


# Filter to sessions that are valid for task analysis
df_sessions = df_sessions[
    df_sessions['session_type'].isin(SESSION_TYPES_TO_ANALYZE) &
    ~df_sessions['subject'].isin(SUBJECTS_TO_EXCLUDE)
]
n_after_type = len(df_sessions)

# Deduplicate: one session per (subject, day_n), prefer sessions without disqualifying errors
dup_log = []
df_sessions = (
    df_sessions.groupby(['subject', 'day_n'], group_keys=False)
    .apply(resolve_duplicate_group, exlog=dup_log, include_groups=True)
    .reset_index(drop=True)
)
df_dup_log = pd.DataFrame(dup_log) if dup_log else pd.DataFrame(columns=LOG_COLUMNS)
n_after_dedup = len(df_sessions)

print("\nSession counts:")
print(f"  Registered: {n_total}")
print(f"  After session type filter: {n_after_type} (-{n_total - n_after_type})")
print(f"  After deduplication: {n_after_dedup} (-{n_after_type - n_after_dedup})")


# =============================================================================
# Derive flags from error logs
# =============================================================================

_errs = df_sessions['logged_errors']

# Matrix 2: raw data present (task + photometry)
df_sessions['has_raw_data'] = _errs.apply(
    lambda e: 'MissingRawData' not in e)

# Matrix 3: complete data (extracted + sufficient trials + trials in photometry time)
_complete_blockers = {'MissingExtractedData', 'InsufficientTrials', 'TrialsNotInPhotometryTime'}
df_sessions['has_complete_data'] = _errs.apply(
    lambda e: not any(err in _complete_blockers for err in e))

# Matrix 4: passes basic photometry QC
_qc_blockers = {'MissingExtractedData', 'InsufficientTrials', 'TrialsNotInPhotometryTime',
                'QCValidationError', 'FewUniqueSamples'}
df_sessions['passes_basic_qc'] = _errs.apply(
    lambda e: not any(err in _qc_blockers for err in e))


# =============================================================================
# Session Overview Matrices
# =============================================================================

print("\nGenerating session matrices...")

n_raw = df_sessions['has_raw_data'].sum()
n_complete = df_sessions['has_complete_data'].sum()
n_qc = df_sessions['passes_basic_qc'].sum()


def _save_matrix(df, highlight, title, filename):
    ax = session_overview_matrix(df, highlight=highlight)
    ax.set_title(title)
    set_plotsize(w=48, h=32, ax=ax)
    ax.get_figure().savefig(figures_dir / filename, dpi=FIGURE_DPI, bbox_inches='tight')
    return ax


_save_matrix(df_sessions, 'all',
             f'All registered sessions (n={n_after_dedup})',
             '1_all_sessions.svg')

_save_matrix(df_sessions, lambda df: df['has_raw_data'],
             f'Sessions with raw data (n={n_raw})',
             '2_raw_data.svg')

_save_matrix(df_sessions, lambda df: df['has_complete_data'],
             f'Sessions with complete data (n={n_complete})',
             '3_complete_data.svg')

_save_matrix(df_sessions, lambda df: df['passes_basic_qc'],
             f'Sessions passing basic QC (n={n_qc})',
             '4_passes_qc.svg')


# =============================================================================
# Dataset Summary
# =============================================================================

print(f"\n{'='*50}")
print("Dataset Summary")
print(f"{'='*50}")
print(f"Total sessions (after filtering): {n_after_dedup}")
print(f"  Has raw data: {n_raw}")
print(f"  Has complete data: {n_complete}")
print(f"  Passes basic QC: {n_qc}")


# =============================================================================
# Target-NM Barplots (per recording, QC-passing sessions only)
# =============================================================================

# Fill empty brain_region/hemisphere from other sessions of the same subject
df_sessions = fill_empty_lists_from_group(df_sessions, 'brain_region')
df_sessions = fill_empty_lists_from_group(df_sessions, 'hemisphere')
n_filled = df_sessions['brain_region'].apply(lambda x: len(x) > 0 if isinstance(x, (list, np.ndarray)) else False).sum()
print(f"  After filling from subject group: {n_filled} sessions with brain_region")

# TEMPFIX: normalize brain_region naming errors from Alyx metadata
_REGION_FIXES = {'DRN': 'DR', 'SNC': 'SNc'}

def _fix_regions(regions):
    if not isinstance(regions, (list, np.ndarray)):
        return regions
    fixed = []
    for r in regions:
        bare = r.rsplit('-', 1)[0] if r.endswith(('-l', '-r')) else r
        suffix = r[len(bare):]
        fixed.append(_REGION_FIXES.get(bare, bare) + suffix)
    return fixed

df_sessions['brain_region'] = df_sessions['brain_region'].apply(_fix_regions)

# Rebuild NM and target_NM from corrected brain_region
def _target_nm_from_region(region):
    bare = region.rsplit('-', 1)[0] if region.endswith(('-l', '-r')) else region
    nm = TARGET2NM.get(bare)
    return f'{bare}-{nm}' if nm else None

df_sessions['target_NM'] = df_sessions['brain_region'].apply(
    lambda rs: [_target_nm_from_region(r) for r in rs]
    if isinstance(rs, (list, np.ndarray)) else rs
)
df_sessions['NM'] = df_sessions['target_NM'].apply(
    lambda ts: ts[0].split('-')[-1]
    if isinstance(ts, (list, np.ndarray)) and len(ts) > 0 and ts[0] else None
)

# Explode to one row per recording (brain_region / target_NM are parallel lists)
df_recordings = (
    df_sessions[df_sessions['passes_basic_qc']]
    .explode(['target_NM', 'brain_region', 'hemisphere'])
    .loc[lambda df: df['target_NM'].isin(VALID_TARGETNMS)]
    .copy()
)

if len(df_recordings) > 0:
    n_rec = len(df_recordings)
    n_ses = df_recordings['eid'].nunique()

    ax = target_overview_barplot(df_recordings)
    ax.set_title(f'Complete recordings by target ({n_rec} recordings, {n_ses} sessions)')
    set_plotsize(w=24, h=12, ax=ax)
    ax.get_figure().savefig(figures_dir / '5_target_overview.svg',
                            dpi=FIGURE_DPI, bbox_inches='tight')

    ax = mouse_overview_barplot(df_recordings)
    ax.set_title(f'Mice by target ({n_rec} recordings, {n_ses} sessions)')
    set_plotsize(w=24, h=12, ax=ax)
    ax.get_figure().savefig(figures_dir / '6_mouse_overview.svg',
                            dpi=FIGURE_DPI, bbox_inches='tight')
else:
    print("No complete recordings to plot.")


# =============================================================================
# Save unified error log
# =============================================================================

upstream_logs = [pd.read_parquet(p) for p in
                 [QUERY_DATABASE_LOG_FPATH, PHOTOMETRY_LOG_FPATH, TASK_LOG_FPATH]
                 if p.exists()]
df_errors = deduplicate_log(concat_logs(upstream_logs + [df_dup_log]))

ERRORS_FPATH.parent.mkdir(parents=True, exist_ok=True)
df_errors.to_parquet(ERRORS_FPATH)
print(f"\nSaved {len(df_errors)} error entries to {ERRORS_FPATH}")
print(f"Figures saved to {figures_dir}")
