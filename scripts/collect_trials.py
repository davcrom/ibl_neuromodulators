"""
Collect Trials Data

Loops over proficient sessions and collects all trials data from H5 files
into a single parquet file with eid, subject, and session_n metadata.

Output: data/trials.pqt

Usage:
    python scripts/collect_trials.py
"""
import argparse

import pandas as pd
from tqdm import tqdm

from iblnm.config import (
    PROJECT_ROOT, SESSIONS_FPATH, SESSIONS_H5_DIR,
    ANALYSIS_QC_BLOCKERS, SESSION_TYPES_TO_ANALYZE, TARGETNMS_TO_ANALYZE,
)
from iblnm.analysis import state_dwell_times
from iblnm.data import PhotometrySessionGroup
from iblnm.io import _get_default_connection

OUTPUT_FPATH = PROJECT_ROOT / 'data' / 'trials.pqt'

# The raw task columns exported verbatim, in export order. A session whose
# stored table lacks any of them is dropped whole, so every run writes the same
# columns and no cell is empty because one extractor never produced it.
RAW_COLUMNS = [
    'intervals_0', 'intervals_1',
    'stimOnTrigger_times', 'stimOn_times',
    'stimOffTrigger_times', 'stimOff_times',
    'goCueTrigger_times', 'goCue_times',
    'firstMovement_times', 'response_times', 'feedback_times',
    'feedbackType', 'choice',
    'contrastLeft', 'contrastRight',
    'probabilityLeft', 'rewardVolume', 'quiescencePeriod',
]

# The columns whose NaNs mean a trial is incomplete. The four excluded ones are
# NaN by construction rather than by failure: exactly one of the two contrast
# columns holds a value on any trial, and `stimOn_times` and
# `firstMovement_times` are routinely absent on no-choice trials, which
# `no_choice` already flags.
SCANNED_COLUMNS = [
    column for column in RAW_COLUMNS
    if column not in ('contrastLeft', 'contrastRight',
                      'stimOn_times', 'firstMovement_times')
]

# Session identity, copied onto every one of that session's trials.
IDENTITY_COLUMNS = ['subject', 'eid', 'day_n', 'session_n', 'session_type']

# Seconds from go cue to feedback below which a response is taken to have been
# committed before the stimulus could have driven it.
FALSE_START_THRESHOLD = 0.05

# The per-trial flags, each reported on its own. They are independent by
# construction, so a mouse's three fractions add rather than nest.
FLAGS = ('false_start', 'no_choice', 'incomplete')


def build_export(trials: pd.DataFrame, session: pd.Series) -> pd.DataFrame | None:
    """Turn one session's stored trials table into its export rows.

    Parameters
    ----------
    trials : pandas.DataFrame
        The session's stored `trials/table`, one row per trial.
    session : pandas.Series
        That session's catalog row, carrying the `IDENTITY_COLUMNS`.

    Returns
    -------
    pandas.DataFrame or None
        One row per trial, carrying the identity columns, `trial_n`, the 18
        `RAW_COLUMNS`, `stim_side`, `reaction_time` in seconds and the three
        flags, in that order. `None` when `trials` lacks any raw column, which
        drops the session from the export.
    """
    if not set(RAW_COLUMNS).issubset(trials.columns):
        return None
    export = pd.DataFrame({column: session[column] for column in IDENTITY_COLUMNS},
                          index=trials.index)
    export['trial_n'] = trials['trial']
    export[RAW_COLUMNS] = trials[RAW_COLUMNS]
    export['stim_side'] = trials['stim_side']
    no_choice = trials['choice'] == 0
    # On a no-choice trial feedback lands at the response-window timeout, a
    # constant rather than a measurement of anything the mouse did.
    export['reaction_time'] = (trials['feedback_times']
                               - trials['goCue_times']).mask(no_choice)
    # A NaN reaction time compares False, so no-choice trials never false-start.
    export['false_start'] = export['reaction_time'] < FALSE_START_THRESHOLD
    export['no_choice'] = no_choice
    one_contrast = trials[['contrastLeft', 'contrastRight']].notna().sum(axis=1) == 1
    export['incomplete'] = trials[SCANNED_COLUMNS].isna().any(axis=1) | ~one_contrast
    return export


def _flag_summary(trials: pd.DataFrame, subject: str, flag: str) -> dict:
    """One report row: how often `flag` fired over `trials`, and in what runs.

    The run lengths are counted per eid, so a stretch of flagged trials ending
    one session and another opening the next are two runs rather than one.
    """
    flagged = trials[flag]
    runs = state_dwell_times(flagged.to_numpy(), trials['eid'].to_numpy())
    lengths = runs.loc[runs['state'].astype(bool), 'length']
    return {'subject': subject, 'flag': flag,
            'n_trials': len(trials), 'n_flagged': int(flagged.sum()),
            'fraction': flagged.mean(),
            'run_q1': lengths.quantile(0.25),
            'run_median': lengths.quantile(0.5),
            'run_q3': lengths.quantile(0.75),
            'run_max': lengths.max()}


def flag_report(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize each flag's prevalence per mouse and over every mouse.

    Parameters
    ----------
    df : pandas.DataFrame
        The trials export, carrying `subject`, `eid` and the `FLAGS` columns.

    Returns
    -------
    pandas.DataFrame
        One row per (subject, flag) plus one per flag with `subject` set to
        `'all'`, carrying `n_trials`, `n_flagged` and `fraction`. The pooled row
        is computed over every trial, not averaged over the per-mouse rows.
    """
    rows = [_flag_summary(trials, subject, flag)
            for subject, trials in df.groupby('subject', sort=True)
            for flag in FLAGS]
    rows += [_flag_summary(df, 'all', flag) for flag in FLAGS]
    return pd.DataFrame(rows)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args(argv)


if __name__ == '__main__':
    parse_args()

    # Load sessions and create group (same pattern as task_encoding.py)
    print(f"Loading sessions from {SESSIONS_FPATH}")
    df = pd.read_parquet(SESSIONS_FPATH)

    one = _get_default_connection()
    group = PhotometrySessionGroup.from_catalog(df, one=one, h5_dir=SESSIONS_H5_DIR)
    group.filter_sessions(
        session_types=SESSION_TYPES_TO_ANALYZE,
        qc_blockers=ANALYSIS_QC_BLOCKERS,
        targetnms=TARGETNMS_TO_ANALYZE,
    )
    # Deduplicate to unique sessions (filter gives one row per recording)
    df_sessions = group.sessions.drop_duplicates(subset='eid')
    print(f"  {len(df_sessions)} sessions after filtering")

    # Collect trials from the store, now that every session in scope holds them.
    all_trials = []
    n_missing = 0
    for _, row in tqdm(df_sessions.iterrows(), total=len(df_sessions), desc='Loading trials'):
        ps = group._get_session(row)
        if not ps.filepath.exists():
            n_missing += 1
            continue
        ps.load_h5(groups=['trials'])
        if not hasattr(ps, 'trials'):
            n_missing += 1
            continue
        trials = ps.trials.copy()
        trials['eid'] = row['eid']
        trials['subject'] = row['subject']
        trials['session_n'] = row['session_n']
        trials['session_type'] = row['session_type']
        all_trials.append(trials)

    if n_missing:
        print(f"  Skipped {n_missing} sessions (no H5 or no trials group)")

    df_trials = pd.concat(all_trials, ignore_index=True)
    print(f"  {len(df_trials)} total trials from {len(all_trials)} sessions")

    df_trials.to_parquet(OUTPUT_FPATH)
    print(f"Saved to {OUTPUT_FPATH}")
