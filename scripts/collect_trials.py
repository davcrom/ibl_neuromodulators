"""
Collect Trials Data

Exports every trial of every biased and ephys session in scope as one ordered
table, for a collaborator's per-mouse DDM-HMM fit. Each trial carries the raw
task columns verbatim, a reaction time, and three independent flags describing
it; no trial is dropped and no column recommends an exclusion policy. A summary
of how often each flag fired, per mouse and pooled, is printed afterwards.

Output: data/trials.csv

Usage:
    python scripts/collect_trials.py
"""
import argparse

import pandas as pd
from tqdm import tqdm

from iblnm.config import (
    SESSIONS_FPATH, SESSIONS_H5_DIR, TRIALS_FPATH, ANALYSIS_QC_BLOCKERS,
)
from iblnm.analysis import state_dwell_times
from iblnm.data import PhotometrySessionGroup
from iblnm.io import _get_default_connection

# The error types that disqualify a session from the export. The three
# photometry-side blockers are dropped from the analysis set: a session removed
# because its fiber failed QC leaves a hole the HMM reads as two consecutive
# days, so only behavioral failures remove a day.
BEHAVIOR_QC_BLOCKERS = ANALYSIS_QC_BLOCKERS - {
    'TrialsNotInPhotometryTime', 'QCValidationError', 'AmbiguousRegionMapping'}

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

    print(f"Loading sessions from {SESSIONS_FPATH}")
    df = pd.read_parquet(SESSIONS_FPATH)

    one = _get_default_connection()
    group = PhotometrySessionGroup.from_catalog(df, one=one, h5_dir=SESSIONS_H5_DIR)
    # Target-NM selects the mice, behavioral criteria alone select their days,
    # so the photometry QC filter is off. Training sessions hold probabilityLeft
    # at 0.5, which would change what the HMM's state transitions mean.
    group.filter_sessions(
        session_types=('biased', 'ephys'),
        qc_blockers=BEHAVIOR_QC_BLOCKERS,
        photometry_qc=False,
    )
    group.deduplicate()
    df_sessions = group.sessions
    print(f"  {len(df_sessions)} sessions in scope")

    # `load_trials` fetches from Alyx when a session holds no stored table, so a
    # first run against a store predating the rebuild downloads what it lacks.
    exports = []
    n_skipped = 0
    for _, row in tqdm(df_sessions.iterrows(), total=len(df_sessions),
                       desc='Loading trials'):
        ps = group._get_session(row)
        ps.load_trials()
        export = build_export(ps.trials, row)
        if export is None:
            n_skipped += 1
            continue
        exports.append(export)

    df_trials = pd.concat(exports, ignore_index=True)
    df_trials.to_csv(TRIALS_FPATH, index=False)
    print(f"Saved to {TRIALS_FPATH}")
    print(f"  {n_skipped} sessions skipped for a missing column")
    print(f"  {len(df_trials)} trials from {len(exports)} sessions")
    print(flag_report(df_trials).to_string(index=False))
