"""Tests for scripts/collect_trials.py, on synthetic frames only."""
import numpy as np
import pandas as pd

from iblnm import config
import scripts.collect_trials as ct


def test_export_is_written_beside_the_other_data_tables():
    """The export lands in `data/trials/`, one CSV per mouse."""
    assert config.TRIALS_DIR == config.PROJECT_ROOT / 'data/trials'


def test_write_per_subject_writes_one_csv_per_mouse(tmp_path):
    """Each mouse's trials round-trip to `{subject}.csv`, that mouse's only."""
    df = mock_export([('A', 'e1'), ('A', 'e2'), ('B', 'e3')])

    written = ct.write_per_subject(df, tmp_path)

    assert sorted(p.name for p in written) == ['A.csv', 'B.csv']
    a = pd.read_csv(tmp_path / 'A.csv')
    assert set(a['subject']) == {'A'}
    assert sorted(a['eid'].unique()) == ['e1', 'e2']
    assert len(a) == (df['subject'] == 'A').sum()


def test_scope_blocks_on_behavior_errors_only():
    """The photometry-side blockers are dropped; the behavioral ones are kept.

    A session whose fiber failed leaves a hole the collaborator's HMM reads as
    two consecutive days, so photometry never removes one.
    """
    assert ct.BEHAVIOR_QC_BLOCKERS == {
        'MissingExtractedData', 'MissingRawData', 'InsufficientTrials',
        'IncompleteEventTimes', 'MissingBlockInfo',
    }
    assert ct.BEHAVIOR_QC_BLOCKERS < config.ANALYSIS_QC_BLOCKERS


def mock_trials(n_trials=5, **overrides):
    """A stored trials table carrying every raw column plus the derived ones.

    Every trial is complete and answered: a right-side stimulus, feedback
    0.5 s after the go cue. ``overrides`` replaces whole columns, so a test
    states only the values its assertion turns on.
    """
    trial_starts = np.arange(n_trials, dtype=float) * 10
    trials = pd.DataFrame({
        'intervals_0': trial_starts,
        'intervals_1': trial_starts + 9,
        'stimOnTrigger_times': trial_starts + 1,
        'stimOn_times': trial_starts + 1.01,
        'stimOffTrigger_times': trial_starts + 2,
        'stimOff_times': trial_starts + 2.01,
        'goCueTrigger_times': trial_starts + 1,
        'goCue_times': trial_starts + 1.02,
        'firstMovement_times': trial_starts + 1.2,
        'response_times': trial_starts + 1.5,
        'feedback_times': trial_starts + 1.52,
        'feedbackType': np.ones(n_trials),
        'choice': -np.ones(n_trials),
        'contrastLeft': np.full(n_trials, np.nan),
        'contrastRight': np.full(n_trials, 1.0),
        'probabilityLeft': np.full(n_trials, 0.5),
        'rewardVolume': np.full(n_trials, 1.5),
        'quiescencePeriod': np.full(n_trials, 0.4),
        'trial': np.arange(n_trials),
        'stim_side': ['right'] * n_trials,
        'signed_contrast': np.ones(n_trials),
        'contrast': np.ones(n_trials),
    })
    return trials.assign(**overrides)


def mock_session():
    """A catalog row carrying the five identity columns."""
    return pd.Series({
        'subject': 'ZFM-001',
        'eid': 'abc-123',
        'day_n': 7,
        'session_n': 3,
        'session_type': 'biased',
        'NM': 'DA',
        'target_NM': ['VTA-DA'],
    })


EXPECTED_COLUMNS = (
    ['subject', 'eid', 'NM', 'day_n', 'session_n', 'session_type', 'trial_n']
    + ct.RAW_COLUMNS
    + ['stim_side', 'reaction_time', 'false_start', 'no_choice', 'incomplete']
)


def mock_export(subjects_eids, **flags):
    """An exported frame carrying only what `flag_report` reads.

    ``subjects_eids`` is a list of ``(subject, eid)`` pairs, one per trial, in
    order. Every flag is False unless ``flags`` names it with a full column.
    ``session_n`` and ``trial_n`` count the rows as given, so the frame is
    already in the order `write_per_subject` sorts to.
    """
    export = pd.DataFrame(subjects_eids, columns=['subject', 'eid'])
    export['session_n'] = export.groupby('subject')['eid'].transform(
        lambda eids: eids.astype('category').cat.codes + 1)
    export['trial_n'] = export.groupby('eid').cumcount()
    for flag in ct.FLAGS:
        export[flag] = np.asarray(flags.get(flag, np.zeros(len(export))), dtype=bool)
    return export


def test_write_per_subject_sorts_by_session_then_trial(tmp_path):
    """Rows come out in the mouse's own order: session, then trial within it."""
    df = mock_export([('A', 'e2')] * 2 + [('A', 'e1')] * 2)
    df['session_n'] = [2, 2, 1, 1]
    df['trial_n'] = [5, 4, 3, 2]

    ct.write_per_subject(df, tmp_path)

    out = pd.read_csv(tmp_path / 'A.csv')
    assert out['session_n'].tolist() == [1, 1, 2, 2]
    assert out['trial_n'].tolist() == [2, 3, 4, 5]


def test_format_flag_report_gives_one_block_per_flag():
    """Each flag gets its own headed table, in `FLAGS` order, mice within it."""
    export = mock_export([('A', 'a')] * 2 + [('B', 'b')] * 2,
                         false_start=[1, 0, 0, 0])

    text = ct.format_flag_report(ct.flag_report(export))

    headings = [line for line in text.splitlines() if line in ct.FLAGS]
    assert headings == list(ct.FLAGS)
    first, second = text.index('false_start'), text.index('no_choice')
    subjects = text[first:second].count('A') + text[first:second].count('B')
    assert subjects == 2                      # both mice inside the one block
    header = text.splitlines()[1].split()     # the column is the heading now
    assert 'flag' not in header and 'subject' in header


def test_flag_report_pools_over_trials_not_over_mice():
    """The `'all'` row's fraction is total flagged over total trials."""
    export = mock_export(
        [('A', 'a')] * 4 + [('B', 'b')] * 6,
        false_start=[1, 0, 0, 0] + [1, 1, 0, 0, 0, 0],
    )

    report = ct.flag_report(export)
    starts = report[report['flag'] == 'false_start'].set_index('subject')

    assert starts.loc['A', 'fraction'] == 0.25
    assert starts.loc['B', 'fraction'] == 2 / 6
    assert starts.loc['all', 'fraction'] == 0.3
    assert starts.loc['all', 'n_trials'] == 10
    assert starts.loc['all', 'n_flagged'] == 3


def test_flag_report_runs_never_cross_a_session_boundary():
    """Four adjacent flagged trials over two eids are two runs of 2, not one of 4."""
    export = mock_export(
        [('A', 'a')] * 4 + [('A', 'b')] * 4,
        no_choice=[0, 0, 1, 1] + [1, 1, 0, 0],
    )

    report = ct.flag_report(export)
    row = report.query("subject == 'A' and flag == 'no_choice'").iloc[0]

    assert row['run_max'] == 2
    assert row['run_median'] == 2
    assert row['n_flagged'] == 4


def test_flag_report_run_quartiles_over_the_run_lengths():
    """The run columns are the quartiles of the flagged-run lengths."""
    export = mock_export(
        [('A', 'a')] * 12,
        incomplete=[1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
    )

    report = ct.flag_report(export)
    row = report.query("subject == 'A' and flag == 'incomplete'").iloc[0]

    # Run lengths, in order: 1, 2, 3, 3.
    assert row['run_q1'] == 1.75
    assert row['run_median'] == 2.5
    assert row['run_q3'] == 3.0
    assert row['run_max'] == 3


def test_flag_report_unflagged_gives_zero_fraction_and_nan_runs():
    """A flag that never fires has no run to measure."""
    export = mock_export([('A', 'a')] * 4)

    report = ct.flag_report(export)
    row = report.query("subject == 'A' and flag == 'false_start'").iloc[0]

    assert row['fraction'] == 0.0
    assert row['n_flagged'] == 0
    assert row[['run_q1', 'run_median', 'run_q3', 'run_max']].isna().all()


def test_flag_report_one_row_per_mouse_per_flag_plus_the_pooled_rows():
    """Three flags for each of two mice, and three pooled."""
    export = mock_export([('A', 'a')] * 3 + [('B', 'b')] * 3)

    report = ct.flag_report(export)

    assert len(report) == 2 * 3 + 3
    assert set(report['flag']) == set(ct.FLAGS)


def test_build_export_columns_in_order():
    """The export's columns are exactly the fixed set, in the fixed order."""
    export = ct.build_export(mock_trials(n_trials=5), mock_session())

    assert list(export.columns) == EXPECTED_COLUMNS
    assert len(export) == 5


def test_build_export_no_choice_trial_has_no_reaction_time():
    """A no-choice trial is flagged, and its timeout is not a reaction time."""
    trials = mock_trials(n_trials=3)
    trials.loc[1, 'choice'] = 0.0
    trials.loc[1, 'feedback_times'] = trials.loc[1, 'goCue_times'] + 60.029

    export = ct.build_export(trials, mock_session())

    assert list(export['no_choice']) == [False, True, False]
    assert np.isnan(export.loc[1, 'reaction_time'])
    assert export.loc[1, 'false_start'] == False  # noqa: E712 — not None/NaN
    assert export.loc[0, 'reaction_time'] == 0.5


def test_build_export_false_start_below_threshold_only():
    """A 20 ms response is a false start; a 200 ms one is not.

    Both ends of the reaction time are Bpod-clock columns, so the sound card's
    latency cannot move a trial across the threshold: `feedback_times` stays
    where the fixture put it and only `response_times` decides.
    """
    trials = mock_trials(n_trials=2)
    trials['response_times'] = trials['stimOnTrigger_times'] + [0.02, 0.2]

    export = ct.build_export(trials, mock_session())

    assert list(export['false_start']) == [True, False]
    assert export['false_start'].dtype == bool


def test_build_export_incomplete_scans_only_the_scanned_columns():
    """A NaN flags a trial incomplete unless its column is off the scan."""
    trials = mock_trials(n_trials=3)
    trials.loc[0, 'rewardVolume'] = np.nan
    trials.loc[1, ['stimOn_times', 'firstMovement_times', 'stimOff_times',
                   'goCue_times', 'feedback_times']] = np.nan

    export = ct.build_export(trials, mock_session())

    assert list(export['incomplete']) == [True, False, False]
    assert export['incomplete'].dtype == bool


def test_build_export_incomplete_requires_exactly_one_contrast():
    """Both contrasts present, or neither, is a malformed trial."""
    trials = mock_trials(n_trials=3)
    trials.loc[0, 'contrastLeft'] = 0.25
    trials.loc[1, 'contrastRight'] = np.nan

    export = ct.build_export(trials, mock_session())

    assert list(export['incomplete']) == [True, True, False]


def test_build_export_rejects_a_table_missing_a_raw_column():
    """A session whose extractor never produced a raw column is dropped whole."""
    trials = mock_trials().drop(columns='stimOff_times')

    assert ct.build_export(trials, mock_session()) is None


def test_build_export_drops_columns_outside_the_exported_set():
    """A column the export does not name never reaches the file."""
    trials = mock_trials().assign(intervals_bpod_0=0.0)

    export = ct.build_export(trials, mock_session())

    assert 'intervals_bpod_0' not in export.columns
    assert 'signed_contrast' not in export.columns


def test_build_export_carries_the_identity_columns_and_trial_index():
    """Identity comes from the catalog row; `trial_n` is the stored index."""
    trials = mock_trials(n_trials=3, trial=[4, 5, 6])

    export = ct.build_export(trials, mock_session())

    assert set(export['subject']) == {'ZFM-001'}
    assert set(export['eid']) == {'abc-123'}
    assert set(export['day_n']) == {7}
    assert set(export['session_n']) == {3}
    assert set(export['session_type']) == {'biased'}
    assert list(export['trial_n']) == [4, 5, 6]
