"""
Recording Capacity Estimation

Combines:
- Sessions to reach biased stage
- Dataset availability
- Photometry baseline QC

Estimates number of mice recordable by a given date, given training capacity.

Usage:
    python scripts/recording_capacity.py 01-06-2025 6

Arguments:
    date: Target date in dd-mm-yyyy format
    n_mice: Number of mice that can be trained per day

Output: Summary table and projections
"""
import argparse
import pandas as pd
from datetime import date, datetime, timedelta

from iblnm.config import SESSIONS_CLEAN_FPATH, QCPHOTOMETRY_FPATH, VALID_TARGETS
from iblnm.task import count_sessions_to_stage


def get_qc_per_session(df_qc):
    """Aggregate QC metrics per session with baseline QC criteria."""
    qc_per_session = df_qc.groupby('eid').agg({
        'n_early_samples': 'max',
        'n_band_inversions': 'max',
        'n_unique_samples': 'min',
    }).reset_index()
    qc_per_session['passes_baseline_qc'] = (
        (qc_per_session['n_early_samples'] == 0) &
        (qc_per_session['n_band_inversions'] == 0) &
        (qc_per_session['n_unique_samples'] > 0.1)
    )
    return qc_per_session


def get_subject_target_mapping(df_sessions):
    """Map subjects to their target_NM."""
    df_targets = df_sessions[['subject', 'target', 'NM']].copy()
    df_targets = df_targets.explode('target').dropna(subset='target')
    df_targets['target_NM'] = df_targets['target'].str.split('-').str[0] + '-' + df_targets['NM']
    df_targets = df_targets.query('target_NM in @VALID_TARGETS')
    return df_targets.groupby('subject')['target_NM'].first().to_dict()


def count_good_biased_sessions(df_sessions):
    """Count biased/ephys sessions with photometry passing baseline QC per subject."""
    df_biased = df_sessions[
        (df_sessions['session_type'].isin(['biased', 'ephys'])) &
        (df_sessions['has_photometry']) &
        (df_sessions['passes_baseline_qc'])
    ]
    return df_biased.groupby('subject').size().reset_index(name='n_good_biased_sessions')


def sessions_to_n_good_biased(df_sessions, n=5):
    """Calculate total sessions needed to reach n good biased sessions per subject."""
    df_biased = df_sessions[
        (df_sessions['session_type'].isin(['biased', 'ephys'])) &
        (df_sessions['has_photometry']) &
        (df_sessions['passes_baseline_qc'])
    ].copy()

    # Sort by session number and find the nth good biased session
    df_biased = df_biased.sort_values(['subject', 'session_n'])
    df_biased['good_biased_rank'] = df_biased.groupby('subject').cumcount() + 1

    # Get session_n when nth good biased was reached
    nth_sessions = df_biased[df_biased['good_biased_rank'] == n][['subject', 'session_n']]
    nth_sessions = nth_sessions.rename(columns={'session_n': f'sessions_to_{n}_good_biased'})

    return nth_sessions


def count_weekdays(start_date, end_date):
    """Count weekdays (Mon-Fri) between two dates."""
    total_days = (end_date - start_date).days
    weeks, remainder = divmod(total_days, 7)
    weekdays = weeks * 5

    # Count remaining days
    current = start_date + timedelta(days=weeks * 7)
    for _ in range(remainder):
        if current.weekday() < 5:  # Mon-Fri
            weekdays += 1
        current += timedelta(days=1)

    return weekdays


def estimate_mice_by_date(target_date, mice_per_day, mean_sessions_to_target, start_date=None):
    """
    Estimate number of mice that can reach recording target by a given date.

    Parameters
    ----------
    target_date : date
        Target completion date
    mice_per_day : int
        Number of mice that can be trained per day
    mean_sessions_to_target : float
        Average sessions needed to reach recording-ready state
    start_date : date, optional
        Start date (default: today)

    Returns
    -------
    int
        Estimated number of mice that can be trained
    """
    if start_date is None:
        start_date = date.today()

    weekdays_available = count_weekdays(start_date, target_date)
    total_training_slots = weekdays_available * mice_per_day

    # Each mouse needs mean_sessions_to_target training slots
    n_mice = int(total_training_slots / mean_sessions_to_target)

    return n_mice, weekdays_available, total_training_slots


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimate recording capacity by target date')
    parser.add_argument('target_date', type=str, help='Target date in dd-mm-yyyy format')
    parser.add_argument('n_mice', type=int, help='Number of mice that can be trained per day')
    args = parser.parse_args()

    # Parse date
    target_date = datetime.strptime(args.target_date, '%d-%m-%Y').date()
    mice_per_day = args.n_mice

    # Load data
    print("Loading data...")
    df_sessions = pd.read_parquet(SESSIONS_CLEAN_FPATH)
    df_qc = pd.read_parquet(QCPHOTOMETRY_FPATH)

    # Add QC flags
    qc_per_session = get_qc_per_session(df_qc)
    df_sessions = df_sessions.merge(qc_per_session[['eid', 'passes_baseline_qc']], on='eid', how='left')
    df_sessions['passes_baseline_qc'] = df_sessions['passes_baseline_qc'].fillna(False)

    # Get subject -> target_NM mapping
    subject_targets = get_subject_target_mapping(df_sessions)

    # Count sessions to stage
    df_stage = count_sessions_to_stage(df_sessions)
    df_stage['target_NM'] = df_stage['subject'].map(subject_targets)
    df_stage = df_stage.dropna(subset=['target_NM'])

    # Count good biased sessions per subject
    good_biased = count_good_biased_sessions(df_sessions)
    df_stage = df_stage.merge(good_biased, on='subject', how='left')
    df_stage['n_good_biased_sessions'] = df_stage['n_good_biased_sessions'].fillna(0).astype(int)
    df_stage['has_5_good_biased'] = df_stage['n_good_biased_sessions'] >= 5

    # Get sessions to 5 good biased
    sessions_to_5 = sessions_to_n_good_biased(df_sessions, n=5)
    df_stage = df_stage.merge(sessions_to_5, on='subject', how='left')

    # Summary table
    print("\n=== Summary by target_NM ===")
    summary = df_stage.groupby('target_NM').agg(
        n_subjects=('subject', 'count'),
        n_reached_biased=('sessions_to_biased', lambda x: x.notna().sum()),
        n_5_good_biased=('has_5_good_biased', 'sum'),
        mean_to_biased=('sessions_to_biased', 'mean'),
        std_to_biased=('sessions_to_biased', 'std'),
        mean_to_5_good=('sessions_to_5_good_biased', 'mean'),
        std_to_5_good=('sessions_to_5_good_biased', 'std'),
    ).round(1)

    summary['reached_biased'] = summary['n_reached_biased'].astype(int).astype(str) + '/' + summary['n_subjects'].astype(int).astype(str)
    summary['has_5_good'] = summary['n_5_good_biased'].astype(int).astype(str) + '/' + summary['n_subjects'].astype(int).astype(str)
    summary['to_biased'] = summary.apply(
        lambda r: f"{r['mean_to_biased']:.0f}±{r['std_to_biased']:.0f}" if pd.notna(r['std_to_biased']) else
                  (f"{r['mean_to_biased']:.0f}" if pd.notna(r['mean_to_biased']) else '-'), axis=1)
    summary['to_5_good'] = summary.apply(
        lambda r: f"{r['mean_to_5_good']:.0f}±{r['std_to_5_good']:.0f}" if pd.notna(r['std_to_5_good']) else
                  (f"{r['mean_to_5_good']:.0f}" if pd.notna(r['mean_to_5_good']) else '-'), axis=1)

    print(summary[['reached_biased', 'has_5_good', 'to_biased', 'to_5_good']].to_markdown())

    # Overall stats for projection
    subjects_with_5_good = df_stage[df_stage['has_5_good_biased']]
    mean_sessions = subjects_with_5_good['sessions_to_5_good_biased'].mean()

    print(f"\n=== Recording Capacity Projection ===")
    print(f"Target date: {target_date}")
    print(f"Mice per day: {mice_per_day}")
    print(f"Mean sessions to 5 good biased: {mean_sessions:.1f}")

    n_mice, weekdays, slots = estimate_mice_by_date(target_date, mice_per_day, mean_sessions)
    print(f"Weekdays until target: {weekdays}")
    print(f"Total training slots: {slots}")
    print(f"Estimated mice to 5 good biased: {n_mice}")
