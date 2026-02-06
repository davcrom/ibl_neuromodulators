"""
Recording Capacity Estimation

Estimates how many more mice can reach recording-ready state by a target date.

Recording-ready = 5+ biased/ephys sessions with:
- Photometry accessible (has_photometry)
- Passes basic QC (n_unique_samples > 0.1, n_band_inversions == 0)
- Trials synced to photometry (trials_in_photometry_time)

Usage:
    python scripts/recording_capacity.py [target_date] [mice_per_day]

Arguments:
    target_date: Target date in dd-mm-yyyy format (default: 6 months from today)
    mice_per_day: Number of mice that can be trained per day (default: 6)
"""
import argparse
import pandas as pd
from datetime import date, datetime, timedelta

from iblnm.config import SESSIONS_FPATH, QCPHOTOMETRY_FPATH, VALID_TARGETS
from iblnm.task import count_sessions_to_stage
from iblnm.util import clean_sessions, drop_junk_duplicates, process_regions, aggregate_qc_per_session


def get_subject_target_mapping(df_sessions):
    """Map subjects to their target_NM."""
    df_targets = df_sessions[['subject', 'target', 'NM']].copy()
    df_targets = df_targets.explode('target').dropna(subset='target')
    df_targets = process_regions(df_targets, region_col='target', add_hemisphere=False)
    return df_targets.groupby('subject')['target_NM'].first().to_dict()


def count_good_sessions(df_sessions):
    """Count biased/ephys sessions passing all QC criteria per subject."""
    df_good = df_sessions[
        (df_sessions['session_type'].isin(['biased', 'ephys'])) &
        (df_sessions['has_photometry']) &
        (df_sessions['passes_basic_qc']) &
        (df_sessions['trials_in_photometry_time'])
    ]
    return df_good.groupby('subject').size().reset_index(name='n_good_sessions')


def sessions_to_n_good(df_sessions, n=5):
    """Calculate total sessions needed to reach n good sessions per subject."""
    df_good = df_sessions[
        (df_sessions['session_type'].isin(['biased', 'ephys'])) &
        (df_sessions['has_photometry']) &
        (df_sessions['passes_basic_qc']) &
        (df_sessions['trials_in_photometry_time'])
    ].copy()

    df_good = df_good.sort_values(['subject', 'session_n'])
    df_good['good_rank'] = df_good.groupby('subject').cumcount() + 1

    nth_sessions = df_good[df_good['good_rank'] == n][['subject', 'session_n']]
    nth_sessions = nth_sessions.rename(columns={'session_n': f'sessions_to_{n}_good'})

    return nth_sessions


def count_weekdays(start_date, end_date):
    """Count weekdays (Mon-Fri) between two dates."""
    total_days = (end_date - start_date).days
    if total_days <= 0:
        return 0
    weeks, remainder = divmod(total_days, 7)
    weekdays = weeks * 5

    current = start_date + timedelta(days=weeks * 7)
    for _ in range(remainder):
        if current.weekday() < 5:
            weekdays += 1
        current += timedelta(days=1)

    return weekdays


def estimate_capacity(target_date, mice_per_day, effective_sessions, start_date=None):
    """Estimate number of mice that can reach recording target by a given date."""
    if start_date is None:
        start_date = date.today()

    weekdays = count_weekdays(start_date, target_date)
    total_slots = weekdays * mice_per_day

    if effective_sessions > 0:
        n_mice = int(total_slots / effective_sessions)
    else:
        n_mice = 0

    return n_mice, weekdays, total_slots


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimate recording capacity')
    parser.add_argument('target_date', nargs='?', type=str, default=None,
                        help='Target date in dd-mm-yyyy format (default: 01-07-2026)')
    parser.add_argument('mice_per_day', nargs='?', type=int, default=8,
                        help='Mice that can be trained per day (default: 8)')
    args = parser.parse_args()

    # Parse target date
    if args.target_date:
        target_date = datetime.strptime(args.target_date, '%d-%m-%Y').date()
    else:
        target_date = date(2026, 7, 1)

    mice_per_day = args.mice_per_day

    # Load data
    print("Loading data...")
    df_sessions = pd.read_parquet(SESSIONS_FPATH)
    df_qc = pd.read_parquet(QCPHOTOMETRY_FPATH)

    # Add QC aggregation
    qc_agg = aggregate_qc_per_session(df_qc, require_all=True)
    df_sessions = df_sessions.merge(qc_agg[['eid', 'passes_basic_qc']], on='eid', how='left')
    df_sessions['passes_basic_qc'] = df_sessions['passes_basic_qc'].fillna(False)

    # =========================================================================
    # Filter to mainenlab mice started in 2024+
    # =========================================================================
    df_sessions = df_sessions[df_sessions['lab'] == 'mainenlab']
    df_sessions['start_time'] = pd.to_datetime(df_sessions['start_time'], format='ISO8601')
    first_session = df_sessions.groupby('subject')['start_time'].min()
    subjects_2024 = first_session[first_session >= '2024-01-01'].index
    df_sessions = df_sessions[df_sessions['subject'].isin(subjects_2024)]

    n_subjects = df_sessions['subject'].nunique()
    print(f"Mainenlab subjects started 2024+: {n_subjects}")

    # Get subject -> target mapping
    subject_targets = get_subject_target_mapping(df_sessions)

    # Count sessions to biased stage
    df_stage = count_sessions_to_stage(df_sessions)
    df_stage['target_NM'] = df_stage['subject'].map(subject_targets)
    df_stage = df_stage.dropna(subset=['target_NM'])

    # Count good sessions per subject
    good_counts = count_good_sessions(df_sessions)
    df_stage = df_stage.merge(good_counts, on='subject', how='left')
    df_stage['n_good_sessions'] = df_stage['n_good_sessions'].fillna(0).astype(int)
    df_stage['reached_target'] = df_stage['n_good_sessions'] >= 5

    # Sessions to reach 5 good
    sessions_to_5 = sessions_to_n_good(df_sessions, n=5)
    df_stage = df_stage.merge(sessions_to_5, on='subject', how='left')

    # Total sessions per subject
    sessions_per_subject = df_sessions.groupby('subject').size()
    df_stage['total_sessions'] = df_stage['subject'].map(sessions_per_subject)

    # Sessions consumed (capped at target for successful mice)
    df_stage['sessions_consumed'] = df_stage.apply(
        lambda r: r['sessions_to_5_good'] if r['reached_target'] else r['total_sessions'],
        axis=1
    )

    # =========================================================================
    # Summary by target
    # =========================================================================
    print("\n" + "=" * 70)
    print("CURRENT STATUS BY TARGET")
    print("=" * 70)

    for target in sorted(df_stage['target_NM'].unique(), key=lambda x: VALID_TARGETS.index(x) if x in VALID_TARGETS else 99):
        t_data = df_stage[df_stage['target_NM'] == target]
        n_total = len(t_data)
        n_reached = t_data['reached_target'].sum()
        n_biased = t_data['sessions_to_biased'].notna().sum()

        mean_to_biased = t_data['sessions_to_biased'].mean()
        mean_to_5 = t_data['sessions_to_5_good'].mean()

        print(f"\n{target}:")
        print(f"  Mice: {n_total}")
        print(f"  Reached biased: {n_biased}/{n_total}")
        print(f"  Reached 5 good: {n_reached}/{n_total}")
        if pd.notna(mean_to_biased):
            print(f"  Sessions to biased: {mean_to_biased:.0f}")
        if pd.notna(mean_to_5):
            print(f"  Sessions to 5 good: {mean_to_5:.0f}")

    # =========================================================================
    # Overall statistics
    # =========================================================================
    n_total = len(df_stage)
    n_successful = df_stage['reached_target'].sum()
    success_rate = n_successful / n_total if n_total > 0 else 0

    total_sessions_consumed = df_stage['sessions_consumed'].sum()
    if n_successful > 0:
        effective_sessions = total_sessions_consumed / n_successful
        mean_sessions_successful = df_stage[df_stage['reached_target']]['sessions_to_5_good'].mean()
    else:
        effective_sessions = df_stage['total_sessions'].mean() * 2  # Rough estimate
        mean_sessions_successful = float('nan')

    print("\n" + "=" * 70)
    print("OVERALL STATISTICS")
    print("=" * 70)
    print(f"Total mice: {n_total}")
    print(f"Reached target (5 good sessions): {n_successful}")
    print(f"Success rate: {100 * success_rate:.0f}%")
    print(f"Mean sessions (successful only): {mean_sessions_successful:.1f}")
    print(f"Effective sessions/mouse (accounting for dropouts): {effective_sessions:.1f}")

    # =========================================================================
    # Projection
    # =========================================================================
    n_mice, weekdays, slots = estimate_capacity(target_date, mice_per_day, effective_sessions)

    print("\n" + "=" * 70)
    print("PROJECTION")
    print("=" * 70)
    print(f"Target date: {target_date.strftime('%d-%m-%Y')}")
    print(f"Mice per day: {mice_per_day}")
    print(f"Weekdays available: {weekdays}")
    print(f"Total training slots: {slots}")
    print(f"Estimated mice reaching target: {n_mice}")
