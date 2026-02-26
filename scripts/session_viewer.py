"""
Session Viewer

Interactive viewer for a single photometry session: raw bands, preprocessed
signal, and peri-event heatmap + mean±SEM traces.

Usage:
    python scripts/session_viewer.py <subject> [session_index]
    python scripts/session_viewer.py --eid <eid>

session_index selects among the subject's sessions sorted by date:
    0 = first session, -1 = most recent (default).
"""
import argparse
import sys

import pandas as pd
from matplotlib import pyplot as plt

from iblnm.config import (
    SESSIONS_FPATH, SESSIONS_H5_DIR,
    QUERY_DATABASE_LOG_FPATH, PHOTOMETRY_LOG_FPATH, TASK_LOG_FPATH,
)
from iblnm.data import PhotometrySession
from iblnm.gui import PhotometrySessionViewer
from iblnm.io import _get_default_connection
from iblnm.util import collect_session_errors, concat_logs, deduplicate_log


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Interactive photometry session viewer',
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('subject', nargs='?', default=None,
                       help='Mouse name (e.g. ZFM-04019)')
    group.add_argument('--eid', default=None,
                       help='Session EID (bypasses subject/index lookup)')
    parser.add_argument('session_index', nargs='?', type=int, default=-1,
                        help='Index into sessions sorted by date '
                             '(0=first, -1=most recent; default: -1)')
    return parser.parse_args(argv)


def load_sessions():
    """Load sessions table with upstream error logs attached."""
    df = pd.read_parquet(SESSIONS_FPATH)
    df = collect_session_errors(
        df, [QUERY_DATABASE_LOG_FPATH, PHOTOMETRY_LOG_FPATH, TASK_LOG_FPATH],
    )
    return df


def find_session(df, subject, session_index):
    """Look up a session row by subject and index into date-sorted sessions."""
    subject_sessions = df[df['subject'] == subject].sort_values('start_time')
    if len(subject_sessions) == 0:
        subjects = sorted(df['subject'].unique())
        print(f"Subject '{subject}' not found. Available subjects:")
        for s in subjects:
            print(f"  {s}")
        sys.exit(1)

    n = len(subject_sessions)
    if session_index >= n or session_index < -n:
        print(f"Session index {session_index} out of range for {subject} "
              f"(valid: -{n} to {n - 1}, {n} sessions)")
        sys.exit(1)

    return subject_sessions.iloc[session_index]


def load_session_data(row, one):
    """Create PhotometrySession and load data from H5 (with raw fallback)."""
    eid = row['eid']
    ps = PhotometrySession(row, one=one)

    h5_path = SESSIONS_H5_DIR / f'{eid}.h5'
    if h5_path.exists():
        try:
            ps.load_photometry()
        except Exception as e:
            print(f"Warning: could not load raw photometry — {e}")
        ps.load_h5(h5_path)
    else:
        print(f"No H5 for {eid}, running pipeline...")
        ps.load_trials()
        ps.load_photometry()
        ps.preprocess()
        ps.extract_responses()
        ps.save_h5()

    return ps


if __name__ == '__main__':
    args = parse_args()

    df = load_sessions()

    if args.eid:
        matches = df[df['eid'] == args.eid]
        if len(matches) == 0:
            print(f"EID '{args.eid}' not found in sessions table.")
            sys.exit(1)
        row = matches.iloc[0]
    else:
        row = find_session(df, args.subject, args.session_index)

    errors = row.get('logged_errors', [])

    # Print full error log for this session
    if errors:
        log_paths = [QUERY_DATABASE_LOG_FPATH, PHOTOMETRY_LOG_FPATH, TASK_LOG_FPATH]
        all_logs = [pd.read_parquet(p) for p in log_paths if p.exists()]
        if all_logs:
            df_log = deduplicate_log(concat_logs(all_logs))
            session_log = df_log[df_log['eid'] == row['eid']]
            print(f"\n{'=' * 60}")
            print(f"Errors for this session ({len(session_log)}):")
            print(f"{'=' * 60}")
            for _, entry in session_log.iterrows():
                print(f"  [{entry['error_type']}] {entry['error_message']}")
            print()

    print(f"Loading: {row['subject']} | {row['start_time']} | {row['eid']}")

    one = _get_default_connection()
    ps = load_session_data(row, one)

    viewer = PhotometrySessionViewer(ps)
    viewer.plot(errors=errors)
    plt.show()
