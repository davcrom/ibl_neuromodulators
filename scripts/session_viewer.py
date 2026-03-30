"""
Session Viewer

Interactive viewer for a single photometry session: raw bands, preprocessed
signal, and peri-event heatmap + mean+/-SEM traces.

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

from iblnm.config import SESSIONS_H5_DIR
from iblnm.data import PhotometrySession
from iblnm.gui import PhotometrySessionViewer
from iblnm.io import _get_default_connection
from iblnm.validation import MissingRawData, MissingExtractedData


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


def find_session(args, one):
    """Look up a session on Alyx and return a PhotometrySession with metadata.

    Queries Alyx REST for the session dict, creates a PhotometrySession,
    and calls from_alyx() to populate and validate all metadata.
    """
    if args.eid:
        sessions = one.alyx.rest('sessions', 'list', id=args.eid,
                                 project='ibl_fibrephotometry')
        if not sessions:
            print(f"EID '{args.eid}' not found on Alyx.")
            sys.exit(1)
        session_dict = sessions[0]
    else:
        sessions = one.alyx.rest('sessions', 'list', subject=args.subject,
                                 project='ibl_fibrephotometry')
        if not sessions:
            print(f"No sessions found for subject '{args.subject}'.")
            sys.exit(1)
        sessions = sorted(sessions, key=lambda s: s['start_time'])
        n = len(sessions)
        idx = args.session_index
        if idx >= n or idx < -n:
            print(f"Session index {idx} out of range for {args.subject} "
                  f"(valid: -{n} to {n - 1}, {n} sessions)")
            sys.exit(1)
        session_dict = sessions[idx]

    row = pd.Series(session_dict).rename(index={'id': 'eid'})
    ps = PhotometrySession(row, one=one, load_data=False)
    ps.from_alyx()
    return ps


def print_session_errors(ps):
    """Print validation errors from a PhotometrySession."""
    if not ps.errors:
        return
    print(f"\n{'=' * 60}")
    print(f"Errors for this session ({len(ps.errors)}):")
    print(f"{'=' * 60}")
    for entry in ps.errors:
        print(f"  [{entry['error_type']}] {entry['error_message']}")
    print()


def load_session_data(ps):
    """Load data into a PhotometrySession from H5 (with raw fallback)."""
    try:
        ps.load_photometry()
    except Exception as e:
        print(f"Warning: could not load raw photometry -- {e}")

    h5_path = SESSIONS_H5_DIR / f'{ps.eid}.h5'

    if h5_path.exists():
        ps.load_h5(h5_path)
    else:
        print(f"No H5 for {ps.eid}, running pipeline...")
        try:
            ps.load_trials()
        except MissingRawData:
            print(f"Warning: task data not yet registered for {ps.eid} "
                  f"-- viewing photometry only")
        except MissingExtractedData:
            print(f"Warning: trials not yet extracted for {ps.eid} "
                  f"-- viewing photometry only")
        try:
            ps.load_photometry()
        except MissingRawData:
            print(f"Photometry data not yet registered for {ps.eid}.")
            sys.exit(1)
        except MissingExtractedData:
            print(f"Photometry data not yet extracted for {ps.eid}.")
            sys.exit(1)
        ps.preprocess()
        if ps.trials is not None:
            ps.extract_responses()

    return ps


if __name__ == '__main__':
    args = parse_args()
    one = _get_default_connection()

    ps = find_session(args, one)
    print_session_errors(ps)

    print(f"Loading: {ps.subject} | {ps.start_time} | {ps.eid}")
    load_session_data(ps)
    viewer = PhotometrySessionViewer(ps)
    viewer.plot()
    plt.show()
