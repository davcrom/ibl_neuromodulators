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
    """Populate ps with the raw bands, the preprocessed signal and the responses.

    Each load method reads its own stored product when the session's H5 holds a
    current one and rebuilds it from Alyx otherwise, so full, partial and empty
    stores need no branching here. The trials table is the exception: nothing
    loads it back from the store, so it is read off the H5 first and left to
    ``load_responses`` to fetch when the file has none.
    """
    if ps.filepath.exists():
        ps.load_h5(ps.filepath, groups=['trials'])

    try:
        ps.load_raw_photometry()
    except (MissingRawData, MissingExtractedData) as e:
        print(f"Photometry data not available for {ps.eid}: {e}")
        sys.exit(1)

    ps.load_photometry()

    try:
        ps.load_responses('photometry')
    except (MissingRawData, MissingExtractedData) as e:
        print(f"Warning: trials not available for {ps.eid} -- {e}")

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
