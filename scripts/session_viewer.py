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

from iblnm.config import SESSIONS_H5_DIR
from iblnm.data import PhotometrySession
from iblnm.gui import PhotometrySessionViewer
from iblnm.io import (
    _get_default_connection, get_subject_info, get_session_dict,
    get_brain_region, get_datasets,
)
from iblnm.util import get_session_type, get_targetNM, get_session_length
from iblnm.validation import (
    fill_hemisphere_from_fiber_insertion_table,
    validate_subject, validate_strain, validate_line,
    validate_neuromodulator, validate_brain_region, validate_hemisphere,
    validate_datasets, MissingRawData, MissingExtractedData,
)


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


def build_session_from_rest(session_dict, one):
    """Build and validate a session Series from an Alyx REST session dict.

    Runs the same io and validation pipeline as query_database.py.
    Errors are attached as 'logged_errors' (list of error_type strings) and
    '_rest_errors' (list of full error dicts for display).
    """
    exlog = []
    session = pd.Series(session_dict).rename(index={'id': 'eid'})

    session = get_subject_info(session, one=one, exlog=exlog)
    validate_subject(session, exlog=exlog)
    validate_strain(session, exlog=exlog)
    validate_line(session, exlog=exlog)
    validate_neuromodulator(session, exlog=exlog)

    session = get_session_dict(session, one=one, exlog=exlog)

    session = get_brain_region(session, one=one, exlog=exlog)
    session = fill_hemisphere_from_fiber_insertion_table(session, exlog=exlog)
    validate_brain_region(session, exlog=exlog)
    validate_hemisphere(session, exlog=exlog)

    session = get_datasets(session, one=one, exlog=exlog)
    validate_datasets(session, exlog=exlog)

    session = get_session_type(session, exlog=exlog)
    session = get_targetNM(session, exlog=exlog)
    session = get_session_length(session, exlog=exlog)

    session['logged_errors'] = [e['error_type'] for e in exlog]
    session['_rest_errors'] = exlog
    return session


def find_session_by_eid(eid, one):
    """Return the session Series for a given EID, queried from Alyx."""
    sessions = one.alyx.rest('sessions', 'list', id=eid, project='ibl_fibrephotometry')
    if not sessions:
        print(f"EID '{eid}' not found on Alyx.")
        sys.exit(1)
    return build_session_from_rest(sessions[0], one)


def find_session_by_subject(subject, session_index, one):
    """Return the session Series for a subject/index, queried from Alyx."""
    sessions = one.alyx.rest('sessions', 'list', subject=subject, project='ibl_fibrephotometry')
    if not sessions:
        print(f"No sessions found for subject '{subject}'.")
        sys.exit(1)
    sessions = sorted(sessions, key=lambda s: s['start_time'])
    n = len(sessions)
    if session_index >= n or session_index < -n:
        print(f"Session index {session_index} out of range for {subject} "
              f"(valid: -{n} to {n - 1}, {n} sessions)")
        sys.exit(1)
    return build_session_from_rest(sessions[session_index], one)


def print_session_errors(session):
    """Print validation errors attached to a session."""
    errors = session.get('_rest_errors', [])
    if not errors:
        return
    print(f"\n{'=' * 60}")
    print(f"Errors for this session ({len(errors)}):")
    print(f"{'=' * 60}")
    for entry in errors:
        print(f"  [{entry['error_type']}] {entry['error_message']}")
    print()


def load_session_data(row, one):
    """Create PhotometrySession and load data from H5 (with raw fallback)."""
    eid = row['eid']
    ps = PhotometrySession(row, one=one)

    try:
        ps.load_photometry()
    except Exception as e:
        print(f"Warning: could not load raw photometry — {e}")

    h5_path = SESSIONS_H5_DIR / f'{eid}.h5'

    if h5_path.exists():
        ps.load_h5(h5_path)
    else:
        print(f"No H5 for {eid}, running pipeline...")
        try:
            ps.load_trials()
        except MissingRawData:
            print(f"Warning: task data not yet registered for {eid} — viewing photometry only")
        except MissingExtractedData:
            print(f"Warning: trials not yet extracted for {eid} — viewing photometry only")
        try:
            ps.load_photometry()
        except MissingRawData:
            print(f"Photometry data not yet registered for {eid}.")
            sys.exit(1)
        except MissingExtractedData:
            print(f"Photometry data not yet extracted for {eid}.")
            sys.exit(1)
        ps.preprocess()
        if ps.trials is not None:
            ps.extract_responses()

    return ps


if __name__ == '__main__':
    args = parse_args()
    one = _get_default_connection()

    if args.eid:
        session = find_session_by_eid(args.eid, one)
    else:
        session = find_session_by_subject(args.subject, args.session_index, one)

    print_session_errors(session)

    print(f"Loading: {session['subject']} | {session['start_time']} | {session['eid']}")
    ps = load_session_data(session, one)
    viewer = PhotometrySessionViewer(ps)
    viewer.plot()
    plt.show()
