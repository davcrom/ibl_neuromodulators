"""Re-cut the photometry responses of every stored session.

A narrow pass over the store: nothing is fetched and nothing else is rebuilt.
Each session's preprocessed band and trials table are read back out of its H5
file, the peri-event matrices are cut again, and `photometry/{region}/responses`
is replaced. Use it after changing `config.RESPONSE_EVENTS`,
`config.RESPONSE_WINDOW` or the cut itself, when a full `scripts/download.py`
run would rebuild every other product to no purpose.

Usage:
    python scripts/rebuild_responses.py                   # every session
    python scripts/rebuild_responses.py --workers 4       # in parallel
    python scripts/rebuild_responses.py --session-type biased
    python scripts/rebuild_responses.py --target-NM LC-NE
"""
import os

# pyarrow's default mimalloc pool retains roughly 85 MB per parquet read and
# never plateaus. The backend is chosen when the pool is first created, so this
# has to run before anything that imports pyarrow — which both `iblnm` and ONE do.
os.environ['ARROW_DEFAULT_MEMORY_POOL'] = 'system'

import argparse  # noqa: E402

from iblnm.config import (  # noqa: E402
    SESSION_TYPES, SESSIONS_H5_DIR, VALID_TARGETNMS,
)
from iblnm.data import (  # noqa: E402
    PREPROCESSED_BAND, PhotometrySession, PhotometrySessionGroup,
)
from iblnm.io import _get_default_connection  # noqa: E402


def rebuild_responses(ps: PhotometrySession) -> None:
    """Re-cut one session's photometry responses and write them back.

    The session arrives holding whatever its file stored, so the trials and the
    preprocessed band are already in memory and the two loads below are reads
    of the store, not trips to Alyx — unless the file is missing that product,
    in which case `load_photometry` fetches the raw bands and preprocesses them
    as the download pass would.

    `save_h5(groups=['photometry'])` replaces the `responses` subgroup of each
    region. The preprocessed signal and the QC beside it are round-tripped
    through the same handlers that read them, so what stands in the file after
    the write is what stood there before it.

    Parameters
    ----------
    ps : PhotometrySession
        The session to re-cut. Top-level so `PhotometrySessionGroup.process`
        can hand it to parallel workers.
    """
    try:
        ps.load_trials()
        signal = ps.load_photometry()
        events = ps.complete_events()
        if events:
            ps.photometry_responses = ps.extract_responses(signal, events=events)
            ps.save_h5(groups=['photometry'])
    except Exception as error:
        ps.log_error(error, product='photometry')


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Re-cut the photometry responses of the stored sessions')
    parser.add_argument('--workers', '-w', type=int, default=1,
                        help='Number of parallel worker processes')
    parser.add_argument('--session-type', nargs='+', choices=SESSION_TYPES,
                        default=None,
                        help='Restrict to these session types (default: all)')
    parser.add_argument('--target-NM', nargs='+', choices=VALID_TARGETNMS,
                        default=None,
                        help='Restrict to sessions carrying a recording from '
                             'one of these target neuromodulators '
                             '(default: all)')
    return parser.parse_args(argv)


def main(argv=None) -> None:
    """Re-cut every stored session's photometry responses.

    The catalog is rebuilt from the store's own `metadata` groups rather than
    read from `sessions.pqt`, so this pass covers what is on disk and queries
    Alyx for no session list of its own. `fix_catalog` still runs: a session
    whose brain regions are unrepaired cannot name its own photometry columns
    if the preprocessed band has to be rebuilt.

    Every analysis filter is off, for the reason the download pass turns them
    off — they name the sessions worth analysing, not the ones worth building —
    and `scan_h5=False` because no filter that is on reads what the scan
    collects.
    """
    args = parse_args(argv)
    one = _get_default_connection()

    group = PhotometrySessionGroup.from_h5_dir(SESSIONS_H5_DIR, one=one,
                                               scan_h5=False)
    group.fix_catalog()
    group.filter_sessions(
        session_types=args.session_type or False, qc_blockers=set(),
        targetnms=args.target_NM or False, photometry_qc=False,
        min_performance=False, required_contrasts=False,
    )
    print(f'Re-cutting {PREPROCESSED_BAND} responses for '
          f'{len(group.sessions)} sessions')

    group.process(rebuild_responses, workers=args.workers)

    errors = group.collect_errors()
    if len(errors):
        print(f'\nLogged {len(errors)} error entries across sessions:')
        print(errors['error_type'].value_counts().to_string())


if __name__ == '__main__':
    main()
