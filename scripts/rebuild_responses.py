"""Re-cut the responses of every stored session, photometry and wheel.

A narrow pass over the store: nothing is fetched and nothing else is rebuilt.
Each session's preprocessed signals and trials table are read back out of its
H5 file, the peri-event matrices are cut again, and
`photometry/{region}/responses`, `wheel/velocity/responses` and the
`wheel/velocity/peak_velocity` reduced from that cut are replaced. Use it after
changing the events, the window or the cut itself — `config.RESPONSE_EVENTS`,
`config.RESPONSE_WINDOW`, `config.WHEEL_RESPONSE_EVENTS`,
`config.WHEEL_RESPONSE_WINDOW` — when a full `scripts/download.py` run would
rebuild every other product to no purpose.

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
    WHEEL_RESPONSE_EVENTS, WHEEL_RESPONSE_WINDOW,
)
from iblnm.data import (  # noqa: E402
    WHEEL_LABEL, PhotometrySession, PhotometrySessionGroup,
)
from iblnm.io import _get_default_connection  # noqa: E402


def rebuild_responses(ps: PhotometrySession) -> None:
    """Re-cut one session's photometry and wheel responses and write them back.

    The session arrives holding whatever its file stored, so the trials and the
    preprocessed signals are already in memory and the loads below are reads of
    the store, not trips to Alyx — unless the file is missing that product, in
    which case `load_photometry` fetches the raw bands and preprocesses them,
    and `load_wheel` fetches the raw encoder samples and differentiates them, as
    the download pass would.

    One `try` per modality, the boundary `scripts/download.py` draws: a fatal
    step abandons its block, logs one error against that modality, and the other
    modality is still re-cut. Each block loads the trials it reads, so a session
    with no trials table logs against both rather than silently re-cutting
    neither. The second call is free — `load_trials` answers from the attribute
    the first one set.

    `save_h5(groups=[...])` replaces the `responses` subgroup of each region and
    of the wheel, and the wheel's `peak_velocity` beside it — the one product
    reduced from a cut, so it is rebuilt here rather than left stale. The
    preprocessed signals and the QC beside them are round-tripped through the
    same handlers that read them, so what stands in the file after the write is
    what stood there before it.

    The errors are written last, once, because `process` persists nothing: a
    pass that rebuilds a product owns that product's error log, and writes it
    with the same call that writes the product. Each block clears its own
    product first, so a cut that succeeds this time drops the entry its last
    attempt left; the products this pass does not touch keep theirs.

    Parameters
    ----------
    ps : PhotometrySession
        The session to re-cut. Top-level so `PhotometrySessionGroup.process`
        can hand it to parallel workers.
    """
    try:
        ps.clear_errors('photometry')
        ps.load_trials()
        signal = ps.load_photometry()
        events = ps.complete_events()
        if events:
            ps.photometry_responses = ps.extract_responses(signal, events=events)
            ps.save_h5(groups=['photometry'])
    except Exception as error:
        ps.log_error(error, product='photometry')

    try:
        ps.clear_errors('wheel')
        ps.load_trials()
        ps.wheel_responses = ps.extract_responses(
            {WHEEL_LABEL: ps.load_wheel()}, events=WHEEL_RESPONSE_EVENTS,
            window=WHEEL_RESPONSE_WINDOW)
        ps.extract_peak_velocity()
        ps.save_h5(groups=['wheel'])
    except Exception as error:
        ps.log_error(error, product='wheel')

    ps.save_h5(groups=['errors'])


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Re-cut the responses of the stored sessions')
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
    """Re-cut every stored session's photometry and wheel responses.

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
    print(f'Re-cutting responses for {len(group.sessions)} sessions')

    group.process(rebuild_responses, workers=args.workers)

    errors = group.collect_errors()
    if len(errors):
        print(f'\nLogged {len(errors)} error entries across sessions:')
        print(errors['error_type'].value_counts().to_string())


if __name__ == '__main__':
    main()
