"""Fetch the session catalog from Alyx and build every product in the store.

Two phases. Phase one queries Alyx for the project's session list, writes each
new session's metadata into `data/sessions/{eid}.h5`, and runs the fixups that
need every session at once, leaving the catalog in `metadata/sessions.pqt`.
Phase two rebuilds every product of every session, through the group's
`process`.

Nothing is skipped and nothing is read back: a session is fetched from Alyx,
processed in memory, and written whole. The flags narrow which sessions run,
never what is built within one.

Usage:
    python scripts/download.py                          # build every session
    python scripts/download.py --workers 4              # in parallel
    python scripts/download.py --session-type biased    # one session type
    python scripts/download.py --target-NM LC-NE        # one target population
"""
import os

# pyarrow's default mimalloc pool retains roughly 85 MB per parquet read and
# never plateaus, which is what the OOM killer ended the last full rebuild on.
# The backend is chosen when the pool is first created, so this has to run
# before anything that imports pyarrow — which both `iblnm` and ONE do.
os.environ['ARROW_DEFAULT_MEMORY_POOL'] = 'system'

import argparse  # noqa: E402

import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from iblnm.config import (  # noqa: E402
    RESPONSE_EVENTS, SESSION_TYPES, SESSIONS_FPATH, SESSIONS_H5_DIR,
    VALID_TARGETNMS,
)
from iblnm.data import (  # noqa: E402
    PREPROCESSED_BAND, WHEEL_LABEL, PhotometrySession, PhotometrySessionGroup,
)
from iblnm.io import _get_default_connection  # noqa: E402
from iblnm.validation import (  # noqa: E402
    BlockStructureBug, IncompleteEventTimes,
)

# The wheel's own cut: each trial's wheel velocity from its stimulus onset to
# its own feedback, rather than the fixed peri-event window the photometry uses.
WHEEL_RESPONSE_EVENTS = ('stimOn_times',)
WHEEL_RESPONSE_WINDOW = (0.0, 'feedback_times')


def query_session(row: pd.Series, one) -> bool:
    """Write one session's Alyx metadata into a fresh H5 file.

    A metadata query that fails is logged into the file it writes rather than
    raised, so one unreadable session does not end the catalog pass. Returns
    whether the file was written; a session whose row cannot even construct a
    `PhotometrySession` is reported and returns False.
    """
    try:
        ps = PhotometrySession(row, one=one, load_data=False)
    except Exception as error:
        print(f"  {row['eid']}: {type(error).__name__}: {error}")
        return False
    try:
        ps.from_alyx()
    except Exception as error:
        ps.log_error(error)
    ps.save_h5(SESSIONS_H5_DIR / f"{row['eid']}.h5",
               groups=['metadata', 'errors'], mode='w')
    return True


def fetch_catalog(one) -> PhotometrySessionGroup:
    """Query Alyx for the project's sessions and return the group to build.

    Sessions already holding an H5 file keep their stored metadata; only new
    eids are queried. The group is then read back out of every file's
    `metadata` group and its catalog repaired in place, so what the build
    iterates carries the fixed brain regions rather than a copy of them.

    The catalog is also written to `config.SESSIONS_FPATH` for the analysis
    scripts that read it; the store's `metadata` groups remain the source, so
    that file is a convenience, not a second source.

    `scan_h5=False` because nothing here reads what `complete_catalog`
    collects: `main` turns off every filter that consults it, and
    `collect_errors` re-reads the store itself. Leaving it on opens every
    stored file a second time to build a table that is then discarded.
    """
    print('Querying database...')
    sessions = one.alyx.rest('sessions', 'list', project='ibl_fibrephotometry')
    alyx = pd.DataFrame(sessions).rename(columns={'id': 'eid'})
    stored = {path.stem for path in SESSIONS_H5_DIR.glob('*.h5')}
    new = alyx[~alyx['eid'].isin(stored)]
    print(f'Found {len(alyx)} sessions on Alyx, {len(new)} without stored metadata')

    written = sum(query_session(row, one)
                  for _, row in tqdm(new.iterrows(), total=len(new),
                                     desc='Querying Alyx'))
    if len(new):
        print(f'Wrote metadata for {written} of {len(new)} sessions')

    group = PhotometrySessionGroup.from_h5_dir(SESSIONS_H5_DIR, one=one,
                                               scan_h5=False)
    group.fix_catalog()
    # No filter has run yet, so this is the whole catalog.
    group.sessions.to_parquet(SESSIONS_FPATH, index=False)
    print(f'Catalogued {len(group.sessions)} sessions in {SESSIONS_FPATH}')
    return group


def build_trials(ps: PhotometrySession) -> None:
    """Fetch the trials table, validate it, and score the session's behavior.

    Fatal: the fetch and the trial-count check, either of which leaves nothing
    to score. Non-fatal: the two structural checks. An incomplete event is
    recorded and left alone here — the photometry block is where it changes
    what is cut — and a corrupted block structure is followed by the fix that
    reconstructs `probabilityLeft` from the session JSON.
    """
    try:
        ps.fetch_trials()
        ps.validate_n_trials()
        try:
            ps.validate_event_completeness()
        except IncompleteEventTimes as error:
            ps.log_error(error, product='trials')
        try:
            ps.validate_block_structure()
        except BlockStructureBug as error:
            ps.log_error(error, product='trials')
            ps.fix_block_structure()
        ps.extract_performance()
    except Exception as error:
        ps.log_error(error, product='trials')


def complete_events(ps: PhotometrySession) -> list[str]:
    """The `config.RESPONSE_EVENTS` whose times are complete enough to cut on.

    `validate_event_completeness` is non-fatal for the response cut: the events
    it names are dropped and the rest are cut. An empty list means no event
    survived, and nothing is cut at all.
    """
    try:
        ps.validate_event_completeness()
    except IncompleteEventTimes as error:
        return [event for event in RESPONSE_EVENTS
                if event not in error.missing_events]
    return list(RESPONSE_EVENTS)


def build_photometry(ps: PhotometrySession) -> None:
    """Fetch both photometry sources, score them, preprocess and cut responses.

    Every step is fatal but the event-completeness check, which degrades the
    cut to the events that survive it.

    The extracted bands are fetched first because `fetch_photometry` is the
    only step that tells an absent recording from an unextracted one, raising
    `MissingRawData` or `MissingExtractedData` — both of which
    `config.ANALYSIS_QC_BLOCKERS` excludes from analysis. Above it,
    `fetch_neurophotometrics` raises a bare `ALFObjectNotFound` for the same
    absent session, which blocks nothing, and abandons the block before the
    classification is reached.

    The neurophotometrics table is still scored before anything is computed
    from the bands: a band inversion means the channels are not the bands they
    are labelled, so nothing below `validate_qc` is worth computing. Fetching
    is not computing, which is what lets it move above the score.
    """
    try:
        ps.fetch_photometry()
        ps.fetch_neurophotometrics()
        ps.run_neurophotometrics_qc()
        ps.validate_qc()
        ps.validate_trials_in_photometry_time()
        ps.run_photometry_qc()
        ps.extract_preprocessed_photometry()
        events = complete_events(ps)
        if events:
            ps.photometry_responses = ps.extract_responses(
                ps.photometry[PREPROCESSED_BAND], events=events)
    except Exception as error:
        ps.log_error(error, product='photometry')


def build_wheel(ps: PhotometrySession) -> None:
    """Fetch the encoder samples, differentiate them, and cut the responses."""
    try:
        ps.fetch_wheel()
        ps.extract_wheel_velocity()
        ps.wheel_responses = ps.extract_responses(
            {WHEEL_LABEL: ps.wheel_velocity},
            events=WHEEL_RESPONSE_EVENTS, window=WHEEL_RESPONSE_WINDOW)
    except Exception as error:
        ps.log_error(error, product='wheel')


def build_video(ps: PhotometrySession) -> None:
    """Fetch the three camera datasets and score the two video QC products.

    The datasets are independent Alyx queries and fail separately, so each is
    caught on its own: a session with no LightningPose still contributes its
    motion energy and its camera clock. The two QC steps follow in one `try`,
    so a failure in the clock check abandons the pose check below it — which
    correlates the paw speed against the wheel velocity the block above this
    one left on the session.
    """
    for fetch, product in ((ps.fetch_camera_times, 'video/times'),
                           (ps.fetch_pose, 'video/pose'),
                           (ps.fetch_motion_energy, 'video/motion_energy')):
        try:
            fetch()
        except Exception as error:
            ps.log_error(error, product=product)
    try:
        ps.run_video_times_qc()
        ps.run_pose_qc()
    except Exception as error:
        ps.log_error(error, product='video')


def build_session(ps: PhotometrySession) -> None:
    """Build one session's whole store from Alyx, four blocks in order.

    Every product is built every run; nothing is read back, skipped or checked
    against what the file already holds. Each block is one `try`, reproducing
    the boundary the pipeline had when every modality was its own script: a
    fatal step abandons its block and logs one error against it, and the next
    block still runs.

    Block order is load-bearing. Trials come first because the response cuts
    read `ps.trials`, and the wheel precedes the video because the pose QC
    reads `ps.wheel_velocity`.

    Parameters
    ----------
    ps : PhotometrySession
        The session to build. Top-level so `PhotometrySessionGroup.process`
        can hand it to parallel workers.

    Notes
    -----
    The file is rewritten rather than merged into. The truncating write carries
    the metadata and the errors, which have no data of their own to detect;
    the append that follows writes whatever products the four blocks built.
    """
    build_trials(ps)
    build_photometry(ps)
    build_wheel(ps)
    build_video(ps)
    ps.save_h5(groups=['metadata', 'errors'], mode='w')
    ps.save_h5()


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Fetch and build the photometry session store')
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
    args = parse_args(argv)
    one = _get_default_connection()

    group = fetch_catalog(one)

    # Every analysis filter is off: these are the criteria a session must clear
    # to be analysed, not to be built. The raw-photometry QC filter especially —
    # it reads the QC this pass is here to compute, so before the store is built
    # it fails every recording, and scans all of `data/sessions` to say so.
    group.filter_sessions(
        session_types=args.session_type or False, qc_blockers=set(),
        targetnms=args.target_NM or False, photometry_qc=False,
        min_performance=False,
        required_contrasts=False,
    )
    print(f'  {len(group.sessions)} sessions after filtering')

    group.process(build_session, workers=args.workers)

    errors = group.collect_errors()
    if len(errors):
        print(f'\nLogged {len(errors)} error entries across sessions:')
        print(errors['error_type'].value_counts().to_string())


if __name__ == '__main__':
    main()
