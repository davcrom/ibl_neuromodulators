"""Fetch the session catalog from Alyx and build every product in the store.

Two phases. Phase one queries Alyx for the project's session list, writes each
new session's metadata into `data/sessions/{eid}.h5`, and runs the fixups that
need every session at once, leaving the catalog in `metadata/sessions.pqt`.
Phase two builds every product each session is missing, through the group's
`process`.

Detection is automatic, rebuilding is manual: missing data affects one session
and is cheap to build, while a stale stamp can mean the whole store, so a
`config.py` change that was not intended stops the run rather than silently
re-deriving 5525 files.

Usage:
    python scripts/download.py                          # build everything missing
    python scripts/download.py --workers 4              # in parallel
    python scripts/download.py --session-type biased    # one session type
    python scripts/download.py --skip video             # leave the camera alone
    python scripts/download.py --rebuild photometry
    python scripts/download.py --retry-failed           # re-attempt failed builds
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
    PRODUCT_SPEC, SESSION_TYPES, SESSIONS_FPATH, SESSIONS_H5_DIR,
)
from iblnm.data import (  # noqa: E402
    PREPROCESSED_BAND, WHEEL_LABEL, _RESPONSE_MODALITIES, PhotometrySession,
    PhotometrySessionGroup,
)
from iblnm.io import _get_default_connection  # noqa: E402
from iblnm.util import build_catalog  # noqa: E402
from iblnm.validation import (  # noqa: E402
    MissingLP, MissingMotionEnergy, StaleProduct,
)

MODALITIES = ('trials', 'photometry', 'wheel', 'video')


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


def fetch_catalog(one) -> pd.DataFrame:
    """Query Alyx for the project's sessions and return the fixed-up catalog.

    Sessions already holding an H5 file keep their stored metadata; only new
    eids are queried. The catalog is rebuilt from every file's `metadata` group
    afterwards, run through `build_catalog`, and written to
    `config.SESSIONS_FPATH` — the rollup script regenerates the same table from
    the same source, so this copy is a convenience, not a second source.
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

    catalog = build_catalog(
        PhotometrySessionGroup.from_h5_dir(SESSIONS_H5_DIR, one=one).sessions)
    catalog.to_parquet(SESSIONS_FPATH, index=False)
    print(f'Catalogued {len(catalog)} sessions in {SESSIONS_FPATH}')
    return catalog


def fetch_movement_sources(ps: PhotometrySession) -> None:
    """Fetch the pose and the motion energy, tolerating a missing one.

    The two signal sources are independent: a session with no LightningPose
    still contributes its motion energy channel. A missing one is logged
    against its own product and leaves the other's channels intact, which is
    also what marks it settled so the next run does not fetch it again.
    """
    for fetch, product, missing in (
            (ps.fetch_pose, 'video/pose', MissingLP),
            (ps.fetch_motion_energy, 'video/motion_energy', MissingMotionEnergy)):
        try:
            fetch()
        except missing as error:
            ps.log_error(error, product=product)


def cut_responses(ps: PhotometrySession, modality: str) -> None:
    """Cut one modality's peri-event matrices from the signals it holds.

    The signals come off the session attributes the block has just populated
    rather than through the modality's `load_*`, which would consult the store
    for what is already in memory, and would refetch the raw under `--rebuild`.
    The events and window are the modality's defaults, the same ones
    `load_responses` falls back on.
    """
    _, attribute, defaults = _RESPONSE_MODALITIES[modality]
    if ps.trials is None:
        ps.load_trials()
    signals = RESPONSE_SIGNALS[modality](ps)
    setattr(ps, attribute, ps.extract_responses(signals, **defaults))


def ensure_wheel_velocity(ps: PhotometrySession) -> None:
    """Put the wheel velocity on the session for the cross-modal pose QC.

    `video/pose/qc` is the one product needing another modality's signal. When
    the wheel block ran, its velocity is already here; when it did not — it was
    skipped, or had nothing left to build — `load_wheel` builds and stores it,
    since the work has to happen either way.
    """
    if ps.wheel_velocity is None:
        ps.load_wheel()


# The preprocessed signals each modality cuts its responses from, taken from
# memory. Photometry's labels are brain regions, the wheel's is its one channel,
# video's are the movement channels.
RESPONSE_SIGNALS = {
    'photometry': lambda ps: ps.photometry[PREPROCESSED_BAND],
    'wheel':      lambda ps: {WHEEL_LABEL: ps.wheel_velocity},
    'video':      lambda ps: ps.movement_signals,
}

# What each modality builds, in the order that lets one fetch serve every
# product made from it: the raw datasets arrive once, and everything cut from
# them is cut while they are still in memory. Each entry is a product and the
# calls that make it; a raise inside one abandons the rest of its modality,
# which is where blocking comes from — there is no dependency walk.
BUILD_STEPS = {
    'trials': (
        ('trials/table',       (lambda ps: ps.fetch_trials(),)),
        ('trials/performance', (lambda ps: ps.extract_performance(),)),
    ),
    'photometry': (
        ('photometry/neurophotometrics/qc',
         (lambda ps: ps.fetch_neurophotometrics(),
          lambda ps: ps.run_neurophotometrics_qc())),
        ('photometry/raw/qc',
         (lambda ps: ps.fetch_photometry(),
          lambda ps: ps.run_photometry_qc())),
        ('photometry/preprocessed',
         (lambda ps: ps.extract_preprocessed_photometry(),)),
        ('photometry/responses', (lambda ps: cut_responses(ps, 'photometry'),)),
    ),
    'wheel': (
        ('wheel/preprocessed',
         (lambda ps: ps.fetch_wheel(), lambda ps: ps.extract_wheel_velocity())),
        ('wheel/responses', (lambda ps: cut_responses(ps, 'wheel'),)),
    ),
    'video': (
        ('video/times/qc',
         (lambda ps: ps.fetch_camera_times(),
          lambda ps: ps.run_video_times_qc())),
        ('video/preprocessed',
         (fetch_movement_sources, lambda ps: ps.extract_movement_signals())),
        ('video/responses', (lambda ps: cut_responses(ps, 'video'),)),
        ('video/pose/qc', (ensure_wheel_velocity, lambda ps: ps.run_pose_qc())),
    ),
}


def modality_products(modalities) -> set[str]:
    """Every `config.PRODUCT_SPEC` product belonging to the named modalities.

    The flags name modalities; a session's `rebuild` set names products, since
    that is what its load methods and stamps are keyed on.
    """
    return {product for product in PRODUCT_SPEC
            if product.split('/')[0] in modalities}


def pending_products(ps: PhotometrySession, modality: str,
                     retry_failed: bool = False) -> set[str]:
    """The modality's products that are worth attempting.

    A product is pending unless the store holds it current, or it is settled —
    tried before, logged an error, and left no data. Without the settled state a
    session whose Alyx data does not exist would refetch it on every run to fail
    the same way; `retry_failed` is how the user asks for exactly that.

    The raw products are not consulted: with `config.store_raw` off nothing
    keeps them, so they are absent on every session and would make every
    modality pending forever. A raw dataset that cannot be fetched settles the
    products built from it instead.
    """
    settled = set() if retry_failed else ps.failed_products()
    return {product for product, _ in BUILD_STEPS[modality]
            if product not in settled
            and ps.product_status(product) != 'current'}


def build_session(ps: PhotometrySession, modalities=MODALITIES,
                  rebuild=frozenset(), retry_failed=False) -> dict[str, str]:
    """Build one session's whole store, one modality at a time.

    Each modality fetches its raw datasets once, computes every product made
    from them while they are in memory, and is written once at the end. This is
    the bulk path: the `load_*` methods answer "give me this, repair it if
    missing", which is the right contract at a prompt and the wrong one here.

    Parameters
    ----------
    ps : PhotometrySession
        The session to build, carrying its own `rebuild` product set. Top-level
        so `PhotometrySessionGroup.process` can pass it to workers.
    modalities : sequence of str
        Keys of `BUILD_STEPS`. A modality left out is not built for its own
        sake, though another modality's product may still build it — the video
        block's pose QC needs the wheel either way.
    rebuild : set of str
        Modalities to rebuild whether or not the store already holds them.
    retry_failed : bool
        Attempt products whose stored errors record a failed build.

    Returns
    -------
    dict
        Product -> 'built' or 'failed', one entry per product attempted. A
        modality that was skipped contributes no entries.
    """
    results = {}
    for modality, steps in BUILD_STEPS.items():
        if modality not in modalities:
            continue
        if (modality not in rebuild
                and not pending_products(ps, modality, retry_failed)):
            continue
        for product, calls in steps:
            try:
                for call in calls:
                    call(ps)
            except (BlockingIOError, StaleProduct):
                raise
            except Exception as error:
                ps.log_error(error, product=product)
                results[product] = 'failed'
                break
            results[product] = 'built'
        ps.save_h5(groups=[modality])
    return results


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Fetch and build the photometry session store')
    parser.add_argument('--skip', nargs='+', default=[], metavar='MODALITY',
                        choices=MODALITIES,
                        help='Modalities not to build')
    parser.add_argument('--rebuild', nargs='+', default=[], metavar='MODALITY',
                        choices=MODALITIES,
                        help='Modalities to rebuild rather than read back')
    parser.add_argument('--workers', '-w', type=int, default=1,
                        help='Number of parallel worker processes')
    parser.add_argument('--retry-failed', action='store_true',
                        help='Re-attempt products whose stored errors record a '
                             'failed build')
    parser.add_argument('--session-type', nargs='+', choices=SESSION_TYPES,
                        default=None,
                        help='Restrict to these session types (default: all)')
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    one = _get_default_connection()

    catalog = fetch_catalog(one)

    group = PhotometrySessionGroup.from_catalog(
        catalog, one=one, h5_dir=SESSIONS_H5_DIR, scan_h5=False)
    # Every analysis filter is off: these are the criteria a session must clear
    # to be analysed, not to be built. The raw-photometry QC filter especially —
    # it reads the QC this pass is here to compute, so before the store is built
    # it fails every recording, and scans all of `data/sessions` to say so.
    group.filter_sessions(
        session_types=args.session_type or False, qc_blockers=set(),
        targetnms=False, photometry_qc=False, min_performance=False,
        required_contrasts=False,
    )
    print(f'  {len(group.sessions)} sessions after filtering')

    # The sessions carry the products to rebuild, the build carries the
    # modalities: one names what a load method may read back, the other what
    # the pass fetches at all.
    group.rebuild = modality_products(args.rebuild)
    group.check_products(*PRODUCT_SPEC, rebuild=group.rebuild)
    group.process(
        build_session,
        modalities=tuple(m for m in MODALITIES if m not in args.skip),
        rebuild=set(args.rebuild),
        retry_failed=args.retry_failed,
        workers=args.workers,
    )

    errors = group.collect_errors()
    if len(errors):
        print(f'\nLogged {len(errors)} error entries across sessions:')
        print(errors['error_type'].value_counts().to_string())


if __name__ == '__main__':
    main()
