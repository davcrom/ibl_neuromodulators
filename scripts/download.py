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
    python scripts/download.py --skip video/pose        # leave LP alone
    python scripts/download.py --rebuild photometry/preprocessed
    python scripts/download.py --retry-failed           # re-attempt failed builds
"""
import argparse
from collections import Counter

import pandas as pd
from tqdm import tqdm

from iblnm.config import (
    PRODUCT_INPUTS, PRODUCT_SPEC, SESSION_TYPES, SESSIONS_FPATH, SESSIONS_H5_DIR,
)
from iblnm.data import VIDEO_QC_ERRORS, PhotometrySession, PhotometrySessionGroup
from iblnm.io import _get_default_connection
from iblnm.util import build_catalog
from iblnm.validation import (
    StaleProduct, validate_video_dropped_frames_qc, validate_video_length,
    validate_video_pin_state_qc, validate_video_timestamps_qc,
)

# The leftCamera QC checks run over every session's video. Their verdicts are
# logged, never blocking: a session with failing video QC still gets its traces.
VIDEO_QC_VALIDATORS = (
    validate_video_length,
    validate_video_timestamps_qc,
    validate_video_dropped_frames_qc,
    validate_video_pin_state_qc,
)


def with_dependents(products) -> set[str]:
    """The named products plus every product built from one of them.

    Both `--skip` and `--rebuild` reach through the `config.PRODUCT_INPUTS`
    graph this way. Skipping a product without its dependents would be a lie —
    the dependent's load method rebuilds the input it is missing — and
    rebuilding one without them would leave stored results cut from a signal
    that no longer exists.
    """
    dropped = set(products)
    while True:
        downstream = {product for product, inputs in PRODUCT_INPUTS.items()
                      if product not in dropped and dropped.intersection(inputs)}
        if not downstream:
            return dropped
        dropped |= downstream


def failed_products(ps) -> set[str]:
    """Products this session has tried and failed to build.

    A product is failed when its stored errors record an attempt and its data
    is absent. An error logged beside data that exists is informational — the
    build had caveats but produced something — and does not count.
    """
    attempted = {entry['product'] for entry in ps.errors if entry['product']}
    return {product for product in attempted
            if ps.product_status(product) == 'absent'}


def _build_trials(ps) -> None:
    """Fetch the trials table and store it; `load_trials` only fetches."""
    ps.load_trials()
    ps.save_h5(groups=['trials'])


def _build_video_times_qc(ps) -> None:
    """Score the camera clock, then run the four leftCamera QC checks over it.

    The checks produce no product of their own: they read the clock measures
    just built and the eight Alyx labels fetched live, and log a failure
    against `video/times/qc` so the pose rollup can disqualify the session
    (`data.VIDEO_QC_DISQUALIFYING_ERRORS`). They never block — every verdict
    still gets its traces extracted.
    """
    ps.load_video_times_qc()
    ps.fetch_video_qc()
    qc_row = {**ps.video_times_qc, **ps.video_qc}
    for validate in VIDEO_QC_VALIDATORS:
        try:
            validate(qc_row)
        except VIDEO_QC_ERRORS as error:
            ps.log_error(error, product='video/times/qc')


# What this script builds, in dependency order, each mapped to the call that
# reads the stored product back or builds it. The raw products are deliberately
# absent: with `config.store_raw` off nothing keeps them, so naming them here
# would fetch from Alyx on every run for data each derived load method already
# fetches on demand. `video/preprocessed` is likewise reached through
# `video/responses`, whose load method builds it — the session's own method for
# it is private.
PRODUCT_BUILDERS = {
    'trials/table':                    _build_trials,
    'trials/performance':              lambda ps: ps.load_performance(),
    'photometry/neurophotometrics/qc': lambda ps: ps.load_neurophotometrics_qc(),
    'photometry/raw/qc':               lambda ps: ps.load_photometry_qc(),
    'photometry/preprocessed':         lambda ps: ps.load_photometry(),
    'photometry/responses':            lambda ps: ps.load_responses('photometry'),
    'wheel/preprocessed':              lambda ps: ps.load_wheel(),
    'wheel/responses':                 lambda ps: ps.load_responses('wheel'),
    'video/times/qc':                  _build_video_times_qc,
    'video/pose/qc':                   lambda ps: ps.load_pose_qc(),
    'video/responses':                 lambda ps: ps.load_responses('video'),
}


def build_session(ps, skip=frozenset(), retry_failed=False) -> dict[str, str]:
    """Build every product this session is missing, in dependency order.

    Parameters
    ----------
    ps : PhotometrySession
        The session to build, carrying its own `rebuild` set. Top-level so
        `PhotometrySessionGroup.process` can pickle this function to workers.
    skip : set of str
        Products not to build, already expanded over their dependents.
    retry_failed : bool
        Re-attempt a product whose stored errors record a failed build. Off by
        default: absent data beside a recorded error is the "already tried,
        do not retry" state, which is otherwise skipped permanently.

    Returns
    -------
    dict
        Product -> 'built' or 'failed', one entry per product attempted.
        Products already current, skipped, or blocked by an upstream failure
        contribute no entry.
    """
    blocked = set(skip)
    if not retry_failed:
        blocked |= with_dependents(failed_products(ps))
    results = {}
    for product, build in PRODUCT_BUILDERS.items():
        if product in blocked:
            continue
        if product not in ps.rebuild and ps.product_status(product) == 'current':
            continue
        try:
            build(ps)
        except (BlockingIOError, StaleProduct):
            raise
        except Exception as error:
            ps.log_error(error, product=product)
            results[product] = 'failed'
            blocked |= with_dependents({product})
        else:
            results[product] = 'built'
    return results


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


def status_report(status: pd.DataFrame) -> str:
    """One line per product: how many sessions hold it current, stale or absent.

    `status` is a `PhotometrySessionGroup.scan_product_status` frame — an `eid`
    column plus one column of verdicts per surveyed product.
    """
    counts = {product: Counter(status[product])
              for product in status.columns if product != 'eid'}
    return '\n'.join(
        f'  {product:<34} ' + ', '.join(
            f'{count[verdict]} {verdict}'
            for verdict in ('current', 'stale', 'absent') if count[verdict])
        for product, count in counts.items())


def rebuild_report(status: pd.DataFrame, rebuild) -> str:
    """How many sessions each rebuilt product will overwrite.

    Only stored products can be counted, so a rebuild of a product this script
    never stores (a raw one, or a dependent skipped from the survey) reports
    nothing — its work shows up in whichever dependent is stored.
    """
    return '\n'.join(
        f'  {product:<34} {(status[product] != "absent").sum()} sessions'
        for product in sorted(rebuild) if product in status.columns)


def stale_products(status: pd.DataFrame, rebuild) -> dict[str, int]:
    """Surveyed products holding a stamp that disagrees with `config.py`."""
    return {product: int((status[product] == 'stale').sum())
            for product in status.columns
            if product != 'eid' and product not in rebuild
            and (status[product] == 'stale').any()}


def build_store(group, skip=frozenset(), rebuild=frozenset(),
                retry_failed=False, workers=1) -> list:
    """Survey the store, report what it holds, then build what is missing.

    Parameters
    ----------
    group : PhotometrySessionGroup
        The sessions in scope. Its `rebuild` set is assigned here, so every
        session it constructs — including in a worker process — carries it.
    skip, rebuild : set of str
        Products named on the command line. Each is expanded over its
        dependents here, so what the group and the sessions see is the full
        set that naming one product implies.
    retry_failed : bool
        Passed to `build_session`.
    workers : int
        `ProcessPoolExecutor` parallelism; 1 runs in this process.

    Returns
    -------
    list
        One `build_session` result per session, in `group.sessions` order,
        None for a session whose run raised.

    Raises
    ------
    SystemExit
        A product not named in `rebuild` is stored with a stale stamp. The
        store and `config.py` disagree, and which one is wrong is the user's
        call: rebuilding thousands of files is not a decision to take silently.
    """
    skip, rebuild = with_dependents(skip), with_dependents(rebuild)
    products = [product for product in PRODUCT_BUILDERS if product not in skip]
    status = group.scan_product_status(*products)
    print(f'Stored products across {len(status)} sessions:\n'
          f'{status_report(status)}')

    if rebuild:
        print(f'Rebuilding:\n{rebuild_report(status, rebuild)}')
    stale = stale_products(status, rebuild)
    if stale:
        raise SystemExit(
            'Stored stamps disagree with config.py for: '
            + ', '.join(f'{product} ({count} sessions)'
                        for product, count in stale.items())
            + '\nPass --rebuild with those products to re-derive them.')

    group.rebuild = set(rebuild)
    return group.process(build_session, workers=workers,
                         skip=set(skip), retry_failed=retry_failed)


def build_summary(results: list) -> str:
    """Per-product built/failed counts over one `build_store` pass."""
    counts = Counter((product, outcome) for result in results if result
                     for product, outcome in result.items())
    lines = [f'  {product:<34} {count} {outcome}'
             for (product, outcome), count in sorted(counts.items())]
    n_fatal = sum(result is None for result in results)
    if n_fatal:
        lines.append(f'  {n_fatal} sessions failed outright')
    return '\n'.join(lines) or '  nothing to build'


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Fetch and build the photometry session store')
    parser.add_argument('--skip', nargs='+', default=[], metavar='PRODUCT',
                        choices=tuple(PRODUCT_SPEC),
                        help='Products not to build, with their dependents '
                             '(config.PRODUCT_SPEC keys)')
    parser.add_argument('--rebuild', nargs='+', default=[], metavar='PRODUCT',
                        choices=tuple(PRODUCT_SPEC),
                        help='Products to rebuild rather than read back, with '
                             'their dependents (config.PRODUCT_SPEC keys)')
    parser.add_argument('--retry-failed', action='store_true',
                        help='Re-attempt products whose stored errors record a '
                             'failed build')
    parser.add_argument('--workers', '-w', type=int, default=1,
                        help='Number of parallel worker processes')
    parser.add_argument('--session-type', nargs='+', choices=SESSION_TYPES,
                        default=None,
                        help='Restrict to these session types (default: all)')
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    one = _get_default_connection()

    catalog = fetch_catalog(one)

    group = PhotometrySessionGroup.from_catalog(
        catalog, one=one, h5_dir=SESSIONS_H5_DIR, scan_h5_errors=False)
    group.filter_sessions(
        session_types=args.session_type or False, qc_blockers=set(),
        targetnms=False, min_performance=False, required_contrasts=False,
    )
    print(f'  {len(group.sessions)} sessions after filtering')

    results = build_store(
        group,
        skip=args.skip,
        rebuild=args.rebuild,
        retry_failed=args.retry_failed,
        workers=args.workers,
    )
    print(f'Built:\n{build_summary(results)}')

    errors = group.collect_errors()
    if len(errors):
        print(f'\nLogged {len(errors)} error entries across sessions:')
        print(errors['error_type'].value_counts().to_string())


if __name__ == '__main__':
    main()
