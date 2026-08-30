"""Build the derived products every script reads out of `data/sessions/*.h5`.

`scripts/download.py` runs this over the whole catalog; every analysis script
runs it over the products it is about to read, so its loop meets a complete
store instead of rebuilding sessions one at a time inside the analysis.

Detection is automatic, rebuilding is manual: missing data affects one session
and is cheap to build, while a stale stamp can mean the whole store, so a
`config.py` change that was not intended stops the run rather than silently
re-deriving thousands of files.
"""
import argparse
from collections import Counter

import pandas as pd

from iblnm.config import PRODUCT_INPUTS, PRODUCT_SPEC
from iblnm.data import VIDEO_QC_ERRORS
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


# What this module builds, in dependency order, each mapped to the call that
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

ALL_PRODUCTS = tuple(PRODUCT_BUILDERS)


def build_session(ps, products=ALL_PRODUCTS, retry_failed=False) -> dict[str, str]:
    """Build every product this session is missing, in dependency order.

    Parameters
    ----------
    ps : PhotometrySession
        The session to build, carrying its own `rebuild` set. Top-level so
        `PhotometrySessionGroup.process` can pickle this function to workers.
    products : set of str
        The products to consider. Everything outside it is left alone, which
        is how both `--skip` and an analysis script's own product list reach
        this loop; `PRODUCT_BUILDERS` still fixes the order they build in.
    retry_failed : bool
        Re-attempt a product whose stored errors record a failed build. Off by
        default: absent data beside a recorded error is the "already tried,
        do not retry" state, which is otherwise skipped permanently.

    Returns
    -------
    dict
        Product -> 'built' or 'failed', one entry per product attempted.
        Products already current, out of scope, or blocked by an upstream
        failure contribute no entry.
    """
    blocked = set() if retry_failed else with_dependents(failed_products(ps))
    results = {}
    for product, build in PRODUCT_BUILDERS.items():
        if product not in products or product in blocked:
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

    Only stored products can be counted, so a rebuild of a product this pass
    never stores (a raw one, or a dependent outside the survey) reports
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


def build_store(group, products=ALL_PRODUCTS, skip=frozenset(),
                rebuild=frozenset(), retry_failed=False, workers=1) -> list:
    """Survey the store, report what it holds, then build what is missing.

    This is the pre-warm an analysis script runs before iterating: naming the
    products it is about to read leaves it looping over a complete store rather
    than discovering gaps one session at a time.

    Parameters
    ----------
    group : PhotometrySessionGroup
        The sessions in scope. Its `rebuild` set is assigned here, so every
        session it constructs — including in a worker process — carries it.
    products : sequence of str
        The products to survey and build. Defaults to everything this module
        knows how to build; an analysis script passes the ones it reads.
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
    StaleProduct
        A surveyed product not named in `rebuild` is stored with a stale
        stamp. The store and `config.py` disagree, and which one is wrong is
        the user's call: rebuilding thousands of files is not a decision to
        take silently.
    """
    skip, rebuild = with_dependents(skip), with_dependents(rebuild)
    products = [product for product in products if product not in skip]
    status = group.scan_product_status(*products)
    print(f'Stored products across {len(status)} sessions:\n'
          f'{status_report(status)}')

    if rebuild:
        print(f'Rebuilding:\n{rebuild_report(status, rebuild)}')
    stale = stale_products(status, rebuild)
    if stale:
        raise StaleProduct(
            'Stored stamps disagree with config.py for: '
            + ', '.join(f'{product} ({count} sessions)'
                        for product, count in stale.items())
            + '\nPass --rebuild with those products to re-derive them.')

    group.rebuild = set(rebuild)
    return group.process(build_session, workers=workers,
                         products=set(products), retry_failed=retry_failed)


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


def add_store_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the `--rebuild` and `--workers` flags every analysis script shares.

    `--rebuild` is validated against `config.PRODUCT_SPEC` at parse time, so a
    mistyped product name fails before any session is touched rather than
    silently rebuilding nothing.
    """
    parser.add_argument('--rebuild', nargs='+', default=[], metavar='PRODUCT',
                        choices=tuple(PRODUCT_SPEC),
                        help='Products to rebuild rather than read back, with '
                             'their dependents (config.PRODUCT_SPEC keys)')
    parser.add_argument('--workers', '-w', type=int, default=1,
                        help='Number of parallel worker processes used to '
                             'build missing products before the analysis runs')
