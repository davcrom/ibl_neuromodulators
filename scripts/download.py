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

import pandas as pd
from tqdm import tqdm

from iblnm.config import (
    PRODUCT_SPEC, SESSION_TYPES, SESSIONS_FPATH, SESSIONS_H5_DIR,
)
from iblnm.data import PhotometrySession, PhotometrySessionGroup
from iblnm.io import _get_default_connection
from iblnm.store import add_store_arguments, build_store, build_summary
from iblnm.util import build_catalog


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


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Fetch and build the photometry session store')
    parser.add_argument('--skip', nargs='+', default=[], metavar='PRODUCT',
                        choices=tuple(PRODUCT_SPEC),
                        help='Products not to build, with their dependents '
                             '(config.PRODUCT_SPEC keys)')
    add_store_arguments(parser)
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
