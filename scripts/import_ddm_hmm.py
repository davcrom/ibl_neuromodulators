"""
Import DDM-HMM Fits

Writes a collaborator's per-mouse DDM-HMM fits (drift-diffusion observation
model, hidden-Markov latent states) into the session store, one
`hmm/ddm-k{K}` group per fitted K. Every K of every fit is imported; which K
to analyse is the overview's choice, not this script's.

Run after every `scripts/download.py`, whose whole-file rewrite erases `hmm`,
and whenever new fits arrive.

Input: data/ddm-hmm/model_comparison.csv, data/ddm-hmm/{subject}/trials_K{K}.csv,
       data/ddm-hmm/{subject}/params_K{K}.csv
Output: hmm/ddm-k{K} in data/sessions/{eid}.h5

Usage:
    python scripts/import_ddm_hmm.py [--subject SUBJECT ...]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from iblnm.config import DDM_HMM_DIR, SESSIONS_FPATH, SESSIONS_H5_DIR
from iblnm.data import PhotometrySession, PhotometrySessionGroup

# The fit files name three DDM parameters in Unicode; the store uses ASCII.
PARAM_NAMES = {'α': 'alpha', 'a₀': 'a0', 'τ': 'tau'}


def check_alignment(fit_trials: pd.DataFrame, trials: pd.DataFrame) -> None:
    """Raise unless one session's fit rows describe its stored trials.

    The fit is keyed by `trial_n`, which must name the stored `trial` column
    one-to-one. Two task columns are then compared trial by trial, because
    each is coded differently in the fit: contrast is a fraction there and a
    percentage in the store, and choice carries the opposite sign.

    Parameters
    ----------
    fit_trials : pandas.DataFrame
        That session's rows of a `trials_K{K}.csv`: `trial_n`, `c`, `choice`.
    trials : pandas.DataFrame
        The session's stored `trials/table`: `trial`, `contrast`, `choice`.

    Raises
    ------
    ValueError
        Row counts differ, the trial identities differ, or any trial's contrast
        or sign-flipped choice disagrees.
    """
    if len(fit_trials) != len(trials):
        raise ValueError(f"fit holds {len(fit_trials)} trials, "
                         f"store holds {len(trials)}")
    if set(fit_trials['trial_n']) != set(trials['trial']):
        raise ValueError("fit trial_n does not match stored trial")
    joined = fit_trials.merge(trials, left_on='trial_n', right_on='trial',
                              suffixes=('_fit', ''))
    if not np.isclose(joined['c'] * 100, joined['contrast']).all():
        raise ValueError("fit contrast does not match stored contrast")
    if not (joined['choice_fit'] == -joined['choice']).all():
        raise ValueError("fit choice is not the sign-flipped stored choice")


def read_fits(subject_dir: Path, comparison: pd.DataFrame) -> dict[int, dict]:
    """Read every K of one mouse's fit into the form `ps.hmm` holds.

    Parameters
    ----------
    subject_dir : pathlib.Path
        `DDM_HMM_DIR / subject`, holding `trials_K{K}.csv` and
        `params_K{K}.csv` per K. The directory name is the subject.
    comparison : pandas.DataFrame
        `model_comparison.csv`, one row per subject x `K_ddm`: the run summary
        of the restart the fit files came from.

    Returns
    -------
    dict
        K -> ``{'trials': DataFrame, 'attrs': dict}``. `trials` is every
        fitted session's rows, unsplit. `attrs` is every field of the
        comparison row plus one array per `params` column, indexed by state
        (K+1 entries, the no-response state last), with the Unicode parameter
        names made ASCII so they are addressable in code.
    """
    fits = {}
    for trials_fpath in sorted(subject_dir.glob('trials_K*.csv')):
        k = int(trials_fpath.stem.removeprefix('trials_K'))
        params = pd.read_csv(subject_dir / f'params_K{k}.csv').rename(
            columns=PARAM_NAMES)
        row = comparison[(comparison['subject'] == subject_dir.name)
                         & (comparison['K_ddm'] == k)].squeeze(axis=0)
        attrs = row.to_dict()
        # Object columns (`kind`) become str arrays, which the attr pair stores.
        attrs |= {column: values.to_numpy(dtype=str if values.dtype == object
                                          else values.dtype)
                  for column, values in params.items()}
        fits[k] = {'trials': pd.read_csv(trials_fpath), 'attrs': attrs}
    return fits


def import_session(ps: PhotometrySession, fits: dict[int, dict]) -> list[int]:
    """Replace one session's `hmm` group with the Ks of `fits` that hold it.

    Every K is checked before anything is written, so a session that fails
    any K keeps whatever `hmm` it held. A session no K holds ends with no
    `hmm` group at all: `_save_hmm` deletes it for an empty mapping, which
    also clears old-format fits of a mouse the new format supersedes.

    Parameters
    ----------
    ps : PhotometrySession
        The session, addressing its file in the store.
    fits : dict
        `read_fits` output for the session's subject.

    Returns
    -------
    list of int
        The Ks written, ascending; empty when no K holds the session.

    Raises
    ------
    ValueError
        The store holds no trials for the session, or `check_alignment` fails.
    """
    ps.load_h5(groups=['trials'])
    if not hasattr(ps, 'trials'):
        raise ValueError(f"{ps.eid}: no stored trials to align the fit to")
    ps.hmm = {}
    for k, fit in sorted(fits.items()):
        rows = fit['trials'][fit['trials']['eid'] == ps.eid]
        if rows.empty:
            continue
        check_alignment(rows, ps.trials)
        ps.hmm[k] = {'trials': rows.assign(trial=rows['trial_n'])
                     .reset_index(drop=True),
                     'attrs': fit['attrs']}
    ps.save_h5(groups=['hmm'])
    return sorted(ps.hmm)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject', nargs='+',
                        help='import only these subjects (default: every '
                             'subject directory in DDM_HMM_DIR)')
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    comparison = pd.read_csv(DDM_HMM_DIR / 'model_comparison.csv')
    subject_dirs = sorted(path for path in DDM_HMM_DIR.iterdir()
                          if path.is_dir()
                          and (args.subject is None or path.name in args.subject))

    catalog = pd.read_parquet(SESSIONS_FPATH)
    # No ONE connection: trials are read from the store, and the filters that
    # the scan would feed are all off.
    group = PhotometrySessionGroup.from_catalog(catalog, h5_dir=SESSIONS_H5_DIR,
                                                scan_h5=False)
    for subject_dir in subject_dirs:
        subject = subject_dir.name
        # Every catalogued session of the mouse, whatever its type or QC: the
        # fit's own rows decide which sessions hold it.
        group.filter_sessions(
            session_types=False,
            exclude_subjects=sorted(set(catalog['subject']) - {subject}),
            exclude_eids=False, qc_blockers=False, targetnms=False,
            photometry_qc=False, min_performance=False,
            required_contrasts=False,
        )
        results = group.process(import_session,
                                fits=read_fits(subject_dir, comparison))
        written = [ks for ks in results if ks is not None]
        print(f"{subject}: {len(written)} sessions written "
              f"({sum(bool(ks) for ks in written)} holding a fit), "
              f"{len(results) - len(written)} skipped")


if __name__ == '__main__':
    main()
