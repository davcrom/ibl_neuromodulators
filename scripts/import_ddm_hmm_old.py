"""
Import Old-Format DDM-HMM Fits

Legacy importer for the collaborator's first-round fits in `data/ddm-hmm.old/`,
kept until those mice are refit in the new format. Writes each mouse's best-K
fit into its sessions' `hmm/ddm-k{K}` under the new format's names, so the
store holds one naming whichever format a fit came from. Every subject with a
directory in `config.DDM_HMM_DIR` is skipped: its new-format fit supersedes the
old one whole, and `scripts/import_ddm_hmm.py` writes it.

The old fit dropped some trials (no-go, long reaction time) and does not name
the ones it kept, so its rows are matched to the stored trials by ordered
reaction time and checked by absolute contrast.

Run after every `scripts/download.py`, alongside `scripts/import_ddm_hmm.py`.

Input: data/ddm-hmm.old/all_mice_bestK_params.csv,
       data/ddm-hmm.old/model_selection_all.csv,
       data/ddm-hmm.old/{subject}_K{K}_{posteriors,params,transition}.csv
Output: hmm/ddm-k{K} in data/sessions/{eid}.h5

Usage:
    python scripts/import_ddm_hmm_old.py [--subject SUBJECT ...]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from iblnm.config import DDM_HMM_DIR, PROJECT_ROOT, SESSIONS_FPATH, SESSIONS_H5_DIR
from iblnm.data import PhotometrySession, PhotometrySessionGroup

DDM_HMM_OLD_DIR = PROJECT_ROOT / 'data' / 'ddm-hmm.old'

# The run summary a new-format fit carries (`model_comparison.csv` columns).
# Every one is written for an old fit too, NaN where the old format has none.
NEW_SUMMARY_FIELDS = (
    'subject', 'K_ddm', 'K_total', 'restart', 'seed', 'criterion', 'score',
    'loglik', 'logpost', 'n_params', 'n_trials', 'n_sessions', 'n_omissions',
    'AIC', 'BIC', 'converged', 'iterations', 'seconds', 'share_alpha', 'rt_max',
    'host', 'finished',
)
# Old run-summary names for quantities the new format also carries.
SUMMARY_NAMES = {'K': 'K_ddm', 'logL': 'loglik', 'bic': 'BIC'}


def align_posteriors_to_trials(
    block: pd.DataFrame, trials: pd.DataFrame, atol: float = 1e-3
) -> np.ndarray:
    """Match each DDM-HMM posterior row to its canonical trial by ordered RT.

    The posteriors CSV is a chronological subsequence of the session's trials
    (the fit dropped some trials by a preprocessing rule not recoverable from
    the trial columns). Both are walked in order — ``block`` by
    ``trial_in_dataset``, ``trials`` by ``stimOn_times`` — matching each block
    ``rt`` to the next trial whose ``response_times - stimOn_times`` equals it.
    Order preservation makes RT collisions harmless.

    ``stimOn_times`` here is not the pipeline's onset clock
    (``config.STIM_ONSET_EVENT``, the Bpod trigger) but the column the
    collaborator's fit measured its RTs from. It is a matching key against a
    foreign file, so it tracks that file's definition; the two clocks differ by
    ~60 ms, far more than ``atol``, and nothing would match if this drifted.

    Parameters
    ----------
    block : pandas.DataFrame
        Posterior rows for one eid, sorted by ``trial_in_dataset``; needs an
        ``rt`` column (seconds).
    trials : pandas.DataFrame
        Session trials sorted by ``stimOn_times``; needs ``response_times`` and
        ``stimOn_times`` (seconds).
    atol : float
        RT match tolerance in seconds.

    Returns
    -------
    numpy.ndarray
        ``trials`` index labels, one per ``block`` row, in block order.

    Raises
    ------
    ValueError
        If a block row has no ordered RT match (the CSV is not a subsequence of
        these trials — fail loud).
    """
    block_rt = block['rt'].to_numpy()
    trial_rt = (trials['response_times'] - trials['stimOn_times']).to_numpy()
    trial_labels = trials.index.to_numpy()
    matched = np.empty(len(block_rt), dtype=trial_labels.dtype)
    j = 0
    for i, rt in enumerate(block_rt):
        while j < len(trial_rt) and abs(trial_rt[j] - rt) > atol:
            j += 1
        if j >= len(trial_rt):
            raise ValueError(
                f"posteriors row {i} (rt={rt}) has no ordered RT match in trials")
        matched[i] = trial_labels[j]
        j += 1
    return matched


def align_session(block: pd.DataFrame, trials: pd.DataFrame) -> pd.DataFrame:
    """Key one session's posterior rows by the stored trials they describe.

    The match is verified by requiring ``|signed_contrast|`` to agree on every
    matched trial (CSV as a fraction, store as a percent), which catches any
    RT-collision misalignment.

    Parameters
    ----------
    block : pandas.DataFrame
        That session's rows of a `{subject}_K{K}_posteriors.csv`, any order.
    trials : pandas.DataFrame
        The session's stored `trials/table`: `trial`, `stimOn_times`,
        `response_times`, `signed_contrast`.

    Returns
    -------
    pandas.DataFrame
        The rows in chronological order, one per matched trial, with a `trial`
        column holding that trial's stored identity. Trials the fit dropped
        have no row.

    Raises
    ------
    ValueError
        A row has no ordered RT match, or a matched trial's ``|contrast|``
        disagrees with the CSV.
    """
    block = block.sort_values('trial_in_dataset')
    matched = align_posteriors_to_trials(block, trials.sort_values('stimOn_times'))
    csv_abs_contrast = block['signed_contrast'].abs().to_numpy() * 100
    h5_abs_contrast = trials.loc[matched, 'signed_contrast'].abs().to_numpy()
    if not np.allclose(csv_abs_contrast, h5_abs_contrast, atol=1e-2):
        raise ValueError("|contrast| mismatch between posteriors and trials "
                         "(RT-collision misalignment)")
    return block.assign(trial=trials.loc[matched, 'trial'].to_numpy()
                        ).reset_index(drop=True)


def read_fit(subject: str, old_dir: Path) -> tuple[int, dict]:
    """Read one mouse's best-K fit into the attrs `hmm/ddm-k{K}` holds.

    Parameters
    ----------
    subject : str
        The mouse, as named in the CSVs.
    old_dir : pathlib.Path
        The old-format fit directory, `DDM_HMM_OLD_DIR` outside tests.

    Returns
    -------
    k : int
        The mouse's best K, from `all_mice_bestK_params.csv`.
    attrs : dict
        Every `NEW_SUMMARY_FIELDS` name, NaN where the old format has no
        counterpart, filled from the `model_selection_all.csv` row for
        (`subject`, `k`); plus one array per state (K entries, in state order)
        for each `params` column, `self_transition`, `occupancy` and each
        `trans_to_{j}` transition column. Old names the new format also
        carries are renamed to the new format's.
    """
    best = pd.read_csv(old_dir / 'all_mice_bestK_params.csv')
    per_state = best[best['mouse'] == subject].sort_values('state')
    k = int(per_state['best_K'].iloc[0])
    selection = pd.read_csv(old_dir / 'model_selection_all.csv')
    row = selection[(selection['subject'] == subject)
                    & (selection['K'] == k)].squeeze(axis=0)
    params = pd.read_csv(old_dir / f'{subject}_K{k}_params.csv').rename(
        columns={'pi0': 'init'})
    transition = (pd.read_csv(old_dir / f'{subject}_K{k}_transition.csv')
                  .sort_values('from_state').drop(columns='from_state')
                  .add_prefix('trans_'))
    attrs = dict.fromkeys(NEW_SUMMARY_FIELDS, np.nan)
    attrs |= row.rename(SUMMARY_NAMES).to_dict()
    attrs |= {column: values.to_numpy()
              for column, values in pd.concat(
                  [params, transition], axis=1).items()}
    attrs |= {column: per_state[column].to_numpy()
              for column in ('self_transition', 'occupancy')}
    return k, attrs


def import_session(ps: PhotometrySession, k: int, posteriors: pd.DataFrame,
                   attrs: dict) -> list[int]:
    """Replace one session's `hmm` group with its rows of the mouse's fit.

    The rows are aligned before anything is written, so a session that fails
    keeps whatever `hmm` it held. A session the fit does not hold ends with no
    `hmm` group, as it does under `scripts/import_ddm_hmm.py`.

    Parameters
    ----------
    ps : PhotometrySession
        The session, addressing its file in the store.
    k : int
        The fit's K, naming the group `hmm/ddm-k{k}`.
    posteriors : pandas.DataFrame
        The mouse's whole `{subject}_K{k}_posteriors.csv`, every session.
    attrs : dict
        `read_fit` attrs for the mouse.

    Returns
    -------
    list of int
        The Ks written: `[k]`, or empty when the fit does not hold the session.

    Raises
    ------
    ValueError
        The store holds no trials for the session, or `align_session` fails.
    """
    ps.load_h5(groups=['trials'])
    if not hasattr(ps, 'trials'):
        raise ValueError(f"{ps.eid}: no stored trials to align the fit to")
    block = posteriors[posteriors['eid'] == ps.eid]
    ps.hmm = {}
    if not block.empty:
        fit_trials = align_session(block, ps.trials).rename(
            columns=lambda column: column.replace('state_', 'p_state_', 1)
            if column.startswith('state_') else column)
        ps.hmm[k] = {'trials': fit_trials, 'attrs': attrs}
    ps.save_h5(groups=['hmm'])
    return sorted(ps.hmm)


def select_subjects(old_dir: Path, new_dir: Path,
                    subjects: list[str] | None = None) -> list[str]:
    """The old-format mice to import: those `new_dir` holds no fit for.

    `subjects` narrows the set (the `--subject` flag) but never overrides the
    skip, since a new-format fit supersedes the old one whole.
    """
    mice = pd.read_csv(old_dir / 'all_mice_bestK_params.csv')['mouse'].unique()
    return sorted(mouse for mouse in mice
                  if not (new_dir / mouse).is_dir()
                  and (subjects is None or mouse in subjects))


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject', nargs='+',
                        help='import only these subjects (default: every '
                             'old-format mouse without a new-format fit)')
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    subjects = select_subjects(DDM_HMM_OLD_DIR, DDM_HMM_DIR, args.subject)

    catalog = pd.read_parquet(SESSIONS_FPATH)
    # No ONE connection: trials are read from the store, and the filters that
    # the scan would feed are all off.
    group = PhotometrySessionGroup.from_catalog(catalog, h5_dir=SESSIONS_H5_DIR,
                                                scan_h5=False)
    for subject in subjects:
        k, attrs = read_fit(subject, DDM_HMM_OLD_DIR)
        posteriors = pd.read_csv(DDM_HMM_OLD_DIR / f'{subject}_K{k}_posteriors.csv')
        # Every catalogued session of the mouse, whatever its type or QC: the
        # fit's own rows decide which sessions hold it.
        group.filter_sessions(
            session_types=False,
            exclude_subjects=sorted(set(catalog['subject']) - {subject}),
            exclude_eids=False, qc_blockers=False, targetnms=False,
            photometry_qc=False, min_performance=False,
            required_contrasts=False,
        )
        results = group.process(import_session, k=k, posteriors=posteriors,
                                attrs=attrs)
        written = [ks for ks in results if ks is not None]
        print(f"{subject}: {len(written)} sessions written "
              f"({sum(bool(ks) for ks in written)} holding a fit), "
              f"{len(results) - len(written)} skipped")


if __name__ == '__main__':
    main()
