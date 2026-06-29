"""Session-swap permutation tests of pre-stimulus baseline coding.

Tests whether a recording's within-session baseline fluctuations carry
information about behaviour. Each --model is one independent analysis; run them
as separate processes to parallelize.

Usage:
    python scripts/baseline.py --model performance
    python scripts/baseline.py --model reaction_time
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from iblnm.config import SESSIONS_FPATH, PROJECT_ROOT, FIGURE_DPI
from iblnm.io import _get_default_connection
from iblnm.data import PhotometrySessionGroup
from iblnm.util import load_or_collect_session_errors
from iblnm.vis import plot_baseline_propsig, plot_baseline_r2

def prepare_session(ps):
    assert len(ps.brain_region) == 1
    ps.load_h5(groups=['trials', 'photometry'])
    ps.trials = ps.trials[
        (ps.trials['choice'] != 0)
        & ((ps.trials['firstMovement_times'] - ps.trials['stimOn_times']) >= 0.05)
        ].copy()
    ps.trials['rt'] = ps.trials['feedback_times'] - ps.trials['stimOn_times']
    ps.trials['log_rt'] = ps.trials['rt'].apply(lambda x: np.log(x) if x > 0 else np.nan)
    ps.correct = ps.trials['feedbackType'].apply(lambda x: 1 if x > 0 else 0).to_numpy()
    ps.relative_log_rt = ps.trials.groupby('contrast')['log_rt'].transform(lambda x: (x - x.mean()) / x.std()).to_numpy()
    responses = ps.extract_responses(events=['stimOn_times'], window=[-0.4, -0.1])
    baseline = responses[ps.brain_region[0]].sel(event='stimOn_times').mean(axis=1).to_numpy()
    # z-score within session so the slope is comparable across recordings and
    # the donor-swap null injects no cross-session scale differences
    ps.baseline = (baseline - np.nanmean(baseline)) / np.nanstd(baseline)
    return ps

def linear_regression(rt, baseline):
    from scipy.stats import linregress
    mask = (~np.isnan(rt) & ~np.isnan(baseline))
    res = linregress(rt[mask], baseline[mask])
    return {'slope': res.slope, 'r2': res.rvalue ** 2}

def logistic_regression(correct, baseline):
    from statsmodels.formula.api import logit
    formula = 'correct ~ baseline'
    data = pd.DataFrame(
        data=np.column_stack([correct, baseline]),
        columns=['correct', 'baseline']
        ).dropna()
    res = logit(formula, data=data).fit(disp=0)
    return {'slope': res.params['baseline'], 'r2': res.prsquared}


# Each model is one independent permutation analysis; run them as separate
# processes (one per --model) to parallelize. Maps the selector to its
# (statistic function, fixed regressor, output filename).
MODELS = {
    'performance': (logistic_regression, ['correct'], 'performance.pqt'),
    'reaction_time': (linear_regression, ['relative_log_rt'], 'reaction_time.pqt'),
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS, required=True,
                        help='which baseline analysis to run')
    parser.add_argument('--reprocess', action='store_true',
                        help='load the saved parquet and re-plot without '
                             'rerunning the ~10h permutation test')
    args = parser.parse_args()
    model_fn, fixed_var, fname = MODELS[args.model]
    out_dir = Path('results/baseline')

    if args.reprocess:
        results = pd.read_parquet(out_dir / fname)
    else:
        catalog = pd.read_parquet(SESSIONS_FPATH)
        catalog = catalog.merge(
            load_or_collect_session_errors(catalog['eid']),
            on='eid', how='left'
            )
        group = PhotometrySessionGroup.from_catalog(
            catalog,
            one=_get_default_connection(),
            scan_h5_errors=False
            )
        group.filter_sessions(
            session_types=('biased', 'ephys',),
        )
        _ = group.deduplicate()

        out_dir.mkdir(parents=True, exist_ok=True)
        results = group.session_permutation_test(
            prepare_session,
            model_fn,
            fixed_var=fixed_var,
            swapped_var=['baseline'],
            statistic_key='r2',
            alternative='greater',
            n_iter=1000,
            eids_to_process=pd.read_csv('eids4parallel.csv').iloc[:, 0].to_list()
            )
        results.to_parquet(out_dir / fname)

    fig_dir = PROJECT_ROOT / 'figures/baseline'
    fig_dir.mkdir(parents=True, exist_ok=True)
    propsig_fig = plot_baseline_propsig(results)
    propsig_fig.savefig(fig_dir / f'{args.model}_propsig.svg',
                        dpi=FIGURE_DPI, bbox_inches='tight')
    r2_fig = plot_baseline_r2(results)
    r2_fig.savefig(fig_dir / f'{args.model}_r2.svg',
                   dpi=FIGURE_DPI, bbox_inches='tight')
