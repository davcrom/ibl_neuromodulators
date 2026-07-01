"""Session-swap permutation tests of pre-stimulus baseline coding.

Tests whether a recording's within-session baseline fluctuations carry
information about behaviour. Each --model is one independent analysis; run them
as separate processes to parallelize.

Usage:
    python scripts/baseline.py --model performance              # plot from saved parquet
    python scripts/baseline.py --model performance --reprocess  # rerun ~10h test, then plot
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from iblnm.config import SESSIONS_FPATH, PROJECT_ROOT, FIGURE_DPI
from iblnm.io import _get_default_connection
from iblnm.data import PhotometrySessionGroup
from iblnm.vis import plot_baseline_propsig, plot_baseline_r2, plot_baseline_slope

from iblphotometry import processing

PIPELINE = [
        dict(
            function=processing.lowpass_bleachcorrect,
            parameters=dict(
                correction_method='subtract-divide',
                N=3,
                Wn=0.001,
            ),
            inputs=('signal',),
            output='signal_bleach_corrected',
        ),
        dict(
            function=processing.lowpass_bleachcorrect,
            parameters=dict(
                correction_method='subtract-divide',
                N=3,
                Wn=0.001,
            ),
            inputs=('reference',),
            output='reference_bleach_corrected',
        ),
        dict(
            function=processing.isosbestic_correct,
            parameters=dict(
                regression_method='mse',
                correction_method='subtract',
            ),
            inputs=('signal_bleach_corrected', 'reference_bleach_corrected'),
            output='result',
        ),
        dict(
            function=processing.zscore,
            parameters=dict(mode='classic'),
            inputs=('result',),
            output='result',
        ),
    ]

def prepare_session(ps):
    assert len(ps.brain_region) == 1
    ps.load_photometry()
    ps.preprocess(pipeline=PIPELINE)
    ps.load_trials()
    # ~ ps.load_h5(groups=['trials', 'photometry'])
    ps.trials = ps.trials[
        (ps.trials['choice'] != 0)
        & ((ps.trials['firstMovement_times'] - ps.trials['stimOn_times']) >= 0.05)
        ].copy()
    ps.trials['rt'] = ps.trials['feedback_times'] - ps.trials['stimOn_times']
    ps.trials['log_rt'] = ps.trials['rt'].apply(lambda x: np.log(x) if x > 0 else np.nan)
    ps.correct = ps.trials['feedbackType'].apply(lambda x: 1 if x > 0 else 0).to_numpy()
    ps.log_rt = ps.trials['log_rt'].to_numpy()
    ps.contrast = ps.trials['contrast'].to_numpy()
    responses = ps.extract_responses(events=['stimOn_times'], window=[-0.4, -0.1])
    baseline = responses[ps.brain_region[0]].sel(event='stimOn_times').mean(axis=1).to_numpy()
    # z-score within session so the slope is comparable across recordings and
    # the donor-swap null injects no cross-session scale differences
    ps.baseline = (baseline - np.nanmean(baseline)) / np.nanstd(baseline)
    return ps

def linear_regression(log_rt, contrast, baseline):
    """OLS fit of log RT on baseline, controlling for contrast.

    Contrast enters as a categorical covariate ``C(contrast)`` so the baseline
    term measures deviation in RT beyond what the trial's visual contrast
    predicts. ``r2`` is baseline's incremental R^2 over the contrast-only model
    (full minus contrast-only), isolating baseline's unique contribution.
    """
    data = pd.DataFrame(
        {'log_rt': log_rt, 'contrast': contrast, 'baseline': baseline}
        ).dropna()
    reduced = smf.ols('log_rt ~ C(contrast)', data=data).fit()
    full = smf.ols('log_rt ~ C(contrast) + baseline', data=data).fit()
    return {'slope': full.params['baseline'], 'r2': full.rsquared - reduced.rsquared}

def logistic_regression(correct, contrast, baseline):
    """Logistic fit of correctness on baseline, controlling for contrast.

    Contrast enters as a categorical covariate ``C(contrast)`` so the baseline
    term measures deviation in accuracy beyond what the trial's visual contrast
    predicts. ``r2`` is baseline's incremental McFadden pseudo-R^2 over the
    contrast-only model (full minus contrast-only), isolating baseline's unique
    contribution.
    """
    data = pd.DataFrame(
        {'correct': correct, 'contrast': contrast, 'baseline': baseline}
        ).dropna()
    reduced = smf.logit('correct ~ C(contrast)', data=data).fit(disp=0)
    full = smf.logit('correct ~ C(contrast) + baseline', data=data).fit(disp=0)
    return {'slope': full.params['baseline'], 'r2': full.prsquared - reduced.prsquared}


# Each model is one independent permutation analysis; run them as separate
# processes (one per --model) to parallelize. Maps the selector to its
# (statistic function, fixed regressor, output filename).
MODELS = {
    'performance': (logistic_regression, ['correct', 'contrast'], 'performance.pqt'),
    'reaction_time': (linear_regression, ['log_rt', 'contrast'], 'reaction_time.pqt'),
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS, required=True,
                        help='which baseline analysis to run')
    parser.add_argument('--reprocess', action='store_true',
                        help='rerun the ~10h permutation test and overwrite the '
                             'saved parquet; default re-plots from the saved '
                             'parquet')
    args = parser.parse_args()
    model_fn, fixed_var, fname = MODELS[args.model]
    out_dir = Path('results/baseline')

    if args.reprocess:
        group = PhotometrySessionGroup.from_catalog(
            pd.read_parquet(SESSIONS_FPATH),
            one=_get_default_connection()
            )
        group.filter_sessions(
            session_types=('biased', 'ephys',)
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
            )
        results.to_parquet(out_dir / fname)
    else:
        results = pd.read_parquet(out_dir / fname)

    fig_dir = PROJECT_ROOT / 'figures/baseline'
    fig_dir.mkdir(parents=True, exist_ok=True)
    propsig_fig = plot_baseline_propsig(results)
    propsig_fig.savefig(fig_dir / f'{args.model}_propsig.svg',
                        dpi=FIGURE_DPI, bbox_inches='tight')
    r2_fig = plot_baseline_r2(results)
    r2_fig.savefig(fig_dir / f'{args.model}_r2.svg',
                   dpi=FIGURE_DPI, bbox_inches='tight')
    slope_fig = plot_baseline_slope(results)
    slope_fig.savefig(fig_dir / f'{args.model}_slope.svg',
                      dpi=FIGURE_DPI, bbox_inches='tight')
