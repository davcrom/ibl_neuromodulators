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
from tqdm import tqdm

from iblnm.config import (SESSIONS_FPATH, SESSION_SCHEMA, PROJECT_ROOT,
                          FIGURE_DPI, STIM_ONSET_EVENT)
from iblnm.io import _get_default_connection
from iblnm.util import enforce_schema
from iblnm.data import PhotometrySession, PhotometrySessionGroup
from iblnm import analysis
from iblnm.vis import (plot_baseline_propsig, plot_baseline_r2,
                       plot_baseline_slope, plot_baseline_schematic,
                       plot_baseline_tercile_curves,
                       plot_baseline_tercile_difference)

from iblphotometry import processing

# VTA-DA recording used for the schematic intro figure. Dominant-direction and
# significant in both models (performance slope < 0, RT slope > 0; both p < 0.05)
# with a flat, non-drifting z-scored baseline across trials. Hardcoded; not
# derived in code.
EXAMPLE_EID = '26d93d1d-97f1-40f0-b84c-28229135f6fa'

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
    # load_trials only fetches, so the stored table is read off the H5 instead;
    # load_photometry then reads the preprocessed signal or rebuilds it.
    ps.load_h5(groups=['trials'])
    ps.load_photometry()
    ps.trials = ps.trials[
        (ps.trials['choice'] != 0)
        & ((ps.trials['firstMovement_times']
            - ps.trials[STIM_ONSET_EVENT]) >= 0.05)
        ].copy()
    ps.trials['rt'] = ps.trials['feedback_times'] - ps.trials[STIM_ONSET_EVENT]
    ps.trials['log_rt'] = ps.trials['rt'].apply(lambda x: np.log(x) if x > 0 else np.nan)
    ps.correct = ps.trials['feedbackType'].apply(lambda x: 1 if x > 0 else 0).to_numpy()
    ps.log_rt = ps.trials['log_rt'].to_numpy()
    ps.contrast = ps.trials['contrast'].to_numpy()
    # Signed contrast (percent, negative = left) for the tercile curves. Read
    # straight from the H5 trials frame (the loader stores it precomputed;
    # contrastLeft/contrastRight are not persisted) so it stays aligned with
    # ps.baseline and the per-trial outcome selectors after the trial filter.
    ps.signed_contrast = ps.trials['signed_contrast'].to_numpy()
    responses = ps.extract_responses(
        ps.photometry['GCaMP_preprocessed'],
        events=[STIM_ONSET_EVENT], window=[-0.4, -0.1],
    )
    baseline = responses[ps.brain_region[0]].sel(
        event=STIM_ONSET_EVENT).mean(axis=1).to_numpy()
    # z-score within session so the slope is comparable across recordings and
    # the donor-swap null injects no cross-session scale differences
    ps.baseline = (baseline - np.nanmean(baseline)) / np.nanstd(baseline)
    return ps

def linear_regression(outcome, contrast, baseline):
    """OLS fit of a behavioral outcome on baseline, controlling for contrast.

    Serves both models: continuous ``log_rt`` and the binary ``correct`` as a
    linear probability model (LPM). OLS on a 0/1 outcome is used deliberately —
    the permutation null supplies inference, so the logit link buys nothing,
    and OLS cannot fail to converge under the perfect separation a ceiling
    (all-correct) contrast level induces in a logistic fit.

    Contrast enters as a categorical covariate ``C(contrast)`` so the baseline
    term measures deviation in the outcome beyond what the trial's visual
    contrast predicts. ``r2`` is baseline's incremental R^2 over the
    contrast-only model (full minus contrast-only), isolating baseline's unique
    contribution.
    """
    data = pd.DataFrame(
        {'outcome': outcome, 'contrast': contrast, 'baseline': baseline}
        ).dropna()
    reduced = smf.ols('outcome ~ C(contrast)', data=data).fit()
    full = smf.ols('outcome ~ C(contrast) + baseline', data=data).fit()
    return {'slope': full.params['baseline'], 'r2': full.rsquared - reduced.rsquared}


# Each model is one independent permutation analysis; run them as separate
# processes (one per --model) to parallelize. Maps the selector to its
# (statistic function, fixed regressor, output filename).
MODELS = {
    'performance': (linear_regression, ['correct', 'contrast'], 'performance.pqt'),
    'reaction_time': (linear_regression, ['log_rt', 'contrast'], 'reaction_time.pqt'),
}

# Per-model per-trial outcome for the tercile behavioral curves (analysis choice,
# so it lives in the script). ``performance`` -> rightward-choice indicator
# (IBL choice == -1 = chose right); ``reaction_time`` -> log reaction time.
TERCILE_OUTCOME = {
    'performance': lambda ps: (ps.trials['choice'] == -1).to_numpy(),
    'reaction_time': lambda ps: ps.log_rt,
}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS, required=True,
                        help='which baseline analysis to run')
    parser.add_argument('--reprocess', action='store_true',
                        help='rerun the ~10h permutation test and overwrite the '
                             'saved parquet; default re-plots from the saved '
                             'parquet')
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_args()
    model_fn, fixed_var, fname = MODELS[args.model]
    out_dir = Path('results/baseline')

    one = _get_default_connection()
    group = PhotometrySessionGroup.from_catalog(
        pd.read_parquet(SESSIONS_FPATH), one=one)
    group.filter_sessions(
        session_types=('biased', 'ephys',)
    )
    _ = group.deduplicate()

    if args.reprocess:
        out_dir.mkdir(parents=True, exist_ok=True)
        results = group.session_permutation_test(
            prepare_session,
            model_fn,
            fixed_var=fixed_var,
            swapped_var=['baseline'],
            statistic_key='r2',
            alternative='greater',
            n_iter=1000,
            # ~ eids_to_process=pd.read_csv('eids4parallel.csv').iloc[:10, 0].to_list()
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

    # Tercile behavioral curves: reload the significant recordings through the
    # group object, split each session's behavior into low/high baseline
    # terciles per signed-contrast level, and aggregate across sessions by target.
    #
    # Match significance at the recording level (eid, brain_region), not by eid:
    # a catalog session may list filled/fixed brain regions absent from its H5,
    # and multi-region sessions have only some recordings significant. Build a
    # fresh PhotometrySession per recording (as session_permutation_test does) so
    # the brain region comes from the catalog recordings row, not a stale
    # per-eid cache -- ps.brain_region[0] then keys the correct H5 response.
    sig = results[results['p_value'] <= 0.05]
    sig_recordings = set(zip(sig['eid'], sig['brain_region']))
    outcome_selector = TERCILE_OUTCOME[args.model]
    recordings = group.recordings[
        [(e, r) in sig_recordings
         for e, r in zip(group.recordings['eid'], group.recordings['brain_region'])]
    ]
    curves_by_target = {}
    for _, rec in tqdm(recordings.iterrows(), total=len(recordings),
                       desc='Tercile curves'):
        ps = prepare_session(PhotometrySession(rec, one=one))
        curve = analysis.tercile_split_curves(
            ps.baseline, ps.signed_contrast, outcome_selector(ps), min_count=5)
        curves_by_target.setdefault(rec['target_NM'], []).append(curve)
    tercile_fig = plot_baseline_tercile_curves(curves_by_target, model=args.model)
    tercile_fig.savefig(fig_dir / f'{args.model}_tercile.svg',
                        dpi=FIGURE_DPI, bbox_inches='tight')
    diff_fig = plot_baseline_tercile_difference(curves_by_target, model=args.model)
    diff_fig.savefig(fig_dir / f'{args.model}_tercile_diff.svg',
                     dpi=FIGURE_DPI, bbox_inches='tight')

    # Schematic intro figure: recompute the example session from the ONE cache
    # (the H5 preprocessing differs from PIPELINE) and draw its modelled traces,
    # scatter, and the permutation cartoon. fixed_var[0] names the behavior
    # attribute prepare_session sets ('correct' or 'log_rt').
    sessions = enforce_schema(pd.read_parquet(SESSIONS_FPATH), SESSION_SCHEMA)
    row = sessions[sessions['eid'] == EXAMPLE_EID].iloc[0]
    ps = prepare_session(PhotometrySession(row, one=one))
    schematic_fig = plot_baseline_schematic(
        ps.baseline, getattr(ps, fixed_var[0]), args.model)
    schematic_fig.savefig(fig_dir / f'{args.model}_schematic.svg',
                          dpi=FIGURE_DPI, bbox_inches='tight')
