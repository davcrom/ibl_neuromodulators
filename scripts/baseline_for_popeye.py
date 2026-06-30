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

from iblnm.config import SESSIONS_FPATH
from iblnm.data import PhotometrySessionGroup

from iblnm.config import PREPROCESSING_PIPELINES

from deploy.iblsdsc import OneSdsc

from uuid import UUID


def is_uuid(s):
    try:
        UUID(s)
    except (ValueError, AttributeError, TypeError):
        return False
    return True


# requires the unmerged PR #121 https://github.com/int-brain-lab/iblscripts/pull/121
one = OneSdsc(location="popeye")


def prepare_session(ps):
    assert len(ps.brain_region) == 1
    # replacing the loading from h5 by
    # ps.load_h5(groups=["trials", "photometry"])
    ps.load_photometry()

    # loading trials
    # ps.trials = ps.trials[
    #     (ps.trials["choice"] != 0)
    #     & ((ps.trials["firstMovement_times"] - ps.trials["stimOn_times"]) >= 0.05)
    # ].copy()
    ps.load_trials()

    pipeline = PREPROCESSING_PIPELINES["isosbestic_correction"]

    #
    pipeline[0]["parameters"] = dict(
        correction_method="subtract-divide",
        N=3,
        Wn=0.001,
    )
    pipeline[1]["parameters"] = dict(
        correction_method="subtract-divide",
        N=3,
        Wn=0.001,
    )

    ps.preprocess(pipeline=pipeline)

    ps.trials["rt"] = ps.trials["feedback_times"] - ps.trials["stimOn_times"]
    ps.trials["log_rt"] = ps.trials["rt"].apply(
        lambda x: np.log(x) if x > 0 else np.nan
    )
    ps.correct = ps.trials["feedbackType"].apply(lambda x: 1 if x > 0 else 0).to_numpy()
    ps.relative_log_rt = (
        ps.trials.groupby("contrast")["log_rt"]
        .transform(lambda x: (x - x.mean()) / x.std())
        .to_numpy()
    )
    responses = ps.extract_responses(events=["stimOn_times"], window=[-0.4, -0.1])
    baseline = (
        responses[ps.brain_region[0]].sel(event="stimOn_times").mean(axis=1).to_numpy()
    )
    # z-score within session so the slope is comparable across recordings and
    # the donor-swap null injects no cross-session scale differences
    ps.baseline = (baseline - np.nanmean(baseline)) / np.nanstd(baseline)
    return ps


def linear_regression(rt, baseline):
    from scipy.stats import linregress

    mask = ~np.isnan(rt) & ~np.isnan(baseline)
    res = linregress(rt[mask], baseline[mask])
    return {"slope": res.slope, "r2": res.rvalue**2}


def logistic_regression(correct, baseline):
    from statsmodels.formula.api import logit

    formula = "correct ~ baseline"
    data = pd.DataFrame(
        data=np.column_stack([correct, baseline]), columns=["correct", "baseline"]
    ).dropna()
    res = logit(formula, data=data).fit(disp=0)
    return {"slope": res.params["baseline"], "r2": res.prsquared}


# Each model is one independent permutation analysis; run them as separate
# processes (one per --model) to parallelize. Maps the selector to its
# (statistic function, fixed regressor, output filename).
MODELS = {
    "performance": (logistic_regression, ["correct"], "performance.pqt"),
    "reaction_time": (linear_regression, ["relative_log_rt"], "reaction_time.pqt"),
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=MODELS,
        required=True,
        help="which baseline analysis to run",
    )
    # for per-session paralellization
    # adding an eid and building an exclusion set of all other eids
    parser.add_argument(
        "--eid",
        required=False,
        help="subsetting to eid",
    )
    # add exclude eid argument here

    args = parser.parse_args()
    model_fn, fixed_var, fname = MODELS[args.model]

    group = PhotometrySessionGroup.from_catalog(
        pd.read_parquet(SESSIONS_FPATH),
        one=one,
    )
    # group has a list of all valid sessions
    # we are exluding all sessions except the session of interest
    assert args.eid is not None
    eids = group.sessions["eid"].values
    eids = [str(eid) for eid in eids if is_uuid(eid)]

    exclude_eids = list({*eids} - {args.eid})

    group.filter_sessions(
        exclude_eids=exclude_eids,
        session_types=("ephys",),
    )
    _ = group.deduplicate()

    out_dir = Path(__file__).parent.parent / "results" / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = group.session_permutation_test(
        prepare_session,
        model_fn,
        fixed_var=fixed_var,
        swapped_var=["baseline"],
        statistic_key="r2",
        alternative="greater",
        n_iter=10,  # originally 1000
    )
    # TODO verify output directory
    results.to_parquet(out_dir / f"{args.eid}_{fname}")
