import numpy as np
import pandas as pd

from iblnm.config import SESSIONS_FPATH
from iblnm.io import _get_default_connection
from iblnm.data import PhotometrySessionGroup

def prepare_session(ps):
    assert len(ps.brain_region) == 1
    ps.load_h5(groups=['trials', 'photometry'])
    ps.trials['rt'] = ps.trials['feedback_times'] - ps.trials['stimOn_times']
    ps.trials['log_rt'] = ps.trials['rt'].apply(lambda x: np.log(x) if x > 0 else np.nan)
    ps.relative_log_rt = ps.trials.groupby('contrast')['log_rt'].transform(lambda x: (x - x.mean()) / x.std()).to_numpy()
    responses = ps.extract_responses(events=['stimOn_times'], window=[-0.4, -0.1])
    ps.baseline = responses[ps.brain_region[0]].sel(event='stimOn_times').mean(axis=1).to_numpy()
    return ps

def my_model(rt, baseline):
    from scipy.stats import linregress
    mask = (~np.isnan(rt) & ~np.isnan(baseline))
    res = linregress(rt[mask], baseline[mask])
    return res

def _model_slope(rt, baseline):
    res = my_model(rt, baseline)
    return res.slope

def _model_rsquared(rt, baseline):
    res = my_model(rt, baseline)
    return res.rvalue ** 2


if __name__ == '__main__':

    group = PhotometrySessionGroup.from_catalog(
        pd.read_parquet(SESSIONS_FPATH),
        one=_get_default_connection()
        )
    group.filter_sessions(
        session_types=('biased','ephys')
    )
    _ = group.deduplicate()

    results_rsquared = group.session_permutation_test(
        prepare_session,
        _model_rsquared,
        fixed_var=['relative_log_rt'],
        swapped_var=['baseline'],
        alternative='two-sided',
        n_iter=1000
        )

    results_rsquared.to_parquet('rsquared_test.pqt')

    #results_slope = group.process(_model_slope, workers=1)
