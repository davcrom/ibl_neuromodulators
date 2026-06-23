import numpy as np
import pandas as pd

from iblnm.config import SESSIONS_FPATH
from iblnm.io import _get_default_connection
from iblnm.data import PhotometrySessionGroup

def prepare_session(ps):
    assert len(ps.brain_region) == 1
    ps.load_h5(groups=['trials', 'photometry'])
    ps.rt = (ps.trials['firstMovement_times'] - ps.trials['stimOn_times']).to_numpy()
    responses = ps.extract_responses(events=['stimOn_times'], window=[-0.4, -0.1])
    ps.baseline = responses[ps.brain_region[0]].sel(event='stimOn_times').mean(axis=1).to_numpy()
    return ps

def my_model(rt, baseline):
    from scipy.stats import linregress
    mask = (~np.isnan(rt) & ~np.isnan(baseline))
    res = linregress(rt[mask], baseline[mask])
    return res.slope


if __name__ == '__main__':

    group = PhotometrySessionGroup.from_catalog(
        pd.read_parquet(SESSIONS_FPATH),
        one=_get_default_connection()
        )
    group.filter_sessions(
        session_types=('ephys',),
        min_performance={'ephys': 0.8}
    )
    _ = group.deduplicate()

    results = group.session_permutation_test(
        prepare_session,
        my_model,
        fixed_var=['rt'],
        swapped_var=['baseline'],
        alternative='greater',
        n_iter=3
        )
