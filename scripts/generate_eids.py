# %%
from pathlib import Path
import pandas as pd

from iblnm.config import SESSIONS_FPATH
from iblnm.data import PhotometrySessionGroup

from deploy.iblsdsc import OneSdsc

# requires the unmerged PR #121 https://github.com/int-brain-lab/iblscripts/pull/121
one = OneSdsc(location="popeye")

group = PhotometrySessionGroup.from_catalog(
    pd.read_parquet(SESSIONS_FPATH),
    one=one,
)

# %%
eids = list(group.sessions["eid"])
out_path = Path(__file__).parent / "eids_for_permutation.txt"
out_path.write_text("\n".join(eids) + "\n", encoding="utf-8")
