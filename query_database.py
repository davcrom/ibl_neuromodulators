import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from one.api import ONE

from iblnm import io
from iblnm.config import *
from iblnm.util import protocol2type, save_timestamped_pqt

one = ONE()

# ~extended = False
# ~io.fetch_sessions(extended=extended)

# Get list of session dicts
print("Querying database...")
sessions = one.alyx.rest('sessions', 'list', project='ibl_fibrephotometry')

# Rename id column to eid
df_sessions = pd.DataFrame(sessions).rename(columns={'id': 'eid'})
# Add useful label for session types
df_sessions['session_type'] = df_sessions['task_protocol'].map(protocol2type)

# Remove mice that shouldn't be part of the analysis
df_sessions = df_sessions.query('subject not in @EXCLUDE_SUBJECTS')

# Add genotype info and tries to infer the GCaMP-expressing NM cell type
print("Adding subject info...")
df_sessions = df_sessions.progress_apply(io.get_subject_info, axis='columns').copy()

print("Checking datasets...")
df_sessions = df_sessions.progress_apply(io.check_datasets, one=one, axis='columns').copy()

# ~print("Unpacking session dicts...")
# ~df_sessions = df_sessions.progress_apply(io.unpack_session_dict, one=one, axis='columns').copy()

# Save as parquet with a timestamp
save_timestamped_pqt(df_sessions, SESSIONS_FPATH)
