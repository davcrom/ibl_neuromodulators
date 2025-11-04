import pandas as pd
from tqdm import tqdm
# ~from pymer4.models import Lmer


from one.api import ONE
from one.alf.exceptions import ALFObjectNotFound

from iblnm.data import PhotometrySession

one = ONE()

sessions_file = 'sessions_dopamine.pqt'
df_sessions = pd.read_parquet(sessions_file)

df = df_sessions[df_sessions['subject'].apply(lambda x: int(x.split('-')[1]) > 8000)]
# ~df = df.query('session_type == "biased"')
dfs = []
for idx, session in tqdm(df.iterrows(), total=len(df)):
    # ~if session['session_type'] != 'biased':
        # ~continue
    try:
        ps = PhotometrySession(session, one=one)
    except ALFObjectNotFound:
        continue
    try:
        ps.trials['contrast'] = ps.trials['contrastRight'].fillna(ps.trials['contrastLeft'])
        ps.get_responses('gcamp', 'stimOn', pre=0, post=0.5, split_by='contrast')
        for target in ps.targets['gcamp']:
            for contrast in ps.trials['contrast'].unique():
                c = contrast * 100
                responses = session.copy()
                responses['target'] = target
                baseline = ps.responses['stimOn'][target]['gcamp'][contrast][0]
                resp_norm = ps.responses['stimOn'][target]['gcamp'][contrast] - baseline
                responses[f'stimOn_{c:.1f}'] = resp_norm.mean(axis=0)
                dfs.append(responses.to_frame().T.explode(f'stimOn_{c:.1f}').reset_index(drop=True))
    except:
        continue
df_responses = pd.concat(dfs)
df_responses.to_parquet('dopamine_responses_biased.pqt')

df_responses = pd.read_parquet('dopamine_responses_biased.pqt')

# Identify the stimOn columns
stimon_cols = [col for col in df_responses.columns if 'stimOn' in col]

# Melt the dataframe
df = df_responses.melt(
    id_vars=['eid', 'subject', 'target', 'session_type'],
    value_vars=stimon_cols,
    var_name='stimOn_col',
    value_name='response'
).dropna(subset='response')

# Extract the contrast value from the column name
df['contrast'] = df['stimOn_col'].str.replace('stimOn_', '').astype(float)
df['target'] = df['target'].astype('category')
df['session_type'] = df['session_type'].astype('category')
df['response_z'] = df.groupby('eid')['response'].transform(lambda x: (x - x.mean()) / x.std())

model = Lmer(
    'response_z ~ contrast + (1|subject)', data=df
    )
model.fit()


print(model.ranef_var)
print(model.coef)
