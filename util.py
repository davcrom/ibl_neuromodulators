import numpy as np
import pandas as pd

def load_recinfo(fpath='recordings.csv'):
    df = pd.read_csv(fpath)
    # Convert acronym strings into lists of strings
    df['acronyms'] = df['_acronyms'].apply(eval)
    # Add additional metadata
    df_insertions = pd.read_csv('insertions.csv')
    def _merge_metadata(row, df=df_insertions):
        subj = df_insertions[df_insertions['subject'] ==  row['subject']]
        for col in [v for v in subj.columns if v != 'subject']:
            row[col] = subj[col].values
        return row
    df = df.apply(_merge_metadata, df=df_insertions, axis='columns')   
    return df

def get_responses(events, data, tpts, pre=0, post=1, baseline=[]):
    dt = np.round(np.diff(tpts).mean(), 3)  # round to nearest ms
    i_pre, i_post = int(pre / dt), int(post / dt)
    responses = np.full((len(events), i_pre + i_post), np.nan)
    for j, t0 in enumerate(events):
        i = tpts.searchsorted(t0)
        i0, i1 = i - i_pre, i + i_post
        if i0 < 0:
            continue 
        if i1 > len(data):
            break
        responses[j] = data[i0:i1]
    tpts = np.linspace(-pre, post, responses.shape[1])
    if baseline:
        b0 = tpts.searchsorted(baseline[0])
        # Ensure baseline is >= 1 datapoint
        b1 = max(b0 + 1, tpts.searchsorted(baseline[1]))
        baseline_resp = responses[:, b0:b1].mean(axis=1)
        responses = (responses.T - baseline_resp).T
    return responses, tpts