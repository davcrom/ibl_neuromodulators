import numpy as np

def get_responses(F, ts, t0=0.1, t1=1):
    tpts = F.index.values
    ts = ts[(ts - t0 > tpts.min()) & (ts + t1 < tpts.max())]
    inds = tpts.searchsorted(ts)
    dt = np.median(np.diff(tpts))
    i0s = inds - int(t0 // dt)
    i1s = inds + int(t1 // dt)
    responses = np.stack([F.values[i0:i1] for i0, i1 in zip(i0s, i1s)])
    tpts_resp = np.linspace(-1 * t0, t1, responses.shape[1])
    responses = (responses.T - responses[:, tpts_resp.searchsorted(0)]).T
    return responses, tpts_resp