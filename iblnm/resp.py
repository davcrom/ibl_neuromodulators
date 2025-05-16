import numpy as np

def get_responses(A, events, window=(0, 1)):
    # signal = A.values.squeeze()
    signal = A.values if isinstance(A, pd.Series) else A
    assert signal.ndim == 1
    tpts = A.index.values
    dt = np.median(np.diff(tpts))
    events = events[events + window[1] < tpts.max()]
    event_inds = tpts.searchsorted(events)
    i0s = event_inds - int(window[0] / dt)
    i1s = event_inds + int(window[1] / dt)
    responses = np.vstack([signal[i0:i1] for i0, i1 in zip(i0s, i1s)])
    responses = (responses.T - signal[event_inds]).T
    tpts = np.arange(window[0], window[1] - dt, dt)
    return responses, tpts