import numpy as np
import pandas as pd

from iblnm.config import *

def get_responses(photometry, events, window=RESPONSE_WINDOW):
    """Get peri-event signals in the given window."""
    times = photometry.index.to_numpy()
    values = photometry.values
    fs = int(1 / np.median(np.diff(times)))

    n_trials = len(events)
    samples = np.arange(int(window[0] * fs), int(window[1] * fs))
    responses = np.full((n_trials, len(samples)), np.nan)

    # Filter valid events
    valid_events = (events + window[0] >= times.min()) & (events + window[1] <= times.max())
    valid_event_times = events[valid_events]
    event_idx = np.searchsorted(times, valid_event_times)

    # Create response indices: (n_valid_events, n_samples)
    response_idx = event_idx[:, None] + samples[None, :]
    response_idx = np.clip(response_idx, 0, len(values) - 1)

    responses[valid_events] = values[response_idx]
    return responses


def get_response_tpts(photometry, window=RESPONSE_WINDOW):
    times = photometry.index.to_numpy()
    fs = int(1 / np.median(np.diff(times)))
    samples = np.arange(int(window[0] * fs), int(window[1] * fs))
    return samples / fs


def normalize_responses(responses, tpts, bwin=(-0.1, 0), divide=True):
    i0, i1 = tpts.searchsorted(bwin)
    bvals = responses[:, i0:i1].mean(axis=1)
    resp_norm = (responses.T - bvals).T
    if divide:
        resp_norm = (resp_norm.T / bvals).T
    return resp_norm
