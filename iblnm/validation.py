"""Custom exceptions and validate_* functions for IBL photometry sessions."""
import traceback as tb_module
from functools import wraps, lru_cache

import numpy as np
import pandas as pd

from iblnm.config import (
    VALID_STRAINS, VALID_LINES, VALID_NEUROMODULATORS, VALID_TARGETS,
    DATASET_CATEGORIES,
    SUBJECTS_TO_EXCLUDE, FIBERS_FPATH,
    LENGTH_MISMATCH_THRESHOLD,
)


# =============================================================================
# Exceptions — metadata / session-level
# =============================================================================

class InvalidSubject(Exception):
    """Mouse does not belong to this project."""

class InvalidStrain(Exception):
    """Mouse strain is not recognized."""

class InvalidLine(Exception):
    """Mouse line is not recognized."""

class InvalidNeuromodulator(Exception):
    """Neuromodulator could not be determined."""

class InvalidBrainRegion(Exception):
    """Brain region not recognized."""

class MissingBrainRegion(Exception):
    """Session has no brain region metadata."""

class MissingHemisphere(Exception):
    """Session has no hemisphere metadata."""

class HemisphereMismatch(Exception):
    """Region name and fiber coordinates disagree on hemisphere."""

class MissingInsertion(Exception):
    """Fiber coordinates for subject not found in lookup table."""

class MissingHemiSuffix(Exception):
    """Brain region did not include a hemisphere suffix."""

class DataNotListed(Exception):
    """Dataset not found in one.list_datasets."""

class InvalidSessionType(Exception):
    """Session type not suitable for analysis."""

class InvalidTargetNM(Exception):
    """Brain region does not map to a valid target-NM combination."""

class InvalidSessionLength(Exception):
    """Session start and end times are on different days"""

class TrueDuplicateSession(Exception):
    """Two or more sessions on the same day pass all quality criteria.

    Carries fallback_row: the best row to return when the exception is caught.
    """
    def __init__(self, msg, fallback_row=None):
        super().__init__(msg)
        self.fallback_row = fallback_row


# =============================================================================
# Exceptions — data loading / photometry
# =============================================================================

class MissingExtractedData(Exception):
    """Extracted dataset not found on Alyx (raw data exists)."""

class MissingResponses(Exception):
    """Pre-computed peri-event responses not found in H5 file."""

class MissingRawData(Exception):
    """Raw dataset not found on Alyx."""

class InsufficientTrials(Exception):
    """Session has too few trials for analysis."""

class BlockStructureBug(Exception):
    """Biased/ephys session has rapidly flipping blocks."""

class IncompleteEventTimes(Exception):
    """Event times below completeness threshold."""
    def __init__(self, missing_events):
        self.missing_events = missing_events
        super().__init__(f"Incomplete events: {', '.join(missing_events)}")

class TrialsNotInPhotometryTime(Exception):
    """Trial times fall outside photometry recording window."""

class BandInversion(Exception):
    """Photometry signal has band inversions."""

class EarlySamples(Exception):
    """Photometry signal has early samples."""

class FewUniqueSamples(Exception):
    """One or more photometry channels have too few unique samples."""

class QCValidationError(Exception):
    """One or more raw QC checks failed (band inversions, early samples)."""

class AmbiguousRegionMapping(Exception):
    """Photometry columns cannot be unambiguously mapped to session brain_region metadata."""


# =============================================================================
# Exceptions — video QC / data
# =============================================================================

class MissingVideoTimestamps(Exception):
    """leftCamera.times could not be loaded."""

class VideoLengthError(Exception):
    """Video length differs from session length beyond threshold."""

class VideoTimestampsQCError(Exception):
    """qc_videoLeft_timestamps is not PASS."""

class VideoDroppedFramesQCError(Exception):
    """qc_videoLeft_dropped_frames is not PASS."""

class VideoPinStateQCError(Exception):
    """qc_videoLeft_pin_state is not PASS."""


# =============================================================================
# Logging helpers
# =============================================================================

def make_log_entry(eid, error=None, error_type=None, error_message=None):
    """Create a standardized log entry.

    Provide either an exception via `error`, or explicit `error_type`/`error_message`.
    When `error` is given, type/message/traceback are extracted from it.
    """
    if error is not None:
        return {
            'eid': eid,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': tb_module.format_exc(),
        }
    if error_type is not None:
        return {
            'eid': eid,
            'error_type': error_type,
            'error_message': error_message,
            'traceback': None,
        }
    raise ValueError("Provide either error or error_type")


def exception_logger(func):
    """
    Decorator that allows session processing functions to log exceptions.
    Use exlog parameter to capture errors instead of raising them.

    Works with both pd.Series (single row, e.g. from df.apply) and pd.DataFrame
    (group, e.g. from groupby.apply). For DataFrames the first eid is used for
    the log entry and the first row is returned on error.
    """
    @wraps(func)
    def wrapper(series, *args, exlog=None, **kwargs):
        try:
            return func(series, *args, **kwargs)
        except Exception as e:
            if exlog is not None:
                if isinstance(series, pd.DataFrame):
                    eid = series['eid'].iloc[0] if len(series) > 0 else 'unknown'
                    exlog.append(make_log_entry(eid, error=e))
                    fallback = getattr(e, 'fallback_row', None)
                    return fallback if fallback is not None else series.iloc[0]
                else:
                    exlog.append(make_log_entry(series.get('eid', 'unknown'), error=e))
                    return series
            else:
                raise
    return wrapper


# =============================================================================
# Validate functions
# =============================================================================

@exception_logger
def validate_subject(session):
    subject = session['subject']
    if subject in SUBJECTS_TO_EXCLUDE:
        raise InvalidSubject(f"Subject {subject} in {SUBJECTS_TO_EXCLUDE}")
    return None


@exception_logger
def validate_strain(session):
    strain = session['strain']
    if strain not in VALID_STRAINS:
        raise InvalidStrain(f"Strain {strain} not in {VALID_STRAINS}")
    return None


@exception_logger
def validate_line(session):
    line = session['line']
    if line not in VALID_LINES:
        raise InvalidLine(f"Line {line} not in {VALID_LINES}")
    return None


@exception_logger
def validate_neuromodulator(session):
    nm = session['NM']
    if nm not in VALID_NEUROMODULATORS:
        raise InvalidNeuromodulator(f"NM {nm} not in {VALID_NEUROMODULATORS}")
    return None


@exception_logger
def validate_brain_region(session):
    regions = session['brain_region']
    if not isinstance(regions, (list, np.ndarray)) or len(regions) == 0:
        raise MissingBrainRegion(
            f"{session.get('subject', '')} {session.get('eid', '')}: "
            f"empty brain_region"
        )
    for region in regions:
        bare = region.rsplit('-', 1)[0] if region.endswith(('-l', '-r')) else region
        if bare not in VALID_TARGETS:
            raise InvalidBrainRegion(f"Region {region} not in {VALID_TARGETS}")
    return None


@lru_cache(maxsize=1)
def _get_fiber_hemisphere_lookup():
    """Build subject+region -> hemisphere lookup from fiber coordinates.

    Returns None for a (subject, region) pair when fibers span both hemispheres.
    """
    df_fibers = pd.read_csv(FIBERS_FPATH)
    df_fibers = df_fibers.copy()
    df_fibers['hemi'] = df_fibers['X-ml_um'].apply(
        lambda x: 'l' if x > 0 else 'r'
    )
    grouped = df_fibers.groupby(['subject', 'targeted_region'])['hemi']
    return {key: vals.iloc[0] if vals.nunique() == 1 else None
            for key, vals in grouped}


@exception_logger
def fill_hemisphere_from_fiber_insertion_table(session, fiber_lookup=None):
    """Fill None hemisphere entries from the fiber insertion coordinates lookup.

    For each brain region where hemisphere is None, the bare region name is
    looked up in fiber_lookup. If a unique hemisphere is found it is filled in;
    otherwise the entry remains None and validate_hemisphere will flag it.

    Parameters
    ----------
    session : pd.Series
        Must have 'subject', 'brain_region' (list[str]), 'hemisphere' (list[str|None]).
    fiber_lookup : dict, optional
        {(subject, bare_region): 'l' | 'r' | None}. Defaults to
        _get_fiber_hemisphere_lookup().

    Returns
    -------
    pd.Series
        Session with updated hemisphere list.
    """
    if fiber_lookup is None:
        fiber_lookup = _get_fiber_hemisphere_lookup()
    subject = session['subject']
    hemi = list(session['hemisphere'])
    for i, (region, h) in enumerate(zip(session['brain_region'], hemi)):
        if h is not None:
            continue
        bare = region.rsplit('-', 1)[0] if region.endswith(('-l', '-r')) else region
        hemi[i] = fiber_lookup.get((subject, bare))
    session = session.copy()
    session['hemisphere'] = hemi
    return session


@exception_logger
def validate_hemisphere(session, fiber_lookup=None):
    if fiber_lookup is None:
        fiber_lookup = _get_fiber_hemisphere_lookup()
    hemis = session['hemisphere']
    if not isinstance(hemis, (list, np.ndarray)) or len(hemis) == 0:
        raise MissingHemisphere(
            f"{session.get('subject', '')} {session.get('eid', '')}: "
            f"empty hemisphere"
        )
    subject = session['subject']
    for region, hemi_name in zip(session['brain_region'], session['hemisphere']):
        bare = region.rsplit('-', 1)[0] if region.endswith(('-l', '-r')) else region
        hemi_fiber = fiber_lookup.get((subject, bare))
        if hemi_name is not None and hemi_fiber is not None and hemi_name != hemi_fiber:
            raise HemisphereMismatch(
                f"{subject} {region}: name={hemi_name}, coordinate={hemi_fiber}"
            )
        elif hemi_fiber is None:
            raise MissingInsertion(
                f"{subject} {region} missing fiber insertion entry"
            )
        elif hemi_name is None:
            raise MissingHemiSuffix(
                f"{subject} {region} missing hemisphere suffix"
            )
    return None


@exception_logger
def validate_datasets(session):
    datasets = session.get('datasets', [])
    def _has(d):
        return isinstance(datasets, (list, np.ndarray)) and d in datasets
    missing = [
        cat for cat in DATASET_CATEGORIES
        if not any(_has(d) for d in DATASET_CATEGORIES.get(cat, []))
    ]
    if missing:
        raise DataNotListed(f"Missing dataset categories: {', '.join(missing)}")
    return None


# =============================================================================
# Video validate functions
# =============================================================================

@exception_logger
def validate_video_length(session):
    """Raise VideoLengthError if video–session length discrepancy exceeds threshold."""
    discrepancy = session.get('length_discrepancy', np.nan)
    if not np.isnan(discrepancy) and discrepancy >= LENGTH_MISMATCH_THRESHOLD:
        raise VideoLengthError(
            f"Video–session length discrepancy {discrepancy:.0f}s "
            f"exceeds {LENGTH_MISMATCH_THRESHOLD}s threshold"
        )
    return None


@exception_logger
def validate_video_timestamps_qc(session):
    """Raise VideoTimestampsQCError if qc_videoLeft_timestamps is not PASS."""
    val = session.get('qc_videoLeft_timestamps')
    if val != 'PASS':
        raise VideoTimestampsQCError(
            f"qc_videoLeft_timestamps is {val!r}, expected PASS"
        )
    return None


@exception_logger
def validate_video_dropped_frames_qc(session):
    """Raise VideoDroppedFramesQCError if qc_videoLeft_dropped_frames is not PASS."""
    val = session.get('qc_videoLeft_dropped_frames')
    if val != 'PASS':
        raise VideoDroppedFramesQCError(
            f"qc_videoLeft_dropped_frames is {val!r}, expected PASS"
        )
    return None


@exception_logger
def validate_video_pin_state_qc(session):
    """Raise VideoPinStateQCError if qc_videoLeft_pin_state is not PASS."""
    val = session.get('qc_videoLeft_pin_state')
    if val != 'PASS':
        raise VideoPinStateQCError(
            f"qc_videoLeft_pin_state is {val!r}, expected PASS"
        )
    return None
