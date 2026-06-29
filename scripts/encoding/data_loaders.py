"""Data loaders for the photometry encoding model.

Each loader fetches one modality for a session and returns it as a pynapple
object (or a trials DataFrame). `load_session_data` returns them together for
the `encoding_model` module to place on its time grid.
"""
import pandas as pd
import pynapple as nap

from one.api import ONE
from iblphotometry.fpio import PhotometrySessionLoader
from iblphotometry.pipelines import sliding_mad_pipeline, run_pipeline
from brainbox.behavior import wheel as wheel_methods

def get_brain_regions(eid: str, one: ONE):
    psl = PhotometrySessionLoader(one=one, eid=eid)
    psl.load_photometry()
    return psl.photometry['GCaMP'].columns

def load_fluorescence(
    psl: PhotometrySessionLoader, brain_region: str, band: str = "GCaMP"
) -> nap.Tsd:
    """Load and preprocess the photometry signal for one brain region.

    Args:
        psl (PhotometrySessionLoader): loader for the target session.
        brain_region (str): region/channel to extract, e.g. "SNc-l".
        band (str): signal band column group. Defaults to "GCaMP".

    Returns:
        nap.Tsd: bleach-corrected fluorescence on its native time base.
    """
    psl.load_photometry()
    signal = run_pipeline(sliding_mad_pipeline, psl.photometry[band][brain_region])
    return nap.Tsd(t=signal.index, d=signal.values)


def load_trials(psl: PhotometrySessionLoader) -> pd.DataFrame:
    """Load the trials table and add a side-invariant contrast column.

    Args:
        psl (PhotometrySessionLoader): loader for the target session.

    Returns:
        pd.DataFrame: trials with an added "contrast" column (left or right).
    """
    psl.load_trials()
    trials = psl.trials
    # merge left/right contrast into a single side-invariant column
    trials["contrast"] = trials["contrastLeft"].fillna(trials["contrastRight"])
    # signed contrast: left is negative
    trials['signed_contrast'] = trials["contrastRight"].fillna(-1*trials["contrastLeft"])
    return trials


def load_pose(one: ONE, eid: str, camera: str = "leftCamera") -> nap.TsdFrame:
    """Load Lightning Pose tracking, dropping likelihood columns.

    Args:
        one (ONE): an open ONE connection.
        eid (str): experiment id.
        camera (str): camera label. Defaults to "leftCamera".

    Returns:
        nap.TsdFrame: pose traces indexed by camera timestamps.
    """
    camera_data = one.load_object(eid, obj=camera, collection="alf")
    pose = pd.DataFrame(camera_data["lightningPose"])
    # keep coordinate traces, drop per-point likelihoods
    # pose = pose[[col for col in pose.columns if not col.endswith("likelihood")]]
    pose = pose[[col for col in pose.columns if 'ens_median' in col]]
    pose.index = camera_data["times"]
    return nap.TsdFrame(pose)


def load_wheel(one: ONE, eid: str, frequency: float = 1000.0) -> nap.TsdFrame:
    """Load wheel data and derive position, velocity and acceleration.

    Args:
        one (ONE): an open ONE connection.
        eid (str): experiment id.
        frequency (float): resampling frequency in Hz. Defaults to 1000.0.

    Returns:
        nap.TsdFrame: position, velocity and acceleration on a uniform grid.
    """
    wheel_data = one.load_object(eid, obj="*wheel", collection="alf")
    position, timestamps = wheel_methods.interpolate_position(
        re_ts=wheel_data["timestamps"], re_pos=wheel_data["position"], freq=frequency
    )
    velocity, acceleration = wheel_methods.velocity_filtered(pos=position, fs=frequency)
    wheel = pd.DataFrame(
        dict(position=position, velocity=velocity, acceleration=acceleration),
        index=timestamps,
    )
    return nap.TsdFrame(wheel)


def load_session_data(
    one: ONE,
    eid: str,
    brain_region: str,
    band: str = "GCaMP",
    camera: str = "leftCamera",
    wheel_frequency: float = 1000.0,
) -> tuple[nap.Tsd, pd.DataFrame, dict[str, nap.TsdFrame]]:
    """Load photometry, trials and continuous regressors for one session.

    Args:
        one (ONE): an open ONE connection.
        eid (str): experiment id.
        brain_region (str): photometry region/channel, e.g. "SNc-l".
        band (str): photometry signal band. Defaults to "GCaMP".
        camera (str): pose camera label. Defaults to "leftCamera".
        wheel_frequency (float): wheel resampling frequency in Hz.

    Returns:
        tuple: (fluorescence, trials, continuous), where continuous maps a
        regressor name -> nap.TsdFrame ({"pose", "wheel"}).
    """
    psl = PhotometrySessionLoader(one=one, eid=eid)
    fluorescence = load_fluorescence(psl, brain_region, band)
    trials = load_trials(psl)
    continuous = {
        "pose": load_pose(one, eid, camera),
        "wheel": load_wheel(one, eid, wheel_frequency),
    }
    return fluorescence, trials, continuous
