from functools import lru_cache
import pandas as pd

from one.api import ONE
from one.alf.spec import QC
from one.alf.exceptions import ALFObjectNotFound

from iblnm.config import STRAIN2NM, LINE2NM

from iblnm.validation import exception_logger

from iblnm.util import traj2coord


# Values to extract from the session dict
SESSIONDICT_KEYS = ['users', 'lab', 'end_time']


@lru_cache(maxsize=1)
def _get_default_connection():
    """
    Create and cache the default database connection. Cached connection allows
    repeated function calls without re-creating connection instance.
    """
    return ONE()


@exception_logger
def get_subject_info(session, one=None):
    """
    Get the mouse strain, line, and genotype for a given session. Try to infer
    the neuromodulatory cell type targeted using this info.

    Parameters
    ----------
    session : pd.Series
        A series representing an IBL session containing the mouse 'nickname' in
        the 'subject' column.

    one : one.api.OneAlyx
        Alyx database connection instance.

    Returns
    -------
    session : pd.Series
        The session series with new entries for mouse 'strain', 'line', and
        'genotype'.
    """
    if one is None:
        one = _get_default_connection()
    subjects = one.alyx.rest('subjects', 'list', nickname=session['subject'])

    if len(subjects) != 1:
        raise ValueError(
            f"Expected exactly 1 subject for '{session['subject']}', found {len(subjects)}"
        )

    subject = subjects[0]
    for key in ['strain', 'line', 'genotype']:
        session[key] = subject[key]

    session['NM'] = STRAIN2NM.get(session['strain']) or LINE2NM.get(session['line'])
    return session


@exception_logger
def get_session_dict(session, one=None):
    """Unpack metadata from the Alyx session dict (end_time, lab, users)."""
    if one is None:
        one = _get_default_connection()

    session_dict = one.alyx.rest('sessions', 'read', id=session['eid'])

    for key in SESSIONDICT_KEYS:
        if key in session_dict:
            session[key] = session_dict[key]

    # TEMPFIX: get datasets for comparison with get_datasets
    if 'data_dataset_session_related' in session_dict:
        session['_datasets_from_session_dict'] = [d['name'] for d in session_dict['data_dataset_session_related']]
    else:
        session['_datasets_from_session_dict'] = []

    return session


def get_block_info(eid, one=None):
    """Fetch block structure fields from the Alyx session JSON.

    Parameters
    ----------
    eid : str
        Session UUID.
    one : ONE, optional
        ONE connection. Uses default if None.

    Returns
    -------
    dict
        Keys ``len_blocks``, ``positions``, ``block_probability_set``.
        Values are None when the corresponding field is absent from the JSON.
    """
    if one is None:
        one = _get_default_connection()

    session_dict = one.alyx.rest('sessions', 'read', id=eid)
    session_json = session_dict.get('json') or {}

    return {
        'len_blocks': session_json.get('LEN_BLOCKS'),
        'positions': session_json.get('POSITIONS'),
        'block_probability_set': session_json.get('BLOCK_PROBABILITY_SET'),
    }


def _hemisphere_from_regions(regions):
    """Derive hemisphere list from region names."""
    return [r[-1] if r.endswith(('-l', '-r')) else '' for r in regions]


@exception_logger
def get_brain_region(session, one=None):
    """Populate brain_region and hemisphere from experiment description or locations file.

    Tries the experiment description first. If missing, falls back to
    photometryROI.locations.pqt. Raises ALFObjectNotFound if neither source
    has brain region data.
    """
    if one is None:
        one = _get_default_connection()

    # Try experiment description
    try:
        session_desc = one.load_dataset(session['eid'], '_ibl_experiment.description.yaml')
        fibers = session_desc.get('devices', {}).get('neurophotometrics', {}).get('fibers', {})
        regions = [fiber.get('location', '') for fiber in fibers.values()]
        session['brain_region'] = regions
        session['hemisphere'] = _hemisphere_from_regions(regions)
        return session
    except ALFObjectNotFound:
        pass

    # Fallback: photometry locations file
    try:
        loc = one.load_dataset(session['eid'], 'photometryROI.locations.pqt')
        regions = list(loc['brain_region'].values)
        session['brain_region'] = regions
        session['hemisphere'] = _hemisphere_from_regions(regions)
        return session
    except ALFObjectNotFound:
        pass

    raise ALFObjectNotFound(
        f"No brain_region source for {session['eid']}: "
        f"experiment description and photometryROI.locations.pqt both missing"
    )


@exception_logger
def get_datasets(session, one=None):
    """Store list of available datasets for the given eid."""
    if one is None:
        one = _get_default_connection()
    session['datasets'] = list(one.list_datasets(session['eid']))
    return session


@exception_logger
def get_extended_qc(series, one=None):
    """Fetch and normalize extended QC values for a session."""
    if one is None:
        one = _get_default_connection()
    session_dict = one.alyx.rest('sessions', 'read', id=series['eid'])

    series['qc_session'] = session_dict.get('qc')

    extended_qc = session_dict.get('extended_qc')
    if extended_qc is None:
        return series

    for key, val in extended_qc.items():
        # Normalize key name
        if key.endswith('_qc'):
            key = key.rstrip('_qc')
        if not key.startswith('_'):
            key = '_' + key
        key = 'qc' + key

        # Handle list values (QC outcome + underlying values)
        if isinstance(val, list):
            series[key.lstrip('qc_')] = val[1:]  # underlying values
            val = val[0]  # QC outcome

        # Normalize QC value to string
        if isinstance(val, bool):
            series[key] = 'PASS' if val else 'FAIL'
        elif isinstance(val, int):
            try:
                series[key] = QC(val).name
            except ValueError:
                series[key] = val
        elif isinstance(val, (float, str)):
            series[key] = val
        elif val is None:
            series[key] = None  # Let pandas handle as NaN
        else:
            raise ValueError(f"Unexpected QC value type for '{key}': {type(val)}")

    return series

@exception_logger
def get_fiber_coordinates(session, all_trajectories=None, one=None):

    if one is None:
        one = _get_default_connection()

    if all_trajectories is None:
        # FIXME: there is something wrong with the session dicts nested in Alyx
        # trajectories for this project (None, no subject, no project)
        # TEMPFIX: use the full list of all trajectories, search by chonic
        # insertion id (see below)
        all_trajectories = one.alyx.rest('trajectories', 'list')

    subject = session['subject'] if isinstance(session, pd.Series) else session
    insertions = one.alyx.rest('chronic-insertions', 'list', subject=subject)
    if len(insertions) == 0:
        print(f"No chronic insertions found for subject {subject}.")
        return []

    iids = [i['id'] for i in insertions]
    coords = []
    for iid in iids:
        trajectory = [
            t for t in all_trajectories if t['chronic_insertion'] == iid
            ]
        assert len(trajectory) == 1
        trajectory = trajectory[0]
        coords.append({
            'subject': subject,
            'fiber': trajectory['probe_name'],
            'coords': traj2coord(**trajectory)
            })

    # FIXME: return session as pd.Series with coords inserted
    return coords






