from functools import lru_cache
import pandas as pd

from one.api import ONE
from one.alf.spec import QC
from one.alf.exceptions import ALFObjectNotFound

from iblnm.config import STRAIN2NM, LINE2NM

from iblnm.util import exception_logger


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
def get_session_info(session, one=None):
    """Unpack metadata from session dict."""
    if one is None:
        one = _get_default_connection()

    # Fetch dicts with session info
    session_dict = one.alyx.rest('sessions', 'read', id=session['eid'])
    session_desc = one.load_dataset(session['eid'], '*experiment.desc*')

    # Add select keys from session dict
    for key in SESSIONDICT_KEYS:
        if key in session_dict:
            session[key] = session_dict[key]

    # TEMPFIX: get datasets for comparison with get_datasets
    if 'data_dataset_session_related' in session_dict:
        session['_datasets_from_session_dict'] = [d['name'] for d in session_dict['data_dataset_session_related']]
    else:
        session['_datasets_from_session_dict'] = []

    # Add brain regions from experiment description
    fibers = session_desc.get('devices', {}).get('neurophotometrics', {}).get('fibers', {})
    regions = [fiber.get('location', '') for fiber in fibers.values()]
    session['brain_region'] = [region.split('-')[0] for region in regions]
    session['hemisphere'] = [
        region[-1] if region.endswith('-l') or region.endswith('-r') else None for region in regions
        ]

    return session


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
