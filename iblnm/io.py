import traceback
from functools import lru_cache, wraps
import pandas as pd

from one.api import ONE
from one.alf.spec import QC
from one.alf.exceptions import ALFObjectNotFound

from iblnm.config import SESSIONDICT_KEYS, STRAIN2NM, LINE2NM, ALYX_DATASETS


def exception_logger(func):
    """
    Decorator that allows session processing functions to log exceptions.
    Use exlog parameter to capture errors instead of raising them.
    """
    @wraps(func)
    def wrapper(series, *args, exlog=None, **kwargs):
        try:
            return func(series, *args, **kwargs)
        except Exception as e:
            if exlog is not None:
                exception_info = {
                    'eid': series.get('eid', 'unknown'),
                    'subject': series.get('subject', 'unknown'),
                    'function': func.__name__,
                    'exception_type': type(e).__name__,
                    'exception_message': str(e),
                    'traceback': traceback.format_exc()
                }
                exlog.append(exception_info)
                # Return series unchanged to continue processing
                return series
            else:
                # Re-raise if no exlog provided
                raise
    return wrapper


@lru_cache(maxsize=1)
def _get_default_connection():
    """
    Create and cache the default database connection. Cached connection allows
    repeated function calls without re-creating connection instance.
    """
    return ONE()


@exception_logger
def unpack_session_dict(series, one=None):
    """Unpack metadata from session dict."""
    if one is None:
        one = _get_default_connection()
    session_dict = one.alyx.rest('sessions', 'read', id=series['eid'])
    for key in SESSIONDICT_KEYS:
        if key in session_dict:
            series[key] = session_dict[key]
    # TODO: temp - comparing with get_datasets to figure out best approach
    if 'data_dataset_session_related' in session_dict:
        series['_datasets_from_session_dict'] = [d['name'] for d in session_dict['data_dataset_session_related']]
    else:
        series['_datasets_from_session_dict'] = []
    return series


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

    session['NM'] = _infer_neuromodulator(session['strain'], session['line'])
    return session


def _infer_neuromodulator(strain, line):
    """Infer neuromodulator from strain/line. Returns NM name, 'none', or 'conflict'."""
    s_nm = STRAIN2NM.get(strain, 'none')
    l_nm = LINE2NM.get(line, 'none')

    if s_nm == 'none':
        return l_nm
    if l_nm == 'none':
        return s_nm
    return s_nm if s_nm == l_nm else 'conflict'


@exception_logger
def get_datasets(series, one=None):
    """Store list of available datasets from ALYX_DATASETS for the given eid."""
    if one is None:
        one = _get_default_connection()
    datasets = one.list_datasets(series['eid'])
    series['datasets'] = [d for d in ALYX_DATASETS if d in datasets]
    return series


@exception_logger
def get_target_regions(session, one=None):
    """Get ROI names and brain region targets from photometry locations dataset."""
    if one is None:
        one = _get_default_connection()
    try:
        locations = one.load_dataset(id=session['eid'], dataset='photometryROI.locations.pqt')
        session['roi'] = locations.index.to_list()
        session['target'] = locations['brain_region'].to_list()
    except ALFObjectNotFound:
        session['roi'] = []
        session['target'] = []
    return session
