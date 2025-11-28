import pandas as pd
from tqdm import tqdm
import traceback
# ~from pymer4.models import Lmer

from one.api import ONE
from one.alf.exceptions import ALFObjectNotFound

from iblnm.data import PhotometrySession

one = ONE()

# TODO: make session file name an arg
SESSIONS_FNAME = 'sessions_2025-11-24-13h41.pqt'
df_sessions = pd.read_parquet(f'metadata/{SESSIONS_FNAME}')

df_sessions = df_sessions.query('NM == "ACh"')

sessions = []
exceptions_log = []
for idx, session in tqdm(df_sessions.iterrows(), total=len(df_sessions)):
    try:
        ps = PhotometrySession(session, one=one)
    except Exception as e:
        exception_info = {  # collect exception info
            'eid': session['eid'],
            'exception_type': type(e).__name__,
            'exception_message': str(e),
            'traceback': traceback.format_exc(),
            'description': "error loading data"
        }
        exceptions_log.append(exception_info)
        continue

    try:
        for event in ['stimOn', 'feedback']:
            ps.get_responses('gcamp', event)
    except Exception as e:
        exception_info = {  # collect exception info
            'eid': session['eid'],
            'exception_type': type(e).__name__,
            'exception_message': str(e),
            'traceback': traceback.format_exc(),
            'description': "error extracting responses"
        }
        exceptions_log.append(exception_info)
        continue

    sessions.append(ps)


# Get photometry location data
df_insertions = pd.read_csv('metadata/insertions.csv')
df_insertions['targeted_region'] = df_insertions['targeted_region'] + df_insertions['X-ml_um'].apply(lambda x: '-l' if x > 0 else '-r')
df_insertions = df_insertions.set_index(['subject', 'targeted_region'])


response_vectors = []
# Add convenience columns for analyses
for session in tqdm(sessions):
    for target in session.targets['gcamp']:
        # Initialize coordinates dict outside event loop
        if not hasattr(session, 'coordinates'):
            session.coordinates = {}
            session.hemisphere = {}
        if not hasattr(session, 'response_vectors'):
            session.response_vectors = {}
        if target not in session.response_vectors.keys():
            session.response_vectors[target] = {}

        for event in session.responses.keys():
            # Find the corresponding insertion
            try:
                subject_insertions = df_insertions.loc[session.subject]
            except KeyError:
                continue
            in_insertions = [
                target in region for region in subject_insertions.index
            ]
            if sum(in_insertions) != 1:
                continue
            # Get the matching insertion
            insertion = subject_insertions[in_insertions].iloc[0]

            # Assign coordinates (only once per target)
            if target not in session.coordinates:
                session.coordinates[target] = {
                    'ML': insertion['X-ml_um'],
                    'AP': insertion['Y-ap_um'],
                    'DV': insertion['Z-dv_um']
                }
                session.hemisphere[target] = -1 if session.coordinates[target]['ML'] > 0 else 1

            # Create a signed contrast column
            session.trials['signed_contrast'] = -1 * session.trials['contrastLeft']
            session.trials['signed_contrast'] = session.trials['signed_contrast'].fillna(
                session.trials['contrastRight']
            )
            session.trials['relative_contrast'] = session.trials.apply(
                lambda x: x['signed_contrast'] * session.hemisphere[target],
                axis='columns'
            )
            session.trials['side'] = session.trials.apply(  # True is contra , False is ipsi
                lambda x: np.signbit(x['relative_contrast']), axis='columns'
            )
            # TEMPFIX: get_responses used to drop NaN trials
            trials = session.trials.dropna(subset=f'{event}_times').copy()
            trial_groups = trials.groupby(
                ['relative_contrast', 'side', 'feedbackType']
            )
            responses = session.responses[event][target]['gcamp']
            tpts = session.get_response_tpts('gcamp')
            responses = normalize_responses(responses, tpts)
            i0, i1 = tpts.searchsorted(RESPONSE_WINDOWS['early'])
            assert len(trials) == len(responses)

            session.response_vectors[target][event] = np.array([
                responses[trial_indices, i0:i1].mean(axis=1).mean(axis=0)
                for group_key, trial_indices in trial_groups.indices.items()
            ])

        # Only add data if coordinates were successfully assigned
        if target in session.coordinates:
            data = {
                'eid': session.eid,
                'target': target,
                'ML': session.coordinates[target]['ML'],
                'AP': session.coordinates[target]['AP'],
                'DV': session.coordinates[target]['DV'],
                'response_vector': np.concatenate([
                    rv for rv in session.response_vectors[target].values()
                ]),
            'session_type': protocol2type(session.task_protocol)
            }
            response_vectors.append(data)

df = pd.DataFrame(response_vectors)










import itertools

# Define all possible trial types
ALL_CONTRASTS = [0.0, 0.0625, 0.125, 0.25, 1.0]
ALL_SIDES = [True, False]
ALL_FEEDBACKS = [1.0, -1.0]
ALL_TRIAL_TYPES = list(itertools.product(ALL_CONTRASTS, ALL_SIDES, ALL_FEEDBACKS))

response_vectors = []
# Add convenience columns for analyses
for session in tqdm(sessions):
    for target in session.targets['gcamp']:
        # Initialize coordinates dict outside event loop
        if not hasattr(session, 'coordinates'):
            session.coordinates = {}
            session.hemisphere = {}
        if not hasattr(session, 'response_vectors'):
            session.response_vectors = {}
        if target not in session.response_vectors.keys():
            session.response_vectors[target] = {}

        for event in session.responses.keys():
            # Find the corresponding insertion
            try:
                subject_insertions = df_insertions.loc[session.subject]
            except KeyError:
                continue
            in_insertions = [
                target in region for region in subject_insertions.index
            ]
            if sum(in_insertions) != 1:
                continue
            # Get the matching insertion
            insertion = subject_insertions[in_insertions].iloc[0]

            # Assign coordinates (only once per target)
            if target not in session.coordinates:
                session.coordinates[target] = {
                    'ML': insertion['X-ml_um'],
                    'AP': insertion['Y-ap_um'],
                    'DV': insertion['Z-dv_um']
                }
                session.hemisphere[target] = -1 if session.coordinates[target]['ML'] > 0 else 1

            # Create a signed contrast column
            session.trials['signed_contrast'] = -1 * session.trials['contrastLeft']
            session.trials['signed_contrast'] = session.trials['signed_contrast'].fillna(
                session.trials['contrastRight']
            )
            session.trials['relative_contrast'] = session.trials.apply(
                lambda x: x['signed_contrast'] * session.hemisphere[target],
                axis='columns'
            )
            session.trials['side'] = session.trials.apply(  # True is contra , False is ipsi
                lambda x: np.signbit(x['relative_contrast']), axis='columns'
            )
            # TEMPFIX: get_responses used to drop NaN trials
            trials = session.trials.dropna(subset=f'{event}_times').copy()
            trial_groups = trials.groupby(
                ['relative_contrast', 'side', 'feedbackType']
            )
            responses = session.responses[event][target]['gcamp']
            tpts = session.get_response_tpts('gcamp')
            responses = normalize_responses(responses, tpts)
            i0, i1 = tpts.searchsorted(RESPONSE_WINDOWS['early'])
            assert len(trials) == len(responses)

            # Create a dictionary mapping trial types to their mean responses
            trial_type_responses = {
                group_key: responses[trial_indices, i0:i1].mean(axis=1).mean(axis=0)
                for group_key, trial_indices in trial_groups.indices.items()
            }

            # Create response vector with NaN for missing trial types
            session.response_vectors[target][event] = np.array([
                trial_type_responses.get(trial_type, np.nan)
                for trial_type in ALL_TRIAL_TYPES
            ])

        # Only add data if coordinates were successfully assigned
        if target in session.coordinates:
            data = {
                'subject': session.subject,
                'eid': session.eid,
                'target': target,
                'ML': session.coordinates[target]['ML'],
                'AP': session.coordinates[target]['AP'],
                'DV': session.coordinates[target]['DV'],
                'response_vector': np.concatenate([
                    rv for rv in session.response_vectors[target].values()
                ]),
                'session_type': protocol2type(session.task_protocol)
            }
            response_vectors.append(data)

df = pd.DataFrame(response_vectors)


