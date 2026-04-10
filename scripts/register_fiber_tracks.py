from ibllib.pipes.histology import register_track, _parse_filename, load_track_csv
from iblatlas.atlas import AllenAtlas
from one.api import ONE

from iblnm.config import PROJECT_ROOT

ba = AllenAtlas()
one = ONE()

track_file = PROJECT_ROOT / 'histology/2025-07-21_ZFM-09139_001_fiber00_pts.csv'

search_filter = _parse_filename(track_file)
chronic_insertion = one.alyx.rest(
    'chronic-insertions', 'list',
    subject=search_filter['subject'],
    name=search_filter['name']
)

# If no matching chronic insertion found we create it
if len(chronic_insertion) == 0:
    subj = one.alyx.rest('subjects', 'list', nickname=search_filter['subject'])[0]
    data = {
        'subject': search_filter['subject'],
        'name': search_filter['name'],
        'serial': search_filter['subject'] + '_' + search_filter['name'],
        'model': 'Fiber',
        'lab': subj['lab']
    }
    chronic = one.alyx.rest('chronic-insertions', 'create', data=data)
else:
    chronic = chronic_insertion[0]

picks = load_track_csv(track_file, brain_atlas=ba)

brain_locations, insertion_histology = register_track(
    chronic['id'], picks=picks, one=one, overwrite=True, channels=False, brain_atlas=ba, endpoint='chronic-insertions'
    )
