"""Tests for scripts/rollup.py — the six rollup files, over synthetic stores."""
import pandas as pd

from iblnm.config import POSE_MEASURES, VIDEO_QC_COLS
from iblnm.util import LOG_COLUMNS
from tests.test_data import (
    _collector_catalog, _write_pose_session, qc_group, store_photometry_qc,
)
from tests.test_util import _write_session_h5

import scripts.rollup as rollup


def _group(h5_dir, session_types, scan_h5_errors=False):
    """Group over `eid -> session_type`, single-region, reading `h5_dir`."""
    from iblnm.data import PhotometrySessionGroup
    return PhotometrySessionGroup.from_catalog(
        _collector_catalog(session_types), one=None, h5_dir=h5_dir,
        scan_h5_errors=scan_h5_errors)


POSE_SERIES = pd.Series({
    'eid': 'placeholder', 'subject': 'mouse_A', 'number': 1,
    'start_time': '2024-01-01T10:00:00', 'session_type': 'biased',
})
POSE_STEPS = {'paw': 1.0, 'nose': 2.0, 'tongue_speed': 3.0,
              'tongue_likelihood': 0.5}


class TestCohortEids:
    """The CSV's cohort: every filter but the task-side three."""

    def test_keeps_every_session_type_and_drops_off_target(self, tmp_path):
        catalog = _collector_catalog({'eid-training': 'training',
                                      'eid-habituation': 'habituation'})
        off_target = _collector_catalog({'eid-off': 'biased'})
        off_target['target_NM'] = [['SI-ACh']]
        catalog = pd.concat([catalog, off_target], ignore_index=True)
        # deduplicate resolves same-subject same-day sessions; every session
        # here is its own subject's first.
        catalog['day_n'] = 0

        eids = rollup.cohort_eids(catalog, h5_dir=tmp_path)

        assert eids == {'eid-training', 'eid-habituation'}


def _csv_pose_table():
    """Pose rows carrying every exported column, in a scrambled order.

    Sorted output must be eid-b (LP, ephys), eid-a (LP, training), eid-c (no
    LP): descending `lp_exists`, then session type rank, then video QC score.
    """
    rows = [
        {'eid': 'eid-a', 'lp_exists': True, 'session_type': 'training',
         'video_qc_score': 0.5},
        {'eid': 'eid-c', 'lp_exists': False, 'session_type': 'ephys',
         'video_qc_score': 0.9},
        {'eid': 'eid-b', 'lp_exists': True, 'session_type': 'ephys',
         'video_qc_score': 0.1},
    ]
    table = pd.DataFrame(rows)
    for column in rollup.LP_SESSIONS_COLUMNS:
        if column not in table.columns and column != 'LP status':
            table[column] = 0.0
    return table


class TestPoseExport:
    """LightningPoseSessions.csv: the analysis-ready view of the pose table."""

    def test_column_list_and_sort_order(self, tmp_path):
        path = tmp_path / 'LightningPoseSessions.csv'

        rollup.pose_export(_csv_pose_table(),
                           {'eid-a', 'eid-b', 'eid-c'}, path)

        written = pd.read_csv(path)
        assert list(written.columns) == rollup.LP_SESSIONS_COLUMNS
        assert list(written['eid']) == ['eid-b', 'eid-a', 'eid-c']
        assert list(written['LP status']) == ['COMPLETE', 'COMPLETE', 'NONE']

    def test_session_outside_the_cohort_is_dropped(self, tmp_path):
        path = tmp_path / 'LightningPoseSessions.csv'

        rollup.pose_export(_csv_pose_table(), {'eid-a'}, path)

        assert list(pd.read_csv(path)['eid']) == ['eid-a']


class TestRollupPose:
    """pose.pqt rolls the filtered sessions' video groups up, one row each."""

    def test_writes_the_pose_table(self, tmp_path):
        for eid in ('eid-1', 'eid-2'):
            _write_pose_session(tmp_path, eid, POSE_STEPS, drift=0.3,
                                peak_lags=[0.1, 0.2, 0.4], qc_lp='PASS',
                                series=POSE_SERIES)
        video_qc = {'eid-1': {col: 'PASS' for col in VIDEO_QC_COLS},
                    'eid-2': {col: 'FAIL' for col in VIDEO_QC_COLS}}
        path = tmp_path / 'pose.pqt'

        rollup.rollup_pose(_group(tmp_path, {'eid-1': 'biased',
                                             'eid-2': 'biased'}),
                           video_qc, path)

        written = pd.read_parquet(path).set_index('eid')
        assert set(written.index) == {'eid-1', 'eid-2'}
        assert {'lp_exists', 'drift', 'peak_lag_early', 'peak_val_late',
                'qc_lp', 'video_qc_score', 'fraction_correct',
                *VIDEO_QC_COLS, *POSE_MEASURES} <= set(written.columns)
        assert written.loc['eid-1', 'drift'] == 0.3
        assert written.loc['eid-2', 'qc_videoLeft_focus'] == 'FAIL'

    def test_filtered_out_session_is_absent(self, tmp_path):
        for eid in ('eid-in', 'eid-out'):
            _write_pose_session(tmp_path, eid, POSE_STEPS, drift=0.1,
                                peak_lags=[0.0, 0.0, 0.0], qc_lp='PASS',
                                series=POSE_SERIES)
        group = _group(tmp_path, {'eid-in': 'biased', 'eid-out': 'training'})
        group.filter_sessions(session_types=('biased',), targetnms=False,
                              qc_blockers=set(), min_performance=False,
                              required_contrasts=False)
        path = tmp_path / 'pose.pqt'

        rollup.rollup_pose(group, {}, path)

        assert list(pd.read_parquet(path)['eid']) == ['eid-in']


class TestRollupCatalog:
    """sessions.pqt is rebuilt from the store's own `metadata` groups."""

    def test_writes_the_fixed_up_catalog(self, tmp_path):
        _write_session_h5(tmp_path, 'eid-1', 'M1', brain_region=['SNC'],
                          hemisphere=['l'], target_NM=['SNc-DA'])
        _write_session_h5(tmp_path, 'eid-2', 'M1')
        path = tmp_path / 'sessions.pqt'

        catalog = rollup.rollup_catalog(tmp_path, path=path)

        written = pd.read_parquet(path).set_index('eid')
        assert set(written.index) == {'eid-1', 'eid-2'}
        # The cross-session fixups ran: eid-2's empty region filled from its
        # subject's other session, and the target NM was derived from it.
        assert list(written.loc['eid-2', 'brain_region']) == ['SNc']
        assert list(written.loc['eid-2', 'target_NM']) == ['SNc-DA']
        assert set(catalog['eid']) == {'eid-1', 'eid-2'}


class TestRollupQc:
    """qc_photometry.pqt is one row per recording, metrics band-suffixed."""

    def test_writes_band_suffixed_metrics_per_recording(self, tmp_path):
        store_photometry_qc(tmp_path, 'eid-1', {
            'VTA': {'n_unique_samples_GCaMP': 0.5,
                    'n_unique_samples_Isosbestic': 0.4},
            'SNc': {'n_unique_samples_GCaMP': 0.002,
                    'n_unique_samples_Isosbestic': 0.3}})
        group = qc_group(tmp_path, {'eid-1': ['VTA', 'SNc']})
        group.filter_sessions(session_types=False, targetnms=False,
                              qc_blockers=set(), min_performance=False,
                              required_contrasts=False, photometry_qc=False)
        path = tmp_path / 'qc_photometry.pqt'

        rollup.rollup_qc(group, path)

        written = pd.read_parquet(path).set_index(['eid', 'brain_region'])
        assert 'band' not in written.columns
        assert 'n_unique_samples' not in written.columns
        assert written.loc[('eid-1', 'VTA'), 'n_unique_samples_GCaMP'] == 0.5
        assert written.loc[('eid-1', 'SNc'), 'n_unique_samples_Isosbestic'] == 0.3

    def test_recording_cut_by_the_qc_threshold_is_absent(self, tmp_path):
        """The threshold filter cuts recordings, so the rollup loses that row."""
        store_photometry_qc(tmp_path, 'eid-1', {
            'VTA': {'n_unique_samples_GCaMP': 0.5,
                    'n_unique_samples_Isosbestic': 0.4},
            'SNc': {'n_unique_samples_GCaMP': 0.0001,
                    'n_unique_samples_Isosbestic': 0.3}})
        group = qc_group(tmp_path, {'eid-1': ['VTA', 'SNc']})
        group.filter_sessions(session_types=False, targetnms=False,
                              qc_blockers=set(), min_performance=False,
                              required_contrasts=False)
        path = tmp_path / 'qc_photometry.pqt'

        rollup.rollup_qc(group, path)

        assert list(pd.read_parquet(path)['brain_region']) == ['VTA']


def _write_performance(h5_dir, eid, performance):
    """Write one session's metadata plus its `trials/performance` product."""
    from unittest.mock import MagicMock
    from iblnm.data import PhotometrySession
    series = _collector_catalog({eid: 'biased'}).iloc[0]
    ps = PhotometrySession(series, one=MagicMock(), load_data=False)
    ps.performance = performance
    ps.save_h5(h5_dir / f'{eid}.h5', groups=['metadata', 'trials'])


class TestRollupPerformance:
    """performance.pqt is the sessions' stored `trials/performance` products."""

    def test_writes_one_row_per_session_with_its_metrics(self, tmp_path):
        _write_performance(tmp_path, 'eid-1', {'fraction_correct': 0.82,
                                               'n_trials': 300.0,
                                               'contrasts': [0.0, 1.0]})
        _write_performance(tmp_path, 'eid-2', {'fraction_correct': 0.61,
                                               'n_trials': 250.0,
                                               'contrasts': [0.0, 0.5, 1.0]})
        path = tmp_path / 'performance.pqt'

        rollup.rollup_performance(_group(tmp_path, {'eid-1': 'biased',
                                                    'eid-2': 'biased'}), path)

        written = pd.read_parquet(path).set_index('eid')
        assert set(written.index) == {'eid-1', 'eid-2'}
        assert written.loc['eid-1', 'fraction_correct'] == 0.82
        assert list(written.loc['eid-2', 'contrasts']) == [0.0, 0.5, 1.0]

    def test_store_without_the_product_writes_an_empty_file(self, tmp_path):
        """A rollup reports what the store holds, including nothing."""
        _write_session_h5(tmp_path, 'eid-1', 'mouse_A')
        path = tmp_path / 'performance.pqt'

        rollup.rollup_performance(_group(tmp_path, {'eid-1': 'biased'}), path)

        assert pd.read_parquet(path).empty

    def test_filtered_out_session_is_absent(self, tmp_path):
        """`load_performance` reads the catalog; the rollup reports the cohort."""
        _write_performance(tmp_path, 'eid-in', {'fraction_correct': 0.8,
                                                'contrasts': [1.0]})
        _write_performance(tmp_path, 'eid-out', {'fraction_correct': 0.5,
                                                 'contrasts': [1.0]})
        group = _group(tmp_path, {'eid-in': 'biased', 'eid-out': 'training'})
        group.filter_sessions(session_types=('biased',), targetnms=False,
                              qc_blockers=set(), min_performance=False,
                              required_contrasts=False)
        path = tmp_path / 'performance.pqt'

        rollup.rollup_performance(group, path)

        assert list(pd.read_parquet(path)['eid']) == ['eid-in']


class TestRollupErrors:
    """errors.pqt aggregates the filtered sessions' per-product error groups."""

    def test_writes_the_log_columns(self, tmp_path):
        _write_session_h5(tmp_path, 'eid-1', 'mouse_A',
                          errors=[ValueError("bad value")])
        _write_session_h5(tmp_path, 'eid-2', 'mouse_B',
                          errors=[TypeError("bad type")])
        path = tmp_path / 'errors.pqt'

        rollup.rollup_errors(_group(tmp_path, {'eid-1': 'biased',
                                               'eid-2': 'biased'}), path)

        written = pd.read_parquet(path)
        assert list(written.columns) == LOG_COLUMNS
        assert set(written['eid']) == {'eid-1', 'eid-2'}
        assert set(written['error_type']) == {'ValueError', 'TypeError'}

    def test_filtered_out_session_is_absent(self, tmp_path):
        _write_session_h5(tmp_path, 'eid-in', 'mouse_A', 'biased',
                          errors=[ValueError("kept")])
        _write_session_h5(tmp_path, 'eid-out', 'mouse_B', 'training',
                          errors=[TypeError("dropped")])
        group = _group(tmp_path, {'eid-in': 'biased', 'eid-out': 'training'})
        group.filter_sessions(session_types=('biased',), targetnms=False,
                              qc_blockers=set(), min_performance=False,
                              required_contrasts=False)
        path = tmp_path / 'errors.pqt'

        rollup.rollup_errors(group, path)

        assert list(pd.read_parquet(path)['eid']) == ['eid-in']
