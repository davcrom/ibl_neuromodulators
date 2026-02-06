"""Tests for iblnm.util functions."""
import numpy as np
import pandas as pd
import pytest

from iblnm.util import (
    has_dataset,
    has_dataset_category,
    add_dataset_flags,
    drop_junk_duplicates,
    resolve_session_status,
    add_hemisphere,
)


class TestHasDataset:
    def test_dataset_present(self):
        session = pd.Series({'datasets': ['a', 'b', 'c']})
        assert has_dataset(session, 'a') is True
        assert has_dataset(session, 'b') is True

    def test_dataset_absent(self):
        session = pd.Series({'datasets': ['a', 'b', 'c']})
        assert has_dataset(session, 'd') is False

    def test_empty_list(self):
        session = pd.Series({'datasets': []})
        assert has_dataset(session, 'a') is False

    def test_not_a_list(self):
        session = pd.Series({'datasets': 'not a list'})
        assert has_dataset(session, 'a') is False

    def test_numpy_array(self):
        session = pd.Series({'datasets': np.array(['a', 'b', 'c'])})
        assert has_dataset(session, 'a') is True
        assert has_dataset(session, 'd') is False


class TestResolveSessionStatus:
    def test_single_good_session(self):
        """Single session meeting requirements should be 'good'."""
        df = pd.DataFrame({
            'eid': ['a'],
            'data_complete': [True],
            'trials_complete': [True],
        })
        result = resolve_session_status(df, columns=['data_complete', 'trials_complete'])
        assert result.iloc[0] == 'good'

    def test_single_junk_session(self):
        """Single session not meeting requirements should be 'junk'."""
        df = pd.DataFrame({
            'eid': ['a'],
            'data_complete': [False],
            'trials_complete': [True],
        })
        result = resolve_session_status(df, columns=['data_complete', 'trials_complete'])
        assert result.iloc[0] == 'junk'

    def test_one_good_one_junk(self):
        """One good, one junk: good stays good, junk stays junk."""
        df = pd.DataFrame({
            'eid': ['a', 'b'],
            'data_complete': [True, False],
            'trials_complete': [True, True],
        })
        result = resolve_session_status(df, columns=['data_complete', 'trials_complete'])
        assert result.iloc[0] == 'good'
        assert result.iloc[1] == 'junk'

    def test_multiple_good_sessions_conflict(self):
        """Multiple sessions meeting requirements should be 'conflict'."""
        df = pd.DataFrame({
            'eid': ['a', 'b'],
            'data_complete': [True, True],
            'trials_complete': [True, True],
        })
        result = resolve_session_status(df, columns=['data_complete', 'trials_complete'])
        assert result.iloc[0] == 'conflict'
        assert result.iloc[1] == 'conflict'

    def test_all_junk(self):
        """All sessions not meeting requirements should be 'junk'."""
        df = pd.DataFrame({
            'eid': ['a', 'b'],
            'data_complete': [False, False],
            'trials_complete': [True, True],
        })
        result = resolve_session_status(df, columns=['data_complete', 'trials_complete'])
        assert all(result == 'junk')


class TestDropJunkDuplicates:
    def test_keeps_good_over_junk(self):
        """When group has good and junk, keep only good."""
        df = pd.DataFrame({
            'subject': ['A', 'A'],
            'session_n': [1, 1],
            'session_type': ['biased', 'biased'],
            'session_status': ['good', 'junk'],
        })
        result = drop_junk_duplicates(df, group_cols=['subject', 'session_n'], verbose=False)
        assert len(result) == 1
        assert result.iloc[0]['session_status'] == 'good'

    def test_keeps_one_junk_when_no_good(self):
        """When group has only junk, keep one."""
        df = pd.DataFrame({
            'subject': ['A', 'A'],
            'session_n': [1, 1],
            'session_type': ['biased', 'biased'],
            'session_status': ['junk', 'junk'],
        })
        result = drop_junk_duplicates(df, group_cols=['subject', 'session_n'], verbose=False)
        assert len(result) == 1
        assert result.iloc[0]['session_status'] == 'junk'

    def test_drops_conflicts(self):
        """Conflicts should be dropped entirely."""
        df = pd.DataFrame({
            'subject': ['A', 'A'],
            'session_n': [1, 1],
            'session_type': ['biased', 'biased'],
            'session_status': ['conflict', 'conflict'],
        })
        result = drop_junk_duplicates(df, group_cols=['subject', 'session_n'], verbose=False)
        assert len(result) == 0

    def test_multiple_groups(self):
        """Each group should have one session."""
        df = pd.DataFrame({
            'subject': ['A', 'A', 'A', 'B', 'B'],
            'session_n': [1, 1, 2, 1, 1],
            'session_type': ['biased', 'biased', 'biased', 'biased', 'biased'],
            'session_status': ['good', 'junk', 'good', 'junk', 'junk'],
        })
        result = drop_junk_duplicates(df, group_cols=['subject', 'session_n'], verbose=False)
        # A has 2 session_n values (1 and 2), B has 1
        assert len(result) == 3
        # Check group A, session_n=1 kept good
        a1 = result[(result['subject'] == 'A') & (result['session_n'] == 1)]
        assert len(a1) == 1
        assert a1.iloc[0]['session_status'] == 'good'

    def test_preserves_unique_sessions(self):
        """Sessions with unique subject/session_n should be preserved."""
        df = pd.DataFrame({
            'subject': ['A', 'B', 'C'],
            'session_n': [1, 1, 1],
            'session_type': ['biased', 'biased', 'biased'],
            'session_status': ['good', 'junk', 'good'],
        })
        result = drop_junk_duplicates(df, group_cols=['subject', 'session_n'], verbose=False)
        assert len(result) == 3


class TestAddDatasetFlags:
    def test_adds_category_columns(self):
        """Should add has_* columns for each category."""
        df = pd.DataFrame({
            'datasets': [
                ['alf/_ibl_trials.table.pqt', 'alf/photometry/photometry.signal.pqt'],
                ['raw_photometry_data/_neurophotometrics_fpData.raw.pqt'],
            ]
        })
        result = add_dataset_flags(df)
        assert 'has_extracted_task' in result.columns
        assert 'has_extracted_photometry_signal' in result.columns
        assert 'has_raw_photometry' in result.columns
        assert result.iloc[0]['has_extracted_task'] == True
        assert result.iloc[0]['has_extracted_photometry_signal'] == True
        assert result.iloc[0]['has_raw_photometry'] == False
        assert result.iloc[1]['has_extracted_task'] == False
        assert result.iloc[1]['has_raw_photometry'] == True


class TestAddHemisphere:
    def test_single_fiber_left(self):
        """Single fiber with positive X should be left hemisphere."""
        df_sessions = pd.DataFrame({
            'subject': ['mouse1'],
            'target': ['VTA-DA'],
        })
        df_fibers = pd.DataFrame({
            'subject': ['mouse1'],
            'targeted_region': ['VTA'],
            'neuromodulator': ['DA'],
            'X-ml_um': [500],  # positive = left
        })
        result = add_hemisphere(df_sessions, region_col='target', df_fibers=df_fibers)
        assert result.iloc[0]['hemisphere'] == 'L'

    def test_single_fiber_right(self):
        """Single fiber with negative X should be right hemisphere."""
        df_sessions = pd.DataFrame({
            'subject': ['mouse1'],
            'target': ['VTA-DA'],
        })
        df_fibers = pd.DataFrame({
            'subject': ['mouse1'],
            'targeted_region': ['VTA'],
            'neuromodulator': ['DA'],
            'X-ml_um': [-500],  # negative = right
        })
        result = add_hemisphere(df_sessions, region_col='target', df_fibers=df_fibers)
        assert result.iloc[0]['hemisphere'] == 'R'

    def test_multiple_fibers_same_target_blank(self):
        """Multiple fibers per subject+target should leave hemisphere blank."""
        df_sessions = pd.DataFrame({
            'subject': ['mouse1'],
            'target': ['VTA-DA'],
        })
        df_fibers = pd.DataFrame({
            'subject': ['mouse1', 'mouse1'],
            'targeted_region': ['VTA', 'VTA'],
            'neuromodulator': ['DA', 'DA'],
            'X-ml_um': [500, -500],  # bilateral
        })
        result = add_hemisphere(df_sessions, region_col='target', df_fibers=df_fibers)
        assert pd.isna(result.iloc[0]['hemisphere'])

    def test_multiple_targets_different_hemispheres(self):
        """Multiple targets in different hemispheres should work correctly."""
        df_sessions = pd.DataFrame({
            'subject': ['mouse1', 'mouse1'],
            'target': ['VTA-DA', 'SNc-DA'],
        })
        df_fibers = pd.DataFrame({
            'subject': ['mouse1', 'mouse1'],
            'targeted_region': ['VTA', 'SNc'],
            'neuromodulator': ['DA', 'DA'],
            'X-ml_um': [500, -500],
        })
        result = add_hemisphere(df_sessions, region_col='target', df_fibers=df_fibers)
        assert result.iloc[0]['hemisphere'] == 'L'  # VTA-DA
        assert result.iloc[1]['hemisphere'] == 'R'  # SNc-DA

    def test_no_matching_fiber(self):
        """Session with no matching fiber should have NaN hemisphere."""
        df_sessions = pd.DataFrame({
            'subject': ['mouse1'],
            'target': ['VTA-DA'],
        })
        df_fibers = pd.DataFrame({
            'subject': ['mouse2'],
            'targeted_region': ['VTA'],
            'neuromodulator': ['DA'],
            'X-ml_um': [500],
        })
        result = add_hemisphere(df_sessions, region_col='target', df_fibers=df_fibers)
        assert pd.isna(result.iloc[0]['hemisphere'])
