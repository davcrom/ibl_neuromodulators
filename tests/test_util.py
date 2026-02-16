"""Tests for iblnm.util functions."""
import numpy as np
import pandas as pd
import pytest

from iblnm.util import (
    has_dataset,
    has_dataset_category,
    add_dataset_flags,
    drop_junk_duplicates,
    add_hemisphere,
    get_sessions,
)


@pytest.fixture
def mock_sessions_pqt(tmp_path):
    """Create a mock sessions.pqt with a mix of subjects and session types."""
    df = pd.DataFrame({
        'eid': [f'eid-{i}' for i in range(8)],
        'subject': ['mouseA', 'mouseA', 'mouseA', 'mouseB', 'mouseB',
                     'SP076', 'mouseC', 'mouseC'],  # SP076 is excluded
        'session_type': ['biased', 'biased', 'training', 'biased', 'biased',
                         'biased', 'habituation', 'biased'],  # habituation excluded
        'day_n': [1, 1, 2, 1, 2, 1, 1, 2],
        'has_raw_task': [True, False, True, True, True, True, True, True],
        'has_raw_photometry': [True, True, True, True, True, True, True, True],
        'has_extracted_task': [True, True, True, True, False, True, True, True],
        'has_extracted_photometry': [True, True, False, True, True, True, True, True],
    })
    path = tmp_path / 'sessions.pqt'
    df.to_parquet(path)
    return path


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


class TestDropJunkDuplicates:
    def test_defaults_to_raw_columns_only(self):
        """Default scoring uses has_raw_* columns, ignoring has_extracted_*."""
        df = pd.DataFrame({
            'subject': ['A', 'A'],
            'session_n': [1, 1],
            'has_raw_task': [False, True],
            'has_raw_photometry': [True, True],
            'has_extracted_task': [True, False],  # opposite of raw
        })
        result = drop_junk_duplicates(df, group_cols=['subject', 'session_n'], verbose=False)
        assert len(result) == 1
        # Should keep the session with more raw data (row 1), not more extracted
        assert result.iloc[0]['has_raw_task'] == True
        assert result.iloc[0]['has_extracted_task'] == False

    def test_keeps_raw_complete_over_incomplete(self):
        """When group has raw-complete and raw-incomplete, keep raw-complete."""
        df = pd.DataFrame({
            'subject': ['A', 'A'],
            'session_n': [1, 1],
            'has_raw_task': [True, False],
            'has_raw_photometry': [True, True],
        })
        result = drop_junk_duplicates(df, group_cols=['subject', 'session_n'], verbose=False)
        assert len(result) == 1
        assert result.iloc[0]['has_raw_task'] == True

    def test_keeps_one_when_equal_score(self):
        """When group has equal raw completeness, keep one."""
        df = pd.DataFrame({
            'subject': ['A', 'A'],
            'session_n': [1, 1],
            'has_raw_task': [True, True],
            'has_raw_photometry': [False, False],
        })
        result = drop_junk_duplicates(df, group_cols=['subject', 'session_n'], verbose=False)
        assert len(result) == 1

    def test_custom_completeness_cols(self):
        """Can specify custom completeness columns."""
        df = pd.DataFrame({
            'subject': ['A', 'A'],
            'session_n': [1, 1],
            'has_raw_task': [True, True],
            'custom_flag': [True, False],
        })
        result = drop_junk_duplicates(
            df, group_cols=['subject', 'session_n'],
            completeness_cols=['custom_flag'], verbose=False
        )
        assert len(result) == 1
        assert result.iloc[0]['custom_flag'] == True

    def test_multiple_groups(self):
        """Each group keeps one session."""
        df = pd.DataFrame({
            'subject': ['A', 'A', 'A', 'B', 'B'],
            'session_n': [1, 1, 2, 1, 1],
            'has_raw_task': [True, False, True, False, False],
            'has_raw_photometry': [True, True, True, True, False],
        })
        result = drop_junk_duplicates(df, group_cols=['subject', 'session_n'], verbose=False)
        assert len(result) == 3
        a1 = result[(result['subject'] == 'A') & (result['session_n'] == 1)]
        assert len(a1) == 1
        assert a1.iloc[0]['has_raw_task'] == True

    def test_preserves_unique_sessions(self):
        """Sessions with unique subject/session_n are preserved."""
        df = pd.DataFrame({
            'subject': ['A', 'B', 'C'],
            'session_n': [1, 1, 1],
            'has_raw_task': [True, False, True],
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


class TestGetSessions:
    def test_loads_and_cleans(self, mock_sessions_pqt):
        """get_sessions with all flags False returns cleaned, deduped sessions."""
        df = get_sessions(
            sessions_path=mock_sessions_pqt,
            require_extracted_task=False,
            require_extracted_photometry=False,
            require_qc=False,
            require_tipt=False,
            verbose=False,
        )
        assert isinstance(df, pd.DataFrame)
        raw = pd.read_parquet(mock_sessions_pqt)
        # Should have removed SP076 (excluded subject), habituation (excluded type),
        # and one duplicate (mouseA day_n=1 has two entries)
        assert len(df) < len(raw)
        # SP076 should be gone
        assert 'SP076' not in df['subject'].values
        # habituation should be gone
        assert 'habituation' not in df['session_type'].values
        # mouseA day_n=1 should have only one entry (deduped)
        assert len(df[(df['subject'] == 'mouseA') & (df['day_n'] == 1)]) == 1

    def test_filters_extracted(self, mock_sessions_pqt):
        """Sessions missing extracted data are excluded."""
        df = get_sessions(
            sessions_path=mock_sessions_pqt,
            require_extracted_task=True,
            require_extracted_photometry=True,
            require_qc=False,
            require_tipt=False,
            verbose=False,
        )
        assert df['has_extracted_task'].all()
        assert df['has_extracted_photometry'].all()
