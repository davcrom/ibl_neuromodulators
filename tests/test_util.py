"""Tests for iblnm.util functions."""
import numpy as np
import pandas as pd
import pytest

from iblnm.util import (
    has_dataset,
    has_dataset_category,
    add_dataset_flags,
    drop_junk_duplicates,
    make_log_entry,
    concat_logs,
    deduplicate_log,
    enforce_schema,
    get_session_type,
    get_targetNM,
    collect_session_errors,
    InvalidSessionType,
    InvalidTargetNM,
    LOG_COLUMNS,
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
        assert 'has_raw_photometry_signals' in result.columns
        assert result.iloc[0]['has_extracted_task'] == True
        assert result.iloc[0]['has_extracted_photometry_signal'] == True
        assert result.iloc[0]['has_raw_photometry_signals'] == False
        assert result.iloc[1]['has_extracted_task'] == False
        assert result.iloc[1]['has_raw_photometry_signals'] == True


# TODO: add tests for validate_hemisphere once it's fixed


class TestMakeLogEntry:
    """Tests for make_log_entry function."""

    def test_from_exception(self):
        """Extracts type, message, traceback from an exception."""
        try:
            raise ValueError("bad data")
        except ValueError as e:
            entry = make_log_entry('eid-1', error=e)

        assert entry['eid'] == 'eid-1'
        assert entry['error_type'] == 'ValueError'
        assert entry['error_message'] == 'bad data'
        assert 'Traceback' in entry['traceback']
        assert set(entry.keys()) == set(LOG_COLUMNS)

    def test_from_explicit_strings(self):
        """Creates entry from explicit error_type and error_message."""
        entry = make_log_entry('eid-2', error_type='UnknownNM', error_message='no NM found')

        assert entry['eid'] == 'eid-2'
        assert entry['error_type'] == 'UnknownNM'
        assert entry['error_message'] == 'no NM found'
        assert entry['traceback'] is None
        assert set(entry.keys()) == set(LOG_COLUMNS)

    def test_exception_overrides_explicit(self):
        """When both error and error_type given, exception wins."""
        try:
            raise KeyError("missing")
        except KeyError as e:
            entry = make_log_entry('eid-3', error=e, error_type='ShouldBeIgnored')

        assert entry['error_type'] == 'KeyError'

    def test_no_error_info_raises(self):
        """Must provide either error or error_type."""
        with pytest.raises(ValueError):
            make_log_entry('eid-4')


class TestConcatLogs:
    """Tests for concat_logs function."""

    def test_concatenates_multiple_logs(self):
        """Multiple logs are stacked, not merged."""
        log1 = pd.DataFrame({
            'eid': ['a'], 'error_type': ['ValueError'],
            'error_message': ['bad'], 'traceback': [None],
        })
        log2 = pd.DataFrame({
            'eid': ['b'], 'error_type': ['KeyError'],
            'error_message': ['missing'], 'traceback': [None],
        })
        result = concat_logs([log1, log2])

        assert len(result) == 2
        assert list(result.columns) == LOG_COLUMNS
        assert set(result['eid']) == {'a', 'b'}

    def test_skips_none_entries(self):
        """None entries in the list are skipped."""
        log1 = pd.DataFrame({
            'eid': ['a'], 'error_type': ['ValueError'],
            'error_message': ['bad'], 'traceback': [None],
        })
        result = concat_logs([None, log1, None])

        assert len(result) == 1

    def test_drops_extra_columns(self):
        """Extra columns beyond LOG_COLUMNS are dropped."""
        log = pd.DataFrame({
            'eid': ['a'], 'error_type': ['ValueError'],
            'error_message': ['bad'], 'traceback': [None],
            'subject': ['mouse1'], 'function': ['foo'],
        })
        result = concat_logs([log])

        assert list(result.columns) == LOG_COLUMNS
        assert 'subject' not in result.columns

    def test_empty_list_returns_empty_df(self):
        """Empty list returns DataFrame with LOG_COLUMNS."""
        result = concat_logs([])

        assert len(result) == 0
        assert list(result.columns) == LOG_COLUMNS

    def test_all_none_returns_empty_df(self):
        """All-None list returns empty DataFrame."""
        result = concat_logs([None, None])

        assert len(result) == 0
        assert list(result.columns) == LOG_COLUMNS

    def test_preserves_duplicate_eids(self):
        """Same eid from different sources produces multiple rows."""
        log1 = pd.DataFrame({
            'eid': ['a'], 'error_type': ['ValueError'],
            'error_message': ['bad data'], 'traceback': [None],
        })
        log2 = pd.DataFrame({
            'eid': ['a'], 'error_type': ['KeyError'],
            'error_message': ['missing key'], 'traceback': [None],
        })
        result = concat_logs([log1, log2])

        assert len(result) == 2
        assert (result['eid'] == 'a').all()


class TestEnforceSchema:
    """Tests for enforce_schema function."""

    def test_fills_missing_list_columns(self):
        """Missing list columns are added with empty list default."""
        schema = {'tags': (list, [])}
        df = pd.DataFrame({'eid': ['a', 'b']})
        result = enforce_schema(df, schema)

        assert 'tags' in result.columns
        assert result.loc[0, 'tags'] == []
        assert result.loc[1, 'tags'] == []

    def test_fills_missing_scalar_columns(self):
        """Missing scalar columns are added with None default."""
        schema = {'NM': (str, None)}
        df = pd.DataFrame({'eid': ['a']})
        result = enforce_schema(df, schema)

        assert 'NM' in result.columns
        assert result.loc[0, 'NM'] is None

    def test_coerces_nan_to_list_default(self):
        """NaN values in list columns are replaced with empty list."""
        schema = {'brain_region': (list, [])}
        df = pd.DataFrame({'brain_region': [['VTA'], float('nan'), ['LC']]})
        result = enforce_schema(df, schema)

        assert result.loc[0, 'brain_region'] == ['VTA']
        assert result.loc[1, 'brain_region'] == []
        assert result.loc[2, 'brain_region'] == ['LC']

    def test_preserves_existing_values(self):
        """Existing valid values are not overwritten."""
        schema = {'NM': (str, None), 'brain_region': (list, [])}
        df = pd.DataFrame({
            'NM': ['DA', None, '5HT'],
            'brain_region': [['VTA'], ['LC'], float('nan')],
        })
        result = enforce_schema(df, schema)

        assert result.loc[0, 'NM'] == 'DA'
        assert result.loc[0, 'brain_region'] == ['VTA']
        assert result.loc[1, 'NM'] is None
        assert result.loc[1, 'brain_region'] == ['LC']
        assert result.loc[2, 'NM'] == '5HT'
        assert result.loc[2, 'brain_region'] == []

    def test_does_not_share_list_references(self):
        """Each row gets its own list instance, not a shared reference."""
        schema = {'tags': (list, [])}
        df = pd.DataFrame({'eid': ['a', 'b']})
        result = enforce_schema(df, schema)

        result.loc[0, 'tags'].append('x')
        assert result.loc[1, 'tags'] == []

    def test_converts_numpy_arrays_to_lists(self):
        """Numpy arrays (from parquet round-trip) are converted to lists."""
        schema = {'brain_region': (list, [])}
        df = pd.DataFrame({'brain_region': [np.array(['VTA', 'LC']), np.array([]), None]})
        result = enforce_schema(df, schema)

        assert result.loc[0, 'brain_region'] == ['VTA', 'LC']
        assert isinstance(result.loc[0, 'brain_region'], list)
        assert result.loc[1, 'brain_region'] == []
        assert result.loc[2, 'brain_region'] == []

    def test_ignores_columns_not_in_schema(self):
        """Columns not in the schema are left untouched."""
        schema = {'NM': (str, None)}
        df = pd.DataFrame({'eid': ['a'], 'subject': ['mouse1']})
        result = enforce_schema(df, schema)

        assert 'subject' in result.columns
        assert result.loc[0, 'subject'] == 'mouse1'


class TestGetSessionType:
    """Tests for get_session_type function."""

    def test_biased_choiceworld(self):
        """Standard biasedChoiceWorld protocol maps to 'biased'."""
        session = pd.Series({
            'eid': 'eid-1',
            'task_protocol': '_iblrig_tasks_biasedChoiceWorld6.4.2',
        })
        result = get_session_type(session)
        assert result['session_type'] == 'biased'

    def test_training_choiceworld(self):
        """Standard trainingChoiceWorld protocol maps to 'training'."""
        session = pd.Series({
            'eid': 'eid-1',
            'task_protocol': '_iblrig_tasks_trainingChoiceWorld6.4.2',
        })
        result = get_session_type(session)
        assert result['session_type'] == 'training'

    def test_ephys_choiceworld(self):
        """Standard ephysChoiceWorld maps to 'ephys'."""
        session = pd.Series({
            'eid': 'eid-1',
            'task_protocol': '_iblrig_tasks_ephysChoiceWorld6.4.2',
        })
        result = get_session_type(session)
        assert result['session_type'] == 'ephys'

    def test_habituation(self):
        """Habituation protocol maps to 'habituation'."""
        session = pd.Series({
            'eid': 'eid-1',
            'task_protocol': '_iblrig_tasks_habituationChoiceWorld6.4.2',
        })
        result = get_session_type(session)
        assert result['session_type'] == 'habituation'

    def test_red_flag_raises(self):
        """Protocol with red flag raises InvalidSessionType."""
        session = pd.Series({
            'eid': 'eid-1',
            'task_protocol': '_iblrig_tasks_biasedChoiceWorld_RPE',
        })
        with pytest.raises(InvalidSessionType, match='red flags'):
            get_session_type(session)

    def test_delay_red_flag_raises(self):
        """Delay variant is flagged."""
        session = pd.Series({
            'eid': 'eid-1',
            'task_protocol': '_iblrig_tasks_biasedChoiceWorld_DELAY',
        })
        with pytest.raises(InvalidSessionType, match='red flags'):
            get_session_type(session)

    def test_no_match_raises(self):
        """Protocol with no recognized type raises InvalidSessionType."""
        session = pd.Series({
            'eid': 'eid-1',
            'task_protocol': '_iblrig_tasks_unknownProtocol',
        })
        with pytest.raises(InvalidSessionType):
            get_session_type(session)

    def test_histology_session(self):
        """Protocol containing 'Histology' maps to 'histology'."""
        session = pd.Series({
            'eid': 'eid-1',
            'task_protocol': 'Histology',
        })
        result = get_session_type(session)
        assert result['session_type'] == 'histology'

    def test_ephyssessions_maps_to_biased(self):
        """biasedChoiceWorld_ephyssessions resolves to biased (strict CW match)."""
        session = pd.Series({
            'eid': 'eid-1',
            'task_protocol': '_iblrig_tasks_biasedChoiceWorld_ephyssessions7.0.4',
        })
        result = get_session_type(session)
        assert result['session_type'] == 'biased'

    def test_training_phase_choiceworld_raises(self):
        """trainingPhaseChoiceWorld is not matched by strict CW mask."""
        session = pd.Series({
            'eid': 'eid-1',
            'task_protocol': '_iblrig_tasks_trainingPhaseChoiceWorld8.29.0',
        })
        with pytest.raises(InvalidSessionType, match='does not match'):
            get_session_type(session)

    def test_exception_logged_when_exlog_provided(self):
        """Invalid protocol logs to exlog instead of raising."""
        session = pd.Series({
            'eid': 'eid-1',
            'task_protocol': 'nonsense_protocol',
        })
        exlog = []
        result = get_session_type(session, exlog=exlog)
        assert len(exlog) == 1
        assert exlog[0]['error_type'] == 'InvalidSessionType'


class TestGetTargetNM:
    """Tests for get_targetNM function."""

    def test_single_region(self):
        """Single brain region + NM produces valid target_NM."""
        session = pd.Series({
            'eid': 'eid-1',
            'NM': 'DA',
            'brain_region': ['VTA'],
        })
        result = get_targetNM(session)
        assert result['target_NM'] == ['VTA-DA']

    def test_multiple_regions(self):
        """Multiple brain regions produce multiple target_NMs."""
        session = pd.Series({
            'eid': 'eid-1',
            'NM': 'DA',
            'brain_region': ['VTA', 'SNc'],
        })
        result = get_targetNM(session)
        assert result['target_NM'] == ['VTA-DA', 'SNc-DA']

    def test_strips_hemisphere_suffix(self):
        """Hemisphere suffix (e.g., 'VTA-l') is stripped before combining."""
        session = pd.Series({
            'eid': 'eid-1',
            'NM': 'DA',
            'brain_region': ['VTA-l', 'SNc-r'],
        })
        result = get_targetNM(session)
        assert result['target_NM'] == ['VTA-DA', 'SNc-DA']

    def test_invalid_target_raises(self):
        """Unknown region-NM combo raises InvalidTargetNM."""
        session = pd.Series({
            'eid': 'eid-1',
            'NM': 'DA',
            'brain_region': ['XYZ'],
        })
        with pytest.raises(InvalidTargetNM, match='XYZ-DA'):
            get_targetNM(session)

    def test_exception_logged_when_exlog_provided(self):
        """Invalid target logs to exlog instead of raising."""
        session = pd.Series({
            'eid': 'eid-1',
            'NM': 'DA',
            'brain_region': ['XYZ'],
        })
        exlog = []
        result = get_targetNM(session, exlog=exlog)
        assert len(exlog) == 1
        assert exlog[0]['error_type'] == 'InvalidTargetNM'
        # target_NM should not be set (original series returned)
        assert 'target_NM' not in result or result.get('target_NM') is None

    def test_empty_brain_region(self):
        """Empty brain_region list produces empty target_NM."""
        session = pd.Series({
            'eid': 'eid-1',
            'NM': 'DA',
            'brain_region': [],
        })
        result = get_targetNM(session)
        assert result['target_NM'] == []


class TestCollectSessionErrors:
    @pytest.fixture
    def df_sessions(self):
        return pd.DataFrame({'eid': ['eid-1', 'eid-2', 'eid-3']})

    def test_adds_logged_errors_column(self, df_sessions, tmp_path):
        log = pd.DataFrame([
            {'eid': 'eid-1', 'error_type': 'InvalidNM', 'error_message': 'x', 'traceback': None},
        ])
        fpath = tmp_path / 'log.pqt'
        log.to_parquet(fpath)
        result = collect_session_errors(df_sessions, [fpath])
        assert 'logged_errors' in result.columns

    def test_errors_attached_to_correct_eid(self, df_sessions, tmp_path):
        log = pd.DataFrame([
            {'eid': 'eid-1', 'error_type': 'InvalidNM', 'error_message': 'x', 'traceback': None},
            {'eid': 'eid-1', 'error_type': 'InvalidStrain', 'error_message': 'y', 'traceback': None},
        ])
        fpath = tmp_path / 'log.pqt'
        log.to_parquet(fpath)
        result = collect_session_errors(df_sessions, [fpath])
        row = result[result['eid'] == 'eid-1'].iloc[0]
        assert set(row['logged_errors']) == {'InvalidNM', 'InvalidStrain'}

    def test_no_errors_gives_empty_list(self, df_sessions, tmp_path):
        log = pd.DataFrame(columns=['eid', 'error_type', 'error_message', 'traceback'])
        fpath = tmp_path / 'log.pqt'
        log.to_parquet(fpath)
        result = collect_session_errors(df_sessions, [fpath])
        assert result['logged_errors'].apply(lambda x: x == []).all()

    def test_missing_log_file_ignored(self, df_sessions, tmp_path):
        missing = tmp_path / 'nonexistent.pqt'
        result = collect_session_errors(df_sessions, [missing])
        assert 'logged_errors' in result.columns
        assert result['logged_errors'].apply(lambda x: x == []).all()

    def test_merges_multiple_log_files(self, df_sessions, tmp_path):
        log1 = pd.DataFrame([
            {'eid': 'eid-1', 'error_type': 'TypeA', 'error_message': '', 'traceback': None},
        ])
        log2 = pd.DataFrame([
            {'eid': 'eid-2', 'error_type': 'TypeB', 'error_message': '', 'traceback': None},
        ])
        p1, p2 = tmp_path / 'a.pqt', tmp_path / 'b.pqt'
        log1.to_parquet(p1); log2.to_parquet(p2)
        result = collect_session_errors(df_sessions, [p1, p2])
        assert 'TypeA' in result[result['eid'] == 'eid-1'].iloc[0]['logged_errors']
        assert 'TypeB' in result[result['eid'] == 'eid-2'].iloc[0]['logged_errors']
        assert result[result['eid'] == 'eid-3'].iloc[0]['logged_errors'] == []

    def test_accepts_dataframe_sources(self, df_sessions):
        log = pd.DataFrame([
            {'eid': 'eid-1', 'error_type': 'TypeError', 'error_message': 'msg', 'traceback': None},
        ])
        result = collect_session_errors(df_sessions, [log])
        assert 'TypeError' in result[result['eid'] == 'eid-1'].iloc[0]['logged_errors']
        assert result[result['eid'] == 'eid-2'].iloc[0]['logged_errors'] == []

    def test_deduplicates_across_sources(self, df_sessions):
        """Same (eid, error_type, error_message) from two sources → counted once."""
        log1 = pd.DataFrame([
            {'eid': 'eid-1', 'error_type': 'TypeError', 'error_message': 'msg', 'traceback': None},
        ])
        log2 = pd.DataFrame([
            {'eid': 'eid-1', 'error_type': 'TypeError', 'error_message': 'msg', 'traceback': None},
        ])
        result = collect_session_errors(df_sessions, [log1, log2])
        assert result[result['eid'] == 'eid-1'].iloc[0]['logged_errors'].count('TypeError') == 1


class TestDeduplicateLog:
    def test_drops_duplicate_entries(self):
        df = pd.DataFrame({
            'eid': ['a', 'a', 'b'],
            'error_type': ['TypeError', 'TypeError', 'KeyError'],
            'error_message': ['msg', 'msg', 'other'],
            'traceback': [None, None, None],
        })
        result = deduplicate_log(df)
        assert len(result) == 2

    def test_keeps_different_messages_same_type(self):
        """Same eid+type but different message → both kept."""
        df = pd.DataFrame({
            'eid': ['a', 'a'],
            'error_type': ['TypeError', 'TypeError'],
            'error_message': ['msg1', 'msg2'],
            'traceback': [None, None],
        })
        result = deduplicate_log(df)
        assert len(result) == 2

    def test_keeps_same_message_different_eid(self):
        """Same type+message but different eid → both kept."""
        df = pd.DataFrame({
            'eid': ['a', 'b'],
            'error_type': ['TypeError', 'TypeError'],
            'error_message': ['msg', 'msg'],
            'traceback': [None, None],
        })
        result = deduplicate_log(df)
        assert len(result) == 2

    def test_empty_df_returns_empty(self):
        df = pd.DataFrame(columns=LOG_COLUMNS)
        result = deduplicate_log(df)
        assert len(result) == 0

    def test_none_returns_none(self):
        assert deduplicate_log(None) is None
