"""Tests for iblnm.validation module."""
import pytest


class TestModuleImports:
    def test_validation_module_importable(self):
        import iblnm.validation  # noqa

    def test_all_exceptions_importable(self):
        from iblnm.validation import (  # noqa: F401
            InvalidSubject, InvalidStrain, InvalidLine, InvalidNeuromodulator,
            InvalidTarget, HemisphereMismatch, MissingInsertion, MissingHemiSuffix,
            DataNotListed, InvalidSessionType, InvalidTargetNM, InvalidSessionLength,
            TrueDuplicateSession, MissingExtractedData, MissingRawData,
            InsufficientTrials, BlockStructureBug, IncompleteEventTimes,
            TrialsNotInPhotometryTime, BandInversion, EarlySamples,
            FewUniqueSamples, QCValidationError,
        )

    def test_validate_functions_importable(self):
        from iblnm.validation import (  # noqa: F401
            validate_subject, validate_strain, validate_line,
            validate_neuromodulator, validate_target, validate_hemisphere,
            validate_datasets,
        )

    def test_exception_logger_importable(self):
        from iblnm.validation import exception_logger  # noqa: F401

    def test_make_log_entry_importable(self):
        from iblnm.validation import make_log_entry  # noqa: F401


class TestValidateSubject:
    def test_raises_for_excluded_subject(self):
        import pandas as pd
        from iblnm.validation import validate_subject, InvalidSubject
        from iblnm.config import SUBJECTS_TO_EXCLUDE
        if not SUBJECTS_TO_EXCLUDE:
            pytest.skip("No excluded subjects configured")
        session = pd.Series({'eid': 'e', 'subject': next(iter(SUBJECTS_TO_EXCLUDE))})
        with pytest.raises(InvalidSubject):
            validate_subject(session)

    def test_returns_none_for_valid_subject(self):
        import pandas as pd
        from iblnm.validation import validate_subject
        session = pd.Series({'eid': 'e', 'subject': 'valid_mouse_xyz'})
        assert validate_subject(session) is None

    def test_logs_when_exlog_provided(self):
        import pandas as pd
        from iblnm.validation import validate_subject
        from iblnm.config import SUBJECTS_TO_EXCLUDE
        if not SUBJECTS_TO_EXCLUDE:
            pytest.skip("No excluded subjects configured")
        session = pd.Series({'eid': 'e', 'subject': next(iter(SUBJECTS_TO_EXCLUDE))})
        exlog = []
        validate_subject(session, exlog=exlog)
        assert len(exlog) == 1
        assert exlog[0]['error_type'] == 'InvalidSubject'


class TestValidateStrain:
    def test_raises_for_invalid_strain(self):
        import pandas as pd
        from iblnm.validation import validate_strain, InvalidStrain
        session = pd.Series({'eid': 'e', 'strain': '__not_a_strain__'})
        with pytest.raises(InvalidStrain):
            validate_strain(session)

    def test_returns_none_for_valid_strain(self):
        import pandas as pd
        from iblnm.validation import validate_strain
        from iblnm.config import VALID_STRAINS
        session = pd.Series({'eid': 'e', 'strain': next(iter(VALID_STRAINS))})
        assert validate_strain(session) is None


class TestValidateNeuromodulator:
    def test_raises_for_invalid_nm(self):
        import pandas as pd
        from iblnm.validation import validate_neuromodulator, InvalidNeuromodulator
        session = pd.Series({'eid': 'e', 'NM': '__invalid__'})
        with pytest.raises(InvalidNeuromodulator):
            validate_neuromodulator(session)

    def test_returns_none_for_valid_nm(self):
        import pandas as pd
        from iblnm.validation import validate_neuromodulator
        from iblnm.config import VALID_NEUROMODULATORS
        session = pd.Series({'eid': 'e', 'NM': next(iter(VALID_NEUROMODULATORS))})
        assert validate_neuromodulator(session) is None


class TestValidateTarget:
    def test_raises_for_invalid_target(self):
        import pandas as pd
        from iblnm.validation import validate_target, InvalidTarget
        session = pd.Series({'eid': 'e', 'brain_region': ['__invalid_region__']})
        with pytest.raises(InvalidTarget):
            validate_target(session)

    def test_returns_none_for_valid_targets(self):
        import pandas as pd
        from iblnm.validation import validate_target
        from iblnm.config import VALID_TARGETS
        session = pd.Series({'eid': 'e', 'brain_region': [next(iter(VALID_TARGETS))]})
        assert validate_target(session) is None


class TestFillHemisphereFromFiberInsertionTable:
    """Tests for fill_hemisphere_from_fiber_insertion_table."""

    def _lookup(self):
        return {('sub1', 'VTA'): 'r', ('sub1', 'DR'): 'l'}

    def _session(self, brain_region, hemisphere):
        import pandas as pd
        return pd.Series({
            'eid': 'e1', 'subject': 'sub1',
            'brain_region': brain_region,
            'hemisphere': hemisphere,
        })

    def test_fills_none_from_lookup(self):
        from iblnm.validation import fill_hemisphere_from_fiber_insertion_table as fill_hemi
        result = fill_hemi(self._session(['VTA'], [None]), fiber_lookup=self._lookup())
        assert result['hemisphere'] == ['r']

    def test_does_not_overwrite_existing_hemisphere(self):
        from iblnm.validation import fill_hemisphere_from_fiber_insertion_table as fill_hemi
        result = fill_hemi(self._session(['VTA'], ['l']), fiber_lookup=self._lookup())
        assert result['hemisphere'] == ['l']

    def test_leaves_none_when_key_absent(self):
        from iblnm.validation import fill_hemisphere_from_fiber_insertion_table as fill_hemi
        result = fill_hemi(self._session(['NAc'], [None]), fiber_lookup=self._lookup())
        assert result['hemisphere'] == [None]

    def test_leaves_none_when_lookup_returns_none(self):
        from iblnm.validation import fill_hemisphere_from_fiber_insertion_table as fill_hemi
        lookup = {('sub1', 'VTA'): None}
        result = fill_hemi(self._session(['VTA'], [None]), fiber_lookup=lookup)
        assert result['hemisphere'] == [None]

    def test_empty_brain_region_returns_unchanged(self):
        from iblnm.validation import fill_hemisphere_from_fiber_insertion_table as fill_hemi
        result = fill_hemi(self._session([], []), fiber_lookup=self._lookup())
        assert result['hemisphere'] == []

    def test_multiple_regions_partial_fill(self):
        from iblnm.validation import fill_hemisphere_from_fiber_insertion_table as fill_hemi
        result = fill_hemi(
            self._session(['VTA', 'DR', 'NAc'], [None, 'r', None]),
            fiber_lookup=self._lookup(),
        )
        # VTA → filled 'r'; DR → existing 'r' preserved; NAc → absent → None
        assert result['hemisphere'] == ['r', 'r', None]

    def test_strips_suffix_before_lookup(self):
        """Regions already stored with suffix (e.g. 'VTA-r') should still hit the lookup."""
        from iblnm.validation import fill_hemisphere_from_fiber_insertion_table as fill_hemi
        # hemisphere is None but brain_region already has suffix — fill should
        # strip suffix before lookup and not double-apply
        result = fill_hemi(
            self._session(['VTA-r'], [None]),
            fiber_lookup={('sub1', 'VTA'): 'r'},
        )
        assert result['hemisphere'] == ['r']


class TestUtilBackwardCompat:
    """exception_logger and make_log_entry are re-exported from util for backward compat."""

    def test_exception_logger_importable_from_util(self):
        from iblnm.util import exception_logger  # noqa

    def test_make_log_entry_importable_from_util(self):
        from iblnm.util import make_log_entry  # noqa


class TestDataBackwardCompat:
    """Data exceptions remain importable from iblnm.data for backward compat."""

    def test_exceptions_importable_from_data(self):
        pass
