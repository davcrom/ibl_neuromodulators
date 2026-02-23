"""Tests for iblnm.validation module."""
import pytest


class TestModuleImports:
    def test_validation_module_importable(self):
        import iblnm.validation  # noqa

    def test_all_exceptions_importable(self):
        from iblnm.validation import (
            InvalidSubject, InvalidStrain, InvalidLine, InvalidNeuromodulator,
            InvalidTarget, HemisphereMismatch, MissingInsertion, MissingHemiSuffix,
            DataNotListed, InvalidSessionType, InvalidTargetNM, InvalidSessionLength,
            TrueDuplicateSession, MissingExtractedData, MissingRawData,
            InsufficientTrials, BlockStructureBug, IncompleteEventTimes,
            TrialsNotInPhotometryTime, BandInversion, EarlySamples,
            FewUniqueSamples, QCValidationError,
        )

    def test_validate_functions_importable(self):
        from iblnm.validation import (
            validate_subject, validate_strain, validate_line,
            validate_neuromodulator, validate_target, validate_hemisphere,
            validate_datasets,
        )

    def test_exception_logger_importable(self):
        from iblnm.validation import exception_logger

    def test_make_log_entry_importable(self):
        from iblnm.validation import make_log_entry


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


class TestUtilBackwardCompat:
    """exception_logger and make_log_entry are re-exported from util for backward compat."""

    def test_exception_logger_importable_from_util(self):
        from iblnm.util import exception_logger  # noqa

    def test_make_log_entry_importable_from_util(self):
        from iblnm.util import make_log_entry  # noqa


class TestDataBackwardCompat:
    """Data exceptions remain importable from iblnm.data for backward compat."""

    def test_exceptions_importable_from_data(self):
        from iblnm.data import (
            MissingExtractedData, MissingRawData, InsufficientTrials,
            BlockStructureBug, IncompleteEventTimes, TrialsNotInPhotometryTime,
            BandInversion, EarlySamples, FewUniqueSamples, QCValidationError,
        )
