"""Tests for iblnm.validation module."""
import pytest


class TestModuleImports:
    def test_validation_module_importable(self):
        import iblnm.validation  # noqa

    def test_all_exceptions_importable(self):
        from iblnm.validation import (  # noqa: F401
            InvalidSubject, InvalidStrain, InvalidLine, InvalidNeuromodulator,
            InvalidBrainRegion, MissingBrainRegion, MissingHemisphere,
            HemisphereMismatch, MissingInsertion, MissingHemiSuffix,
            DataNotListed, InvalidSessionType, InvalidTargetNM, InvalidSessionLength,
            TrueDuplicateSession, MissingExtractedData, MissingRawData,
            InsufficientTrials, BlockStructureBug, IncompleteEventTimes,
            TrialsNotInPhotometryTime, BandInversion, EarlySamples,
            QCValidationError,
        )

    def test_video_exceptions_importable(self):
        from iblnm.validation import (  # noqa: F401
            MissingVideoTimestamps, VideoLengthError,
        )

    def test_validate_functions_importable(self):
        from iblnm.validation import (  # noqa: F401
            validate_subject, validate_strain, validate_line,
            validate_neuromodulator, validate_brain_region, validate_hemisphere,
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


class TestValidateBrainRegion:
    def test_raises_for_invalid_region(self):
        import pandas as pd
        from iblnm.validation import validate_brain_region, InvalidBrainRegion
        session = pd.Series({'eid': 'e', 'subject': 's',
                             'brain_region': ['__invalid_region__']})
        with pytest.raises(InvalidBrainRegion):
            validate_brain_region(session)

    def test_returns_none_for_valid_regions(self):
        import pandas as pd
        from iblnm.validation import validate_brain_region
        from iblnm.config import VALID_TARGETS
        session = pd.Series({'eid': 'e', 'subject': 's',
                             'brain_region': [next(iter(VALID_TARGETS))]})
        assert validate_brain_region(session) is None

    def test_raises_for_empty_brain_region(self):
        import pandas as pd
        from iblnm.validation import validate_brain_region, MissingBrainRegion
        session = pd.Series({'eid': 'e', 'subject': 's', 'brain_region': []})
        with pytest.raises(MissingBrainRegion):
            validate_brain_region(session)


class TestValidateHemisphere:
    def test_raises_for_empty_hemisphere(self):
        import pandas as pd
        from iblnm.validation import validate_hemisphere, MissingHemisphere
        session = pd.Series({'eid': 'e', 'subject': 's',
                             'brain_region': ['VTA'], 'hemisphere': []})
        with pytest.raises(MissingHemisphere):
            validate_hemisphere(session)
