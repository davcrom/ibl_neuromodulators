"""Tests for iblnm.validation module."""
import numpy as np
import pandas as pd
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
            FewUniqueSamples, QCValidationError,
        )

    def test_video_exceptions_importable(self):
        from iblnm.validation import (  # noqa: F401
            MissingVideoTimestamps, VideoLengthError,
            VideoTimestampsQCError, VideoDroppedFramesQCError,
            VideoPinStateQCError,
        )

    def test_validate_functions_importable(self):
        from iblnm.validation import (  # noqa: F401
            validate_subject, validate_strain, validate_line,
            validate_neuromodulator, validate_brain_region, validate_hemisphere,
            validate_datasets,
        )

    def test_video_validate_functions_importable(self):
        from iblnm.validation import (  # noqa: F401
            validate_video_length,
            validate_video_timestamps_qc,
            validate_video_dropped_frames_qc,
            validate_video_pin_state_qc,
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


class TestValidateVideoLength:
    def test_raises_when_discrepancy_exceeds_threshold(self):
        from iblnm.validation import validate_video_length, VideoLengthError
        session = pd.Series({'eid': 'e1', 'length_discrepancy': 200.0})
        with pytest.raises(VideoLengthError):
            validate_video_length(session)

    def test_returns_none_when_within_threshold(self):
        from iblnm.validation import validate_video_length
        session = pd.Series({'eid': 'e1', 'length_discrepancy': 50.0})
        assert validate_video_length(session) is None

    def test_returns_none_when_nan(self):
        from iblnm.validation import validate_video_length
        session = pd.Series({'eid': 'e1', 'length_discrepancy': np.nan})
        assert validate_video_length(session) is None

    def test_raises_at_exact_threshold(self):
        from iblnm.validation import validate_video_length, VideoLengthError
        from iblnm.config import LENGTH_MISMATCH_THRESHOLD
        session = pd.Series({
            'eid': 'e1',
            'length_discrepancy': float(LENGTH_MISMATCH_THRESHOLD),
        })
        with pytest.raises(VideoLengthError):
            validate_video_length(session)

    def test_logs_when_exlog_provided(self):
        from iblnm.validation import validate_video_length
        session = pd.Series({'eid': 'e1', 'length_discrepancy': 200.0})
        exlog = []
        validate_video_length(session, exlog=exlog)
        assert len(exlog) == 1
        assert exlog[0]['error_type'] == 'VideoLengthError'


class TestValidateVideoTimestampsQC:
    def test_raises_for_fail(self):
        from iblnm.validation import validate_video_timestamps_qc, VideoTimestampsQCError
        session = pd.Series({'eid': 'e1', 'qc_videoLeft_timestamps': 'FAIL'})
        with pytest.raises(VideoTimestampsQCError):
            validate_video_timestamps_qc(session)

    def test_raises_for_nan(self):
        from iblnm.validation import validate_video_timestamps_qc, VideoTimestampsQCError
        session = pd.Series({'eid': 'e1', 'qc_videoLeft_timestamps': np.nan})
        with pytest.raises(VideoTimestampsQCError):
            validate_video_timestamps_qc(session)

    def test_raises_for_not_set(self):
        from iblnm.validation import validate_video_timestamps_qc, VideoTimestampsQCError
        session = pd.Series({'eid': 'e1', 'qc_videoLeft_timestamps': 'NOT_SET'})
        with pytest.raises(VideoTimestampsQCError):
            validate_video_timestamps_qc(session)

    def test_returns_none_for_pass(self):
        from iblnm.validation import validate_video_timestamps_qc
        session = pd.Series({'eid': 'e1', 'qc_videoLeft_timestamps': 'PASS'})
        assert validate_video_timestamps_qc(session) is None

    def test_logs_when_exlog_provided(self):
        from iblnm.validation import validate_video_timestamps_qc
        session = pd.Series({'eid': 'e1', 'qc_videoLeft_timestamps': 'CRITICAL'})
        exlog = []
        validate_video_timestamps_qc(session, exlog=exlog)
        assert len(exlog) == 1
        assert exlog[0]['error_type'] == 'VideoTimestampsQCError'


class TestValidateVideoDroppedFramesQC:
    def test_raises_for_fail(self):
        from iblnm.validation import validate_video_dropped_frames_qc, VideoDroppedFramesQCError
        session = pd.Series({'eid': 'e1', 'qc_videoLeft_dropped_frames': 'FAIL'})
        with pytest.raises(VideoDroppedFramesQCError):
            validate_video_dropped_frames_qc(session)

    def test_returns_none_for_pass(self):
        from iblnm.validation import validate_video_dropped_frames_qc
        session = pd.Series({'eid': 'e1', 'qc_videoLeft_dropped_frames': 'PASS'})
        assert validate_video_dropped_frames_qc(session) is None

    def test_logs_when_exlog_provided(self):
        from iblnm.validation import validate_video_dropped_frames_qc
        session = pd.Series({'eid': 'e1', 'qc_videoLeft_dropped_frames': 'WARNING'})
        exlog = []
        validate_video_dropped_frames_qc(session, exlog=exlog)
        assert len(exlog) == 1
        assert exlog[0]['error_type'] == 'VideoDroppedFramesQCError'


class TestValidateVideoPinStateQC:
    def test_raises_for_fail(self):
        from iblnm.validation import validate_video_pin_state_qc, VideoPinStateQCError
        session = pd.Series({'eid': 'e1', 'qc_videoLeft_pin_state': 'FAIL'})
        with pytest.raises(VideoPinStateQCError):
            validate_video_pin_state_qc(session)

    def test_returns_none_for_pass(self):
        from iblnm.validation import validate_video_pin_state_qc
        session = pd.Series({'eid': 'e1', 'qc_videoLeft_pin_state': 'PASS'})
        assert validate_video_pin_state_qc(session) is None

    def test_logs_when_exlog_provided(self):
        from iblnm.validation import validate_video_pin_state_qc
        session = pd.Series({'eid': 'e1', 'qc_videoLeft_pin_state': 'CRITICAL'})
        exlog = []
        validate_video_pin_state_qc(session, exlog=exlog)
        assert len(exlog) == 1
        assert exlog[0]['error_type'] == 'VideoPinStateQCError'


