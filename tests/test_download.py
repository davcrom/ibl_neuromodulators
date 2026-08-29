"""Tests for scripts/download.py — catalog fixups and per-session building."""
from unittest.mock import MagicMock

import pandas as pd
import pytest

import scripts.download as download


@pytest.fixture
def fake_ps():
    """A PhotometrySession mock whose stored products are all absent.

    ``product_status`` is driven by the ``stored`` dict, so a test names only
    the products it wants reported as 'current' or 'stale'.
    """
    ps = MagicMock()
    ps.eid = 'test-eid'
    ps.errors = []
    ps.rebuild = set()
    ps.stored = {}
    ps.product_status.side_effect = lambda product: ps.stored.get(product, 'absent')
    return ps


class TestBuildCatalog:
    """Phase one: the cross-session fixups run over the whole catalog."""

    def test_fixups_ranking_and_schema(self):
        """Empty lists fill from the subject, names are fixed, days are ranked."""
        sessions = pd.DataFrame({
            'eid': ['a', 'b'],
            'subject': ['M1', 'M1'],
            'start_time': ['2024-01-01T10:00:00', '2024-01-03T10:00:00'],
            'brain_region': [['SNC'], []],
            'hemisphere': [['l'], []],
        })

        catalog = download.build_catalog(sessions)

        assert list(catalog['brain_region']) == [['SNc'], ['SNc']]
        assert list(catalog['hemisphere']) == [['l'], ['l']]
        assert list(catalog['target_NM']) == [['SNc-DA'], ['SNc-DA']]
        assert list(catalog['day_n']) == [0, 2]
        assert list(catalog['session_n']) == [1, 2]
        # enforce_schema fills every catalogued column, so the frame is usable
        # by PhotometrySessionGroup without further patching.
        assert 'session_type' in catalog.columns
        assert 'date' not in catalog.columns


class TestBuildSession:
    """Phase two: which products one session builds, and which it skips."""

    def test_current_products_are_not_rebuilt(self, fake_ps):
        """A session whose products are all stored current calls no loader."""
        fake_ps.stored = {product: 'current' for product in download.PRODUCT_BUILDERS}

        built = download.build_session(fake_ps)

        assert built == {}
        fake_ps.load_trials.assert_not_called()
        fake_ps.load_photometry.assert_not_called()
        fake_ps.load_responses.assert_not_called()

    def test_absent_products_are_built(self, fake_ps):
        """Nothing stored: every product in the table is built, in order."""
        built = download.build_session(fake_ps)

        assert built == {product: 'built' for product in download.PRODUCT_BUILDERS}
        fake_ps.load_trials.assert_called_once()
        fake_ps.load_photometry.assert_called_once()
        assert [call.args[0] for call in fake_ps.load_responses.call_args_list] == [
            'photometry', 'wheel', 'video']

    def test_skipped_product_is_not_built(self, fake_ps):
        """--skip drops that product and leaves the rest of the table alone."""
        built = download.build_session(
            fake_ps, skip=download.with_dependents({'photometry/responses'}))

        assert 'photometry/responses' not in built
        assert built['photometry/preprocessed'] == 'built'
        assert [call.args[0] for call in fake_ps.load_responses.call_args_list] == [
            'wheel', 'video']

    def test_skip_reaches_dependents(self, fake_ps):
        """Skipping a raw product skips everything built from it."""
        skip = download.with_dependents({'video/pose'})

        built = download.build_session(fake_ps, skip=skip)

        assert 'video/pose/qc' not in built
        assert 'video/responses' not in built
        # The camera clock is a separate raw product, so its QC still runs.
        assert built['video/times/qc'] == 'built'
        assert built['photometry/responses'] == 'built'

    def test_failed_build_blocks_its_dependents(self, fake_ps):
        """A raising builder is logged against its product; dependents stop."""
        fake_ps.load_photometry.side_effect = ValueError('no signal')

        built = download.build_session(fake_ps)

        assert built['photometry/preprocessed'] == 'failed'
        assert 'photometry/responses' not in built
        fake_ps.log_error.assert_called_once()
        assert fake_ps.log_error.call_args.kwargs == {
            'product': 'photometry/preprocessed'}

    def test_recorded_failure_is_not_retried(self, fake_ps):
        """Absent data beside a recorded error: skip the product and its dependents."""
        fake_ps.errors = [{'product': 'video/pose', 'error_type': 'MissingLP'}]

        built = download.build_session(fake_ps)

        assert 'video/pose/qc' not in built
        assert 'video/responses' not in built
        assert built['video/times/qc'] == 'built'

    def test_retry_failed_reattempts_them(self, fake_ps):
        """--retry-failed ignores the recorded failure and builds anyway."""
        fake_ps.errors = [{'product': 'video/pose', 'error_type': 'MissingLP'}]

        built = download.build_session(fake_ps, retry_failed=True)

        assert built['video/pose/qc'] == 'built'
        assert built['video/responses'] == 'built'

    def test_error_beside_stored_data_is_informational(self, fake_ps):
        """An error logged for a product that still has data blocks nothing."""
        fake_ps.errors = [{'product': 'video/pose', 'error_type': 'ValueError'}]
        fake_ps.stored = {'video/pose': 'current'}

        built = download.build_session(fake_ps)

        assert built['video/pose/qc'] == 'built'
        assert built['video/responses'] == 'built'


class TestBuildSessionOnStoredFile:
    """The same gate against a real PhotometrySession and a real H5 file."""

    def test_stored_product_is_not_refetched(self, tmp_path):
        """A stamped `trials/table` is left alone, so nothing touches Alyx."""
        from iblnm.config import PRODUCT_SPEC
        from iblnm.data import PhotometrySession

        one = MagicMock()
        session = pd.Series({
            'eid': 'test-eid-123', 'subject': 'test_mouse', 'number': 1,
            'start_time': '2024-01-01T10:00:00', 'session_type': 'training',
            'task_protocol': 'test_protocol', 'lab': 'test_lab',
        })
        ps = PhotometrySession(session, one=one, load_data=False)
        ps.filepath = tmp_path / f'{ps.eid}.h5'
        ps.trials = pd.DataFrame({'trial': [0, 1], 'choice': [1, -1]})
        ps.save_h5(groups=['trials'])
        one.reset_mock()  # the loader base class resolves a path at construction

        built = download.build_session(
            ps, skip=set(PRODUCT_SPEC) - {'trials/table'})

        assert built == {}
        assert one.mock_calls == []


@pytest.fixture
def fake_group(request):
    """A PhotometrySessionGroup mock over three sessions with a status scan.

    ``request.param`` (optional) maps a product to the status reported for
    every session; unnamed products come back 'absent'.
    """
    stored = getattr(request, 'param', {})
    group = MagicMock()
    group.rebuild = set()
    group.scan_product_status.side_effect = lambda *products: pd.DataFrame(
        {'eid': ['a', 'b', 'c'],
         **{product: [stored.get(product, 'absent')] * 3 for product in products}}
    )
    return group


class TestBuildStore:
    """The group-level pass: report the store, then build what is missing."""

    @pytest.mark.parametrize(
        'fake_group', [{'photometry/preprocessed': 'current'}], indirect=True)
    def test_rebuild_is_reported_before_any_build(self, fake_group, capsys):
        """The blast radius is printed before `process` is given the work."""
        fake_group.process.side_effect = RuntimeError('built too early')

        with pytest.raises(RuntimeError):
            download.build_store(fake_group, rebuild={'photometry/preprocessed'})

        assert '3 sessions' in capsys.readouterr().out

    @pytest.mark.parametrize(
        'fake_group', [{'photometry/preprocessed': 'current'}], indirect=True)
    def test_rebuild_set_reaches_the_group(self, fake_group):
        """Named products and their dependents are what the group rebuilds."""
        download.build_store(fake_group, rebuild={'photometry/preprocessed'})

        assert fake_group.rebuild == {'photometry/preprocessed',
                                      'photometry/responses'}
        assert fake_group.process.call_args.args[0] is download.build_session

    @pytest.mark.parametrize(
        'fake_group', [{'wheel/preprocessed': 'stale'}], indirect=True)
    def test_stale_product_aborts_the_run(self, fake_group):
        """A stamp disagreeing with config.py stops the run instead of rebuilding."""
        with pytest.raises(SystemExit, match='wheel/preprocessed'):
            download.build_store(fake_group)

        fake_group.process.assert_not_called()

    @pytest.mark.parametrize(
        'fake_group', [{'wheel/preprocessed': 'stale'}], indirect=True)
    def test_rebuilding_a_stale_product_is_allowed(self, fake_group):
        """--rebuild is how the user accepts the config.py change."""
        download.build_store(fake_group, rebuild={'wheel/preprocessed'})

        fake_group.process.assert_called_once()


class TestParseArgs:
    def test_unknown_product_is_rejected(self):
        """Both product flags validate at parse time, not silently at build time."""
        with pytest.raises(SystemExit):
            download.parse_args(['--skip', 'photometry/nonsense'])
        with pytest.raises(SystemExit):
            download.parse_args(['--rebuild', 'photometry/nonsense'])

    def test_products_and_defaults(self):
        args = download.parse_args(['--skip', 'video/pose', '--workers', '4'])

        assert args.skip == ['video/pose']
        assert args.rebuild == []
        assert args.workers == 4
        assert args.retry_failed is False
