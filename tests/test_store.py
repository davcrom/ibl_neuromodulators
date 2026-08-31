"""Tests for iblnm/store.py — the per-session and group-level product builds.

Also covers the store flags the analysis scripts share, since `--rebuild` and
the pre-warm product list mean the same thing in every one of them.
"""
import importlib
from unittest.mock import MagicMock

import pandas as pd
import pytest

from iblnm import store
from iblnm.validation import StaleProduct

# Every script that pre-warms the store, mapped to the arguments it cannot run
# without, so the store flags can be appended to a parsable command line.
ANALYSIS_SCRIPTS = {
    'baseline': ['--model', 'performance'],
    'collect_trials': [],
    'ddm_hmm_overview': [],
    'encoding': ['some-eid', 'SNc'],
    'example_session': [],
    'responses': [],
    'task_encoding': [],
    'task_performance': [],
}


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


class TestBuildSession:
    """Phase two: which products one session builds, and which it skips."""

    def test_current_products_are_not_rebuilt(self, fake_ps):
        """A session whose products are all stored current calls no loader."""
        fake_ps.stored = {product: 'current' for product in store.PRODUCT_BUILDERS}

        built = store.build_session(fake_ps)

        assert built == {}
        fake_ps.load_trials.assert_not_called()
        fake_ps.load_photometry.assert_not_called()
        fake_ps.load_responses.assert_not_called()

    def test_absent_products_are_built(self, fake_ps):
        """Nothing stored: every product in the table is built, in order."""
        built = store.build_session(fake_ps)

        assert built == {product: 'built' for product in store.PRODUCT_BUILDERS}
        fake_ps.load_trials.assert_called_once()
        fake_ps.load_photometry.assert_called_once()
        assert [call.args[0] for call in fake_ps.load_responses.call_args_list] == [
            'photometry', 'wheel', 'video']

    def test_product_outside_the_scope_is_not_built(self, fake_ps):
        """A product left out of `products` is skipped; the rest still build."""
        built = store.build_session(
            fake_ps,
            products=set(store.ALL_PRODUCTS) - {'photometry/responses'})

        assert 'photometry/responses' not in built
        assert built['photometry/preprocessed'] == 'built'
        assert [call.args[0] for call in fake_ps.load_responses.call_args_list] == [
            'wheel', 'video']

    def test_failed_build_blocks_its_dependents(self, fake_ps):
        """A raising builder is logged against its product; dependents stop."""
        fake_ps.load_photometry.side_effect = ValueError('no signal')

        built = store.build_session(fake_ps)

        assert built['photometry/preprocessed'] == 'failed'
        assert 'photometry/responses' not in built
        fake_ps.log_error.assert_called_once()
        assert fake_ps.log_error.call_args.kwargs == {
            'product': 'photometry/preprocessed'}

    def test_recorded_failure_is_not_retried(self, fake_ps):
        """Absent data beside a recorded error: skip the product and its dependents."""
        fake_ps.errors = [{'product': 'video/pose', 'error_type': 'MissingLP'}]

        built = store.build_session(fake_ps)

        assert 'video/pose/qc' not in built
        assert 'video/responses' not in built
        assert built['video/times/qc'] == 'built'

    def test_retry_failed_reattempts_them(self, fake_ps):
        """--retry-failed ignores the recorded failure and builds anyway."""
        fake_ps.errors = [{'product': 'video/pose', 'error_type': 'MissingLP'}]

        built = store.build_session(fake_ps, retry_failed=True)

        assert built['video/pose/qc'] == 'built'
        assert built['video/responses'] == 'built'

    def test_error_beside_stored_data_is_informational(self, fake_ps):
        """An error logged for a product that still has data blocks nothing."""
        fake_ps.errors = [{'product': 'video/pose', 'error_type': 'ValueError'}]
        fake_ps.stored = {'video/pose': 'current'}

        built = store.build_session(fake_ps)

        assert built['video/pose/qc'] == 'built'
        assert built['video/responses'] == 'built'


class TestBuildSessionOnStoredFile:
    """The same gate against a real PhotometrySession and a real H5 file."""

    def test_stored_product_is_not_refetched(self, tmp_path):
        """A stamped `trials/table` is left alone, so nothing touches Alyx."""
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

        built = store.build_session(ps, products={'trials/table'})

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
            store.build_store(fake_group, rebuild={'photometry/preprocessed'})

        assert '3 sessions' in capsys.readouterr().out

    @pytest.mark.parametrize(
        'fake_group', [{'photometry/preprocessed': 'current'}], indirect=True)
    def test_rebuild_set_reaches_the_group(self, fake_group):
        """Named products and their dependents are what the group rebuilds."""
        store.build_store(fake_group, rebuild={'photometry/preprocessed'})

        assert fake_group.rebuild == {'photometry/preprocessed',
                                      'photometry/responses'}
        assert fake_group.process.call_args.args[0] is store.build_session

    @pytest.mark.parametrize(
        'fake_group', [{'wheel/preprocessed': 'stale'}], indirect=True)
    def test_stale_product_aborts_the_run(self, fake_group):
        """A stamp disagreeing with config.py stops the run instead of rebuilding."""
        with pytest.raises(StaleProduct, match='wheel/preprocessed'):
            store.build_store(fake_group)

        fake_group.process.assert_not_called()

    @pytest.mark.parametrize(
        'fake_group', [{'wheel/preprocessed': 'stale'}], indirect=True)
    def test_stale_product_outside_the_scope_is_ignored(self, fake_group):
        """A pre-warm surveys only the products its analysis reads."""
        store.build_store(fake_group, products=['trials/table'])

        assert fake_group.process.call_args.kwargs['products'] == {'trials/table'}

    def test_skip_reaches_dependents(self, fake_group):
        """Skipping a raw product drops everything built from it."""
        store.build_store(fake_group, skip={'video/pose'})

        products = fake_group.process.call_args.kwargs['products']
        assert {'video/pose/qc', 'video/responses'}.isdisjoint(products)
        # The camera clock is a separate raw product, so its QC still runs.
        assert {'video/times/qc', 'photometry/responses'} <= products

    @pytest.mark.parametrize(
        'fake_group', [{'wheel/preprocessed': 'stale'}], indirect=True)
    def test_rebuilding_a_stale_product_is_allowed(self, fake_group):
        """--rebuild is how the user accepts the config.py change."""
        store.build_store(fake_group, rebuild={'wheel/preprocessed'})

        fake_group.process.assert_called_once()


@pytest.mark.parametrize('script', sorted(ANALYSIS_SCRIPTS))
class TestAnalysisScriptStoreFlags:
    """Every analysis script takes `--rebuild` and names what it pre-warms."""

    def test_rebuild_reaches_the_parsed_arguments(self, script):
        """A named product is carried through to the pre-warm unchanged."""
        module = importlib.import_module(f'scripts.{script}')

        args = module.parse_args(
            ANALYSIS_SCRIPTS[script] + ['--rebuild', 'photometry/responses'])

        assert args.rebuild == ['photometry/responses']

    def test_unknown_product_is_rejected_at_parse_time(self, script):
        """A mistyped product fails before any session is touched."""
        module = importlib.import_module(f'scripts.{script}')

        with pytest.raises(SystemExit):
            module.parse_args(
                ANALYSIS_SCRIPTS[script] + ['--rebuild', 'photometry/nonsense'])

    def test_required_products_are_buildable(self, script):
        """The pre-warm list names products `store` knows how to build."""
        module = importlib.import_module(f'scripts.{script}')

        assert set(module.REQUIRED_PRODUCTS) <= set(store.ALL_PRODUCTS)
        assert module.REQUIRED_PRODUCTS


class TestStatusReport:
    """The printed survey is what `scan_product_status` found, not a re-count."""

    def test_counts_match_the_scan(self):
        """Each verdict is reported with the number of sessions holding it."""
        status = pd.DataFrame({
            'eid': ['a', 'b', 'c', 'd'],
            'trials/table': ['current', 'current', 'stale', 'absent'],
        })

        report = store.status_report(status)

        counts = status['trials/table'].value_counts()
        assert set(counts.index) == {'current', 'stale', 'absent'}
        assert all(f'{count} {verdict}' in report
                   for verdict, count in counts.items())


class TestPreWarmOverAStore:
    """The pre-warm against real H5 files: build the gap, then loop over it."""

    @staticmethod
    def _catalog(*eids):
        return pd.DataFrame([
            {'eid': eid, 'subject': 'test_mouse', 'number': 1,
             'start_time': '2024-01-01T10:00:00', 'session_type': 'training',
             'task_protocol': 'test_protocol', 'lab': 'test_lab'}
            for eid in eids])

    def test_absent_product_is_built_before_the_loop(self, tmp_path, monkeypatch):
        """One session holds the product, the other gains it; both then read back."""
        from iblnm.data import PhotometrySessionGroup

        def build_trials(ps):
            ps.trials = pd.DataFrame({'trial': [0, 1], 'choice': [1, -1]})
            ps.save_h5(groups=['trials'])

        monkeypatch.setattr(store, 'PRODUCT_BUILDERS', {'trials/table': build_trials})
        group = PhotometrySessionGroup.from_catalog(
            self._catalog('stored-eid', 'empty-eid'), one=None, h5_dir=tmp_path,
            scan_h5_errors=False)
        stored = group._get_session(group.sessions.iloc[0])
        build_trials(stored)
        stored.save_h5(groups=['metadata'])  # so `process` can rebuild it from file
        before = group.scan_product_status('trials/table')['trials/table']
        assert list(before) == ['current', 'absent']

        store.build_store(group, products=['trials/table'])

        after = group.scan_product_status('trials/table')['trials/table']
        assert list(after) == ['current', 'current']
