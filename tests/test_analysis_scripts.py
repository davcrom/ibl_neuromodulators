"""The command line and product list every analysis script shares.

Each script names the products it reads in `REQUIRED_PRODUCTS` and surveys them
with `PhotometrySessionGroup.check_products` before it starts. A name that is
not a `config.PRODUCT_SPEC` key survives import and only fails once the survey
runs against a real store, so it is checked here instead. Kept in one module
rather than eight because the invariant is the script layer's, not any one
script's.
"""
import importlib

import pytest

from iblnm.config import PRODUCT_SPEC

# Every script that surveys the store, mapped to the arguments it cannot run
# without, so `parse_args` can be exercised on a parsable command line.
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


@pytest.mark.parametrize('script', sorted(ANALYSIS_SCRIPTS))
class TestAnalysisScriptProducts:

    def test_required_products_are_known_products(self, script):
        """The surveyed list names `config.PRODUCT_SPEC` keys, and is not empty."""
        module = importlib.import_module(f'scripts.{script}')

        assert set(module.REQUIRED_PRODUCTS) <= set(PRODUCT_SPEC)
        assert module.REQUIRED_PRODUCTS

    def test_minimum_command_line_parses(self, script):
        """The script's own flags still parse without the retired store flags."""
        module = importlib.import_module(f'scripts.{script}')

        assert module.parse_args(ANALYSIS_SCRIPTS[script]) is not None
