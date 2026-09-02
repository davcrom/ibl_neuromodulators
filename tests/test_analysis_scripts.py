"""The command line every analysis script shares.

Each script is importable and parses its own flags. Kept in one module rather
than eight because the invariant is the script layer's, not any one script's.
"""
import importlib

import pytest

# Every analysis script, mapped to the arguments it cannot run without, so
# `parse_args` can be exercised on a parsable command line.
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
class TestAnalysisScriptCommandLine:

    def test_minimum_command_line_parses(self, script):
        """The script's own flags still parse without the retired store flags."""
        module = importlib.import_module(f'scripts.{script}')

        assert module.parse_args(ANALYSIS_SCRIPTS[script]) is not None
