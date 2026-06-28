"""Tests for scripts/task_encoding.py orchestration helpers."""
import numpy as np
import pandas as pd
import pytest

from scripts.task_encoding import _block_label, assemble_dispersion_frame


@pytest.mark.parametrize('event_name, block, expected', [
    ('stimOn_times', 'task', 'stimOn_task'),
    ('feedback_times', 'movement', 'feedback_movement'),
])
def test_block_label(event_name, block, expected):
    assert _block_label(event_name, block) == expected


def _neural_long(records, rng):
    """Expand (subject, target_NM, eids, terms) records to a long coef frame."""
    rows = []
    for subject, target_NM, eids, terms in records:
        for eid in eids:
            for term in terms:
                rows.append({
                    'subject': subject, 'target_NM': target_NM, 'eid': eid,
                    'term': term, 'coef': rng.normal(),
                })
    return pd.DataFrame(rows)


def _behavioral_long(subject_eids, params, rng):
    """Expand {subject: [eids]} to a long behavioral-param frame."""
    rows = []
    for subject, eids in subject_eids.items():
        for eid in eids:
            for param in params:
                rows.append({
                    'subject': subject, 'eid': eid,
                    'param': param, 'value': rng.normal(),
                })
    return pd.DataFrame(rows)


class TestAssembleDispersionFrame:
    @pytest.fixture
    def frame(self):
        rng = np.random.default_rng(0)
        task_terms = ['contrast', 'side', 'reward', 'contrast:side']
        move_terms = ['choice_side', 'peak_velocity']
        terms = ['Intercept'] + task_terms + move_terms
        # s1: two target-NMs, each with 3 sessions; s2: one target with 2 sessions
        neural = _neural_long([
            ('s1', 'VTA-DA', ['e1', 'e2', 'e3'], terms),
            ('s1', 'SNc-DA', ['e4', 'e5', 'e6'], terms),
            ('s2', 'VTA-DA', ['e7', 'e8'], terms),
        ], rng)
        behavioral = _behavioral_long(
            {'s1': ['e1', 'e2', 'e3', 'e4', 'e5', 'e6'], 's2': ['e7', 'e8']},
            ['p1', 'p2', 'p3', 'p4'], rng,
        )
        block_mains = {
            'task': ['contrast', 'side', 'reward'],
            'movement': ['choice_side', 'log_reaction_time', 'peak_velocity'],
        }
        return assemble_dispersion_frame(
            {'stimOn_times': neural}, behavioral, block_mains, min_sessions=3)

    def test_drops_units_below_min_sessions(self, frame):
        # s2 has only 2 sessions (neural and behavioral) -> excluded everywhere
        assert 's2' not in set(frame['subject'])
        # s1's two target-NMs survive in every block
        for block in ('task', 'movement'):
            targets = set(frame[frame['block'] == block]['target_NM'])
            assert targets == {'VTA-DA', 'SNc-DA'}

    def test_two_targetnms_share_behavioral_dispersion(self, frame):
        task = frame[frame['block'] == 'task']
        shared = task.groupby('subject')['behavioral_dispersion'].nunique()
        assert (shared == 1).all()
        assert task['behavioral_dispersion'].notna().all()

    def test_columns(self, frame):
        assert list(frame.columns) == [
            'subject', 'target_NM', 'event', 'block',
            'neural_dispersion', 'behavioral_dispersion',
        ]
