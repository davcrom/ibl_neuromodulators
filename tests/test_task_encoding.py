"""Tests for scripts/task_encoding.py CCA orchestration helpers."""
import pytest

from scripts.task_encoding import _block_label


@pytest.mark.parametrize('event_name, block, expected', [
    ('stimOn_times', 'task', 'stimOn_task'),
    ('feedback_times', 'movement', 'feedback_movement'),
])
def test_block_label(event_name, block, expected):
    assert _block_label(event_name, block) == expected
