# test_action_queue.py — per-timestep action queue (v1.2.5).

import pytest

from lerobot_coreai.action_queue import ActionQueue
from lerobot_coreai.errors import CoreAIPolicyError


def test_load_and_drain_sequentially():
    q = ActionQueue()
    q.load_chunk([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    assert len(q) == 3
    assert q.pop_next() == [1.0, 2.0]
    assert q.pop_next() == [3.0, 4.0]
    assert q.pop_next() == [5.0, 6.0]
    assert q.empty is True


def test_exhaustion_raises():
    q = ActionQueue()
    q.load_chunk([[1.0, 2.0]])
    q.pop_next()
    with pytest.raises(CoreAIPolicyError):
        q.pop_next()


def test_reset_clears():
    q = ActionQueue()
    q.load_chunk([[1.0], [2.0]])
    q.reset()
    assert q.empty is True


def test_ragged_chunk_rejected():
    q = ActionQueue()
    with pytest.raises(CoreAIPolicyError):
        q.load_chunk([[1.0, 2.0], [3.0]])


def test_nonfinite_chunk_rejected():
    q = ActionQueue()
    with pytest.raises(CoreAIPolicyError):
        q.load_chunk([[1.0, float("nan")]])
    with pytest.raises(CoreAIPolicyError):
        q.load_chunk([[float("inf"), 2.0]])


def test_empty_chunk_rejected():
    q = ActionQueue()
    with pytest.raises(CoreAIPolicyError):
        q.load_chunk([])
    with pytest.raises(CoreAIPolicyError):
        q.load_chunk([[]])


def test_single_action_wrapped_as_one_row():
    # A single action [A] (scalar rows) must load as one row and pop as [A].
    q = ActionQueue()
    q.load_chunk([1.0, 2.0, 3.0])
    assert len(q) == 1
    assert q.pop_next() == [1.0, 2.0, 3.0]


def test_values_normalized_to_float():
    q = ActionQueue()
    q.load_chunk([[1, 2], [3, 4]])  # ints in → floats out
    row = q.pop_next()
    assert row == [1.0, 2.0]
    assert all(isinstance(x, float) for x in row)


def test_non_numeric_string_rejected():
    q = ActionQueue()
    with pytest.raises(CoreAIPolicyError):
        q.load_chunk([["not-a-number", 2.0]])
