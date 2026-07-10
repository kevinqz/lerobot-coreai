# test_batching.py — split-and-stack batching fallback (v1.2.5).

import pytest

from lerobot_coreai.action_contract import BatchContract
from lerobot_coreai.batching import (
    detect_batch_size, run_batched_with_fallback, split_observation, stack_actions,
)
from lerobot_coreai.errors import CoreAIPolicyError


def test_detect_single_observation_returns_none():
    assert detect_batch_size({"observation.state": [0.0] * 7, "task": "do it"}) is None


def test_detect_batched_task_list():
    obs = {"observation.state": [[0.0] * 7, [1.0] * 7], "task": ["a", "b"]}
    assert detect_batch_size(obs) == 2


def test_inconsistent_batch_sizes_raise():
    obs = {"observation.state": [[0.0], [1.0], [2.0]], "task": ["a", "b"]}
    with pytest.raises(CoreAIPolicyError):
        detect_batch_size(obs)


def test_split_observation():
    obs = {"observation.state": [[0.0], [1.0]], "task": ["a", "b"]}
    parts = split_observation(obs, 2)
    assert parts[0] == {"observation.state": [0.0], "task": "a"}
    assert parts[1] == {"observation.state": [1.0], "task": "b"}


def test_run_single_when_not_batched():
    obs = {"observation.state": [0.0] * 7, "task": "x"}
    out = run_batched_with_fallback(obs, BatchContract(), lambda o: [1.0, 2.0])
    assert out == [1.0, 2.0]


def test_split_and_stack_runs_each_and_stacks():
    obs = {"observation.state": [[0.0], [1.0], [2.0]], "task": ["a", "b", "c"]}
    calls = []

    def _run(o):
        calls.append(o["task"])
        return [len(o["task"])]

    out = run_batched_with_fallback(obs, BatchContract(fallback="split_and_stack"), _run)
    assert calls == ["a", "b", "c"]
    assert out == stack_actions([[1], [1], [1]])


def test_reject_fallback_raises_on_batch():
    obs = {"observation.state": [[0.0], [1.0]], "task": ["a", "b"]}
    with pytest.raises(CoreAIPolicyError):
        run_batched_with_fallback(obs, BatchContract(fallback="reject"), lambda o: [0.0])
