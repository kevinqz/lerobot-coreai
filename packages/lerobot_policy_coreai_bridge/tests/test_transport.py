# test_transport.py — LeRobot batch -> CoreAI observation boundary (v1.3.2).

import json

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

from lerobot_coreai.errors import CoreAIPolicyError  # noqa: E402
from lerobot_policy_coreai_bridge.transport import (  # noqa: E402
    infer_batch_size, prepare_single_coreai_observation,
)


class _Manifest:
    observation_features = {"observation.state": object(), "observation.images.wrist": object()}


def test_b1_batch_stripped_and_task_unwrapped():
    batch = {
        "observation.state": torch.zeros(1, 7),
        "observation.images.wrist": torch.zeros(1, 3, 2, 2),
        "task": ["pick up the cube"],
        "action": torch.ones(1, 7),   # ground truth — must be dropped
        "reward": torch.ones(1),
    }
    obs, sha = prepare_single_coreai_observation(batch, _Manifest())
    assert "action" not in obs and "reward" not in obs        # no label leakage
    assert obs["task"] == "pick up the cube"                  # list -> str
    # Tensors become the JSON-safe __array__ envelope with the leading batch dim
    # stripped and shape/dtype preserved.
    assert obs["observation.state"]["__array__"] == [0.0] * 7
    assert obs["observation.state"]["shape"] == [7]
    assert obs["observation.images.wrist"]["shape"] == [3, 2, 2]
    json.dumps(obs)                                           # JSON-safe
    assert sha.startswith("sha256:")


def test_b_gt_1_fails_closed():
    batch = {"observation.state": torch.zeros(4, 7), "task": ["a", "b", "c", "d"]}
    with pytest.raises(CoreAIPolicyError):
        prepare_single_coreai_observation(batch, _Manifest())


def test_inconsistent_batch_sizes_fail():
    batch = {"observation.state": torch.zeros(2, 7), "task": ["only-one"]}
    with pytest.raises(CoreAIPolicyError):
        infer_batch_size(batch)


def test_only_manifest_features_kept():
    batch = {"observation.state": torch.zeros(1, 7),
             "observation.unexpected": torch.zeros(1, 3), "task": ["t"]}
    obs, _ = prepare_single_coreai_observation(batch, _Manifest())
    assert "observation.unexpected" not in obs
