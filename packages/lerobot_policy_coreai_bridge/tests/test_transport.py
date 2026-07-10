# test_transport.py — LeRobot batch -> CoreAI observation boundary (v1.3.2).

import json

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

from lerobot_coreai.errors import CoreAIPolicyError  # noqa: E402
from lerobot_coreai.errors import CoreAIPolicyError as _CE  # noqa: E402
from lerobot_policy_coreai_bridge.transport import (  # noqa: E402
    TYPED_ARRAY_ENVELOPE_V1, infer_batch_size, prepare_single_coreai_observation,
)


class _Feat:
    def __init__(self, shape):
        self.shape = shape


class _Manifest:
    observation_features = {"observation.state": _Feat([7]),
                            "observation.images.wrist": _Feat([3, 2, 2]),
                            "task": _Feat(None)}   # declared -> forwarded (v1.3.12)


def test_b1_batch_stripped_task_unwrapped_nested_json_default():
    batch = {
        "observation.state": torch.zeros(1, 7),
        "observation.images.wrist": torch.zeros(1, 3, 2, 2),
        "task": ["pick up the cube"],
        "action": torch.ones(1, 7),   # ground truth — must be dropped
        "reward": torch.ones(1),
    }
    audit: dict = {}
    obs, sha = prepare_single_coreai_observation(batch, _Manifest(), audit=audit)
    assert "action" not in obs and "reward" not in obs        # no label leakage
    assert obs["task"] == "pick up the cube"                  # list -> str
    # Default nested_json_v1 sends PLAIN nested lists (no typed envelope).
    assert obs["observation.state"] == [0.0] * 7
    assert isinstance(obs["observation.images.wrist"], list)
    # Shape is audited separately (validated before encoding).
    assert audit["observation.state"]["shape"] == [7]
    json.dumps(obs)                                           # JSON-safe
    assert sha.startswith("sha256:")


def test_typed_envelope_only_when_selected():
    batch = {"observation.state": torch.zeros(1, 7), "task": ["t"]}
    obs, _ = prepare_single_coreai_observation(
        batch, _Manifest(), encoding=TYPED_ARRAY_ENVELOPE_V1)
    assert obs["observation.state"]["__array__"] == [0.0] * 7
    assert obs["observation.state"]["shape"] == [7]


def test_shape_mismatch_fails_before_encoding():
    batch = {"observation.state": torch.zeros(1, 99), "task": ["t"]}  # want 7
    with pytest.raises(_CE):
        prepare_single_coreai_observation(batch, _Manifest())


def test_unknown_encoding_fails():
    with pytest.raises(_CE):
        prepare_single_coreai_observation(
            {"observation.state": torch.zeros(1, 7)}, _Manifest(), encoding="bogus")


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


# --- v1.3.10: strict runtime validation + canonical batch hash ---

class _ReqFeat:
    def __init__(self, shape, required=True):
        self.shape = shape
        self.required = required


class _ReqManifest:
    observation_features = {"observation.state": _ReqFeat([7], required=True),
                            "task": _ReqFeat(None, required=False)}


def test_task_none_fails():
    batch = {"observation.state": torch.zeros(1, 7), "task": [None]}
    with pytest.raises(CoreAIPolicyError):
        prepare_single_coreai_observation(batch, _ReqManifest())


def test_task_int_fails():
    batch = {"observation.state": torch.zeros(1, 7), "task": [42]}
    with pytest.raises(CoreAIPolicyError):
        prepare_single_coreai_observation(batch, _ReqManifest())


def test_missing_required_feature_fails():
    batch = {"task": ["only task, no state"]}
    with pytest.raises(CoreAIPolicyError):
        prepare_single_coreai_observation(batch, _ReqManifest())


def test_canonical_batch_hash_is_order_sensitive():
    from lerobot_policy_coreai_bridge.transport import canonical_batch_sha256
    a = canonical_batch_sha256(2, ["sha256:aa", "sha256:bb"], "split_and_stack")
    b = canonical_batch_sha256(2, ["sha256:bb", "sha256:aa"], "split_and_stack")
    assert a != b                                   # order matters
    assert a != "sha256:aa"                          # not just the first sample
    assert a == canonical_batch_sha256(2, ["sha256:aa", "sha256:bb"], "split_and_stack")


# --- v1.3.12: task requiredness + undeclared-task drop ---

class _TaskReqManifest:
    observation_features = {"observation.state": _ReqFeat([7], required=True),
                            "task": _ReqFeat(None, required=True)}


class _NoTaskManifest:
    observation_features = {"observation.state": _ReqFeat([7], required=True)}


def test_required_task_absent_fails():
    with pytest.raises(CoreAIPolicyError):
        prepare_single_coreai_observation(
            {"observation.state": torch.zeros(1, 7)}, _TaskReqManifest())


def test_undeclared_task_is_dropped():
    obs, _ = prepare_single_coreai_observation(
        {"observation.state": torch.zeros(1, 7), "task": ["ignored"]},
        _NoTaskManifest())
    assert "task" not in obs           # manifest does not declare task -> dropped
