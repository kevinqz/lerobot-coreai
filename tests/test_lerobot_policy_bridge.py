# test_lerobot_policy_bridge.py — CoreAILeRobotPolicyBridge behavior (v1.1.0).

import pytest

from lerobot_coreai.errors import CoreAIPolicyError
from lerobot_coreai.lerobot_policy import CoreAILeRobotPolicyBridge
from lerobot_coreai.manifest import LeRobotCoreAIManifest
from lerobot_coreai.policy import CoreAIPolicy


def _bridge(valid_manifest_dict, action=None):
    m = LeRobotCoreAIManifest.from_dict(valid_manifest_dict)
    policy = CoreAIPolicy(m, validate_io=False)
    # Patch inference so no runner is needed.
    canned = {"action": action if action is not None else [[0.0] * 7] * 16,
              "metadata": {"timing": {"total_ms": 4.2}}}
    policy.predict_action = lambda batch, **kw: (  # type: ignore
        canned if kw.get("return_metadata", True) else {"action": canned["action"]})
    return CoreAILeRobotPolicyBridge(policy)


def test_select_action_returns_raw_action(valid_manifest_dict):
    b = _bridge(valid_manifest_dict)
    action = b.select_action({"observation.state": [0.0] * 7})
    assert action == [[0.0] * 7] * 16  # raw action, not a dict


def test_predict_action_returns_dict_with_metadata(valid_manifest_dict):
    b = _bridge(valid_manifest_dict)
    out = b.predict_action({"observation.state": [0.0] * 7})
    assert "action" in out and "metadata" in out


def test_train_true_raises(valid_manifest_dict):
    b = _bridge(valid_manifest_dict)
    with pytest.raises(CoreAIPolicyError):
        b.train(True)


def test_train_false_and_eval_return_self(valid_manifest_dict):
    b = _bridge(valid_manifest_dict)
    assert b.train(False) is b
    assert b.eval() is b


def test_to_is_safe_noop_for_any_device(valid_manifest_dict):
    b = _bridge(valid_manifest_dict)
    # LeRobot code may call .to("cuda"); the bridge must not raise.
    assert b.to("cuda") is b
    assert b.to("cpu") is b
    assert b.to() is b


def test_metadata_is_honest(valid_manifest_dict):
    b = _bridge(valid_manifest_dict)
    md = b.metadata()
    assert md["runtime"] == "coreai"
    assert md["training_supported"] is False
    assert md["native_registry"] is False
    assert b.policy_type == "coreai_bridge"
