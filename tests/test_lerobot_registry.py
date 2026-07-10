# test_lerobot_registry.py — local, opt-in LeRobot registry adapter (v1.1.3).

from unittest.mock import patch

import pytest

from lerobot_coreai.errors import CoreAIPolicyError
from lerobot_coreai.lerobot_config import BRIDGE_POLICY_TYPE
from lerobot_coreai.lerobot_policy import CoreAILeRobotPolicyBridge
from lerobot_coreai.lerobot_registry import (
    CoreAILeRobotRegistry, evaluate_registry_check,
)
from lerobot_coreai.manifest import LeRobotCoreAIManifest
from lerobot_coreai.policy import CoreAIPolicy


def _fake_policy(valid_manifest_dict):
    m = LeRobotCoreAIManifest.from_dict(valid_manifest_dict)
    p = CoreAIPolicy(m, validate_io=False)
    canned = {"action": [[0.0] * 7] * 16, "metadata": {}}
    p.predict_action = lambda batch, **kw: (  # type: ignore
        canned if kw.get("return_metadata", True) else {"action": canned["action"]})
    return p


def test_register_and_query():
    reg = CoreAILeRobotRegistry()
    assert not reg.is_registered(BRIDGE_POLICY_TYPE)
    reg.register(BRIDGE_POLICY_TYPE)
    assert reg.is_registered(BRIDGE_POLICY_TYPE)
    assert BRIDGE_POLICY_TYPE in reg.registered()


def test_register_refuses_coreai():
    reg = CoreAILeRobotRegistry()
    with pytest.raises(CoreAIPolicyError):
        reg.register("coreai")


def test_load_returns_bridge(valid_manifest_dict):
    reg = CoreAILeRobotRegistry()
    reg.register(BRIDGE_POLICY_TYPE)
    with patch.object(CoreAIPolicy, "from_pretrained",
                      return_value=_fake_policy(valid_manifest_dict)):
        bridge = reg.load(BRIDGE_POLICY_TYPE, policy_path="kevinqz/EVO1-SO100-CoreAI")
    assert isinstance(bridge, CoreAILeRobotPolicyBridge)


def test_load_unknown_type_fails_closed():
    reg = CoreAILeRobotRegistry()
    with pytest.raises(CoreAIPolicyError):
        reg.load("nope", policy_path="x")


def test_registry_check_report_ok_and_honest():
    report = evaluate_registry_check()
    assert report["ok"] is True
    assert report["claims"]["native_upstream_registry"] is False
    assert report["claims"]["supports_training"] is False
    names = {c["name"] for c in report["checks"]}
    assert "reserved_coreai_refused" in names
    assert "registry_registers" in names
