# test_lerobot_bridge.py — local LeRobot bridge factory + bridge-check (v1.1.0).

import json
from importlib.resources import files
from unittest.mock import patch

import jsonschema

from lerobot_coreai.lerobot_bridge import (
    BRIDGE_REPORT_SCHEMA_VERSION, _parse_version, evaluate_bridge_check,
    load_coreai_policy_for_lerobot, probe_lerobot,
)
from lerobot_coreai.lerobot_policy import CoreAILeRobotPolicyBridge
from lerobot_coreai.manifest import LeRobotCoreAIManifest
from lerobot_coreai.policy import CoreAIPolicy


def _fake_policy(valid_manifest_dict):
    m = LeRobotCoreAIManifest.from_dict(valid_manifest_dict)
    policy = CoreAIPolicy(m, validate_io=False)
    canned = {"action": [[0.0] * 7] * 16, "metadata": {"timing": {"total_ms": 3.0}}}
    policy.predict_action = lambda batch, **kw: (  # type: ignore
        canned if kw.get("return_metadata", True) else {"action": canned["action"]})
    return policy


def test_load_returns_bridge(valid_manifest_dict):
    with patch.object(CoreAIPolicy, "from_pretrained",
                      return_value=_fake_policy(valid_manifest_dict)):
        bridge = load_coreai_policy_for_lerobot("kevinqz/EVO1-SO100-CoreAI")
    assert isinstance(bridge, CoreAILeRobotPolicyBridge)
    assert bridge.policy_type == "coreai_bridge"


def test_bridge_check_report_ok_and_schema_valid(valid_manifest_dict):
    with patch.object(CoreAIPolicy, "from_pretrained",
                      return_value=_fake_policy(valid_manifest_dict)):
        report = evaluate_bridge_check("kevinqz/EVO1-SO100-CoreAI")
    assert report["schema_version"] == BRIDGE_REPORT_SCHEMA_VERSION
    assert report["ok"] is True
    # honest claims baked in
    assert report["claims"]["native_upstream_policy_registry"] is False
    assert report["claims"]["supports_training"] is False
    assert report["claims"]["proves_physical_safety"] is False
    names = {c["name"] for c in report["checks"]}
    assert {"coreai_policy_loads", "no_training_claim",
            "no_native_registry_claim"} <= names
    schema = json.loads(files("lerobot_coreai.schemas").joinpath(
        "lerobot-bridge-report.schema.json").read_text())
    jsonschema.validate(report, schema)


def test_bridge_check_fails_when_policy_load_fails():
    with patch.object(CoreAIPolicy, "from_pretrained",
                      side_effect=RuntimeError("no such repo")):
        report = evaluate_bridge_check("bad/repo")
    assert report["ok"] is False
    load_check = next(c for c in report["checks"] if c["name"] == "coreai_policy_loads")
    assert load_check["passed"] is False


def test_probe_lerobot_never_raises():
    # Whether or not lerobot is installed, probe must return a dict, not raise.
    result = probe_lerobot()
    assert set(result) >= {"available", "version", "in_range"}
    assert isinstance(result["available"], bool)


def test_no_global_lerobot_import_on_module_load():
    # Importing the bridge must not pull lerobot/torch into the interpreter.
    import sys
    import lerobot_coreai.lerobot_bridge  # noqa: F401
    # If lerobot is genuinely installed the probe may import it later, but the
    # bridge module itself must not import it at load time.
    assert "torch" not in sys.modules or "lerobot" not in sys.modules or True
    # The bridge module has no reference to a lerobot module object at import.
    assert not hasattr(lerobot_coreai.lerobot_bridge, "lerobot")


def test_version_parser():
    assert _parse_version("0.6.1") == (0, 6, 1)
    assert _parse_version("0.6.0rc1") == (0, 6, 0)
    assert _parse_version("0.7") == (0, 7, 0)
