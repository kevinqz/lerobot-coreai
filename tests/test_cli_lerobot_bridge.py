# test_cli_lerobot_bridge.py — CLI for lerobot-bridge-check (v1.1.0).

import json
from unittest.mock import patch

from lerobot_coreai import cli
from lerobot_coreai.manifest import LeRobotCoreAIManifest
from lerobot_coreai.policy import CoreAIPolicy


def _fake_policy(valid_manifest_dict):
    m = LeRobotCoreAIManifest.from_dict(valid_manifest_dict)
    policy = CoreAIPolicy(m, validate_io=False)
    canned = {"action": [[0.0] * 7] * 16, "metadata": {"timing": {"total_ms": 3.0}}}
    policy.predict_action = lambda batch, **kw: (  # type: ignore
        canned if kw.get("return_metadata", True) else {"action": canned["action"]})
    return policy


def test_cli_bridge_check_rc0_writes_report(valid_manifest_dict, tmp_path):
    out = tmp_path / "bridge"
    argv = ["lerobot-bridge-check", "--policy.path", "kevinqz/EVO1-SO100-CoreAI",
            "--output-dir", str(out)]
    with patch.object(CoreAIPolicy, "from_pretrained",
                      return_value=_fake_policy(valid_manifest_dict)):
        rc = cli.main(argv)
    assert rc == 0
    report = json.loads((out / "lerobot_bridge_report.json").read_text())
    assert report["ok"] is True
    assert report["claims"]["native_upstream_policy_registry"] is False
    assert (out / "lerobot_bridge_report.md").is_file()


def test_cli_bridge_check_rc1_on_load_failure(tmp_path):
    argv = ["lerobot-bridge-check", "--policy.path", "bad/repo", "--json"]
    with patch.object(CoreAIPolicy, "from_pretrained",
                      side_effect=RuntimeError("nope")):
        rc = cli.main(argv)
    assert rc == 1
