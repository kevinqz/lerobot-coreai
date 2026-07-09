# test_shadow_supervisor_integration.py — shadow + safety supervisor (v0.9.0).
#
# Shadow ALWAYS blocks every action via ActionBlocker. Enabling the supervisor
# only adds auditable diagnostic decisions; it never enables any egress.

import json
from unittest.mock import MagicMock, patch

from lerobot_coreai.shadow import ShadowConfig, run_shadow_mode


def _make_mock_policy(manifest_dict, action=None):
    from lerobot_coreai.manifest import LeRobotCoreAIManifest
    mock = MagicMock()
    mock.predict_action.return_value = {
        "action": action if action is not None else [[0.01 * i] * 7 for i in range(16)],
        "metadata": {"timing": {"total_ms": 12.3}},
    }
    mock.manifest = LeRobotCoreAIManifest.from_dict(manifest_dict)
    mock.policy_type = "evo1"
    mock.robot_type = "so100"
    mock.parity_passed = True
    mock.policy_repo_id = "kevinqz/EVO1-SO100-CoreAI"
    return mock


def _fixtures(tmp_path, n=3):
    tmp_path.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (tmp_path / f"{i:06d}.json").write_text(json.dumps({
            "observation.state": [0.0] * 7,
            "observation.images.wrist": "wrist.png",
            "task": "pick up the cube",
        }))
    return tmp_path


def _cfg(tmp_path, **over):
    base = dict(
        policy_path="kevinqz/EVO1-SO100-CoreAI",
        observation_source="fixtures",
        fixtures_dir=_fixtures(tmp_path / "fixtures", n=3),
        runner_url="http://localhost:8710",
        output_dir=tmp_path / "run",
        max_steps=3, fps=0,
    )
    base.update(over)
    return ShadowConfig(**base)


def test_shadow_writes_supervisor_decisions(tmp_path, valid_manifest_dict):
    mock_policy = _make_mock_policy(valid_manifest_dict)
    with patch("lerobot_coreai.shadow.CoreAIPolicy.from_pretrained", return_value=mock_policy):
        result = run_shadow_mode(_cfg(tmp_path, supervisor_mode="report_only"))
    assert (tmp_path / "run" / "safety_report.jsonl").is_file()
    sec = result.report["safety_supervisor"]
    assert sec["enabled"] is True
    assert sec["actions_supervised"] == 3


def test_shadow_still_sends_zero_actions_with_supervisor(tmp_path, valid_manifest_dict):
    mock_policy = _make_mock_policy(valid_manifest_dict)
    with patch("lerobot_coreai.shadow.CoreAIPolicy.from_pretrained", return_value=mock_policy):
        result = run_shadow_mode(_cfg(tmp_path, supervisor_mode="enforce"))
    m = result.report["metrics"]
    assert m["actions_sent"] == 0
    assert m["actions_blocked"] == 3
    # Every action was blocked by ActionBlocker regardless of supervisor verdict.


def test_shadow_off_by_default_no_safety_section(tmp_path, valid_manifest_dict):
    mock_policy = _make_mock_policy(valid_manifest_dict)
    with patch("lerobot_coreai.shadow.CoreAIPolicy.from_pretrained", return_value=mock_policy):
        result = run_shadow_mode(_cfg(tmp_path))
    assert "safety_supervisor" not in result.report
    assert not (tmp_path / "run" / "safety_report.jsonl").exists()
