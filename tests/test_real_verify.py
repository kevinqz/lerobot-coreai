# test_real_verify.py — offline real-session audit verifier (v1.0.2).

from unittest.mock import MagicMock, patch

from lerobot_coreai.real_mode import RealModeConfig, run_real_mode
from lerobot_coreai.real_verify import verify_real_session


def _mock_policy(valid_manifest_dict, action=None):
    from lerobot_coreai.manifest import LeRobotCoreAIManifest
    m = MagicMock()
    m.predict_action.return_value = {"action": action if action is not None else [[0.0] * 7] * 16,
                                     "metadata": {"timing": {"total_ms": 5.0}}}
    m.manifest = LeRobotCoreAIManifest.from_dict(valid_manifest_dict)
    m.policy_type = "evo1"
    m.robot_type = "so100"
    m.policy_repo_id = "test/policy"
    return m


def _run_guarded(sc, tmp_path, valid_manifest_dict, action=None, max_steps=3):
    cfg = RealModeConfig(
        mode="guarded", policy_path="test/p", runner_url="http://127.0.0.1:8710",
        robot_adapter="mock", robot_type=sc["robot_type"],
        safety_profile=sc["profile"], readiness_report=sc["readiness"],
        approval=sc["approval"], bundle_dir=sc["bundle_dir"],
        output_dir=tmp_path / "real_out", operator="K", max_steps=max_steps, fps=10.0,
        attest_real_hardware=True, attest_physical_estop=True, attest_workspace_clear=True)
    with patch("lerobot_coreai.real_mode.CoreAIPolicy.from_pretrained",
               return_value=_mock_policy(valid_manifest_dict, action=action)):
        return run_real_mode(cfg)


def test_verify_clean_session_passes(real_ready_scenario, tmp_path, valid_manifest_dict):
    sc = real_ready_scenario()
    result = _run_guarded(sc, tmp_path, valid_manifest_dict)
    assert result.ok
    v = verify_real_session(tmp_path / "real_out",
                            bundle_dir=sc["bundle_dir"], approval=sc["approval"],
                            readiness_report=sc["readiness"])
    assert v.ok, [c for c in v.checks if not c["passed"]]


def test_verify_detects_action_count_tamper(real_ready_scenario, tmp_path, valid_manifest_dict):
    import json
    sc = real_ready_scenario()
    _run_guarded(sc, tmp_path, valid_manifest_dict)
    report_path = tmp_path / "real_out" / "real_report.json"
    report = json.loads(report_path.read_text())
    report["egress"]["actions_sent_to_robot"] = 999  # lie about egress count
    report_path.write_text(json.dumps(report))
    v = verify_real_session(tmp_path / "real_out")
    assert not v.ok
    assert any(c["name"] == "actions_sent_count_matches" and not c["passed"] for c in v.checks)


def test_verify_detects_missing_report(tmp_path):
    (tmp_path / "empty").mkdir()
    v = verify_real_session(tmp_path / "empty")
    assert not v.ok
    assert any(c["name"] == "real_report_exists" and not c["passed"] for c in v.checks)


def test_verify_blocked_session(real_ready_scenario, tmp_path, valid_manifest_dict):
    # A NaN action blocks; verify should still find the artifacts internally
    # consistent (zero sent, blocked action not sent).
    sc = real_ready_scenario()
    bad = [[float("nan")] * 7] * 16
    result = _run_guarded(sc, tmp_path, valid_manifest_dict, action=bad, max_steps=3)
    assert not result.ok
    v = verify_real_session(tmp_path / "real_out")
    # accounting invariants hold even though the session failed.
    assert any(c["name"] == "no_blocked_action_sent" and c["passed"] for c in v.checks)
    assert any(c["name"] == "actions_sent_count_matches" and c["passed"] for c in v.checks)
