# test_cli_real.py — CLI tests for guarded real mode (v1.0.0).

from unittest.mock import MagicMock, patch

from lerobot_coreai import cli


def _mock_policy(valid_manifest_dict):
    from lerobot_coreai.manifest import LeRobotCoreAIManifest
    m = MagicMock()
    m.predict_action.return_value = {"action": [[0.0] * 7] * 16,
                                     "metadata": {"timing": {"total_ms": 5.0}}}
    m.manifest = LeRobotCoreAIManifest.from_dict(valid_manifest_dict)
    m.policy_type = "evo1"
    m.robot_type = "so100"
    m.policy_repo_id = "test/policy"
    return m


def _base_argv(sc, tmp_path, mode):
    return [
        "real", "--mode", mode,
        "--policy.path", "test/p", "--runner.url", "http://127.0.0.1:8710",
        "--robot.adapter", "mock", "--robot.type", sc["robot_type"],
        "--safety.profile", str(sc["profile"]),
        "--readiness-report", str(sc["readiness"]),
        "--approval", str(sc["approval"]),
        "--bundle-dir", str(sc["bundle_dir"]),
        "--output-dir", str(tmp_path / "out"),
    ]


_ATTEST = [
    "--i-understand-this-may-move-real-hardware",
    "--i-have-physical-emergency-stop-ready",
    "--i-confirm-robot-workspace-is-clear",
]


def test_preflight_rc0_zero_actions(real_ready_scenario, tmp_path):
    sc = real_ready_scenario()
    rc = cli.main(_base_argv(sc, tmp_path, "preflight"))
    assert rc == 0
    assert (tmp_path / "out" / "real_preflight_report.json").is_file()


def test_guarded_rc0_with_mock(real_ready_scenario, tmp_path, valid_manifest_dict):
    sc = real_ready_scenario()
    argv = _base_argv(sc, tmp_path, "guarded") + [
        "--operator", "Kevin Saltarelli", "--max-steps", "3", "--fps", "10"] + _ATTEST
    with patch("lerobot_coreai.real_mode.CoreAIPolicy.from_pretrained",
               return_value=_mock_policy(valid_manifest_dict)):
        rc = cli.main(argv)
    assert rc == 0
    assert (tmp_path / "out" / "real_report.json").is_file()


def test_guarded_rc1_missing_attestation(real_ready_scenario, tmp_path, valid_manifest_dict):
    sc = real_ready_scenario()
    argv = _base_argv(sc, tmp_path, "guarded") + [
        "--operator", "K", "--max-steps", "3", "--fps", "10",
        "--i-understand-this-may-move-real-hardware"]  # only one attestation
    with patch("lerobot_coreai.real_mode.CoreAIPolicy.from_pretrained",
               return_value=_mock_policy(valid_manifest_dict)):
        rc = cli.main(argv)
    assert rc == 1


def test_guarded_rc1_not_ready(real_ready_scenario, tmp_path, valid_manifest_dict):
    sc = real_ready_scenario(ready=False)
    argv = _base_argv(sc, tmp_path, "guarded") + [
        "--operator", "K", "--max-steps", "3", "--fps", "10"] + _ATTEST
    with patch("lerobot_coreai.real_mode.CoreAIPolicy.from_pretrained",
               return_value=_mock_policy(valid_manifest_dict)):
        rc = cli.main(argv)
    assert rc == 1


def test_guarded_json_output(real_ready_scenario, tmp_path, valid_manifest_dict, capsys):
    import json
    sc = real_ready_scenario()
    argv = _base_argv(sc, tmp_path, "guarded") + [
        "--operator", "K", "--max-steps", "2", "--fps", "10", "--json"] + _ATTEST
    with patch("lerobot_coreai.real_mode.CoreAIPolicy.from_pretrained",
               return_value=_mock_policy(valid_manifest_dict)):
        rc = cli.main(argv)
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["egress"]["actions_sent_to_robot"] == 2
    assert payload["claims"]["proves_physical_safety"] is False
