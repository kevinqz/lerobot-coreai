# test_sim_supervisor_integration.py — sim mode + safety supervisor (v0.9.0).

import json
from unittest.mock import MagicMock, patch

from lerobot_coreai.sim import SimConfig, run_sim_mode


def _make_mock_policy(manifest_dict, action=None):
    from lerobot_coreai.manifest import LeRobotCoreAIManifest
    mock = MagicMock()
    mock.predict_action.return_value = {
        "action": action if action is not None else [[0.0] * 7] * 16,
        "metadata": {"timing": {"total_ms": 12.3}},
    }
    mock.manifest = LeRobotCoreAIManifest.from_dict(manifest_dict)
    mock.policy_type = "evo1"
    mock.robot_type = "so100"
    mock.parity_passed = True
    mock.policy_repo_id = "test/policy"
    return mock


def _cfg(tmp_path, **over):
    base = dict(
        policy_path="test/p", output_dir=tmp_path / "run", env_type="fake",
        confirm_sim_egress=True, episodes=1, max_steps_per_episode=3, fps=0,
    )
    base.update(over)
    return SimConfig(**base)


class TestSupervisorEnforce:
    def test_enforce_allows_valid_action_and_writes_reports(self, tmp_path, valid_manifest_dict):
        mock_policy = _make_mock_policy(valid_manifest_dict)
        with patch("lerobot_coreai.sim.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            result = run_sim_mode(_cfg(tmp_path, supervisor_mode="enforce"))
        assert result.ok
        out = tmp_path / "run"
        assert (out / "safety_report.jsonl").is_file()
        assert (out / "safety_summary.json").is_file()
        assert (out / "safety_summary.md").is_file()
        sec = result.report["safety_supervisor"]
        assert sec["enabled"] is True
        assert sec["mode"] == "enforce"
        assert sec["actions_supervised"] >= 1
        assert sec["actions_blocked"] == 0

    def test_report_includes_software_supervision_claim(self, tmp_path, valid_manifest_dict):
        mock_policy = _make_mock_policy(valid_manifest_dict)
        with patch("lerobot_coreai.sim.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            result = run_sim_mode(_cfg(tmp_path, supervisor_mode="enforce"))
        assert result.report["claims"]["proves_software_supervision"] is True
        assert result.report["claims"]["proves_physical_safety"] is False

    def test_enforce_blocks_nan_action(self, tmp_path, valid_manifest_dict):
        # A NaN action must be blocked and never sent to the simulator.
        bad = [[float("nan")] * 7] * 16
        mock_policy = _make_mock_policy(valid_manifest_dict, action=bad)
        with patch("lerobot_coreai.sim.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            result = run_sim_mode(_cfg(tmp_path, supervisor_mode="enforce",
                                       safety_profile_name="default-sim-safe"))
        sec = result.report["safety_supervisor"]
        assert sec["actions_blocked"] >= 1
        assert result.report["metrics"]["actions_sent_to_simulator"] == 0
        # No robot egress, ever.
        assert result.report["safety"]["actions_sent_to_robot"] == 0
        assert result.report["safety"]["robot_egress_enabled"] is False

    def test_blocked_episode_marked_safety_terminated(self, tmp_path, valid_manifest_dict):
        bad = [[float("inf")] * 7] * 16
        mock_policy = _make_mock_policy(valid_manifest_dict, action=bad)
        with patch("lerobot_coreai.sim.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            result = run_sim_mode(_cfg(tmp_path, supervisor_mode="enforce"))
        episodes = [json.loads(l) for l in
                    (tmp_path / "run" / "episodes.jsonl").read_text().strip().splitlines()]
        assert episodes[0]["terminated_by"] == "safety_supervisor"
        assert episodes[0]["success"] is False


class TestSupervisorModes:
    def test_off_writes_no_safety_artifacts(self, tmp_path, valid_manifest_dict):
        mock_policy = _make_mock_policy(valid_manifest_dict)
        with patch("lerobot_coreai.sim.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            result = run_sim_mode(_cfg(tmp_path, supervisor_mode="off"))
        assert "safety_supervisor" not in result.report
        assert not (tmp_path / "run" / "safety_report.jsonl").exists()

    def test_report_only_does_not_block_egress(self, tmp_path, valid_manifest_dict):
        # NaN action in report_only: recorded as would-block, but still egressed.
        bad = [[float("nan")] * 7] * 16
        mock_policy = _make_mock_policy(valid_manifest_dict, action=bad)
        with patch("lerobot_coreai.sim.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            result = run_sim_mode(_cfg(tmp_path, supervisor_mode="report_only"))
        sec = result.report["safety_supervisor"]
        assert sec["mode"] == "report_only"
        assert sec["actions_blocked"] == 0        # report-only never blocks
        assert result.report["metrics"]["actions_sent_to_simulator"] >= 1

    def test_report_only_unsafe_does_not_pass_summary(self, tmp_path, valid_manifest_dict):
        # The P1: report_only must not mask an unsafe finding as passed.
        bad = [[float("nan")] * 7] * 16
        mock_policy = _make_mock_policy(valid_manifest_dict, action=bad)
        with patch("lerobot_coreai.sim.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            result = run_sim_mode(_cfg(tmp_path, supervisor_mode="report_only"))
        sec = result.report["safety_supervisor"]
        assert sec["actions_blocked"] == 0
        assert sec["would_block_actions"] >= 1
        assert sec["passed"] is False

    def test_bundle_includes_safety_artifacts_and_verifies(self, tmp_path, valid_manifest_dict):
        from lerobot_coreai.sim_bundle import verify_sim_bundle
        bundle_dir = tmp_path / "bundle"
        mock_policy = _make_mock_policy(valid_manifest_dict)
        with patch("lerobot_coreai.sim.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            result = run_sim_mode(_cfg(
                tmp_path, supervisor_mode="enforce",
                package_run=True, package_output_dir=bundle_dir,
            ))
        assert result.ok
        assert (bundle_dir / "source_run" / "safety_summary.json").is_file()
        assert (bundle_dir / "source_run" / "safety_report.jsonl").is_file()
        manifest = json.loads((bundle_dir / "bundle_manifest.json").read_text())
        assert manifest["safety_supervisor"]["enabled"] is True
        # The bundle still verifies cleanly (no safety overclaim, checksums ok).
        v = verify_sim_bundle(bundle_dir)
        assert v.ok, v.invariant_failures + v.checksum_failures

    def test_invalid_profile_fails_clear(self, tmp_path, valid_manifest_dict):
        import pytest
        from lerobot_coreai.errors import CoreAIPolicyError
        bad_profile = tmp_path / "bad.json"
        bad_profile.write_text('{"schema_version": "wrong", "name": "x", "mode": "fail_closed"}')
        mock_policy = _make_mock_policy(valid_manifest_dict)
        with patch("lerobot_coreai.sim.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            with pytest.raises(CoreAIPolicyError):
                run_sim_mode(_cfg(tmp_path, supervisor_mode="enforce", safety_profile=bad_profile))
