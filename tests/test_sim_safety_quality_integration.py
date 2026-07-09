# test_sim_safety_quality_integration.py — sim + safety quality gates (v0.9.2).

from unittest.mock import MagicMock, patch

from lerobot_coreai.safety_quality import SafetyQualityConfig
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


class TestSimSafetyQuality:
    def test_clean_run_passes_gate(self, tmp_path, valid_manifest_dict):
        mock_policy = _make_mock_policy(valid_manifest_dict)
        with patch("lerobot_coreai.sim.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            result = run_sim_mode(_cfg(
                tmp_path, safety_quality=SafetyQualityConfig(), fail_on_safety_quality=True))
        assert result.ok
        assert (tmp_path / "run" / "safety_quality_report.json").is_file()
        assert result.report["safety_quality"]["passed"] is True

    def test_report_includes_safety_quality_section(self, tmp_path, valid_manifest_dict):
        mock_policy = _make_mock_policy(valid_manifest_dict)
        with patch("lerobot_coreai.sim.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            result = run_sim_mode(_cfg(tmp_path, safety_quality=SafetyQualityConfig()))
        assert "safety_quality" in result.report
        assert "checks" in result.report["safety_quality"]

    def test_blocked_action_fails_run_when_fail_on(self, tmp_path, valid_manifest_dict):
        # NaN action → blocked → safety summary not passed → gate fails the run.
        bad = [[float("nan")] * 7] * 16
        mock_policy = _make_mock_policy(valid_manifest_dict, action=bad)
        with patch("lerobot_coreai.sim.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            result = run_sim_mode(_cfg(
                tmp_path, supervisor_mode="enforce",
                safety_profile_name="default-sim-safe",
                safety_quality=SafetyQualityConfig(), fail_on_safety_quality=True))
        assert result.ok is False
        assert result.report["safety_quality"]["passed"] is False
        # No robot egress, ever.
        assert result.report["safety"]["actions_sent_to_robot"] == 0
        assert result.report["safety"]["robot_egress_enabled"] is False

    def test_gate_report_only_does_not_fail_run(self, tmp_path, valid_manifest_dict):
        bad = [[float("nan")] * 7] * 16
        mock_policy = _make_mock_policy(valid_manifest_dict, action=bad)
        with patch("lerobot_coreai.sim.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            result = run_sim_mode(_cfg(
                tmp_path, supervisor_mode="enforce",
                safety_profile_name="default-sim-safe",
                safety_quality=SafetyQualityConfig(), fail_on_safety_quality=False))
        # Gate found problems but was not set to fail the run.
        assert result.report["safety_quality"]["passed"] is False
        assert result.ok is True

    def test_safety_gate_requires_supervisor_enabled(self, tmp_path, valid_manifest_dict):
        import pytest
        from lerobot_coreai.errors import CoreAIPolicyError
        mock_policy = _make_mock_policy(valid_manifest_dict)
        with patch("lerobot_coreai.sim.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            with pytest.raises(CoreAIPolicyError, match="Safety quality gates require"):
                run_sim_mode(_cfg(
                    tmp_path, supervisor_mode="off",
                    safety_quality=SafetyQualityConfig(), fail_on_safety_quality=True))

    def test_no_gate_config_no_section(self, tmp_path, valid_manifest_dict):
        mock_policy = _make_mock_policy(valid_manifest_dict)
        with patch("lerobot_coreai.sim.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            result = run_sim_mode(_cfg(tmp_path))
        assert "safety_quality" not in result.report
