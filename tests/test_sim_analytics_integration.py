# test_sim_analytics_integration.py — end-to-end analytics integration (v0.8.2).

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from lerobot_coreai.sim import SimConfig, run_sim_mode


def _make_mock_policy(manifest_dict):
    from lerobot_coreai.manifest import LeRobotCoreAIManifest
    mock = MagicMock()
    mock.predict_action.return_value = {
        "action": [[0.0] * 7] * 16, "metadata": {"timing": {"total_ms": 12.3}},
    }
    mock.manifest = LeRobotCoreAIManifest.from_dict(manifest_dict)
    mock.policy_type = "evo1"
    mock.robot_type = "so100"
    mock.parity_passed = True
    mock.policy_repo_id = "test/policy"
    return mock


class TestSimAnalyticsIntegration:
    def test_summary_and_taxonomy_written_by_default(self, tmp_path, valid_manifest_dict):
        output_dir = tmp_path / "run"
        mock_policy = _make_mock_policy(valid_manifest_dict)
        with patch("lerobot_coreai.sim.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            config = SimConfig(
                policy_path="test/p", output_dir=output_dir, env_type="fake",
                confirm_sim_egress=True, episodes=2, max_steps_per_episode=3, fps=0,
            )
            result = run_sim_mode(config)
        assert (output_dir / "sim_summary.md").exists()
        assert (output_dir / "failure_taxonomy.json").exists()
        # CSV not written by default.
        assert not (output_dir / "episode_metrics.csv").exists()

    def test_csv_written_with_flag(self, tmp_path, valid_manifest_dict):
        output_dir = tmp_path / "run"
        mock_policy = _make_mock_policy(valid_manifest_dict)
        with patch("lerobot_coreai.sim.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            config = SimConfig(
                policy_path="test/p", output_dir=output_dir, env_type="fake",
                confirm_sim_egress=True, episodes=1, max_steps_per_episode=2, fps=0,
                export_csv=True,
            )
            run_sim_mode(config)
        assert (output_dir / "episode_metrics.csv").exists()
        assert (output_dir / "step_metrics.csv").exists()

    def test_report_includes_analytics_sections(self, tmp_path, valid_manifest_dict):
        output_dir = tmp_path / "run"
        mock_policy = _make_mock_policy(valid_manifest_dict)
        with patch("lerobot_coreai.sim.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            config = SimConfig(
                policy_path="test/p", output_dir=output_dir, env_type="fake",
                confirm_sim_egress=True, episodes=1, max_steps_per_episode=2, fps=0,
            )
            result = run_sim_mode(config)
        r = result.report
        assert "episode_metrics" in r
        assert "latency_metrics" in r
        assert "action_metrics" in r
        assert "failure_metrics" in r

    def test_latency_includes_env_step_ms(self, tmp_path, valid_manifest_dict):
        output_dir = tmp_path / "run"
        mock_policy = _make_mock_policy(valid_manifest_dict)
        with patch("lerobot_coreai.sim.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            config = SimConfig(
                policy_path="test/p", output_dir=output_dir, env_type="fake",
                confirm_sim_egress=True, episodes=1, max_steps_per_episode=2, fps=0,
            )
            result = run_sim_mode(config)
        assert "env_step_p95_ms" in result.report["latency_metrics"]

    def test_actions_jsonl_has_env_step_timing(self, tmp_path, valid_manifest_dict):
        output_dir = tmp_path / "run"
        mock_policy = _make_mock_policy(valid_manifest_dict)
        with patch("lerobot_coreai.sim.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            config = SimConfig(
                policy_path="test/p", output_dir=output_dir, env_type="fake",
                confirm_sim_egress=True, episodes=1, max_steps_per_episode=2, fps=0,
            )
            run_sim_mode(config)
        lines = (output_dir / "actions.jsonl").read_text().strip().split("\n")
        for line in lines:
            rec = json.loads(line)
            if rec["ok"]:
                assert "env_step_ms" in rec["timing"]

    def test_safety_invariants_preserved(self, tmp_path, valid_manifest_dict):
        output_dir = tmp_path / "run"
        mock_policy = _make_mock_policy(valid_manifest_dict)
        with patch("lerobot_coreai.sim.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            config = SimConfig(
                policy_path="test/p", output_dir=output_dir, env_type="fake",
                confirm_sim_egress=True, episodes=1, max_steps_per_episode=2, fps=0,
            )
            result = run_sim_mode(config)
        s = result.report["safety"]
        assert s["robot_egress_enabled"] is False
        assert s["actions_sent_to_robot"] == 0
        assert s["action_egress"] == "simulator_only"

    def test_summary_contains_no_overclaim(self, tmp_path, valid_manifest_dict):
        output_dir = tmp_path / "run"
        mock_policy = _make_mock_policy(valid_manifest_dict)
        with patch("lerobot_coreai.sim.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            config = SimConfig(
                policy_path="test/p", output_dir=output_dir, env_type="fake",
                confirm_sim_egress=True, episodes=1, max_steps_per_episode=2, fps=0,
            )
            run_sim_mode(config)
        summary = (output_dir / "sim_summary.md").read_text()
        assert "Proves real task success: False" in summary
        assert "Proves robot safety: False" in summary
        assert "real-world success: True" not in summary.lower()

    def test_no_summary_when_disabled(self, tmp_path, valid_manifest_dict):
        output_dir = tmp_path / "run"
        mock_policy = _make_mock_policy(valid_manifest_dict)
        with patch("lerobot_coreai.sim.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            config = SimConfig(
                policy_path="test/p", output_dir=output_dir, env_type="fake",
                confirm_sim_egress=True, episodes=1, max_steps_per_episode=2, fps=0,
                summary_md=False,
            )
            run_sim_mode(config)
        assert not (output_dir / "sim_summary.md").exists()
