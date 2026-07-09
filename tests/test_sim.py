# test_sim.py — integration tests for sim mode (v0.8).

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from lerobot_coreai.errors import CoreAIPolicyError
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


class TestSimSuccess:
    def test_sim_writes_all_files(self, tmp_path, valid_manifest_dict):
        output_dir = tmp_path / "run"
        mock_policy = _make_mock_policy(valid_manifest_dict)

        with patch("lerobot_coreai.sim.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            config = SimConfig(
                policy_path="test/policy",
                output_dir=output_dir,
                env_type="fake",
                confirm_sim_egress=True,
                episodes=2,
                max_steps_per_episode=4,
                fps=0,
            )
            result = run_sim_mode(config)

        assert result.ok is True
        assert result.report_path.exists()
        assert result.trace_path.exists()
        assert result.actions_path.exists()
        assert result.observations_path.exists()
        assert result.episodes_path.exists()

    def test_sim_actions_sent_to_simulator(self, tmp_path, valid_manifest_dict):
        output_dir = tmp_path / "run"
        mock_policy = _make_mock_policy(valid_manifest_dict)

        with patch("lerobot_coreai.sim.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            config = SimConfig(
                policy_path="test/policy",
                output_dir=output_dir,
                env_type="fake",
                confirm_sim_egress=True,
                episodes=1,
                max_steps_per_episode=5,
                fps=0,
            )
            result = run_sim_mode(config)

        m = result.report["metrics"]
        assert m["actions_sent_to_simulator"] > 0
        assert m["actions_sent_to_robot"] == 0
        assert m["episodes_completed"] == 1

    def test_sim_multiple_episodes(self, tmp_path, valid_manifest_dict):
        output_dir = tmp_path / "run"
        mock_policy = _make_mock_policy(valid_manifest_dict)

        with patch("lerobot_coreai.sim.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            config = SimConfig(
                policy_path="test/policy",
                output_dir=output_dir,
                env_type="fake",
                confirm_sim_egress=True,
                episodes=3,
                max_steps_per_episode=2,
                fps=0,
            )
            result = run_sim_mode(config)

        # episodes.jsonl should have 3 lines.
        lines = result.episodes_path.read_text().strip().split("\n")
        assert len(lines) == 3
        for line in lines:
            rec = json.loads(line)
            assert rec["actions_sent_to_robot"] == 0
            assert rec["actions_sent_to_simulator"] == 2

    def test_sim_report_safety_invariants(self, tmp_path, valid_manifest_dict):
        output_dir = tmp_path / "run"
        mock_policy = _make_mock_policy(valid_manifest_dict)

        with patch("lerobot_coreai.sim.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            config = SimConfig(
                policy_path="test/policy",
                output_dir=output_dir,
                env_type="fake",
                confirm_sim_egress=True,
                episodes=1,
                max_steps_per_episode=3,
                fps=0,
            )
            result = run_sim_mode(config)

        s = result.report["safety"]
        assert s["simulator_egress_enabled"] is True
        assert s["robot_egress_enabled"] is False
        assert s["physical_actuation_possible"] is False
        assert s["motor_commands_available"] is False
        assert s["robot_connected"] is False
        assert s["actions_sent_to_robot"] == 0
        assert s["action_egress"] == "simulator_only"

    def test_sim_claims_never_claim_real_success(self, tmp_path, valid_manifest_dict):
        output_dir = tmp_path / "run"
        mock_policy = _make_mock_policy(valid_manifest_dict)

        with patch("lerobot_coreai.sim.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            config = SimConfig(
                policy_path="test/policy",
                output_dir=output_dir,
                env_type="fake",
                confirm_sim_egress=True,
                episodes=1,
                max_steps_per_episode=3,
                fps=0,
            )
            result = run_sim_mode(config)

        c = result.report["claims"]
        assert c["proves_real_task_success"] is False
        assert c["proves_robot_safety"] is False
        assert c["proves_real_world_safety"] is False


class TestSimConfirmGate:
    def test_confirm_required(self, tmp_path, valid_manifest_dict):
        """Sim mode must refuse to start without --confirm-sim-egress."""
        config = SimConfig(
            policy_path="test/policy",
            output_dir=tmp_path / "run",
            env_type="fake",
            confirm_sim_egress=False,
        )
        with pytest.raises(CoreAIPolicyError, match="confirm-sim-egress"):
            run_sim_mode(config)

    def test_confirm_message_mentions_no_robot_commands(self, tmp_path, valid_manifest_dict):
        config = SimConfig(
            policy_path="test/policy",
            output_dir=tmp_path / "run",
            env_type="fake",
            confirm_sim_egress=False,
        )
        with pytest.raises(CoreAIPolicyError, match="No robot commands were sent"):
            run_sim_mode(config)


class TestSimFailFast:
    def test_fail_fast_false_continues_on_runner_error(self, tmp_path, valid_manifest_dict):
        output_dir = tmp_path / "run"
        mock_policy = _make_mock_policy(valid_manifest_dict)
        # First call fails, then succeeds.
        mock_policy.predict_action.side_effect = [
            RuntimeError("transient"),
            {"action": [[0.0] * 7] * 16, "metadata": {"timing": {"total_ms": 5.0}}},
        ]

        with patch("lerobot_coreai.sim.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            config = SimConfig(
                policy_path="test/policy",
                output_dir=output_dir,
                env_type="fake",
                confirm_sim_egress=True,
                episodes=1,
                max_steps_per_episode=3,
                fps=0,
                fail_fast=False,
            )
            result = run_sim_mode(config)

        # Step 0 failed (runner error), but the run continued.
        assert result.ok is True
        assert result.report["metrics"]["runner_errors"] >= 1

    def test_fail_fast_true_raises_on_runner_error(self, tmp_path, valid_manifest_dict):
        output_dir = tmp_path / "run"
        mock_policy = _make_mock_policy(valid_manifest_dict)
        mock_policy.predict_action.side_effect = RuntimeError("runner down")

        with patch("lerobot_coreai.sim.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            config = SimConfig(
                policy_path="test/policy",
                output_dir=output_dir,
                env_type="fake",
                confirm_sim_egress=True,
                episodes=1,
                max_steps_per_episode=3,
                fps=0,
                fail_fast=True,
            )
            with pytest.raises(RuntimeError):
                run_sim_mode(config)

        # Failure report still written.
        report = json.loads((output_dir / "sim_report.json").read_text())
        assert report["ok"] is False
        assert report["safety"]["actions_sent_to_robot"] == 0


class TestSimEnvironmentClose:
    def test_env_close_called_on_success(self, tmp_path, valid_manifest_dict):
        output_dir = tmp_path / "run"
        mock_policy = _make_mock_policy(valid_manifest_dict)

        closed = {"called": False}
        with patch("lerobot_coreai.sim.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            config = SimConfig(
                policy_path="test/policy",
                output_dir=output_dir,
                env_type="fake",
                confirm_sim_egress=True,
                episodes=1,
                max_steps_per_episode=2,
                fps=0,
            )
            result = run_sim_mode(config)

        assert result.report["environment"]["closed"] is True

    def test_env_close_called_on_failure(self, tmp_path, valid_manifest_dict):
        output_dir = tmp_path / "run"
        mock_policy = _make_mock_policy(valid_manifest_dict)
        mock_policy.predict_action.side_effect = RuntimeError("boom")

        with patch("lerobot_coreai.sim.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            config = SimConfig(
                policy_path="test/policy",
                output_dir=output_dir,
                env_type="fake",
                confirm_sim_egress=True,
                episodes=1,
                max_steps_per_episode=2,
                fps=0,
                fail_fast=True,
            )
            with pytest.raises(RuntimeError):
                run_sim_mode(config)

        report = json.loads((output_dir / "sim_report.json").read_text())
        assert report["environment"]["closed"] is True


class TestSimOverwrite:
    def test_overwrite_required_for_nonempty_dir(self, tmp_path, valid_manifest_dict):
        output_dir = tmp_path / "run"
        output_dir.mkdir()
        (output_dir / "existing.txt").write_text("data")

        config = SimConfig(
            policy_path="test/policy",
            output_dir=output_dir,
            env_type="fake",
            confirm_sim_egress=True,
        )
        with pytest.raises(CoreAIPolicyError, match="not empty"):
            run_sim_mode(config)

    def test_overwrite_replaces(self, tmp_path, valid_manifest_dict):
        output_dir = tmp_path / "run"
        output_dir.mkdir()
        (output_dir / "existing.txt").write_text("data")
        mock_policy = _make_mock_policy(valid_manifest_dict)

        with patch("lerobot_coreai.sim.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            config = SimConfig(
                policy_path="test/policy",
                output_dir=output_dir,
                env_type="fake",
                confirm_sim_egress=True,
                episodes=1,
                max_steps_per_episode=2,
                fps=0,
                overwrite=True,
            )
            result = run_sim_mode(config)

        assert result.ok is True


class TestSimActionsJsonl:
    def test_actions_jsonl_has_egress_block(self, tmp_path, valid_manifest_dict):
        output_dir = tmp_path / "run"
        mock_policy = _make_mock_policy(valid_manifest_dict)

        with patch("lerobot_coreai.sim.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            config = SimConfig(
                policy_path="test/policy",
                output_dir=output_dir,
                env_type="fake",
                confirm_sim_egress=True,
                episodes=1,
                max_steps_per_episode=2,
                fps=0,
            )
            result = run_sim_mode(config)

        lines = result.actions_path.read_text().strip().split("\n")
        for line in lines:
            rec = json.loads(line)
            assert "egress" in rec
            if rec["ok"]:
                assert rec["egress"]["sent_to_simulator"] is True
                assert rec["egress"]["sent_to_robot"] is False
                assert rec["egress"]["destination"] == "simulator"
            assert "reward" in rec
            assert "done" in rec
