# test_rollout_dry_run.py — tests for run_dry_run_rollout with mocked policy.

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from lerobot_coreai.rollout import DryRunRolloutConfig, run_dry_run_rollout
from lerobot_coreai.errors import CoreAIPolicyError, RunnerNotReachableError


def _make_fixture(tmp_path):
    f = tmp_path / "obs.json"
    f.write_text(json.dumps({
        "observation.images.wrist": "wrist.png",
        "observation.state": [0.0] * 7,
        "task": "pick up the cube",
    }))
    return f


class TestDryRunRolloutSuccess:
    def test_dry_run_writes_all_files(self, tmp_path, valid_manifest_dict):
        """dry_run success writes action.json, observation.json, rollout_report.json, trace.jsonl."""
        from lerobot_coreai.manifest import LeRobotCoreAIManifest
        from lerobot_coreai.types import ActionPredictResponse

        fixture = _make_fixture(tmp_path)
        output_dir = tmp_path / "run1"

        mock_policy = MagicMock()
        action = [[0.01] * 7 for _ in range(16)]
        mock_policy.predict_action.return_value = {"action": action, "metadata": {"timing": {}}}
        mock_policy.manifest = LeRobotCoreAIManifest.from_dict(valid_manifest_dict)
        mock_policy.policy_type = "evo1"
        mock_policy.robot_type = "so100"
        mock_policy.parity_passed = True

        with patch("lerobot_coreai.rollout.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            config = DryRunRolloutConfig(
                policy_path="kevinqz/EVO1-SO100-CoreAI",
                robot_type="so100",
                fixture_path=fixture,
                runner_url="http://localhost:8710",
                output_dir=output_dir,
            )
            result = run_dry_run_rollout(config)

        assert result.ok is True
        assert result.action_path.exists()
        assert result.observation_path.exists()
        assert result.report_path.exists()
        assert result.trace_path.exists()

    def test_report_ok_true(self, tmp_path, valid_manifest_dict):
        from lerobot_coreai.manifest import LeRobotCoreAIManifest

        fixture = _make_fixture(tmp_path)
        mock_policy = MagicMock()
        mock_policy.predict_action.return_value = {"action": [[0.0]*7]*16, "metadata": {}}
        mock_policy.manifest = LeRobotCoreAIManifest.from_dict(valid_manifest_dict)
        mock_policy.policy_type = "evo1"
        mock_policy.robot_type = "so100"
        mock_policy.parity_passed = True

        with patch("lerobot_coreai.rollout.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            config = DryRunRolloutConfig(
                policy_path="test", robot_type="so100",
                fixture_path=fixture, runner_url="http://x",
                output_dir=tmp_path / "run",
            )
            result = run_dry_run_rollout(config)

        assert result.report["ok"] is True
        assert result.report["action"]["generated"] is True
        assert result.report["robot"]["connected"] is False
        assert result.report["robot"]["actions_sent"] == 0
        assert result.report["safety"]["physical_actuation_possible"] is False
        assert result.report["safety"]["motor_commands_available"] is False

    def test_output_dir_non_empty_fails(self, tmp_path, valid_manifest_dict):
        fixture = _make_fixture(tmp_path)
        output_dir = tmp_path / "existing"
        output_dir.mkdir()
        (output_dir / "stale.txt").write_text("data")

        with pytest.raises(CoreAIPolicyError, match="not empty"):
            config = DryRunRolloutConfig(
                policy_path="test", fixture_path=fixture,
                runner_url="http://x", output_dir=output_dir,
            )
            run_dry_run_rollout(config)


class TestDryRunRolloutFailure:
    def test_runner_error_writes_failure_report(self, tmp_path, valid_manifest_dict):
        from lerobot_coreai.manifest import LeRobotCoreAIManifest

        fixture = _make_fixture(tmp_path)
        mock_policy = MagicMock()
        mock_policy.predict_action.side_effect = RunnerNotReachableError("down")
        mock_policy.manifest = LeRobotCoreAIManifest.from_dict(valid_manifest_dict)
        mock_policy.policy_type = "evo1"
        mock_policy.robot_type = "so100"
        mock_policy.parity_passed = True

        with patch("lerobot_coreai.rollout.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            config = DryRunRolloutConfig(
                policy_path="test", robot_type="so100",
                fixture_path=fixture, runner_url="http://x",
                output_dir=tmp_path / "run",
            )
            with pytest.raises(RunnerNotReachableError):
                run_dry_run_rollout(config)

        report_path = tmp_path / "run" / "rollout_report.json"
        assert report_path.exists()
        report = json.loads(report_path.read_text())
        assert report["ok"] is False
        assert report["robot"]["actions_sent"] == 0
        assert report["safety"]["physical_actuation_possible"] is False
