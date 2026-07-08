# test_eval.py — tests for run_lerobot_dataset_eval with mocked dataset + policy.

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from lerobot_coreai.eval import EvalConfig, run_lerobot_dataset_eval
from lerobot_coreai.errors import CoreAIPolicyError


def _mock_dataset(num_frames=10):
    """Create a mock dataset that returns items with the right keys."""
    ds = MagicMock()
    ds.__len__ = MagicMock(return_value=num_frames)
    ds.__getitem__ = MagicMock(side_effect=lambda idx: {
        "observation.images.wrist": f"/tmp/img_{idx}.png",
        "observation.state": [0.0] * 7,
        "task": "pick up the cube",
        "action": [0.0] * 7,
    })
    return ds


class TestEvalSuccess:
    def test_eval_processes_frames(self, tmp_path, valid_manifest_dict):
        from lerobot_coreai.manifest import LeRobotCoreAIManifest

        mock_ds = _mock_dataset(10)
        mock_policy = MagicMock()
        action = [[0.01] * 7 for _ in range(16)]
        mock_policy.predict_action.return_value = {"action": action, "metadata": {"timing": {"total_ms": 12.0}}}
        mock_policy.manifest = LeRobotCoreAIManifest.from_dict(valid_manifest_dict)
        mock_policy.policy_type = "evo1"
        mock_policy.robot_type = "so100"
        mock_policy.parity_passed = True

        with patch("lerobot_coreai.eval.load_lerobot_dataset", return_value=mock_ds), \
             patch("lerobot_coreai.eval.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            config = EvalConfig(
                policy_path="kevinqz/EVO1-SO100-CoreAI",
                dataset_repo_id="lerobot/test",
                runner_url="http://localhost:8710",
                output_dir=tmp_path / "eval",
                max_frames=5,
            )
            result = run_lerobot_dataset_eval(config)

        assert result.ok is True
        assert result.actions_path.exists()
        assert result.trace_path.exists()
        assert result.report_path.exists()
        assert result.report["metrics"]["actions_generated"] == 5
        assert result.report["metrics"]["actions_failed"] == 0

    def test_eval_report_safety_invariants(self, tmp_path, valid_manifest_dict):
        from lerobot_coreai.manifest import LeRobotCoreAIManifest

        mock_ds = _mock_dataset(5)
        mock_policy = MagicMock()
        mock_policy.predict_action.return_value = {"action": [[0.0]*7]*16, "metadata": {}}
        mock_policy.manifest = LeRobotCoreAIManifest.from_dict(valid_manifest_dict)
        mock_policy.policy_type = "evo1"
        mock_policy.robot_type = "so100"
        mock_policy.parity_passed = True

        with patch("lerobot_coreai.eval.load_lerobot_dataset", return_value=mock_ds), \
             patch("lerobot_coreai.eval.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            config = EvalConfig(
                policy_path="test", dataset_repo_id="test",
                runner_url="http://x", output_dir=tmp_path / "e",
                max_frames=3,
            )
            result = run_lerobot_dataset_eval(config)

        assert result.report["safety"]["actions_sent"] == 0
        assert result.report["safety"]["physical_actuation_possible"] is False
        assert result.report["safety"]["motor_commands_available"] is False
        assert result.report["safety"]["robot_connected"] is False

    def test_eval_actions_jsonl_has_entries(self, tmp_path, valid_manifest_dict):
        from lerobot_coreai.manifest import LeRobotCoreAIManifest

        mock_ds = _mock_dataset(5)
        mock_policy = MagicMock()
        mock_policy.predict_action.return_value = {"action": [[0.0]*7]*16, "metadata": {}}
        mock_policy.manifest = LeRobotCoreAIManifest.from_dict(valid_manifest_dict)
        mock_policy.policy_type = "evo1"
        mock_policy.robot_type = "so100"
        mock_policy.parity_passed = True

        with patch("lerobot_coreai.eval.load_lerobot_dataset", return_value=mock_ds), \
             patch("lerobot_coreai.eval.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            config = EvalConfig(
                policy_path="test", dataset_repo_id="test",
                runner_url="http://x", output_dir=tmp_path / "e",
                max_frames=3,
            )
            result = run_lerobot_dataset_eval(config)

        lines = result.actions_path.read_text().strip().split("\n")
        assert len(lines) == 3
        data = json.loads(lines[0])
        assert "action" in data
        assert data["ok"] is True


class TestEvalFailures:
    def test_eval_frame_error_continues_non_fail_fast(self, tmp_path, valid_manifest_dict):
        from lerobot_coreai.manifest import LeRobotCoreAIManifest
        from lerobot_coreai.errors import ObservationValidationError

        mock_ds = _mock_dataset(10)
        mock_policy = MagicMock()
        mock_policy.manifest = LeRobotCoreAIManifest.from_dict(valid_manifest_dict)
        mock_policy.policy_type = "evo1"
        mock_policy.robot_type = "so100"
        mock_policy.parity_passed = True
        # First call fails, rest succeed.
        mock_policy.predict_action.side_effect = [
            ObservationValidationError("bad obs"),
            {"action": [[0.0]*7]*16, "metadata": {}},
            {"action": [[0.0]*7]*16, "metadata": {}},
            {"action": [[0.0]*7]*16, "metadata": {}},
            {"action": [[0.0]*7]*16, "metadata": {}},
        ]

        with patch("lerobot_coreai.eval.load_lerobot_dataset", return_value=mock_ds), \
             patch("lerobot_coreai.eval.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            config = EvalConfig(
                policy_path="test", dataset_repo_id="test",
                runner_url="http://x", output_dir=tmp_path / "e",
                max_frames=5, fail_fast=False,
            )
            result = run_lerobot_dataset_eval(config)

        assert result.report["metrics"]["actions_failed"] == 1
        assert result.report["metrics"]["actions_generated"] == 4
        assert result.ok is False  # had failures

    def test_eval_fail_fast_stops_on_first(self, tmp_path, valid_manifest_dict):
        from lerobot_coreai.manifest import LeRobotCoreAIManifest
        from lerobot_coreai.errors import ObservationValidationError

        mock_ds = _mock_dataset(10)
        mock_policy = MagicMock()
        mock_policy.manifest = LeRobotCoreAIManifest.from_dict(valid_manifest_dict)
        mock_policy.policy_type = "evo1"
        mock_policy.robot_type = "so100"
        mock_policy.parity_passed = True
        mock_policy.predict_action.side_effect = ObservationValidationError("bad obs")

        with patch("lerobot_coreai.eval.load_lerobot_dataset", return_value=mock_ds), \
             patch("lerobot_coreai.eval.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            config = EvalConfig(
                policy_path="test", dataset_repo_id="test",
                runner_url="http://x", output_dir=tmp_path / "e",
                max_frames=5, fail_fast=True,
            )
            with pytest.raises(ObservationValidationError):
                run_lerobot_dataset_eval(config)


class FakeTensor:
    """Simulates a torch.Tensor for serialization testing."""
    def __init__(self, data):
        self._data = data
    def detach(self):
        return self
    def cpu(self):
        return self
    def tolist(self):
        return self._data


class TestEvalSerialization:
    def test_eval_serializes_tensor_state_before_predict(self, tmp_path, valid_manifest_dict):
        """Integration test: eval must serialize tensors to lists before calling predict_action."""
        from lerobot_coreai.manifest import LeRobotCoreAIManifest

        # Mock dataset returns FakeTensor for observation.state.
        mock_ds = MagicMock()
        mock_ds.__len__ = MagicMock(return_value=5)
        mock_ds.__getitem__ = MagicMock(side_effect=lambda idx: {
            "observation.images.wrist": f"/tmp/img_{idx}.png",
            "observation.state": FakeTensor([float(idx)] * 7),  # tensor-like
            "task": "pick up the cube",
        })

        mock_policy = MagicMock()
        mock_policy.predict_action.return_value = {"action": [[0.0]*7]*16, "metadata": {}}
        mock_policy.manifest = LeRobotCoreAIManifest.from_dict(valid_manifest_dict)
        mock_policy.policy_type = "evo1"
        mock_policy.robot_type = "so100"
        mock_policy.parity_passed = True

        with patch("lerobot_coreai.eval.load_lerobot_dataset", return_value=mock_ds), \
             patch("lerobot_coreai.eval.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            config = EvalConfig(
                policy_path="test", dataset_repo_id="test",
                runner_url="http://x", output_dir=tmp_path / "e",
                max_frames=1,
            )
            run_lerobot_dataset_eval(config)

        # Verify predict_action received a serialized list, not a FakeTensor.
        called_batch = mock_policy.predict_action.call_args.args[0]
        assert called_batch["observation.state"] == [0.0] * 7
        assert not isinstance(called_batch["observation.state"], FakeTensor)
