# test_compare.py — tests for run_lerobot_policy_compare with mocks.

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from lerobot_coreai.compare import CompareConfig, run_lerobot_policy_compare
from lerobot_coreai.errors import ActionParityError


def _mock_dataset(num_frames=10):
    ds = MagicMock()
    ds.__len__ = MagicMock(return_value=num_frames)
    ds.__getitem__ = MagicMock(side_effect=lambda idx: {
        "observation.images.wrist": f"/tmp/img_{idx}.png",
        "observation.state": [0.0] * 7,
        "task": "pick up the cube",
        "action": [0.0] * 7,
    })
    return ds


def _make_mocks(valid_manifest_dict, action=None):
    from lerobot_coreai.manifest import LeRobotCoreAIManifest
    action = action or [[0.0] * 7 for _ in range(16)]
    mock_coreai = MagicMock()
    mock_coreai.predict_action.return_value = {"action": action, "metadata": {"timing": {"total_ms": 12.0}}}
    mock_coreai.manifest = LeRobotCoreAIManifest.from_dict(valid_manifest_dict)
    mock_coreai.policy_type = "evo1"
    mock_coreai.robot_type = "so100"
    mock_coreai.parity_passed = True

    mock_torch = MagicMock()
    mock_torch.select_action.return_value = action  # identical action

    return mock_coreai, mock_torch


class TestCompareSuccess:
    def test_compare_identical_actions_ok(self, tmp_path, valid_manifest_dict):
        mock_coreai, mock_torch = _make_mocks(valid_manifest_dict)
        mock_ds = _mock_dataset(10)

        with patch("lerobot_coreai.compare.load_lerobot_dataset", return_value=mock_ds), \
             patch("lerobot_coreai.compare.CoreAIPolicy.from_pretrained", return_value=mock_coreai), \
             patch("lerobot_coreai.compare.load_lerobot_policy", return_value=mock_torch):
            config = CompareConfig(
                torch_policy_path="lerobot/evo1",
                coreai_policy_path="kevinqz/EVO1-CoreAI",
                dataset_repo_id="lerobot/test",
                runner_url="http://x",
                output_dir=tmp_path / "compare",
                max_frames=3,
            )
            result = run_lerobot_policy_compare(config)

        assert result.ok is True
        assert result.report["metrics"]["frames_compared"] == 3
        assert result.report["metrics"]["frames_passed"] == 3
        assert result.report["claims"]["proves_numeric_action_fidelity"] is True

    def test_compare_writes_files(self, tmp_path, valid_manifest_dict):
        mock_coreai, mock_torch = _make_mocks(valid_manifest_dict)
        mock_ds = _mock_dataset(5)

        with patch("lerobot_coreai.compare.load_lerobot_dataset", return_value=mock_ds), \
             patch("lerobot_coreai.compare.CoreAIPolicy.from_pretrained", return_value=mock_coreai), \
             patch("lerobot_coreai.compare.load_lerobot_policy", return_value=mock_torch):
            config = CompareConfig(
                torch_policy_path="t", coreai_policy_path="c",
                dataset_repo_id="d", runner_url="http://x",
                output_dir=tmp_path / "cmp", max_frames=2,
            )
            result = run_lerobot_policy_compare(config)

        assert result.actions_path.exists()
        assert result.trace_path.exists()
        assert result.report_path.exists()

    def test_compare_safety_invariants(self, tmp_path, valid_manifest_dict):
        mock_coreai, mock_torch = _make_mocks(valid_manifest_dict)
        mock_ds = _mock_dataset(5)

        with patch("lerobot_coreai.compare.load_lerobot_dataset", return_value=mock_ds), \
             patch("lerobot_coreai.compare.CoreAIPolicy.from_pretrained", return_value=mock_coreai), \
             patch("lerobot_coreai.compare.load_lerobot_policy", return_value=mock_torch):
            config = CompareConfig(
                torch_policy_path="t", coreai_policy_path="c",
                dataset_repo_id="d", runner_url="http://x",
                output_dir=tmp_path / "cmp", max_frames=1,
            )
            result = run_lerobot_policy_compare(config)

        assert result.report["safety"]["actions_sent"] == 0
        assert result.report["safety"]["physical_actuation_possible"] is False
        assert result.report["safety"]["motor_commands_available"] is False
        assert result.report["safety"]["robot_connected"] is False

    def test_compare_manifest_patch_when_passed(self, tmp_path, valid_manifest_dict):
        mock_coreai, mock_torch = _make_mocks(valid_manifest_dict)
        mock_ds = _mock_dataset(5)

        with patch("lerobot_coreai.compare.load_lerobot_dataset", return_value=mock_ds), \
             patch("lerobot_coreai.compare.CoreAIPolicy.from_pretrained", return_value=mock_coreai), \
             patch("lerobot_coreai.compare.load_lerobot_policy", return_value=mock_torch):
            config = CompareConfig(
                torch_policy_path="t", coreai_policy_path="c",
                dataset_repo_id="d", runner_url="http://x",
                output_dir=tmp_path / "cmp", max_frames=1,
            )
            result = run_lerobot_policy_compare(config)

        patch_path = tmp_path / "cmp" / "manifest-evaluation-patch.json"
        assert patch_path.exists()
        eval_data = json.loads(patch_path.read_text())
        assert eval_data["evaluation"]["status"] == "passed"

    def test_compare_reset_called(self, tmp_path, valid_manifest_dict):
        mock_coreai, mock_torch = _make_mocks(valid_manifest_dict)
        mock_ds = _mock_dataset(5)

        with patch("lerobot_coreai.compare.load_lerobot_dataset", return_value=mock_ds), \
             patch("lerobot_coreai.compare.CoreAIPolicy.from_pretrained", return_value=mock_coreai), \
             patch("lerobot_coreai.compare.load_lerobot_policy", return_value=mock_torch):
            config = CompareConfig(
                torch_policy_path="t", coreai_policy_path="c",
                dataset_repo_id="d", runner_url="http://x",
                output_dir=tmp_path / "cmp", max_frames=1,
            )
            run_lerobot_policy_compare(config)

        mock_torch.reset.assert_called_once()
        mock_coreai.reset.assert_called_once()


class TestCompareMismatch:
    def test_compare_action_mismatch_fails(self, tmp_path, valid_manifest_dict):
        torch_action = [[0.0] * 7] * 16
        coreai_action = [[1.0] * 7] * 16  # very different
        mock_coreai, mock_torch = _make_mocks(valid_manifest_dict, action=coreai_action)
        mock_torch.select_action.return_value = torch_action
        mock_ds = _mock_dataset(5)

        with patch("lerobot_coreai.compare.load_lerobot_dataset", return_value=mock_ds), \
             patch("lerobot_coreai.compare.CoreAIPolicy.from_pretrained", return_value=mock_coreai), \
             patch("lerobot_coreai.compare.load_lerobot_policy", return_value=mock_torch):
            config = CompareConfig(
                torch_policy_path="t", coreai_policy_path="c",
                dataset_repo_id="d", runner_url="http://x",
                output_dir=tmp_path / "cmp", max_frames=3,
                tolerance_cosine=0.9999,
            )
            result = run_lerobot_policy_compare(config)

        assert result.ok is False
        assert result.report["metrics"]["frames_failed"] > 0

    def test_compare_shape_mismatch_counted(self, tmp_path, valid_manifest_dict):
        torch_action = [[0.0] * 7] * 16
        coreai_action = [[0.0] * 7]  # wrong shape [1, 7] vs [16, 7]
        mock_coreai, mock_torch = _make_mocks(valid_manifest_dict, action=coreai_action)
        mock_torch.select_action.return_value = torch_action
        mock_ds = _mock_dataset(5)

        with patch("lerobot_coreai.compare.load_lerobot_dataset", return_value=mock_ds), \
             patch("lerobot_coreai.compare.CoreAIPolicy.from_pretrained", return_value=mock_coreai), \
             patch("lerobot_coreai.compare.load_lerobot_policy", return_value=mock_torch):
            config = CompareConfig(
                torch_policy_path="t", coreai_policy_path="c",
                dataset_repo_id="d", runner_url="http://x",
                output_dir=tmp_path / "cmp", max_frames=3,
            )
            result = run_lerobot_policy_compare(config)

        assert result.report["metrics"]["shape_mismatches"] > 0

    def test_compare_fail_fast_stops(self, tmp_path, valid_manifest_dict):
        torch_action = [[0.0] * 7] * 16
        coreai_action = [[1.0] * 7] * 16
        mock_coreai, mock_torch = _make_mocks(valid_manifest_dict, action=coreai_action)
        mock_torch.select_action.return_value = torch_action
        mock_ds = _mock_dataset(10)

        with patch("lerobot_coreai.compare.load_lerobot_dataset", return_value=mock_ds), \
             patch("lerobot_coreai.compare.CoreAIPolicy.from_pretrained", return_value=mock_coreai), \
             patch("lerobot_coreai.compare.load_lerobot_policy", return_value=mock_torch):
            config = CompareConfig(
                torch_policy_path="t", coreai_policy_path="c",
                dataset_repo_id="d", runner_url="http://x",
                output_dir=tmp_path / "cmp", max_frames=5,
                tolerance_cosine=0.9999, fail_fast=True,
            )
            with pytest.raises(ActionParityError):
                run_lerobot_policy_compare(config)

    def test_compare_non_fail_fast_continues(self, tmp_path, valid_manifest_dict):
        torch_action = [[0.0] * 7] * 16
        coreai_action = [[1.0] * 7] * 16
        mock_coreai, mock_torch = _make_mocks(valid_manifest_dict, action=coreai_action)
        mock_torch.select_action.return_value = torch_action
        mock_ds = _mock_dataset(10)

        with patch("lerobot_coreai.compare.load_lerobot_dataset", return_value=mock_ds), \
             patch("lerobot_coreai.compare.CoreAIPolicy.from_pretrained", return_value=mock_coreai), \
             patch("lerobot_coreai.compare.load_lerobot_policy", return_value=mock_torch):
            config = CompareConfig(
                torch_policy_path="t", coreai_policy_path="c",
                dataset_repo_id="d", runner_url="http://x",
                output_dir=tmp_path / "cmp", max_frames=5,
                tolerance_cosine=0.9999, fail_fast=False,
            )
            result = run_lerobot_policy_compare(config)

        # All 5 frames failed but didn't stop.
        assert result.report["metrics"]["frames_compared"] == 5
        assert result.report["metrics"]["frames_failed"] == 5
