# test_shadow.py — tests for run_shadow_mode with mocked policy.

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from lerobot_coreai.shadow import ShadowConfig, run_shadow_mode
from lerobot_coreai.errors import CoreAIPolicyError


def _make_mock_policy(manifest_dict):
    from lerobot_coreai.manifest import LeRobotCoreAIManifest
    mock = MagicMock()
    action = [[0.01 * i] * 7 for i in range(16)]
    mock.predict_action.return_value = {
        "action": action,
        "metadata": {"timing": {"total_ms": 12.3}},
    }
    mock.manifest = LeRobotCoreAIManifest.from_dict(manifest_dict)
    mock.policy_type = "evo1"
    mock.robot_type = "so100"
    mock.parity_passed = True
    mock.policy_repo_id = "kevinqz/EVO1-SO100-CoreAI"
    return mock


def _make_fixture_dir(tmp_path, n=4):
    """Create a fixtures dir with N ordered fixtures."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (tmp_path / f"{i:06d}.json").write_text(json.dumps({
            "observation.state": [0.0] * 7,
            "observation.images.wrist": "wrist.png",
            "task": "pick up the cube",
        }))
    return tmp_path


class TestShadowSuccess:
    def test_shadow_writes_all_files(self, tmp_path, valid_manifest_dict):
        fixtures_dir = _make_fixture_dir(tmp_path / "fixtures", n=4)
        output_dir = tmp_path / "run"
        mock_policy = _make_mock_policy(valid_manifest_dict)

        with patch("lerobot_coreai.shadow.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            config = ShadowConfig(
                policy_path="kevinqz/EVO1-SO100-CoreAI",
                observation_source="fixtures",
                fixtures_dir=fixtures_dir,
                runner_url="http://localhost:8710",
                output_dir=output_dir,
                max_steps=4,
                fps=0,  # no sleeping in tests
            )
            result = run_shadow_mode(config)

        assert result.ok is True
        assert result.report_path.exists()
        assert result.trace_path.exists()
        assert result.actions_path.exists()
        assert result.observations_path.exists()
        assert result.blocked_actions_path.exists()

    def test_shadow_metrics_correct(self, tmp_path, valid_manifest_dict):
        fixtures_dir = _make_fixture_dir(tmp_path / "fixtures", n=4)
        output_dir = tmp_path / "run"
        mock_policy = _make_mock_policy(valid_manifest_dict)

        with patch("lerobot_coreai.shadow.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            config = ShadowConfig(
                policy_path="test",
                observation_source="fixtures",
                fixtures_dir=fixtures_dir,
                output_dir=output_dir,
                max_steps=4,
                fps=0,
            )
            result = run_shadow_mode(config)

        m = result.report["metrics"]
        assert m["observations_read"] == 4
        assert m["actions_generated"] == 4
        assert m["actions_blocked"] == 4
        assert m["actions_sent"] == 0
        assert m["observation_errors"] == 0
        assert m["runner_errors"] == 0

    def test_actions_sent_always_zero(self, tmp_path, valid_manifest_dict):
        fixtures_dir = _make_fixture_dir(tmp_path / "fixtures", n=2)
        mock_policy = _make_mock_policy(valid_manifest_dict)

        with patch("lerobot_coreai.shadow.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            config = ShadowConfig(
                policy_path="test",
                observation_source="fixtures",
                fixtures_dir=fixtures_dir,
                output_dir=tmp_path / "run",
                max_steps=2,
                fps=0,
            )
            result = run_shadow_mode(config)

        assert result.report["safety"]["actions_sent"] == 0
        assert result.report["metrics"]["actions_sent"] == 0

    def test_predict_action_called(self, tmp_path, valid_manifest_dict):
        fixtures_dir = _make_fixture_dir(tmp_path / "fixtures", n=3)
        mock_policy = _make_mock_policy(valid_manifest_dict)

        with patch("lerobot_coreai.shadow.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            config = ShadowConfig(
                policy_path="test",
                observation_source="fixtures",
                fixtures_dir=fixtures_dir,
                output_dir=tmp_path / "run",
                max_steps=3,
                fps=0,
            )
            run_shadow_mode(config)

        assert mock_policy.predict_action.call_count == 3

    def test_safety_invariants_in_report(self, tmp_path, valid_manifest_dict):
        fixtures_dir = _make_fixture_dir(tmp_path / "fixtures", n=2)
        mock_policy = _make_mock_policy(valid_manifest_dict)

        with patch("lerobot_coreai.shadow.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            config = ShadowConfig(
                policy_path="test",
                observation_source="fixtures",
                fixtures_dir=fixtures_dir,
                output_dir=tmp_path / "run",
                max_steps=2,
                fps=0,
            )
            result = run_shadow_mode(config)

        s = result.report["safety"]
        assert s["physical_actuation_possible"] is False
        assert s["motor_commands_available"] is False
        assert s["actuation_device_connected"] is False
        assert s["robot_connected"] is False
        assert s["actions_sent"] == 0
        assert s["action_egress"] == "blocked"

    def test_claims_in_report(self, tmp_path, valid_manifest_dict):
        fixtures_dir = _make_fixture_dir(tmp_path / "fixtures", n=2)
        mock_policy = _make_mock_policy(valid_manifest_dict)

        with patch("lerobot_coreai.shadow.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            config = ShadowConfig(
                policy_path="test",
                observation_source="fixtures",
                fixtures_dir=fixtures_dir,
                output_dir=tmp_path / "run",
                max_steps=2,
                fps=0,
            )
            result = run_shadow_mode(config)

        c = result.report["claims"]
        assert c["proves_runtime_action_generation"] is True
        assert c["proves_task_success"] is False
        assert c["proves_robot_safety"] is False
        assert c["proves_real_world_safety"] is False

    def test_source_closed_on_success(self, tmp_path, valid_manifest_dict):
        fixtures_dir = _make_fixture_dir(tmp_path / "fixtures", n=2)
        mock_policy = _make_mock_policy(valid_manifest_dict)

        with patch("lerobot_coreai.shadow.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            config = ShadowConfig(
                policy_path="test",
                observation_source="fixtures",
                fixtures_dir=fixtures_dir,
                output_dir=tmp_path / "run",
                max_steps=2,
                fps=0,
            )
            run_shadow_mode(config)

        # Verify trace contains observation_source.closed
        trace_text = (tmp_path / "run" / "shadow_trace.jsonl").read_text()
        assert "observation_source.closed" in trace_text


class TestShadowFailFast:
    def test_fail_fast_false_continues_after_error(self, tmp_path, valid_manifest_dict):
        """With fail_fast=False, a runner error mid-loop should not abort the run."""
        fixtures_dir = _make_fixture_dir(tmp_path / "fixtures", n=4)
        mock_policy = _make_mock_policy(valid_manifest_dict)
        # Fail on the 2nd call only.
        call_count = [0]
        original = mock_policy.predict_action.return_value

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("runner hiccup")
            return original

        mock_policy.predict_action.side_effect = side_effect

        with patch("lerobot_coreai.shadow.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            config = ShadowConfig(
                policy_path="test",
                observation_source="fixtures",
                fixtures_dir=fixtures_dir,
                output_dir=tmp_path / "run",
                max_steps=4,
                fps=0,
                fail_fast=False,
            )
            result = run_shadow_mode(config)

        # Run succeeded overall (best-effort), but 1 error was recorded.
        assert result.ok is True
        assert result.report["metrics"]["runner_errors"] == 1
        assert result.report["metrics"]["actions_generated"] == 3  # 4 reads - 1 failed
        assert len(result.report["errors"]) == 1

    def test_fail_fast_true_raises_on_step_error(self, tmp_path, valid_manifest_dict):
        fixtures_dir = _make_fixture_dir(tmp_path / "fixtures", n=4)
        mock_policy = _make_mock_policy(valid_manifest_dict)
        call_count = [0]
        original = mock_policy.predict_action.return_value

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("runner hiccup")
            return original

        mock_policy.predict_action.side_effect = side_effect

        with patch("lerobot_coreai.shadow.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            config = ShadowConfig(
                policy_path="test",
                observation_source="fixtures",
                fixtures_dir=fixtures_dir,
                output_dir=tmp_path / "run",
                max_steps=4,
                fps=0,
                fail_fast=True,
            )
            with pytest.raises(RuntimeError, match="runner hiccup"):
                run_shadow_mode(config)

    def test_source_closed_on_failure(self, tmp_path, valid_manifest_dict):
        """Source must be closed even when fail_fast raises."""
        fixtures_dir = _make_fixture_dir(tmp_path / "fixtures", n=4)
        mock_policy = _make_mock_policy(valid_manifest_dict)
        call_count = [0]
        original = mock_policy.predict_action.return_value

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("boom")
            return original

        mock_policy.predict_action.side_effect = side_effect

        with patch("lerobot_coreai.shadow.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            config = ShadowConfig(
                policy_path="test",
                observation_source="fixtures",
                fixtures_dir=fixtures_dir,
                output_dir=tmp_path / "run",
                max_steps=4,
                fps=0,
                fail_fast=True,
            )
            with pytest.raises(RuntimeError):
                run_shadow_mode(config)

        # Trace should still contain observation_source.closed.
        trace_text = (tmp_path / "run" / "shadow_trace.jsonl").read_text()
        assert "observation_source.closed" in trace_text


class TestShadowFolderSource:
    def test_shadow_with_folder_source(self, tmp_path, valid_manifest_dict):
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        for i in range(3):
            (frames_dir / f"{i:06d}.png").write_bytes(b"\x89PNG fake")
        mock_policy = _make_mock_policy(valid_manifest_dict)

        with patch("lerobot_coreai.shadow.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            config = ShadowConfig(
                policy_path="test",
                observation_source="folder",
                frames_dir=frames_dir,
                task="pick up the cube",
                output_dir=tmp_path / "run",
                max_steps=3,
                fps=0,
            )
            result = run_shadow_mode(config)

        assert result.ok is True
        assert result.report["metrics"]["actions_generated"] == 3
        assert result.report["observation_source"]["type"] == "folder"


class TestShadowOverwrite:
    def test_non_empty_dir_without_overwrite_raises(self, tmp_path, valid_manifest_dict):
        output_dir = tmp_path / "run"
        output_dir.mkdir()
        (output_dir / "existing.txt").write_text("data")

        mock_policy = _make_mock_policy(valid_manifest_dict)
        with patch("lerobot_coreai.shadow.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            config = ShadowConfig(
                policy_path="test",
                observation_source="fixtures",
                fixtures_dir=_make_fixture_dir(tmp_path / "fx", n=1),
                output_dir=output_dir,
                max_steps=1,
                fps=0,
            )
            with pytest.raises(CoreAIPolicyError, match="not empty"):
                run_shadow_mode(config)
