# test_cli_shadow_camera.py — tests for camera source CLI args.

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from lerobot_coreai import cli
from lerobot_coreai.shadow import ShadowResult


def _mock_shadow_result(tmp_path):
    r = MagicMock(spec=ShadowResult)
    r.ok = True
    r.report = {
        "ok": True,
        "schema_version": "lerobot-coreai.shadow_report.v0",
        "metrics": {"observations_read": 1, "actions_generated": 1, "actions_blocked": 1, "actions_sent": 0},
    }
    r.report_path = tmp_path / "shadow_report.json"
    r.trace_path = tmp_path / "shadow_trace.jsonl"
    r.actions_path = tmp_path / "actions.jsonl"
    r.observations_path = tmp_path / "observations.jsonl"
    r.blocked_actions_path = tmp_path / "blocked_actions.jsonl"
    return r


class TestCLIShadowCamera:
    def test_camera_args_parsed(self, tmp_path):
        """--camera.* args should be passed into ShadowConfig."""
        mock_result = _mock_shadow_result(tmp_path)
        with patch("lerobot_coreai.cli.run_shadow_mode", return_value=mock_result) as mock_run:
            cli.main([
                "shadow",
                "--policy.path", "test",
                "--observation-source", "camera",
                "--camera.index", "2",
                "--camera.width", "1280",
                "--camera.height", "720",
                "--camera.fps", "30",
                "--output-dir", str(tmp_path / "run"),
            ])
        config = mock_run.call_args[0][0]
        assert config.camera_index == 2
        assert config.camera_width == 1280
        assert config.camera_height == 720
        assert config.camera_fps == 30.0
        assert config.observation_source == "camera"

    def test_no_save_camera_frames_flag_removed(self, tmp_path, capsys):
        """--no-save-camera-frames is no longer accepted (frames always saved)."""
        with pytest.raises(SystemExit):
            cli.main([
                "shadow",
                "--policy.path", "test",
                "--observation-source", "camera",
                "--no-save-camera-frames",
                "--output-dir", str(tmp_path / "run"),
            ])
        err = capsys.readouterr().err
        assert "unrecognized arguments" in err or "unknown" in err.lower()

    def test_camera_index_default_zero(self, tmp_path):
        """Default camera index should be 0."""
        mock_result = _mock_shadow_result(tmp_path)
        with patch("lerobot_coreai.cli.run_shadow_mode", return_value=mock_result) as mock_run:
            cli.main([
                "shadow",
                "--policy.path", "test",
                "--observation-source", "camera",
                "--output-dir", str(tmp_path / "run"),
            ])
        config = mock_run.call_args[0][0]
        assert config.camera_index == 0

    def test_camera_error_prints_no_robot_commands(self, capsys):
        """If camera source fails, CLI should print 'No robot commands were sent.'"""
        from lerobot_coreai.errors import CoreAIPolicyError
        with patch("lerobot_coreai.cli.run_shadow_mode",
                   side_effect=CoreAIPolicyError("Could not open camera index 99")):
            rc = cli.main([
                "shadow",
                "--policy.path", "test",
                "--observation-source", "camera",
                "--camera.index", "99",
                "--output-dir", "/tmp/camera-run",
            ])
        assert rc == 1
        err = capsys.readouterr().err
        assert "No robot commands were sent" in err
