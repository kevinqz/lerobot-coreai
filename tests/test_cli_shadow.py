# test_cli_shadow.py — tests for the lerobot-coreai shadow CLI command.

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
        "metrics": {"observations_read": 3, "actions_generated": 3, "actions_blocked": 3, "actions_sent": 0},
    }
    r.report_path = tmp_path / "shadow_report.json"
    r.trace_path = tmp_path / "shadow_trace.jsonl"
    r.actions_path = tmp_path / "actions.jsonl"
    r.observations_path = tmp_path / "observations.jsonl"
    r.blocked_actions_path = tmp_path / "blocked_actions.jsonl"
    return r


class TestCLIShadow:
    def test_shadow_success_rc0(self, tmp_path, capsys):
        mock_result = _mock_shadow_result(tmp_path)
        with patch("lerobot_coreai.cli.run_shadow_mode", return_value=mock_result):
            rc = cli.main([
                "shadow",
                "--policy.path", "kevinqz/EVO1-SO100-CoreAI",
                "--observation-source", "folder",
                "--frames-dir", str(tmp_path / "frames"),
                "--runner.url", "http://localhost:8710",
                "--output-dir", str(tmp_path / "run"),
                "--max-steps", "3",
            ])
        assert rc == 0
        out = capsys.readouterr().out
        assert "No robot commands sent" in out
        assert "Shadow run completed" in out

    def test_shadow_failure_rc1(self, capsys):
        from lerobot_coreai.errors import CoreAIPolicyError
        with patch("lerobot_coreai.cli.run_shadow_mode",
                   side_effect=CoreAIPolicyError("runner not reachable")):
            rc = cli.main([
                "shadow",
                "--policy.path", "test",
                "--observation-source", "folder",
                "--frames-dir", "/tmp/none",
                "--output-dir", "/tmp/run",
            ])
        assert rc == 1
        err = capsys.readouterr().err
        assert "No robot commands were sent" in err

    def test_shadow_json_prints_report(self, tmp_path, capsys):
        mock_result = _mock_shadow_result(tmp_path)
        mock_result.report = {"ok": True, "mode": "shadow", "schema_version": "lerobot-coreai.shadow_report.v0"}
        with patch("lerobot_coreai.cli.run_shadow_mode", return_value=mock_result):
            rc = cli.main([
                "shadow",
                "--policy.path", "test",
                "--observation-source", "folder",
                "--frames-dir", str(tmp_path),
                "--output-dir", str(tmp_path / "run"),
                "--json",
            ])
        assert rc == 0
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert parsed["mode"] == "shadow"

    def test_max_steps_passed_through(self, tmp_path):
        mock_result = _mock_shadow_result(tmp_path)
        with patch("lerobot_coreai.cli.run_shadow_mode", return_value=mock_result) as mock_run:
            cli.main([
                "shadow",
                "--policy.path", "test",
                "--observation-source", "folder",
                "--frames-dir", str(tmp_path),
                "--output-dir", str(tmp_path / "run"),
                "--max-steps", "42",
            ])
        config = mock_run.call_args[0][0]
        assert config.max_steps == 42

    def test_fps_passed_through(self, tmp_path):
        mock_result = _mock_shadow_result(tmp_path)
        with patch("lerobot_coreai.cli.run_shadow_mode", return_value=mock_result) as mock_run:
            cli.main([
                "shadow",
                "--policy.path", "test",
                "--observation-source", "folder",
                "--frames-dir", str(tmp_path),
                "--output-dir", str(tmp_path / "run"),
                "--fps", "30",
            ])
        config = mock_run.call_args[0][0]
        assert config.fps == 30.0

    def test_folder_source_passes_frames_dir(self, tmp_path):
        mock_result = _mock_shadow_result(tmp_path)
        with patch("lerobot_coreai.cli.run_shadow_mode", return_value=mock_result) as mock_run:
            cli.main([
                "shadow",
                "--policy.path", "test",
                "--observation-source", "folder",
                "--frames-dir", str(tmp_path / "myframes"),
                "--output-dir", str(tmp_path / "run"),
            ])
        config = mock_run.call_args[0][0]
        assert config.frames_dir == tmp_path / "myframes"

    def test_state_vector_parsed(self, tmp_path):
        mock_result = _mock_shadow_result(tmp_path)
        with patch("lerobot_coreai.cli.run_shadow_mode", return_value=mock_result) as mock_run:
            cli.main([
                "shadow",
                "--policy.path", "test",
                "--observation-source", "folder",
                "--frames-dir", str(tmp_path),
                "--output-dir", str(tmp_path / "run"),
                "--state-vector", "0.1,0.2,0.3",
            ])
        config = mock_run.call_args[0][0]
        assert config.state_vector == [0.1, 0.2, 0.3]

    def test_unexpected_error_rc1(self, capsys):
        with patch("lerobot_coreai.cli.run_shadow_mode",
                   side_effect=RuntimeError("unexpected")):
            rc = cli.main([
                "shadow",
                "--policy.path", "test",
                "--observation-source", "folder",
                "--frames-dir", "/tmp/x",
                "--output-dir", "/tmp/r",
            ])
        assert rc == 1
        assert "No robot commands were sent" in capsys.readouterr().err
