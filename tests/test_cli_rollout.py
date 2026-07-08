# test_cli_rollout.py — tests for the lerobot-coreai rollout CLI command.

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from lerobot_coreai import cli


class TestCLIRolloutModes:
    def test_rollout_real_blocked(self, capsys):
        """real mode should be blocked in v0.3."""
        rc = cli.main([
            "rollout", "--policy.path", "test",
            "--mode", "real",
            "--confirm-real-robot-actuation",
        ])
        assert rc == 1
        err = capsys.readouterr().err
        assert "not implemented" in err.lower() or "No robot commands" in err

    def test_rollout_shadow_blocked(self, capsys):
        rc = cli.main(["rollout", "--policy.path", "test", "--mode", "shadow"])
        assert rc == 1

    def test_rollout_sim_blocked(self, capsys):
        rc = cli.main(["rollout", "--policy.path", "test", "--mode", "sim"])
        assert rc == 1

    def test_rollout_dry_run_missing_fixture(self, capsys):
        """dry_run without --fixture should fail."""
        rc = cli.main([
            "rollout", "--policy.path", "test", "--mode", "dry_run",
        ])
        assert rc == 1
        assert "--fixture" in capsys.readouterr().err


class TestCLIRolloutDryRun:
    def test_rollout_dry_run_success(self, tmp_path, capsys, valid_manifest_dict):
        from lerobot_coreai.rollout import DryRunRolloutResult

        fixture = tmp_path / "obs.json"
        fixture.write_text(json.dumps({
            "observation.state": [0.0] * 7,
            "observation.images.wrist": "wrist.png",
        }))

        mock_result = MagicMock(spec=DryRunRolloutResult)
        mock_result.ok = True
        mock_result.report = {"ok": True, "schema_version": "lerobot-coreai.rollout_report.v0"}
        mock_result.action_path = tmp_path / "action.json"
        mock_result.observation_path = tmp_path / "observation.json"
        mock_result.trace_path = tmp_path / "trace.jsonl"
        mock_result.report_path = tmp_path / "rollout_report.json"

        with patch("lerobot_coreai.cli.run_dry_run_rollout", return_value=mock_result):
            rc = cli.main([
                "rollout",
                "--policy.path", "kevinqz/EVO1-SO100-CoreAI",
                "--robot.type", "so100",
                "--mode", "dry_run",
                "--fixture", str(fixture),
                "--runner.url", "http://localhost:8710",
                "--output-dir", str(tmp_path / "run"),
            ])
            assert rc == 0
            out = capsys.readouterr().out
            assert "No robot commands sent" in out
            assert "Dry-run completed" in out
