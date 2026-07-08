# test_cli_predict.py — tests for the lerobot-coreai predict CLI command.

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from lerobot_coreai import cli
from lerobot_coreai.manifest import LeRobotCoreAIManifest


class TestCLIPredict:
    def test_predict_fixture_missing(self, capsys):
        """Missing fixture file should return rc 1."""
        rc = cli.main([
            "predict",
            "--policy.path", "kevinqz/EVO1-SO100-CoreAI",
            "--observation", "/nonexistent/obs.json",
            "--runner.url", "http://localhost:8710",
        ])
        assert rc == 1

    def test_predict_success(self, tmp_path, capsys, valid_manifest_dict):
        """Successful predict with mocked policy."""
        fixture = tmp_path / "obs.json"
        fixture.write_text(json.dumps({
            "observation.images.wrist": "/tmp/wrist.png",
            "observation.state": [0.0] * 7,
        }))

        mock_policy = MagicMock()
        mock_policy.select_action.return_value = {"action": [[0.01] * 7]}

        with patch("lerobot_coreai.policy.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            rc = cli.main([
                "predict",
                "--policy.path", "kevinqz/EVO1-SO100-CoreAI",
                "--observation", str(fixture),
                "--runner.url", "http://localhost:8710",
                "--json",
            ])
            assert rc == 0
            out = capsys.readouterr().out
            data = json.loads(out)
            assert "action" in data

    def test_predict_output_file(self, tmp_path, valid_manifest_dict):
        """--output writes action to file."""
        fixture = tmp_path / "obs.json"
        fixture.write_text(json.dumps({"observation.state": [0.0] * 7}))

        output_file = tmp_path / "action.json"
        mock_policy = MagicMock()
        mock_policy.select_action.return_value = {"action": [[0.0] * 7]}

        with patch("lerobot_coreai.policy.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            rc = cli.main([
                "predict",
                "--policy.path", "kevinqz/EVO1-SO100-CoreAI",
                "--observation", str(fixture),
                "--runner.url", "http://localhost:8710",
                "--output", str(output_file),
            ])
            assert rc == 0
            assert output_file.exists()
            data = json.loads(output_file.read_text())
            assert "action" in data

    def test_predict_policy_error_prints_no_robot(self, tmp_path, capsys, valid_manifest_dict):
        """On policy/runner error, stderr should say 'No robot commands were sent.'"""
        from lerobot_coreai.errors import RunnerNotReachableError

        fixture = tmp_path / "obs.json"
        fixture.write_text(json.dumps({"observation.state": [0.0] * 7}))

        mock_policy = MagicMock()
        mock_policy.select_action.side_effect = RunnerNotReachableError("down")

        with patch("lerobot_coreai.policy.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            rc = cli.main([
                "predict",
                "--policy.path", "kevinqz/EVO1-SO100-CoreAI",
                "--observation", str(fixture),
                "--runner.url", "http://localhost:8710",
            ])
            assert rc == 1
            err = capsys.readouterr().err
            assert "No robot commands were sent" in err
