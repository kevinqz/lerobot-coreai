# test_cli.py — tests for the lerobot-coreai CLI.

import json
from unittest.mock import patch, MagicMock

import pytest

from lerobot_coreai import cli
from lerobot_coreai.manifest import LeRobotCoreAIManifest


class TestCLIParser:
    def test_version_flag(self):
        with pytest.raises(SystemExit):
            cli.main(["--version"])

    def test_no_command_shows_help(self):
        rc = cli.main([])
        assert rc == 1

    def test_inspect_requires_policy_path(self):
        with pytest.raises(SystemExit):
            cli.main(["inspect"])

    def test_not_implemented_command(self, capsys):
        # serve is still not implemented in v0.6
        rc = cli.main(["serve"])
        assert rc == 1
        captured = capsys.readouterr()
        assert "not implemented" in captured.err


class TestCLIInspect:
    @patch("lerobot_coreai.cli.load_manifest")
    def test_inspect_pretty_output(self, mock_load, valid_manifest_dict, capsys):
        mock_load.return_value = LeRobotCoreAIManifest.from_dict(valid_manifest_dict)

        rc = cli.main(["inspect", "--policy.path", "kevinqz/EVO1-SO100-CoreAI"])

        assert rc == 0
        out = capsys.readouterr().out
        assert "EVO1" in out
        assert "CoreAI" in out
        assert "so100" in out
        assert "passed" in out
        assert "dry_run" in out

    @patch("lerobot_coreai.cli.load_manifest")
    def test_inspect_json_output(self, mock_load, valid_manifest_dict, capsys):
        mock_load.return_value = LeRobotCoreAIManifest.from_dict(valid_manifest_dict)

        rc = cli.main(["inspect", "--policy.path", "kevinqz/EVO1-SO100-CoreAI", "--json"])

        assert rc == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["schema_version"] == "lerobot-coreai.v0"
        assert data["runtime"] == "coreai"

    @patch("lerobot_coreai.cli.load_manifest")
    def test_inspect_manifest_error(self, mock_load, capsys):
        from lerobot_coreai.errors import ManifestError
        mock_load.side_effect = ManifestError("not found")

        rc = cli.main(["inspect", "--policy.path", "bad/re"])
        assert rc == 2


class TestCLIDoctor:
    @patch("lerobot_coreai.cli.load_manifest")
    def test_doctor_with_manifest(self, mock_load, valid_manifest_dict, capsys):
        mock_load.return_value = LeRobotCoreAIManifest.from_dict(valid_manifest_dict)

        rc = cli.main(["doctor", "--policy.path", "kevinqz/EVO1-SO100-CoreAI"])

        assert rc == 0
        out = capsys.readouterr().out
        assert "✓" in out
        assert "action parity passed" in out.lower() or "Action parity passed" in out

    @patch("lerobot_coreai.cli.load_manifest")
    def test_doctor_robot_type_match(self, mock_load, valid_manifest_dict, capsys):
        mock_load.return_value = LeRobotCoreAIManifest.from_dict(valid_manifest_dict)

        rc = cli.main([
            "doctor",
            "--policy.path", "kevinqz/EVO1-SO100-CoreAI",
            "--robot.type", "so100",
        ])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Robot type matches" in out

    @patch("lerobot_coreai.cli.load_manifest")
    def test_doctor_robot_type_mismatch(self, mock_load, valid_manifest_dict, capsys):
        mock_load.return_value = LeRobotCoreAIManifest.from_dict(valid_manifest_dict)

        rc = cli.main([
            "doctor",
            "--policy.path", "kevinqz/EVO1-SO100-CoreAI",
            "--robot.type", "so101",
        ])
        assert rc == 1
        out = capsys.readouterr().out
        assert "mismatch" in out.lower()

    def test_doctor_no_policy_path(self, capsys):
        """Doctor without --policy.path should still check lerobot-coreai and LeRobot."""
        rc = cli.main(["doctor"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "lerobot-coreai" in out
