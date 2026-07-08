# test_list.py — tests for the lerobot-coreai list command.

import json
from unittest.mock import patch

import pytest

from lerobot_coreai import cli


class TestCLIList:
    @patch("lerobot_coreai.cli.list_lerobot_policies")
    def test_list_basic(self, mock_list, capsys):
        mock_list.return_value = [
            {"repo_id": "kevinqz/EVO1-SO100-CoreAI", "policy_type": "evo1",
             "robot_type": "so100", "status": "action_parity_passed"},
            {"repo_id": "kevinqz/ACT-SO101-CoreAI", "policy_type": "act",
             "robot_type": "so101", "status": "action_parity_passed"},
        ]

        rc = cli.main(["list"])
        assert rc == 0

        out = capsys.readouterr().out
        assert "LeRobot CoreAI policies (2)" in out
        assert "EVO1-SO100-CoreAI" in out
        assert "ACT-SO101-CoreAI" in out
        assert "evo1" in out
        assert "so100" in out

    @patch("lerobot_coreai.cli.list_lerobot_policies")
    def test_list_filter_robot_type(self, mock_list, capsys):
        mock_list.return_value = [
            {"repo_id": "kevinqz/EVO1-SO100-CoreAI", "policy_type": "evo1",
             "robot_type": "so100", "status": "action_parity_passed"},
            {"repo_id": "kevinqz/ACT-SO101-CoreAI", "policy_type": "act",
             "robot_type": "so101", "status": "action_parity_passed"},
        ]

        rc = cli.main(["list", "--robot.type", "so100"])
        assert rc == 0

        out = capsys.readouterr().out
        assert "EVO1-SO100-CoreAI" in out
        assert "ACT-SO101-CoreAI" not in out

    @patch("lerobot_coreai.cli.list_lerobot_policies")
    def test_list_filter_policy_type(self, mock_list, capsys):
        mock_list.return_value = [
            {"repo_id": "kevinqz/EVO1-SO100-CoreAI", "policy_type": "evo1",
             "robot_type": "so100", "status": "action_parity_passed"},
            {"repo_id": "kevinqz/ACT-SO101-CoreAI", "policy_type": "act",
             "robot_type": "so101", "status": "action_parity_passed"},
        ]

        rc = cli.main(["list", "--policy.type", "act"])
        assert rc == 0

        out = capsys.readouterr().out
        assert "ACT-SO101-CoreAI" in out
        assert "EVO1" not in out

    @patch("lerobot_coreai.cli.list_lerobot_policies")
    def test_list_json_output(self, mock_list, capsys):
        mock_list.return_value = [
            {"repo_id": "kevinqz/EVO1-SO100-CoreAI", "policy_type": "evo1",
             "robot_type": "so100", "status": "action_parity_passed"},
        ]

        rc = cli.main(["list", "--json"])
        assert rc == 0

        out = capsys.readouterr().out
        data = json.loads(out)
        assert len(data["policies"]) == 1
        assert data["policies"][0]["repo_id"] == "kevinqz/EVO1-SO100-CoreAI"

    @patch("lerobot_coreai.cli.list_lerobot_policies")
    def test_list_empty(self, mock_list, capsys):
        mock_list.return_value = []

        rc = cli.main(["list"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "No LeRobot CoreAI policies found" in out

    @patch("lerobot_coreai.cli.list_lerobot_policies")
    def test_list_filter_status(self, mock_list, capsys):
        mock_list.return_value = [
            {"repo_id": "kevinqz/EVO1-SO100-CoreAI", "policy_type": "evo1",
             "robot_type": "so100", "status": "action_parity_passed"},
            {"repo_id": "kevinqz/test", "policy_type": "act",
             "robot_type": "so101", "status": "indexed"},
        ]

        rc = cli.main(["list", "--status", "action_parity_passed"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "EVO1" in out
        assert "policies (1)" in out
