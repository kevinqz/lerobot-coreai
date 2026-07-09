# test_cli_sim.py — CLI tests for the sim command (v0.8).

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from lerobot_coreai import cli
from lerobot_coreai.sim import SimResult


def _mock_sim_result(tmp_path):
    mock = MagicMock(spec=SimResult)
    mock.ok = True
    mock.output_dir = tmp_path / "run"
    mock.report_path = tmp_path / "run" / "sim_report.json"
    mock.trace_path = tmp_path / "run" / "sim_trace.jsonl"
    mock.actions_path = tmp_path / "run" / "actions.jsonl"
    mock.observations_path = tmp_path / "run" / "observations.jsonl"
    mock.episodes_path = tmp_path / "run" / "episodes.jsonl"
    mock.report = {
        "mode": "sim",
        "ok": True,
        "metrics": {
            "episodes_completed": 1,
            "steps_completed": 3,
            "actions_generated": 3,
            "actions_sent_to_simulator": 3,
            "actions_sent_to_robot": 0,
        },
        "safety": {"robot_egress_enabled": False, "actions_sent_to_robot": 0},
    }
    return mock


class TestCliSim:
    def test_sim_success_rc0(self, tmp_path, capsys):
        mock_result = _mock_sim_result(tmp_path)
        with patch("lerobot_coreai.cli.run_sim_mode", return_value=mock_result):
            rc = cli.main([
                "sim",
                "--policy.path", "kevinqz/EVO1-SO100-CoreAI",
                "--env.type", "fake",
                "--runner.url", "http://localhost:8710",
                "--output-dir", str(tmp_path / "run"),
                "--episodes", "1",
                "--max-steps-per-episode", "3",
                "--confirm-sim-egress",
            ])
        assert rc == 0
        out = capsys.readouterr().out
        assert "No robot commands sent" in out
        assert "Sim run completed" in out

    def test_sim_json_output(self, tmp_path, capsys):
        mock_result = _mock_sim_result(tmp_path)
        with patch("lerobot_coreai.cli.run_sim_mode", return_value=mock_result):
            rc = cli.main([
                "sim",
                "--policy.path", "test",
                "--env.type", "fake",
                "--output-dir", str(tmp_path / "run"),
                "--confirm-sim-egress",
                "--json",
            ])
        assert rc == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["mode"] == "sim"

    def test_sim_missing_confirm_rc1(self, tmp_path, capsys):
        from lerobot_coreai.errors import CoreAIPolicyError

        def _raise(config):
            raise CoreAIPolicyError(
                "Sim mode sends actions to a simulator. Re-run with --confirm-sim-egress.\n"
                "No robot commands were sent."
            )

        with patch("lerobot_coreai.cli.run_sim_mode", side_effect=_raise):
            rc = cli.main([
                "sim",
                "--policy.path", "test",
                "--env.type", "fake",
                "--output-dir", str(tmp_path / "run"),
            ])
        assert rc == 1
        err = capsys.readouterr().err
        assert "No robot commands were sent." in err

    def test_sim_passes_confirm_flag(self, tmp_path):
        mock_result = _mock_sim_result(tmp_path)
        with patch("lerobot_coreai.cli.run_sim_mode", return_value=mock_result) as mock_run:
            cli.main([
                "sim",
                "--policy.path", "test",
                "--env.type", "fake",
                "--output-dir", str(tmp_path / "run"),
                "--confirm-sim-egress",
            ])
        config = mock_run.call_args[0][0]
        assert config.confirm_sim_egress is True

    def test_sim_unexpected_error_rc1(self, tmp_path, capsys):
        with patch("lerobot_coreai.cli.run_sim_mode", side_effect=RuntimeError("boom")):
            rc = cli.main([
                "sim",
                "--policy.path", "test",
                "--env.type", "fake",
                "--output-dir", str(tmp_path / "run"),
                "--confirm-sim-egress",
            ])
        assert rc == 1
        err = capsys.readouterr().err
        assert "No robot commands were sent." in err

    def test_sim_env_type_restricted(self, tmp_path, capsys):
        """--env.type should only accept fake|replay in v0.8.0."""
        with pytest.raises(SystemExit):
            cli.main([
                "sim",
                "--policy.path", "test",
                "--env.type", "mujoco",
                "--output-dir", str(tmp_path / "run"),
                "--confirm-sim-egress",
            ])
