# test_cli_sim_analytics.py — CLI flag tests for v0.8.2 analytics artifacts.

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
        "mode": "sim", "ok": True,
        "metrics": {"episodes_completed": 1, "steps_completed": 3,
                    "actions_generated": 3, "actions_sent_to_simulator": 3},
        "safety": {"actions_sent_to_robot": 0},
        "files": {"report": "sim_report.json"},
    }
    return mock


class TestCliSimAnalyticsFlags:
    def test_export_csv_flag_passed(self, tmp_path):
        mock_result = _mock_sim_result(tmp_path)
        with patch("lerobot_coreai.cli.run_sim_mode", return_value=mock_result) as mock_run:
            cli.main([
                "sim", "--policy.path", "test", "--env.type", "fake",
                "--output-dir", str(tmp_path / "run"), "--confirm-sim-egress",
                "--export-csv",
            ])
        config = mock_run.call_args[0][0]
        assert config.export_csv is True

    def test_no_summary_md_flag_passed(self, tmp_path):
        mock_result = _mock_sim_result(tmp_path)
        with patch("lerobot_coreai.cli.run_sim_mode", return_value=mock_result) as mock_run:
            cli.main([
                "sim", "--policy.path", "test", "--env.type", "fake",
                "--output-dir", str(tmp_path / "run"), "--confirm-sim-egress",
                "--no-summary-md",
            ])
        config = mock_run.call_args[0][0]
        assert config.summary_md is False

    def test_no_failure_taxonomy_flag_passed(self, tmp_path):
        mock_result = _mock_sim_result(tmp_path)
        with patch("lerobot_coreai.cli.run_sim_mode", return_value=mock_result) as mock_run:
            cli.main([
                "sim", "--policy.path", "test", "--env.type", "fake",
                "--output-dir", str(tmp_path / "run"), "--confirm-sim-egress",
                "--no-failure-taxonomy",
            ])
        config = mock_run.call_args[0][0]
        assert config.failure_taxonomy is False

    def test_defaults_on(self, tmp_path):
        """summary_md and failure_taxonomy default to True; export_csv defaults to False."""
        mock_result = _mock_sim_result(tmp_path)
        with patch("lerobot_coreai.cli.run_sim_mode", return_value=mock_result) as mock_run:
            cli.main([
                "sim", "--policy.path", "test", "--env.type", "fake",
                "--output-dir", str(tmp_path / "run"), "--confirm-sim-egress",
            ])
        config = mock_run.call_args[0][0]
        assert config.summary_md is True
        assert config.failure_taxonomy is True
        assert config.export_csv is False

    def test_csv_disabled_message_when_not_exported(self, tmp_path, capsys):
        mock_result = _mock_sim_result(tmp_path)
        with patch("lerobot_coreai.cli.run_sim_mode", return_value=mock_result):
            cli.main([
                "sim", "--policy.path", "test", "--env.type", "fake",
                "--output-dir", str(tmp_path / "run"), "--confirm-sim-egress",
            ])
        out = capsys.readouterr().out
        assert "CSV exports:  disabled" in out
