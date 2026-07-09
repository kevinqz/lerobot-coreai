# test_cli_sim_quality.py — CLI tests for sim quality gates (v0.8.3).

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


class TestCliSimQuality:
    def test_quality_flags_build_config(self, tmp_path):
        mock_result = _mock_sim_result(tmp_path)
        with patch("lerobot_coreai.cli.run_sim_mode", return_value=mock_result) as mock_run:
            cli.main([
                "sim", "--policy.path", "test", "--env.type", "fake",
                "--output-dir", str(tmp_path / "run"), "--confirm-sim-egress",
                "--quality.min-success-rate", "0.8",
                "--quality.max-runner-p95-ms", "15.0",
            ])
        config = mock_run.call_args[0][0]
        assert config.quality_config is not None
        assert config.quality_config.min_success_rate == 0.8
        assert config.quality_config.max_runner_p95_ms == 15.0
        assert config.fail_on_quality is False

    def test_fail_on_quality_flag_passed(self, tmp_path):
        mock_result = _mock_sim_result(tmp_path)
        with patch("lerobot_coreai.cli.run_sim_mode", return_value=mock_result) as mock_run:
            cli.main([
                "sim", "--policy.path", "test", "--env.type", "fake",
                "--output-dir", str(tmp_path / "run"), "--confirm-sim-egress",
                "--quality.min-success-rate", "0.8",
                "--quality.fail-on-quality",
            ])
        config = mock_run.call_args[0][0]
        assert config.fail_on_quality is True

    def test_no_quality_flags_means_none(self, tmp_path):
        mock_result = _mock_sim_result(tmp_path)
        with patch("lerobot_coreai.cli.run_sim_mode", return_value=mock_result) as mock_run:
            cli.main([
                "sim", "--policy.path", "test", "--env.type", "fake",
                "--output-dir", str(tmp_path / "run"), "--confirm-sim-egress",
            ])
        config = mock_run.call_args[0][0]
        assert config.quality_config is None
