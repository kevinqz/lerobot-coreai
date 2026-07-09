# test_sim_regression.py — tests for sim regression comparison (v0.8.3).

import json
import pytest
from pathlib import Path

from lerobot_coreai.errors import CoreAIPolicyError
from lerobot_coreai.sim_regression import SimRegressionConfig, run_sim_regression


def _write_report(path: Path, *, success_rate=0.8, mean_reward=42.0, runner_p95=12.0):
    report = {
        "mode": "sim",
        "episode_metrics": {"success_rate": success_rate, "mean_reward": mean_reward},
        "latency_metrics": {"runner_p95_ms": runner_p95},
    }
    path.write_text(json.dumps(report))


class TestSimRegression:
    def test_no_regression_passes(self, tmp_path):
        baseline = tmp_path / "baseline.json"
        candidate = tmp_path / "candidate.json"
        _write_report(baseline, success_rate=0.8, mean_reward=42.0, runner_p95=12.0)
        _write_report(candidate, success_rate=0.79, mean_reward=41.0, runner_p95=13.0)
        r = run_sim_regression(baseline, candidate, SimRegressionConfig(
            max_success_drop=0.05, max_reward_drop=2.0, max_runner_p95_increase_ms=2.0,
        ))
        assert r.passed is True
        assert r.deltas["success_rate_delta"] == pytest.approx(-0.01)

    def test_success_rate_regression_fails(self, tmp_path):
        baseline = tmp_path / "baseline.json"
        candidate = tmp_path / "candidate.json"
        _write_report(baseline, success_rate=0.9)
        _write_report(candidate, success_rate=0.7)
        r = run_sim_regression(baseline, candidate, SimRegressionConfig(max_success_drop=0.05))
        assert r.passed is False
        check = [c for c in r.checks if c["name"] == "max_success_drop"][0]
        assert check["passed"] is False

    def test_reward_regression_fails(self, tmp_path):
        baseline = tmp_path / "baseline.json"
        candidate = tmp_path / "candidate.json"
        _write_report(baseline, mean_reward=50.0)
        _write_report(candidate, mean_reward=40.0)
        r = run_sim_regression(baseline, candidate, SimRegressionConfig(max_reward_drop=5.0))
        assert r.passed is False

    def test_latency_regression_fails(self, tmp_path):
        baseline = tmp_path / "baseline.json"
        candidate = tmp_path / "candidate.json"
        _write_report(baseline, runner_p95=10.0)
        _write_report(candidate, runner_p95=20.0)
        r = run_sim_regression(baseline, candidate, SimRegressionConfig(max_runner_p95_increase_ms=5.0))
        assert r.passed is False

    def test_no_thresholds_passes(self, tmp_path):
        baseline = tmp_path / "baseline.json"
        candidate = tmp_path / "candidate.json"
        _write_report(baseline)
        _write_report(candidate, success_rate=0.1)
        # No thresholds configured -> no checks -> passes.
        r = run_sim_regression(baseline, candidate, SimRegressionConfig())
        assert r.passed is True

    def test_missing_report_raises(self, tmp_path):
        with pytest.raises(CoreAIPolicyError, match="not found"):
            run_sim_regression(tmp_path / "nope.json", tmp_path / "nope2.json", SimRegressionConfig())

    def test_non_sim_report_rejected(self, tmp_path):
        baseline = tmp_path / "baseline.json"
        candidate = tmp_path / "candidate.json"
        baseline.write_text(json.dumps({"mode": "shadow"}))
        candidate.write_text(json.dumps({"mode": "sim"}))
        with pytest.raises(CoreAIPolicyError, match="not a sim report"):
            run_sim_regression(baseline, candidate, SimRegressionConfig())
