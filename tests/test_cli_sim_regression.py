# test_cli_sim_regression.py — CLI tests for sim-regression command (v0.8.3).

import json
from pathlib import Path

from lerobot_coreai import cli


def _write_report(path: Path, *, success_rate=0.8, mean_reward=42.0, runner_p95=12.0):
    report = {
        "mode": "sim",
        "episode_metrics": {"success_rate": success_rate, "mean_reward": mean_reward},
        "latency_metrics": {"runner_p95_ms": runner_p95},
    }
    path.write_text(json.dumps(report))


class TestCliSimRegression:
    def test_regression_pass_rc0(self, tmp_path, capsys):
        baseline = tmp_path / "baseline.json"
        candidate = tmp_path / "candidate.json"
        _write_report(baseline)
        _write_report(candidate, success_rate=0.79)
        rc = cli.main([
            "sim-regression", "--baseline", str(baseline), "--candidate", str(candidate),
            "--max-success-drop", "0.05",
        ])
        assert rc == 0
        out = capsys.readouterr().out
        assert "passed" in out.lower()

    def test_regression_fail_rc1(self, tmp_path, capsys):
        baseline = tmp_path / "baseline.json"
        candidate = tmp_path / "candidate.json"
        _write_report(baseline, success_rate=0.9)
        _write_report(candidate, success_rate=0.5)
        rc = cli.main([
            "sim-regression", "--baseline", str(baseline), "--candidate", str(candidate),
            "--max-success-drop", "0.05",
        ])
        assert rc == 1
        out = capsys.readouterr().out
        assert "FAILED" in out

    def test_json_output(self, tmp_path, capsys):
        baseline = tmp_path / "baseline.json"
        candidate = tmp_path / "candidate.json"
        _write_report(baseline)
        _write_report(candidate)
        rc = cli.main([
            "sim-regression", "--baseline", str(baseline), "--candidate", str(candidate),
            "--max-success-drop", "0.05", "--json",
        ])
        assert rc == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "passed" in data
        assert "deltas" in data

    def test_missing_report_rc1(self, tmp_path, capsys):
        rc = cli.main([
            "sim-regression", "--baseline", str(tmp_path / "nope.json"),
            "--candidate", str(tmp_path / "nope2.json"),
        ])
        assert rc == 1
        err = capsys.readouterr().err
        assert "failed" in err.lower()
