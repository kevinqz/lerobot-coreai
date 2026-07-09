# test_cli_package_sim_run.py — CLI tests for package-sim-run / verify-sim-bundle (v0.8.4).

import json
from pathlib import Path

from lerobot_coreai import cli


def _write_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "lerobot-coreai.sim_report.v0",
        "lerobot_coreai_version": "0.8.4",
        "ok": True,
        "mode": "sim",
        "policy": {"path": "test/policy", "runtime": "coreai", "type": "evo1"},
        "runner": {"url": "http://localhost:8710", "reachable": True, "supports_action": True},
        "environment": {"type": "gym", "id": "PushT-v0", "episodes": 1,
                        "max_steps_per_episode": 10, "seed": 42,
                        "simulator_egress_enabled": True},
        "loop": {"episodes_completed": 1, "steps_completed": 10},
        "metrics": {"episodes_completed": 1, "steps_completed": 10, "mean_episode_reward": 1.0},
        "episode_metrics": {"success_rate": 1.0, "mean_reward": 1.0},
        "claims": {
            "proves_sim_task_success": True,
            "proves_real_task_success": False,
            "proves_robot_safety": False,
            "proves_real_world_safety": False,
        },
        "safety": {
            "simulator_egress_enabled": True,
            "robot_egress_enabled": False,
            "physical_actuation_possible": False,
            "actions_sent_to_robot": 0,
            "action_egress": "simulator_only",
        },
        "files": {"report": "sim_report.json"},
        "errors": [],
    }
    (run_dir / "sim_report.json").write_text(json.dumps(report))
    (run_dir / "actions.jsonl").write_text('{"step":0}\n')
    (run_dir / "episodes.jsonl").write_text('{"episode":0}\n')
    return run_dir


class TestPackageCli:
    def test_success_rc0(self, tmp_path):
        run_dir = _write_run(tmp_path)
        out = tmp_path / "bundle"
        rc = cli.main(["package-sim-run", "--run-dir", str(run_dir),
                       "--output-dir", str(out)])
        assert rc == 0
        assert (out / "bundle_manifest.json").is_file()

    def test_json_output(self, tmp_path, capsys):
        run_dir = _write_run(tmp_path)
        out = tmp_path / "bundle"
        rc = cli.main(["package-sim-run", "--run-dir", str(run_dir),
                       "--output-dir", str(out), "--json"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["ok"] is True
        assert "source_run/sim_report.json" in payload["files_copied"]

    def test_missing_run_dir_rc1(self, tmp_path):
        out = tmp_path / "bundle"
        rc = cli.main(["package-sim-run", "--run-dir", str(tmp_path / "nope"),
                       "--output-dir", str(out)])
        assert rc == 1

    def test_missing_report_rc1(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        out = tmp_path / "bundle"
        rc = cli.main(["package-sim-run", "--run-dir", str(run_dir),
                       "--output-dir", str(out)])
        assert rc == 1

    def test_existing_dir_without_overwrite_rc1(self, tmp_path):
        run_dir = _write_run(tmp_path)
        out = tmp_path / "bundle"
        assert cli.main(["package-sim-run", "--run-dir", str(run_dir),
                         "--output-dir", str(out)]) == 0
        # Second run without --overwrite fails.
        assert cli.main(["package-sim-run", "--run-dir", str(run_dir),
                         "--output-dir", str(out)]) == 1

    def test_overwrite_works(self, tmp_path):
        run_dir = _write_run(tmp_path)
        out = tmp_path / "bundle"
        assert cli.main(["package-sim-run", "--run-dir", str(run_dir),
                         "--output-dir", str(out)]) == 0
        assert cli.main(["package-sim-run", "--run-dir", str(run_dir),
                         "--output-dir", str(out), "--overwrite"]) == 0

    def test_rejects_robot_egress_report_rc1(self, tmp_path):
        run_dir = _write_run(tmp_path)
        report = json.loads((run_dir / "sim_report.json").read_text())
        report["safety"]["actions_sent_to_robot"] = 7
        (run_dir / "sim_report.json").write_text(json.dumps(report))
        out = tmp_path / "bundle"
        rc = cli.main(["package-sim-run", "--run-dir", str(run_dir),
                       "--output-dir", str(out)])
        assert rc == 1


class TestVerifyCli:
    def test_verify_valid_bundle_rc0(self, tmp_path):
        run_dir = _write_run(tmp_path)
        out = tmp_path / "bundle"
        cli.main(["package-sim-run", "--run-dir", str(run_dir), "--output-dir", str(out)])
        rc = cli.main(["verify-sim-bundle", "--bundle-dir", str(out)])
        assert rc == 0

    def test_verify_json_output(self, tmp_path, capsys):
        run_dir = _write_run(tmp_path)
        out = tmp_path / "bundle"
        cli.main(["package-sim-run", "--run-dir", str(run_dir), "--output-dir", str(out)])
        capsys.readouterr()
        rc = cli.main(["verify-sim-bundle", "--bundle-dir", str(out), "--json"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["ok"] is True

    def test_verify_fails_after_tamper_rc1(self, tmp_path):
        run_dir = _write_run(tmp_path)
        out = tmp_path / "bundle"
        cli.main(["package-sim-run", "--run-dir", str(run_dir), "--output-dir", str(out)])
        (out / "source_run" / "sim_report.json").write_text('{"x":1}')
        rc = cli.main(["verify-sim-bundle", "--bundle-dir", str(out)])
        assert rc == 1
