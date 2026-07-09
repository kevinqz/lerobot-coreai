# test_sim_bundle.py — unit tests for the reproducibility bundle (v0.8.4).

import json
from pathlib import Path

import pytest

from lerobot_coreai.errors import CoreAIPolicyError
from lerobot_coreai.sim_bundle import (
    SimBundleConfig,
    build_checksums,
    package_sim_run,
    sha256_file,
    verify_bundle_checksums,
    verify_sim_bundle,
)


def _base_report() -> dict:
    """A minimal, invariant-valid sim_report.json dict."""
    return {
        "schema_version": "lerobot-coreai.sim_report.v0",
        "lerobot_coreai_version": "0.8.4",
        "ok": True,
        "mode": "sim",
        "policy": {
            "path": "kevinqz/EVO1-SO100-CoreAI",
            "repo_id": "kevinqz/EVO1-SO100-CoreAI",
            "source_repo_id": "lerobot/evo1_so100",
            "type": "evo1",
            "runtime": "coreai",
            "model_id": "model-abc",
        },
        "runner": {"url": "http://127.0.0.1:8710", "reachable": True, "supports_action": True},
        "environment": {
            "type": "gym", "id": "PushT-v0", "episodes": 10,
            "max_steps_per_episode": 300, "seed": 42,
            "simulator_egress_enabled": True,
        },
        "loop": {"fps_target": 0, "episodes_completed": 10, "steps_completed": 3000},
        "metrics": {
            "episodes_completed": 10, "steps_completed": 3000,
            "mean_episode_reward": 42.0,
        },
        "episode_metrics": {"success_rate": 0.8, "mean_reward": 42.0},
        "latency_metrics": {"runner_p95_ms": 5.0},
        "action_metrics": {"nan_action_steps": 0, "inf_action_steps": 0, "shape_changes": 0},
        "failure_metrics": {"error_rate": 0.0},
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


def _write_run(tmp_path: Path, *, report: dict | None = None,
               optional: bool = True) -> Path:
    """Write a run dir with sim_report.json and (optionally) all extras."""
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "sim_report.json").write_text(json.dumps(report or _base_report()))
    if optional:
        (run_dir / "actions.jsonl").write_text('{"step":0}\n')
        (run_dir / "episodes.jsonl").write_text('{"episode":0}\n')
        (run_dir / "observations.jsonl").write_text('{"obs":0}\n')
        (run_dir / "sim_trace.jsonl").write_text('{"event":"x"}\n')
        (run_dir / "sim_summary.md").write_text("# Summary\n")
        (run_dir / "failure_taxonomy.json").write_text("{}\n")
        (run_dir / "episode_metrics.csv").write_text("episode,reward\n0,42\n")
        (run_dir / "step_metrics.csv").write_text("step,latency\n0,5\n")
    return run_dir


class TestPackaging:
    def test_missing_report_fails(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        with pytest.raises(CoreAIPolicyError, match="sim_report.json not found"):
            package_sim_run(SimBundleConfig(run_dir=run_dir, output_dir=tmp_path / "out"))

    def test_valid_run_packages_core_files(self, tmp_path):
        run_dir = _write_run(tmp_path)
        out = tmp_path / "bundle"
        result = package_sim_run(SimBundleConfig(run_dir=run_dir, output_dir=out))
        assert result.ok
        assert (out / "bundle_manifest.json").is_file()
        assert (out / "checksums.json").is_file()
        assert (out / "README.md").is_file()
        assert (out / "reproducibility.md").is_file()
        assert (out / "policy.json").is_file()
        assert (out / "environment.json").is_file()
        assert (out / "runner.json").is_file()
        assert (out / "source_run" / "sim_report.json").is_file()
        assert "source_run/sim_report.json" in result.files_copied

    def test_optional_missing_files_warn_not_fail(self, tmp_path):
        run_dir = _write_run(tmp_path, optional=False)
        out = tmp_path / "bundle"
        result = package_sim_run(SimBundleConfig(run_dir=run_dir, output_dir=out))
        assert result.ok
        assert any("step_metrics.csv" in w for w in result.warnings)
        assert any("actions.jsonl" in w for w in result.warnings)

    def test_manifest_has_required_fields(self, tmp_path):
        run_dir = _write_run(tmp_path)
        out = tmp_path / "bundle"
        package_sim_run(SimBundleConfig(run_dir=run_dir, output_dir=out))
        manifest = json.loads((out / "bundle_manifest.json").read_text())
        for key in ["schema_version", "lerobot_coreai_version", "created_at",
                    "bundle_type", "mode", "policy", "environment", "results",
                    "safety", "claims", "files"]:
            assert key in manifest
        assert manifest["mode"] == "sim"
        assert manifest["bundle_type"] == "sim_run"
        assert manifest["safety"]["robot_egress_enabled"] is False
        assert manifest["safety"]["actions_sent_to_robot"] == 0
        assert manifest["safety"]["action_egress"] == "simulator_only"
        assert manifest["claims"]["proves_real_task_success"] is False

    def test_manifest_is_schema_valid_at_package_time(self, tmp_path):
        # package_sim_run validates the manifest against the schema before
        # writing it; a produced bundle's manifest must validate.
        import jsonschema
        from importlib.resources import files as _files
        run_dir = _write_run(tmp_path)
        out = tmp_path / "bundle"
        package_sim_run(SimBundleConfig(run_dir=run_dir, output_dir=out))
        schema = json.loads(
            _files("lerobot_coreai.schemas").joinpath("sim-bundle.schema.json").read_text())
        manifest = json.loads((out / "bundle_manifest.json").read_text())
        jsonschema.validate(manifest, schema)  # must not raise

    def test_optional_path_that_is_a_dir_warns_not_crashes(self, tmp_path):
        # A directory named like an optional file must not crash packaging.
        run_dir = _write_run(tmp_path, optional=False)
        (run_dir / "actions.jsonl").mkdir()  # a dir where a file is expected
        out = tmp_path / "bundle"
        result = package_sim_run(SimBundleConfig(run_dir=run_dir, output_dir=out))
        assert result.ok
        assert any("actions.jsonl" in w and "not a file" in w for w in result.warnings)
        assert not (out / "source_run" / "actions.jsonl").exists()

    def test_overwrite_required_for_nonempty(self, tmp_path):
        run_dir = _write_run(tmp_path)
        out = tmp_path / "bundle"
        package_sim_run(SimBundleConfig(run_dir=run_dir, output_dir=out))
        with pytest.raises(CoreAIPolicyError, match="not empty"):
            package_sim_run(SimBundleConfig(run_dir=run_dir, output_dir=out))
        # With overwrite, succeeds.
        result = package_sim_run(SimBundleConfig(run_dir=run_dir, output_dir=out, overwrite=True))
        assert result.ok


class TestInvariants:
    def test_rejects_robot_egress_enabled(self, tmp_path):
        report = _base_report()
        report["safety"]["robot_egress_enabled"] = True
        run_dir = _write_run(tmp_path, report=report)
        with pytest.raises(CoreAIPolicyError, match="no-robot-egress"):
            package_sim_run(SimBundleConfig(run_dir=run_dir, output_dir=tmp_path / "out"))

    def test_rejects_actions_sent_to_robot(self, tmp_path):
        report = _base_report()
        report["safety"]["actions_sent_to_robot"] = 3
        run_dir = _write_run(tmp_path, report=report)
        with pytest.raises(CoreAIPolicyError, match="no-robot-egress"):
            package_sim_run(SimBundleConfig(run_dir=run_dir, output_dir=tmp_path / "out"))

    def test_rejects_bad_action_egress(self, tmp_path):
        report = _base_report()
        report["safety"]["action_egress"] = "robot"
        run_dir = _write_run(tmp_path, report=report)
        with pytest.raises(CoreAIPolicyError, match="no-robot-egress"):
            package_sim_run(SimBundleConfig(run_dir=run_dir, output_dir=tmp_path / "out"))

    def test_rejects_non_sim_mode(self, tmp_path):
        report = _base_report()
        report["mode"] = "real"
        run_dir = _write_run(tmp_path, report=report)
        with pytest.raises(CoreAIPolicyError, match="no-robot-egress"):
            package_sim_run(SimBundleConfig(run_dir=run_dir, output_dir=tmp_path / "out"))

    def test_rejects_proves_real_task_success(self, tmp_path):
        report = _base_report()
        report["claims"]["proves_real_task_success"] = True
        run_dir = _write_run(tmp_path, report=report)
        with pytest.raises(CoreAIPolicyError, match="proves_real_task_success"):
            package_sim_run(SimBundleConfig(run_dir=run_dir, output_dir=tmp_path / "out"))

    def test_rejects_proves_robot_safety(self, tmp_path):
        report = _base_report()
        report["claims"]["proves_robot_safety"] = True
        run_dir = _write_run(tmp_path, report=report)
        with pytest.raises(CoreAIPolicyError, match="proves_robot_safety"):
            package_sim_run(SimBundleConfig(run_dir=run_dir, output_dir=tmp_path / "out"))

    def test_rejects_proves_real_world_safety(self, tmp_path):
        report = _base_report()
        report["claims"]["proves_real_world_safety"] = True
        run_dir = _write_run(tmp_path, report=report)
        with pytest.raises(CoreAIPolicyError, match="proves_real_world_safety"):
            package_sim_run(SimBundleConfig(run_dir=run_dir, output_dir=tmp_path / "out"))

    def test_rejects_missing_claims_block(self, tmp_path):
        report = _base_report()
        del report["claims"]
        run_dir = _write_run(tmp_path, report=report)
        with pytest.raises(CoreAIPolicyError, match="claims block missing"):
            package_sim_run(SimBundleConfig(run_dir=run_dir, output_dir=tmp_path / "out"))


class TestRedaction:
    def test_redacts_runner_url(self, tmp_path):
        run_dir = _write_run(tmp_path)
        out = tmp_path / "bundle"
        package_sim_run(SimBundleConfig(run_dir=run_dir, output_dir=out, redact_runner_url=True))
        runner = json.loads((out / "runner.json").read_text())
        assert runner["url"] == "<redacted>"
        assert runner["redacted"] is True
        manifest = json.loads((out / "bundle_manifest.json").read_text())
        assert manifest["runner"]["url"] == "<redacted>"

    def test_runner_url_present_by_default(self, tmp_path):
        run_dir = _write_run(tmp_path)
        out = tmp_path / "bundle"
        package_sim_run(SimBundleConfig(run_dir=run_dir, output_dir=out))
        runner = json.loads((out / "runner.json").read_text())
        assert runner["url"] == "http://127.0.0.1:8710"
        assert runner["redacted"] is False

    def test_redacts_absolute_run_dir_by_default(self, tmp_path):
        # run_dir is absolute (tmp_path) → manifest must not leak the home prefix.
        run_dir = _write_run(tmp_path)
        assert run_dir.is_absolute()
        out = tmp_path / "bundle"
        package_sim_run(SimBundleConfig(run_dir=run_dir, output_dir=out))
        manifest = json.loads((out / "bundle_manifest.json").read_text())
        run_dir_field = manifest["source_run"]["run_dir"]
        assert run_dir_field == "run"  # basename only
        assert str(tmp_path) not in run_dir_field

    def test_keeps_run_dir_when_redaction_disabled(self, tmp_path):
        run_dir = _write_run(tmp_path)
        out = tmp_path / "bundle"
        package_sim_run(SimBundleConfig(
            run_dir=run_dir, output_dir=out, redact_local_paths=False))
        manifest = json.loads((out / "bundle_manifest.json").read_text())
        assert manifest["source_run"]["run_dir"] == str(run_dir)


class TestObservationsDir:
    def test_observations_dir_excluded_by_default(self, tmp_path):
        run_dir = _write_run(tmp_path)
        (run_dir / "observations").mkdir()
        (run_dir / "observations" / "0.png").write_bytes(b"img")
        out = tmp_path / "bundle"
        package_sim_run(SimBundleConfig(run_dir=run_dir, output_dir=out))
        assert not (out / "source_run" / "observations").exists()

    def test_observations_dir_included_with_flag(self, tmp_path):
        run_dir = _write_run(tmp_path)
        (run_dir / "observations").mkdir()
        (run_dir / "observations" / "0.png").write_bytes(b"img")
        out = tmp_path / "bundle"
        package_sim_run(SimBundleConfig(
            run_dir=run_dir, output_dir=out, include_observations_dir=True))
        assert (out / "source_run" / "observations" / "0.png").is_file()


class TestChecksums:
    def test_sha256_file(self, tmp_path):
        p = tmp_path / "f.txt"
        p.write_text("hello")
        h = sha256_file(p)
        assert h.startswith("sha256:")
        # Known SHA256 of "hello".
        assert h == "sha256:2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"

    def test_checksums_exclude_self(self, tmp_path):
        run_dir = _write_run(tmp_path)
        out = tmp_path / "bundle"
        package_sim_run(SimBundleConfig(run_dir=run_dir, output_dir=out))
        checksums = json.loads((out / "checksums.json").read_text())
        assert "checksums.json" not in checksums["files"]
        assert "bundle_manifest.json" in checksums["files"]

    def test_checksums_match_content(self, tmp_path):
        run_dir = _write_run(tmp_path)
        out = tmp_path / "bundle"
        package_sim_run(SimBundleConfig(run_dir=run_dir, output_dir=out))
        v = verify_bundle_checksums(out)
        assert v["ok"]
        assert v["checked"] > 0
        assert v["failures"] == []

    def test_tamper_detected(self, tmp_path):
        run_dir = _write_run(tmp_path)
        out = tmp_path / "bundle"
        package_sim_run(SimBundleConfig(run_dir=run_dir, output_dir=out))
        # Tamper with a copied file after packaging.
        (out / "source_run" / "sim_report.json").write_text('{"tampered": true}')
        v = verify_bundle_checksums(out)
        assert not v["ok"]
        assert any("sim_report.json" in f for f in v["failures"])

    def test_build_checksums_direct(self, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "checksums.json").write_text("{}")
        result = build_checksums(tmp_path)
        assert result["algorithm"] == "sha256"
        assert "a.txt" in result["files"]
        assert "checksums.json" not in result["files"]


class TestVerify:
    def test_verify_passes_on_valid_bundle(self, tmp_path):
        run_dir = _write_run(tmp_path)
        out = tmp_path / "bundle"
        package_sim_run(SimBundleConfig(run_dir=run_dir, output_dir=out))
        result = verify_sim_bundle(out)
        assert result.ok
        assert result.checksum_failures == []
        assert result.invariant_failures == []

    def test_verify_fails_after_tamper(self, tmp_path):
        run_dir = _write_run(tmp_path)
        out = tmp_path / "bundle"
        package_sim_run(SimBundleConfig(run_dir=run_dir, output_dir=out))
        (out / "source_run" / "sim_report.json").write_text('{"x": 1}')
        result = verify_sim_bundle(out)
        assert not result.ok
        assert len(result.checksum_failures) >= 1

    def test_verify_missing_manifest(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        result = verify_sim_bundle(empty)
        assert not result.ok
        assert any("manifest" in f for f in result.invariant_failures)


def _tamper_manifest(out: Path, mutate) -> None:
    """Edit the manifest via mutate(dict) and rebuild checksums so the only
    verification failure is the invariant, not the checksum."""
    manifest = json.loads((out / "bundle_manifest.json").read_text())
    mutate(manifest)
    (out / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    checksums = build_checksums(out)
    (out / "checksums.json").write_text(json.dumps(checksums, indent=2) + "\n")


class TestVerifyInvariants:
    def _bundle(self, tmp_path):
        run_dir = _write_run(tmp_path)
        out = tmp_path / "bundle"
        package_sim_run(SimBundleConfig(run_dir=run_dir, output_dir=out))
        return out

    def test_verify_fails_mode_not_sim(self, tmp_path):
        out = self._bundle(tmp_path)
        _tamper_manifest(out, lambda m: m.update({"mode": "real"}))
        result = verify_sim_bundle(out)
        assert not result.ok
        assert result.checksum_failures == []  # isolated: only invariant failed
        assert any("mode" in f for f in result.invariant_failures)

    def test_verify_fails_bundle_type_wrong(self, tmp_path):
        out = self._bundle(tmp_path)
        _tamper_manifest(out, lambda m: m.update({"bundle_type": "real_run"}))
        result = verify_sim_bundle(out)
        assert not result.ok
        assert any("bundle_type" in f for f in result.invariant_failures)

    def test_verify_fails_bad_schema_version(self, tmp_path):
        out = self._bundle(tmp_path)
        _tamper_manifest(out, lambda m: m.update({"schema_version": "bogus"}))
        result = verify_sim_bundle(out)
        assert not result.ok
        assert any("schema_version" in f for f in result.invariant_failures)

    def test_verify_fails_physical_actuation_possible(self, tmp_path):
        out = self._bundle(tmp_path)
        _tamper_manifest(out, lambda m: m["safety"].update({"physical_actuation_possible": True}))
        result = verify_sim_bundle(out)
        assert not result.ok
        assert any("physical_actuation_possible" in f for f in result.invariant_failures)

    def test_verify_fails_proves_real_world_safety(self, tmp_path):
        out = self._bundle(tmp_path)
        _tamper_manifest(out, lambda m: m["claims"].update({"proves_real_world_safety": True}))
        result = verify_sim_bundle(out)
        assert not result.ok
        assert any("proves_real_world_safety" in f for f in result.invariant_failures)

    def test_verify_fails_if_source_report_violates_invariants(self, tmp_path):
        out = self._bundle(tmp_path)
        # Tamper the bundled source report to claim robot egress, then rebuild
        # checksums so only the source-report invariant fails.
        src = out / "source_run" / "sim_report.json"
        report = json.loads(src.read_text())
        report["safety"]["actions_sent_to_robot"] = 9
        src.write_text(json.dumps(report))
        checksums = build_checksums(out)
        (out / "checksums.json").write_text(json.dumps(checksums, indent=2) + "\n")
        result = verify_sim_bundle(out)
        assert not result.ok
        assert any("source report" in f for f in result.invariant_failures)
