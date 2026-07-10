# test_benchmark_pack.py — reproducible bridge benchmark packs (v1.1.7).

import json
from importlib.resources import files

import jsonschema
import pytest

from lerobot_coreai.benchmark_pack import (
    BRIDGE_BENCHMARK_SCHEMA_VERSION, BenchmarkError, BenchmarkInputs,
    package_bridge_benchmark, verify_bridge_benchmark,
)


def _write(path, obj):
    path.write_text(json.dumps(obj))
    return path


def _good_reports(tmp_path):
    compat = _write(tmp_path / "compat.json", {
        "schema_version": "lerobot-coreai.lerobot_compat.v0", "ok": True,
        "policy_path": "kevinqz/EVO1-SO100-CoreAI",
        "claims": {"native_upstream_registry": False, "supports_training": False,
                   "supports_physical_safety": False}})
    bridge = _write(tmp_path / "bridge.json", {
        "schema_version": "lerobot-coreai.lerobot_bridge.v0", "ok": True,
        "policy_path": "kevinqz/EVO1-SO100-CoreAI",
        "claims": {"native_upstream_registry": False, "supports_training": False,
                   "proves_physical_safety": False}})
    return compat, bridge


def _inputs(tmp_path):
    compat, bridge = _good_reports(tmp_path)
    return BenchmarkInputs(compat=compat, bridge=bridge)


def test_package_and_verify_roundtrip(tmp_path):
    out = tmp_path / "pack"
    manifest = package_bridge_benchmark(_inputs(tmp_path), out)
    assert manifest["schema_version"] == BRIDGE_BENCHMARK_SCHEMA_VERSION
    assert manifest["policy_path"] == "kevinqz/EVO1-SO100-CoreAI"
    assert (out / "benchmark_manifest.json").is_file()
    assert (out / "checksums.json").is_file()
    assert (out / "README.md").is_file()
    assert (out / "reports" / "lerobot_compatibility_report.json").is_file()

    result = verify_bridge_benchmark(out)
    assert result.ok is True


def test_manifest_schema_valid(tmp_path):
    manifest = package_bridge_benchmark(_inputs(tmp_path), tmp_path / "pack")
    schema = json.loads(files("lerobot_coreai.schemas").joinpath(
        "bridge-benchmark-pack.schema.json").read_text())
    jsonschema.validate(manifest, schema)


def test_no_reports_fails_closed(tmp_path):
    with pytest.raises(BenchmarkError):
        package_bridge_benchmark(BenchmarkInputs(), tmp_path / "pack")


def test_overclaiming_report_refused(tmp_path):
    bad = _write(tmp_path / "bad.json", {
        "schema_version": "x", "claims": {"proves_physical_safety": True}})
    with pytest.raises(BenchmarkError):
        package_bridge_benchmark(BenchmarkInputs(bridge=bad), tmp_path / "pack")


def test_tamper_detected(tmp_path):
    out = tmp_path / "pack"
    package_bridge_benchmark(_inputs(tmp_path), out)
    # Tamper with a bundled report after packaging.
    victim = out / "reports" / "lerobot_compatibility_report.json"
    victim.write_text(victim.read_text().replace("true", "false"))
    result = verify_bridge_benchmark(out)
    assert result.ok is False
    names = {c["name"]: c["passed"] for c in result.checks}
    assert names["checksums_match"] is False


def test_missing_report_detected(tmp_path):
    out = tmp_path / "pack"
    package_bridge_benchmark(_inputs(tmp_path), out)
    (out / "reports" / "lerobot_bridge_report.json").unlink()
    result = verify_bridge_benchmark(out)
    assert result.ok is False
    names = {c["name"]: c["passed"] for c in result.checks}
    assert names["all_reports_present"] is False


def test_verify_missing_manifest(tmp_path):
    result = verify_bridge_benchmark(tmp_path / "empty")
    assert result.ok is False
