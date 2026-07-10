# test_provenance.py — provenance manifests (v1.2.0).

import json
from importlib.resources import files

import jsonschema
import pytest

from lerobot_coreai.errors import CoreAIPolicyError
from lerobot_coreai.provenance import (
    PROVENANCE_SCHEMA_VERSION, build_provenance, sha256_file,
)


def _artifact(tmp_path):
    d = tmp_path / "bundle"
    (d / "reports").mkdir(parents=True)
    (d / "benchmark_manifest.json").write_text(json.dumps({
        "schema_version": "lerobot-coreai.bridge_benchmark_pack.v0",
        "bundle_type": "bridge_benchmark",
        "reports": {"compat": "reports/lerobot_compatibility_report.json"},
        "claims": {"proves_task_success": False, "proves_physical_safety": False,
                   "authorizes_robot_actuation": False}}))
    (d / "checksums.json").write_text(json.dumps({"benchmark_manifest.json": "x"}))
    (d / "reports" / "lerobot_compatibility_report.json").write_text(json.dumps({"ok": True}))
    return d


def test_build_provenance_shape_and_schema(tmp_path):
    prov = build_provenance(_artifact(tmp_path), "bridge_benchmark",
                            created_at="2026-07-10T00:00:00Z")
    assert prov["schema_version"] == PROVENANCE_SCHEMA_VERSION
    assert prov["artifact_type"] == "bridge_benchmark"
    assert "benchmark_manifest.json" in prov["artifact_hashes"]
    assert prov["source_reports"]["compat"] == "reports/lerobot_compatibility_report.json"
    schema = json.loads(files("lerobot_coreai.schemas").joinpath(
        "provenance-manifest.schema.json").read_text())
    jsonschema.validate(prov, schema)


def test_provenance_claims_honest(tmp_path):
    prov = build_provenance(_artifact(tmp_path), "bridge_benchmark")
    assert prov["claims"]["proves_physical_safety"] is False
    assert prov["claims"]["authorizes_robot_actuation"] is False
    assert prov["claims"]["proves_task_success"] is False


def test_provenance_hashes_match_files(tmp_path):
    d = _artifact(tmp_path)
    prov = build_provenance(d, "bridge_benchmark")
    assert prov["artifact_hashes"]["benchmark_manifest.json"] == \
        sha256_file(d / "benchmark_manifest.json")


def test_missing_artifact_dir_fails(tmp_path):
    with pytest.raises(CoreAIPolicyError):
        build_provenance(tmp_path / "nope", "bridge_benchmark")
