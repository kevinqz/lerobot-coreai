# test_artifact.py — canonical plugin artifact build + hardened verify (v1.3.6/1.3.7).

import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("lerobot")

from lerobot_policy_coreai_bridge.artifact import (  # noqa: E402
    ArtifactError, build_plugin_artifact, check_version_compatibility,
    verify_plugin_artifact,
)
from lerobot_policy_coreai_bridge.processor_coreai_bridge import (  # noqa: E402
    ProcessorOwnershipError,
)

HORIZON, ACTION_DIM = 3, 7


def _manifest(*, ownership="exact"):
    m = {
        "schema_version": "lerobot-coreai.v0", "runtime": "coreai",
        "framework": {"name": "lerobot", "version": "0.6.0", "commit": None},
        "policy": {"repo_id": "kevinqz/E2E", "source_repo_id": "lerobot/e2e",
                   "type": "evo1", "class": None, "config_class": None},
        "robot": {"type": "so100", "action_representation": "joint_position_delta",
                  "fps": 30},
        "features": {
            "observation": {
                "observation.state": {"dtype": "float32", "shape": [ACTION_DIM],
                                      "required": True},
                "task": {"dtype": "string", "required": False}},
            "action": {"action": {"dtype": "float32", "shape": [HORIZON, ACTION_DIM]}}},
        "normalization": {"format": "lerobot", "path": "norm_stats.json",
                          "sha256": None},
        "coreai": {"artifact_format": "aimodel", "runner": "coreai-runner",
                   "graphs": [{"name": "g", "role": "denoise_step"}],
                   "host_loop_required": False},
        "evaluation": {"metric": "action_parity", "status": "passed", "n_obs": 8,
                       "min_chunk_cosine": 0.9999, "max_action_mae": None,
                       "max_relative_action_mae": None, "proves_numeric_fidelity": True,
                       "proves_task_success": False, "proves_robot_safety": False},
        "safety": {"default_mode": "dry_run",
                   "real_actuation_requires_confirmation": True},
    }
    if ownership == "exact":
        m["contracts"] = {"processor": {
            "observation_input": {"owner": "coreai_runner",
                                  "expects": "raw_lerobot_observation"},
            "action_output": {"owner": "coreai_runner",
                              "returns": "postprocessed_environment_action"}}}
    elif ownership == "wrong_expects":
        m["contracts"] = {"processor": {
            "observation_input": {"owner": "coreai_runner",
                                  "expects": "policy_preprocessed_observation"},
            "action_output": {"owner": "coreai_runner",
                              "returns": "postprocessed_environment_action"}}}
    elif ownership == "wrong_returns":
        m["contracts"] = {"processor": {
            "observation_input": {"owner": "coreai_runner",
                                  "expects": "raw_lerobot_observation"},
            "action_output": {"owner": "coreai_runner", "returns": "normalized_action"}}}
    # ownership == "none" -> no contracts block
    return m


def _src(tmp_path, **kw):
    d = tmp_path / "coreai"
    d.mkdir()
    (d / "lerobot-coreai.json").write_text(json.dumps(_manifest(**kw)))
    return str(d)


def _build(tmp_path, **kw):
    out = tmp_path / "plugin"
    build_plugin_artifact(_src(tmp_path, **kw), str(out),
                          runner_url_env="COREAI_RUNNER_URL")
    return out


# --- layout + secrets ---

def test_build_writes_hardened_layout(tmp_path):
    out = _build(tmp_path)
    for fn in ("config.json", "policy_preprocessor.json", "policy_postprocessor.json",
               "lerobot-coreai.json", "plugin_artifact_manifest.json",
               "plugin_artifact_inventory.json", "checksums.json", "README.md"):
        assert (out / fn).exists(), fn


def test_build_persists_no_local_path(tmp_path):
    out = _build(tmp_path)
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["coreai_artifact"] == ""
    assert cfg["runner_url_env"] == "COREAI_RUNNER_URL"


def test_verify_passes_and_writes_report_outside(tmp_path):
    out = _build(tmp_path)
    rdir = tmp_path / "reports"
    res = verify_plugin_artifact(str(out), deep=True, report_dir=str(rdir))
    assert res.ok, {k: v for k, v in res.checks.items() if v != "passed"}
    assert res.claims["integrity_verified"] is True
    assert res.claims["authenticity_verified"] is False   # unsigned
    assert res.claims["processor_contract_verified"] is True
    assert res.claims["semantics_verified"] is True
    assert res.claims["official_eval_certified"] is False
    # report is OUTSIDE the sealed artifact.
    assert (rdir / "plugin_artifact_verification_report.json").exists()
    assert not (out / "plugin_artifact_verification_report.json").exists()


def test_verify_is_idempotent(tmp_path):
    out = _build(tmp_path)
    root1 = json.loads((out / "plugin_artifact_inventory.json").read_text())["artifact_root_sha256"]
    files_before = sorted(p.name for p in out.iterdir())
    verify_plugin_artifact(str(out), deep=True, report_dir=str(tmp_path / "r1"))
    verify_plugin_artifact(str(out), deep=True, report_dir=str(tmp_path / "r2"))
    root2 = json.loads((out / "plugin_artifact_inventory.json").read_text())["artifact_root_sha256"]
    assert root1 == root2
    assert sorted(p.name for p in out.iterdir()) == files_before   # artifact untouched


def test_semantics_detects_config_mismatch(tmp_path):
    from lerobot_policy_coreai_bridge.artifact import verify_artifact_semantics
    out = _build(tmp_path)
    p = out / "config.json"
    d = json.loads(p.read_text()); d["expected_action_dim"] = 999
    p.write_text(json.dumps(d))
    sem = verify_artifact_semantics(out)
    assert sem["config_action_dim"].startswith("failed")


# --- integrity: tamper / coverage / traversal ---

def test_verify_detects_content_tamper(tmp_path):
    out = _build(tmp_path)
    p = out / "config.json"
    d = json.loads(p.read_text()); d["expected_action_dim"] = 999
    p.write_text(json.dumps(d))
    res = verify_plugin_artifact(str(out), deep=False)
    assert not res.ok
    assert res.checks["inv:config.json"].startswith("failed")


def test_verify_detects_missing_file(tmp_path):
    out = _build(tmp_path)
    (out / "policy_preprocessor.json").unlink()
    assert not verify_plugin_artifact(str(out), deep=False).ok


def test_verify_detects_undeclared_file(tmp_path):
    out = _build(tmp_path)
    (out / "sneaky.bin").write_text("x")
    res = verify_plugin_artifact(str(out), deep=False)
    assert not res.ok
    assert res.checks["no_undeclared_files"].startswith("failed")


def test_verify_detects_checksums_extra_entry(tmp_path):
    out = _build(tmp_path)
    cs = json.loads((out / "checksums.json").read_text())
    cs["ghost.json"] = "sha256:" + "0" * 64
    (out / "checksums.json").write_text(json.dumps(cs))
    res = verify_plugin_artifact(str(out), deep=False)
    assert not res.ok
    assert res.checks["checksums_exact_coverage"].startswith("failed")


def test_verify_detects_traversal_path_in_inventory(tmp_path):
    out = _build(tmp_path)
    inv = json.loads((out / "plugin_artifact_inventory.json").read_text())
    inv["files"][0]["path"] = "../escape.json"
    (out / "plugin_artifact_inventory.json").write_text(json.dumps(inv))
    res = verify_plugin_artifact(str(out), deep=False)
    assert not res.ok
    assert res.checks["inventory_paths_safe"].startswith("failed")


def test_verify_detects_artifact_root_tamper(tmp_path):
    out = _build(tmp_path)
    inv = json.loads((out / "plugin_artifact_inventory.json").read_text())
    inv["artifact_root_sha256"] = "sha256:" + "1" * 64
    (out / "plugin_artifact_inventory.json").write_text(json.dumps(inv))
    assert not verify_plugin_artifact(str(out), deep=False).ok


# --- processor contract semantics ---

def test_build_fails_without_ownership(tmp_path):
    with pytest.raises(ProcessorOwnershipError):
        _build(tmp_path, ownership="none")


def test_build_fails_with_wrong_expects(tmp_path):
    with pytest.raises(ProcessorOwnershipError):
        _build(tmp_path, ownership="wrong_expects")


def test_build_fails_with_wrong_returns(tmp_path):
    with pytest.raises(ProcessorOwnershipError):
        _build(tmp_path, ownership="wrong_returns")


# --- source provenance ---

def test_external_reference_requires_resolved_commit(tmp_path):
    # a mutable ref like "main" without a resolved commit sha is refused.
    with pytest.raises(ArtifactError):
        build_plugin_artifact(_src(tmp_path), str(tmp_path / "p"), external=True,
                              requested_ref="main")


def test_external_reference_records_resolved_commit(tmp_path):
    out = tmp_path / "p"
    sha = "a" * 40
    build_plugin_artifact(_src(tmp_path), str(out), external=True,
                          requested_ref="main", resolved_commit_sha=sha)
    ref = json.loads((out / "plugin_artifact_manifest.json").read_text())[
        "source_coreai_artifact_reference"]
    assert ref["mode"] == "external" and ref["requested_ref"] == "main"
    assert ref["resolved_commit_sha"] == sha
    assert ref["embedded_manifest_sha256"].startswith("sha256:")
    assert verify_plugin_artifact(str(out), deep=True).ok


def test_external_invalid_commit_fails(tmp_path):
    with pytest.raises(ArtifactError):
        build_plugin_artifact(_src(tmp_path), str(tmp_path / "p"), external=True,
                              requested_ref="main", resolved_commit_sha="not-a-sha")


# --- version binding (pure) ---

def test_version_lockstep_required():
    ok, _ = check_version_compatibility(
        {"lerobot_coreai": "1.3.7", "lerobot_policy_coreai_bridge": "1.3.6"},
        {"lerobot_coreai": "1.3.7", "lerobot_policy_coreai_bridge": "1.3.7"})
    assert not ok


def test_version_installed_older_than_artifact_fails():
    ok, _ = check_version_compatibility(
        {"lerobot_coreai": "1.3.9", "lerobot_policy_coreai_bridge": "1.3.9"},
        {"lerobot_coreai": "1.3.7", "lerobot_policy_coreai_bridge": "1.3.7"})
    assert not ok


def test_version_major_mismatch_fails():
    ok, _ = check_version_compatibility(
        {"lerobot_coreai": "2.0.0", "lerobot_policy_coreai_bridge": "2.0.0"},
        {"lerobot_coreai": "1.3.7", "lerobot_policy_coreai_bridge": "1.3.7"})
    assert not ok


def test_version_compatible_passes():
    ok, _ = check_version_compatibility(
        {"lerobot_coreai": "1.3.7", "lerobot_policy_coreai_bridge": "1.3.7",
         "lerobot": "0.6.0"},
        {"lerobot_coreai": "1.3.7", "lerobot_policy_coreai_bridge": "1.3.7",
         "lerobot": "0.6.1"})
    assert ok


def test_lerobot_minor_mismatch_fails():
    ok, _ = check_version_compatibility(
        {"lerobot_coreai": "1.3.7", "lerobot_policy_coreai_bridge": "1.3.7",
         "lerobot": "0.7.0"},
        {"lerobot_coreai": "1.3.7", "lerobot_policy_coreai_bridge": "1.3.7",
         "lerobot": "0.6.0"})
    assert not ok


# --- honest claims ---

def test_manifest_claims_stay_false(tmp_path):
    out = _build(tmp_path)
    c = json.loads((out / "plugin_artifact_manifest.json").read_text())["claims"]
    for k in ("official_eval_certified", "upstream_native", "supports_training",
              "proves_task_success", "proves_physical_safety"):
        assert c[k] is False
