# test_artifact.py — canonical plugin artifact build + verify (v1.3.6).

import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("lerobot")

from lerobot_policy_coreai_bridge.artifact import (  # noqa: E402
    ArtifactError, build_plugin_artifact, verify_plugin_artifact,
)
from lerobot_policy_coreai_bridge.processor_coreai_bridge import (  # noqa: E402
    ProcessorOwnershipError,
)

HORIZON, ACTION_DIM = 3, 7


def _manifest(*, ownership=True):
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
    if ownership:
        m["contracts"] = {"processor": {
            "observation_input": {"owner": "coreai_runner", "expects": "raw"},
            "action_output": {"owner": "coreai_runner", "returns": "post"}}}
    return m


def _src(tmp_path, **kw):
    d = tmp_path / "coreai"
    d.mkdir()
    (d / "lerobot-coreai.json").write_text(json.dumps(_manifest(**kw)))
    return str(d)


def _build(tmp_path, **kw):
    src = _src(tmp_path, **kw)
    out = tmp_path / "plugin"
    build_plugin_artifact(src, str(out), runner_url_env="COREAI_RUNNER_URL")
    return out


def test_build_writes_canonical_layout(tmp_path):
    out = _build(tmp_path)
    for fn in ("config.json", "policy_preprocessor.json", "policy_postprocessor.json",
               "lerobot-coreai.json", "plugin_artifact_manifest.json",
               "checksums.json", "README.md"):
        assert (out / fn).exists(), fn


def test_build_persists_no_local_path_or_secret(tmp_path):
    out = _build(tmp_path)
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["coreai_artifact"] == ""          # root, not a local path
    assert cfg["runner_url_env"] == "COREAI_RUNNER_URL"  # only the env var NAME
    assert "http://" not in (out / "config.json").read_text()


def test_verify_passes_on_fresh_artifact(tmp_path):
    out = _build(tmp_path)
    res = verify_plugin_artifact(str(out), deep=True)
    assert res.ok, {k: v for k, v in res.checks.items() if v != "passed"}


def test_verify_detects_tamper(tmp_path):
    out = _build(tmp_path)
    cfg_path = out / "config.json"
    d = json.loads(cfg_path.read_text())
    d["expected_action_dim"] = 999
    cfg_path.write_text(json.dumps(d))
    res = verify_plugin_artifact(str(out), deep=False)
    assert not res.ok
    assert res.checks["checksum:config.json"].startswith("failed")


def test_verify_detects_missing_file(tmp_path):
    out = _build(tmp_path)
    (out / "policy_preprocessor.json").unlink()
    res = verify_plugin_artifact(str(out), deep=False)
    assert not res.ok


def test_build_fails_without_processor_ownership(tmp_path):
    src = _src(tmp_path, ownership=False)
    out = tmp_path / "plugin"
    with pytest.raises(ProcessorOwnershipError):
        build_plugin_artifact(src, str(out), runner_url_env="COREAI_RUNNER_URL")


def test_external_reference_requires_revision_and_sha(tmp_path):
    src = _src(tmp_path)
    out = tmp_path / "plugin"
    with pytest.raises(ArtifactError):
        build_plugin_artifact(src, str(out), external=True)  # no revision/sha256


def test_external_reference_recorded_when_pinned(tmp_path):
    src = _src(tmp_path)
    out = tmp_path / "plugin"
    build_plugin_artifact(src, str(out), external=True,
                          external_revision="abc123", external_sha256="sha256:deadbeef")
    pm = json.loads((out / "plugin_artifact_manifest.json").read_text())
    ref = pm["coreai_artifact_reference"]
    assert ref["mode"] == "external" and ref["revision"] == "abc123"
    assert verify_plugin_artifact(str(out), deep=True).ok


def test_manifest_claims_stay_false(tmp_path):
    out = _build(tmp_path)
    pm = json.loads((out / "plugin_artifact_manifest.json").read_text())
    c = pm["claims"]
    assert c["official_eval_certified"] is False
    assert c["upstream_native"] is False
    assert c["supports_training"] is False
    assert c["proves_task_success"] is False
    assert c["proves_physical_safety"] is False
