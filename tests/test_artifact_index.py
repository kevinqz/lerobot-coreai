# test_artifact_index.py — local artifact registry (v1.2.2).

import hashlib
import json
from importlib.resources import files

import jsonschema
import pytest

from lerobot_coreai.artifact_index import (
    ARTIFACT_INDEX_ENTRY_SCHEMA_VERSION, ArtifactIndexError, add_artifact,
    find_entries, init_index, list_entries, verify_index,
)


def _bundle(tmp_path, name="evo1-pusht-bridge-benchmark", *, overclaim=False,
            secret=False):
    d = tmp_path / name
    (d / "reports").mkdir(parents=True)
    report = {"ok": True, "claims": {"proves_physical_safety": overclaim}}
    if secret:
        report["auth"] = {"token": "rawsecretvalue"}
    (d / "reports" / "compat.json").write_text(json.dumps(report))
    checks = {"reports/compat.json":
              hashlib.sha256((d / "reports" / "compat.json").read_bytes()).hexdigest()}
    (d / "benchmark_manifest.json").write_text(json.dumps({
        "schema_version": "lerobot-coreai.bridge_benchmark_pack.v0",
        "bundle_type": "bridge_benchmark",
        "policy_path": "kevinqz/EVO1-SO100-CoreAI", "dataset_repo_id": "lerobot/pusht",
        "reports": {"compat": "reports/compat.json"},
        "claims": {"proves_task_success": False, "proves_physical_safety": False,
                   "authorizes_robot_actuation": False}}))
    (d / "checksums.json").write_text(json.dumps(checks))
    return d


def test_init_creates_index(tmp_path):
    root = init_index(tmp_path / "idx")
    assert (tmp_path / "idx" / "index.json").is_file()
    assert root["entries"] == []


def test_add_and_list(tmp_path):
    idx = tmp_path / "idx"
    init_index(idx)
    res = add_artifact(idx, _bundle(tmp_path), "bridge_benchmark",
                       release_channel="public-demo", created_at="2026-07-10T00:00:00Z")
    assert res.entry["policy_path"] == "kevinqz/EVO1-SO100-CoreAI"
    assert res.entry["signature_verified"] is False  # no signature provided
    entries = list_entries(idx)
    assert len(entries) == 1
    schema = json.loads(files("lerobot_coreai.schemas").joinpath(
        "artifact-index-entry.schema.json").read_text())
    jsonschema.validate(entries[0], schema)


def test_index_root_schema_valid(tmp_path):
    idx = tmp_path / "idx"
    init_index(idx)
    add_artifact(idx, _bundle(tmp_path), "bridge_benchmark",
                 created_at="2026-07-10T00:00:00Z")
    root = json.loads((idx / "index.json").read_text())
    schema = json.loads(files("lerobot_coreai.schemas").joinpath(
        "artifact-index.schema.json").read_text())
    jsonschema.validate(root, schema)


def test_overclaim_blocks_add(tmp_path):
    idx = tmp_path / "idx"
    init_index(idx)
    with pytest.raises(ArtifactIndexError):
        add_artifact(idx, _bundle(tmp_path, overclaim=True), "bridge_benchmark")


def test_secret_blocks_add(tmp_path):
    idx = tmp_path / "idx"
    init_index(idx)
    with pytest.raises(ArtifactIndexError):
        add_artifact(idx, _bundle(tmp_path, secret=True), "bridge_benchmark")


def test_tampered_artifact_blocks_add(tmp_path):
    idx = tmp_path / "idx"
    init_index(idx)
    d = _bundle(tmp_path)
    victim = d / "reports" / "compat.json"
    victim.write_text(victim.read_text().replace("true", "false"))
    with pytest.raises(ArtifactIndexError):
        add_artifact(idx, d, "bridge_benchmark")


def test_duplicate_id_not_silently_overwritten(tmp_path):
    idx = tmp_path / "idx"
    init_index(idx)
    d = _bundle(tmp_path)
    add_artifact(idx, d, "bridge_benchmark", created_at="2026-07-10T00:00:00Z")
    with pytest.raises(ArtifactIndexError):
        add_artifact(idx, d, "bridge_benchmark", created_at="2026-07-10T00:00:00Z")
    # force allowed.
    add_artifact(idx, d, "bridge_benchmark", created_at="2026-07-10T00:00:00Z", force=True)


def test_find_filters(tmp_path):
    idx = tmp_path / "idx"
    init_index(idx)
    add_artifact(idx, _bundle(tmp_path, "a"), "bridge_benchmark",
                 release_channel="public-demo", created_at="2026-07-10T00:00:01Z")
    add_artifact(idx, _bundle(tmp_path, "b"), "demo_pack",
                 release_channel="internal", created_at="2026-07-10T00:00:02Z")
    assert len(find_entries(idx, artifact_type="bridge_benchmark")) == 1
    assert len(find_entries(idx, release_channel="internal")) == 1
    assert len(find_entries(idx, policy_path="kevinqz/EVO1-SO100-CoreAI")) == 2
    assert len(find_entries(idx, dataset_repo_id="nope")) == 0


def test_verify_detects_post_add_tamper(tmp_path):
    idx = tmp_path / "idx"
    init_index(idx)
    d = _bundle(tmp_path)
    add_artifact(idx, d, "bridge_benchmark", created_at="2026-07-10T00:00:00Z")
    # Tamper after indexing.
    victim = d / "reports" / "compat.json"
    victim.write_text(json.dumps({"ok": False, "claims": {}}))
    result = verify_index(idx)
    assert result.ok is False


def test_verify_clean_index_ok(tmp_path):
    idx = tmp_path / "idx"
    init_index(idx)
    add_artifact(idx, _bundle(tmp_path), "bridge_benchmark",
                 created_at="2026-07-10T00:00:00Z")
    assert verify_index(idx).ok is True


def test_signature_verified_only_true_after_real_check(tmp_path):
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from lerobot_coreai.provenance import build_provenance
    from lerobot_coreai.signing import (
        build_signature_manifest, canonical_payload, load_private_key,
        public_key_bytes, sign_payload,
    )
    from lerobot_coreai.trust_policy import build_signed_payload_fields

    idx = tmp_path / "idx"
    init_index(idx)
    d = _bundle(tmp_path)
    prov = tmp_path / "provenance.json"
    prov.write_text(json.dumps(build_provenance(d, "bridge_benchmark",
                                                created_at="2026-07-10T00:00:00Z")))
    seed = Ed25519PrivateKey.generate().private_bytes(
        serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
        serialization.NoEncryption()).hex()
    priv = load_private_key(seed)
    fields = build_signed_payload_fields(d, prov)
    sig = tmp_path / "signature.json"
    sig.write_text(json.dumps(build_signature_manifest(
        signer_name="T", public_bytes=public_key_bytes(priv), signed_fields=fields,
        signature_b64=sign_payload(canonical_payload(fields), priv),
        signed_at="2026-07-10T00:01:00Z")))

    res = add_artifact(idx, d, "bridge_benchmark", signature=sig, provenance=prov,
                       created_at="2026-07-10T00:02:00Z")
    assert res.entry["signature_verified"] is True
    assert res.entry["signature_fingerprint"].startswith("sha256:")
