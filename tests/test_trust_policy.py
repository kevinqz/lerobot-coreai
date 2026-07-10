# test_trust_policy.py — signed-artifact verification + trust policy (v1.2.0).

import base64
import json

import pytest

pytest.importorskip("cryptography")

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from lerobot_coreai.provenance import build_provenance
from lerobot_coreai.signing import (
    build_signature_manifest, canonical_payload, load_private_key,
    public_key_bytes, sign_payload,
)
from lerobot_coreai.trust_policy import (
    build_signed_payload_fields, verify_signed_artifact,
)


def _seed():
    k = Ed25519PrivateKey.generate()
    return k.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
                           serialization.NoEncryption()).hex()


def _make_signed(tmp_path, seed=None, forbidden_claim=False):
    d = tmp_path / "bundle"
    (d / "reports").mkdir(parents=True)
    report = {"ok": True, "claims": {"proves_physical_safety": forbidden_claim}}
    (d / "reports" / "compat.json").write_text(json.dumps(report))
    (d / "benchmark_manifest.json").write_text(json.dumps({
        "schema_version": "lerobot-coreai.bridge_benchmark_pack.v0",
        "bundle_type": "bridge_benchmark",
        "reports": {"compat": "reports/compat.json"},
        "claims": {"proves_task_success": False, "proves_physical_safety": False,
                   "authorizes_robot_actuation": False}}))
    # checksums.json binds every report file (bare hex, like the packer writes).
    import hashlib
    checks = {"reports/compat.json":
              hashlib.sha256((d / "reports" / "compat.json").read_bytes()).hexdigest()}
    (d / "checksums.json").write_text(json.dumps(checks))

    prov_path = tmp_path / "provenance.json"
    prov_path.write_text(json.dumps(build_provenance(d, "bridge_benchmark",
                                                     created_at="2026-07-10T00:00:00Z")))

    seed = seed or _seed()
    priv = load_private_key(seed)
    fields = build_signed_payload_fields(d, prov_path)
    manifest = build_signature_manifest(
        signer_name="Tester", public_bytes=public_key_bytes(priv),
        signed_fields=fields, signature_b64=sign_payload(canonical_payload(fields), priv),
        signed_at="2026-07-10T00:01:00Z")
    sig_path = tmp_path / "signature.json"
    sig_path.write_text(json.dumps(manifest))
    return d, prov_path, sig_path, manifest


def test_valid_signature_verifies(tmp_path):
    d, prov, sig, _ = _make_signed(tmp_path)
    result = verify_signed_artifact(d, prov, sig)
    assert result.ok is True


def test_tampered_report_fails(tmp_path):
    d, prov, sig, _ = _make_signed(tmp_path)
    victim = d / "reports" / "compat.json"
    victim.write_text(victim.read_text().replace("true", "false"))
    result = verify_signed_artifact(d, prov, sig)
    assert result.ok is False
    names = {c["name"]: c["passed"] for c in result.checks}
    assert names["artifact_files_untampered"] is False


def test_tampered_manifest_fails(tmp_path):
    d, prov, sig, _ = _make_signed(tmp_path)
    m = d / "benchmark_manifest.json"
    m.write_text(m.read_text() + " ")  # whitespace changes the hash
    result = verify_signed_artifact(d, prov, sig)
    assert result.ok is False
    names = {c["name"]: c["passed"] for c in result.checks}
    assert names["anchor_hashes_match_signed_payload"] is False


def test_tampered_provenance_fails(tmp_path):
    d, prov, sig, _ = _make_signed(tmp_path)
    prov.write_text(prov.read_text() + " ")
    result = verify_signed_artifact(d, prov, sig)
    assert result.ok is False


def test_forged_signature_fails(tmp_path):
    d, prov, sig, manifest = _make_signed(tmp_path)
    manifest["signature"] = base64.b64encode(b"\x00" * 64).decode()
    sig.write_text(json.dumps(manifest))
    result = verify_signed_artifact(d, prov, sig)
    assert result.ok is False
    names = {c["name"]: c["passed"] for c in result.checks}
    assert names["signature_cryptographically_valid"] is False


def test_untrusted_signer_fails(tmp_path):
    d, prov, sig, manifest = _make_signed(tmp_path)
    policy = {"schema_version": "lerobot-coreai.trust_policy.v0",
              "trusted_keys": [{"name": "someone else", "fingerprint": "sha256:deadbeef"}]}
    result = verify_signed_artifact(d, prov, sig, trust_policy=policy)
    assert result.ok is False
    names = {c["name"]: c["passed"] for c in result.checks}
    assert names["signer_trusted"] is False


def test_trusted_signer_passes(tmp_path):
    d, prov, sig, manifest = _make_signed(tmp_path)
    policy = {"schema_version": "lerobot-coreai.trust_policy.v0",
              "trusted_keys": [{"name": "Tester",
                                "fingerprint": manifest["signer"]["key_fingerprint"]}]}
    result = verify_signed_artifact(d, prov, sig, trust_policy=policy)
    assert result.ok is True


def test_forbidden_claim_fails_policy(tmp_path):
    d, prov, sig, manifest = _make_signed(tmp_path, forbidden_claim=True)
    policy = {"schema_version": "lerobot-coreai.trust_policy.v0",
              "trusted_keys": [{"name": "Tester",
                                "fingerprint": manifest["signer"]["key_fingerprint"]}],
              "forbidden_claims": ["proves_physical_safety"]}
    result = verify_signed_artifact(d, prov, sig, trust_policy=policy)
    assert result.ok is False
    names = {c["name"]: c["passed"] for c in result.checks}
    assert names["no_forbidden_claims"] is False
