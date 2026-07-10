# test_cli_sign_verify.py — provenance/sign/verify CLI flow (v1.2.0).

import json

import pytest

pytest.importorskip("cryptography")

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from lerobot_coreai import cli


def _seed_hex():
    k = Ed25519PrivateKey.generate()
    return k.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
                           serialization.NoEncryption()).hex()


def _bundle(tmp_path):
    d = tmp_path / "bundle"
    (d / "reports").mkdir(parents=True)
    (d / "reports" / "compat.json").write_text(json.dumps({"ok": True}))
    (d / "benchmark_manifest.json").write_text(json.dumps({
        "schema_version": "lerobot-coreai.bridge_benchmark_pack.v0",
        "bundle_type": "bridge_benchmark",
        "reports": {"compat": "reports/compat.json"},
        "claims": {"proves_task_success": False, "proves_physical_safety": False,
                   "authorizes_robot_actuation": False}}))
    import hashlib
    (d / "checksums.json").write_text(json.dumps({
        "reports/compat.json":
        hashlib.sha256((d / "reports" / "compat.json").read_bytes()).hexdigest()}))
    return d


def test_full_cli_flow(tmp_path, monkeypatch):
    d = _bundle(tmp_path)
    prov = tmp_path / "provenance.json"
    sig = tmp_path / "signature.json"

    assert cli.main(["provenance-create", "--artifact-dir", str(d),
                     "--artifact-type", "bridge_benchmark", "--output", str(prov)]) == 0

    monkeypatch.setenv("LEROBOT_COREAI_SIGNING_KEY", _seed_hex())
    assert cli.main(["sign-artifact", "--artifact-dir", str(d), "--provenance", str(prov),
                     "--key-env", "LEROBOT_COREAI_SIGNING_KEY", "--signer-name", "Tester",
                     "--output", str(sig)]) == 0

    # The private key must never appear in the signature file.
    assert "LEROBOT_COREAI_SIGNING_KEY" not in sig.read_text()
    seed = __import__("os").environ["LEROBOT_COREAI_SIGNING_KEY"]
    assert seed not in sig.read_text()

    assert cli.main(["verify-signature", "--artifact-dir", str(d),
                     "--provenance", str(prov), "--signature", str(sig), "--json"]) == 0


def test_cli_sign_missing_env_fails(tmp_path):
    d = _bundle(tmp_path)
    prov = tmp_path / "provenance.json"
    cli.main(["provenance-create", "--artifact-dir", str(d),
              "--artifact-type", "bridge_benchmark", "--output", str(prov)])
    rc = cli.main(["sign-artifact", "--artifact-dir", str(d), "--provenance", str(prov),
                   "--key-env", "DEFINITELY_UNSET_KEY_VAR", "--output", str(tmp_path / "s.json")])
    assert rc == 1


def test_cli_verify_detects_tamper(tmp_path, monkeypatch):
    d = _bundle(tmp_path)
    prov = tmp_path / "provenance.json"
    sig = tmp_path / "signature.json"
    cli.main(["provenance-create", "--artifact-dir", str(d),
              "--artifact-type", "bridge_benchmark", "--output", str(prov)])
    monkeypatch.setenv("SK", _seed_hex())
    cli.main(["sign-artifact", "--artifact-dir", str(d), "--provenance", str(prov),
              "--key-env", "SK", "--output", str(sig)])
    victim = d / "reports" / "compat.json"
    victim.write_text(json.dumps({"ok": False}))
    assert cli.main(["verify-signature", "--artifact-dir", str(d),
                     "--provenance", str(prov), "--signature", str(sig)]) == 1
