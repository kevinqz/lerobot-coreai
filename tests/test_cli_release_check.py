# test_cli_release_check.py — release-check CLI incl. signed public-demo (v1.2.1).

import json

import pytest

from lerobot_coreai import cli

_CANON = ["lerobot_compatibility_report.json", "lerobot_bridge_report.json",
          "feature_mapping.json", "eval_v2_report.json", "obs_bridge_report.json"]


def _bundle(tmp_path):
    d = tmp_path / "bundle"
    (d / "reports").mkdir(parents=True)
    for r in _CANON:
        (d / "reports" / r).write_text(json.dumps({"ok": True, "claims": {}}))
    import hashlib
    checks = {}
    for r in _CANON:
        fp = d / "reports" / r
        checks[f"reports/{r}"] = hashlib.sha256(fp.read_bytes()).hexdigest()
    (d / "benchmark_manifest.json").write_text(json.dumps({
        "schema_version": "lerobot-coreai.bridge_benchmark_pack.v0",
        "bundle_type": "bridge_benchmark",
        "reports": {r.split(".")[0]: f"reports/{r}" for r in _CANON},
        "claims": {"proves_task_success": False, "proves_physical_safety": False,
                   "authorizes_robot_actuation": False}}))
    (d / "checksums.json").write_text(json.dumps(checks))
    return d


def test_internal_channel_rc0(tmp_path):
    d = _bundle(tmp_path)
    assert cli.main(["release-check", "--artifact-dir", str(d),
                     "--channel", "internal", "--json"]) == 0


def test_public_demo_without_signature_rc1(tmp_path):
    d = _bundle(tmp_path)
    assert cli.main(["release-check", "--artifact-dir", str(d),
                     "--channel", "public-demo", "--json"]) == 1


def test_unknown_channel_rc1(tmp_path):
    d = _bundle(tmp_path)
    assert cli.main(["release-check", "--artifact-dir", str(d),
                     "--channel", "nonsense"]) == 1


def test_public_demo_with_valid_signature_rc0(tmp_path, monkeypatch):
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    d = _bundle(tmp_path)
    prov = tmp_path / "provenance.json"
    sig = tmp_path / "signature.json"
    cli.main(["provenance-create", "--artifact-dir", str(d),
              "--artifact-type", "bridge_benchmark", "--output", str(prov)])
    seed = Ed25519PrivateKey.generate().private_bytes(
        serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
        serialization.NoEncryption()).hex()
    monkeypatch.setenv("SK", seed)
    cli.main(["sign-artifact", "--artifact-dir", str(d), "--provenance", str(prov),
              "--key-env", "SK", "--output", str(sig)])

    rc = cli.main(["release-check", "--artifact-dir", str(d), "--channel", "public-demo",
                   "--provenance", str(prov), "--signature", str(sig),
                   "--output-dir", str(tmp_path / "rc"), "--json"])
    assert rc == 0
    assert (tmp_path / "rc" / "release_check_report.json").is_file()


def test_release_policy_override_file(tmp_path):
    d = _bundle(tmp_path)
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({
        "schema_version": "lerobot-coreai.release_policy.v0", "channel": "custom",
        "require_signature": False, "require_no_overclaims": True}))
    assert cli.main(["release-check", "--artifact-dir", str(d),
                     "--release-policy", str(policy), "--json"]) == 0
