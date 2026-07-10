# trust_policy.py — trust policy + signed-artifact verification (v1.2.0).
#
# Ties provenance + signature + policy together. Verification fails closed on:
# artifact tamper (any file's checksum mismatch), anchor tamper (manifest /
# checksums / provenance), an invalid or forged signature, an untrusted signer,
# missing required artifacts, or a forbidden claim set true. Proves signature
# validity + integrity only — never physical safety or actuation authorization.

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .provenance import sha256_bytes, sha256_file
from .signing import (
    canonical_payload, fingerprint, load_public_key_b64, verify_payload,
)

TRUST_POLICY_SCHEMA_VERSION = "lerobot-coreai.trust_policy.v0"

_DEFAULT_FORBIDDEN = [
    "proves_physical_safety",
    "authorizes_robot_actuation",
    "native_upstream_registry",
    "supports_training",
]


@dataclass
class VerifyResult:
    ok: bool
    checks: list[dict[str, Any]] = field(default_factory=list)


def load_trust_policy(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _find_true_claims(obj: Any, forbidden: set[str]) -> list[str]:
    found: list[str] = []

    def _walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if k in forbidden and v is True:
                    found.append(k)
                _walk(v)
        elif isinstance(node, list):
            for x in node:
                _walk(x)

    _walk(obj)
    return found


def build_signed_payload_fields(artifact_dir: Path, provenance_path: Path) -> dict[str, str]:
    """Compute the three anchor hashes the signature commits to."""
    artifact_dir = Path(artifact_dir)
    fields = {"provenance_sha256": sha256_file(provenance_path)}
    manifest = artifact_dir / "benchmark_manifest.json"
    checksums = artifact_dir / "checksums.json"
    if manifest.is_file():
        fields["manifest_sha256"] = sha256_file(manifest)
    if checksums.is_file():
        fields["checksums_sha256"] = sha256_file(checksums)
    return fields


def verify_signed_artifact(
    artifact_dir: Path, provenance_path: Path, signature_path: Path,
    trust_policy: dict[str, Any] | None = None,
) -> VerifyResult:
    """Verify a signed artifact end-to-end. Fail-closed."""
    artifact_dir = Path(artifact_dir)
    checks: list[dict[str, Any]] = []

    def _c(name, passed, detail=""):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    try:
        signature = json.loads(Path(signature_path).read_text())
        provenance = json.loads(Path(provenance_path).read_text())
        _c("signature_and_provenance_readable", True)
    except Exception as e:
        _c("signature_and_provenance_readable", False, str(e))
        return VerifyResult(ok=False, checks=checks)

    # 1. Anchor hashes recomputed from files must equal the signed payload.
    signed = signature.get("signed_payload", {})
    recomputed = build_signed_payload_fields(artifact_dir, provenance_path)
    anchors_ok = all(signed.get(k) == v for k, v in recomputed.items()) and \
        set(signed.keys()) == set(recomputed.keys())
    _c("anchor_hashes_match_signed_payload", anchors_ok,
       "" if anchors_ok else f"signed={signed} recomputed={recomputed}")

    # 2. Every checksums.json entry must match the actual file (report tamper).
    checksums_path = artifact_dir / "checksums.json"
    tampered: list[str] = []
    if checksums_path.is_file():
        checksums = json.loads(checksums_path.read_text())
        for rel, expected in checksums.items():
            fp = artifact_dir / rel
            # checksums.json stores bare hex; normalize both sides.
            actual = sha256_file(fp).split(":", 1)[1] if fp.is_file() else None
            if actual != expected:
                tampered.append(rel)
    _c("artifact_files_untampered", not tampered,
       "" if not tampered else f"tampered/missing: {tampered}")

    # 3. Provenance's own anchored hashes must still match the files.
    prov_ok = True
    for name, h in provenance.get("artifact_hashes", {}).items():
        fp = artifact_dir / name
        if not fp.is_file() or sha256_file(fp) != h:
            prov_ok = False
            break
    _c("provenance_hashes_match_files", prov_ok)

    # 4. Cryptographic signature over the canonical signed payload.
    signer = signature.get("signer", {})
    pub_b64 = signer.get("public_key")
    sig_valid = False
    fp_ok = False
    if pub_b64:
        try:
            import base64
            pub_raw = base64.b64decode(pub_b64)
            fp_ok = fingerprint(pub_raw) == signer.get("key_fingerprint")
            public_key = load_public_key_b64(pub_b64)
            sig_valid = verify_payload(
                canonical_payload(signed), signature.get("signature", ""), public_key)
        except Exception:
            sig_valid = False
    _c("key_fingerprint_matches_public_key", fp_ok)
    _c("signature_cryptographically_valid", sig_valid)

    # 5. Trust policy (optional).
    if trust_policy is not None:
        trusted = {k.get("fingerprint") for k in trust_policy.get("trusted_keys", [])}
        signer_fp = signer.get("key_fingerprint")
        _c("signer_trusted", signer_fp in trusted,
           "" if signer_fp in trusted else f"untrusted signer {signer_fp}")

        required = trust_policy.get("required_artifacts", [])
        missing = [r for r in required if not (artifact_dir / r).is_file()
                   and r not in ("provenance.json",)]
        # provenance.json may live outside the artifact dir; accept the passed path.
        if "provenance.json" in required and not Path(provenance_path).is_file():
            missing.append("provenance.json")
        _c("required_artifacts_present", not missing,
           "" if not missing else f"missing: {missing}")

        forbidden = set(trust_policy.get("forbidden_claims", _DEFAULT_FORBIDDEN))
        violations: list[str] = []
        for name in ("benchmark_manifest.json",):
            fp = artifact_dir / name
            if fp.is_file():
                violations += _find_true_claims(json.loads(fp.read_text()), forbidden)
        violations += _find_true_claims(provenance, forbidden)
        for rel in provenance.get("source_reports", {}).values():
            fp = artifact_dir / rel
            if fp.is_file():
                try:
                    violations += _find_true_claims(json.loads(fp.read_text()), forbidden)
                except Exception:
                    pass
        _c("no_forbidden_claims", not violations,
           "" if not violations else f"forbidden: {sorted(set(violations))}")

    ok = all(c["passed"] for c in checks)
    return VerifyResult(ok=ok, checks=checks)
