# rollout_verify.py — OFFLINE independent verifier for rollout evidence (v1.3.14).
#
# Given only a bundle directory, re-prove what the producer claimed WITHOUT trusting
# it: recompute every checksum, validate schemas, recompute bundle/matrix roots,
# require the full case matrix, and refuse any promoted/forbidden claim. Pure files +
# JSON + jsonschema — no lerobot, no network.

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jsonschema

from .rollout_evidence_schema import (
    BUNDLE_MANIFEST_SCHEMA,
    MATRIX_SCHEMA,
    READINESS_SCHEMA,
    REQUIRED_CASES,
    canonical_json_sha256,
)

REPORT_FILE = "official_rollout_readiness_report.json"
BUNDLE_MANIFEST_FILE = "bundle_manifest.json"
CHECKSUMS_FILE = "checksums.json"
MATRIX_FILE = "official_rollout_matrix_manifest.json"

_FORBIDDEN_TRUE = ("official_eval_certified", "authenticity_verified",
                   "proves_task_success", "proves_physical_safety")
# Sensitive substrings that must never appear in evidence JSON values.
_SECRET_RE = ("://", "token", "secret", "password", "authorization", "bearer",
              "api_key")


def _sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


@dataclass
class VerifyEvidenceResult:
    ok: bool
    checks: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "checks": self.checks}


def _scan_secret(obj: Any) -> str | None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            kl = str(k).lower()
            if any(s in kl for s in ("token", "secret", "password", "authorization",
                                     "bearer", "api_key")) and v not in (None, "", {}, []):
                return f"key:{k}"
            hit = _scan_secret(v)
            if hit:
                return hit
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            hit = _scan_secret(v)
            if hit:
                return hit
    elif isinstance(obj, str) and "://" in obj and "@" in obj.split("://", 1)[1][:64]:
        return "credential-in-url"
    return None


def _verify_case(case_dir: Path, checks: dict, prefix: str) -> tuple[bool, str | None]:
    """Verify one case bundle; return (ok, bundle_root_sha256)."""
    def ok(name, cond, reason=""):
        checks[f"{prefix}:{name}"] = "passed" if cond else f"failed: {reason}"
        return cond

    for fn in (REPORT_FILE, BUNDLE_MANIFEST_FILE, CHECKSUMS_FILE):
        if not ok(f"present:{fn}", (case_dir / fn).exists(), "missing"):
            return False, None

    # checksums cover exactly the recorded files, and match.
    try:
        recorded = json.loads((case_dir / CHECKSUMS_FILE).read_text())
    except Exception as exc:  # noqa: BLE001
        ok("checksums_parse", False, str(exc))
        return False, None
    tamper_ok = True
    for name, digest in recorded.items():
        actual = _sha256_file(case_dir / name) if (case_dir / name).exists() else None
        if actual != digest:
            tamper_ok = False
    ok("checksums_match", tamper_ok, "a listed file was modified")

    # report schema + claims + secrets.
    try:
        report = json.loads((case_dir / REPORT_FILE).read_text())
        jsonschema.validate(report, READINESS_SCHEMA)
        ok("report_schema", True)
    except Exception as exc:  # noqa: BLE001
        ok("report_schema", False, str(exc))
        return False, None
    ok("no_forbidden_claims",
       all(report["claims"].get(k) is not True for k in _FORBIDDEN_TRUE),
       "a forbidden claim is true")
    ok("no_secrets", _scan_secret(report) is None, "secret in report")

    # bundle manifest schema + recomputed bundle root over its file digests.
    try:
        bm = json.loads((case_dir / BUNDLE_MANIFEST_FILE).read_text())
        jsonschema.validate(bm, BUNDLE_MANIFEST_SCHEMA)
    except Exception as exc:  # noqa: BLE001
        ok("bundle_manifest_schema", False, str(exc))
        return False, None
    recomputed = canonical_json_sha256(
        sorted((p, h) for p, h in bm["files"].items()))
    root_ok = recomputed == bm["bundle_root_sha256"]
    ok("bundle_root", root_ok, "bundle root mismatch")
    # every manifest file digest matches the actual file.
    files_ok = all(
        (case_dir / p).exists() and _sha256_file(case_dir / p) == h
        for p, h in bm["files"].items())
    ok("bundle_files_match", files_ok, "a bundle-manifest file digest mismatch")

    case_ok = all(v == "passed" for k, v in checks.items() if k.startswith(prefix))
    return case_ok, bm.get("bundle_root_sha256")


def verify_official_rollout_evidence(
    bundle_dir: str, *, require_complete_matrix: bool = True,
) -> VerifyEvidenceResult:
    """Independently verify a rollout evidence directory offline (v1.3.14)."""
    root = Path(bundle_dir)
    checks: dict[str, str] = {}

    def ok(name, cond, reason=""):
        checks[name] = "passed" if cond else f"failed: {reason}"
        return cond

    if not ok("bundle_dir_exists", root.is_dir(), "missing"):
        return VerifyEvidenceResult(False, checks)

    case_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    seen_cases = {p.name: p for p in case_dirs}
    all_ok = True
    case_roots: dict[str, str] = {}
    for name, cdir in seen_cases.items():
        cok, broot = _verify_case(cdir, checks, f"case[{name}]")
        all_ok &= cok
        if broot:
            case_roots[name] = broot

    if require_complete_matrix:
        for req in REQUIRED_CASES:
            present = ok(f"required_case:{req}", req in seen_cases, "missing case")
            all_ok &= present

    # matrix manifest (if present) must schema-validate and its root recompute.
    if (root / MATRIX_FILE).exists():
        try:
            mx = json.loads((root / MATRIX_FILE).read_text())
            jsonschema.validate(mx, MATRIX_SCHEMA)
            recomputed = canonical_json_sha256(
                sorted((c, v["bundle_root_sha256"]) for c, v in mx["cases"].items()))
            all_ok &= ok("matrix_root", recomputed == mx["matrix_root_sha256"],
                         "matrix root mismatch")
        except Exception as exc:  # noqa: BLE001
            all_ok &= ok("matrix_schema", False, str(exc))
    elif require_complete_matrix:
        all_ok &= ok("matrix_present", False, f"{MATRIX_FILE} missing")

    return VerifyEvidenceResult(bool(all_ok), checks)
