# rollout_verify.py — OFFLINE independent verifier for rollout evidence (v1.3.14/1.3.15).
#
# Given only a bundle directory, re-prove what the producer claimed WITHOUT trusting
# it: path-safe file access, exact checksum coverage, schema validation, SEMANTIC
# replay (re-derive every check from raw records), recomputed bundle/matrix roots,
# matrix<->case binding, the full case matrix, and refusal of promoted/forbidden
# claims or secrets. Pure files + JSON + jsonschema — no lerobot, no network.

from __future__ import annotations

import hashlib
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
from .rollout_replay import replay_rollout_evidence

REPORT_FILE = "official_rollout_readiness_report.json"
BUNDLE_MANIFEST_FILE = "bundle_manifest.json"
CHECKSUMS_FILE = "checksums.json"
MEASUREMENTS_FILE = "measurements.json"
TRACE_FILE = "official_rollout_trace.jsonl"
README_MD = "official_rollout_readiness_report.md"
MATRIX_FILE = "official_rollout_matrix_manifest.json"

# content files (checksums must cover exactly these + the bundle manifest).
_CONTENT = (REPORT_FILE, README_MD, TRACE_FILE, MEASUREMENTS_FILE)
_EXPECTED_CHECKSUMS = set(_CONTENT) | {BUNDLE_MANIFEST_FILE}
_SECRET_KEYS = ("token", "secret", "password", "authorization", "bearer", "api_key")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _safe_name(name: str) -> bool:
    return (name == Path(name).name and not Path(name).is_absolute()
            and ".." not in name.split("/") and name not in ("", ".", ".."))


def _scan_secret(obj: Any) -> str | None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if any(s in str(k).lower() for s in _SECRET_KEYS) and v not in (None, "", {}, []):
                return f"key:{k}"
            hit = _scan_secret(v)
            if hit:
                return hit
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            hit = _scan_secret(v)
            if hit:
                return hit
    elif isinstance(obj, str) and "://" in obj and "@" in obj.split("://", 1)[1][:80]:
        return "credential-in-url"
    return None


@dataclass
class VerifyEvidenceResult:
    ok: bool
    checks: dict[str, str] = field(default_factory=dict)
    case_roots: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "checks": self.checks}


def _verify_case(case_dir: Path, checks: dict, prefix: str) -> tuple[bool, str | None]:
    def ok(name, cond, reason=""):
        checks[f"{prefix}:{name}"] = "passed" if cond else f"failed: {reason}"
        return cond

    if not ok("no_symlinks", not any(p.is_symlink() for p in case_dir.iterdir()),
              "symlink present"):
        return False, None
    for fn in (REPORT_FILE, BUNDLE_MANIFEST_FILE, CHECKSUMS_FILE, MEASUREMENTS_FILE):
        if not ok(f"present:{fn}", (case_dir / fn).exists(), "missing"):
            return False, None

    # checksums: path-safe, exact coverage, no tamper.
    try:
        recorded = json.loads((case_dir / CHECKSUMS_FILE).read_text())
    except Exception as exc:  # noqa: BLE001
        ok("checksums_parse", False, str(exc)); return False, None
    if not ok("checksums_paths_safe", all(_safe_name(k) for k in recorded),
              "unsafe path in checksums"):
        return False, None
    ok("checksums_exact_coverage", set(recorded) == _EXPECTED_CHECKSUMS,
       f"{sorted(recorded)} != {sorted(_EXPECTED_CHECKSUMS)}")
    ok("checksums_match", all(
        (case_dir / n).exists() and _sha256_file(case_dir / n) == d
        for n, d in recorded.items()), "a listed file was modified")

    # schema + claims + secrets.
    try:
        report = json.loads((case_dir / REPORT_FILE).read_text())
        jsonschema.validate(report, READINESS_SCHEMA)
        ok("report_schema", True)
    except Exception as exc:  # noqa: BLE001
        ok("report_schema", False, str(exc)); return False, None
    ok("no_forbidden_claims", all(
        report["claims"].get(k) is not True for k in
        ("official_eval_certified", "authenticity_verified", "proves_task_success",
         "proves_physical_safety")), "a forbidden claim is true")
    ok("no_secrets", _scan_secret(report) is None
       and _scan_secret(json.loads((case_dir / MEASUREMENTS_FILE).read_text())) is None,
       "secret detected")

    # SEMANTIC replay (v1.3.15): re-derive every check from raw records.
    rep = replay_rollout_evidence(str(case_dir))
    ok("semantic_replay", rep.ok, "; ".join(rep.errors)[:200])

    # bundle manifest schema + recomputed root + per-file digests.
    try:
        bm = json.loads((case_dir / BUNDLE_MANIFEST_FILE).read_text())
        jsonschema.validate(bm, BUNDLE_MANIFEST_SCHEMA)
    except Exception as exc:  # noqa: BLE001
        ok("bundle_manifest_schema", False, str(exc)); return False, None
    ok("bundle_manifest_paths_safe", all(_safe_name(p) for p in bm["files"]),
       "unsafe path in bundle manifest")
    ok("bundle_manifest_coverage", set(bm["files"]) == set(_CONTENT),
       "manifest does not cover exactly the content files")
    ok("bundle_root", canonical_json_sha256(sorted(bm["files"].items()))
       == bm["bundle_root_sha256"], "bundle root mismatch")
    ok("bundle_files_match", all(
        _safe_name(p) and (case_dir / p).exists() and _sha256_file(case_dir / p) == h
        for p, h in bm["files"].items()), "a bundle-manifest digest mismatch")

    case_ok = all(v == "passed" for k, v in checks.items() if k.startswith(prefix))
    return case_ok, bm.get("bundle_root_sha256")


def verify_official_rollout_evidence(
    bundle_dir: str, *, require_complete_matrix: bool = True,
) -> VerifyEvidenceResult:
    root = Path(bundle_dir)
    checks: dict[str, str] = {}

    def ok(name, cond, reason=""):
        checks[name] = "passed" if cond else f"failed: {reason}"
        return cond

    if not ok("bundle_dir_exists", root.is_dir(), "missing"):
        return VerifyEvidenceResult(False, checks)

    case_roots: dict[str, str] = {}
    case_results: dict[str, bool] = {}
    seen = {p.name: p for p in sorted(root.iterdir()) if p.is_dir()}
    all_ok = True
    for name, cdir in seen.items():
        cok, broot = _verify_case(cdir, checks, f"case[{name}]")
        all_ok &= cok
        case_results[name] = cok
        if broot:
            case_roots[name] = broot

    if require_complete_matrix:
        for req in REQUIRED_CASES:
            all_ok &= ok(f"required_case:{req}", req in seen, "missing case")

    if (root / MATRIX_FILE).exists():
        try:
            mx = json.loads((root / MATRIX_FILE).read_text())
            jsonschema.validate(mx, MATRIX_SCHEMA)
        except Exception as exc:  # noqa: BLE001
            return VerifyEvidenceResult(False, {**checks, "matrix_schema": f"failed: {exc}"})
        all_ok &= ok("matrix_root",
                     canonical_json_sha256(sorted(
                         (c, v["bundle_root_sha256"]) for c, v in mx["cases"].items()))
                     == mx["matrix_root_sha256"], "matrix root mismatch")
        # v1.3.15: matrix entries must match INDEPENDENTLY verified case roots/results.
        all_ok &= ok("matrix_case_set", set(mx["cases"]) == set(case_roots),
                     "matrix cases != verified case bundles")
        for c, entry in mx["cases"].items():
            all_ok &= ok(f"matrix_binding:{c}",
                         case_roots.get(c) == entry["bundle_root_sha256"]
                         and case_results.get(c) == entry["passed"],
                         "matrix entry != verified case")
    elif require_complete_matrix:
        all_ok &= ok("matrix_present", False, f"{MATRIX_FILE} missing")

    return VerifyEvidenceResult(bool(all_ok), checks, case_roots)
