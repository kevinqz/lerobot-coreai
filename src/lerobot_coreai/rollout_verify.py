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
    EXECUTION_ENVELOPE_SCHEMA,
    FAILURE_BUNDLE_MANIFEST_SCHEMA,
    FAILURE_REPORT_SCHEMA,
    MATRIX_SCHEMA,
    READINESS_SCHEMA,
    REQUIRED_CASES,
    TERMINAL_FAILURE_EVENTS,
    TRACE_EVENT_SCHEMA,
    _ENVIRONMENT_SCHEMA,
    canonical_json_sha256,
    capabilities_sha256_from_raw,
)
from .rollout_replay import replay_rollout_evidence

FAILURE_REPORT_FILE = "failure_report.json"
FAILURE_MANIFEST_FILE = "failure_bundle_manifest.json"
CAPABILITIES_RAW_FILE = "runner_capabilities_raw.json"
CAPABILITIES_NORM_FILE = "runner_capabilities_normalized.json"
_FAILURE_CONTENT = ("failure_report.json", "execution_envelope.json",
                    "environment_identity.json", "partial_trace.jsonl")

REPORT_FILE = "official_rollout_readiness_report.json"
BUNDLE_MANIFEST_FILE = "bundle_manifest.json"
CHECKSUMS_FILE = "checksums.json"
MEASUREMENTS_FILE = "measurements.json"
TRACE_FILE = "official_rollout_trace.jsonl"
README_MD = "official_rollout_readiness_report.md"
MATRIX_FILE = "official_rollout_matrix_manifest.json"

# content files (checksums must cover exactly these + the bundle manifest).
_CONTENT = (REPORT_FILE, README_MD, TRACE_FILE, MEASUREMENTS_FILE,
            CAPABILITIES_RAW_FILE, CAPABILITIES_NORM_FILE)
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


def _verify_failure_case(case_dir: Path, checks: dict, prefix: str,
                         evidence_grade: str = "diagnostic",
                         ) -> tuple[bool, bool, str | None]:
    """Verify a FailureEvidence v2 bundle (v1.3.19): schema-valid, all-claims-false,
    manifest/checksum-recomputable — independently, like a success bundle.

    v1.3.22 (P1.8): in ``certificate`` grade the terminal event must have been emitted
    by the runtime at the failing boundary — a writer-synthesized or post-hoc terminal
    is diagnostic-only."""
    def ok(name, cond, reason=""):
        checks[f"{prefix}:{name}"] = "passed" if cond else f"failed: {reason}"
        return cond

    if not ok("no_symlinks", not any(p.is_symlink() for p in case_dir.iterdir()),
              "symlink present"):
        return False, False, None
    actual = {p.name for p in case_dir.iterdir()}
    expected = set(_FAILURE_CONTENT) | {FAILURE_MANIFEST_FILE, CHECKSUMS_FILE}
    if not ok("exact_file_coverage", actual == expected,
              f"unexpected: {sorted(actual ^ expected)}"):
        return False, False, None
    try:
        recorded = json.loads((case_dir / CHECKSUMS_FILE).read_text())
    except Exception as exc:  # noqa: BLE001
        ok("checksums_parse", False, str(exc)); return False, False, None
    ok("checksums_paths_safe", all(_safe_name(k) for k in recorded), "unsafe path")
    # v1.3.20 (P1.7): exact checksum coverage, like the success bundle.
    ok("checksums_exact_coverage",
       set(recorded) == set(_FAILURE_CONTENT) | {FAILURE_MANIFEST_FILE},
       f"{sorted(recorded)}")
    ok("checksums_match", all(
        (case_dir / n).exists() and _sha256_file(case_dir / n) == d
        for n, d in recorded.items()), "a listed file was modified")
    try:
        report = json.loads((case_dir / FAILURE_REPORT_FILE).read_text())
        jsonschema.validate(report, FAILURE_REPORT_SCHEMA)
        envelope = json.loads((case_dir / "execution_envelope.json").read_text())
        jsonschema.validate(envelope, EXECUTION_ENVELOPE_SCHEMA)
        identity = json.loads((case_dir / "environment_identity.json").read_text())
        jsonschema.validate(identity, _ENVIRONMENT_SCHEMA)   # P1.5
        ok("schemas", True)
    except Exception as exc:  # noqa: BLE001
        ok("schemas", False, str(exc)); return False, False, None
    # every claim must be false (schema pins them, but re-check defensively).
    ok("no_forbidden_claims", all(v is False for v in report["claims"].values()),
       "a claim is true")
    ok("envelope_status", envelope["status"] in ("failed", "aborted"),
       "envelope status not terminal-failure")
    if evidence_grade == "certificate":
        ok("certificate_runtime_terminal",
           report.get("terminal_event_origin") == "runtime_exception_boundary",
           f"terminal_event_origin={report.get('terminal_event_origin')} "
           "is not certificate-grade (needs runtime_exception_boundary)")
    ok("no_secrets", _scan_secret(report) is None and _scan_secret(envelope) is None
       and _scan_secret(identity) is None, "secret detected")
    # partial trace: each event schema-valid, and it must END with a terminal
    # failure event whose execution_id + failed_stage match the report/envelope (P1.8).
    try:
        from .rollout_evidence_schema import EXECUTION_EVENT_SCHEMA
        events = [json.loads(ln) for ln in
                  (case_dir / "partial_trace.jsonl").read_text().splitlines() if ln.strip()]
        for e in events:
            jsonschema.validate(e, EXECUTION_EVENT_SCHEMA)
        ok("partial_trace_schema", True)
    except Exception as exc:  # noqa: BLE001
        ok("partial_trace_schema", False, str(exc)); events = []
    term = events[-1] if events else {}
    ok("failure_trace_terminal", bool(events)
       and term.get("event") in TERMINAL_FAILURE_EVENTS,
       "partial trace does not end with a terminal failure event")
    ok("failure_cross_binding",
       bool(events)
       and term.get("execution_id") == envelope["execution_id"] == report.get("execution_id")
       and term.get("failed_stage") == report["failed_stage"],
       "execution id / failed_stage not cross-bound across trace/report/envelope")
    try:
        bm = json.loads((case_dir / FAILURE_MANIFEST_FILE).read_text())
        jsonschema.validate(bm, FAILURE_BUNDLE_MANIFEST_SCHEMA)   # P1.6
        ok("failure_manifest_schema", True)
    except Exception as exc:  # noqa: BLE001
        ok("failure_manifest_schema", False, str(exc)); return False, False, None
    ok("failure_manifest_coverage", set(bm.get("files", {})) == set(_FAILURE_CONTENT),
       "manifest does not cover exactly the failure content")
    root = bm.get("bundle_root_sha256")
    ok("failure_root", canonical_json_sha256(sorted(bm.get("files", {}).items())) == root,
       "failure bundle root mismatch")
    ok("failure_files_match", all(
        _safe_name(p) and (case_dir / p).exists() and _sha256_file(case_dir / p) == h
        for p, h in bm.get("files", {}).items()), "a failure-bundle digest mismatch")
    case_ok = all(v == "passed" for k, v in checks.items() if k.startswith(prefix))
    return case_ok, False, root       # a failure case never counts as rollout-passed


def _verify_negotiation(case_dir: Path, checks: dict, prefix: str,
                        require_negotiation: bool) -> None:
    """Re-prove the NegotiationRecord OFFLINE (v1.3.20): capabilities hash recomputed
    from the persisted file, and the negotiation algorithm RE-RUN from the record's
    inputs must reproduce the persisted protocol/encoding (P1.2, P1.3)."""
    from .negotiation_algorithm import NegotiationError, expected_negotiation

    def ok(name, cond, reason=""):
        checks[f"{prefix}:{name}"] = "passed" if cond else f"failed: {reason}"
        return cond

    try:
        raw = json.loads((case_dir / MEASUREMENTS_FILE).read_text())
    except Exception as exc:  # noqa: BLE001
        ok("negotiation_load", False, str(exc)); return
    from .capabilities_normalize import (
        CapabilitiesNormalizationError, NORMALIZED_CAPABILITIES_SCHEMA,
        normalize_capabilities, normalized_capabilities_sha256,
    )
    neg = raw.get("negotiation")
    if neg is None:
        # certificate-grade evidence must carry a NegotiationRecord (P1.1).
        ok("negotiation_present", not require_negotiation,
           "no NegotiationRecord (legacy unnegotiated evidence)")
        return
    # raw capabilities hash recomputed from the persisted raw payload (audit).
    try:
        caps_raw = json.loads((case_dir / CAPABILITIES_RAW_FILE).read_text())
        ok("capabilities_raw_hash",
           capabilities_sha256_from_raw(caps_raw) == neg.get("runner_capabilities_sha256"),
           "raw capabilities != NegotiationRecord.runner_capabilities_sha256")
    except Exception as exc:  # noqa: BLE001
        ok("capabilities_raw_hash", False, str(exc)); return
    # v1.3.21 (P1.4/P1.5): re-run normalization OFFLINE from the raw payload; it must
    # equal both the persisted normalized file and the record's normalized hash, and
    # the normalized file must be schema-valid.
    try:
        norm_persisted = json.loads((case_dir / CAPABILITIES_NORM_FILE).read_text())
        jsonschema.validate(norm_persisted, NORMALIZED_CAPABILITIES_SCHEMA)
        recomputed = normalize_capabilities(caps_raw)
        ok("capabilities_normalized",
           recomputed == norm_persisted
           and normalized_capabilities_sha256(caps_raw)
           == neg.get("normalized_capabilities_sha256"),
           "normalized capabilities mismatch (offline re-run != persisted)")
    except (CapabilitiesNormalizationError, Exception) as exc:  # noqa: BLE001
        ok("capabilities_normalized", False, str(exc))
    # re-run the exact negotiation algorithm and require equality (P1.2).
    try:
        proto, enc = expected_negotiation(neg)
        ok("negotiation_recomputed",
           proto == neg["negotiated_protocol"] and enc == neg["negotiated_encoding"],
           "recomputed negotiation != persisted result")
    except (NegotiationError, KeyError) as exc:
        ok("negotiation_recomputed", False, f"invalid negotiation inputs: {exc}")


def _verify_case(case_dir: Path, checks: dict, prefix: str, *,
                 require_negotiation: bool = True,
                 evidence_grade: str = "diagnostic") -> tuple[bool, bool, str | None]:
    """Verify one case bundle. Returns (bundle_verified, rollout_passed, root)."""
    if (case_dir / FAILURE_REPORT_FILE).exists():
        return _verify_failure_case(case_dir, checks, prefix, evidence_grade)

    def ok(name, cond, reason=""):
        checks[f"{prefix}:{name}"] = "passed" if cond else f"failed: {reason}"
        return cond

    if not ok("no_symlinks", not any(p.is_symlink() for p in case_dir.iterdir()),
              "symlink present"):
        return False, False, None
    for fn in (REPORT_FILE, BUNDLE_MANIFEST_FILE, CHECKSUMS_FILE, MEASUREMENTS_FILE):
        if not ok(f"present:{fn}", (case_dir / fn).exists(), "missing"):
            return False, False, None

    # checksums: path-safe, exact coverage, no tamper.
    try:
        recorded = json.loads((case_dir / CHECKSUMS_FILE).read_text())
    except Exception as exc:  # noqa: BLE001
        ok("checksums_parse", False, str(exc)); return False, False, None
    if not ok("checksums_paths_safe", all(_safe_name(k) for k in recorded),
              "unsafe path in checksums"):
        return False, False, None
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
        ok("report_schema", False, str(exc)); return False, False, None
    ok("no_forbidden_claims", all(
        report["claims"].get(k) is not True for k in
        ("official_eval_certified", "authenticity_verified", "proves_task_success",
         "proves_physical_safety")), "a forbidden claim is true")
    ok("no_secrets", _scan_secret(report) is None
       and _scan_secret(json.loads((case_dir / MEASUREMENTS_FILE).read_text())) is None,
       "secret detected")

    # exact actual-file coverage (v1.3.16, P1.6): no extras/hidden/dirs.
    actual = {p.name for p in case_dir.iterdir()}
    expected_files = set(_CONTENT) | {BUNDLE_MANIFEST_FILE, CHECKSUMS_FILE}
    ok("exact_file_coverage", actual == expected_files,
       f"unexpected files: {sorted(actual ^ expected_files)}")

    # trace cross-binding (P1.2): trace hashes must equal report + recomputed.
    try:
        events = [json.loads(ln) for ln in
                  (case_dir / TRACE_FILE).read_text().splitlines() if ln.strip()]
        for ev in events:                          # apply the schema (P1.1)
            jsonschema.validate(ev, TRACE_EVENT_SCHEMA)
        raw = json.loads((case_dir / MEASUREMENTS_FILE).read_text())
        tr_req = [e["request_sha256"] for e in events]
        tr_resp = [e["response_sha256"] for e in events]
        idx_ok = [e["index"] for e in events] == list(range(len(events)))
        rep_req = report["observation"]["ordered_request_sha256s"]
        rep_resp = report["action"]["ordered_response_sha256s"]
        raw_req = [canonical_json_sha256(b) for b in raw["request_bodies"]]
        raw_resp = [canonical_json_sha256(b) for b in raw["response_bodies"]]
        ok("trace_cross_binding",
           idx_ok and tr_req == rep_req == raw_req
           and tr_resp == rep_resp == raw_resp,     # self-sufficient (P1.7)
           "trace does not match report/measurements")
    except Exception as exc:  # noqa: BLE001
        ok("trace_cross_binding", False, str(exc))

    # SEMANTIC replay (v1.3.15): re-derive every check from raw records.
    rep = replay_rollout_evidence(str(case_dir))
    ok("semantic_replay", rep.ok, "; ".join(rep.errors)[:200])

    # v1.3.20: re-prove negotiation (capabilities hash + recomputed algorithm).
    _verify_negotiation(case_dir, checks, prefix, require_negotiation)

    # bundle manifest schema + recomputed root + per-file digests.
    try:
        bm = json.loads((case_dir / BUNDLE_MANIFEST_FILE).read_text())
        jsonschema.validate(bm, BUNDLE_MANIFEST_SCHEMA)
    except Exception as exc:  # noqa: BLE001
        ok("bundle_manifest_schema", False, str(exc)); return False, False, None
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
    passed = bool(report["claims"].get("official_rollout_pipeline_smoke_passed"))
    return case_ok, passed, bm.get("bundle_root_sha256")


def verify_official_rollout_evidence(
    bundle_dir: str, *, require_complete_matrix: bool = True,
    require_negotiation: bool = True, evidence_grade: str = "diagnostic",
) -> VerifyEvidenceResult:
    # v1.3.22 (P1.8): certificate grade is strictly stronger than diagnostic —
    # negotiation + a runtime-boundary failure terminal + the RuntimeSupportProfile
    # are all mandatory.
    if evidence_grade == "certificate":
        require_negotiation = True
    root = Path(bundle_dir)
    checks: dict[str, str] = {}

    def ok(name, cond, reason=""):
        checks[name] = "passed" if cond else f"failed: {reason}"
        return cond

    if not ok("bundle_dir_exists", root.is_dir(), "missing"):
        return VerifyEvidenceResult(False, checks)

    case_roots: dict[str, str] = {}
    case_passed: dict[str, bool] = {}
    seen = {p.name: p for p in sorted(root.iterdir()) if p.is_dir()}
    all_ok = True
    for name, cdir in seen.items():
        cok, passed, broot = _verify_case(
            cdir, checks, f"case[{name}]", require_negotiation=require_negotiation,
            evidence_grade=evidence_grade)
        all_ok &= cok
        case_passed[name] = passed
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
        # v1.3.16: root binds target + passed + bundle root (a target/pass flip
        # changes the root). v1.3.23 (P1.7): the RuntimeSupportProfile hash is folded
        # in too — recompute it from the on-disk profile and require it to match.
        profile_sha = mx.get("runtime_support_profile_sha256")
        rsp = root / "runtime_support_profile.json"
        if profile_sha is not None:
            all_ok &= ok("runtime_support_profile_bound",
                         rsp.exists()
                         and canonical_json_sha256(json.loads(rsp.read_text())) == profile_sha,
                         "runtime_support_profile.json != matrix profile hash")
        recomputed = canonical_json_sha256(
            {"schema_version": mx["schema_version"], "target": mx["target"],
             **({"runtime_support_profile_sha256": profile_sha}
                if profile_sha is not None else {}),
             "cases": sorted((c, bool(v["passed"]), v["bundle_root_sha256"])
                             for c, v in mx["cases"].items())})
        all_ok &= ok("matrix_root", recomputed == mx["matrix_root_sha256"],
                     "matrix root mismatch")
        all_ok &= ok("matrix_case_set", set(mx["cases"]) == set(case_roots),
                     "matrix cases != verified case bundles")
        for c, entry in mx["cases"].items():
            all_ok &= ok(f"matrix_binding:{c}",
                         case_roots.get(c) == entry["bundle_root_sha256"]
                         and case_passed.get(c) == entry["passed"],
                         "matrix entry != verified case")
        # matrix.target must equal every case's target (success: report environment;
        # failure: execution envelope) — v1.3.19 supports mixed matrices.
        target_ok = True
        for c in mx["cases"]:
            try:
                cdir = root / c
                if (cdir / FAILURE_REPORT_FILE).exists():
                    env_t = json.loads(
                        (cdir / "execution_envelope.json").read_text())["target"]
                else:
                    env_t = json.loads(
                        (cdir / REPORT_FILE).read_text())["environment"]["target"]
                target_ok &= (env_t == mx["target"])
            except Exception:  # noqa: BLE001
                target_ok = False
        all_ok &= ok("matrix_target_binding", target_ok,
                     "matrix.target != a case target")
    elif require_complete_matrix:
        all_ok &= ok("matrix_present", False, f"{MATRIX_FILE} missing")

    # v1.3.21 (P1.14): the RuntimeSupportProfile, if present, must be schema-valid and
    # equal the canonical profile (a tamper adding native `shared` / split `global`
    # fails), separating the universal contract language from the supported subset.
    rsp = root / "runtime_support_profile.json"
    if rsp.exists():
        from .runtime_support import RUNTIME_SUPPORT_SCHEMA, runtime_support_profile
        try:
            prof = json.loads(rsp.read_text())
            jsonschema.validate(prof, RUNTIME_SUPPORT_SCHEMA)
            all_ok &= ok("runtime_support_profile", prof == runtime_support_profile(),
                         "profile != canonical runtime support profile")
        except Exception as exc:  # noqa: BLE001
            all_ok &= ok("runtime_support_profile", False, str(exc))
    elif evidence_grade == "certificate":
        # a certificate-grade matrix MUST declare the runtime support profile (P1.10).
        all_ok &= ok("runtime_support_profile", False,
                     "runtime_support_profile.json missing (required for certificate)")

    return VerifyEvidenceResult(bool(all_ok), checks, case_roots)
