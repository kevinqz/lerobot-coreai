# rollout_evidence.py — real, verifiable official-rollout evidence (v1.3.14).
#
# v1.3.13 built a hash-bound report; v1.3.14 makes it SELF-SUFFICIENT and
# INDEPENDENTLY VERIFIABLE: exact environment identity, request AND response/action
# hashes, canonical (JSON-only) hashing, derived fixture-semantics, a per-case bundle
# manifest with a recomputable root, an aggregate matrix manifest, and failure
# evidence. Schemas live in the base package (lerobot_coreai.rollout_evidence_schema)
# so the offline verifier needs no lerobot. Still NOT a certificate.

from __future__ import annotations

import json
import math
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jsonschema

from lerobot_coreai.rollout_evidence_schema import (
    BUNDLE_MANIFEST_SCHEMA_VERSION,
    CANONICAL_HASH_ALGORITHM,
    MATRIX_SCHEMA_VERSION,
    READINESS_SCHEMA,
    READINESS_SCHEMA_VERSION,
    REQUIRED_CHECKS,
    canonical_json_sha256,
)

from . import __version__ as PLUGIN_VERSION


class EvidenceBindingError(RuntimeError):
    """Raised when certificate-grade evidence cannot be bound to real inputs."""


def _pkg_version(name: str) -> str | None:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:  # noqa: BLE001
        return None


def capture_environment_identity(target: str) -> dict[str, Any]:
    """Exact execution environment (P1.1) — no generic placeholders.

    v1.3.19 EnvironmentIdentity v2: records the separate provenance SHAs a PR merge
    conflates (source-branch head vs merge commit vs base) plus run attempt and
    runner image, so a rerun/merge is distinguishable. The stable wheel-distribution
    digest is still deferred (v1.3.20).
    """
    import os
    return {
        "target": target,
        "lerobot_version": _pkg_version("lerobot"),
        "lerobot_source": os.environ.get("COREAI_ROLLOUT_LEROBOT_SOURCE", "unknown"),
        "lerobot_commit": os.environ.get("LEROBOT_DEV_REF"),
        "lerobot_distribution_sha256": None,
        "python_version": platform.python_version(),
        "torch_version": _pkg_version("torch"),
        "numpy_version": _pkg_version("numpy"),
        "platform": platform.platform(),
        "lerobot_coreai_version": _pkg_version("lerobot-coreai"),
        "companion_version": PLUGIN_VERSION,
        "repository_head_sha": os.environ.get("GITHUB_SHA"),
        "workflow_run_id": os.environ.get("GITHUB_RUN_ID"),
        "workflow_job": os.environ.get("GITHUB_JOB"),
        # v1.3.20 (P1.12): the workflow resolves these EXACTLY via dedicated env vars
        # — GITHUB_SHA is a synthetic merge commit on PRs, and GITHUB_BASE_REF is a
        # branch name, not a SHA — so we prefer the explicit COREAI_* values.
        "source_head_sha": os.environ.get("COREAI_SOURCE_HEAD_SHA")
        or os.environ.get("GITHUB_SHA"),
        "base_sha": os.environ.get("COREAI_BASE_SHA"),
        "merge_sha": os.environ.get("COREAI_MERGE_SHA") or os.environ.get("GITHUB_SHA"),
        "workflow_sha": os.environ.get("COREAI_WORKFLOW_SHA"),
        "workflow_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "runner_image": os.environ.get("ImageOS") or os.environ.get("RUNNER_OS"),
    }


# MARK: - measurements + evaluation

@dataclass(frozen=True)
class RolloutMeasurements:
    batch_size: int
    mode: str
    sequence_length: int
    horizon: int
    action_dim: int
    terminate_at: tuple[int, ...]
    request_bodies: tuple[dict, ...]
    response_bodies: tuple[dict, ...]
    done_mask: tuple[tuple[int, ...], ...]
    final_action: Any                       # nested list [B, seq, A]
    required_obs_keys: tuple[str, ...] = ()
    fixture_contract: dict = field(default_factory=dict)   # {key: expected per-sample shape}
    queue_events: tuple[dict, ...] = ()
    negotiation: dict | None = None                        # persisted NegotiationRecord
    runner_capabilities: dict | None = None                # normalized announced caps

    def to_raw(self) -> dict:
        """Canonical, replayable raw record (persisted as measurements.json)."""
        raw = {
            "batch_size": self.batch_size, "mode": self.mode,
            "sequence_length": self.sequence_length, "horizon": self.horizon,
            "action_dim": self.action_dim, "terminate_at": list(self.terminate_at),
            "request_bodies": list(self.request_bodies),
            "response_bodies": list(self.response_bodies),
            "done_mask": [list(r) for r in self.done_mask],
            "final_action": self.final_action,
            "required_obs_keys": list(self.required_obs_keys),
            "fixture_contract": dict(self.fixture_contract),
            "queue_events": list(self.queue_events),
        }
        if self.negotiation is not None:
            raw["negotiation"] = dict(self.negotiation)
        return raw


@dataclass(frozen=True)
class RolloutEvaluation:
    passed: bool
    checks: dict[str, bool]
    request_count: int
    ordered_request_sha256s: tuple[str, ...]
    ordered_response_sha256s: tuple[str, ...]
    errors: tuple[str, ...] = ()
    failed_stage: str | None = None


def evaluate_rollout_measurements(m: RolloutMeasurements) -> RolloutEvaluation:
    """Derive checks via the SAME base engine the offline verifier replays with."""
    from lerobot_coreai.rollout_replay import derive_checks
    raw = m.to_raw()
    checks = derive_checks(raw)
    req_hashes = tuple(canonical_json_sha256(b) for b in m.request_bodies)
    resp_hashes = tuple(canonical_json_sha256(b) for b in m.response_bodies)
    failed = next((stage for stage, key in (
        ("done_mask_validation", "done_mask_matches_terminate_at"),
        ("request_accounting", "request_count_exact"),
        ("response_chain", "response_action_chain_valid"),
        ("wire_validation", "wire_payload_valid"),
        ("fixture_semantics", "fixture_feature_semantics_verified"))
        if not checks[key]), None)
    return RolloutEvaluation(
        passed=all(checks.values()), checks=checks, request_count=len(m.request_bodies),
        ordered_request_sha256s=req_hashes, ordered_response_sha256s=resp_hashes,
        errors=() if failed is None else (f"failed_stage={failed}",),
        failed_stage=failed)


# MARK: - report + bundle

def build_rollout_readiness_report(
    evaluation: RolloutEvaluation, measurements: RolloutMeasurements, *,
    environment: dict, artifact_root_sha256: str, batch_contract_sha256: str,
    runner_capabilities_sha256: str, preprocessor_sha256: str,
    postprocessor_sha256: str, artifact_integrity_verified: bool,
) -> dict[str, Any]:
    import re
    _re = re.compile(r"^sha256:[0-9a-f]{64}$")
    for name, h in (("artifact_root", artifact_root_sha256),
                    ("batch_contract", batch_contract_sha256),
                    ("runner_capabilities", runner_capabilities_sha256),
                    ("preprocessor", preprocessor_sha256),
                    ("postprocessor", postprocessor_sha256)):
        if not (isinstance(h, str) and _re.match(h)):
            raise EvidenceBindingError(f"{name}_sha256 must be a real sha256; got {h!r}.")
    req = evaluation.ordered_request_sha256s
    # v1.3.19 execution envelope: pull the id + terminal accounting from the sealed
    # event stream, and bind the persisted NegotiationRecord hash into the report.
    events = list(measurements.queue_events)
    exec_id = events[0]["execution_id"] if events else None
    last = events[-1] if events else {}
    neg_sha = measurements.negotiation["record_sha256"] if measurements.negotiation else None
    report = {
        "schema_version": READINESS_SCHEMA_VERSION,
        "hash_algorithm": CANONICAL_HASH_ALGORITHM,
        "environment": environment,
        "execution": {
            "batch_size": measurements.batch_size, "mode": measurements.mode,
            "sequence_length": measurements.sequence_length,
            "horizon": measurements.horizon, "request_count": evaluation.request_count,
            "failed_stage": evaluation.failed_stage, "errors": list(evaluation.errors),
            "execution_id": exec_id,
            "status": "completed" if evaluation.passed else "failed",
            "termination_reason": last.get("termination_reason"),
            "unused_action_count": last.get("unused_action_count", 0),
            "negotiation_sha256": neg_sha},
        "contracts": {
            "artifact_root_sha256": artifact_root_sha256,
            "batch_contract_sha256": batch_contract_sha256,
            "runner_capabilities_sha256": runner_capabilities_sha256,
            "preprocessor_sha256": preprocessor_sha256,
            "postprocessor_sha256": postprocessor_sha256,
            "artifact_integrity_verified": bool(artifact_integrity_verified)},
        "observation": {
            "ordered_request_sha256s": list(req),
            "distinct_request_hashes": len(set(req)) == len(req) and len(req) > 1},
        "action": {
            "ordered_response_sha256s": list(evaluation.ordered_response_sha256s),
            "final_action_sha256": canonical_json_sha256(measurements.final_action),
            "done_mask_sha256": canonical_json_sha256(
                [list(r) for r in measurements.done_mask])},
        "checks": {k: bool(v) for k, v in evaluation.checks.items()},
        "claims": {
            "official_rollout_pipeline_smoke_passed": bool(evaluation.passed),
            "official_eval_certified": False, "authenticity_verified": False,
            "proves_task_success": False, "proves_physical_safety": False},
    }
    jsonschema.validate(report, READINESS_SCHEMA)
    return report


def render_readiness_md(report: dict) -> str:
    lines = ["# Official Rollout Readiness Report", "",
             f"target: {report['environment']['target']} · "
             f"lerobot {report['environment']['lerobot_version']}", "", "## Checks", ""]
    for k, v in report["checks"].items():
        lines.append(f"- {'✓' if v else '✗'} {k}: {v}")
    lines += ["", "## Claims", ""]
    for k, v in report["claims"].items():
        lines.append(f"- **{k}**: {v}")
    lines += ["", "_Pipeline smoke only — not lerobot-eval certification._"]
    return "\n".join(lines) + "\n"


def _sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def write_evidence_bundle(out_dir: str, report: dict, measurements: RolloutMeasurements) -> str:
    """Persist a per-case bundle (report/md/jsonl/manifest/checksums); return root."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "official_rollout_readiness_report.json").write_text(json.dumps(report, indent=2))
    (out / "official_rollout_readiness_report.md").write_text(render_readiness_md(report))
    # Raw, canonical, REPLAYABLE record (v1.3.15) — the offline verifier re-derives
    # every semantic check from this, never trusting the report's own booleans.
    (out / "measurements.json").write_text(json.dumps(measurements.to_raw(), indent=2))
    with open(out / "official_rollout_trace.jsonl", "w") as fh:
        for j, (rq, rs) in enumerate(zip(measurements.request_bodies,
                                         measurements.response_bodies)):
            fh.write(json.dumps({"index": j, "request_sha256": canonical_json_sha256(rq),
                                 "response_sha256": canonical_json_sha256(rs)}) + "\n")
    # v1.3.20 (P1.3): persist the normalized announced capabilities so the verifier
    # can recompute the hash bound in the NegotiationRecord OFFLINE.
    (out / "runner_capabilities.json").write_text(
        json.dumps(measurements.runner_capabilities or {}, indent=2, sort_keys=True))
    content = ["official_rollout_readiness_report.json",
               "official_rollout_readiness_report.md", "official_rollout_trace.jsonl",
               "measurements.json", "runner_capabilities.json"]
    files = {f: _sha256_file(out / f) for f in content}
    bundle_root = canonical_json_sha256(sorted(files.items()))
    manifest = {"schema_version": BUNDLE_MANIFEST_SCHEMA_VERSION,
                "case": out.name, "files": files, "bundle_root_sha256": bundle_root}
    (out / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2))
    # checksums cover the content files AND the bundle manifest.
    checks = dict(files)
    checks["bundle_manifest.json"] = _sha256_file(out / "bundle_manifest.json")
    (out / "checksums.json").write_text(json.dumps(checks, indent=2))
    return bundle_root


def write_failure_evidence(
    out_dir: str, *, case: str, failed_stage: str, exception_type: str, message: str,
    target: str = "local", batch_size: int = 1, mode: str = "single_only",
    execution_id: str | None = None, environment: dict | None = None,
    partial_events: tuple[dict, ...] = (), status: str = "failed",
    negotiation_sha256: str | None = None,
) -> str:
    """Write a schema-valid, independently verifiable FailureEvidence v2 bundle.

    v1.3.18 wrote a single schema-free record; v1.3.19 (P1.13) writes a full bundle —
    failure_report.json (stage-typed, all claims false), execution_envelope.json,
    environment_identity.json, partial_trace.jsonl — with a manifest + checksums the
    offline verifier can re-prove exactly like a success bundle. Returns the root.
    """
    from lerobot_coreai.rollout_evidence_schema import (
        EXECUTION_ENVELOPE_SCHEMA, EXECUTION_ENVELOPE_SCHEMA_VERSION,
        FAILURE_BUNDLE_MANIFEST_SCHEMA_VERSION, FAILURE_REPORT_SCHEMA,
        FAILURE_REPORT_SCHEMA_VERSION,
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    exec_id = execution_id or f"failed-{case}"     # shared across report + envelope + trace
    report = {
        "schema_version": FAILURE_REPORT_SCHEMA_VERSION, "case": case,
        "target": target, "failed_stage": failed_stage,
        "exception_type": exception_type, "message": message[:2000],
        "execution_id": exec_id,
        "claims": {"official_rollout_pipeline_smoke_passed": False,
                   "official_eval_certified": False, "authenticity_verified": False,
                   "proves_task_success": False, "proves_physical_safety": False}}
    jsonschema.validate(report, FAILURE_REPORT_SCHEMA)
    envelope = {
        "schema_version": EXECUTION_ENVELOPE_SCHEMA_VERSION,
        "execution_id": exec_id, "case": case,
        "target": target, "mode": mode, "batch_size": batch_size,
        "status": status if status in ("failed", "aborted") else "failed",
        "negotiation_sha256": negotiation_sha256, "termination_reason": failed_stage}
    jsonschema.validate(envelope, EXECUTION_ENVELOPE_SCHEMA)

    (out / "failure_report.json").write_text(json.dumps(report, indent=2))
    (out / "execution_envelope.json").write_text(json.dumps(envelope, indent=2))
    (out / "environment_identity.json").write_text(
        json.dumps(environment or capture_environment_identity(target), indent=2))
    # v1.3.20 (P1.8): the partial trace must END with a terminal failure event that
    # names the same failed_stage + execution_id as the report/envelope, so the
    # causal trace demonstrably terminated at that failure.
    events = list(partial_events)
    exec_id = envelope["execution_id"]
    terminal = "execution.aborted" if status == "aborted" else "execution.failed"
    if not events or events[-1].get("event") not in ("execution.failed",
                                                      "execution.aborted"):
        events.append({
            "event_index": events[-1]["event_index"] + 1 if events else 0,
            "event": terminal, "execution_id": exec_id,
            "relative_monotonic_ns": (events[-1].get("relative_monotonic_ns", 0) + 1
                                      if events else 0),
            "failed_stage": failed_stage})
    with open(out / "partial_trace.jsonl", "w") as fh:
        for e in events:
            fh.write(json.dumps(e) + "\n")
    content = ["failure_report.json", "execution_envelope.json",
               "environment_identity.json", "partial_trace.jsonl"]
    files = {f: _sha256_file(out / f) for f in content}
    root = canonical_json_sha256(sorted(files.items()))
    manifest = {"schema_version": FAILURE_BUNDLE_MANIFEST_SCHEMA_VERSION,
                "case": case, "files": files, "bundle_root_sha256": root}
    (out / "failure_bundle_manifest.json").write_text(json.dumps(manifest, indent=2))
    checks = dict(files)
    checks["failure_bundle_manifest.json"] = _sha256_file(out / "failure_bundle_manifest.json")
    (out / "checksums.json").write_text(json.dumps(checks, indent=2))
    return root


def write_matrix_manifest(matrix_dir: str, target: str, cases: dict[str, dict]) -> dict:
    """Aggregate per-case {passed, bundle_root_sha256} into a matrix + root."""
    out = Path(matrix_dir)
    out.mkdir(parents=True, exist_ok=True)
    # v1.3.16 (P1.1): the matrix root binds target + passed + bundle root, so a
    # target/pass flip changes the root (not just the bundle-root list).
    root = canonical_json_sha256(
        {"schema_version": MATRIX_SCHEMA_VERSION, "target": target,
         "cases": sorted((c, bool(v["passed"]), v["bundle_root_sha256"])
                         for c, v in cases.items())})
    mx = {"schema_version": MATRIX_SCHEMA_VERSION, "target": target,
          "cases": cases, "matrix_root_sha256": root}
    (out / "official_rollout_matrix_manifest.json").write_text(json.dumps(mx, indent=2))
    (out / "matrix_checksums.json").write_text(json.dumps(
        {"official_rollout_matrix_manifest.json":
         _sha256_file(out / "official_rollout_matrix_manifest.json")}, indent=2))
    return mx
