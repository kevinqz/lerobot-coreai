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
    """Exact execution environment (P1.1) — no generic placeholders."""
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
    }


# MARK: - measurements + evaluation

@dataclass(frozen=True)
class RolloutMeasurements:
    batch_size: int
    mode: str
    sequence_length: int
    horizon: int
    terminate_at: tuple[int, ...]
    request_bodies: tuple[dict, ...]
    response_bodies: tuple[dict, ...]
    done_mask: tuple[tuple[int, ...], ...]
    final_action: Any                       # nested list [B, seq, A]
    required_obs_keys: tuple[str, ...] = ()
    fixture_contract: dict = field(default_factory=dict)   # {key: expected per-sample shape}


@dataclass(frozen=True)
class RolloutEvaluation:
    passed: bool
    checks: dict[str, bool]
    request_count: int
    ordered_request_sha256s: tuple[str, ...]
    ordered_response_sha256s: tuple[str, ...]
    errors: tuple[str, ...] = ()
    failed_stage: str | None = None


def _done_cumulative(done) -> bool:
    for row in done:
        seen = False
        for v in row:
            if v:
                seen = True
            elif seen:
                return False
    return True


def _first_done_matches(done, terminate_at) -> bool:
    for i, ta in enumerate(terminate_at):
        row, fd = done[i], ta - 1
        if any(row[:fd]) or not all(row[fd:]):
            return False
    return True


def _per_sample_shape(v):
    if isinstance(v, list):
        s = [len(v)]
        cur = v
        while cur and isinstance(cur[0], list):
            s.append(len(cur[0]))
            cur = cur[0]
        return s
    return []


def _fixture_semantics_ok(bodies, native, B, fixture) -> bool:
    if not fixture:
        return False
    for body in bodies:
        obs = body.get("observation", {})
        for key, per_sample in fixture.items():
            if key not in obs:
                return False
            shape = _per_sample_shape(obs[key])
            expected = ([B] + list(per_sample)) if (native and B > 1) else list(per_sample)
            if shape != expected:
                return False
    return True


def _response_chain_ok(responses, requests, native, B, seq, horizon, A) -> bool:
    if len(responses) != len(requests) or not responses:
        return False
    for r in responses:
        act = r.get("action")
        shape = _per_sample_shape(act)
        if native and B > 1:
            if len(shape) != 3 or shape[0] != B or shape[-1] != A:
                return False
        else:
            if len(shape) != 2 or shape[-1] != A:      # [H, A]
                return False
    return True


def evaluate_rollout_measurements(m: RolloutMeasurements, *, action_dim: int) -> RolloutEvaluation:
    predictions = math.ceil(m.sequence_length / m.horizon)
    native = m.mode == "native_batch"
    expected_requests = (m.batch_size * predictions) if m.mode == "split_and_stack" \
        else predictions
    req_count = len(m.request_bodies)
    req_hashes = tuple(canonical_json_sha256(b) for b in m.request_bodies)
    resp_hashes = tuple(canonical_json_sha256(b) for b in m.response_bodies)

    checks = {
        "official_rollout_called": req_count > 0,
        "all_environments_reached_done": all(any(r) for r in m.done_mask),
        "done_mask_cumulative": _done_cumulative(m.done_mask),
        "done_mask_matches_terminate_at": _first_done_matches(m.done_mask, m.terminate_at),
        "queue_refilled": predictions > 1,
        "wire_payload_valid": all(
            k in b.get("observation", {}) for b in m.request_bodies
            for k in m.required_obs_keys) and bool(m.request_bodies),
        "request_count_exact": req_count == expected_requests,
        "response_action_chain_valid": _response_chain_ok(
            m.response_bodies, m.request_bodies, native, m.batch_size,
            m.sequence_length, m.horizon, action_dim),
        "fixture_feature_semantics_verified": _fixture_semantics_ok(
            m.request_bodies, native, m.batch_size, m.fixture_contract),
    }
    assert set(checks) == set(REQUIRED_CHECKS), "check set drifted from schema"
    errors = []
    if req_count != expected_requests:
        errors.append(f"request_count {req_count} != {expected_requests}")
    failed = None
    for stage, key in (("done_mask_validation", "done_mask_matches_terminate_at"),
                       ("request_accounting", "request_count_exact"),
                       ("response_chain", "response_action_chain_valid"),
                       ("fixture_semantics", "fixture_feature_semantics_verified")):
        if not checks[key]:
            failed = stage
            break
    return RolloutEvaluation(
        passed=all(checks.values()), checks=checks, request_count=req_count,
        ordered_request_sha256s=req_hashes, ordered_response_sha256s=resp_hashes,
        errors=tuple(errors), failed_stage=failed)


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
    report = {
        "schema_version": READINESS_SCHEMA_VERSION,
        "hash_algorithm": CANONICAL_HASH_ALGORITHM,
        "environment": environment,
        "execution": {
            "batch_size": measurements.batch_size, "mode": measurements.mode,
            "sequence_length": measurements.sequence_length,
            "horizon": measurements.horizon, "request_count": evaluation.request_count,
            "failed_stage": evaluation.failed_stage, "errors": list(evaluation.errors)},
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
    with open(out / "official_rollout_trace.jsonl", "w") as fh:
        for j, (rq, rs) in enumerate(zip(measurements.request_bodies,
                                         measurements.response_bodies)):
            fh.write(json.dumps({"index": j, "request_sha256": canonical_json_sha256(rq),
                                 "response_sha256": canonical_json_sha256(rs)}) + "\n")
    content = ["official_rollout_readiness_report.json",
               "official_rollout_readiness_report.md", "official_rollout_trace.jsonl"]
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


def write_failure_evidence(out_dir: str, *, case: str, failed_stage: str,
                           exception_type: str, message: str) -> None:
    """Write a minimal, schema-free failure record for any stage exception (P1.7)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rec = {"case": case, "passed": False, "failed_stage": failed_stage,
           "exception_type": exception_type, "message": message[:2000],
           "claims": {"official_rollout_pipeline_smoke_passed": False,
                      "official_eval_certified": False}}
    (out / "failure_evidence.json").write_text(json.dumps(rec, indent=2))


def write_matrix_manifest(matrix_dir: str, target: str, cases: dict[str, dict]) -> dict:
    """Aggregate per-case {passed, bundle_root_sha256} into a matrix + root."""
    out = Path(matrix_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = canonical_json_sha256(
        sorted((c, v["bundle_root_sha256"]) for c, v in cases.items()))
    mx = {"schema_version": MATRIX_SCHEMA_VERSION, "target": target,
          "cases": cases, "matrix_root_sha256": root}
    (out / "official_rollout_matrix_manifest.json").write_text(json.dumps(mx, indent=2))
    (out / "matrix_checksums.json").write_text(json.dumps(
        {"official_rollout_matrix_manifest.json":
         _sha256_file(out / "official_rollout_matrix_manifest.json")}, indent=2))
    return mx
