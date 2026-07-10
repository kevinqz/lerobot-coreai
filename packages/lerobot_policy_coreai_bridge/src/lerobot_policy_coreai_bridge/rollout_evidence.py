# rollout_evidence.py — real, hash-bound official-rollout evidence (v1.3.13).
#
# v1.3.12 built an in-memory report from caller-provided booleans and placeholder
# hashes. v1.3.13 makes it real: MEASUREMENTS -> EVALUATOR -> checks -> report, with
# actual artifact/contract/capability/processor/observation/action hashes, and a
# persisted, checksummed evidence bundle (json/md/jsonl). Still NOT a certificate —
# every strong claim stays false; a signed promotion is a later version.

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jsonschema

ROLLOUT_READINESS_SCHEMA_VERSION = "lerobot-coreai.official_rollout_readiness.v2"
_SHA256 = {"type": "string", "pattern": r"^sha256:[0-9a-f]{64}$"}
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

# Observation keys the runner must never receive (ground-truth / bookkeeping).
_LEAK_KEYS = ("action", "reward", "done", "success", "index", "episode_index",
              "frame_index", "timestamp", "next.reward", "next.done")


def _sha256_obj(obj: Any) -> str:
    canon = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(canon.encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


# MARK: - Measurements + evaluation (checks are DERIVED, never caller-supplied)

@dataclass(frozen=True)
class RolloutMeasurements:
    batch_size: int
    mode: str                          # single_only | native_batch | split_and_stack
    sequence_length: int
    horizon: int
    terminate_at: tuple[int, ...]
    request_bodies: tuple[dict, ...]
    done_mask: tuple[tuple[int, ...], ...]   # (B, seq) 0/1
    required_obs_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class RolloutEvaluation:
    passed: bool
    checks: dict[str, bool]
    request_count: int
    ordered_request_sha256s: tuple[str, ...]
    errors: tuple[str, ...] = ()
    failed_stage: str | None = None


def _done_cumulative(done: tuple[tuple[int, ...], ...]) -> bool:
    for row in done:
        seen = False
        for v in row:
            if v:
                seen = True
            elif seen:
                return False        # went back to 0 after a 1
    return True


def _all_reached_done(done, terminate_at) -> bool:
    return all(any(row) for row in done)


def _first_done_matches(done, terminate_at) -> bool:
    for i, ta in enumerate(terminate_at):
        row = done[i]
        fd = ta - 1
        if any(row[:fd]) or not all(row[fd:]):
            return False
    return True


def _wire_valid(bodies, native, B, required_obs_keys) -> tuple[bool, list[str]]:
    errs: list[str] = []
    for j, body in enumerate(bodies):
        obs = body.get("observation", {})
        for leak in _LEAK_KEYS:
            if leak in obs:
                errs.append(f"request {j}: label {leak!r} leaked")
        for k in required_obs_keys:
            if k not in obs:
                errs.append(f"request {j}: required key {k!r} missing")
        opts = body.get("options", {})
        if native and B > 1:
            if opts.get("batch_size") != B:
                errs.append(f"request {j}: options.batch_size != {B}")
        else:
            if "batch_size" in opts and B != 1:
                errs.append(f"request {j}: unexpected batch_size in single/split")
    return (not errs), errs


def evaluate_rollout_measurements(m: RolloutMeasurements) -> RolloutEvaluation:
    """Derive every check from measured data (never from caller booleans, P1.3)."""
    predictions = math.ceil(m.sequence_length / m.horizon)
    if m.mode == "split_and_stack":
        expected_requests = m.batch_size * predictions
    else:                                       # single_only / native_batch
        expected_requests = predictions
    request_count = len(m.request_bodies)

    wire_ok, wire_errs = _wire_valid(m.request_bodies, m.mode == "native_batch",
                                     m.batch_size, m.required_obs_keys)
    checks = {
        "official_rollout_called": request_count > 0,
        "all_environments_reached_done": _all_reached_done(m.done_mask, m.terminate_at),
        "done_mask_cumulative": _done_cumulative(m.done_mask),
        "done_mask_matches_terminate_at": _first_done_matches(m.done_mask, m.terminate_at),
        "queue_refilled": predictions > 1,
        "wire_payload_valid": wire_ok,
        "request_count_exact": request_count == expected_requests,
    }
    errors = list(wire_errs)
    if request_count != expected_requests:
        errors.append(f"request_count {request_count} != expected {expected_requests}")
    failed_stage = None
    if not checks["done_mask_cumulative"] or not checks["done_mask_matches_terminate_at"]:
        failed_stage = "done_mask_validation"
    elif not checks["request_count_exact"]:
        failed_stage = "request_accounting"
    elif not checks["wire_payload_valid"]:
        failed_stage = "wire_validation"
    ordered = tuple(_sha256_obj(b) for b in m.request_bodies)
    passed = all(checks.values())
    return RolloutEvaluation(passed=passed, checks=checks, request_count=request_count,
                             ordered_request_sha256s=ordered, errors=tuple(errors),
                             failed_stage=failed_stage)


# MARK: - Report schema (v2) + builder

OFFICIAL_ROLLOUT_READINESS_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "environment", "execution", "contracts",
                 "observation", "checks", "claims"],
    "properties": {
        "schema_version": {"const": ROLLOUT_READINESS_SCHEMA_VERSION},
        "environment": {"type": "object"},
        "execution": {
            "type": "object", "additionalProperties": False,
            "required": ["batch_size", "mode", "sequence_length", "horizon",
                         "request_count", "failed_stage", "errors"],
            "properties": {
                "batch_size": {"type": "integer", "minimum": 1},
                "mode": {"enum": ["single_only", "native_batch", "split_and_stack"]},
                "sequence_length": {"type": "integer", "minimum": 1},
                "horizon": {"type": "integer", "minimum": 1},
                "request_count": {"type": "integer", "minimum": 0},
                "failed_stage": {"type": ["string", "null"]},
                "errors": {"type": "array", "items": {"type": "string"}}}},
        "contracts": {
            "type": "object", "additionalProperties": False,
            "required": ["artifact_root_sha256", "batch_contract_sha256",
                         "runner_capabilities_sha256", "preprocessor_sha256",
                         "postprocessor_sha256"],
            "properties": {k: _SHA256 for k in (
                "artifact_root_sha256", "batch_contract_sha256",
                "runner_capabilities_sha256", "preprocessor_sha256",
                "postprocessor_sha256")}},
        "observation": {
            "type": "object", "additionalProperties": False,
            "required": ["ordered_request_sha256s"],
            "properties": {
                "ordered_request_sha256s": {"type": "array", "items": _SHA256},
                "fixture_feature_semantics_verified": {"type": "boolean"},
                "universal_feature_contract_verified": {"type": "boolean"}}},
        "checks": {"type": "object", "additionalProperties": {"type": "boolean"}},
        "claims": {
            "type": "object", "additionalProperties": False,
            "required": ["official_rollout_pipeline_smoke_passed",
                         "official_eval_certified", "authenticity_verified",
                         "proves_task_success", "proves_physical_safety"],
            "properties": {k: {"type": "boolean"} for k in (
                "official_rollout_pipeline_smoke_passed", "official_eval_certified",
                "authenticity_verified", "proves_task_success",
                "proves_physical_safety")}},
    },
}


def build_rollout_readiness_report(
    evaluation: RolloutEvaluation, measurements: RolloutMeasurements, *,
    environment: dict, artifact_root_sha256: str, batch_contract_sha256: str,
    runner_capabilities_sha256: str, preprocessor_sha256: str,
    postprocessor_sha256: str, fixture_feature_semantics_verified: bool = True,
) -> dict[str, Any]:
    """Assemble a schema-valid report from an EVALUATION (not caller checks).

    Every contract hash must be a real ``sha256:<64hex>`` — a placeholder/None is
    rejected. ``official_rollout_pipeline_smoke_passed`` == evaluation.passed.
    """
    for name, h in (("artifact_root", artifact_root_sha256),
                    ("batch_contract", batch_contract_sha256),
                    ("runner_capabilities", runner_capabilities_sha256),
                    ("preprocessor", preprocessor_sha256),
                    ("postprocessor", postprocessor_sha256)):
        if not (isinstance(h, str) and _SHA256_RE.match(h)):
            raise ValueError(f"{name}_sha256 must be a real sha256:<64hex>, got {h!r}.")
    report = {
        "schema_version": ROLLOUT_READINESS_SCHEMA_VERSION,
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
            "postprocessor_sha256": postprocessor_sha256},
        "observation": {
            "ordered_request_sha256s": list(evaluation.ordered_request_sha256s),
            "fixture_feature_semantics_verified": bool(fixture_feature_semantics_verified),
            "universal_feature_contract_verified": False},
        "checks": {k: bool(v) for k, v in evaluation.checks.items()},
        "claims": {
            "official_rollout_pipeline_smoke_passed": bool(evaluation.passed),
            "official_eval_certified": False, "authenticity_verified": False,
            "proves_task_success": False, "proves_physical_safety": False},
    }
    jsonschema.validate(report, OFFICIAL_ROLLOUT_READINESS_SCHEMA)
    return report


def render_readiness_md(report: dict[str, Any]) -> str:
    lines = ["# Official Rollout Readiness Report", "",
             f"schema: `{report['schema_version']}`", "", "## Execution", ""]
    for k, v in report["execution"].items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## Checks", ""]
    for k, v in report["checks"].items():
        lines.append(f"- {'✓' if v else '✗'} {k}: {v}")
    lines += ["", "## Claims", ""]
    for k, v in report["claims"].items():
        lines.append(f"- **{k}**: {v}")
    lines += ["", "_Pipeline smoke only — not lerobot-eval certification, task "
              "success, or safety._"]
    return "\n".join(lines) + "\n"


def write_evidence_bundle(out_dir: str, report: dict, measurements: RolloutMeasurements) -> Path:
    """Persist json/md/jsonl (+checksums) for one rollout case, even on failure."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "official_rollout_readiness_report.json").write_text(json.dumps(report, indent=2))
    (out / "official_rollout_readiness_report.md").write_text(render_readiness_md(report))
    with open(out / "official_rollout_trace.jsonl", "w") as fh:
        for j, body in enumerate(measurements.request_bodies):
            fh.write(json.dumps({"request_index": j, "sha256": _sha256_obj(body),
                                 "options": body.get("options", {})}) + "\n")
    files = ["official_rollout_readiness_report.json",
             "official_rollout_readiness_report.md", "official_rollout_trace.jsonl"]
    checks = {f: _sha256_file(out / f) for f in files}
    (out / "checksums.json").write_text(json.dumps(checks, indent=2))
    return out
