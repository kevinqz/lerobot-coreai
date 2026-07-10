# rollout_evidence.py — official-rollout readiness evidence (v1.3.12).
#
# Produces an unsigned, schema-valid readiness report from a real lerobot_eval
# rollout run: what ran, how many requests, done-mask/queue checks, and the
# contract/capability hashes it was bound to. NOT a certificate — every strong
# claim stays false; a signed promotion is a later version.

from __future__ import annotations

import json
from typing import Any

import jsonschema

ROLLOUT_READINESS_SCHEMA_VERSION = "lerobot-coreai.official_rollout_readiness.v1"

OFFICIAL_ROLLOUT_READINESS_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "lerobot", "execution", "contracts", "checks",
                 "claims"],
    "properties": {
        "schema_version": {"const": ROLLOUT_READINESS_SCHEMA_VERSION},
        "lerobot": {
            "type": "object", "additionalProperties": False,
            "required": ["version"],
            "properties": {"version": {"type": ["string", "null"]},
                           "commit": {"type": ["string", "null"]}}},
        "execution": {
            "type": "object", "additionalProperties": False,
            "required": ["batch_size", "mode", "sequence_length", "horizon",
                         "request_count"],
            "properties": {
                "batch_size": {"type": "integer", "minimum": 1},
                "mode": {"enum": ["single_only", "native_batch", "split_and_stack"]},
                "sequence_length": {"type": "integer", "minimum": 1},
                "horizon": {"type": "integer", "minimum": 1},
                "request_count": {"type": "integer", "minimum": 0}}},
        "contracts": {
            "type": "object", "additionalProperties": False,
            "required": ["artifact_root_sha256", "batch_contract_sha256",
                         "runner_capabilities_sha256"],
            "properties": {
                "artifact_root_sha256": {"type": ["string", "null"]},
                "batch_contract_sha256": {"type": ["string", "null"]},
                "runner_capabilities_sha256": {"type": ["string", "null"]}}},
        "checks": {
            "type": "object", "additionalProperties": False,
            "required": ["official_rollout_called", "all_environments_reached_done",
                         "done_mask_cumulative", "queue_refilled",
                         "wire_payload_valid", "request_count_exact"],
            "properties": {k: {"type": "boolean"} for k in (
                "official_rollout_called", "all_environments_reached_done",
                "done_mask_cumulative", "queue_refilled", "wire_payload_valid",
                "request_count_exact")}},
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
    *, lerobot_version, lerobot_commit, batch_size, mode, sequence_length, horizon,
    request_count, artifact_root_sha256, batch_contract_sha256,
    runner_capabilities_sha256, checks: dict[str, bool],
) -> dict[str, Any]:
    """Assemble + schema-validate an official-rollout readiness report.

    ``official_rollout_pipeline_smoke_passed`` is True only when every check passed;
    all other claims stay False (this is a pipeline smoke, not a certificate).
    """
    smoke = all(bool(v) for v in checks.values())
    report = {
        "schema_version": ROLLOUT_READINESS_SCHEMA_VERSION,
        "lerobot": {"version": lerobot_version, "commit": lerobot_commit},
        "execution": {"batch_size": int(batch_size), "mode": mode,
                      "sequence_length": int(sequence_length), "horizon": int(horizon),
                      "request_count": int(request_count)},
        "contracts": {"artifact_root_sha256": artifact_root_sha256,
                      "batch_contract_sha256": batch_contract_sha256,
                      "runner_capabilities_sha256": runner_capabilities_sha256},
        "checks": {k: bool(v) for k, v in checks.items()},
        "claims": {
            "official_rollout_pipeline_smoke_passed": smoke,
            "official_eval_certified": False, "authenticity_verified": False,
            "proves_task_success": False, "proves_physical_safety": False},
    }
    jsonschema.validate(report, OFFICIAL_ROLLOUT_READINESS_SCHEMA)
    return report


def render_readiness_md(report: dict[str, Any]) -> str:
    lines = ["# Official Rollout Readiness Report", "",
             f"LeRobot: {report['lerobot']['version']} "
             f"({report['lerobot'].get('commit')})", "", "## Execution", ""]
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
