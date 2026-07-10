# rollout_evidence_schema.py — canonical schemas + hashing for rollout evidence (v1.3.14).
#
# Lives in the BASE package (lerobot-free) so the offline verifier CLI can validate
# a bundle without importing lerobot/torch. The plugin's evidence builder imports
# these same schemas — one source of truth.

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

READINESS_SCHEMA_VERSION = "lerobot-coreai.official_rollout_readiness.v3"
BUNDLE_MANIFEST_SCHEMA_VERSION = "lerobot-coreai.official_rollout_bundle.v1"
MATRIX_SCHEMA_VERSION = "lerobot-coreai.official_rollout_matrix.v1"
CANONICAL_HASH_ALGORITHM = "canonical-json-sha256.v1"

_SHA256 = {"type": "string", "pattern": r"^sha256:[0-9a-f]{64}$"}

# The closed, required check set (v1.3.14, P1.12).
REQUIRED_CHECKS = (
    "official_rollout_called", "all_environments_reached_done", "done_mask_cumulative",
    "done_mask_matches_terminate_at", "queue_refilled", "wire_payload_valid",
    "request_count_exact", "response_action_chain_valid",
    "fixture_feature_semantics_verified",
)
REQUIRED_CASES = ("single_only-b1", "native_batch-b2", "native_batch-b4",
                  "split_and_stack-b2", "split_and_stack-b4")


class CanonicalJSONError(ValueError):
    """Raised when a value is not canonical-JSON serialisable."""


def canonical_json_sha256(obj: Any) -> str:
    """sha256 over canonical JSON — JSON types only, no str() coercion (P1.11)."""
    def _check(v: Any):
        if v is None or isinstance(v, (str, bool)):
            return
        if isinstance(v, int):
            return
        if isinstance(v, float):
            if not math.isfinite(v):
                raise CanonicalJSONError(f"non-finite float {v!r} is not canonical.")
            return
        if isinstance(v, dict):
            for k, sub in v.items():
                if not isinstance(k, str):
                    raise CanonicalJSONError(f"non-string key {k!r}.")
                _check(sub)
            return
        if isinstance(v, (list, tuple)):
            for sub in v:
                _check(sub)
            return
        raise CanonicalJSONError(f"non-JSON value of type {type(v).__name__}.")
    _check(obj)
    canon = json.dumps(obj, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False)
    return "sha256:" + hashlib.sha256(canon.encode("utf-8")).hexdigest()


_ENVIRONMENT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["target", "lerobot_version", "lerobot_source", "python_version",
                 "torch_version", "numpy_version", "platform",
                 "lerobot_coreai_version", "companion_version"],
    "properties": {
        "target": {"enum": ["stable", "development", "local"]},
        "lerobot_version": {"type": ["string", "null"]},
        "lerobot_source": {"enum": ["pypi", "git", "unknown"]},
        "lerobot_commit": {"type": ["string", "null"]},
        "lerobot_distribution_sha256": {"type": ["string", "null"]},
        "python_version": {"type": "string"},
        "torch_version": {"type": ["string", "null"]},
        "numpy_version": {"type": ["string", "null"]},
        "platform": {"type": "string"},
        "lerobot_coreai_version": {"type": ["string", "null"]},
        "companion_version": {"type": ["string", "null"]},
        "repository_head_sha": {"type": ["string", "null"]},
        "workflow_run_id": {"type": ["string", "null"]},
        "workflow_job": {"type": ["string", "null"]},
    },
}

READINESS_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "hash_algorithm", "environment", "execution",
                 "contracts", "observation", "action", "checks", "claims"],
    "properties": {
        "schema_version": {"const": READINESS_SCHEMA_VERSION},
        "hash_algorithm": {"const": CANONICAL_HASH_ALGORITHM},
        "environment": _ENVIRONMENT_SCHEMA,
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
                         "postprocessor_sha256", "artifact_integrity_verified"],
            "properties": {
                "artifact_root_sha256": _SHA256, "batch_contract_sha256": _SHA256,
                "runner_capabilities_sha256": _SHA256, "preprocessor_sha256": _SHA256,
                "postprocessor_sha256": _SHA256,
                "artifact_integrity_verified": {"type": "boolean"}}},
        "observation": {
            "type": "object", "additionalProperties": False,
            "required": ["ordered_request_sha256s"],
            "properties": {
                "ordered_request_sha256s": {"type": "array", "items": _SHA256},
                "distinct_request_hashes": {"type": "boolean"}}},
        "action": {
            "type": "object", "additionalProperties": False,
            "required": ["ordered_response_sha256s", "final_action_sha256",
                         "done_mask_sha256"],
            "properties": {
                "ordered_response_sha256s": {"type": "array", "items": _SHA256},
                "final_action_sha256": _SHA256, "done_mask_sha256": _SHA256}},
        "checks": {
            "type": "object", "additionalProperties": False,
            "required": list(REQUIRED_CHECKS),
            "properties": {k: {"type": "boolean"} for k in REQUIRED_CHECKS}},
        "claims": {
            "type": "object", "additionalProperties": False,
            "required": ["official_rollout_pipeline_smoke_passed",
                         "official_eval_certified", "authenticity_verified",
                         "proves_task_success", "proves_physical_safety"],
            "properties": {
                "official_rollout_pipeline_smoke_passed": {"type": "boolean"},
                "official_eval_certified": {"const": False},
                "authenticity_verified": {"const": False},
                "proves_task_success": {"const": False},
                "proves_physical_safety": {"const": False}}},
    },
}

BUNDLE_MANIFEST_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "case", "files", "bundle_root_sha256"],
    "properties": {
        "schema_version": {"const": BUNDLE_MANIFEST_SCHEMA_VERSION},
        "case": {"type": "string"},
        "files": {"type": "object", "additionalProperties": _SHA256},
        "bundle_root_sha256": _SHA256,
    },
}

MEASUREMENTS_SCHEMA_VERSION = "lerobot-coreai.official_rollout_measurements.v1"
MEASUREMENTS_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "additionalProperties": False,
    "required": ["batch_size", "mode", "sequence_length", "horizon", "action_dim",
                 "terminate_at", "request_bodies", "response_bodies", "done_mask",
                 "final_action", "required_obs_keys", "fixture_contract"],
    "properties": {
        "batch_size": {"type": "integer", "minimum": 1},
        "mode": {"enum": ["single_only", "native_batch", "split_and_stack"]},
        "sequence_length": {"type": "integer", "minimum": 1},
        "horizon": {"type": "integer", "minimum": 1},
        "action_dim": {"type": "integer", "minimum": 1},
        "terminate_at": {"type": "array", "items": {"type": "integer", "minimum": 1}},
        "request_bodies": {"type": "array", "items": {"type": "object"}},
        "response_bodies": {"type": "array", "items": {"type": "object"}},
        "done_mask": {"type": "array",
                      "items": {"type": "array", "items": {"type": "integer"}}},
        "final_action": {"type": "array"},
        "required_obs_keys": {"type": "array", "items": {"type": "string"}},
        "fixture_contract": {"type": "object"},
    },
}

TRACE_EVENT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "additionalProperties": False,
    "required": ["index", "request_sha256", "response_sha256"],
    "properties": {
        "index": {"type": "integer", "minimum": 0},
        "request_sha256": _SHA256, "response_sha256": _SHA256,
    },
}

MATRIX_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "target", "cases", "matrix_root_sha256"],
    "properties": {
        "schema_version": {"const": MATRIX_SCHEMA_VERSION},
        "target": {"type": "string"},
        "cases": {
            "type": "object",
            "additionalProperties": {
                "type": "object", "additionalProperties": False,
                "required": ["passed", "bundle_root_sha256"],
                "properties": {"passed": {"type": "boolean"},
                               "bundle_root_sha256": _SHA256}}},
        "matrix_root_sha256": _SHA256,
    },
}
