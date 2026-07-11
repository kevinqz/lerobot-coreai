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

# The closed, required check set (v1.3.14; v1.3.17 replaces the queue_refilled proxy
# with event-derived queue_lifecycle_valid + queue_refill_count_exact).
REQUIRED_CHECKS = (
    "official_rollout_called", "all_environments_reached_done", "done_mask_cumulative",
    "done_mask_matches_terminate_at", "queue_lifecycle_valid",
    "queue_refill_count_exact", "wire_payload_valid", "request_count_exact",
    "response_action_chain_valid", "fixture_feature_semantics_verified",
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
        # v1.3.19 EnvironmentIdentity v2: separate the provenance SHAs a PR merge
        # conflates (source branch head vs merge commit vs base), plus run attempt
        # and runner image, so a rerun/merge is distinguishable in evidence.
        "source_head_sha": {"type": ["string", "null"]},
        "base_sha": {"type": ["string", "null"]},
        "merge_sha": {"type": ["string", "null"]},
        "workflow_sha": {"type": ["string", "null"]},
        "workflow_run_attempt": {"type": ["string", "null"]},
        "runner_image": {"type": ["string", "null"]},
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
                "errors": {"type": "array", "items": {"type": "string"}},
                # v1.3.19 execution envelope (optional; present for real rollouts).
                "execution_id": {"type": ["string", "null"]},
                "status": {"enum": ["completed", "failed", "aborted"]},
                "termination_reason": {"type": ["string", "null"]},
                "unused_action_count": {"type": ["integer", "null"], "minimum": 0},
                "negotiation_sha256": {"anyOf": [_SHA256, {"type": "null"}]}}},
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
                      "items": {"type": "array", "items": {"enum": [0, 1]}}},
        "final_action": {"type": "array"},
        "required_obs_keys": {"type": "array", "items": {"type": "string"},
                              "uniqueItems": True},
        "fixture_contract": {"type": "object"},
        "queue_events": {"type": "array", "items": {"type": "object"}},
        # v1.3.19: the persisted NegotiationRecord (bound into wire validation).
        "negotiation": {"type": "object"},
    },
    # single_only must be B=1 (P1.5).
    "if": {"properties": {"mode": {"const": "single_only"}}},
    "then": {"properties": {"batch_size": {"const": 1}}},
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

# v1.3.19: Runtime Evidence Protocol v3 — a DISCRIMINATED trace-event schema.
# v1.3.18's QUEUE_EVENT_SCHEMA was closed on properties but every field was optional
# for every event, so {"event":"execution.started","chunk_sha256":null} was
# structurally valid (P1.1). The v3 schema is a oneOf keyed on `event` where each
# branch declares exactly the fields that event must and may carry.
EXECUTION_EVENT_SCHEMA_VERSION = "lerobot-coreai.execution_trace.v3"
QUEUE_EVENT_TYPES = (
    "execution.started", "execution.completed", "policy.reset", "queue.empty",
    "queue.refill_requested", "runner.request_started", "runner.response_received",
    "chunk.validated", "chunk.committed", "action.popped", "queue.exhausted",
)
# v1.3.20: discriminated TERMINAL failure events for the failure path (P1.8). A
# failure bundle's partial trace must end with one of the terminal events below.
FAILURE_EVENT_TYPES = (
    "execution.failed", "execution.aborted", "negotiation.failed",
    "runner.request_failed", "runner.response_invalid", "chunk.assembly_failed",
    "chunk.validation_failed", "chunk.commit_failed", "bundle.write_failed",
    "offline.verify_failed",
)
TERMINAL_FAILURE_EVENTS = ("execution.failed", "execution.aborted")

# reusable field schemas
_NON_NEG = {"type": "integer", "minimum": 0}
_POS = {"type": "integer", "minimum": 1}
_NE_STR = {"type": "string", "minLength": 1}
_HASH_OR = {"type": "string", "pattern": r"^sha256:[0-9a-f]{64}$"}
_COMMON_REQ = ["event_index", "event", "execution_id", "relative_monotonic_ns"]
_COMMON_PROPS = {
    "event_index": _NON_NEG, "event": {"type": "string"},
    "execution_id": _NE_STR, "relative_monotonic_ns": _NON_NEG,
}

# per-event: (required-beyond-common, {optional-prop: schema})
_EVENT_SPEC: dict[str, tuple[dict, list]] = {
    "execution.started": ({}, []),
    # v1.3.20 (P1.11): completion must ALWAYS declare its terminal accounting.
    "execution.completed": ({"termination_reason": _NE_STR,
                             "unused_action_count": _NON_NEG,
                             "unused_action_sha256s": {"type": "array",
                                                       "items": _HASH_OR}},
                            ["termination_reason", "unused_action_count",
                             "unused_action_sha256s"]),
    "policy.reset": ({"reset_kind": {"enum": ["normal", "abort"]},
                      "queue_size_after": _NON_NEG,
                      "discarded_action_count": _NON_NEG,
                      "discarded_queue_sha256": _HASH_OR},
                     ["reset_kind", "queue_size_after"]),
    "queue.empty": ({"queue_size_after": _NON_NEG}, ["queue_size_after"]),
    "queue.refill_requested": ({"prediction_id": _NON_NEG, "chunk_id": _NE_STR,
                                "queue_size_before": _NON_NEG,
                                "queue_size_after": _NON_NEG},
                               ["prediction_id", "chunk_id",
                                "queue_size_before", "queue_size_after"]),
    "runner.request_started": ({"prediction_id": _NON_NEG, "chunk_id": _NE_STR,
                                "request_id": _NE_STR,
                                "sample_index": {"type": ["integer", "null"],
                                                 "minimum": 0}},
                               ["prediction_id", "chunk_id", "request_id"]),
    "runner.response_received": ({"prediction_id": _NON_NEG, "chunk_id": _NE_STR,
                                  "request_id": _NE_STR, "response_sha256": _HASH_OR,
                                  "sample_index": {"type": ["integer", "null"],
                                                   "minimum": 0}},
                                 ["prediction_id", "chunk_id", "request_id",
                                  "response_sha256"]),
    "chunk.validated": ({"prediction_id": _NON_NEG, "chunk_id": _NE_STR,
                         "chunk_sha256": _HASH_OR, "horizon": _POS,
                         "ordered_response_sha256s": {"type": "array",
                                                      "items": _HASH_OR}},
                        ["prediction_id", "chunk_id", "chunk_sha256", "horizon",
                         "ordered_response_sha256s"]),
    "chunk.committed": ({"prediction_id": _NON_NEG, "chunk_id": _NE_STR,
                         "chunk_sha256": _HASH_OR, "committed": _POS,
                         "queue_size_before": _NON_NEG,
                         "queue_size_after": _NON_NEG},
                        ["prediction_id", "chunk_id", "chunk_sha256", "committed",
                         "queue_size_before", "queue_size_after"]),
    "action.popped": ({"prediction_id": _NON_NEG, "chunk_id": _NE_STR,
                       "action_id": _NE_STR, "rollout_step": _NON_NEG,
                       "chunk_timestep": _NON_NEG,
                       "selected_action_sha256": _HASH_OR,
                       "queue_size_before": _NON_NEG, "queue_size_after": _NON_NEG},
                      ["prediction_id", "chunk_id", "action_id", "rollout_step",
                       "chunk_timestep", "selected_action_sha256",
                       "queue_size_before", "queue_size_after"]),
    "queue.exhausted": ({"queue_size_after": _NON_NEG}, ["queue_size_after"]),
}
# Terminal / stage failure events all carry an optional failed_stage + detail; the
# two execution-level terminals require a failed_stage so the trace is self-describing.
for _fe in FAILURE_EVENT_TYPES:
    _req = ["failed_stage"] if _fe in TERMINAL_FAILURE_EVENTS else []
    _EVENT_SPEC[_fe] = ({"failed_stage": _NE_STR, "detail": {"type": "string"},
                         "prediction_id": _NON_NEG, "chunk_id": _NE_STR,
                         "request_id": _NE_STR}, _req)
ALL_EVENT_TYPES = QUEUE_EVENT_TYPES + FAILURE_EVENT_TYPES


def _event_branch(name: str) -> dict:
    extra_props, extra_req = _EVENT_SPEC[name]
    props = {**_COMMON_PROPS, **extra_props, "event": {"const": name}}
    return {"type": "object", "additionalProperties": False, "properties": props,
            "required": _COMMON_REQ + list(extra_req)}


EXECUTION_EVENT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "oneOf": [_event_branch(n) for n in ALL_EVENT_TYPES],
}
# Back-compat alias (the offline verifier + replay import the current name).
QUEUE_EVENT_SCHEMA = EXECUTION_EVENT_SCHEMA

EXECUTION_ENVELOPE_SCHEMA_VERSION = "lerobot-coreai.execution_envelope.v1"
EXECUTION_ENVELOPE_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object", "additionalProperties": False,
    "required": ["schema_version", "execution_id", "case", "target", "mode",
                 "batch_size", "status", "negotiation_sha256"],
    "properties": {
        "schema_version": {"const": EXECUTION_ENVELOPE_SCHEMA_VERSION},
        "execution_id": _NE_STR, "case": _NE_STR,
        "target": {"enum": ["stable", "development", "local"]},
        "mode": {"enum": ["single_only", "native_batch", "split_and_stack"]},
        "batch_size": _POS,
        "status": {"enum": ["completed", "failed", "aborted"]},
        "negotiation_sha256": {"anyOf": [_HASH_OR, {"type": "null"}]},
        "termination_reason": {"type": ["string", "null"]},
    },
}

# v1.3.20 NegotiationRecord v2: adds selection_policy + the runner's declared
# backward-compatibility list, so the offline verifier can RE-RUN the negotiation
# algorithm and reject a self-consistent-but-invalid record (P1.2, P1.4).
NEGOTIATION_SCHEMA_VERSION = "lerobot-coreai.negotiation_record.v2"
# v1.3.21 (P1.1): only minimum_compatible is implemented, so the schema pins it —
# a record can no longer declare a policy whose semantics the verifier ignores.
# exact / highest_compatible return when they have real use cases + tests.
NEGOTIATION_SELECTION_POLICIES = ("minimum_compatible",)
NEGOTIATION_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object", "additionalProperties": False,
    "required": ["schema_version", "selection_policy", "minimum_protocol",
                 "runner_protocol", "runner_backward_compatible_with",
                 "negotiated_protocol", "runner_encodings", "negotiated_encoding",
                 "runner_capabilities_sha256", "normalized_capabilities_sha256",
                 "record_sha256"],
    "properties": {
        "schema_version": {"const": NEGOTIATION_SCHEMA_VERSION},
        "selection_policy": {"enum": list(NEGOTIATION_SELECTION_POLICIES)},
        "requested_protocol": {"type": ["string", "null"]},
        "minimum_protocol": _NE_STR,
        "runner_protocol": _NE_STR,
        "runner_backward_compatible_with": {"type": "array",
                                            "items": {"type": "string"}},
        "negotiated_protocol": _NE_STR,
        "requested_encoding": {"type": ["string", "null"]},
        "runner_encodings": {"type": "array", "items": {"type": "string"}},
        "negotiated_encoding": _NE_STR,
        "runner_capabilities_sha256": _HASH_OR,          # raw payload (audit)
        "normalized_capabilities_sha256": _HASH_OR,      # canonical (certificate-grade)
        "record_sha256": _HASH_OR,
    },
}

FAILURE_STAGES = (
    "setup", "artifact_verify", "factory_load", "processor_load",
    "runner_negotiate", "request", "response", "chunk_assembly", "validation",
    "queue_commit", "rollout", "measurement_build", "bundle_write",
    "semantic_replay", "offline_verify",
)
FAILURE_REPORT_SCHEMA_VERSION = "lerobot-coreai.official_rollout_failure.v2"
FAILURE_REPORT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object", "additionalProperties": False,
    "required": ["schema_version", "case", "target", "failed_stage",
                 "exception_type", "message", "terminal_event_origin", "claims"],
    "properties": {
        "schema_version": {"const": FAILURE_REPORT_SCHEMA_VERSION},
        "case": _NE_STR, "target": {"enum": ["stable", "development", "local"]},
        "failed_stage": {"enum": list(FAILURE_STAGES)},
        "exception_type": _NE_STR,
        "message": {"type": "string"},
        "execution_id": {"type": ["string", "null"]},
        # v1.3.22 (P1.6/P1.7/L): the PRECISE origin of the terminal event.
        #  - runtime_exception_boundary: emitted automatically at the failing boundary
        #    (certificate-grade — the runtime classified the stage itself);
        #  - runtime_api_posthoc: emitted by an explicit abort call after a caught
        #    exception (the caller chose the stage — diagnostic);
        #  - writer_synthesized: no runtime session at all (diagnostic).
        "terminal_event_origin": {"enum": ["runtime_exception_boundary",
                                           "runtime_api_posthoc", "writer_synthesized"]},
        "claims": {
            "type": "object", "additionalProperties": False,
            "required": ["official_rollout_pipeline_smoke_passed",
                         "official_eval_certified", "authenticity_verified",
                         "proves_task_success", "proves_physical_safety"],
            "properties": {
                "official_rollout_pipeline_smoke_passed": {"const": False},
                "official_eval_certified": {"const": False},
                "authenticity_verified": {"const": False},
                "proves_task_success": {"const": False},
                "proves_physical_safety": {"const": False}}},
    },
}
FAILURE_BUNDLE_MANIFEST_SCHEMA_VERSION = "lerobot-coreai.official_rollout_failure_bundle.v2"
FAILURE_BUNDLE_MANIFEST_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object", "additionalProperties": False,
    "required": ["schema_version", "case", "files", "bundle_root_sha256"],
    "properties": {
        "schema_version": {"const": FAILURE_BUNDLE_MANIFEST_SCHEMA_VERSION},
        "case": _NE_STR,
        "files": {"type": "object", "additionalProperties": _HASH_OR},
        "bundle_root_sha256": _HASH_OR,
    },
}


def capabilities_sha256_from_raw(raw: dict) -> str:
    """The runner-capabilities fingerprint, recomputable OFFLINE from the persisted
    normalized capabilities object (mirrors runner.capabilities_sha256)."""
    canon = json.dumps(raw or {}, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canon.encode()).hexdigest()

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
        # v1.3.23 (P1.7): the RuntimeSupportProfile hash is folded into the matrix
        # root, so removing or tampering with the profile changes the root.
        "runtime_support_profile_sha256": _SHA256,
        "matrix_root_sha256": _SHA256,
    },
}
