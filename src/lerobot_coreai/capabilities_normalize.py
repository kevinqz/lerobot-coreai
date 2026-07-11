# capabilities_normalize.py — canonical RunnerCapabilities normalization (v1.3.21).
#
# The runner's announced capabilities arrive as a loose JSON payload; two semantically
# equivalent payloads can differ byte-for-byte (missing defaults, list order). v1.3.20
# hashed only the RAW payload (P1.4). This module derives a canonical NORMALIZED form
# — explicit defaults, sorted unordered lists, duplicates rejected — and hashes THAT.
# Decisions/certificates bind the normalized hash; raw stays for audit. Pure Python.

from __future__ import annotations

from .rollout_evidence_schema import _HASH_OR, _NE_STR, canonical_json_sha256

NORMALIZE_ALGORITHM_VERSION = "coreai-capabilities-normalize.v1"

RAW_CAPABILITIES_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",   # the announced payload is loose by nature; keep it permissive
}

NORMALIZED_CAPABILITIES_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object", "additionalProperties": False,
    "required": ["normalization_algorithm_version", "runtime", "protocol_version",
                 "backward_compatible_with", "observation_encodings",
                 "supports_action", "action_batching", "inference_state"],
    "properties": {
        "normalization_algorithm_version": {"const": NORMALIZE_ALGORITHM_VERSION},
        "runtime": {"type": ["string", "null"]},
        "protocol_version": {"type": ["string", "null"]},
        "backward_compatible_with": {"type": "array", "items": {"type": "string"},
                                     "uniqueItems": True},
        "observation_encodings": {"type": "array", "items": {"type": "string"},
                                  "uniqueItems": True},
        "supports_action": {"type": "boolean"},
        "action_batching": {
            "type": "object", "additionalProperties": False,
            "required": ["supported", "max_batch_size", "semantics", "slot_isolation"],
            "properties": {"supported": {"type": "boolean"},
                           "max_batch_size": {"type": "integer", "minimum": 1},
                           "semantics": {"type": ["string", "null"]},
                           "slot_isolation": {"type": ["string", "null"]}}},
        "inference_state": {
            "type": "object", "additionalProperties": False,
            "required": ["scope", "supports_session_ids", "reset_scope"],
            "properties": {"scope": {"type": ["string", "null"]},
                           "supports_session_ids": {"type": "boolean"},
                           "reset_scope": {"type": ["string", "null"]}}},
    },
}


class CapabilitiesNormalizationError(ValueError):
    """Raised when a raw capabilities payload cannot be canonically normalized."""


def _sorted_unique(values, field: str) -> list:
    seq = list(values or [])
    if len(seq) != len(set(seq)):
        raise CapabilitiesNormalizationError(f"duplicate value in {field!r}: {seq}.")
    if not all(isinstance(v, str) for v in seq):
        raise CapabilitiesNormalizationError(f"{field!r} must be list[str].")
    return sorted(seq)


def normalize_capabilities(raw: dict) -> dict:
    """Canonicalize an announced capabilities payload, fail-closed on duplicates."""
    if not isinstance(raw, dict):
        raise CapabilitiesNormalizationError("capabilities payload must be an object.")
    ab = raw.get("action_batching") or {}
    if not isinstance(ab, dict):
        raise CapabilitiesNormalizationError("action_batching must be an object.")
    ist = raw.get("inference_state") or {}
    if not isinstance(ist, dict):
        raise CapabilitiesNormalizationError("inference_state must be an object.")
    supports = raw.get("supports") or {}
    return {
        "normalization_algorithm_version": NORMALIZE_ALGORITHM_VERSION,
        "runtime": raw.get("runtime"),
        "protocol_version": raw.get("protocol_version"),
        "backward_compatible_with": _sorted_unique(
            raw.get("backward_compatible_with"), "backward_compatible_with"),
        "observation_encodings": _sorted_unique(
            raw.get("observation_encodings"), "observation_encodings"),
        "supports_action": bool(supports.get("action", False)),
        "action_batching": {
            "supported": bool(ab.get("supported", False)),
            "max_batch_size": int(ab.get("max_batch_size", 1) or 1),
            "semantics": ab.get("semantics"),
            "slot_isolation": ab.get("slot_isolation")},
        "inference_state": {
            "scope": ist.get("scope"),
            "supports_session_ids": bool(ist.get("supports_session_ids", False)),
            "reset_scope": ist.get("reset_scope")},
    }


def normalized_capabilities_sha256(raw: dict) -> str:
    """Canonical hash of the NORMALIZED capabilities (the certificate-grade id)."""
    return canonical_json_sha256(normalize_capabilities(raw))


# re-exported for schema consumers that only want the hash pattern.
__all__ = ["NORMALIZE_ALGORITHM_VERSION", "RAW_CAPABILITIES_SCHEMA",
           "NORMALIZED_CAPABILITIES_SCHEMA", "CapabilitiesNormalizationError",
           "normalize_capabilities", "normalized_capabilities_sha256", "_HASH_OR",
           "_NE_STR"]
