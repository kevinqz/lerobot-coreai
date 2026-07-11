# capabilities_normalize.py — canonical RunnerCapabilities normalization (v1.3.22).
#
# The runner's announced capabilities arrive as a loose JSON payload; two semantically
# equivalent payloads can differ byte-for-byte (missing defaults, list order). This
# module validates the RAW payload against a strict contract, then derives a canonical
# NORMALIZED form — explicit defaults, sorted unordered lists, duplicates rejected —
# and hashes THAT. Decisions/certificates bind the normalized hash; raw stays for audit.
#
# v1.3.22 (P1.2/P1.3/P1.4): normalization is now FAIL-CLOSED. It NEVER coerces — a
# string where a bool/int is required, or an unknown enum, RAISES instead of being
# silently promoted (the old bool("false") is True / int("4") == 4 traps). The raw
# payload is validated against a strict schema before normalization runs. Pure Python.

from __future__ import annotations

from dataclasses import dataclass

from .rollout_evidence_schema import _HASH_OR, _NE_STR, canonical_json_sha256

NORMALIZE_ALGORITHM_VERSION = "coreai-capabilities-normalize.v2"

# closed enums for the semantic fields (None = not announced, allowed).
_SEMANTICS = ("native", "split_and_stack")
_SLOT_ISO = ("independent", "shared", "unknown")
_STATE_SCOPE = ("stateless", "request_scoped", "session_scoped", "global")
_RESET_SCOPE = ("none", "session", "global")

_BOOL = {"type": "boolean"}
_POS_INT = {"type": "integer", "minimum": 1}
RAW_CAPABILITIES_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    # strict on KNOWN fields (types + enums); a runner may still announce extra keys.
    "type": "object",
    "properties": {
        "runtime": {"type": ["string", "null"]},
        "protocol_version": {"type": ["string", "null"]},
        "backward_compatible_with": {"type": "array", "items": {"type": "string"}},
        "observation_encodings": {"type": "array", "items": {"type": "string"}},
        "supports": {"type": "object",
                     "properties": {"action": _BOOL}},
        "action_batching": {
            "type": "object",
            "properties": {"supported": _BOOL, "max_batch_size": _POS_INT,
                           "semantics": {"enum": [*_SEMANTICS, None]},
                           "slot_isolation": {"enum": [*_SLOT_ISO, None]}}},
        "inference_state": {
            "type": "object",
            "properties": {"scope": {"enum": [*_STATE_SCOPE, None]},
                           "supports_session_ids": _BOOL,
                           "reset_scope": {"enum": [*_RESET_SCOPE, None]}}},
    },
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
                           "semantics": {"enum": [*_SEMANTICS, None]},
                           "slot_isolation": {"enum": [*_SLOT_ISO, None]}}},
        "inference_state": {
            "type": "object", "additionalProperties": False,
            "required": ["scope", "supports_session_ids", "reset_scope"],
            "properties": {"scope": {"enum": [*_STATE_SCOPE, None]},
                           "supports_session_ids": {"type": "boolean"},
                           "reset_scope": {"enum": [*_RESET_SCOPE, None]}}},
    },
}


class CapabilitiesNormalizationError(ValueError):
    """Raised when a raw capabilities payload cannot be canonically normalized."""


def _req_bool(v, field: str, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    raise CapabilitiesNormalizationError(
        f"{field!r} must be a JSON boolean, got {type(v).__name__} {v!r} "
        "(no coercion).")


def _req_int(v, field: str, default: int = 1) -> int:
    if v is None:
        return default
    if isinstance(v, bool) or not isinstance(v, int):   # bool is a subclass of int
        raise CapabilitiesNormalizationError(
            f"{field!r} must be a JSON integer, got {type(v).__name__} {v!r} "
            "(no coercion).")
    return v


def _req_enum(v, field: str, allowed: tuple):
    if v is not None and v not in allowed:
        raise CapabilitiesNormalizationError(
            f"{field!r}={v!r} is not one of {allowed} (no enum repair).")
    return v


def _sorted_unique(values, field: str) -> list:
    seq = list(values or [])
    if not all(isinstance(v, str) for v in seq):
        raise CapabilitiesNormalizationError(f"{field!r} must be list[str].")
    if len(seq) != len(set(seq)):
        raise CapabilitiesNormalizationError(f"duplicate value in {field!r}: {seq}.")
    return sorted(seq)


@dataclass(frozen=True)
class NormalizedRunnerCapabilities:
    """The single, typed authority for capability-driven runtime decisions (v1.3.22).

    Built ONLY from a strictly-validated, coercion-free normalization so the runtime
    and the evidence share one representation."""
    runtime: str | None
    protocol_version: str | None
    backward_compatible_with: tuple[str, ...]
    observation_encodings: tuple[str, ...]
    supports_action: bool
    batching_supported: bool
    max_batch_size: int
    batching_semantics: str | None
    slot_isolation: str | None
    state_scope: str | None
    supports_session_ids: bool
    reset_scope: str | None

    def to_dict(self) -> dict:
        return normalize_from_typed(self)

    # v1.3.23 (P1.1/P1.4): duck-typed accessors so the NORMALIZED object is a
    # drop-in authority for select_batch_execution_mode — the runtime batch decision
    # reads capability facts from here, not from the raw/parsed object.
    @property
    def supports_batch(self) -> bool:
        return self.batching_supported

    @property
    def inference_state_scope(self) -> str | None:
        return self.state_scope

    @property
    def action_batching_semantics(self) -> str | None:
        return self.batching_semantics

    @property
    def action_batching_slot_isolation(self) -> str | None:
        return self.slot_isolation

    @property
    def action_batching_state_isolation(self) -> str | None:
        return self.slot_isolation      # alias already resolved during normalization


def normalize_capabilities(raw: dict) -> dict:
    """Canonicalize an announced capabilities payload, fail-closed (no coercion)."""
    if not isinstance(raw, dict):
        raise CapabilitiesNormalizationError("capabilities payload must be an object.")
    import jsonschema
    try:
        jsonschema.validate(raw, RAW_CAPABILITIES_SCHEMA)      # strict raw contract
    except jsonschema.ValidationError as exc:
        raise CapabilitiesNormalizationError(f"raw capabilities invalid: {exc.message}")
    supports = raw.get("supports") or {}
    ab = raw.get("action_batching") or {}
    ist = raw.get("inference_state") or {}
    if not isinstance(supports, dict) or not isinstance(ab, dict) \
            or not isinstance(ist, dict):
        raise CapabilitiesNormalizationError("supports/action_batching/inference_state "
                                             "must be objects.")
    normalized = {
        "normalization_algorithm_version": NORMALIZE_ALGORITHM_VERSION,
        "runtime": raw.get("runtime"),
        "protocol_version": raw.get("protocol_version"),
        "backward_compatible_with": _sorted_unique(
            raw.get("backward_compatible_with"), "backward_compatible_with"),
        "observation_encodings": _sorted_unique(
            raw.get("observation_encodings"), "observation_encodings"),
        "supports_action": _req_bool(supports.get("action"), "supports.action"),
        "action_batching": {
            "supported": _req_bool(ab.get("supported"), "action_batching.supported"),
            "max_batch_size": _req_int(ab.get("max_batch_size"),
                                       "action_batching.max_batch_size", 1),
            "semantics": _req_enum(ab.get("semantics"),
                                   "action_batching.semantics", _SEMANTICS),
            "slot_isolation": _resolve_slot_isolation(ab)},
        "inference_state": {
            "scope": _req_enum(ist.get("scope"), "inference_state.scope", _STATE_SCOPE),
            "supports_session_ids": _req_bool(
                ist.get("supports_session_ids"), "inference_state.supports_session_ids"),
            "reset_scope": _req_enum(ist.get("reset_scope"),
                                     "inference_state.reset_scope", _RESET_SCOPE)},
    }
    _validate_conditional(normalized)              # v1.3.23 (P1.3): closed contract
    return normalized


def _resolve_slot_isolation(ab: dict) -> str | None:
    """Resolve the ``slot_isolation`` / ``state_isolation`` alias, fail on conflict
    (v1.3.23, P1.2). A payload that declares both with different values is rejected."""
    canonical = ab.get("slot_isolation")
    alias = ab.get("state_isolation")
    if canonical is not None and alias is not None and canonical != alias:
        raise CapabilitiesNormalizationError(
            f"conflicting slot_isolation {canonical!r} vs state_isolation alias "
            f"{alias!r}.")
    value = canonical if canonical is not None else alias
    return _req_enum(value, "action_batching.slot_isolation", _SLOT_ISO)


def _validate_conditional(n: dict) -> None:
    """Enforce the conditional capabilities contract (v1.3.23, P1.3)."""
    ab, ist = n["action_batching"], n["inference_state"]
    if ab["supported"]:
        if ab["semantics"] is None:
            raise CapabilitiesNormalizationError(
                "action_batching.supported requires a semantics.")
        if ab["slot_isolation"] is None:
            raise CapabilitiesNormalizationError(
                "action_batching.supported requires a slot_isolation.")
        if ab["max_batch_size"] < 2:
            raise CapabilitiesNormalizationError(
                "action_batching.supported requires max_batch_size >= 2.")
    if ab["semantics"] == "native" and ab["slot_isolation"] is None:
        raise CapabilitiesNormalizationError(
            "native batching requires a slot_isolation.")
    if ist["scope"] == "session_scoped" and not ist["supports_session_ids"]:
        raise CapabilitiesNormalizationError(
            "session_scoped inference requires supports_session_ids.")
    if ist["reset_scope"] == "session" and not ist["supports_session_ids"]:
        raise CapabilitiesNormalizationError(
            "reset_scope=session requires supports_session_ids.")


def typed_normalized_capabilities(raw: dict) -> NormalizedRunnerCapabilities:
    """Strictly normalize + validate into the typed authority object."""
    n = normalize_capabilities(raw)
    import jsonschema
    jsonschema.validate(n, NORMALIZED_CAPABILITIES_SCHEMA)
    ab, ist = n["action_batching"], n["inference_state"]
    return NormalizedRunnerCapabilities(
        runtime=n["runtime"], protocol_version=n["protocol_version"],
        backward_compatible_with=tuple(n["backward_compatible_with"]),
        observation_encodings=tuple(n["observation_encodings"]),
        supports_action=n["supports_action"],
        batching_supported=ab["supported"], max_batch_size=ab["max_batch_size"],
        batching_semantics=ab["semantics"], slot_isolation=ab["slot_isolation"],
        state_scope=ist["scope"], supports_session_ids=ist["supports_session_ids"],
        reset_scope=ist["reset_scope"])


def normalize_from_typed(n: NormalizedRunnerCapabilities) -> dict:
    return {
        "normalization_algorithm_version": NORMALIZE_ALGORITHM_VERSION,
        "runtime": n.runtime, "protocol_version": n.protocol_version,
        "backward_compatible_with": list(n.backward_compatible_with),
        "observation_encodings": list(n.observation_encodings),
        "supports_action": n.supports_action,
        "action_batching": {"supported": n.batching_supported,
                            "max_batch_size": n.max_batch_size,
                            "semantics": n.batching_semantics,
                            "slot_isolation": n.slot_isolation},
        "inference_state": {"scope": n.state_scope,
                            "supports_session_ids": n.supports_session_ids,
                            "reset_scope": n.reset_scope},
    }


def normalized_capabilities_sha256(raw: dict) -> str:
    """Canonical hash of the NORMALIZED capabilities (the certificate-grade id)."""
    return canonical_json_sha256(normalize_capabilities(raw))


__all__ = ["NORMALIZE_ALGORITHM_VERSION", "RAW_CAPABILITIES_SCHEMA",
           "NORMALIZED_CAPABILITIES_SCHEMA", "CapabilitiesNormalizationError",
           "NormalizedRunnerCapabilities", "normalize_capabilities",
           "typed_normalized_capabilities", "normalized_capabilities_sha256",
           "_HASH_OR", "_NE_STR"]
