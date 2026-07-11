# model_conversion_evidence.py — ModelConversionEvidence v1 (v1.3.26.2 / RFC v1.3.26A).
#
# A real .aimodel with a real hash is not enough: certification could otherwise prove
# that the WRONG artifact was executed correctly. This binds the conversion chain —
# source checkpoint → exporter/config → graph partition/operator coverage →
# quantization → .aimodel → numeric parity vs reference outputs — into one verifiable
# record. model_conversion_verified is promoted ONLY with a complete identity chain +
# passing numeric parity within an explicit tolerance. Pure Python; no exporter needed
# to VERIFY (the exporter runs Apple/CoreAI-side; here we schema + hash + parity-gate).

from __future__ import annotations

from .processor_parity_metrics import compute_parity_metrics
from .rollout_evidence_schema import canonical_json_sha256

MODEL_CONVERSION_EVIDENCE_SCHEMA_VERSION = "lerobot-coreai.model-conversion-evidence.v1"
_SHA256 = {"type": "string", "pattern": r"^sha256:[0-9a-f]{64}$"}
_NE_STR = {"type": "string", "minLength": 1}

MODEL_CONVERSION_EVIDENCE_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object", "additionalProperties": False,
    "required": ["schema_version", "source", "exporter", "export_configuration_sha256",
                 "artifact", "numeric_parity", "claims"],
    "properties": {
        "schema_version": {"const": MODEL_CONVERSION_EVIDENCE_SCHEMA_VERSION},
        "source": {
            "type": "object", "additionalProperties": False,
            "required": ["repository", "revision", "weights_sha256"],
            "properties": {"repository": _NE_STR, "revision": _NE_STR,
                           "weights_sha256": _SHA256}},
        "exporter": {
            "type": "object", "additionalProperties": False,
            "required": ["name", "build"],
            "properties": {"name": _NE_STR, "build": _NE_STR}},
        "export_configuration_sha256": _SHA256,
        "operator_coverage": {"type": "array", "items": {"type": "string"}},
        "quantization": {"anyOf": [{"type": "null"}, {
            "type": "object", "additionalProperties": False,
            "required": ["scheme"],
            "properties": {"scheme": _NE_STR, "parameters": {"type": "object"}}}]},
        "artifact": {
            "type": "object", "additionalProperties": False,
            "required": ["aimodel_sha256", "aimodel_schema_version", "manifest_sha256"],
            "properties": {"aimodel_sha256": _SHA256,
                           "aimodel_schema_version": _NE_STR,
                           "manifest_sha256": _SHA256}},
        "numeric_parity": {
            "type": "object", "additionalProperties": False,
            "required": ["reference_outputs_sha256", "candidate_outputs_sha256",
                         "tolerance", "metrics", "passed"],
            "properties": {
                "reference_inputs_sha256": {"type": ["string", "null"]},
                "reference_outputs_sha256": _SHA256,
                "candidate_outputs_sha256": _SHA256,
                "tolerance": {"type": "object"},
                "metrics": {"type": "object"},
                "passed": {"type": "boolean"}}},
        "claims": {
            "type": "object", "additionalProperties": False,
            "required": ["model_conversion_verified", "numeric_parity_verified",
                         "proves_task_success", "proves_physical_safety"],
            "properties": {
                "model_conversion_verified": {"type": "boolean"},
                "numeric_parity_verified": {"type": "boolean"},
                "proves_task_success": {"const": False},
                "proves_physical_safety": {"const": False}}},
    },
}


class ModelConversionError(ValueError):
    """Raised when conversion evidence is malformed or a tolerance is missing."""


def _output_hash(arr) -> str:
    """Canonical hash that TOLERATES non-finite leaves (a failed conversion may emit
    NaN/Inf); such leaves are encoded as sentinels so the record still binds exact
    bytes while numeric parity independently fails on them."""
    import math

    def sanitize(v):
        if isinstance(v, list):
            return [sanitize(x) for x in v]
        if isinstance(v, float) and not math.isfinite(v):
            return "__inf__" if v > 0 else ("__ninf__" if v < 0 else "__nan__")
        return v
    return canonical_json_sha256(sanitize(arr))


def _parity(reference_outputs, candidate_outputs, tolerance: dict) -> tuple[dict, bool, list]:
    if not tolerance:
        raise ModelConversionError("numeric parity requires an explicit tolerance")
    m = compute_parity_metrics(reference_outputs, candidate_outputs)
    reasons: list[str] = []
    if m.nonfinite_ref or m.nonfinite_cand:
        reasons.append("non-finite value")
    if not m.shape_match:
        reasons.append("shape mismatch")
    for key, limit in tolerance.items():
        if key == "min_cosine_similarity":
            if m.cosine_similarity < limit:
                reasons.append(f"cosine {m.cosine_similarity:.6f} < {limit}")
        else:
            val = getattr(m, key, None)
            if val is None:
                reasons.append(f"unknown tolerance {key}")
            elif val > limit:
                reasons.append(f"{key} {val} > {limit}")
    return m.to_dict(), (not reasons), reasons


def build_model_conversion_evidence(
    *, source: dict, exporter: dict, export_configuration: dict, artifact: dict,
    reference_outputs, candidate_outputs, tolerance: dict,
    operator_coverage: list | None = None, quantization: dict | None = None,
    reference_inputs=None,
) -> dict:
    """Assemble conversion evidence. model_conversion_verified is true ONLY when the
    identity chain is complete AND numeric parity passes within the tolerance."""
    metrics, parity_ok, _ = _parity(reference_outputs, candidate_outputs, tolerance)
    ev = {
        "schema_version": MODEL_CONVERSION_EVIDENCE_SCHEMA_VERSION,
        "source": source, "exporter": exporter,
        "export_configuration_sha256": canonical_json_sha256(export_configuration),
        "operator_coverage": list(operator_coverage or []),
        "quantization": quantization,
        "artifact": artifact,
        "numeric_parity": {
            "reference_inputs_sha256": (canonical_json_sha256(reference_inputs)
                                        if reference_inputs is not None else None),
            "reference_outputs_sha256": _output_hash(reference_outputs),
            "candidate_outputs_sha256": _output_hash(candidate_outputs),
            "tolerance": dict(tolerance), "metrics": metrics, "passed": parity_ok},
        "claims": {"model_conversion_verified": parity_ok,
                   "numeric_parity_verified": parity_ok,
                   "proves_task_success": False, "proves_physical_safety": False},
    }
    import jsonschema
    jsonschema.validate(ev, MODEL_CONVERSION_EVIDENCE_SCHEMA)
    return ev


def verify_model_conversion_evidence(
    evidence: dict, *, reference_outputs=None, candidate_outputs=None,
) -> tuple[bool, list]:
    """Offline: schema-valid + claim consistent with recorded parity. If the raw
    reference/candidate arrays are supplied, the recorded hashes + metrics + pass flag
    are RE-DERIVED from them (independent re-proof, not just internal consistency)."""
    import jsonschema
    errors: list[str] = []
    try:
        jsonschema.validate(evidence, MODEL_CONVERSION_EVIDENCE_SCHEMA)
    except Exception as exc:  # noqa: BLE001
        return False, [f"schema: {exc}"]
    np_block = evidence["numeric_parity"]
    if evidence["claims"]["model_conversion_verified"] != np_block["passed"]:
        errors.append("model_conversion_verified inconsistent with numeric parity")
    if evidence["claims"]["numeric_parity_verified"] != np_block["passed"]:
        errors.append("numeric_parity_verified inconsistent with numeric parity")
    if reference_outputs is not None and candidate_outputs is not None:
        if _output_hash(reference_outputs) != np_block["reference_outputs_sha256"]:
            errors.append("reference_outputs hash mismatch")
        if _output_hash(candidate_outputs) != np_block["candidate_outputs_sha256"]:
            errors.append("candidate_outputs hash mismatch")
        _, ok, reasons = _parity(reference_outputs, candidate_outputs,
                                 np_block["tolerance"])
        if ok != np_block["passed"]:
            errors.append(f"recomputed parity {ok} != recorded {np_block['passed']} "
                          f"({reasons})")
    return (not errors), errors
