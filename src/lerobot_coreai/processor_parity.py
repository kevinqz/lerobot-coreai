# processor_parity.py — ProcessorParityReport v1 (v1.3.26).
#
# Prove that a REFERENCE processor path and a CANDIDATE (CoreAI) path produce
# semantically equivalent tensors at each stage. Exact transforms must match by hash;
# numeric transforms must satisfy EXPLICIT thresholds carried in the case (never
# hardcoded). NaN/Inf and shape mismatch always fail. Offline-verifiable. Pure Python.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .processor_parity_metrics import compute_parity_metrics
from .rollout_evidence_schema import canonical_json_sha256

PROCESSOR_PARITY_SCHEMA_VERSION = "lerobot-coreai.processor-parity.v1"
_SHA256 = {"type": "string", "pattern": r"^sha256:[0-9a-f]{64}$"}

PROCESSOR_PARITY_REPORT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object", "additionalProperties": False,
    "required": ["schema_version", "cases", "passed", "claims"],
    "properties": {
        "schema_version": {"const": PROCESSOR_PARITY_SCHEMA_VERSION},
        "dataset_metadata_sha256": {"type": ["string", "null"]},
        "feature_contract_sha256": {"type": ["string", "null"]},
        "processor_stage_contract_sha256": {"type": ["string", "null"]},
        "cases": {"type": "array"},
        "passed": {"type": "boolean"},
        "claims": {
            "type": "object", "additionalProperties": False,
            "required": ["processor_parity_verified", "model_output_parity_verified",
                         "proves_task_success"],
            "properties": {
                "processor_parity_verified": {"type": "boolean"},
                "model_output_parity_verified": {"const": False},
                "proves_task_success": {"const": False}}},
    },
}


@dataclass
class ParityCase:
    feature_id: str
    source_stage: str
    target_stage: str
    mode: str                       # "exact" | "numeric"
    reference: Any                  # nested list
    candidate: Any                  # nested list
    thresholds: dict = field(default_factory=dict)   # numeric gates (explicit)

    def evaluate(self) -> dict:
        m = compute_parity_metrics(self.reference, self.candidate)
        reasons: list[str] = []
        if m.nonfinite_ref or m.nonfinite_cand:
            reasons.append("non-finite value present")
        if not m.shape_match:
            reasons.append(f"shape {m.shape_ref} != {m.shape_cand}")
        if self.mode == "exact":
            ref_h = canonical_json_sha256(self.reference)
            cand_h = canonical_json_sha256(self.candidate)
            if ref_h != cand_h:
                reasons.append("exact hash mismatch")
            entry_hash = {"reference_sha256": ref_h, "candidate_sha256": cand_h}
        else:  # numeric — thresholds MUST be explicit (fail if absent)
            entry_hash = {}
            if not self.thresholds:
                reasons.append("numeric parity requires explicit thresholds")
            for key, limit in self.thresholds.items():
                if key == "min_cosine_similarity":
                    if m.cosine_similarity < limit:
                        reasons.append(f"cosine {m.cosine_similarity:.6f} < {limit}")
                else:  # *_error style upper bounds
                    val = getattr(m, key, None)
                    if val is None:
                        reasons.append(f"unknown threshold {key}")
                    elif val > limit:
                        reasons.append(f"{key} {val} > {limit}")
        passed = not reasons
        return {"feature_id": self.feature_id, "source_stage": self.source_stage,
                "target_stage": self.target_stage, "mode": self.mode,
                "metrics": m.to_dict(), "thresholds": dict(self.thresholds),
                **entry_hash, "passed": passed, "reasons": reasons}


def build_processor_parity_report(
    cases: list[ParityCase], *, dataset_metadata_sha256: str | None = None,
    feature_contract_sha256: str | None = None,
    processor_stage_contract_sha256: str | None = None,
) -> dict:
    evaluated = [c.evaluate() for c in cases]
    passed = bool(evaluated) and all(e["passed"] for e in evaluated)
    return {
        "schema_version": PROCESSOR_PARITY_SCHEMA_VERSION,
        "dataset_metadata_sha256": dataset_metadata_sha256,
        "feature_contract_sha256": feature_contract_sha256,
        "processor_stage_contract_sha256": processor_stage_contract_sha256,
        "cases": evaluated, "passed": passed,
        "claims": {"processor_parity_verified": passed,
                   "model_output_parity_verified": False,
                   "proves_task_success": False},
    }


def verify_processor_parity_report(report: dict) -> tuple[bool, list]:
    """Offline: schema-valid, every case's pass/fail is consistent with its reasons,
    and the top-level claim matches. Re-derivation of metrics is done by re-running
    the report against raw arrays elsewhere; here we enforce internal consistency."""
    import jsonschema
    errors: list[str] = []
    try:
        jsonschema.validate(report, PROCESSOR_PARITY_REPORT_SCHEMA)
    except Exception as exc:  # noqa: BLE001
        return False, [f"schema: {exc}"]
    for c in report["cases"]:
        if c["passed"] != (not c["reasons"]):
            errors.append(f"{c['feature_id']}: passed flag inconsistent with reasons")
    expected = bool(report["cases"]) and all(c["passed"] for c in report["cases"])
    if report["passed"] != expected:
        errors.append("top-level passed inconsistent with cases")
    if report["claims"]["processor_parity_verified"] != report["passed"]:
        errors.append("processor_parity_verified inconsistent with passed")
    return (not errors), errors
