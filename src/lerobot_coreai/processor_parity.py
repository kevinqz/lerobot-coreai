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
    "required": ["schema_version", "evidence_grade", "cases", "passed", "claims"],
    "properties": {
        "schema_version": {"const": PROCESSOR_PARITY_SCHEMA_VERSION},
        "evidence_grade": {"enum": ["diagnostic", "certificate"]},
        "dataset_metadata_sha256": {"type": ["string", "null"]},
        "feature_contract_sha256": {"type": ["string", "null"]},
        "processor_stage_contract_sha256": {"type": ["string", "null"]},
        "cases": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["feature_id", "source_stage", "target_stage", "mode", "metrics",
                         "thresholds", "reference_sha256", "candidate_sha256", "passed",
                         "reasons"],
            "properties": {
                "feature_id": {"type": "string"}, "source_stage": {"type": "string"},
                "target_stage": {"type": "string"},
                "mode": {"enum": ["exact", "numeric"]},     # closed (no implicit numeric)
                "metrics": {"type": "object"}, "thresholds": {"type": "object"},
                "reference_sha256": _SHA256, "candidate_sha256": _SHA256,
                "passed": {"type": "boolean"},
                "reasons": {"type": "array", "items": {"type": "string"}},
                # raw arrays present only in certificate grade (for replay).
                "reference": {}, "candidate": {}}}},
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

    def evaluate(self, *, include_arrays: bool = False) -> dict:
        if self.mode not in ("exact", "numeric"):
            raise ValueError(f"unknown parity mode {self.mode!r} (exact|numeric)")
        entry = _evaluate_arrays(self.reference, self.candidate, self.mode,
                                 self.thresholds)
        out = {"feature_id": self.feature_id, "source_stage": self.source_stage,
               "target_stage": self.target_stage, "mode": self.mode,
               "metrics": entry["metrics"], "thresholds": dict(self.thresholds),
               "reference_sha256": entry["reference_sha256"],
               "candidate_sha256": entry["candidate_sha256"],
               "passed": entry["passed"], "reasons": entry["reasons"]}
        if include_arrays:      # certificate grade persists the raw arrays for replay
            out["reference"] = self.reference
            out["candidate"] = self.candidate
        return out


def _array_hash(arr) -> str:
    """Content hash that tolerates non-finite leaves (encoded as sentinels) so the
    record binds exact bytes while numeric parity independently fails on them."""
    import math

    def sanitize(v):
        if isinstance(v, list):
            return [sanitize(x) for x in v]
        if isinstance(v, float) and not math.isfinite(v):
            return "__inf__" if v > 0 else ("__ninf__" if v < 0 else "__nan__")
        return v
    return canonical_json_sha256(sanitize(arr))


def _evaluate_arrays(reference, candidate, mode: str, thresholds: dict) -> dict:
    """Pure re-derivation from raw arrays — used at build time AND by the verifier so a
    certificate-grade report is independently replayable, not merely self-consistent."""
    m = compute_parity_metrics(reference, candidate)
    reasons: list[str] = []
    if m.nonfinite_ref or m.nonfinite_cand:
        reasons.append("non-finite value present")
    if not m.shape_match:
        reasons.append(f"shape {m.shape_ref} != {m.shape_cand}")
    ref_h, cand_h = _array_hash(reference), _array_hash(candidate)
    if mode == "exact":
        if ref_h != cand_h:
            reasons.append("exact hash mismatch")
    else:  # numeric — thresholds MUST be explicit (fail if absent)
        if not thresholds:
            reasons.append("numeric parity requires explicit thresholds")
        for key, limit in thresholds.items():
            if key == "min_cosine_similarity":
                if m.cosine_similarity < limit:
                    reasons.append(f"cosine {m.cosine_similarity:.6f} < {limit}")
            else:  # *_error style upper bounds
                val = getattr(m, key, None)
                if val is None:
                    reasons.append(f"unknown threshold {key}")
                elif val > limit:
                    reasons.append(f"{key} {val} > {limit}")
    return {"metrics": m.to_dict(), "reference_sha256": ref_h,
            "candidate_sha256": cand_h, "passed": not reasons, "reasons": reasons}


def build_processor_parity_report(
    cases: list[ParityCase], *, dataset_metadata_sha256: str | None = None,
    feature_contract_sha256: str | None = None,
    processor_stage_contract_sha256: str | None = None,
    evidence_grade: str = "certificate",
) -> dict:
    """Build the report. In ``certificate`` grade (default) each case persists its raw
    reference/candidate arrays so a verifier can RE-DERIVE the metrics + pass/fail
    independently (P1.1). ``diagnostic`` grade omits the arrays (self-consistency only)."""
    if evidence_grade not in ("diagnostic", "certificate"):
        raise ValueError(f"unknown evidence_grade {evidence_grade!r}")
    certificate = evidence_grade == "certificate"
    evaluated = [c.evaluate(include_arrays=certificate) for c in cases]
    passed = bool(evaluated) and all(e["passed"] for e in evaluated)
    return {
        "schema_version": PROCESSOR_PARITY_SCHEMA_VERSION,
        "evidence_grade": evidence_grade,
        "dataset_metadata_sha256": dataset_metadata_sha256,
        "feature_contract_sha256": feature_contract_sha256,
        "processor_stage_contract_sha256": processor_stage_contract_sha256,
        "cases": evaluated, "passed": passed,
        "claims": {"processor_parity_verified": passed,
                   "model_output_parity_verified": False,
                   "proves_task_success": False},
    }


def verify_processor_parity_report(report: dict) -> tuple[bool, list]:
    """Offline. In ``certificate`` grade the verifier RE-DERIVES every case's metrics,
    array hashes, and pass/fail from the persisted raw arrays (independent replay, not
    just internal consistency) — a case whose recorded metrics/hashes/verdict don't
    match a fresh recomputation is rejected (P1.1). ``diagnostic`` grade enforces only
    internal consistency."""
    import jsonschema
    errors: list[str] = []
    try:
        jsonschema.validate(report, PROCESSOR_PARITY_REPORT_SCHEMA)
    except Exception as exc:  # noqa: BLE001
        return False, [f"schema: {exc}"]
    certificate = report["evidence_grade"] == "certificate"
    for c in report["cases"]:
        if c["passed"] != (not c["reasons"]):
            errors.append(f"{c['feature_id']}: passed flag inconsistent with reasons")
        if not certificate:
            continue
        if "reference" not in c or "candidate" not in c:
            errors.append(f"{c['feature_id']}: certificate grade requires raw arrays "
                          "for replay")
            continue
        replay = _evaluate_arrays(c["reference"], c["candidate"], c["mode"],
                                  c.get("thresholds", {}))
        if replay["reference_sha256"] != c.get("reference_sha256"):
            errors.append(f"{c['feature_id']}: replayed reference hash mismatch (tamper)")
        if replay["candidate_sha256"] != c.get("candidate_sha256"):
            errors.append(f"{c['feature_id']}: replayed candidate hash mismatch (tamper)")
        if replay["metrics"] != c.get("metrics"):
            errors.append(f"{c['feature_id']}: replayed metrics != recorded (tamper)")
        if replay["passed"] != c["passed"]:
            errors.append(f"{c['feature_id']}: replayed verdict {replay['passed']} != "
                          f"recorded {c['passed']}")
    expected = bool(report["cases"]) and all(c["passed"] for c in report["cases"])
    if report["passed"] != expected:
        errors.append("top-level passed inconsistent with cases")
    if report["claims"]["processor_parity_verified"] != report["passed"]:
        errors.append("processor_parity_verified inconsistent with passed")
    return (not errors), errors
