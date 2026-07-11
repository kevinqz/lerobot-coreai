# feature_contract_reports.py — FeatureContract validation reports (v1.3.24).
#
# An offline-verifiable report binding a FeatureContract (by hash) to the payload
# validations run against it. feature_contract_verified is promoted ONLY when the
# contract is structurally valid AND every payload validated cleanly — never by
# version. Pure Python + JSON.

from __future__ import annotations

from .feature_contract import FeatureContract, feature_contract_from_dict
from .feature_contract_validation import (
    FeatureValidationResult, validate_contract_structure,
)
from .rollout_evidence_schema import canonical_json_sha256

VALIDATION_REPORT_SCHEMA_VERSION = "lerobot-coreai.feature-contract-validation-report.v1"

VALIDATION_REPORT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object", "additionalProperties": False,
    "required": ["schema_version", "feature_contract_sha256", "structural_errors",
                 "stage_results", "claims"],
    "properties": {
        "schema_version": {"const": VALIDATION_REPORT_SCHEMA_VERSION},
        "feature_contract_sha256": {"type": "string", "pattern": r"^sha256:[0-9a-f]{64}$"},
        "structural_errors": {"type": "array", "items": {"type": "string"}},
        "stage_results": {"type": "array"},
        "claims": {
            "type": "object", "additionalProperties": False,
            "required": ["feature_contract_verified", "proves_task_success",
                         "proves_physical_safety"],
            "properties": {
                "feature_contract_verified": {"type": "boolean"},
                "proves_task_success": {"const": False},
                "proves_physical_safety": {"const": False}}},
    },
}


def build_validation_report(contract: FeatureContract,
                            results: list[FeatureValidationResult]) -> dict:
    structural = validate_contract_structure(contract)
    verified = not structural and all(r.ok for r in results) and bool(results)
    return {
        "schema_version": VALIDATION_REPORT_SCHEMA_VERSION,
        "feature_contract_sha256": contract.sha256(),
        "structural_errors": structural,
        "stage_results": [r.to_dict() for r in results],
        "claims": {"feature_contract_verified": verified,
                   "proves_task_success": False, "proves_physical_safety": False},
    }


def verify_validation_report(report: dict, contract: FeatureContract) -> tuple[bool, list]:
    """Offline: the report must bind the contract hash and its claim must be
    consistent with the recorded structural errors + stage results."""
    import jsonschema
    errors: list[str] = []
    try:
        jsonschema.validate(report, VALIDATION_REPORT_SCHEMA)
    except Exception as exc:  # noqa: BLE001
        return False, [f"schema: {exc}"]
    if report["feature_contract_sha256"] != contract.sha256():
        errors.append("feature_contract_sha256 does not bind the supplied contract")
    stage_ok = all(r.get("ok") for r in report["stage_results"]) \
        and bool(report["stage_results"])
    expected = not report["structural_errors"] and stage_ok
    if report["claims"]["feature_contract_verified"] != expected:
        errors.append("feature_contract_verified inconsistent with results")
    return (not errors), errors


def load_feature_contract(path: str) -> FeatureContract:
    import json
    with open(path) as fh:
        return feature_contract_from_dict(json.load(fh))
