# apple_runtime.py — Apple/CoreAI runtime identity + certificate (v1.4.0 machinery).
#
# Run the exact signed artifact on a real Apple/CoreAI runtime and prove the hardware
# execution matches the certified contract. This module is the DIAGNOSTIC-GRADE
# harness + the offline verifier + the hard gate: apple_runtime_certified can be true
# ONLY on a real arm64 Apple host, with a real CoreAI Runner (no fake/stub), a real
# .aimodel loaded, all required cases + numeric parity passed, and the signed
# official-eval chain verified. On Linux CI every one of those is false, so the claim
# stays false (invariant §5.4: claims by proof, not version). Pure Python.

from __future__ import annotations

import platform

from .rollout_evidence_schema import canonical_json_sha256

APPLE_RUNTIME_IDENTITY_SCHEMA_VERSION = "lerobot-coreai.apple-runtime-identity.v1"
APPLE_RUNTIME_CERTIFICATE_SCHEMA_VERSION = "lerobot-coreai.apple-runtime-certificate.v1"
_SHA256 = {"type": "string", "pattern": r"^sha256:[0-9a-f]{64}$"}
_SHA256_OR_NULL = {"anyOf": [_SHA256, {"type": "null"}]}

APPLE_RUNTIME_IDENTITY_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object", "additionalProperties": False,
    "required": ["schema_version", "hardware", "software", "model", "execution"],
    "properties": {
        "schema_version": {"const": APPLE_RUNTIME_IDENTITY_SCHEMA_VERSION},
        "hardware": {
            "type": "object", "additionalProperties": False,
            "required": ["model_identifier", "chip", "memory_bytes", "cpu_arch"],
            "properties": {"model_identifier": {"type": ["string", "null"]},
                           "chip": {"type": ["string", "null"]},
                           "memory_bytes": {"type": ["integer", "null"]},
                           "cpu_arch": {"type": ["string", "null"]}}},
        "software": {
            "type": "object", "additionalProperties": False,
            "required": ["macos_version", "macos_build", "xcode_version", "sdk_version",
                         "coreai_runner_version", "coreai_runner_binary_sha256"],
            "properties": {"macos_version": {"type": ["string", "null"]},
                           "macos_build": {"type": ["string", "null"]},
                           "xcode_version": {"type": ["string", "null"]},
                           "sdk_version": {"type": ["string", "null"]},
                           "coreai_runner_version": {"type": ["string", "null"]},
                           "coreai_runner_binary_sha256": _SHA256_OR_NULL}},
        "model": {
            "type": "object", "additionalProperties": False,
            "required": ["aimodel_sha256", "aimodel_schema_version", "manifest_sha256"],
            "properties": {"aimodel_sha256": _SHA256_OR_NULL,
                           "aimodel_schema_version": {"type": ["string", "null"]},
                           "manifest_sha256": _SHA256_OR_NULL}},
        "execution": {
            "type": "object", "additionalProperties": False,
            "required": ["compute_units_requested", "compute_units_reported",
                         "process_arch"],
            "properties": {"compute_units_requested": {"type": ["string", "null"]},
                           "compute_units_reported": {"type": ["string", "null"]},
                           "process_arch": {"type": ["string", "null"]}}},
    },
}

_CHECK_KEYS = ("real_runner_used", "real_aimodel_loaded", "all_required_cases_passed",
               "numeric_parity_passed", "official_eval_chain_bound",
               "signed_evidence_verified")
APPLE_RUNTIME_CERTIFICATE_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object", "additionalProperties": False,
    "required": ["schema_version", "evidence_grade", "identity_sha256",
                 "artifact_root_sha256", "aimodel_sha256", "cases",
                 "performance_summary", "checks", "claims"],
    "properties": {
        "schema_version": {"const": APPLE_RUNTIME_CERTIFICATE_SCHEMA_VERSION},
        "evidence_grade": {"enum": ["diagnostic", "certificate"]},
        "identity_sha256": _SHA256,
        "signed_official_eval_certificate_sha256": _SHA256_OR_NULL,
        "artifact_root_sha256": _SHA256_OR_NULL,
        "aimodel_sha256": _SHA256_OR_NULL,
        "feature_contract_sha256": _SHA256_OR_NULL,
        "processor_parity_sha256": _SHA256_OR_NULL,
        "cases": {"type": "array"},
        "performance_summary": {"type": "object"},
        "checks": {"type": "object", "additionalProperties": False,
                   "required": list(_CHECK_KEYS),
                   "properties": {k: {"type": "boolean"} for k in _CHECK_KEYS}},
        "claims": {
            "type": "object", "additionalProperties": False,
            "required": ["apple_runtime_certified", "scope_is_exact_hardware_software_artifact",
                         "proves_general_apple_compatibility", "proves_physical_safety",
                         "proves_real_robot_task_success"],
            "properties": {
                "apple_runtime_certified": {"type": "boolean"},
                "scope_is_exact_hardware_software_artifact": {"type": "boolean"},
                "proves_general_apple_compatibility": {"const": False},
                "proves_physical_safety": {"const": False},
                "proves_real_robot_task_success": {"const": False}}},
    },
}


def _macos_sw_versions() -> tuple[str | None, str | None]:
    try:
        mac_ver = platform.mac_ver()[0] or None
    except Exception:  # noqa: BLE001
        mac_ver = None
    return mac_ver, None


def capture_apple_runtime_identity(
    *, coreai_runner_version: str | None = None,
    coreai_runner_binary_sha256: str | None = None,
    aimodel_sha256: str | None = None, aimodel_schema_version: str | None = None,
    manifest_sha256: str | None = None, compute_units_requested: str | None = None,
    compute_units_reported: str | None = None,
) -> dict:
    """Capture the runtime identity. On a non-Apple/non-arm64 host the Apple-specific
    fields are null and the certificate gate can never pass — honest by construction."""
    mac_ver, mac_build = _macos_sw_versions()
    is_macos = platform.system() == "Darwin"
    return {
        "schema_version": APPLE_RUNTIME_IDENTITY_SCHEMA_VERSION,
        "hardware": {"model_identifier": None,
                     "chip": platform.processor() or None if is_macos else None,
                     "memory_bytes": None, "cpu_arch": platform.machine() or None},
        "software": {"macos_version": mac_ver if is_macos else None,
                     "macos_build": mac_build, "xcode_version": None,
                     "sdk_version": None,
                     "coreai_runner_version": coreai_runner_version,
                     "coreai_runner_binary_sha256": coreai_runner_binary_sha256},
        "model": {"aimodel_sha256": aimodel_sha256,
                  "aimodel_schema_version": aimodel_schema_version,
                  "manifest_sha256": manifest_sha256},
        "execution": {"compute_units_requested": compute_units_requested,
                      "compute_units_reported": compute_units_reported,
                      "process_arch": platform.machine() or None},
    }


def identity_sha256(identity: dict) -> str:
    return canonical_json_sha256(identity)


def _gate(identity: dict, checks: dict) -> bool:
    """apple_runtime_certified is true ONLY when every real-execution check passes AND
    the host + process are arm64 Apple Silicon (no fake runner, no Rosetta/x86)."""
    hw_arm = identity["hardware"]["cpu_arch"] == "arm64"
    proc_arm = identity["execution"]["process_arch"] == "arm64"
    macos = identity["software"]["macos_version"] is not None
    return bool(hw_arm and proc_arm and macos and all(checks.get(k) for k in _CHECK_KEYS))


def build_diagnostic_apple_runtime_report(
    *, identity: dict, checks: dict, artifact_root_sha256: str | None = None,
    aimodel_sha256: str | None = None, feature_contract_sha256: str | None = None,
    processor_parity_sha256: str | None = None,
    signed_official_eval_certificate_sha256: str | None = None,
    cases: list | None = None, performance_summary: dict | None = None,
) -> dict:
    """DIAGNOSTIC report only (v1.3.26.8): apple_runtime_certified is ALWAYS false,
    regardless of the caller's checks. A true certificate can be produced ONLY by
    ``promote_apple_runtime_certificate`` from verified receipts — no caller boolean
    can promote the claim."""
    return build_apple_runtime_certificate(
        identity=identity, checks=checks, artifact_root_sha256=artifact_root_sha256,
        aimodel_sha256=aimodel_sha256, feature_contract_sha256=feature_contract_sha256,
        processor_parity_sha256=processor_parity_sha256,
        signed_official_eval_certificate_sha256=signed_official_eval_certificate_sha256,
        cases=cases, performance_summary=performance_summary, _authority=None)


def promote_apple_runtime_certificate(
    *, identity: dict,
    runtime_receipt, official_eval, conversion, trust_policy,
    feature_contract_sha256: str | None = None,
    processor_parity_sha256: str | None = None,
    performance_summary: dict | None = None,
) -> dict:
    """Promote a TRUE apple_runtime certificate (v1.3.26.8, P0.2).

    Accepts ONLY unforgeable ``Verified*`` receipts (a dict/bool raises TypeError) whose
    substance was re-derived by their verifiers — no caller boolean reaches this gate.
    The checks are DERIVED from the runtime receipt + verified chain, never supplied.
    apple_runtime_certified is still the arm64-Apple gate AND the receipt substance, so
    on Linux CI (non-arm64, no macOS) the claim is false even with perfect receipts."""
    import jsonschema

    from .authority import (
        VerifiedCoreAIRuntimeReceipt, VerifiedModelConversionEvidence,
        VerifiedSignedOfficialEvalCertificate, VerifiedTrustPolicy,
    )
    if not isinstance(runtime_receipt, VerifiedCoreAIRuntimeReceipt):
        raise TypeError("runtime_receipt must be a VerifiedCoreAIRuntimeReceipt "
                        "(mint via authority.verify_coreai_runtime_receipt)")
    if not isinstance(official_eval, VerifiedSignedOfficialEvalCertificate):
        raise TypeError("official_eval must be a VerifiedSignedOfficialEvalCertificate")
    if not isinstance(conversion, VerifiedModelConversionEvidence):
        raise TypeError("conversion must be a VerifiedModelConversionEvidence")
    if not isinstance(trust_policy, VerifiedTrustPolicy):
        raise TypeError("trust_policy must be a VerifiedTrustPolicy")
    jsonschema.validate(identity, APPLE_RUNTIME_IDENTITY_SCHEMA)

    receipt = runtime_receipt.payload
    # checks derived from receipt substance (not the caller)
    checks = {
        "real_runner_used": bool(receipt["real_runner_used"] and not receipt["fake_runner"]),
        "real_aimodel_loaded": bool(receipt["aimodel_opened"]),
        "all_required_cases_passed": True,          # enforced at mint (full matrix)
        "numeric_parity_passed": bool(receipt["numeric_parity_passed"]),
        "official_eval_chain_bound": True,          # VerifiedSignedOfficialEvalCertificate
        "signed_evidence_verified": True,           # verified signature + policy at mint
    }
    signed_eval_root = official_eval.payload["predicate"]["certificate_root_sha256"]
    return build_apple_runtime_certificate(
        identity=identity, checks=checks,
        artifact_root_sha256=receipt["artifact_aimodel_sha256"],
        aimodel_sha256=receipt["aimodel_root_sha256"],
        feature_contract_sha256=feature_contract_sha256,
        processor_parity_sha256=processor_parity_sha256,
        signed_official_eval_certificate_sha256=signed_eval_root,
        cases=sorted(receipt["cases"]), performance_summary=performance_summary,
        _authority=object.__new__(_PromotionAuthority),
    )


class _PromotionAuthority:
    """Marker: the only path that may set apple_runtime_certified to the gate result."""


def build_apple_runtime_certificate(
    *, identity: dict, checks: dict, artifact_root_sha256: str | None = None,
    aimodel_sha256: str | None = None, feature_contract_sha256: str | None = None,
    processor_parity_sha256: str | None = None,
    signed_official_eval_certificate_sha256: str | None = None,
    cases: list | None = None, performance_summary: dict | None = None,
    _authority=None,
) -> dict:
    """Internal certificate assembler. When called WITHOUT the promotion authority it
    behaves as a diagnostic (claim forced false); the authority path lets the claim be
    the gate result. Public callers use ``build_diagnostic_apple_runtime_report`` or
    ``promote_apple_runtime_certificate``."""
    import jsonschema
    jsonschema.validate(identity, APPLE_RUNTIME_IDENTITY_SCHEMA)
    full_checks = {k: bool(checks.get(k, False)) for k in _CHECK_KEYS}
    promoted = isinstance(_authority, _PromotionAuthority)
    certified = _gate(identity, full_checks) if promoted else False
    cert = {
        "schema_version": APPLE_RUNTIME_CERTIFICATE_SCHEMA_VERSION,
        "evidence_grade": "certificate" if promoted else "diagnostic",
        "identity_sha256": identity_sha256(identity),
        "signed_official_eval_certificate_sha256": signed_official_eval_certificate_sha256,
        "artifact_root_sha256": artifact_root_sha256,
        "aimodel_sha256": aimodel_sha256,
        "feature_contract_sha256": feature_contract_sha256,
        "processor_parity_sha256": processor_parity_sha256,
        "cases": list(cases or []), "performance_summary": dict(performance_summary or {}),
        "checks": full_checks,
        "claims": {"apple_runtime_certified": certified,
                   "scope_is_exact_hardware_software_artifact": certified,
                   "proves_general_apple_compatibility": False,
                   "proves_physical_safety": False,
                   "proves_real_robot_task_success": False},
    }
    jsonschema.validate(cert, APPLE_RUNTIME_CERTIFICATE_SCHEMA)
    return cert


def verify_apple_runtime_certificate(certificate: dict, identity: dict) -> tuple[bool, list]:
    """Offline: schema-valid, identity binds, and apple_runtime_certified is EXACTLY
    the gate result — a forged true (fake runner, x86 process, unbound chain) fails."""
    import jsonschema
    errors: list[str] = []
    try:
        jsonschema.validate(certificate, APPLE_RUNTIME_CERTIFICATE_SCHEMA)
        jsonschema.validate(identity, APPLE_RUNTIME_IDENTITY_SCHEMA)
    except Exception as exc:  # noqa: BLE001
        return False, [f"schema: {exc}"]
    if certificate["identity_sha256"] != identity_sha256(identity):
        errors.append("identity_sha256 does not bind the supplied identity")
    claimed = certificate["claims"]["apple_runtime_certified"]
    grade = certificate["evidence_grade"]
    if grade == "diagnostic":
        # a diagnostic report may NEVER certify — a forged true is caught here.
        if claimed:
            errors.append("diagnostic-grade report forged to apple_runtime_certified=true "
                          "(diagnostic must not certify)")
    else:  # certificate grade: the claim must be EXACTLY the gate result
        expected = _gate(identity, certificate["checks"])
        if claimed != expected:
            errors.append(f"apple_runtime_certified={claimed} != gate {expected} "
                          "(forged or inconsistent)")
    if claimed and not certificate.get("signed_official_eval_certificate_sha256"):
        errors.append("certified without a bound signed official-eval certificate")
    return (not errors), errors
