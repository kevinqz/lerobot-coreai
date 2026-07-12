# official_eval_certificate.py — OfficialEvalCertificate v1 (v1.3.27 core).
#
# Certify the command users actually run — `lerobot-eval` — not merely the internal
# function it eventually calls. official_eval_certified can be true ONLY when a
# VerifiedOfficialEvalExecutionReceipt (minted from a REAL lerobot-eval subprocess: the
# official entrypoint, an installed executable — not a /tmp shim —, the coreai env
# instantiated, every required case passing with a clean exit) is promoted via
# `promote_official_eval_certificate`. The public builder is DIAGNOSTIC only (claim
# always false); a hand-built execution dict of booleans can NEVER set the claim
# (v1.3.26.8, P0.1). Pure Python; offline.

from __future__ import annotations

from .rollout_evidence_schema import canonical_json_sha256

OFFICIAL_EVAL_CERTIFICATE_SCHEMA_VERSION = "lerobot-coreai.official-eval-certificate.v1"
_SHA256 = {"type": "string", "pattern": r"^sha256:[0-9a-f]{64}$"}
_SHA256_OR_NULL = {"anyOf": [_SHA256, {"type": "null"}]}
REQUIRED_CASES = ("single-b1", "native-b2", "native-b4", "split-b2", "split-b4")
# the full evidence-graph root set an official-eval certificate binds (v1.3.26.11,
# P0.4/WS5). In certificate grade EVERY root must be present + non-null, so the run is
# tied to concrete artifacts/contracts, not a nullable free input.
_ROOT_KEYS = ("artifact_root_sha256", "feature_contract_sha256",
              "dataset_metadata_sha256", "processor_parity_sha256",
              "policy_execution_contract_sha256", "model_conversion_sha256",
              "processor_stage_contract_sha256", "runtime_support_profile_sha256",
              "negotiation_record_sha256", "runner_capabilities_sha256",
              "rollout_matrix_sha256")
_CHECK_KEYS = ("official_cli_entrypoint_used", "third_party_plugin_registration_used",
               "all_required_cases_passed", "outputs_schema_valid",
               "evidence_replay_passed")

OFFICIAL_EVAL_CERTIFICATE_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object", "additionalProperties": False,
    "required": ["schema_version", "evidence_grade", "evidence_namespace", "scope",
                 "inputs", "execution", "checks", "claims"],
    "properties": {
        "schema_version": {"const": OFFICIAL_EVAL_CERTIFICATE_SCHEMA_VERSION},
        "evidence_grade": {"enum": ["diagnostic", "certificate"]},
        # production ⇒ derived from an executor-signed receipt under a pinned release
        # key; test_only ⇒ synthetic/declarative (never a production claim).
        "evidence_namespace": {"enum": ["production", "test_only"]},
        "scope": {
            "type": "object", "additionalProperties": False,
            "required": ["lerobot_version", "environment", "cases"],
            "properties": {"lerobot_version": {"type": ["string", "null"]},
                           "lerobot_distribution_sha256": _SHA256_OR_NULL,
                           "environment": {"type": "string"},
                           "cases": {"type": "array", "items": {"type": "string"}}}},
        "inputs": {
            "type": "object", "additionalProperties": False,
            "required": ["artifact_root_sha256"],
            "properties": {k: _SHA256_OR_NULL for k in _ROOT_KEYS}},
        "execution": {
            "type": "object", "additionalProperties": False,
            "required": ["argv", "command_sha256", "resolved_config_sha256",
                         "output_tree_sha256", "exit_code"],
            "properties": {"argv": {"type": "array", "items": {"type": "string"}},
                           "command_sha256": _SHA256, "resolved_config_sha256": _SHA256,
                           "output_tree_sha256": _SHA256,
                           "exit_code": {"type": "integer"}}},
        "checks": {"type": "object", "additionalProperties": False,
                   "required": list(_CHECK_KEYS),
                   "properties": {k: {"type": "boolean"} for k in _CHECK_KEYS}},
        "claims": {
            "type": "object", "additionalProperties": False,
            "required": ["official_eval_certified", "certification_scope_is_bounded",
                         "proves_general_task_success", "proves_physical_safety"],
            "properties": {
                "official_eval_certified": {"type": "boolean"},
                "certification_scope_is_bounded": {"type": "boolean"},
                "proves_general_task_success": {"const": False},
                "proves_physical_safety": {"const": False}}},
    },
}


def _argv_is_official_eval(argv: list) -> bool:
    """True iff the subprocess argv actually invoked the public lerobot-eval CLI
    (the console script, or `python -m lerobot.scripts.lerobot_eval`)."""
    if not argv:
        return False
    joined = " ".join(argv)
    if argv[0].rsplit("/", 1)[-1] == "lerobot-eval":
        return True
    return "lerobot.scripts.lerobot_eval" in joined and "-m" in argv


def _gate(execution: dict, checks: dict, cases: list) -> bool:
    argv_ok = _argv_is_official_eval(execution.get("argv", []))
    cases_ok = set(cases) == set(REQUIRED_CASES)
    return bool(argv_ok and cases_ok and execution.get("exit_code") == 0
                and all(checks.get(k) for k in _CHECK_KEYS))


class _PromotionAuthority:
    """Marker: the only path that may set official_eval_certified to the gate result."""


def build_diagnostic_official_eval_report(*, scope: dict, inputs: dict, execution: dict,
                                          checks: dict) -> dict:
    """DIAGNOSTIC report only (v1.3.26.8): official_eval_certified is ALWAYS false,
    regardless of the caller's checks/argv. A true certificate is produced ONLY by
    ``promote_official_eval_certificate`` from a verified execution receipt — no
    hand-built execution dict can certify."""
    return _assemble(scope=scope, inputs=inputs, execution=execution, checks=checks,
                     authority=None)


def promote_official_eval_certificate(*, receipt, bundle) -> dict:
    """Promote a TRUE official-eval certificate (v1.3.26.8 → v1.3.26.13). Accepts ONLY a
    ``VerifiedOfficialEvalExecutionReceipt`` (execution + checks derived from its own
    verifier reports, P0.3) AND a ``VerifiedCertificationBundle`` whose eleven roots were
    RE-DERIVED by content-addressing real leaf evidence and re-verified where possible
    (v1.3.26.13). The certificate's ``inputs`` come from the verified bundle — a bare
    ``{root: "sha256:aaa"}`` dict can no longer certify."""
    from .authority import (
        VerifiedCertificationBundle, VerifiedOfficialEvalExecutionReceipt,
    )
    if not isinstance(receipt, VerifiedOfficialEvalExecutionReceipt):
        raise TypeError("receipt must be a VerifiedOfficialEvalExecutionReceipt "
                        "(mint via authority.verify_official_eval_execution_receipt)")
    if not isinstance(bundle, VerifiedCertificationBundle):
        raise TypeError("bundle must be a VerifiedCertificationBundle "
                        "(mint via authority.verify_certification_bundle)")
    inputs = dict(bundle.payload["roots"])      # re-derived, content-addressed roots
    r = receipt.payload
    scope = {"lerobot_version": None,
             "lerobot_distribution_sha256": r["lerobot_distribution_sha256"],
             "environment": "coreai_cert_env", "cases": sorted(r["cases"])}
    execution = {"argv": list(r["argv"]), "command_sha256": r["command_sha256"],
                 "resolved_config_sha256": r["resolved_config_sha256"],
                 "output_tree_sha256": r["output_tree_sha256"],
                 "exit_code": r["exit_code"]}
    # checks DERIVED from the receipt's verifier reports (P0.3) — not hardcoded True.
    checks = {"third_party_plugin_registration_used": bool(r["coreai_env_instantiated"]),
              "all_required_cases_passed": set(r["cases"]) == set(REQUIRED_CASES),
              "outputs_schema_valid": bool(r["schema_report"]["outputs_schema_valid"]),
              "evidence_replay_passed": bool(r["replay_report"]["evidence_replay_passed"])}
    # namespace propagates from the receipt: a declarative receipt is test_only, so the
    # promoted certificate is test_only — a production claim needs an executor-signed
    # receipt (WS1) that does not exist yet (v1.3.26.12).
    namespace = r.get("_namespace", "test_only")
    return _assemble(scope=scope, inputs=inputs, execution=execution, checks=checks,
                     authority=object.__new__(_PromotionAuthority), namespace=namespace)


def _assemble(*, scope: dict, inputs: dict, execution: dict, checks: dict,
              authority, namespace: str = "test_only") -> dict:
    """Internal assembler. Only the promotion authority lets the claim be the gate
    result; otherwise the report is diagnostic (claim forced false)."""
    import jsonschema
    full_checks = {k: bool(checks.get(k, False)) for k in _CHECK_KEYS}
    # the entrypoint check is itself pinned to the recorded argv (no free assertion).
    full_checks["official_cli_entrypoint_used"] = _argv_is_official_eval(
        execution.get("argv", []))
    promoted = isinstance(authority, _PromotionAuthority)
    # certificate grade requires the FULL evidence-graph root set, non-null (P0.4/WS5).
    roots_complete = all(inputs.get(k) for k in _ROOT_KEYS)
    certified = (promoted and roots_complete
                 and _gate(execution, full_checks, scope.get("cases", [])))
    cert = {
        "schema_version": OFFICIAL_EVAL_CERTIFICATE_SCHEMA_VERSION,
        "evidence_grade": "certificate" if promoted else "diagnostic",
        "evidence_namespace": namespace if promoted else "test_only",
        "scope": scope, "inputs": inputs, "execution": execution,
        "checks": full_checks,
        "claims": {"official_eval_certified": certified,
                   "certification_scope_is_bounded": True,
                   "proves_general_task_success": False,
                   "proves_physical_safety": False},
    }
    jsonschema.validate(cert, OFFICIAL_EVAL_CERTIFICATE_SCHEMA)
    return cert


def verify_official_eval_certificate(cert: dict, *, output_tree_sha256: str | None = None,
                                     require_complete_cases: bool = True) -> tuple[bool, list]:
    """Offline: schema-valid, the entrypoint check matches the recorded argv, the claim
    is consistent with the grade (a diagnostic report forged to true fails; a
    certificate grade must equal the gate), and — if supplied — the recomputed
    output-tree digest matches."""
    import jsonschema
    errors: list[str] = []
    try:
        jsonschema.validate(cert, OFFICIAL_EVAL_CERTIFICATE_SCHEMA)
    except Exception as exc:  # noqa: BLE001
        return False, [f"schema: {exc}"]
    execution, checks, scope = cert["execution"], cert["checks"], cert["scope"]
    claimed = cert["claims"]["official_eval_certified"]
    if checks["official_cli_entrypoint_used"] != _argv_is_official_eval(execution["argv"]):
        errors.append("official_cli_entrypoint_used inconsistent with recorded argv")
    if cert["evidence_grade"] == "diagnostic":
        if claimed:
            errors.append("diagnostic-grade report forged to official_eval_certified=true "
                          "(diagnostic must not certify)")
    else:
        expected = _gate(execution, checks, scope["cases"])
        if claimed != expected:
            errors.append(f"official_eval_certified={claimed} != gate {expected} "
                          "(direct-rollout / hand-built evidence can never certify)")
    if require_complete_cases and claimed and set(scope["cases"]) != set(REQUIRED_CASES):
        errors.append(f"incomplete case set {sorted(scope['cases'])}")
    # a certified certificate must bind the FULL evidence-graph root set, non-null.
    if claimed:
        missing = [k for k in _ROOT_KEYS if not cert["inputs"].get(k)]
        if missing:
            errors.append(f"certified without a complete root graph (null: {missing})")
    if output_tree_sha256 is not None and output_tree_sha256 != execution["output_tree_sha256"]:
        errors.append("recomputed output_tree_sha256 mismatch (output tamper)")
    return (not errors), errors
