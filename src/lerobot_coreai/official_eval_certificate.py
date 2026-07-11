# official_eval_certificate.py — OfficialEvalCertificate v1 (v1.3.27 core).
#
# Certify the command users actually run — `lerobot-eval` — not merely the internal
# function it eventually calls. The critical gate: official_eval_certified can be true
# ONLY when the OFFICIAL CLI entrypoint was used (argv invokes lerobot-eval), the
# third-party plugin registration resolved the policy, every required case passed with
# a clean exit, the outputs are schema-valid, and semantic replay passed. Direct
# rollout() evidence (no CLI argv) can NEVER set the claim. Pure Python; offline.

from __future__ import annotations

from .rollout_evidence_schema import canonical_json_sha256

OFFICIAL_EVAL_CERTIFICATE_SCHEMA_VERSION = "lerobot-coreai.official-eval-certificate.v1"
_SHA256 = {"type": "string", "pattern": r"^sha256:[0-9a-f]{64}$"}
_SHA256_OR_NULL = {"anyOf": [_SHA256, {"type": "null"}]}
REQUIRED_CASES = ("single-b1", "native-b2", "native-b4", "split-b2", "split-b4")
_CHECK_KEYS = ("official_cli_entrypoint_used", "third_party_plugin_registration_used",
               "all_required_cases_passed", "outputs_schema_valid",
               "evidence_replay_passed")

OFFICIAL_EVAL_CERTIFICATE_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object", "additionalProperties": False,
    "required": ["schema_version", "scope", "inputs", "execution", "checks", "claims"],
    "properties": {
        "schema_version": {"const": OFFICIAL_EVAL_CERTIFICATE_SCHEMA_VERSION},
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
            "properties": {"artifact_root_sha256": _SHA256_OR_NULL,
                           "feature_contract_sha256": _SHA256_OR_NULL,
                           "dataset_metadata_sha256": _SHA256_OR_NULL,
                           "processor_parity_sha256": _SHA256_OR_NULL}},
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


def build_official_eval_certificate(*, scope: dict, inputs: dict, execution: dict,
                                    checks: dict) -> dict:
    """Assemble the certificate. official_eval_certified is DERIVED from the gate —
    a direct rollout() run (argv not lerobot-eval) can never certify."""
    import jsonschema
    full_checks = {k: bool(checks.get(k, False)) for k in _CHECK_KEYS}
    # the entrypoint check is itself pinned to the recorded argv (no free assertion).
    full_checks["official_cli_entrypoint_used"] = _argv_is_official_eval(
        execution.get("argv", []))
    certified = _gate(execution, full_checks, scope.get("cases", []))
    cert = {
        "schema_version": OFFICIAL_EVAL_CERTIFICATE_SCHEMA_VERSION,
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
    """Offline: schema-valid, the entrypoint check matches the recorded argv, the gate
    is consistent (a forged official_eval_certified fails), and — if supplied — the
    recomputed output-tree digest matches."""
    import jsonschema
    errors: list[str] = []
    try:
        jsonschema.validate(cert, OFFICIAL_EVAL_CERTIFICATE_SCHEMA)
    except Exception as exc:  # noqa: BLE001
        return False, [f"schema: {exc}"]
    execution, checks, scope = cert["execution"], cert["checks"], cert["scope"]
    if checks["official_cli_entrypoint_used"] != _argv_is_official_eval(execution["argv"]):
        errors.append("official_cli_entrypoint_used inconsistent with recorded argv")
    expected = _gate(execution, checks, scope["cases"])
    if cert["claims"]["official_eval_certified"] != expected:
        errors.append(f"official_eval_certified={cert['claims']['official_eval_certified']} "
                      f"!= gate {expected} (direct-rollout evidence can never certify)")
    if require_complete_cases and set(scope["cases"]) != set(REQUIRED_CASES):
        errors.append(f"incomplete case set {sorted(scope['cases'])}")
    if output_tree_sha256 is not None and output_tree_sha256 != execution["output_tree_sha256"]:
        errors.append("recomputed output_tree_sha256 mismatch (output tamper)")
    return (not errors), errors
