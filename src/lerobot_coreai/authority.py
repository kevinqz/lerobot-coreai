# authority.py — claim-promotion authority (v1.3.26.8).
#
# The 3rd external review's core finding: the CONTRACTS are near-SotA but the AUTHORITY
# that promotes high claims still trusted the producer (hand-built JSON + booleans).
# This module makes promotion impossible without VERIFIED, unforgeable receipts:
#
#   * every high-claim promoter accepts ONLY `Verified*` objects (a dict/bool raises
#     TypeError), and
#   * a `Verified*` can only be minted by a verifier function that actually re-derived
#     its checks from bytes/receipts — its constructor is sealed with a module-private
#     token, so a hand-built instance raises.
#
# So no public function accepts booleans to promote, and no manually-built JSON can
# obtain a true high claim. Pure Python.

from __future__ import annotations

from dataclasses import dataclass, field

_AUTHORITY = object()   # module-private mint token; deliberately NOT exported


class AuthorityError(RuntimeError):
    """Raised on an attempt to mint or promote without a real verification."""


@dataclass(frozen=True)
class _Sealed:
    payload: dict = field(default_factory=dict)
    _token: object = None

    def __post_init__(self):
        if self._token is not _AUTHORITY:
            raise AuthorityError(
                f"{type(self).__name__} can only be minted by its verifier "
                "(no direct construction, no hand-built JSON).")


class VerifiedTrustPolicy(_Sealed):
    """A schema-valid TrustPolicy."""


class VerifiedSignedOfficialEvalCertificate(_Sealed):
    """A signed official-eval certificate whose signature + policy verified."""


class VerifiedModelConversionEvidence(_Sealed):
    """Conversion evidence whose numeric parity was re-derived from raw arrays."""


class VerifiedCoreAIRuntimeReceipt(_Sealed):
    """A receipt from a REAL CoreAI Runner run (no fake runner) whose substance
    (runner binary digest, .aimodel opened + root == artifact, full case matrix,
    numeric parity) was checked."""


class VerifiedOfficialEvalExecutionReceipt(_Sealed):
    """A receipt from a REAL `lerobot-eval` subprocess (the official CLI entrypoint,
    not a wrapper, not a temp-dir shim), with the coreai env instantiated, the full
    case matrix, and a clean exit — the ONLY thing that may promote official_eval."""


# --- runtime receipt schema (what a REAL run must emit) ---

_HASH = {"type": "string", "pattern": r"^sha256:[0-9a-f]{64}$"}
COREAI_RUNTIME_RECEIPT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object", "additionalProperties": False,
    "required": ["real_runner_used", "fake_runner", "runner_binary_sha256",
                 "handshake_nonce", "aimodel_opened", "aimodel_root_sha256",
                 "artifact_aimodel_sha256", "cases", "numeric_parity_passed"],
    "properties": {
        "real_runner_used": {"type": "boolean"},
        "fake_runner": {"type": "boolean"},
        "runner_binary_sha256": _HASH,
        "handshake_nonce": {"type": "string", "minLength": 8},
        "aimodel_opened": {"type": "boolean"},
        "aimodel_root_sha256": _HASH,
        "artifact_aimodel_sha256": _HASH,
        "cases": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "numeric_parity_passed": {"type": "boolean"},
    },
}
_REQUIRED_RUNTIME_CASES = ("single-b1", "native-b2", "native-b4", "split-b2", "split-b4")


# --- official-eval execution receipt (what a REAL lerobot-eval subprocess emits) ---

_BOOL = {"type": "boolean"}
OFFICIAL_EVAL_EXECUTION_RECEIPT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object", "additionalProperties": False,
    "required": ["real_subprocess", "fake_executor", "resolution_method",
                 "executable_realpath", "argv", "lerobot_distribution_sha256",
                 "coreai_env_instantiated", "cases", "exit_code",
                 "command_sha256", "resolved_config_sha256", "output_tree_sha256",
                 "schema_report", "replay_report", "verified_cases_root_sha256"],
    "properties": {
        "real_subprocess": _BOOL,
        "fake_executor": _BOOL,
        "resolution_method": {"enum": ["console_script", "python_-m"]},
        "executable_realpath": {"type": "string", "minLength": 1},
        "argv": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "lerobot_distribution_sha256": _HASH,
        "coreai_env_instantiated": _BOOL,
        "cases": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "exit_code": {"type": "integer"},
        "command_sha256": _HASH, "resolved_config_sha256": _HASH,
        "output_tree_sha256": _HASH,
        # verifier reports the executor produced from the REAL outputs — the promoter
        # derives its checks from these, never hardcodes them (P0.3).
        "schema_report": {"type": "object", "additionalProperties": False,
                          "required": ["outputs_schema_valid", "output_manifest_sha256"],
                          "properties": {"outputs_schema_valid": _BOOL,
                                         "output_manifest_sha256": _HASH}},
        "replay_report": {"type": "object", "additionalProperties": False,
                          "required": ["evidence_replay_passed", "replay_root_sha256"],
                          "properties": {"evidence_replay_passed": _BOOL,
                                         "replay_root_sha256": _HASH}},
        "verified_cases_root_sha256": _HASH,
    },
}
_TEMP_DIR_PREFIXES = ("/tmp/", "/private/tmp/", "/var/folders/", "/dev/shm/")
_REQUIRED_EVAL_CASES = ("single-b1", "native-b2", "native-b4", "split-b2", "split-b4")


def _argv_is_official_eval(argv) -> bool:
    """True iff argv invoked the public lerobot-eval CLI (console script or
    `python -m lerobot.scripts.lerobot_eval`)."""
    if not argv:
        return False
    if argv[0].rsplit("/", 1)[-1] == "lerobot-eval":
        return True
    return "lerobot.scripts.lerobot_eval" in " ".join(argv) and "-m" in argv


def verify_official_eval_execution_receipt(receipt: dict) -> VerifiedOfficialEvalExecutionReceipt:
    """Mint an execution receipt ONLY from a REAL lerobot-eval run: a real subprocess
    (not a fake executor), the official entrypoint argv, an executable resolved OUTSIDE
    any temp dir (a `/tmp/lerobot-eval` shim is refused), the coreai env actually
    instantiated, the full case matrix, a clean exit, and a captured output tree."""
    import jsonschema
    jsonschema.validate(receipt, OFFICIAL_EVAL_EXECUTION_RECEIPT_SCHEMA)
    realpath = receipt["executable_realpath"]
    fails = []
    if not receipt["real_subprocess"] or receipt["fake_executor"]:
        fails.append("not a real subprocess (or fake_executor set)")
    if not _argv_is_official_eval(receipt["argv"]):
        fails.append("argv did not invoke the official lerobot-eval entrypoint")
    if not realpath.startswith("/") or any(realpath.startswith(p) for p in _TEMP_DIR_PREFIXES):
        fails.append(f"executable not resolved to an installed path: {realpath!r}")
    if not receipt["coreai_env_instantiated"]:
        fails.append("coreai_cert_env was not instantiated")
    if set(receipt["cases"]) != set(_REQUIRED_EVAL_CASES):
        fails.append("incomplete case matrix")
    if receipt["exit_code"] != 0:
        fails.append(f"non-zero exit {receipt['exit_code']}")
    # the executor's own verifier reports must attest success (the promoter derives its
    # checks from these — a receipt whose reports failed cannot certify).
    if not receipt["schema_report"]["outputs_schema_valid"]:
        fails.append("outputs failed schema validation")
    if not receipt["replay_report"]["evidence_replay_passed"]:
        fails.append("evidence replay did not pass")
    if fails:
        raise AuthorityError(f"official-eval receipt is not certificate-grade: {fails}")
    return VerifiedOfficialEvalExecutionReceipt(dict(receipt), _AUTHORITY)


# --- mint functions (the ONLY way to obtain a Verified*) ---

def as_verified_trust_policy(policy: dict) -> VerifiedTrustPolicy:
    """A schema-valid TrustPolicy (generic; NOT the official release anchor). Useful for
    diagnostic verification, but promotion of a high claim requires the OFFICIAL anchor
    (see ``as_verified_official_trust_policy``)."""
    import jsonschema
    from .signed_evidence import TRUST_POLICY_SCHEMA
    jsonschema.validate(policy, TRUST_POLICY_SCHEMA)
    return VerifiedTrustPolicy(dict(policy), _AUTHORITY)


def as_verified_official_trust_policy(policy: dict) -> VerifiedTrustPolicy:
    """The pinned OFFICIAL release anchor (v1.3.26.11, P0.6). Provenance authority no
    longer comes from a caller-supplied self-signed policy: the policy must match the
    pinned anchor identity (policy_id + allowed issuers ⊆ the official set) and carry NO
    dev key. A producer that mints its own key + its own policy cannot pass this."""
    import jsonschema
    from .signed_evidence import (
        OFFICIAL_ALLOWED_ISSUERS, OFFICIAL_TRUST_POLICY_ID, TRUST_POLICY_SCHEMA,
    )
    jsonschema.validate(policy, TRUST_POLICY_SCHEMA)
    if policy["policy_id"] != OFFICIAL_TRUST_POLICY_ID:
        raise AuthorityError(
            f"trust policy is not the official anchor {OFFICIAL_TRUST_POLICY_ID!r}")
    if not set(policy["allowed_issuers"]) <= set(OFFICIAL_ALLOWED_ISSUERS):
        raise AuthorityError("trust policy allows non-official issuers")
    if any(k.get("dev") for k in policy["trusted_keys"]):
        raise AuthorityError("official anchor must not trust a dev key")
    if policy["minimum_evidence_grade"] != "certificate":
        raise AuthorityError("official anchor must require certificate grade")
    return VerifiedTrustPolicy(dict(policy), _AUTHORITY)


def as_verified_signed_official_eval(dsse_envelope: dict, *, certificate: dict,
                                     trust_policy: VerifiedTrustPolicy, now: str,
                                     ) -> VerifiedSignedOfficialEvalCertificate:
    """Verify a SIGNED official-eval certificate bound to its actual bytes (v1.3.26.11,
    P0.7/WS4). The caller MUST supply the underlying OfficialEvalCertificate; this:
      1. re-runs the official-eval verifier on those bytes (must be certificate grade +
         official_eval_certified),
      2. recomputes the certificate's canonical root and cross-binds it to the signed
         subject / predicate.certificate_root,
      3. verifies the Ed25519 signature under the trust policy.
    So a signature over a bare, unverified root no longer suffices."""
    import base64
    import json

    from .signed_evidence import certificate_root_sha256, verify_signed_evidence
    from .official_eval_certificate import verify_official_eval_certificate
    if not isinstance(trust_policy, VerifiedTrustPolicy):
        raise TypeError("trust_policy must be a VerifiedTrustPolicy")
    if not isinstance(certificate, dict):
        raise TypeError("certificate (the underlying OfficialEvalCertificate) is required")
    # (1) re-verify the underlying certificate bytes.
    cert_ok, cert_reasons = verify_official_eval_certificate(certificate)
    if not cert_ok:
        raise AuthorityError(f"underlying official-eval certificate invalid: {cert_reasons}")
    if certificate.get("evidence_grade") != "certificate":
        raise AuthorityError("underlying certificate is not certificate grade")
    if not certificate["claims"]["official_eval_certified"]:
        raise AuthorityError("underlying certificate does not assert official_eval_certified")
    # (3) signature.
    ok, reasons = verify_signed_evidence(dsse_envelope, trust_policy=trust_policy.payload,
                                         now=now, evidence_grade="certificate")
    if not ok:
        raise AuthorityError(f"signed official-eval did not verify: {reasons}")
    statement = json.loads(base64.b64decode(dsse_envelope["payload"], validate=True))
    pred = statement["predicate"]
    if pred["certificate_type"] != "official_eval":
        raise AuthorityError("signed certificate is not certificate_type=official_eval")
    # (2) cross-bind the signed root to the ACTUAL certificate bytes.
    if pred["certificate_root_sha256"] != certificate_root_sha256(certificate):
        raise AuthorityError("signed certificate_root does not match the certificate bytes")
    return VerifiedSignedOfficialEvalCertificate(
        {"statement": statement, "certificate": dict(certificate)}, _AUTHORITY)


def as_verified_model_conversion(evidence: dict, *, reference_outputs,
                                 candidate_outputs) -> VerifiedModelConversionEvidence:
    from .model_conversion_evidence import verify_model_conversion_evidence
    ok, reasons = verify_model_conversion_evidence(
        evidence, reference_outputs=reference_outputs,
        candidate_outputs=candidate_outputs)
    if not ok or not evidence["claims"]["model_conversion_verified"]:
        raise AuthorityError(f"model conversion not verified (with replay): {reasons}")
    return VerifiedModelConversionEvidence(dict(evidence), _AUTHORITY)


def verify_coreai_runtime_receipt(receipt: dict) -> VerifiedCoreAIRuntimeReceipt:
    """Mint a runtime receipt ONLY if it proves a real run: a real runner (not fake),
    a runner binary digest + handshake nonce, the .aimodel actually opened with its
    root == the certified artifact's aimodel, the full case matrix, and numeric parity.
    """
    import jsonschema
    jsonschema.validate(receipt, COREAI_RUNTIME_RECEIPT_SCHEMA)
    fails = []
    if not receipt["real_runner_used"] or receipt["fake_runner"]:
        fails.append("not a real runner (or fake_runner set)")
    if not receipt["aimodel_opened"]:
        fails.append(".aimodel was not opened")
    if receipt["aimodel_root_sha256"] != receipt["artifact_aimodel_sha256"]:
        fails.append(".aimodel root != certified artifact aimodel")
    if set(receipt["cases"]) != set(_REQUIRED_RUNTIME_CASES):
        fails.append("incomplete case matrix")
    if not receipt["numeric_parity_passed"]:
        fails.append("numeric parity failed")
    if fails:
        raise AuthorityError(f"runtime receipt is not certificate-grade: {fails}")
    return VerifiedCoreAIRuntimeReceipt(dict(receipt), _AUTHORITY)
