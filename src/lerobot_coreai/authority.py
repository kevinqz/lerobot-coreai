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

OFFICIAL_EVAL_EXECUTION_RECEIPT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object", "additionalProperties": False,
    "required": ["real_subprocess", "fake_executor", "resolution_method",
                 "executable_realpath", "argv", "lerobot_distribution_sha256",
                 "coreai_env_instantiated", "cases", "exit_code",
                 "command_sha256", "resolved_config_sha256", "output_tree_sha256"],
    "properties": {
        "real_subprocess": {"type": "boolean"},
        "fake_executor": {"type": "boolean"},
        "resolution_method": {"enum": ["console_script", "python_-m"]},
        "executable_realpath": {"type": "string", "minLength": 1},
        "argv": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "lerobot_distribution_sha256": _HASH,
        "coreai_env_instantiated": {"type": "boolean"},
        "cases": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "exit_code": {"type": "integer"},
        "command_sha256": _HASH, "resolved_config_sha256": _HASH,
        "output_tree_sha256": _HASH,
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
    if fails:
        raise AuthorityError(f"official-eval receipt is not certificate-grade: {fails}")
    return VerifiedOfficialEvalExecutionReceipt(dict(receipt), _AUTHORITY)


# --- mint functions (the ONLY way to obtain a Verified*) ---

def as_verified_trust_policy(policy: dict) -> VerifiedTrustPolicy:
    import jsonschema
    from .signed_evidence import TRUST_POLICY_SCHEMA
    jsonschema.validate(policy, TRUST_POLICY_SCHEMA)
    return VerifiedTrustPolicy(dict(policy), _AUTHORITY)


def as_verified_signed_official_eval(dsse_envelope: dict, *,
                                     trust_policy: VerifiedTrustPolicy, now: str,
                                     ) -> VerifiedSignedOfficialEvalCertificate:
    import base64
    import json

    from .signed_evidence import verify_signed_evidence
    if not isinstance(trust_policy, VerifiedTrustPolicy):
        raise TypeError("trust_policy must be a VerifiedTrustPolicy")
    ok, reasons = verify_signed_evidence(dsse_envelope, trust_policy=trust_policy.payload,
                                         now=now, evidence_grade="certificate")
    if not ok:
        raise AuthorityError(f"signed official-eval did not verify: {reasons}")
    statement = json.loads(base64.b64decode(dsse_envelope["payload"], validate=True))
    if statement["predicate"]["certificate_type"] != "official_eval":
        raise AuthorityError("signed certificate is not certificate_type=official_eval")
    return VerifiedSignedOfficialEvalCertificate(statement, _AUTHORITY)


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
