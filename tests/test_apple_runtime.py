# test_apple_runtime.py — Apple/CoreAI runtime identity + PROMOTION AUTHORITY (v1.4.0
# machinery, closed in v1.3.26.8). Runs on Linux CI: the gate must hold
# apple_runtime_certified=false here. The "all-real" case is SIMULATED via explicit
# identity + verified receipts (no real hardware). The public builder is DIAGNOSTIC
# only — a true claim can be produced ONLY by promote_apple_runtime_certificate from
# unforgeable Verified* receipts, never from caller booleans.

import copy

import pytest

from lerobot_coreai.apple_runtime import (
    APPLE_RUNTIME_IDENTITY_SCHEMA, build_diagnostic_apple_runtime_report,
    capture_apple_runtime_identity, promote_apple_runtime_certificate,
    verify_apple_runtime_certificate,
)
from lerobot_coreai.authority import (
    AuthorityError, as_verified_model_conversion, as_verified_official_trust_policy,
    as_verified_signed_official_eval, verify_coreai_runtime_receipt,
    verify_official_eval_execution_receipt,
)

_H = "sha256:" + "a" * 64
_NOW = "2026-07-11T00:00:00Z"
_ALL_TRUE = {"real_runner_used": True, "real_aimodel_loaded": True,
             "all_required_cases_passed": True, "numeric_parity_passed": True,
             "official_eval_chain_bound": True, "signed_evidence_verified": True}
_CASES = ["single-b1", "native-b2", "native-b4", "split-b2", "split-b4"]


def _apple_identity():
    """A SIMULATED real Apple-Silicon identity (explicit inputs; not this host)."""
    return {
        "schema_version": "lerobot-coreai.apple-runtime-identity.v1",
        "hardware": {"model_identifier": "Mac15,3", "chip": "Apple M3",
                     "memory_bytes": 17179869184, "cpu_arch": "arm64"},
        "software": {"macos_version": "15.5", "macos_build": "24F74",
                     "xcode_version": "16.4", "sdk_version": "macosx15.5",
                     "coreai_runner_version": "coreai-runner.v2",
                     "coreai_runner_binary_sha256": _H},
        "model": {"aimodel_sha256": _H, "aimodel_schema_version": "aimodel.v1",
                  "manifest_sha256": _H},
        "execution": {"compute_units_requested": "cpu_and_ne",
                      "compute_units_reported": "unknown", "process_arch": "arm64"},
    }


def _non_apple_identity():
    """An explicit non-Apple identity (Linux/x86) — deterministic on any machine."""
    idy = _apple_identity()
    idy["hardware"] = {"model_identifier": None, "chip": None, "memory_bytes": None,
                       "cpu_arch": "x86_64"}
    idy["software"]["macos_version"] = None
    idy["execution"]["process_arch"] = "x86_64"
    return idy


# --- helpers that mint REAL Verified* receipts (the only way to promote) ---

def _verified_trust_policy(key):
    from lerobot_coreai.signed_evidence import (
        OFFICIAL_ALLOWED_ISSUERS, OFFICIAL_TRUST_POLICY_ID, TRUST_POLICY_SCHEMA_VERSION,
    )
    policy = {"schema_version": TRUST_POLICY_SCHEMA_VERSION,
              "policy_id": OFFICIAL_TRUST_POLICY_ID,
              "allowed_issuers": list(OFFICIAL_ALLOWED_ISSUERS),
              "trusted_keys": [{"key_id": key["key_id"],
                                "public_key_hex": key["public_key_hex"],
                                "valid_from": None, "valid_until": None, "revoked": False,
                                "dev": False,
                                "allowed_certificate_types": ["official_eval"]}],
              "require_unexpired": True, "minimum_evidence_grade": "certificate",
              "required_claims_false": ["proves_physical_safety"]}
    return as_verified_official_trust_policy(policy)


def _official_eval_certificate():
    """A REAL certificate-grade OfficialEvalCertificate whose artifact_root == the
    certified .aimodel (_H) so the Apple cross-binding holds."""
    from lerobot_coreai.official_eval_certificate import (
        _ROOT_KEYS, REQUIRED_CASES, promote_official_eval_certificate,
    )
    receipt = verify_official_eval_execution_receipt({
        "real_subprocess": True, "fake_executor": False,
        "resolution_method": "console_script",
        "executable_realpath": "/usr/local/bin/lerobot-eval",
        "argv": ["/usr/local/bin/lerobot-eval", "--env.type=coreai_cert_env"],
        "lerobot_distribution_sha256": _H, "coreai_env_instantiated": True,
        "cases": list(REQUIRED_CASES), "exit_code": 0, "command_sha256": _H,
        "resolved_config_sha256": _H, "output_tree_sha256": _H,
        "schema_report": {"outputs_schema_valid": True, "output_manifest_sha256": _H},
        "replay_report": {"evidence_replay_passed": True, "replay_root_sha256": _H},
        "verified_cases_root_sha256": _H})
    return promote_official_eval_certificate(receipt=receipt,
                                             inputs={k: _H for k in _ROOT_KEYS})


def _verified_official_eval(key, vtp):
    from lerobot_coreai.signed_evidence import (
        build_evidence_statement, certificate_root_sha256, sign_statement,
    )
    cert = _official_eval_certificate()
    root = certificate_root_sha256(cert)
    roots = {"artifact_root_sha256": _H, "feature_contract_sha256": _H,
             "dataset_metadata_sha256": _H, "processor_parity_sha256": _H}
    st = build_evidence_statement(certificate_type="official_eval",
                                  certificate_root_sha256=root, roots=roots,
                                  issuer="lerobot-coreai-release-ci", issued_at=_NOW)
    env = sign_statement(st, private_key_hex=key["private_key_hex"], key_id=key["key_id"])
    return as_verified_signed_official_eval(env, certificate=cert, trust_policy=vtp,
                                            now=_NOW)


def _verified_conversion():
    from lerobot_coreai.model_conversion_evidence import build_model_conversion_evidence
    ref = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    cand = copy.deepcopy(ref)                    # exact parity
    ev = build_model_conversion_evidence(
        source={"repository": "org/policy", "revision": "abc123", "weights_sha256": _H},
        exporter={"name": "coreai-exporter", "build": "2.0.0"},
        export_configuration={"opset": 18}, artifact={
            "aimodel_sha256": _H, "aimodel_schema_version": "aimodel.v1",
            "manifest_sha256": _H},
        reference_outputs=ref, candidate_outputs=cand,
        tolerance={"max_abs_error": 0.0})
    return as_verified_model_conversion(ev, reference_outputs=ref, candidate_outputs=cand)


def _runtime_receipt(**overrides):
    receipt = {"real_runner_used": True, "fake_runner": False,
               "runner_binary_sha256": _H, "handshake_nonce": "nonce-12345678",
               "aimodel_opened": True, "aimodel_root_sha256": _H,
               "artifact_aimodel_sha256": _H, "cases": list(_CASES),
               "numeric_parity_passed": True}
    receipt.update(overrides)
    return receipt


def _verified_receipts():
    from lerobot_coreai.signed_evidence import generate_keypair
    key = generate_keypair(dev=False)
    vtp = _verified_trust_policy(key)
    return {
        "runtime_receipt": verify_coreai_runtime_receipt(_runtime_receipt()),
        "official_eval": _verified_official_eval(key, vtp),
        "conversion": _verified_conversion(),
        "trust_policy": vtp,
    }


# --- identity capture ---

def test_captured_identity_is_schema_valid():
    import jsonschema
    idy = capture_apple_runtime_identity(coreai_runner_version="coreai-runner.v2")
    jsonschema.validate(idy, APPLE_RUNTIME_IDENTITY_SCHEMA)


def test_capture_reflects_real_host():
    # P1.7: on macOS the fields are read from the real host (sysctl/sw_vers); on a
    # non-Apple host the Apple-specific fields are null so the gate can never pass.
    import platform
    idy = capture_apple_runtime_identity()
    if platform.system() == "Darwin":
        assert idy["software"]["macos_version"] is not None
        assert idy["hardware"]["chip"] is not None          # sysctl brand string
        assert idy["hardware"]["memory_bytes"] is not None
    else:
        assert idy["software"]["macos_version"] is None
        assert idy["hardware"]["chip"] is None


def test_capture_hashes_real_aimodel_path(tmp_path):
    # P1.7: a real .aimodel PATH is hashed from bytes (no free-arg hash trust).
    p = tmp_path / "model.aimodel"
    p.write_bytes(b"coreai-model-bytes")
    idy = capture_apple_runtime_identity(aimodel_path=str(p))
    import hashlib
    expected = "sha256:" + hashlib.sha256(b"coreai-model-bytes").hexdigest()
    assert idy["model"]["aimodel_sha256"] == expected


# --- diagnostic builder NEVER certifies (P0.2) ---

def test_diagnostic_report_never_certifies_even_with_all_true():
    # the public builder is diagnostic: caller booleans cannot promote the claim.
    idy = _apple_identity()
    cert = build_diagnostic_apple_runtime_report(
        identity=idy, checks=_ALL_TRUE,
        signed_official_eval_certificate_sha256=_H, artifact_root_sha256=_H,
        aimodel_sha256=_H)
    assert cert["claims"]["apple_runtime_certified"] is False
    ok, errs = verify_apple_runtime_certificate(cert, idy)
    assert ok, errs        # correctly-false claim verifies as consistent


# --- promotion authority (the only true-claim path) ---

def test_promote_on_apple_identity_certifies():
    idy = _apple_identity()
    cert = promote_apple_runtime_certificate(identity=idy, **_verified_receipts())
    assert cert["claims"]["apple_runtime_certified"] is True
    ok, errs = verify_apple_runtime_certificate(cert, idy)
    assert ok, errs


def test_promote_on_non_apple_identity_stays_false():
    # even with perfect verified receipts, a non-arm64/non-macOS host cannot certify.
    idy = _non_apple_identity()
    cert = promote_apple_runtime_certificate(identity=idy, **_verified_receipts())
    assert cert["claims"]["apple_runtime_certified"] is False
    ok, errs = verify_apple_runtime_certificate(cert, idy)
    assert ok, errs


@pytest.mark.parametrize("field,name", [
    ("runtime_receipt", "runtime_receipt"), ("official_eval", "official_eval"),
    ("conversion", "conversion"), ("trust_policy", "trust_policy")])
def test_promote_rejects_plain_dict_or_bool(field, name):
    # a hand-built dict / bool for ANY slot is refused at the type boundary.
    idy = _apple_identity()
    receipts = _verified_receipts()
    receipts[field] = {"apple_runtime_certified": True}     # forged JSON
    with pytest.raises(TypeError):
        promote_apple_runtime_certificate(identity=idy, **receipts)
    receipts[field] = True                                  # forged bool
    with pytest.raises(TypeError):
        promote_apple_runtime_certificate(identity=idy, **receipts)


# --- unforgeable Verified* types ---

def test_verified_types_cannot_be_hand_constructed():
    from lerobot_coreai.authority import VerifiedCoreAIRuntimeReceipt
    with pytest.raises(AuthorityError):
        VerifiedCoreAIRuntimeReceipt({"real_runner_used": True})   # no token


def test_runtime_receipt_with_fake_runner_is_refused():
    with pytest.raises(AuthorityError):
        verify_coreai_runtime_receipt(_runtime_receipt(fake_runner=True))


def test_runtime_receipt_with_incomplete_matrix_is_refused():
    with pytest.raises(AuthorityError):
        verify_coreai_runtime_receipt(_runtime_receipt(cases=["single-b1"]))


def test_runtime_receipt_aimodel_root_mismatch_is_refused():
    with pytest.raises(AuthorityError):
        verify_coreai_runtime_receipt(
            _runtime_receipt(aimodel_root_sha256="sha256:" + "b" * 64))


# --- verifier-side integrity (unchanged threat model) ---

def test_forged_true_claim_detected():
    idy = _non_apple_identity()                 # gate can never be true here
    cert = build_diagnostic_apple_runtime_report(identity=idy, checks=_ALL_TRUE)
    assert cert["claims"]["apple_runtime_certified"] is False
    cert["claims"]["apple_runtime_certified"] = True    # forge it
    ok, errs = verify_apple_runtime_certificate(cert, idy)
    assert not ok and any("forged" in e or "gate" in e for e in errs)


def test_identity_tamper_detected():
    idy = _apple_identity()
    cert = promote_apple_runtime_certificate(identity=idy, **_verified_receipts())
    tampered = copy.deepcopy(idy); tampered["hardware"]["chip"] = "Apple M4"
    ok, errs = verify_apple_runtime_certificate(cert, tampered)
    assert not ok and any("identity_sha256" in e for e in errs)


def test_certified_requires_signed_official_eval_chain():
    idy = _apple_identity()
    cert = promote_apple_runtime_certificate(identity=idy, **_verified_receipts())
    cert["signed_official_eval_certificate_sha256"] = None      # strip the bound chain
    ok, errs = verify_apple_runtime_certificate(cert, idy)
    assert not ok and any("signed official-eval" in e for e in errs)


def test_physical_safety_and_task_success_always_false():
    idy = _apple_identity()
    cert = promote_apple_runtime_certificate(identity=idy, **_verified_receipts())
    assert cert["claims"]["proves_physical_safety"] is False
    assert cert["claims"]["proves_real_robot_task_success"] is False
    assert cert["claims"]["proves_general_apple_compatibility"] is False


# --- v1.3.26.11 provenance closures ---

def test_apple_cross_binding_mismatch_refused():
    # the runner executed a DIFFERENT .aimodel than the conversion/identity/official-eval
    # bound → provenance contradiction → promotion refused (P0.5).
    idy = _apple_identity()
    idy["model"]["aimodel_sha256"] = "sha256:" + "d" * 64      # different artifact
    with pytest.raises(AuthorityError):
        promote_apple_runtime_certificate(identity=idy, **_verified_receipts())


def test_official_anchor_rejects_dev_key():
    from lerobot_coreai.signed_evidence import (
        OFFICIAL_ALLOWED_ISSUERS, OFFICIAL_TRUST_POLICY_ID, TRUST_POLICY_SCHEMA_VERSION,
        generate_keypair,
    )
    key = generate_keypair(dev=True)
    policy = {"schema_version": TRUST_POLICY_SCHEMA_VERSION,
              "policy_id": OFFICIAL_TRUST_POLICY_ID,
              "allowed_issuers": list(OFFICIAL_ALLOWED_ISSUERS),
              "trusted_keys": [{"key_id": key["key_id"],
                                "public_key_hex": key["public_key_hex"],
                                "valid_from": None, "valid_until": None, "revoked": False,
                                "dev": True,        # a dev key
                                "allowed_certificate_types": ["official_eval"]}],
              "require_unexpired": True, "minimum_evidence_grade": "certificate",
              "required_claims_false": ["proves_physical_safety"]}
    with pytest.raises(AuthorityError):
        as_verified_official_trust_policy(policy)


def test_official_anchor_rejects_wrong_policy_id():
    from lerobot_coreai.signed_evidence import (
        OFFICIAL_ALLOWED_ISSUERS, TRUST_POLICY_SCHEMA_VERSION, generate_keypair,
    )
    key = generate_keypair(dev=False)
    policy = {"schema_version": TRUST_POLICY_SCHEMA_VERSION,
              "policy_id": "attacker-self-signed",       # not the pinned anchor
              "allowed_issuers": list(OFFICIAL_ALLOWED_ISSUERS),
              "trusted_keys": [{"key_id": key["key_id"],
                                "public_key_hex": key["public_key_hex"],
                                "valid_from": None, "valid_until": None, "revoked": False,
                                "dev": False,
                                "allowed_certificate_types": ["official_eval"]}],
              "require_unexpired": True, "minimum_evidence_grade": "certificate",
              "required_claims_false": ["proves_physical_safety"]}
    with pytest.raises(AuthorityError):
        as_verified_official_trust_policy(policy)


def test_signed_official_eval_requires_matching_certificate_bytes():
    from lerobot_coreai.signed_evidence import (
        build_evidence_statement, sign_statement,
    )
    from lerobot_coreai.signed_evidence import generate_keypair
    key = generate_keypair(dev=False)
    vtp = _verified_trust_policy(key)
    cert = _official_eval_certificate()
    # sign a statement whose root does NOT match the certificate bytes.
    st = build_evidence_statement(
        certificate_type="official_eval", certificate_root_sha256="sha256:" + "e" * 64,
        roots={"artifact_root_sha256": _H, "feature_contract_sha256": _H,
               "dataset_metadata_sha256": _H, "processor_parity_sha256": _H},
        issuer="lerobot-coreai-release-ci", issued_at=_NOW)
    env = sign_statement(st, private_key_hex=key["private_key_hex"], key_id=key["key_id"])
    with pytest.raises(AuthorityError):
        as_verified_signed_official_eval(env, certificate=cert, trust_policy=vtp, now=_NOW)


def test_signed_official_eval_rejects_uncertified_certificate():
    from lerobot_coreai.official_eval_certificate import (
        build_diagnostic_official_eval_report,
    )
    from lerobot_coreai.signed_evidence import (
        build_evidence_statement, certificate_root_sha256, generate_keypair,
        sign_statement,
    )
    key = generate_keypair(dev=False)
    vtp = _verified_trust_policy(key)
    diag = build_diagnostic_official_eval_report(
        scope={"lerobot_version": None, "lerobot_distribution_sha256": _H,
               "environment": "coreai_cert_env", "cases": []},
        inputs={"artifact_root_sha256": _H}, checks={},
        execution={"argv": ["pytest"], "command_sha256": _H,
                   "resolved_config_sha256": _H, "output_tree_sha256": _H,
                   "exit_code": 0})
    st = build_evidence_statement(
        certificate_type="official_eval",
        certificate_root_sha256=certificate_root_sha256(diag),
        roots={"artifact_root_sha256": _H, "feature_contract_sha256": _H,
               "dataset_metadata_sha256": _H, "processor_parity_sha256": _H},
        issuer="lerobot-coreai-release-ci", issued_at=_NOW)
    env = sign_statement(st, private_key_hex=key["private_key_hex"], key_id=key["key_id"])
    with pytest.raises(AuthorityError):        # diagnostic cert never a signed authority
        as_verified_signed_official_eval(env, certificate=diag, trust_policy=vtp, now=_NOW)
