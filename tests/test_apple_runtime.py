# test_apple_runtime.py — Apple/CoreAI runtime identity + certificate gate (v1.4.0
# machinery). Runs on Linux CI: the gate must hold apple_runtime_certified=false here.
# The "all-real" case is SIMULATED via explicit identity inputs (no real hardware).

import copy
import platform

import pytest

from lerobot_coreai.apple_runtime import (
    APPLE_RUNTIME_CERTIFICATE_SCHEMA, APPLE_RUNTIME_IDENTITY_SCHEMA,
    build_apple_runtime_certificate, capture_apple_runtime_identity, identity_sha256,
    verify_apple_runtime_certificate,
)

_H = "sha256:" + "a" * 64
_ALL_TRUE = {"real_runner_used": True, "real_aimodel_loaded": True,
             "all_required_cases_passed": True, "numeric_parity_passed": True,
             "official_eval_chain_bound": True, "signed_evidence_verified": True}


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
    """An explicit non-Apple identity (Linux/x86) — deterministic regardless of the
    machine the test runs on."""
    idy = _apple_identity()
    idy["hardware"] = {"model_identifier": None, "chip": None, "memory_bytes": None,
                       "cpu_arch": "x86_64"}
    idy["software"]["macos_version"] = None
    idy["execution"]["process_arch"] = "x86_64"
    return idy


def test_captured_identity_is_schema_valid():
    import jsonschema
    idy = capture_apple_runtime_identity(coreai_runner_version="coreai-runner.v2")
    jsonschema.validate(idy, APPLE_RUNTIME_IDENTITY_SCHEMA)


def test_non_apple_host_cannot_certify():
    # on a non-Apple host, even all-true checks cannot certify (macos None / x86).
    idy = _non_apple_identity()
    cert = build_apple_runtime_certificate(
        identity=idy, checks=_ALL_TRUE,
        signed_official_eval_certificate_sha256=_H, artifact_root_sha256=_H)
    assert cert["claims"]["apple_runtime_certified"] is False
    ok, errs = verify_apple_runtime_certificate(cert, idy)
    assert ok, errs        # correctly-false claim verifies as consistent


def test_simulated_apple_host_certifies():
    idy = _apple_identity()
    cert = build_apple_runtime_certificate(
        identity=idy, checks=_ALL_TRUE,
        signed_official_eval_certificate_sha256=_H, artifact_root_sha256=_H,
        aimodel_sha256=_H)
    assert cert["claims"]["apple_runtime_certified"] is True
    ok, errs = verify_apple_runtime_certificate(cert, idy)
    assert ok, errs


@pytest.mark.parametrize("missing", list(_ALL_TRUE))
def test_any_missing_check_blocks_certification(missing):
    idy = _apple_identity()
    checks = {**_ALL_TRUE, missing: False}      # e.g. fake runner, no aimodel, ...
    cert = build_apple_runtime_certificate(
        identity=idy, checks=checks, signed_official_eval_certificate_sha256=_H)
    assert cert["claims"]["apple_runtime_certified"] is False


def test_x86_process_blocks_certification():
    # Rosetta / x86 process on Apple hardware must not certify.
    idy = _apple_identity(); idy["execution"]["process_arch"] = "x86_64"
    cert = build_apple_runtime_certificate(
        identity=idy, checks=_ALL_TRUE, signed_official_eval_certificate_sha256=_H)
    assert cert["claims"]["apple_runtime_certified"] is False


def test_forged_true_claim_detected():
    idy = _non_apple_identity()                 # gate can never be true here
    cert = build_apple_runtime_certificate(identity=idy, checks=_ALL_TRUE)
    assert cert["claims"]["apple_runtime_certified"] is False
    cert["claims"]["apple_runtime_certified"] = True    # forge it
    ok, errs = verify_apple_runtime_certificate(cert, idy)
    assert not ok and any("forged" in e or "gate" in e for e in errs)


def test_identity_tamper_detected():
    idy = _apple_identity()
    cert = build_apple_runtime_certificate(
        identity=idy, checks=_ALL_TRUE, signed_official_eval_certificate_sha256=_H)
    tampered = copy.deepcopy(idy); tampered["hardware"]["chip"] = "Apple M4"
    ok, errs = verify_apple_runtime_certificate(cert, tampered)
    assert not ok and any("identity_sha256" in e for e in errs)


def test_certified_requires_signed_official_eval_chain():
    idy = _apple_identity()
    cert = build_apple_runtime_certificate(
        identity=idy, checks=_ALL_TRUE,
        signed_official_eval_certificate_sha256=None)     # gate true but chain unbound
    # gate passes on identity+checks, but verify requires the signed chain when certified.
    ok, errs = verify_apple_runtime_certificate(cert, idy)
    assert not ok and any("signed official-eval" in e for e in errs)


def test_physical_safety_and_task_success_always_false():
    idy = _apple_identity()
    cert = build_apple_runtime_certificate(
        identity=idy, checks=_ALL_TRUE, signed_official_eval_certificate_sha256=_H)
    assert cert["claims"]["proves_physical_safety"] is False
    assert cert["claims"]["proves_real_robot_task_success"] is False
    assert cert["claims"]["proves_general_apple_compatibility"] is False
