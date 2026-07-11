# test_signed_evidence.py — Signed Evidence Certificate (v1.3.28 core). Covers the
# external review's signature threat model. Pure base + cryptography (Ed25519).

import base64
import copy

import pytest

from lerobot_coreai.signed_evidence import (
    CERTIFICATE_TYPES, TRUST_POLICY_SCHEMA_VERSION, build_evidence_statement,
    generate_keypair, sign_statement, verify_signed_evidence,
)

_NOW = "2026-07-11T00:00:00Z"
_H = "sha256:" + "a" * 64


def _roots():
    return {"matrix_root_sha256": _H, "artifact_root_sha256": _H,
            "feature_contract_sha256": _H, "dataset_metadata_sha256": _H,
            "processor_parity_sha256": _H}


def _statement(certificate_type="official_eval", expires_at=None):
    return build_evidence_statement(
        certificate_type=certificate_type, certificate_root_sha256=_H, roots=_roots(),
        issuer="lerobot-coreai-release-ci", issued_at=_NOW, expires_at=expires_at)


def _trust_policy(key, *, issuers=("lerobot-coreai-release-ci",),
                  allowed_types=("official_eval", "apple_runtime"),
                  valid_until=None, revoked=False):
    return {"schema_version": TRUST_POLICY_SCHEMA_VERSION, "policy_id": "official-release",
            "allowed_issuers": list(issuers),
            "trusted_keys": [{"key_id": key["key_id"],
                              "public_key_hex": key["public_key_hex"],
                              "valid_from": None, "valid_until": valid_until,
                              "revoked": revoked,
                              "allowed_certificate_types": list(allowed_types)}],
            "require_unexpired": True, "minimum_evidence_grade": "certificate",
            "required_claims_false": ["proves_physical_safety",
                                      "proves_general_task_success"]}


def _sign(statement=None, key=None):
    key = key or generate_keypair(dev=False)
    st = statement or _statement()
    env = sign_statement(st, private_key_hex=key["private_key_hex"], key_id=key["key_id"])
    return env, key


def test_valid_signature_verifies():
    env, key = _sign()
    ok, reasons = verify_signed_evidence(env, trust_policy=_trust_policy(key), now=_NOW)
    assert ok, reasons


def test_payload_byte_tamper_fails():
    env, key = _sign()
    st = _statement(); st["predicate"]["roots"]["matrix_root_sha256"] = "sha256:" + "b" * 64
    import json
    env["payload"] = base64.b64encode(
        json.dumps(st, sort_keys=True, separators=(",", ":")).encode()).decode()
    ok, reasons = verify_signed_evidence(env, trust_policy=_trust_policy(key), now=_NOW)
    assert not ok and any("signature" in r for r in reasons)


def test_wrong_key_fails():
    env, _ = _sign()
    other = generate_keypair(dev=False)
    ok, _ = verify_signed_evidence(env, trust_policy=_trust_policy(other), now=_NOW)
    assert not ok


def test_untrusted_issuer_fails():
    env, key = _sign()
    tp = _trust_policy(key, issuers=("some-other-issuer",))
    ok, reasons = verify_signed_evidence(env, trust_policy=tp, now=_NOW)
    assert not ok and any("issuer" in r for r in reasons)


def test_expired_key_fails():
    env, key = _sign()
    tp = _trust_policy(key, valid_until="2026-01-01T00:00:00Z")
    ok, reasons = verify_signed_evidence(env, trust_policy=tp, now=_NOW)
    assert not ok and any("expired" in r for r in reasons)


def test_revoked_key_fails():
    env, key = _sign()
    ok, reasons = verify_signed_evidence(env, trust_policy=_trust_policy(key, revoked=True),
                                         now=_NOW)
    assert not ok and any("revoked" in r for r in reasons)


def test_certificate_replay_after_expiry_fails():
    env, key = _sign(_statement(expires_at="2026-06-01T00:00:00Z"))
    ok, reasons = verify_signed_evidence(env, trust_policy=_trust_policy(key), now=_NOW)
    assert not ok and any("replay" in r or "expired" in r for r in reasons)


def test_certificate_type_not_allowed_fails():
    env, key = _sign(_statement(certificate_type="feature_contract"))
    tp = _trust_policy(key, allowed_types=("official_eval",))   # feature_contract not allowed
    ok, reasons = verify_signed_evidence(env, trust_policy=tp, now=_NOW)
    assert not ok and any("certificate_type" in r for r in reasons)


def test_diagnostic_grade_rejected_by_certificate_policy():
    env, key = _sign()
    ok, reasons = verify_signed_evidence(env, trust_policy=_trust_policy(key), now=_NOW,
                                         evidence_grade="diagnostic")
    assert not ok and any("grade" in r for r in reasons)


def test_algorithm_type_confusion_fails():
    env, key = _sign()
    env["payloadType"] = "application/x-evil"
    ok, reasons = verify_signed_evidence(env, trust_policy=_trust_policy(key), now=_NOW)
    assert not ok and any("payloadType" in r for r in reasons)


def test_trust_policy_tamper_fails_schema():
    env, key = _sign()
    tp = _trust_policy(key); del tp["allowed_issuers"]
    ok, reasons = verify_signed_evidence(env, trust_policy=tp, now=_NOW)
    assert not ok and any("schema" in r for r in reasons)


def test_private_key_never_in_envelope():
    env, key = _sign()
    blob = str(env)
    assert key["private_key_hex"] not in blob


def test_signature_asserts_no_forbidden_claim():
    # the statement pins asserts_task_success / asserts_physical_safety to False; a
    # forged True must be rejected even with an otherwise-valid signature.
    key = generate_keypair(dev=False)
    st = _statement(); st["predicate"]["asserts_physical_safety"] = True
    env = sign_statement(st, private_key_hex=key["private_key_hex"], key_id=key["key_id"])
    ok, reasons = verify_signed_evidence(env, trust_policy=_trust_policy(key), now=_NOW)
    assert not ok and any("forbidden claim" in r for r in reasons)


def test_dsse_and_intoto_shape():
    env, _ = _sign()
    import json
    st = json.loads(base64.b64decode(env["payload"]))
    assert st["_type"].endswith("in-toto.io/Statement/v1")
    assert st["subject"][0]["digest"]["sha256"]
    assert env["payloadType"] and env["signatures"][0]["keyid"].startswith("ed25519:")
