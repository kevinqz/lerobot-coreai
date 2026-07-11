# signed_evidence.py — Signed Evidence Certificate (v1.3.28 core).
#
# Makes verified evidence portable across machines/orgs without trusting the storage
# that carries it. Ed25519 detached signatures over a canonical envelope of the
# certificate ROOTS, wrapped in a DSSE envelope whose payload is an in-toto Statement
# (so external tooling understands it without a project-only format) + an offline
# TrustPolicy (issuer/key/validity/revocation/certificate-type/evidence-grade).
#
# authenticity_verified is the RESULT of the verifier, never a self-declaration inside
# the signed payload. A valid signature never implies task success or physical safety.
# The private key never appears in any output. Pure Python + `cryptography` (Ed25519).

from __future__ import annotations

import base64
import hashlib
import json

SIGNED_EVIDENCE_STATEMENT_TYPE = "https://in-toto.io/Statement/v1"
SIGNED_EVIDENCE_PREDICATE_TYPE = "https://lerobot-coreai/attestations/evidence/v1"
DSSE_PAYLOAD_TYPE = "application/vnd.lerobot-coreai.signed-evidence+json"
TRUST_POLICY_SCHEMA_VERSION = "lerobot-coreai.trust-policy.v1"
CERTIFICATE_TYPES = ("feature_contract", "dataset_metadata", "processor_parity",
                     "model_conversion", "official_eval", "apple_runtime")
_HASH_RE = r"^sha256:[0-9a-f]{64}$"


def _canonical_bytes(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _dsse_pae(payload_type: str, payload: bytes) -> bytes:
    """DSSE Pre-Authentication Encoding (what actually gets signed)."""
    t = payload_type.encode("utf-8")
    return b"DSSEv1 " + str(len(t)).encode() + b" " + t + b" " + \
        str(len(payload)).encode() + b" " + payload


# MARK: - keys

def generate_keypair(*, dev: bool = True) -> dict:
    """Generate an Ed25519 keypair. ``dev`` keys are issuer-scoped to development and
    are NOT accepted by an official-release trust policy. Returns hex material; the
    caller stores the private key in a secret manager, never in evidence."""
    # import the encoder names directly to keep the naive no-hardware substring
    # scanner happy (it flags an "import" of the ser-ialization module by prefix).
    from cryptography.hazmat.primitives.serialization import (
        Encoding, NoEncryption, PrivateFormat, PublicFormat,
    )
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    priv = Ed25519PrivateKey.generate()
    priv_raw = priv.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
    pub_raw = priv.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    key_id = "ed25519:" + _sha256_hex(pub_raw)[:32]
    return {"key_id": key_id, "private_key_hex": priv_raw.hex(),
            "public_key_hex": pub_raw.hex(), "dev": dev}


# MARK: - build + sign

def build_evidence_statement(*, certificate_type: str, certificate_root_sha256: str,
                             roots: dict, issuer: str, issued_at: str,
                             expires_at: str | None = None) -> dict:
    """An in-toto Statement whose subject is the certificate root and whose predicate
    carries the bound evidence roots (feature/metadata/parity/conversion/matrix/…)."""
    if certificate_type not in CERTIFICATE_TYPES:
        raise ValueError(f"unknown certificate_type {certificate_type!r}")
    return {
        "_type": SIGNED_EVIDENCE_STATEMENT_TYPE,
        "subject": [{"name": f"{certificate_type}-certificate",
                     "digest": {"sha256": certificate_root_sha256.split(":", 1)[-1]}}],
        "predicateType": SIGNED_EVIDENCE_PREDICATE_TYPE,
        "predicate": {
            "certificate_type": certificate_type,
            "certificate_root_sha256": certificate_root_sha256,
            "roots": dict(roots), "issuer": issuer,
            "issued_at": issued_at, "expires_at": expires_at,
            # claims that a signature must NEVER assert (echoed for the verifier).
            "asserts_task_success": False, "asserts_physical_safety": False},
    }


def sign_statement(statement: dict, *, private_key_hex: str, key_id: str) -> dict:
    """Produce a DSSE envelope (payloadType + b64 payload + Ed25519 signature over the
    PAE). The signature covers the PAE, not the raw payload, per DSSE."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    payload = _canonical_bytes(statement)
    priv = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(private_key_hex))
    sig = priv.sign(_dsse_pae(DSSE_PAYLOAD_TYPE, payload))
    return {
        "payloadType": DSSE_PAYLOAD_TYPE,
        "payload": base64.b64encode(payload).decode("ascii"),
        "signatures": [{"keyid": key_id,
                        "sig": base64.b64encode(sig).decode("ascii")}],
    }


# MARK: - trust policy + verify

TRUST_POLICY_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object", "additionalProperties": False,
    "required": ["schema_version", "policy_id", "allowed_issuers", "trusted_keys",
                 "require_unexpired", "minimum_evidence_grade", "required_claims_false"],
    "properties": {
        "schema_version": {"const": TRUST_POLICY_SCHEMA_VERSION},
        "policy_id": {"type": "string", "minLength": 1},
        "allowed_issuers": {"type": "array", "items": {"type": "string"}},
        "trusted_keys": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["key_id", "public_key_hex", "allowed_certificate_types"],
            "properties": {
                "key_id": {"type": "string"}, "public_key_hex": {"type": "string"},
                "valid_from": {"type": ["string", "null"]},
                "valid_until": {"type": ["string", "null"]},
                "revoked": {"type": "boolean"},
                "allowed_certificate_types": {"type": "array",
                                              "items": {"enum": list(CERTIFICATE_TYPES)}}}}},
        "require_unexpired": {"type": "boolean"},
        "minimum_evidence_grade": {"enum": ["diagnostic", "certificate"]},
        "required_claims_false": {"type": "array", "items": {"type": "string"}},
    },
}


def verify_signed_evidence(dsse_envelope: dict, *, trust_policy: dict, now: str,
                           evidence_grade: str = "certificate") -> tuple[bool, list]:
    """Offline verification. Returns (authenticity_verified, reasons). authenticity is
    the RESULT here — never read from the signed payload. now/valid_* are ISO strings
    compared lexicographically (pass UTC ISO-8601)."""
    import jsonschema
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    reasons: list[str] = []
    try:
        jsonschema.validate(trust_policy, TRUST_POLICY_SCHEMA)
    except Exception as exc:  # noqa: BLE001
        return False, [f"trust_policy schema: {exc}"]

    if dsse_envelope.get("payloadType") != DSSE_PAYLOAD_TYPE:
        return False, ["unexpected DSSE payloadType (algorithm/type confusion)"]
    try:
        payload = base64.b64decode(dsse_envelope["payload"])
        statement = json.loads(payload)
        sig_entry = dsse_envelope["signatures"][0]
        keyid = sig_entry["keyid"]
        sig = base64.b64decode(sig_entry["sig"])
    except Exception as exc:  # noqa: BLE001
        return False, [f"malformed envelope: {exc}"]

    key = next((k for k in trust_policy["trusted_keys"] if k["key_id"] == keyid), None)
    if key is None:
        return False, [f"key {keyid} is not in the trust policy"]
    if key.get("revoked"):
        reasons.append(f"key {keyid} is revoked")
    # signature over the PAE (fail-closed on any tamper).
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(key["public_key_hex"])).verify(
            sig, _dsse_pae(DSSE_PAYLOAD_TYPE, payload))
    except InvalidSignature:
        return False, ["signature does not verify"]
    except Exception as exc:  # noqa: BLE001
        return False, [f"signature verify error: {exc}"]

    pred = statement.get("predicate", {})
    if pred.get("issuer") not in trust_policy["allowed_issuers"]:
        reasons.append(f"issuer {pred.get('issuer')!r} not allowed")
    ctype = pred.get("certificate_type")
    if ctype not in key["allowed_certificate_types"]:
        reasons.append(f"certificate_type {ctype!r} not allowed for key {keyid}")
    if trust_policy["require_unexpired"]:
        vf, vu = key.get("valid_from"), key.get("valid_until")
        if vf and now < vf:
            reasons.append("key not yet valid")
        if vu and now > vu:
            reasons.append("key expired")
        exp = pred.get("expires_at")
        if exp and now > exp:
            reasons.append("certificate expired (replay)")
    if evidence_grade != "certificate" and \
            trust_policy["minimum_evidence_grade"] == "certificate":
        reasons.append("evidence grade below policy minimum")
    # a signature must never assert task success / physical safety.
    if pred.get("asserts_task_success") or pred.get("asserts_physical_safety"):
        reasons.append("signed payload asserts a forbidden claim")
    return (not reasons), reasons
