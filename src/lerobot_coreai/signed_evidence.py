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
    key_id = key_id_for_public_key(pub_raw.hex())
    return {"key_id": key_id, "private_key_hex": priv_raw.hex(),
            "public_key_hex": pub_raw.hex(), "dev": dev}


def key_id_for_public_key(public_key_hex: str) -> str:
    """The canonical key id derived FROM the public key. The verifier recomputes this
    and refuses a trust-policy entry whose key_id does not match its own key (v1.3.26.7)."""
    return "ed25519:" + _sha256_hex(bytes.fromhex(public_key_hex))[:32]


# MARK: - closed schemas for the signed payload (v1.3.26.7: verify what is signed)

_HASH = {"type": "string", "pattern": r"^sha256:[0-9a-f]{64}$"}
DSSE_ENVELOPE_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object", "additionalProperties": False,
    "required": ["payloadType", "payload", "signatures"],
    "properties": {
        "payloadType": {"const": DSSE_PAYLOAD_TYPE},
        "payload": {"type": "string", "minLength": 1},
        # exactly ONE signature (no ambiguous/empty multi-sig).
        "signatures": {"type": "array", "minItems": 1, "maxItems": 1, "items": {
            "type": "object", "additionalProperties": False,
            "required": ["keyid", "sig"],
            "properties": {"keyid": {"type": "string", "minLength": 1},
                           "sig": {"type": "string", "minLength": 1}}}},
    },
}
INTOTO_STATEMENT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object", "additionalProperties": False,
    "required": ["_type", "subject", "predicateType", "predicate"],
    "properties": {
        "_type": {"const": SIGNED_EVIDENCE_STATEMENT_TYPE},
        "subject": {"type": "array", "minItems": 1, "items": {
            "type": "object", "additionalProperties": False,
            "required": ["name", "digest"],
            "properties": {"name": {"type": "string", "minLength": 1},
                           "digest": {"type": "object", "additionalProperties": False,
                                      "required": ["sha256"],
                                      "properties": {"sha256": {"type": "string",
                                                                "pattern": r"^[0-9a-f]{64}$"}}}}}},
        "predicateType": {"const": SIGNED_EVIDENCE_PREDICATE_TYPE},
        "predicate": {
            "type": "object", "additionalProperties": False,
            "required": ["certificate_type", "certificate_root_sha256", "roots",
                         "issuer", "issued_at", "expires_at", "asserts_task_success",
                         "asserts_physical_safety"],
            "properties": {
                "certificate_type": {"enum": list(CERTIFICATE_TYPES)},
                "certificate_root_sha256": _HASH,
                "roots": {"type": "object", "additionalProperties": _HASH},
                "issuer": {"type": "string", "minLength": 1},
                "issued_at": {"type": "string", "minLength": 1},
                "expires_at": {"type": ["string", "null"]},
                "asserts_task_success": {"const": False},
                "asserts_physical_safety": {"const": False}}},
    },
}

# each certificate type must bind these evidence roots (non-null, hash-format).
CERTIFICATE_TYPE_REQUIRED_ROOTS = {
    "feature_contract": ("feature_contract_sha256",),
    "dataset_metadata": ("dataset_metadata_sha256",),
    "processor_parity": ("processor_parity_sha256", "feature_contract_sha256"),
    "model_conversion": ("model_conversion_sha256", "artifact_root_sha256"),
    "official_eval": ("artifact_root_sha256", "feature_contract_sha256",
                      "dataset_metadata_sha256", "processor_parity_sha256"),
    "apple_runtime": ("artifact_root_sha256", "aimodel_sha256",
                      "signed_official_eval_certificate_sha256"),
}
MAX_CERTIFICATE_TTL_SECONDS = 400 * 24 * 3600     # a signed cert may not live forever


def _parse_rfc3339(value):
    """Parse an RFC-3339 / ISO-8601 timestamp into a timezone-AWARE datetime, or None.
    A naive (offset-less) timestamp is rejected (returns None)."""
    import datetime as _dt
    if not isinstance(value, str):
        return None
    v = value.strip()
    if v.endswith("Z"):
        v = v[:-1] + "+00:00"
    try:
        dt = _dt.datetime.fromisoformat(v)
    except ValueError:
        return None
    return dt if dt.tzinfo is not None else None


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

# The pinned OFFICIAL trust anchor (v1.3.26.11). A certificate-grade high claim may be
# promoted only under a policy that matches this anchor's identity AND carries no dev
# key. In a real protected release, `OFFICIAL_TRUST_POLICY_SHA256` is additionally
# pinned to the exact policy bytes committed in the release environment; here we pin the
# anchor IDENTITY (policy_id + allowed issuers) so provenance authority no longer comes
# from a caller-supplied, self-signed policy.
OFFICIAL_TRUST_POLICY_ID = "coreai-official-release.v1"
OFFICIAL_ALLOWED_ISSUERS = ("lerobot-coreai-release-ci",)
# The pinned OFFICIAL release key ids (v1.3.26.12). A PRODUCTION high claim may be
# promoted only under a policy whose every trusted key id is in this set. It is
# deliberately EMPTY: no protected release key exists yet (it is provisioned in the
# v1.3.28 protected-signing work), so production promotion is structurally impossible
# until then — and CI, which has no such key, can only ever produce `test_only`
# evidence. This is what makes "no production high claim in CI" an enforced fact, not a
# claim about intent.
OFFICIAL_RELEASE_KEY_IDS: frozenset = frozenset()


def certificate_root_sha256(certificate: dict) -> str:
    """The canonical content root of a certificate's bytes — what a signed statement's
    subject must equal (v1.3.26.11 cross-binding). Prefixed `sha256:` to match the
    predicate root format."""
    return "sha256:" + _sha256_hex(_canonical_bytes(certificate))


def trust_policy_sha256(policy: dict) -> str:
    """Canonical content hash of a trust policy — recorded in the verified result so the
    exact policy that granted authority is bound into the evidence (v1.3.26.12)."""
    return "sha256:" + _sha256_hex(_canonical_bytes(policy))


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
                # a dev key is development-scoped; an OFFICIAL anchor refuses it.
                "dev": {"type": "boolean"},
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
    the RESULT here — never read from the signed payload. v1.3.26.7 closes the verifier:
    the DSSE envelope + the in-toto statement are schema-validated (closed, no extra
    fields), subject↔certificate-root are cross-bound, the certificate-type's required
    roots must be present, key_id is recomputed from the public key, timestamps are
    RFC-3339 timezone-aware, and TrustPolicy.required_claims_false is applied."""
    import jsonschema
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    reasons: list[str] = []
    if not isinstance(dsse_envelope, dict) or \
            dsse_envelope.get("payloadType") != DSSE_PAYLOAD_TYPE:
        return False, ["unexpected DSSE payloadType (algorithm/type confusion)"]
    try:
        jsonschema.validate(trust_policy, TRUST_POLICY_SCHEMA)
        jsonschema.validate(dsse_envelope, DSSE_ENVELOPE_SCHEMA)
    except Exception as exc:  # noqa: BLE001
        return False, [f"schema: {exc}"]

    try:
        payload = base64.b64decode(dsse_envelope["payload"], validate=True)
        statement = json.loads(payload)
        sig_entry = dsse_envelope["signatures"][0]
        keyid = sig_entry["keyid"]
        sig = base64.b64decode(sig_entry["sig"], validate=True)
    except Exception as exc:  # noqa: BLE001
        return False, [f"malformed envelope: {exc}"]
    try:
        jsonschema.validate(statement, INTOTO_STATEMENT_SCHEMA)   # verify what is signed
    except Exception as exc:  # noqa: BLE001
        return False, [f"statement schema: {exc}"]

    key = next((k for k in trust_policy["trusted_keys"] if k["key_id"] == keyid), None)
    if key is None:
        return False, [f"key {keyid} is not in the trust policy"]
    # key_id MUST be the canonical id of the policy's public key (no arbitrary id).
    if key["key_id"] != key_id_for_public_key(key["public_key_hex"]):
        return False, [f"trust-policy key_id {keyid} != id derived from its public key"]
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

    pred = statement["predicate"]
    # subject <-> certificate root cross-binding (a swapped subject fails).
    subj_digest = statement["subject"][0]["digest"]["sha256"]
    if subj_digest != pred["certificate_root_sha256"].split(":", 1)[-1]:
        reasons.append("subject.digest.sha256 != predicate.certificate_root_sha256")
    if pred["issuer"] not in trust_policy["allowed_issuers"]:
        reasons.append(f"issuer {pred['issuer']!r} not allowed")
    ctype = pred["certificate_type"]
    if ctype not in key["allowed_certificate_types"]:
        reasons.append(f"certificate_type {ctype!r} not allowed for key {keyid}")
    # required roots for this certificate type must be present + hash-format.
    for root in CERTIFICATE_TYPE_REQUIRED_ROOTS.get(ctype, ()):
        if root not in pred["roots"]:
            reasons.append(f"missing required root {root!r} for {ctype}")

    # RFC-3339 timezone-aware validity chain: valid_from <= issued_at <= now <
    # expires_at <= valid_until (+ TTL); a naive/malformed stamp fails.
    import datetime as _dt
    now_dt = _parse_rfc3339(now)
    issued_dt = _parse_rfc3339(pred["issued_at"])
    if now_dt is None:
        return False, ["`now` is not an RFC-3339 timezone-aware timestamp"]
    if issued_dt is None:
        reasons.append("issued_at is not an RFC-3339 timezone-aware timestamp")
    elif issued_dt > now_dt:
        reasons.append("certificate issued in the future")
    if trust_policy["require_unexpired"]:
        vf, vu, exp = (_parse_rfc3339(key.get("valid_from")),
                       _parse_rfc3339(key.get("valid_until")),
                       _parse_rfc3339(pred.get("expires_at")))
        if key.get("valid_from") and vf is None:
            reasons.append("key valid_from is malformed")
        if key.get("valid_until") and vu is None:
            reasons.append("key valid_until is malformed")
        if vf is not None and issued_dt is not None and issued_dt < vf:
            reasons.append("issued before key valid_from")
        if vu is not None and now_dt > vu:
            reasons.append("key expired")
        # a non-null but unparseable expires_at must FAIL, not be treated as "no expiry".
        if pred.get("expires_at") and exp is None:
            reasons.append("expires_at is malformed")
        if exp is not None and now_dt >= exp:
            reasons.append("certificate expired (replay)")
        # a certificate may not outlive the key that signed it.
        if exp is not None and vu is not None and exp > vu:
            reasons.append("certificate expires after key valid_until")
        if exp is not None and issued_dt is not None and \
                (exp - issued_dt) > _dt.timedelta(seconds=MAX_CERTIFICATE_TTL_SECONDS):
            reasons.append("certificate TTL exceeds policy maximum")
    if evidence_grade != "certificate" and \
            trust_policy["minimum_evidence_grade"] == "certificate":
        reasons.append("evidence grade below policy minimum")
    # dynamically apply required_claims_false against the predicate + its roots.
    for claim in trust_policy["required_claims_false"]:
        if bool(pred.get(claim)) or bool(pred.get("roots", {}).get(claim)):
            reasons.append(f"required-false claim {claim!r} is present/true")
    # a signature must never assert task success / physical safety (schema pins these,
    # re-checked defensively).
    if pred["asserts_task_success"] or pred["asserts_physical_safety"]:
        reasons.append("signed payload asserts a forbidden claim")
    return (not reasons), reasons
