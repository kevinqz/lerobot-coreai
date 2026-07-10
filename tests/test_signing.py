# test_signing.py — Ed25519 signing primitives (v1.2.0). Requires [signing].

import base64

import pytest

pytest.importorskip("cryptography")

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from lerobot_coreai.errors import CoreAIPolicyError
from lerobot_coreai.signing import (
    canonical_payload, fingerprint, load_private_key, public_key_bytes,
    sign_payload, verify_payload,
)


def _seed_hex():
    k = Ed25519PrivateKey.generate()
    raw = k.private_bytes(serialization.Encoding.Raw,
                          serialization.PrivateFormat.Raw,
                          serialization.NoEncryption())
    return raw.hex()


def test_sign_and_verify_roundtrip():
    priv = load_private_key(_seed_hex())
    from lerobot_coreai.signing import load_public_key_b64
    pub_b64 = base64.b64encode(public_key_bytes(priv)).decode()
    payload = canonical_payload({"a": "1", "b": "2"})
    sig = sign_payload(payload, priv)
    assert verify_payload(payload, sig, load_public_key_b64(pub_b64)) is True


def test_verify_fails_on_tampered_payload():
    priv = load_private_key(_seed_hex())
    from lerobot_coreai.signing import load_public_key_b64
    pub = load_public_key_b64(base64.b64encode(public_key_bytes(priv)).decode())
    sig = sign_payload(canonical_payload({"a": "1"}), priv)
    assert verify_payload(canonical_payload({"a": "2"}), sig, pub) is False


def test_verify_fails_with_wrong_key():
    priv1 = load_private_key(_seed_hex())
    priv2 = load_private_key(_seed_hex())
    from lerobot_coreai.signing import load_public_key_b64
    pub2 = load_public_key_b64(base64.b64encode(public_key_bytes(priv2)).decode())
    sig = sign_payload(canonical_payload({"a": "1"}), priv1)
    assert verify_payload(canonical_payload({"a": "1"}), sig, pub2) is False


def test_base64_seed_accepted():
    k = Ed25519PrivateKey.generate()
    raw = k.private_bytes(serialization.Encoding.Raw,
                          serialization.PrivateFormat.Raw,
                          serialization.NoEncryption())
    priv = load_private_key(base64.b64encode(raw).decode())
    assert public_key_bytes(priv)


def test_bad_key_material_fails():
    with pytest.raises(CoreAIPolicyError):
        load_private_key("not-a-valid-key")


def test_fingerprint_is_sha256_prefixed():
    priv = load_private_key(_seed_hex())
    fp = fingerprint(public_key_bytes(priv))
    assert fp.startswith("sha256:")


def test_canonical_payload_is_order_independent():
    assert canonical_payload({"a": "1", "b": "2"}) == canonical_payload({"b": "2", "a": "1"})
