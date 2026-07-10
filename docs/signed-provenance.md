# Signed Provenance (v1.2.0)

> Checksums detect tampering; **signatures** say *who* produced and stands behind
> an artifact. This adds optional Ed25519 provenance + signing + trust-policy
> verification for publishable artifacts. Software-only — it proves signature
> validity and integrity, never task success, physical safety, or actuation
> authorization.

Signing needs the optional extra:

```bash
pip install 'lerobot-coreai[signing]'   # pulls `cryptography`
```

The base package never imports `cryptography` — the crypto backend is loaded
lazily, only when you sign or verify.

## Flow

```bash
# 1. Provenance: what the artifact is + which evidence backs it.
lerobot-coreai provenance-create \
  --artifact-dir publish/evo1-pusht-bridge-benchmark \
  --artifact-type bridge_benchmark \
  --output provenance.json

# 2. Sign: covers provenance + checksums + manifest. Key from an env var.
export LEROBOT_COREAI_SIGNING_KEY="$(python - <<'PY'
from cryptography.hazmat.primitives import serialization as s
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
print(Ed25519PrivateKey.generate().private_bytes(
    s.Encoding.Raw, s.PrivateFormat.Raw, s.NoEncryption()).hex())
PY
)"
lerobot-coreai sign-artifact \
  --artifact-dir publish/evo1-pusht-bridge-benchmark \
  --provenance provenance.json \
  --key-env LEROBOT_COREAI_SIGNING_KEY \
  --signer-name "Kevin" \
  --output signature.json

# 3. Verify: tamper, signature, and (optionally) trust policy.
lerobot-coreai verify-signature \
  --artifact-dir publish/evo1-pusht-bridge-benchmark \
  --provenance provenance.json \
  --signature signature.json \
  --trust-policy trust-policy.json
```

## Key handling

The private key is a 32-byte Ed25519 seed (hex or base64) supplied via
`--key-env NAME` (preferred) or `--key-file`. It is **never** written to a
report, log, or the signature file — only the **public key** (base64) and its
`sha256:` **fingerprint** are persisted. `sign-artifact` fails closed if the
named env var is unset.

## What verification checks

| Check | Catches |
|-------|---------|
| `anchor_hashes_match_signed_payload` | Tamper of manifest / checksums / provenance |
| `artifact_files_untampered` | Tamper of any bundled report (via `checksums.json`) |
| `provenance_hashes_match_files` | Provenance drift from the files |
| `key_fingerprint_matches_public_key` | Fingerprint/public-key mismatch |
| `signature_cryptographically_valid` | Forged or wrong-key signatures |
| `signer_trusted` (policy) | Untrusted signer fingerprint |
| `required_artifacts_present` (policy) | Missing required files |
| `no_forbidden_claims` (policy) | A report setting a forbidden claim true |

## Trust policy

```json
{
  "schema_version": "lerobot-coreai.trust_policy.v0",
  "trusted_keys": [{"name": "Kevin release key", "fingerprint": "sha256:abcd..."}],
  "required_artifacts": ["benchmark_manifest.json", "checksums.json", "provenance.json"],
  "forbidden_claims": ["proves_physical_safety", "authorizes_robot_actuation",
                       "native_upstream_registry", "supports_training"]
}
```

## Scope

Ed25519 local signing is the v1.2.0 baseline. The abstraction is intentionally
small so a future release can add Sigstore/cosign or SLSA provenance without
changing these manifests. A valid signature proves origin + integrity — it does
**not** prove physical safety and authorizes no robot actuation.
