# Signed Evidence Certificate (v1.3.28 core — v1.3.26.4)

> Make verified evidence portable across machines and organizations without trusting
> the storage location that carries it.

Until now evidence could be integral + semantically valid, but an attacker who could
replace a bundle + its roots could produce another self-consistent set. This adds
**authenticity**: an Ed25519 signature over the certificate roots + an offline
TrustPolicy.

## What is signed

Not each file. A canonical **in-toto Statement** whose subject is the certificate
root and whose predicate binds the evidence roots (`matrix`, `artifact`,
`feature_contract`, `dataset_metadata`, `processor_parity`, `model_conversion`, …),
the issuer, and validity. It is wrapped in a **DSSE** envelope (`payloadType` +
base64 payload + Ed25519 signature over the DSSE **PAE**) so external tooling
(in-toto / SLSA verifiers) can read it without a project-only format. The Ed25519
offline path stays the base; DSSE/in-toto is the interoperability layer the review
asked for.

## TrustPolicy v1

Offline, fail-closed: `allowed_issuers`, `trusted_keys` (each with
`public_key_hex`, `valid_from/until`, `revoked`, `allowed_certificate_types`),
`require_unexpired`, `minimum_evidence_grade`, `required_claims_false`. The verifier
returns `authenticity_verified` — it is the **result**, never a self-declaration
inside the signed payload. A valid signature never implies task success or physical
safety (the statement pins `asserts_task_success`/`asserts_physical_safety` false and
the verifier rejects a forged true).

## Threat model covered (tests)

byte/root tamper · wrong key · untrusted issuer · expired/revoked key · certificate
replay after expiry · certificate-type not allowed for the key · diagnostic evidence
rejected by a certificate-grade policy · DSSE `payloadType` (algorithm/type)
confusion · trust-policy tamper · **private key never present in any output**.

## Keys

`generate_keypair(dev=…)` — a **dev** key is issuer-scoped to development and is not
accepted by an official-release policy. The **release** key lives only in a protected
CI environment (tag/release workflow), never in a PR, never persisted in evidence or
logs. Signing happens only after full re-verification.

## CLI

```bash
lerobot-coreai sign-evidence --statement st.json \
  --private-key-env COREAI_SIGNING_KEY --key-id ed25519:… --output signed.json
lerobot-coreai verify-signed-evidence --envelope signed.json \
  --trust-policy trust/official-release.json --now 2026-07-11T00:00:00Z --json
```

## Not yet

- **Wiring the signing onto the official-eval certificate** (v1.3.27 → v1.3.28) — this
  ships the signing *primitive* over generic roots; the official-eval-specific signed
  certificate lands when the CLI certification (v1.3.27) exists.
- **Sigstore / keyless** as an optional online mode (Ed25519 offline stays the base).
- **Protected-CI release signing job** (dev-key path is used until then).
