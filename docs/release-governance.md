# Release Channel Governance (v1.2.1)

> Not every verifiable artifact should be publishable on every channel.
> `release-check` evaluates an artifact against a per-channel **release policy**:
> which reports are required, whether a valid signature is required, and whether
> overclaims / raw secrets / real-session / external-http artifacts are allowed.
> Fail-closed — anything not explicitly permitted blocks the release. Proves
> publishability-under-policy only, never physical safety.

## Channels (built-in defaults)

| Channel | Signature | No overclaims | No raw secrets | Real-session | External-http | Notes |
|---------|-----------|---------------|----------------|--------------|---------------|-------|
| `dev` | – | – | – | ✓ | ✓ | Loosest; local iteration |
| `internal` | – | required | required | ✓ | ✓ | Team-internal |
| `public-demo` | **required** | required | required | **blocked** | **blocked** | Bridge reports required |
| `research` | required | required | required | ✓ | ✓ | Signed research artifacts |
| `guarded-real-evidence` | required | required | required | ✓ | ✓ | Requires approval + readiness + verify-real-session |

## CLI

```bash
lerobot-coreai release-check \
  --artifact-dir publish/evo1-pusht-bridge-benchmark \
  --artifact-type bridge_benchmark \
  --channel public-demo \
  --provenance provenance.json \
  --signature signature.json \
  --trust-policy trust-policy.json \
  --output-dir release-checks/evo1
```

Writes `release_check_report.json` / `.md`. Override the built-in channel policy
with `--release-policy policy.json`.

## Checks

- `required_reports_present` — the channel's required report filenames are present.
- `signature_valid` — when the channel requires it, the signature verifies
  (delegates to `verify-signature`, honoring the trust policy).
- `no_overclaims` — no bundled report sets a forbidden claim (`proves_physical_safety`,
  `authorizes_robot_actuation`, `native_upstream_registry`, `supports_training`, …) true.
- `no_raw_secrets` — no report leaks a token/secret/key value (redacted values,
  `sha256:` fingerprints, and env-var **names** are allowed).
- `no_real_session_artifacts` / `no_external_http_artifacts` — blocked on public
  channels by default.
- `guarded_real_evidence_complete` — approval + readiness + verify-real-session
  artifacts all present.

## Custom policy

```json
{
  "schema_version": "lerobot-coreai.release_policy.v0",
  "channel": "public-demo",
  "required_reports": ["lerobot_compatibility_report.json", "lerobot_bridge_report.json",
                       "feature_mapping.json", "eval_v2_report.json", "obs_bridge_report.json"],
  "require_signature": true,
  "require_no_overclaims": true,
  "allow_real_session_artifacts": false,
  "allow_external_http_artifacts": false
}
```

A passing `release-check` proves the artifact satisfies the channel policy — it
does not prove task success or physical safety, and authorizes no actuation.
