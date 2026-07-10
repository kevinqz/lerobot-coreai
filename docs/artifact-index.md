# Artifact Index + Local Registry (v1.2.2)

> Once artifacts are verifiable and signable, they need to be **findable**. The
> artifact index is a local registry of signed/verified lerobot-coreai bundles:
> `init`, `add`, `list`, `find`, `verify`. Fail-closed — an `add` is refused if
> the artifact is tampered, overclaims, leaks a secret, or (with a trust policy)
> is signed by an untrusted key. Proves discovery/integrity only, never physical
> safety.

## Commands

```bash
lerobot-coreai artifact-index init --index-dir .lerobot-coreai/index

lerobot-coreai artifact-index add \
  --artifact-dir publish/evo1-pusht-bridge-benchmark \
  --artifact-type bridge_benchmark \
  --release-channel public-demo \
  --provenance provenance.json \
  --signature signature.json \
  --trust-policy trust-policy.json \
  --release-check-report release-checks/evo1/release_check_report.json \
  --index-dir .lerobot-coreai/index

lerobot-coreai artifact-index list --index-dir .lerobot-coreai/index
lerobot-coreai artifact-index find --policy.path kevinqz/EVO1-SO100-CoreAI \
  --dataset.repo_id lerobot/pusht --artifact-type bridge_benchmark \
  --release-channel public-demo --index-dir .lerobot-coreai/index
lerobot-coreai artifact-index verify --index-dir .lerobot-coreai/index
```

## What `add` records (and guarantees)

Each entry tracks artifact type, policy path, dataset, release channel, manifest
and provenance SHA256, signer fingerprint, `signature_verified`,
`release_check_passed`, and creation time. Before writing an entry, `add`:

- recomputes `checksums.json` and **refuses a tampered artifact**;
- scans every report and **refuses overclaims** (`proves_physical_safety`,
  `authorizes_robot_actuation`, `native_upstream_registry`, `supports_training`, …);
- scans for **raw secrets** (redacted values / `sha256:` fingerprints / env-var
  names are allowed) and refuses leaks;
- sets `signature_verified=true` **only after** a real `verify-signature` pass
  (honoring the trust policy when given);
- **refuses to silently overwrite** an existing `artifact_id` (use `--force`).

## `verify`

Re-checks every indexed artifact: the directory still exists, the manifest hash
still matches, `checksums.json` still validates, and no indexed artifact has
started overclaiming. Fails if any indexed artifact has drifted or been tampered.

## Scope

The index proves artifact discovery + integrity. It does not prove task success
or physical safety, and authorizes no robot actuation. Secrets never enter the
index. `native_upstream_registry` is always `false`.
