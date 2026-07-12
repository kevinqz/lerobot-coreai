# Conformance Levels L0–L6 (RFC-0700 §8)

> The ecosystem RFC pack (2026-07-12) defines a single, honest ladder for the whole
> Apple/CoreAI **deployment** path — distinct from the older
> [LeRobot Compatibility Levels](lerobot-compatibility-levels.md), which grades the
> individual rungs of the LeRobot *policy contract*.

Run it:

```bash
lerobot-coreai conformance-level          # human table
lerobot-coreai conformance-level --json   # machine-readable
```

## The ladder

| Level | Title | Meaning |
|-------|-------|---------|
| **L0** | Metadata | Artifact can be inspected |
| **L1** | Protocol | Runner action-profile handshake |
| **L2** | Factory | Official LeRobot factory/plugin loads |
| **L3** | Official Eval | Real official CLI completes the controlled matrix |
| **L4** | Real Core AI | Real Swift Runner executes a real `.aimodel` |
| **L5** | Device Certified | Signed, scoped device/artifact/runtime certificate |
| **L6** | Robot Task Evidence | Guarded physical-run evidence (never a safety certification) |

The ladder is **monotonic**: a level is achieved only when every level below it is. A
gap stops the climb — you cannot be L5 without L4.

## Current state (honest)

**`L3` (Official Eval), namespace `test_only`.**

- **L0–L3 achieved.** Metadata inspection, the Runner protocol handshake, official
  LeRobot factory/plugin loading, and the **real** official `lerobot-eval` five-case
  matrix (`single-b1` / `native-b2` / `native-b4` / `split-b2` / `split-b4`, v1.3.27.3)
  all exist and pass in CI.
- **`test_only`, not production.** The L3 matrix runs against a protocol-compatible
  **stub** Runner, and the executor receipt is **unsigned**. A production claim requires
  an executor-signed receipt under a **pinned release key** (still pending).
- **L4+ not achieved.** There is no real Swift Runner executing a real `.aimodel` (L4),
  no production-signed device certificate (L5), and no guarded robot-task evidence (L6).

## What each remaining level requires

- **L4 Real Core AI** — a real Swift CoreAI Runner (RFC-0400) executing a real `.aimodel`
  produced by Fabric (RFC-0300), driven through the same official `lerobot-eval` matrix,
  with parity against PyTorch. Needs real Apple/CoreAI software artifacts.
- **L5 Device Certified** — L4 plus a signed certificate scoped to the exact
  hardware / OS / SDK / Runner / `.aimodel` tuple, issued under a pinned protected
  release key (a maintainer-held secret, never minted by the toolchain itself).
- **L6 Robot Task Evidence** — guarded physical-run evidence. This is bounded, fail-closed
  operational evidence and is **never** equivalent to mechanical/physical-safety
  certification.

No high claim is ever derived from `test_only` evidence.

## Runtime providers (RFC-0700 §13)

The certification contracts are backend-neutral, and the provider identities are
declared now with honest status (implementation deferred where it doesn't exist):

| Provider | Status | Role |
|----------|--------|------|
| `coreai` | **implemented** | Apple Core AI deployment (the only real path today) |
| `pytorch_reference` | reference | Upstream LeRobot/PyTorch parity oracle |
| `mlx` | deferred | Reserved identity only — no premature port |

`require_available()` fails closed on a reserved provider, so `mlx`/`pytorch_reference`
can never be routed as if they were real deployment targets. Every provider is compared
against one LeRobot semantic contract.
