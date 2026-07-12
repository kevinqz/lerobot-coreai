# RFC-0800 — Cross-Repository Delivery Plan, Release Train and Migration Program

> **Status:** Proposed  
> **Date:** 2026-07-12  
> **Target:** Apple-first, upstream-compatible Core AI ecosystem  
> **Normative language:** MUST, MUST NOT, SHOULD, SHOULD NOT and MAY are used as in RFC 2119.  
> **Snapshot:** See `SOURCE-SNAPSHOT.md`. The RFCs describe the target architecture; they do not claim that unfinished capabilities already exist.


## 1. Objective

Deliver a coherent ecosystem without freezing independent repository progress or introducing a monorepo.

The program optimizes for:

- direct Apple compatibility;
- truthful capabilities;
- independently releasable repositories;
- one cross-process contract;
- one evidence chain;
- minimal repository proliferation;
- demonstrable end-to-end value.

## 2. Workstream map

| Workstream | Lead repository | Dependents |
|---|---|---|
| Interop contracts | new `coreai-interop` | all |
| Discovery/profile index | Catalog | Runner, Fabric, consumers |
| Conversion profiles | Fabric | Catalog, LeRobot |
| Runtime kernel/providers | Runner | Server, ComfyUI, LeRobot |
| LAN transport | Server | remote consumers |
| Creative consumer | ComfyUI | Runner, Catalog |
| Robotics provider | LeRobot | Runner, Fabric, Catalog |

## 3. Phase 0 — Truth repair

This phase is a stop-the-line for cross-repository contradictions.

Required:

- Runner action and host-loop capability false until implemented.
- Align action request fixture between Swift and Python.
- Add explicit protocol version.
- Use unique embedded socket ownership.
- Comfy Runner binary verification fails closed.
- Server pins a Runner release.
- Fabric disables load-bearing name inference in release mode.
- Runner and Server documentation reflect actual implementation.
- LeRobot docs distinguish L3 test-only from L4 real Core AI.
- Catalog unknown values remain unknown.
- Every P0 receives a regression test.

No new feature should introduce another parallel schema.

## 4. Phase 1 — Interop bootstrap

Create `coreai-interop` 0.1:

- capture legacy protocol;
- define v1 envelope;
- Python and Swift generated types;
- golden fixtures;
- breaking-change checker;
- profile identifiers;
- release and governance policy.

Consumers add fixture CI before migration.

Exit criteria:

- one fixture is decoded identically by Swift and Python;
- legacy contradiction is documented and rejected in v1;
- protocol release is immutable and tagged.

## 5. Phase 2 — Runner foundation

Runner work:

- provider registry;
- Apple official provider;
- direct raw `AIModel` provider;
- community CoreAIKit provider;
- generic invoke;
- truthful capabilities;
- embedded lifecycle;
- signed binary releases;
- stable SPM product.

Keep legacy endpoints behind explicit compatibility version.

Exit criteria:

- Comfy existing stable paths still work;
- provider selection is observable;
- no domain-specific field exists in common kernel;
- Catalog network outage does not block local artifact execution.

## 6. Phase 3 — Catalog and Fabric profiles

Catalog 3.0:

- profiles;
- A0–A5;
- compatibility records;
- signed dist manifest;
- federation adapters.

Fabric 0.2:

- recipe v2;
- action profile;
- executor-owned receipts;
- Apple-first drivers;
- outer policy bundles.

Exit criteria:

- one existing LLM and one ACT recipe use v2;
- Catalog indexes their profiles;
- old static clients have a migration path.

## 7. Phase 4 — Consumer migration

### ComfyUI 1.0

- generated client;
- signed Runner acquisition;
- unique lifecycle;
- conformance-filtered nodes;
- offline cache.

### LeRobot 1.4

- generated client;
- real ACT artifact;
- real Swift Runner matrix;
- signed evidence.

Exit criteria:

- both consumers use the same protocol release;
- one Runner binary serves each independently;
- no shared-socket conflict;
- no client-specific Runner fork.

## 8. Phase 5 — Server

Only after Runner 1.0 and protocol 1.0:

- TLS/pairing;
- Bonjour;
- macOS node;
- immutable artifact resolution;
- remote provider;
- later iOS/iPadOS host.

Exit criteria:

- authenticated remote invocation;
- local/remote equivalence;
- cancellation;
- no sensitive pre-auth disclosure.

## 9. Release BOM

Maintain a machine-readable BOM in `coreai-interop/releases/`:

```yaml
release: 2026.10
components:
  interop: 1.0.0
  catalog: 3.0.0
  fabric: 0.2.0
  runner: 1.0.0
  comfyui_coreai: 1.0.0
  lerobot_coreai: 1.4.0
  server: 0.1.0
upstreams:
  apple_coreai_models: <sha>
  lerobot: ">=0.6,<0.7"
```

The BOM is signed and content-addressed. It records:

- source revisions;
- released artifacts;
- checksums;
- protocol/profile versions;
- minimum OS;
- known limitations.

## 10. Cross-repo CI

Create a scheduled and release-triggered matrix:

1. Verify interop generated code.
2. Build Runner.
3. Run Runner conformance.
4. Validate Catalog fixture.
5. Consume Fabric artifacts.
6. Run Comfy smoke.
7. Run LeRobot official matrix.
8. Optionally run Server remote smoke.
9. Publish signed result.

Failures do not automatically mutate source repositories. They open issues with exact BOM, logs and owners.

## 11. Compatibility policy

- Repositories release independently.
- Public protocol support spans current and previous major during migration.
- Catalog static schema changes require compatibility fixtures.
- Consumers declare supported interop and Runner ranges.
- `main` branches are never production dependencies.
- Experimental profiles are explicitly namespaced and cannot become stable by implication.
- Deprecated fields have a removal version and migration guide.

## 12. Ownership

Recommended CODEOWNERS domains:

- Apple compatibility;
- protocol/common kernel;
- action/LeRobot;
- conversion/provenance;
- Runner security;
- Server networking;
- ComfyUI UX;
- Catalog data governance.

A cross-repo contract PR requires review by producer and consumer owners.

## 13. RFC process

RFC states:

- Draft;
- Proposed;
- Accepted;
- Implementing;
- Final;
- Superseded;
- Rejected.

Each RFC includes:

- problem;
- decisions;
- non-goals;
- compatibility;
- security;
- migration;
- test plan;
- rollback;
- owners.

Implementation PRs link the accepted RFC and update its status.

## 14. Documentation

Each repository README begins with:

- exact mission;
- current maturity;
- implemented capabilities;
- unavailable capabilities;
- compatibility versions;
- relation to Apple and other repos;
- no overclaims;
- minimal quick start.

Master architecture links back to RFC-0000.

Documentation claim tests SHOULD verify key phrases and current versions.

## 15. Issue decomposition

Use labels:

- `interop-blocker`
- `ecosystem-p0`
- `cross-repo`
- `breaking-contract`
- `apple-compat`
- `action-profile`
- `supply-chain`
- `evidence`
- `docs-truth`
- `security`
- `migration`

Cross-repo issues contain affected versions and owner checklist.

## 16. Milestone gates

### Gate A — Protocol Truth

No divergent payloads or false capabilities.

### Gate B — Native Execution

Real Apple artifact runs through Runner.

### Gate C — Consumer Integration

Both ComfyUI and LeRobot use generated clients.

### Gate D — Evidence

Conversion and execution receipts replay.

### Gate E — Distribution

Signed Runner binary and signed Catalog dist.

### Gate F — Remote

Authenticated Server.

### Gate G — Policy Breadth

ACT, Diffusion and one VLA run through the architecture without core special cases.

## 17. Risk register

| Risk | Mitigation |
|---|---|
| Over-generalized protocol | Profiles and bloat test |
| Apple API changes | Upstream pin and compatibility CI |
| Community runtime divergence | Provider isolation |
| Catalog schema churn | Generated clients and versioned exports |
| Repository explosion | One mandatory new repo only |
| False certification | Production/test namespaces and protected signing |
| Robot safety overclaim | Explicit boundary and no safety promotion |
| Shared process conflicts | Embedded unique Runner |
| Supply-chain compromise | Signatures, attestations and fail closed |
| MLX duplication | Evaluate existing projects; no premature port |
| Server product distraction | Gate behind Runner 1.0 |
| Fabric becoming model zoo | Recipes/evidence only; bytes stay on HF |
| Consumer-specific Runner forks | Shared conformance and provider registry |

## 18. Rollback strategy

Every migration maintains:

- legacy protocol endpoint for a bounded window;
- previous signed Runner binary;
- previous Catalog static export;
- recipe schema converter;
- BOM rollback pointer.

Security revocations override rollback availability.

## 19. Success metrics

Technical:

- zero protocol divergence defects;
- percentage of capabilities backed by conformance;
- artifact-to-runtime success rate;
- reproducible conversion rate;
- client compatibility across BOMs;
- mean time to diagnose a failed load.

Adoption:

- external Catalog artifacts;
- external Fabric recipes;
- Comfy workflows shared;
- LeRobot policies deployed;
- independent conformance runs.

Metrics never replace qualitative safety/security review.

## 20. Definition of Program Complete

The first complete program release has:

- one protocol source;
- one signed BOM;
- one ACT artifact;
- one real Runner action execution;
- one official LeRobot matrix;
- ComfyUI regression-free;
- Catalog A0–A5 records;
- Fabric reproducible receipt;
- optional authenticated remote node;
- documentation matching reality;
- no competing Apple spec;
- no extra robot-specific Runner/Fabric repositories.
