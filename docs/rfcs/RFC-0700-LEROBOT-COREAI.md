# RFC-0700 — `lerobot-coreai` as the Apple Deployment Provider for LeRobot

> **Status:** Proposed  
> **Date:** 2026-07-12  
> **Target:** Apple-first, upstream-compatible Core AI ecosystem  
> **Normative language:** MUST, MUST NOT, SHOULD, SHOULD NOT and MAY are used as in RFC 2119.  
> **Snapshot:** See `SOURCE-SNAPSHOT.md`. The RFCs describe the target architecture; they do not claim that unfinished capabilities already exist.


## 1. Repository snapshot

- Repository: `https://github.com/kevinqz/lerobot-coreai`
- Reviewed commit: `77d9840fd9926c2c8c4154d02e7056d6604cf4a5`
- Current role: v1.3.27.3; real official `lerobot-eval` five-case matrix and executor-derived `test_only` receipt; real Swift Runner/`.aimodel` production path remains open.

The repository's public documentation still says official `lerobot-eval` certification is pending. Documentation MUST distinguish:

- official CLI conformance through a real subprocess and registered environment: achieved;
- controlled five-case matrix against protocol-compatible stub Runners: achieved;
- real Swift Runner + real `.aimodel`: not achieved;
- production signed certification: not achieved.

## 2. Positioning

Canonical description:

> Deploy, execute and verify LeRobot policies as Apple Core AI artifacts.

The repository SHALL be three products in one distribution family:

1. **LeRobot Core AI Provider**
2. **LeRobot Core AI Packager**
3. **LeRobot Core AI Conformance and Certification Suite**

It is not a full LeRobot port.

## 3. Responsibility boundary

LeRobot remains authoritative for:

- policies/configs;
- training;
- datasets;
- processors;
- robot interfaces;
- recording/teleop;
- official eval semantics;
- async inference/RTC.

`lerobot-coreai` owns:

- Core AI artifact binding;
- Runner client/provider;
- feature and processor mapping;
- action queue semantics;
- LeRobot plugin integration;
- packaging;
- parity/conformance;
- evidence and guarded deployment workflows.

Fabric converts. Catalog indexes. Runner executes.

## 4. Package structure

Recommended eventual structure:

```text
packages/
├── lerobot-coreai/
│   ├── provider/
│   ├── packaging/
│   ├── evidence/
│   ├── safety/
│   └── cli/
├── lerobot-policy-coreai-bridge/
└── lerobot-coreai-conformance/
```

These MAY remain in one repository and coordinated release.

The base package and official plugin must have clear version compatibility and a shared release BOM.

## 5. Provider interface

The official plugin should implement the minimum LeRobot policy contract and delegate:

```text
LeRobot batch
→ processor plumbing
→ profile-valid observation
→ generated Runner client
→ action chunk
→ per-timestep action queue
```

No duplicate policy implementation.

The provider never imports Fabric at inference time.

## 6. Protocol adoption

Replace custom hand-built `RunnerClient` payloads with the generated interop client.

Action profile mapping MUST be exact:

- feature names;
- dtype;
- shape;
- layout;
- task field;
- batching;
- session/reset;
- action chunk;
- deadlines;
- deterministic seed.

A Swift/Python golden fixture prevents recurrence of the observation nesting mismatch.

## 7. Packager

Packaging consumes Fabric outputs and emits or validates the outer `.coreaipolicy` bundle.

It SHALL bind:

- source LeRobot config/checkpoint;
- artifact components;
- FeatureContract;
- ProcessorStageContract;
- PolicyExecutionContract;
- normalization ownership;
- Runner profile;
- LeRobot version range;
- evidence refs;
- policy family;
- optional tokenizer/scheduler resources.

It does not perform conversion itself except orchestration through Fabric.

## 8. Conformance levels

Define user-facing levels:

| Level | Meaning |
|---|---|
| L0 Metadata | Artifact can be inspected |
| L1 Protocol | Runner action profile handshake |
| L2 Factory | Official LeRobot factory/plugin loads |
| L3 Official Eval | Real official CLI completes controlled matrix |
| L4 Real Core AI | Real Swift Runner executes real `.aimodel` |
| L5 Device Certified | Signed, scoped device/artifact/runtime certificate |
| L6 Robot Task Evidence | Guarded physical run evidence; never equivalent to safety certification |

Current state is L3 test-only, not L4 production.

## 9. Official eval matrix

Maintain five cases:

- single B=1;
- native B=2;
- native B=4;
- split-and-stack B=2;
- split-and-stack B=4.

The next milestone replaces protocol stubs with the real Swift Runner.

Receipt fields derive from:

- real subprocess;
- registered environment nonce;
- output tree;
- replay verifier;
- actual Runner/artifact identity;
- exact policy config;
- case manifests.

## 10. Real Action Runtime milestone

ACT first:

```text
real LeRobot ACT checkpoint
→ Fabric
→ real .aimodel
→ Catalog
→ Swift Runner Action provider
→ official lerobot-eval matrix
→ parity against PyTorch
```

Only after ACT should Diffusion and VLA policies expand the execution-plan and state model.

ACT is selected because it demonstrates:

- image/state input;
- deterministic single-pass model;
- action chunk;
- queue semantics;
- real robot-policy artifact;
- manageable debugging surface.

## 11. Diffusion and VLA progression

### Diffusion

Adds:

- iterative execution plan;
- RNG and seed;
- scheduler state;
- host loop;
- deadline/performance sensitivity.

### SmolVLA/Pi0

Adds:

- language/tokenizer;
- multi-component model;
- VLM resources;
- flow matching/autoregressive variants;
- larger memory and device constraints.

Each family receives a separate conformance profile and evidence policy.

## 12. Training

No Core AI training implementation.

Official recommendations:

- LeRobot + PyTorch MPS for upstream-compatible Mac training;
- cloud/CUDA when appropriate;
- evaluate community LeRobot-MLX as a future provider/oracle;
- do not fork/port policies into this repository.

The CLI may guide users to supported training routes but does not own them.

## 13. MLX future

Add an abstract provider identity now, but defer implementation.

Possible future modes:

- PyTorch reference;
- Core AI deployment;
- MLX dynamic/training.

Common evidence compares all against one LeRobot semantic contract.

A separate MLX repository is created only if the provider becomes real and upstream collaboration cannot host it. Existing LeRobot-MLX community work should be conformance-tested before any duplicate implementation.

## 14. Safety boundary

Preserve all current safeguards:

- dry run sends nothing;
- shadow sends nothing;
- sim only drives simulator;
- guarded real remains bounded and fail closed;
- no software report proves physical safety;
- hardware e-stop remains external;
- operator approval is not safety certification.

Runner Action profile produces action tensors only. Hardware egress stays in LeRobot/guarded adapters.

Safety profiles bind a robot family and action representation; they do not imply mechanical certification.

## 15. Evidence architecture

Continue full-chain evidence but focus effort on executed reality:

- conversion receipt from Fabric;
- real artifact root;
- Runner binary/provider identity;
- official eval receipt;
- parity replay;
- trust policy;
- scoped Apple device identity;
- exact LeRobot and plugin versions.

No high claim from synthetic or `test_only` evidence.

The verifier traverses every leaf rather than checking only root string formats.

## 16. Catalog integration

Catalog entries for LeRobot policies include:

- policy family;
- LeRobot version range;
- profile IDs;
- artifact root;
- supported batching/state;
- conversion status;
- L0–L6 conformance;
- device-scoped evidence;
- robot/embodiment compatibility as a semantic profile, not an execution guarantee.

`lerobot-coreai list/doctor` should use the stable Catalog client.

## 17. Async inference and RTC

After L4:

- map LeRobot async server/client lifecycle;
- propagate deadlines and cancellation;
- action chunk streaming;
- session affinity;
- backpressure;
- reconnect behavior;
- Real-Time Chunking semantics.

These features belong in the LeRobot provider plus generic Runner session primitives.

## 18. Remote Server provider

Remote execution is optional and separate from local Core AI:

```text
LeRobot → remote provider → coreai-server → Runner
```

The provider MUST account for network latency, cancellation and node loss. It never silently falls back from local to remote during a guarded real session.

## 19. Roadmap

### LR1 — Contract convergence

Adopt interop generated client and fix docs.

### LR2 — Real Fabric bundle

Consume a real ACT output.

### LR3 — Real Swift Runner

Run local Action provider.

### LR4 — L4 matrix

Complete real matrix and parity.

### LR5 — Protected signing

Issue first scoped L5 certificate.

### LR6 — Diffusion

Execution plan and deterministic evidence.

### LR7 — VLA

SmolVLA/Pi0 family.

### LR8 — Async/RTC

Modern LeRobot execution.

### LR9 — Remote Server

Authenticated remote provider.

### LR10 — Robot Gateway

Implement the authenticated reference gateway, watchdog, session receipts and mock/SO-101 adapters.

### LR11 — Mobile recording bridge

Validate and convert app episode-staging manifests to upstream LeRobotDataset.

### LR12 — Apple app conformance

Ship fixtures proving that the Swift app binds the exact FeatureContract, action queue and reset semantics.

## 19.1 Apple app and Robot Gateway reference

`lerobot-coreai` SHALL provide the Python-side reference implementation required by RFC-0900:

```text
lerobot-coreai gateway
├── generated Robot Gateway Protocol server
├── upstream LeRobot Robot factory
├── observation-state publisher
├── action validator
├── sequence/deadline enforcement
├── watchdog
├── supervisor integration
└── session receipt
```

The gateway is distinct from the Core AI Runner and Server.

### Gateway responsibilities

- load the selected upstream LeRobot robot implementation;
- expose robot identity and feature contract;
- publish joint state and hardware telemetry;
- accept only authenticated, ordered, unexpired action chunks;
- enforce software safety profile and bounded session policy;
- stop on heartbeat loss;
- record accepted, rejected and executed actions;
- expose explicit stop/fault state.

### Swift conformance package

The repository SHOULD publish or generate policy manifests and fixtures consumed by `LeRobotCoreAIKit` in `lerobot-coreai-apple`.

It MUST NOT contain the SwiftUI application itself.

### Dataset export

`lerobot-coreai` SHALL convert the app's `org.lerobot.episode-staging.v1` recording into the exact upstream LeRobotDataset format, validating synchronization, required features and artifact/session identities.

### Gemma boundary

The repository MAY define skill schemas and planner-policy handoff fixtures. It MUST NOT make Gemma a policy or allow planner output to bypass action-policy and gateway validation.

## 20. CI

- base without LeRobot;
- stable LeRobot;
- pinned dev LeRobot;
- real official CLI lanes;
- interop conformance;
- Fabric bundle fixture;
- Runner real asset lane on Apple Silicon;
- evidence replay;
- safety invariants;
- no physical egress in CI;
- documentation claim checks;
- protocol legacy compatibility during migration;
- test namespace and production namespace separation.

## 21. Definition of Done for v1.4

- Real ACT artifact produced by Fabric.
- Real Swift Runner executes it.
- Official five-case matrix passes.
- PyTorch/Core AI parity passes under pinned policy.
- Catalog records L4.
- Production certificate is signed under protected trust.
- Claims are scoped to exact artifact/device/runtime.
- No claim of total LeRobot parity or physical safety.
- The reference gateway rejects replayed, stale, mismatched and unauthenticated action chunks.
- Mobile recordings convert deterministically to a valid upstream LeRobotDataset.
- The Swift app passes policy-contract fixtures without redefining LeRobot semantics.
- Training remains upstream and documented.
