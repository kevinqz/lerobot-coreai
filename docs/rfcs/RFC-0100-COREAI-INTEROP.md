# RFC-0100 — Create `coreai-interop`: Apple-Subordinate Interoperability Profiles and Conformance

> **Status:** Proposed  
> **Date:** 2026-07-12  
> **Target:** Apple-first, upstream-compatible Core AI ecosystem  
> **Normative language:** MUST, MUST NOT, SHOULD, SHOULD NOT and MAY are used as in RFC 2119.  
> **Snapshot:** See `SOURCE-SNAPSHOT.md`. The RFCs describe the target architecture; they do not claim that unfinished capabilities already exist.


## 1. Decision

Create one new public repository:

```text
kevinqz/coreai-interop
```

The repository MUST NOT present itself as “the Core AI specification.” It is an interoperability layer subordinate to Apple's official formats and APIs.

Recommended tagline:

> Versioned cross-process protocols, domain profiles, generated types and conformance tests for the open Core AI ecosystem.

## 2. Why a repository is required

The same contracts currently exist in multiple incompatible forms:

- Swift Codables in Runner;
- Python payload construction in `lerobot-coreai`;
- Python payload construction and Catalog parsing in ComfyUI-CoreAI;
- Fabric recipe and generated manifest schemas;
- Catalog closed enums and IO contracts;
- Server's future LAN protocol.

Keeping these contracts inside one implementation causes circular ownership. A language-neutral contract repository establishes:

- a single released protocol version;
- generated Swift and Python types;
- golden request/response fixtures;
- compatibility tests runnable by every repository;
- an explicit place for profile evolution;
- a release process independent of any consumer.

## 3. Scope

### Included

- local Runner HTTP/UDS OpenAPI;
- optional Server transport envelope extensions;
- error taxonomy;
- version negotiation;
- artifact descriptors;
- named values and blob references;
- sessions, resets, deadlines and cancellation;
- telemetry envelope;
- provider and profile capability manifests;
- language, vision, diffusion and action profile schemas;
- LeRobot consumer profile;
- signed receipt references, not private signing keys;
- generated Swift/Python models and clients;
- conformance fixtures and test harnesses;
- RFCs for cross-repository contracts.

### Excluded

- `.aimodel` internals;
- Apple's `ModelBundle` redefinition;
- converter implementation;
- model registry data;
- runtime implementation;
- LeRobot policy code;
- ComfyUI nodes;
- robot drivers;
- model weights.

## 4. Repository layout

```text
coreai-interop/
├── README.md
├── GOVERNANCE.md
├── VERSIONING.md
├── rfcs/
├── schemas/
│   ├── common/
│   │   ├── artifact-descriptor.v1.json
│   │   ├── named-value.v1.json
│   │   ├── blob-reference.v1.json
│   │   ├── telemetry.v1.json
│   │   └── error.v1.json
│   ├── capabilities/
│   │   ├── runner-capabilities.v1.json
│   │   ├── provider-manifest.v1.json
│   │   └── profile-manifest.v1.json
│   └── profiles/
│       ├── language.v1.json
│       ├── vision.v1.json
│       ├── diffusion.v1.json
│       ├── action.v1.json
│       ├── language-planner.v1.json
│       ├── lerobot-policy.v1.json
│       ├── robot-brain-skill-plan.v1.json
│       ├── mobile-runtime.v1.json
│       └── episode-staging.v1.json
│   └── robot-gateway/
│       ├── capabilities.v1.json
│       ├── session.v1.json
│       ├── observation-state.v1.json
│       ├── action-chunk.v1.json
│       ├── action-ack.v1.json
│       ├── heartbeat.v1.json
│       ├── stop.v1.json
│       └── fault.v1.json
├── openapi/
│   ├── runner-v1.yaml
│   └── server-v1.yaml
├── generated/
│   ├── python/coreai_interop/
│   └── swift/Sources/CoreAIInterop/
├── fixtures/
│   ├── valid/
│   ├── invalid/
│   └── compatibility/
├── conformance/
│   ├── python/
│   ├── swift/
│   └── scripts/
├── tools/
│   ├── generate.py
│   ├── breaking_change.py
│   └── lint_examples.py
└── .github/workflows/
```

## 5. Compatibility with Apple

Every schema MUST clearly classify its relationship to Apple:

```yaml
native_authority: apple
native_contract:
  repository: apple/coreai-models
  metadata_version: "0.2"
extension_namespace: "org.coreai-interop"
```

The repository SHALL never:

- add unsupported values to Apple's `BundleKind` and call them official;
- mutate official `metadata.json` semantics;
- wrap official bundle fields under conflicting names;
- claim Apple endorsement;
- require Runner-specific metadata in a native Apple bundle.

Profiles SHALL compose or reference official artifacts.

## 6. Common invocation kernel

### 6.1 NamedValue

Supported initial kinds:

- `tensor`
- `image`
- `text`
- `json`
- `blob`

Tensor MUST specify dtype and shape. Large values SHOULD be represented by content-addressed blobs.

### 6.2 ArtifactDescriptor

Must support:

- Catalog ID, when known;
- local path;
- immutable root digest;
- media type;
- native format;
- optional outer-profile bundle;
- native Apple bundle metadata relation;
- required provider/profile.

A request MUST NOT rely solely on a mutable model ID.

### 6.3 ExecutionContext

Fields:

- request ID;
- session ID;
- deadline;
- cancellation token relation;
- deterministic seed, when the profile allows one;
- compute policy;
- trace/evidence mode;
- privacy flags.

## 7. Profiles

### 7.1 `coreai.language.v1`

Defines prompt/messages, generation parameters, streaming events and token telemetry. OpenAI-compatible routes MAY map into it.

### 7.2 `coreai.vision.v1`

Defines image inputs and named outputs without forcing one result shape. Detection and segmentation MAY be subprofiles.

### 7.3 `coreai.diffusion.v1`

Defines generation resources, scheduler/execution-plan reference, seed and progressive outputs.

### 7.4 `coreai.action.v1`

Defines:

- observation named values;
- state/image/task modalities;
- action tensor/chunk;
- batching semantics;
- inference-state scope;
- reset semantics;
- host execution plan;
- deterministic/random sampling declaration;
- action representation metadata.

It MUST NOT define hardware buses, motor commands or safety certification.

### 7.5 `org.huggingface.lerobot.policy.v1`

Defines the mapping between LeRobot feature names and action profile inputs/outputs:

- LeRobot version range;
- policy type/family;
- `FeatureContract`;
- processor ownership;
- action chunk semantics;
- task/language feature;
- config and checkpoint identity;
- official plugin compatibility.

LeRobot owns the semantics; this profile serializes the binding.

## 7.6 `coreai.language-planner.v1`

Defines a bounded planning surface for on-device language models:

- planner artifact identity;
- supported structured-generation mode;
- skill schema references;
- context and memory limits;
- plan timeout and cancellation;
- tool/skill capability declarations;
- `SkillPlan` output;
- explicit prohibition on raw motor commands.

## 7.7 `org.lerobot.robot-brain.skill-plan.v1`

Defines a validated plan composed only of registered skills, typed arguments, bounded retry policy and operator-confirmation requirements.

## 7.8 `org.lerobot.robot-gateway.v1`

Defines pairing, session, state streaming, action-chunk submission, acknowledgements, heartbeat, stop, fault and receipt messages.

Every actuation-bearing message includes session identity, sequence, expiry/deadline, robot identity, policy root and action-contract version.

## 7.9 `coreai.mobile-runtime.v1`

Defines device-scoped runtime capabilities:

- OS and chip class;
- provider availability;
- AOT and entitlement requirements;
- memory/load estimates;
- current thermal/memory state;
- supported coexistence sets;
- foreground/background restrictions.

## 7.10 `org.lerobot.episode-staging.v1`

Defines the append-only mobile recording manifest used before canonical conversion to LeRobotDataset.

## 8. Capabilities

Capability manifests MUST distinguish:

- compiled-in;
- available on the current OS/SDK;
- available for the current artifact;
- conformant;
- experimental;
- unavailable.

A boolean-only capability model is insufficient.

## 9. Code generation

Generated clients MUST be committed and reproducible.

Targets:

- Python 3.10+ Pydantic/dataclass models;
- Swift 6 Codable/Sendable types;
- optional TypeScript later.

Generation CI MUST fail if generated code is stale.

## 10. Conformance suite

Each implementation SHALL run a black-box conformance executable:

```bash
coreai-interop-conformance runner --endpoint unix:///...
coreai-interop-conformance server --endpoint https://...
```

Required checks:

- version negotiation;
- malformed request rejection;
- unknown profile rejection;
- named tensor round trip;
- deadline behavior;
- cancellation;
- session reset;
- truthful capability agreement;
- error code stability;
- artifact digest mismatch;
- profile-specific fixtures.

Action conformance MUST include the exact payload that previously diverged between Swift and Python.

## 11. Versioning

- Semantic versioning for the repository.
- Protocol major versions in the URL/media type or explicit field.
- Additive optional fields MAY be minor.
- New required fields require a major version.
- Generated libraries share the protocol version.
- Profiles version independently but are released in one BOM.

## 12. Governance

Changes to common kernel require approval from at least two implementation owners representing different consumers.

Examples:

- Runner + LeRobot;
- Runner + ComfyUI;
- Fabric + Catalog.

Profile-only changes require the profile owner and one runtime owner.

## 13. Initial bootstrap plan

### Release 0.1

Capture current protocol exactly, including legacy fixtures. Mark known contradictions.

### Release 0.2

Introduce corrected request envelope and generated clients. Runner supports both legacy and v1 under explicit negotiation.

### Release 0.3

Add action and LeRobot profiles plus black-box conformance.

### Release 0.4

Add mobile runtime, language-planner, SkillPlan, Robot Gateway and episode-staging contracts. Ship an app↔gateway mock conformance harness.

### Release 1.0

Declared stable only after Runner, ComfyUI and LeRobot use generated clients and the release BOM pins 1.0.

## 14. Definition of Done

- Runner's Swift Codables are generated or wrap generated types.
- ComfyUI no longer hand-constructs protocol payloads.
- LeRobot no longer hand-constructs protocol payloads.
- Server uses the same request and response types.
- Fabric outputs profile-valid manifests.
- Catalog validates and indexes profile references.
- Golden fixtures run in all ecosystem repositories, including `lerobot-coreai-apple`.
- App↔gateway replay, expiry, disconnect and stop fixtures pass in Swift and Python.
- No Apple-native schema is redefined.
