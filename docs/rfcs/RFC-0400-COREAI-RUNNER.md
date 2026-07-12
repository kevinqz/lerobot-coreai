# RFC-0400 — `coreai-runner` Microkernel, Provider Registry and Truthful Runtime Service

> **Status:** Proposed  
> **Date:** 2026-07-12  
> **Target:** Apple-first, upstream-compatible Core AI ecosystem  
> **Normative language:** MUST, MUST NOT, SHOULD, SHOULD NOT and MAY are used as in RFC 2119.  
> **Snapshot:** See `SOURCE-SNAPSHOT.md`. The RFCs describe the target architecture; they do not claim that unfinished capabilities already exist.


## 1. Repository snapshot

- Repository: `https://github.com/kevinqz/coreai-runner`
- Reviewed commit: `54c7031267b7b02b3193aef09639966c9c038941`
- Current role: Swift runtime/service with UDS HTTP, Catalog client, cache, LLM/vision adapters; action route is scaffolded but returns 501.

The README's “Architecture phase” label is obsolete relative to the code and MUST be corrected.

## 2. Mission

Runner is the reusable local Swift execution engine and optional localhost service for Apple Core AI artifacts.

It owns:

- artifact loading;
- provider selection;
- native graph/function execution;
- sessions and state;
- cache and memory pressure;
- scheduling and concurrency;
- telemetry;
- local UDS service;
- capability discovery;
- execution receipts.

It does not own model conversion, Catalog truth, consumer UX, LeRobot semantics, robot hardware or LAN authorization.

## 3. Immediate P0 corrections

Before adding features:

1. `supports.action` MUST be false while action inference returns 501.
2. `supports.host_loop` MUST be false unless a conformance case executes a host loop.
3. Swift and Python action request nesting MUST be aligned.
4. Protocol version MUST be explicit.
5. Every advertised capability MUST be generated from built providers and passing conformance.
6. README, architecture and package status MUST reflect current implementation.
7. `coreai-server` dependency MUST use a released Runner version, not `branch: main`.
8. The Catalog decoder MUST follow a released schema instead of accepting drift through ad-hoc flexible decoding.
9. No consumer may assume `/tmp/coreai-runner.sock` ownership without a lifecycle token.

## 4. Target module structure

```text
Sources/
├── CoreAIRuntimeCore/
│   ├── Artifact/
│   ├── Providers/
│   ├── Execution/
│   ├── Sessions/
│   ├── Cache/
│   ├── Telemetry/
│   └── Evidence/
├── CoreAIProfileLanguage/
├── CoreAIProfileVision/
├── CoreAIProfileDiffusion/
├── CoreAIProfileAction/
├── CoreAIRunnerService/
└── CoreAIRunnerCLI/
```

Package products MAY expose:

- `CoreAIRuntimeCore`;
- selected profile libraries;
- `CoreAIRunnerService`;
- the full executable.

## 5. Provider registry

Required interface:

```swift
public protocol RuntimeProvider: Sendable {
    static var descriptor: ProviderDescriptor { get }
    func probe(_ artifact: ArtifactDescriptor) async -> ProbeResult
    func load(_ artifact: VerifiedArtifact, options: LoadOptions) async throws -> any ModelSession
}
```

Provider priority:

1. Apple official domain pipeline.
2. Direct Apple `AIModel`.
3. Community CoreAIKit.
4. Explicit custom provider.

A provider MUST report:

- why it matched;
- which native APIs it uses;
- supported OS/SDK;
- profiles;
- limitations;
- evidence identity.

Provider choice is included in telemetry and receipts.

## 6. Apple compatibility

Runner MUST directly support:

- raw `.aimodel`/`.aimodelc`;
- official `ModelBundle` 0.2;
- `FunctionMap`;
- official Apple Swift libraries when present;
- OS/SDK availability probing;
- direct named `AIModel` functions and state.

CoreAIKit remains supported but is labeled community-provider, not the single foundation.

The build SHOULD consume official `apple/coreai-models` releases or pinned revisions directly where license and packaging permit.

## 7. Generic invocation kernel

Replace the growing `AdapterInput`/`AdapterOutput` union with generated interop types:

```swift
func invoke(
    session: SessionID,
    profile: ProfileID,
    inputs: [String: NamedValue],
    context: ExecutionContext
) async throws -> InvocationResult
```

Domain adapters map convenience APIs into this function.

The kernel MUST NOT include prompt, box, point, action or robot fields directly. It only knows named values and execution context.

## 8. Action profile

`CoreAIProfileAction` is first-party.

It owns:

- observation input mapping;
- action tensor/chunk output;
- execution plan;
- host iterative loops;
- state/session scope;
- native and split batching;
- deterministic seed;
- action telemetry.

It does not know:

- robot brands;
- USB/serial/CAN;
- calibration;
- emergency stop;
- motor safety;
- LeRobot `Robot` classes;
- physical task success.

## 9. Execution-plan interpreter

Provide a small non-Turing-complete interpreter with bounded operations:

- `call`
- `bind`
- `repeat` with maximum count
- `select_static_variant`
- `emit`
- `reset_state`

Plans are validated before execution and resource bounded.

This engine is shared by diffusion and action profiles. A profile-specific hand-written host loop MAY exist temporarily but SHOULD migrate to an execution plan when semantics stabilize.

## 10. Sessions and concurrency

Define:

- session identity;
- model session versus request;
- inference state scope;
- reset scope;
- isolation;
- concurrent invoke guarantees;
- cancellation;
- deadlines;
- per-session queue limits;
- idempotency where possible.

Global mutable inference state MUST be explicitly advertised and incompatible with isolated batching claims.

Native batching and split-and-stack MUST be separate capability modes, each with maximum size and slot-isolation behavior.

## 11. Model cache

Current actor/LRU is a useful foundation. Extend it with:

- verified artifact root as cache key;
- provider in key;
- compiled specialization identity;
- memory-cost estimate;
- lease/reference count;
- consumer ownership;
- sticky lease expiry;
- atomic load state;
- structured eviction reason;
- thermal policy;
- no Catalog requirement for local paths;
- failed-load cooldown;
- corruption quarantine.

A model ID alone is not a cache identity.

## 12. Service lifecycle

### Embedded mode

Consumer launches a private Runner with:

- unique socket path;
- parent PID;
- lifecycle token;
- binary/protocol compatibility check;
- graceful shutdown;
- ownership proof before cleanup;
- sanitized environment.

This is the default for ComfyUI and local LeRobot.

### Broker mode

Deferred. A shared daemon requires:

- lock and PID file;
- authentication;
- reference counting;
- multi-tenant isolation;
- version negotiation;
- upgrade protocol;
- lease semantics.

It MUST NOT be approximated by several applications sharing `/tmp/coreai-runner.sock`.

## 13. Security

- UDS socket permissions restricted to the user.
- No arbitrary filesystem reads from request paths.
- Inputs use controlled blob directories, file descriptors or validated paths.
- Artifact roots verified before load.
- Downloaded binaries and artifacts fail closed on signature/digest failure.
- Resource limits prevent decompression/tensor bombs.
- Logs redact prompts, images and action arrays by default.
- Environment and executable identity captured in evidence mode.
- Model loading and inference may have different authorization in broker/server mode.
- Temporary files are atomic and cleaned by ownership token.

## 14. Catalog integration

Catalog client is optional and should use generated/stable types.

Runner APIs:

```swift
load(path:digest:)
load(catalogID:resolvedArtifact:)
```

The second resolves before load; runtime execution always uses a verified immutable descriptor.

Runner MUST support offline operation after installation.

## 15. Receipts

A runtime receipt SHOULD include:

- Runner binary root and code signature;
- provider ID/version;
- artifact root;
- profile and execution-plan root;
- request/response transcript root;
- session/reset events;
- device/OS/toolchain;
- compute policy;
- timings and thermal state;
- nonce/challenge for certification;
- error/retry events.

No caller-supplied `real_runner_used` boolean is evidence.

## 16. HTTP API

Required endpoints:

- `/v1/health`
- `/v1/capabilities`
- `/v1/artifacts/resolve`
- `/v1/models/load`
- `/v1/models/unload`
- `/v1/invoke`
- `/v1/sessions/reset`
- `/v1/receipts/<built-in function id>` when evidence enabled

Convenience endpoints remain adapters:

- `/v1/chat/completions`
- `/v1/action/predict`
- vision-specific routes as justified.

All endpoints use one error taxonomy.

## 17. Capability generation

At build or startup:

1. Discover compiled profile targets.
2. Probe OS frameworks.
3. Run or load conformance attestations.
4. Produce capability response.

States:

- unavailable;
- compiled;
- available;
- conformant;
- experimental;
- degraded.

Booleans may be derived for legacy clients but are not authoritative.

## 18. Release variants

A single codebase MAY publish:

- `coreai-runner-minimal`
- `coreai-runner-comfy`
- `coreai-runner-action`
- `coreai-runner-full`

These are build compositions, not separate repositories.

Every binary release contains:

- version;
- protocol range;
- included providers/profiles;
- SHA-256;
- code-signing identity;
- notarization;
- artifact attestation;
- SBOM.

## 18.1 RuntimeCore and Apple mobile

The current monolithic product boundary SHALL be split into:

```text
CoreAIRuntimeCore
├── macOS
├── iOS
└── no HTTP/Catalog requirement

CoreAIRunnerService
├── macOS
├── UDS/HTTP
└── process lifecycle
```

`CoreAIRuntimeCore` is the only Runner product imported by `lerobot-coreai-apple` and Ditto.

It MUST expose:

- local artifact loading by URL/path and digest;
- provider registry;
- language and action sessions;
- invocation, cancellation and deadlines;
- reset;
- telemetry;
- memory/load estimates when available;
- no dependency on Hummingbird;
- no assumption that Catalog is online.

The service target wraps RuntimeCore. It MUST NOT contain model semantics unavailable to in-process consumers.

### Mobile model admission

RuntimeCore SHALL report enough data for the app to decide whether planner and policy can coexist. It MUST surface load failure, specialization pressure and memory pressure distinctly.

### Mobile lifecycle

RuntimeCore MUST support explicit suspension/reset. It MUST NOT imply that an inference session survives iOS backgrounding or process suspension.

### Action provider

The first mobile Action provider MUST run the same profile and fixtures as the macOS service provider. Device-specific graph variants are selected by manifest/provider capability, not by model-name inference.

## 19. Migration plan

### R1 — Truth fix

Capabilities and action payload alignment.

### R2 — Interop adoption

Generated Swift types, legacy endpoint compatibility.

### R3 — Provider extraction

Wrap current adapters behind providers.

### R4 — Apple official provider

Direct official bundle/pipeline integration.

### R5 — Generic invoke kernel

Adapters delegate to it.

### R6 — Action provider

ACT real graph, execution plan and conformance.

### R7 — Evidence

Signed production runtime receipts.

### R8 — RuntimeCore mobile split

Extract an iOS-compatible library, build a sample app and prove the same language/action fixtures in-process.

### R9 — Stable release

Versioned SPM dependencies and signed binaries/libraries.

## 20. CI

- Swift unit tests;
- generated-code freshness;
- interop conformance;
- malicious input fixtures;
- provider probe matrix;
- real Apple asset smoke on macOS;
- concurrency and cancellation;
- cache/thermal simulation;
- backward protocol tests;
- Server import test;
- ComfyUI and LeRobot consumer contract tests;
- release signature verification;
- no false capability check.

## 21. Definition of Done

- No capability contradicts behavior.
- No domain requires editing a god request type.
- Official Apple providers are first-class.
- CoreAIKit is isolated as a provider.
- Private embedded lifecycle is safe.
- ACT executes as a real `.aimodel`.
- LeRobot's five-case matrix passes against the Swift Runner.
- Receipts derive from observed execution.
- The macOS library and binary have stable released versions.
- The released RuntimeCore builds for iOS and executes language/action conformance fixtures in-process.
