# lerobot-coreai

**Apple CoreAI runtime backend for LeRobot policies.**

This repo ships **two distributions**:

- **`lerobot-coreai`** (base package) — a CoreAI runtime toolkit for LeRobot-shaped policy artifacts: it runs LeRobot-compatible policies as Apple CoreAI `.aimodel` artifacts, with custom dataset replay, simulation, safety, governance, and guarded-real workflows. The base package's local bridge (`CoreAIPolicy`) is duck-typed and runtime-only (not a `PreTrainedPolicy`); its `select_action()` follows the project-local contract (see below).
- **`lerobot_policy_coreai_bridge`** (companion plugin, `packages/`) — a real **official out-of-tree LeRobot plugin**: `CoreAIBridgePolicy` **is** a `PreTrainedConfig`-registered `PreTrainedPolicy`/`nn.Module` whose `select_action(batch) -> torch.Tensor(B, A)` follows LeRobot per-timestep semantics. It is discovered by `register_third_party_plugins()`, loads through `make_policy`/`make_pre_post_processors`, and has a hermetic B=1 factory E2E on LeRobot 0.6.0 + 0.6.1-dev. Official `lerobot-eval` certification is still pending. See [docs/official-lerobot-plugin.md](docs/official-lerobot-plugin.md) and [docs/lerobot-compatibility-levels.md](docs/lerobot-compatibility-levels.md).

Use **LeRobot** for recording, training, datasets, robots, processors, and PyTorch policy deployment.

Use **`lerobot-coreai`** when you want to export, inspect, evaluate, dry-run, shadow-run, simulate, or roll out a LeRobot policy through Apple CoreAI; install the **companion plugin** to run a CoreAI policy through the official LeRobot factory.

> **LeRobot-shaped custom workflows + an official out-of-tree plugin. CoreAI runtime.** (Official `lerobot-eval` certification is still pending — see the compatibility levels doc.)

> **Current:** `inspect`, `doctor`, `predict`, `rollout --mode dry_run`, `shadow` (motor-blocked), `eval` (LeRobotDataset replay), `compare` (PyTorch vs CoreAI parity), `export`, `sim` (simulator-only egress), the safety/governance chain (`supervisor-check`, `profile-*`, `safety-gate`, `safety-regression`, `approval-request`/`approve-bundle`/`verify-approval`, `release-readiness`), and — since v1.0.0 — `real --mode guarded` (guarded real egress) with `verify-real-session`.
> Up to v0.9.3 **no robot commands are ever sent**; v1.0.0 introduces real egress **only** through `real --mode guarded`, behind every gate. This is guarded real egress, not native LeRobot robot integration, and proves nothing about physical safety.
> **Base package** `CoreAIPolicy.select_action()` follows the project-local CoreAI contract and may return an action **chunk** for chunked policies; use `select_next_action()` for per-timestep actions and `predict_action()` for dict+metadata. The **companion plugin** `CoreAIBridgePolicy.select_action(batch)` returns per-timestep `torch.Tensor(B, A)` (LeRobot semantics) and is the sanctioned official path.

---

## What this is

`lerobot-coreai` is **not** a new robotics framework. It is the CoreAI runtime backend for LeRobot-compatible policies. LeRobot remains the source of truth for robot learning concepts. CoreAI Fabric exports and verifies artifacts. CoreAI Catalog indexes compatibility and provenance. CoreAI Runner executes `.aimodel` graphs. CoreAI Server exposes Runner remotely. `lerobot-coreai` makes all of that feel like one additional LeRobot runtime.

## What this is not

- Not an alternative to LeRobot or LeLab
- Not a GUI, teleop stack, or training stack
- Not a new dataset format or simulator
- Not a fork of LeRobot
- Not a substitute for coreai-fabric, coreai-runner, or coreai-catalog

## Install

```bash
# Minimal — inspect, metadata parsing, catalog lookup
pip install lerobot-coreai

# With LeRobot — eval, rollout helpers, LeRobotDataset replay
pip install "lerobot-coreai[lerobot]"

# With Fabric — export orchestration
pip install "lerobot-coreai[fabric]"

# With Gym — gymnasium simulator adapter for sim mode
pip install "lerobot-coreai[sim]"

# Full
pip install "lerobot-coreai[all]"
```

## Quick start

### Inspect a CoreAI-backed LeRobot policy

```bash
lerobot-coreai inspect --policy.path kevinqz/EVO1-SO100-CoreAI
```

```
Policy: EVO1
Runtime: CoreAI
Artifact: kevinqz/EVO1-SO100-CoreAI
Robot type: so100
LeRobot version: 0.6.0
Observation features:
  - observation.images.wrist: [3, 224, 224]
  - observation.state: [7]
Action features:
  - action: [16, 7]
CoreAI parity: passed
Recommended next step: rollout --mode dry_run
```

### Python API

```python
from lerobot_coreai import CoreAIPolicy

policy = CoreAIPolicy.from_pretrained(
    "kevinqz/EVO1-SO100-CoreAI",
    runner_url="http://127.0.0.1:8710",
)

batch = {
    "observation.images.wrist": "/tmp/wrist.png",
    "observation.state": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "task": "pick up the cube",
}

# Legacy CoreAI contract: returns the raw action (a chunk [H,A] for chunked policies)
action = policy.select_action(batch)

# LeRobot-correct per-timestep semantics: returns one action [A] from an internal queue
next_action = policy.select_next_action(batch)

# Debug/CLI-style: returns dict with action + metadata
result = policy.predict_action(batch, return_metadata=True)
```

> **Without a runner:** `from_pretrained(repo_id)` without `runner_url` loads metadata only.
> `select_action()` / `predict_action()` will raise `RunnerNotReachableError`.

### Doctor — check compatibility

```bash
lerobot-coreai doctor --policy.path kevinqz/EVO1-SO100-CoreAI --robot.type so100
```

## CLI commands

| Command | Status | Purpose |
|---------|--------|---------|
| `inspect` | v0.1 ✅ | Inspect a CoreAI-backed LeRobot policy |
| `doctor` | v0.1 ✅ | Metadata + runner compatibility checks |
| `list` | v0.1 ✅ | List LeRobot policies from the catalog |
| `predict` | v0.2 ✅ | Predict action from single observation |
| `rollout --mode dry_run` | v0.3 ✅ | Fixture-based dry-run; no robot actuation |
| `eval` | v0.4 ✅ | LeRobotDataset replay/eval; no robot actuation |
| `compare` | v0.5 ✅ | PyTorch vs CoreAI action parity on LeRobotDataset |
| `export` | v0.6 ✅ | Export/verify/package LeRobot policy as CoreAI artifact |
| `shadow` | v0.7 ✅ | Motor-blocked observation loop; actions generated and logged, never sent |
| `sim` | v0.8 ✅ | Simulator-only action egress; actions drive a simulator, never a robot |
| `sim-regression` | v0.8.3 ✅ | Compare two sim runs for regression |
| `package-sim-run` | v0.8.4 ✅ | Package a sim run into a reproducibility bundle |
| `verify-sim-bundle` | v0.8.4 ✅ | Verify a sim bundle (manifest, checksums, invariants) |
| `supervisor-check` | v0.9.0 ✅ | Evaluate an actions file against a safety profile |
| `profile-list` / `profile-show` | v0.9.1 ✅ | List / inspect built-in safety profiles |
| `profile-validate` | v0.9.1 ✅ | Validate a safety profile |
| `profile-recommend` | v0.9.1 ✅ | Recommend a built-in profile from policy/actions |
| `profile-calibrate` | v0.9.1 ✅ | Calibrate a profile from an actions log |
| `profile-compare` | v0.9.1 ✅ | Compare two profiles over the same actions |
| `safety-gate` | v0.9.2 ✅ | Evaluate a safety summary/run/bundle against safety quality gates |
| `safety-regression` | v0.9.2 ✅ | Compare baseline vs candidate safety summaries for regressions |
| `approval-request` | v0.9.3 ✅ | Build an operator approval checklist for a sim evidence bundle |
| `approve-bundle` | v0.9.3 ✅ | Create an operator approval manifest bound to artifact hashes |
| `verify-approval` | v0.9.3 ✅ | Verify an approval manifest against a bundle |
| `release-readiness` | v0.9.3 ✅ | Produce a final readiness report from bundle + approval |
| `real` | v1.0.0 ✅ | Guarded real mode: preflight or bounded guarded session |
| `verify-real-session` | v1.0.2 ✅ | Offline audit of a completed guarded real session |
| `lerobot-bridge-check` | v1.1.0 ✅ | Probe the local LeRobot bridge for a CoreAI policy (no robot action) |
| `lerobot-compat-check` | v1.1.2 ✅ | LeRobot compatibility certificate; `--contract` for the leveled v1 report (v1.2.4) |
| `lerobot-registry-check` | v1.1.3 ✅ | Check the local (non-upstream) LeRobot registry adapter |
| `eval-v2` | v1.1.4 ✅ | Auditable dataset↔policy feature mapping (not an action replay) |
| `obs-bridge-check` | v1.1.5 ✅ | Confirm a LeRobot frame maps to the CoreAI observation |
| `hf-metadata` | v1.1.6 ✅ | Emit + validate honest HF-style artifact metadata |
| `package-bridge-benchmark` / `verify-bridge-benchmark` | v1.1.7 ✅ | Bundle + verify reproducible bridge benchmark packs |
| `provenance-create` / `sign-artifact` / `verify-signature` | v1.2.0 ✅ | Ed25519 provenance + signing + trust-policy verification (`[signing]`) |
| `release-check` | v1.2.1 ✅ | Evaluate an artifact against a per-channel release policy |
| `artifact-index` | v1.2.2 ✅ | Local registry for signed/verified artifacts (init/add/list/find/verify) |
| `policy-card` | v1.2.3 ✅ | Generate an honest policy card from verified evidence |
| `compare-v2` | v1.2.6 ⚠️ | Processor-inclusive parity via the official loader — **experimental**; parity claimed only when numeric tolerance gates pass (v1.2.7) |
| `eval-v3` | v1.2.9 ✅ | Real action replay through the policy (frames evaluated, per-action validation, latency); sends nothing to a robot/sim |
| `lerobot-compat-check --plugin` | v1.3.1 ✅ | Companion-plugin compatibility profile (discovery/config/class/factory/processors); `official_eval` stays `not_tested` |
| `package-lerobot-plugin-artifact` | v1.3.6 ✅ | Build a canonical, factory-loadable plugin artifact (needs the companion plugin) |
| `verify-lerobot-plugin-artifact` | v1.3.7 ✅ | Verify artifact integrity/semantics fail-closed; `--output-dir` writes a report outside the sealed artifact (v1.3.8) |

## Safety model

v0.7 adds motor-blocked shadow mode.
v0.7.1 adds optional local camera observation source for shadow mode.
v0.7.2 adds observation adapters, live metrics, and run quality diagnostics.
v0.8 adds simulator-only sim mode: actions drive a simulator, never a robot.
v0.8.1 adds a gymnasium simulator adapter (`[sim]` extra) for sim mode.
v0.8.2 adds sim analytics: CSV exports, markdown summaries, failure taxonomy, and richer report sections for simulator-only runs.
v0.8.3 adds sim quality gates and a sim-regression command to compare two sim runs for regression.
v0.8.4 adds reproducibility bundles for simulator-only runs, including manifests, checksums, environment metadata, runner metadata, and audit-ready package outputs.
v0.9.0 adds a runtime safety supervisor that validates, bounds, clips, blocks, and audits actions before egress. It is a software safety layer for simulator and future guarded real-mode workflows. It does not prove physical robot safety and does not enable unrestricted real-world actuation.
v0.9.1 adds robot-family safety profiles and a calibration toolkit (software action-bound contracts; not hardware certification).
v0.9.2 adds supervisor quality gates and a safety regression harness that can fail CI on unsafe actions or safety regressions. Software CI layer only; does not prove physical robot safety.
v0.9.3 adds an operator approval protocol and release-readiness evidence: a named operator must explicitly approve a checksummed evidence bundle (safety gates, regression, calibration) before it is marked release-ready. Approval does not prove physical safety and does not authorize real-world actuation.
v1.0.0 adds guarded real mode: the first sanctioned real-egress path. An action reaches a robot adapter only in `real --mode guarded`, only after verified readiness, valid operator approval, enforced supervisor, a bounded session, and explicit operator attestations — and only through the fail-closed RealEgressGuard. It does not prove physical robot safety and does not authorize unrestricted real-world actuation.
Shadow mode can read observations and generate actions.
Shadow mode cannot send actions to a robot, motor, simulator, or actuator.
Sim mode can send actions to a simulator.
Sim mode cannot send actions to a robot.
Sim and shadow never connect to a robot or send motor commands; only guarded real mode can, behind every gate above.
Export verification can prove numeric action fidelity only when compare passes.
It cannot prove task success or physical robot safety.

| Mode | Status | Behavior |
|------|--------|----------|
| `dry_run` | v0.3 ✅ | No physical robot. Fixture-based action generation. |
| `shadow` | v0.7 ✅ | Observations streamed/replayed, actions generated and logged, never sent. |
| `sim` | v0.8 ✅ | Actions drive a simulator; never a robot. Requires `--confirm-sim-egress`. |
| `real` | v1.0.0 ✅ | Guarded real egress. Requires verified readiness, valid approval, enforced supervisor, bounded session, and explicit operator attestations. |

> v0.8 implements sim, shadow, dry_run, eval, compare, and export.
> Sim mode sends actions to a simulator. Sim mode never sends actions to a robot.
> Sim task success is not real-world task success.
> Shadow mode is not real mode. Shadow mode is not sim mode.
> Shadow mode does not prove task success or physical safety.
> Shadow mode proves runtime action generation and no-actuation logging.
> No robot commands are sent by v0.8.

## Version policy

`lerobot-coreai` 0.8.x supports LeRobot `>=0.6.0,<0.7.0`.
v0.7.1 adds optional `[camera]` extra (OpenCV) for shadow mode camera source.
v0.8 adds simulator-only sim mode (`fake` and `replay` environments).
v0.8.1 adds a gymnasium simulator adapter (`[sim]` extra).
v0.8.2 adds sim analytics (CSV exports, markdown summaries, failure taxonomy).
v0.8.3 adds sim quality gates and the `sim-regression` command.
v0.8.4 adds reproducibility bundles (`package-sim-run` / `verify-sim-bundle`).
v0.9.0 adds a runtime safety supervisor (`--supervisor.mode`, `supervisor-check`).
v0.9.1 adds robot-family safety profiles and an offline profile calibration toolkit (validate/recommend/calibrate/compare), plus fail-closed delta verification across shape changes. Profiles are software action-bound contracts — they do not certify robot safety.
v0.9.2 turns supervisor findings into enforceable safety quality gates (`safety-gate`, `sim --safety.*`) and safety regression checks (`safety-regression`). Gates prove only that an artifact met configured software thresholds — not physical or real-world safety.
v0.9.3 adds the operator approval protocol + release-readiness evidence (`approval-request`, `approve-bundle`, `verify-approval`, `release-readiness`) — the last software gate before guarded real mode. The pre-v1.0 workflow: sim → safety-gate → safety-regression → package-sim-run → verify-sim-bundle → approval-request → approve-bundle → verify-approval → release-readiness.
v0.9.4 hardens the pre-1.0 governance layer: stricter approval/readiness schemas (conditional invariants), fail-closed type validation of safety-summary counts, explicit parseable/finite/non-finite calibration sample counts, and clearer approval-request required-vs-warnings signalling.
v1.0.0 adds guarded real mode (`real --mode preflight|guarded`): the first real-egress path, gated on the entire pre-real-mode evidence chain. Guarded real egress for CoreAI-backed, LeRobot-shaped policies — not a native upstream LeRobot integration, and not proof of physical safety.
v1.0.1 hardens the external-http adapter with an optional bearer token (`--robot.token` / `LEROBOT_COREAI_ROBOT_TOKEN`), completing the loopback-only egress boundary.
v1.0.2 adds `verify-real-session` (offline audit of a completed guarded real session: schema, action accounting, sent⇒allowed, trace order), a conditional real-report schema, a post-session real safety-quality gate, and loopback URL canonicalization.
v1.0.3 adds an external-http controller capability contract: in guarded mode the controller's `/preflight` must declare `robot_type`/`action_shape`/`supports_stop`/`supports_ready`/`max_fps`, validated against a schema and cross-checked with the requested robot type, safety-profile shape, and fps.
v1.0.4 adds real observation config (`--obs.config` / `--obs.*`, required for non-mock adapters) and evidence cross-binding (the run's policy/robot type must match the bundle's `sim_report`).
v1.0.5 adds per-step real-session metrics (`real_metrics.json/csv/md`: latency, effective fps, missed deadlines) and report/session redaction (`--redact-runner-url` / `--redact-operator` / `--redact-paths`).
v1.0.6 adds an arming manifest (`real_arming_manifest.json/md`: the armed envelope — limits, attestations, and SHA256 bindings of the readiness report / approval / safety profile, written before the first action) and operator abort controls (SIGINT / `--abort-file <path>` polled each step → e-stop + `operator_abort` stop reason).
v1.0.7 is a docs-consistency release: the README command surface and `docs/lerobot-compatibility.md` now reflect the full v0.9/v1.0 chain (guarded real egress exists via `real --mode guarded`; still not native LeRobot robot integration, still no physical-safety claim).
v1.1.0 adds a **local LeRobot bridge** (`load_coreai_policy_for_lerobot()` → a LeRobot-shaped, runtime-only `CoreAILeRobotPolicyBridge`; `lerobot-bridge-check` command) so CoreAI policies can be used where LeRobot expects a policy-shaped object. Duck-typed (no `PreTrainedPolicy` subclass), no global monkeypatch, no `torch`/`lerobot` import at load, works without the `[lerobot]` extra. It is a **local** bridge — `policy_type="coreai"` is not registered upstream, training is not supported, and it proves nothing about physical safety. See [docs/lerobot-native-bridge.md](docs/lerobot-native-bridge.md).
v1.1.1 hardens the external-http controller boundary: a `GET /safety-state` contract (e-stop armed + workspace clear + no faults + ready, else fail-closed) gated before any guarded egress, extended `GET /preflight` capability checks (`supports_observation` / `supports_safety_state` / `physical_estop_required`), stricter loopback URL rules (explicit `http://` scheme **and** port; remote/`0.0.0.0`/obfuscated hosts rejected), and local auth via `--robot.auth-token-env` (env-name indirection, `X-LeRobot-CoreAI-*` headers; the raw token never touches a report/trace — only a `sha256:` prefix). See [docs/external-http-controller-contract.md](docs/external-http-controller-contract.md).
v1.1.2 turns the LeRobot 0.6.x compatibility claim into a **tested certificate**: a `lerobot-compat-check` command emits `lerobot_compatibility_report.json/md`, and a dedicated Python 3.12 CI job installs the `[lerobot]` extra and runs it `--strict` (version-in-range, `PreTrainedPolicy`/`LeRobotDataset` importable, bridge shape, honest claims). The base `test` matrix still passes without LeRobot — the base package imports neither `torch` nor `lerobot`. See [docs/lerobot-compatibility-ci.md](docs/lerobot-compatibility-ci.md).
v1.1.3 adds a **local, opt-in registry adapter** (`CoreAILeRobotRegistry`, `local_lerobot_registry_patch()`, `lerobot-registry-check`): registry-style ergonomics for CoreAI policies with no global monkeypatch — the LeRobot factory is untouched by default, and the opt-in context manager patches `get_policy_class("coreai_bridge")` only inside its block and restores it on exit (even on error). `policy_type="coreai"` is refused; still not upstream registration. See [docs/lerobot-local-registry.md](docs/lerobot-local-registry.md).
v1.1.4 adds **eval-v2** — an auditable dataset↔policy feature mapping (`eval-v2` command → `lerobot_feature_mapping.json` + `lerobot_eval_v2_report.json/md`). `--strict-features` fails on missing required keys or shape mismatches; non-strict surfaces the same as warnings (nothing dropped silently). The mapping logic is pure and LeRobot-free; only dataset loading needs the `[lerobot]` extra. Proves observation-mapping coherence only — not task success, not physical safety. See [docs/lerobot-eval-v2.md](docs/lerobot-eval-v2.md).
v1.1.5 adds the **observation pipeline bridge** (`obs-bridge-check` → `obs_bridge_report.json/md`): takes a concrete LeRobot frame and confirms it becomes exactly the observation dict the CoreAI manifest expects — required-key presence, state shape, image-key resolution, task handling — with no silent drops (dropped keys are listed). Reuses the `--obs.*` config. Proves the sample's mapping only. See [docs/observation-pipeline-bridge.md](docs/observation-pipeline-bridge.md).
v1.1.6 adds **golden examples + honest HF metadata**: runnable `examples/lerobot_bridge/*` and a mock `examples/real_guarded_mock/golden_path.md`, three `docs/tutorials/*`, and an `hf-metadata` command (`build_hf_metadata`/`validate_hf_metadata`) that emits validated metadata refusing any overclaim (`native_registry`/`upstream_native`/`training`/`physical_safety_proof`/`unrestricted_actuation` must be false). See [docs/tutorials/coreai-policy-bridge.md](docs/tutorials/coreai-policy-bridge.md).
v1.1.7 adds **bridge benchmark packs** (`package-bridge-benchmark` / `verify-bridge-benchmark`): bundles the compat/bridge/registry/eval-v2/obs-bridge reports into one reproducibility pack with per-file SHA256 checksums, an auto-generated README, and a verifier with tamper detection. Fail-closed on overclaim — a report claiming physical safety, task success, or actuation authorization is refused at packaging and flagged at verify. Software artifacts only. See [docs/bridge-benchmark-packs.md](docs/bridge-benchmark-packs.md).
v1.2.0 adds **signed provenance** (optional `[signing]` extra): `provenance-create`, `sign-artifact` (Ed25519, key from `--key-env`/`--key-file`, never persisted — only the public key + `sha256:` fingerprint), and `verify-signature` with a trust policy. Verification fails closed on artifact/anchor tamper, forged signatures, untrusted signers, missing required artifacts, or forbidden claims. The base package imports no crypto. Proves origin + integrity only — not physical safety, and authorizes no actuation. See [docs/signed-provenance.md](docs/signed-provenance.md).
v1.2.1 adds **release channel governance** (`release-check`): per-channel policies (`dev`/`internal`/`public-demo`/`research`/`guarded-real-evidence`) that fail closed on missing required reports, a missing/invalid signature (when required), overclaims, raw secrets, or — on public channels — real-session/external-http artifacts; `guarded-real-evidence` requires the approval + readiness + verify-real-session chain. Overridable via `--release-policy`. See [docs/release-governance.md](docs/release-governance.md).
v1.2.2 adds an **artifact index** (`artifact-index init/add/list/find/verify`): a local registry of signed/verified bundles. `add` fails closed on a tampered artifact, an overclaim, a raw secret, or an untrusted signer, sets `signature_verified` only after a real `verify-signature` pass, and refuses to silently overwrite an existing id; `verify` re-checks every indexed artifact for drift/tamper. See [docs/artifact-index.md](docs/artifact-index.md).
v1.2.3 adds a **policy card generator** (`policy-card`): deterministic, honest model cards built from verified evidence (an artifact-index entry or direct report paths). It verifies the source bundle/index first, fails closed on tamper or overclaim, summarizes the compat/bridge/registry/eval-v2/obs/benchmark/provenance/signature/release evidence, and always includes the mandatory non-claims. See [docs/policy-card-generator.md](docs/policy-card-generator.md).
v1.2.4 is a **compatibility-truth** release: a leveled contract report (`lerobot-compat-check --contract` → `lerobot_compatibility_report_v1.json/md`) that reports each rung of the official LeRobot contract separately and honestly (action semantics, plugin discovery, config registry, processor pipeline, official eval/rollout all reported `failed`/`not_supported` today, never assumed), a CI split into a blocking **stable** (LeRobot 0.6.0) job and a pinned, non-blocking **development** job, and corrected docs (removed "same LeRobot workflow intact"; documented the local opt-in `get_policy_class` patch and that `eval-v2` is feature-mapping only). See [docs/lerobot-compatibility-levels.md](docs/lerobot-compatibility-levels.md).
v1.2.5 adds **action contract v2 + batch/reset semantics**: explicit `ActionContract`/`BatchContract` (parsed from the manifest), a per-timestep `select_next_action()` backed by an `ActionQueue` (legacy `select_action` unchanged — still a chunk for chunked policies), `reset()` that clears the queue + runner session, and a split-and-stack batching fallback. The contract report's `action_batch_contract` moves to `partial`; action semantics/tensor stay `failed` until the official plugin. See [docs/action-contract-v2.md](docs/action-contract-v2.md).
v1.2.6 adds **source loader v2 + processor-inclusive compare** (`compare-v2`): loads the source PyTorch policy via the official LeRobot API (`PreTrainedConfig.from_pretrained` → `LeRobotDatasetMetadata` → `make_policy(cfg, ds_meta=...)` → `make_pre_post_processors`, never a string `policy_type` and never the abstract base), declares a processor-ownership contract (`--strict-processors` fails closed on ambiguity), and compares the **final** action after each side's processing (`mae`/`max_abs_error`/`cosine`/`relative_mae`, with shape/finite gates). Software-only; sends no robot/sim/real action. See [docs/compare-v2.md](docs/compare-v2.md).
v1.2.7 is a **compare-v2 correctness hardening** (evidence integrity): parity is claimed **only when explicit numeric tolerance gates pass** (`--tolerance.*`; a large finite error no longer reads as parity), actions must match **structural** shape (not just flattened length), the compare target is explicit (`--compare-target next_action|action_chunk`, never mixed), and the source loader binds `cfg.pretrained_path` (+ `--policy.revision`/`--dataset.revision`) so the trained checkpoint is loaded rather than a random init. compare-v2 is marked **experimental** — not release evidence until manifest-v1 processor contracts, real processor execution, temporal alignment, and live fixtures land (v1.2.8+). Also fixes the single-action `ActionQueue` case (with float normalization). See [docs/compare-v2.md](docs/compare-v2.md).
v1.2.8 adds **manifest v1 contracts + a JSON-safe observation boundary**: the manifest schema now accepts an optional `contracts` block (`action`/`batch`/`processor`) so a published, schema-valid artifact can declare its semantics (parsers resolve v1 → v0 → inference), and `serialize_observation()` converts LeRobotDataset tensors/arrays to JSON-safe values (`{"__array__",dtype,shape}`), hashes the exact payload, and **refuses unknown objects** rather than silently coercing them. See [docs/manifest-contracts-v1.md](docs/manifest-contracts-v1.md).
v1.2.9 adds **eval-v3 — real action replay** (`eval-v3`): unlike `eval-v2` (feature-mapping only, zero frames), it actually replays frames through the policy — `frames_evaluated > 0`, per-action validation (finite + dimension vs the action contract), latency p50/p95/max, and per-episode `reset()`. Reports pin `actions_sent_to_robot`/`actions_sent_to_simulator = 0`; proves neither task success nor physical safety. See [docs/lerobot-eval-v3.md](docs/lerobot-eval-v3.md).
v1.3.0 adds the **official out-of-tree LeRobot plugin** — a companion package `packages/lerobot_policy_coreai_bridge` that registers `coreai_bridge` via LeRobot's official `PreTrainedConfig.register_subclass` (no monkeypatch). `CoreAIBridgePolicy` is a real `PreTrainedPolicy`/`nn.Module` whose `select_action(batch) → torch.Tensor(B, action_dim)`; `forward`/`get_optim_params`/`train(True)` raise (runtime-only), `eval()`/`train(False)` work. `local_lerobot_registry_patch()` is now deprecated. Not `policy_type="coreai"`; not upstream-native; proves no physical safety. See [docs/official-lerobot-plugin.md](docs/official-lerobot-plugin.md).
v1.3.1 makes the plugin **factory-loadable**: `CoreAIBridgePolicy.from_pretrained` binds a CoreAI runtime (runner URL from `runner_url_env`, fail-closed, never persisted) and cross-binds the CoreAI manifest (action_dim/horizon/robot_type) fail-closed; the constructor accepts `make_policy`'s `dataset_stats`/`dataset_meta`; `select_action` returns a tensor on the policy's device; `batch_size>1` fails clearly (`single_only` until v1.3.2). Adds a separate **plugin compatibility profile** (`lerobot-compat-check --plugin`) — the base contract's `official_*` levels stay `failed` for the local bridge, while the plugin profile reports its own rungs and keeps `official_eval_certified=false` until a live E2E eval. Also fixes base evidence-integrity items: compare-v2 configured-gate-with-missing-metric now **fails** (not omitted), and eval-v3 no longer feeds ground-truth `action`/`reward` to the policy (label-leakage), resets at the start even without `episode_index`, and honors `--policy.revision`/`--episodes`. See [docs/official-lerobot-plugin.md](docs/official-lerobot-plugin.md).
v1.3.2 builds the plugin **data plane**: `transport.prepare_single_coreai_observation` bridges a real LeRobot batch to the CoreAI observation contract (strips the leading batch dim, `task: list[str]`→`str`, keeps only manifest-declared observation features — never ground-truth `action`/`reward`, converts torch/numpy to JSON-safe values, hashes the payload). `select_action` now always routes through `predict_action_chunk`; cross-binding is fully fail-closed (declared-expectation + unknown-manifest-value fails). Real plugin discovery is proven in a clean subprocess (`register_third_party_plugins()` with no manual import), and the `lerobot-dev` CI job is pinned to an exact 0.6.1-dev commit. See [docs/official-lerobot-plugin.md](docs/official-lerobot-plugin.md).
v1.3.3 hardens the plugin runtime: **local manifest loading** (`load_manifest`/`resolve_manifest` accept a local dir/file — no network — or an HF repo; a local-looking nonexistent path fails rather than hitting the Hub), a configurable observation codec (`nested_json_v1` plain lists by default; shape validated against the manifest **before** encoding; payload sha256 recorded), a **strict action-output validator** (rejects ragged/rank-4/non-finite/wrong-dim outputs), and fully fail-closed `expected_robot_type`. See [docs/official-lerobot-plugin.md](docs/official-lerobot-plugin.md).
v1.3.4 makes the codec **genuinely negotiated** and integrates the action contract: `RunnerCapabilities` now carries `protocol_version`/`observation_encodings`/`supports_batch`/`max_batch_size`, and `negotiate_observation_encoding()` picks the encoding as the intersection of config ∩ plugin-supported ∩ runner-announced (fail-closed; legacy opt-in), sending the encoding + protocol + observation sha256 in `request.options`. The plugin now passes the manifest action contract's `representation`/`horizon`/`action_dim` to the strict validator (single-action artifacts work; wrong-horizon runner outputs now fail). See [docs/official-lerobot-plugin.md](docs/official-lerobot-plugin.md).
v1.3.5 hardens the runner protocol and proves the chain composes **without mocks**: a `NegotiatedRunnerProtocol` validates the runner's announced `protocol_version` against `minimum_runner_protocol` and sends the *negotiated* version (never a hardcoded constant); a capabilities/transport failure now **propagates** instead of silently falling back to legacy; the `TypeError` compatibility fallback around `predict_action_chunk` is removed (companion requires `lerobot-coreai>=1.3.5`); action-contract parsing **fails closed** (invalid contract raises; `representation=single ⇒ horizon=1` is enforced). A hermetic B=1 end-to-end test runs a real local HTTP runner through a real `CoreAIBridgePolicy.from_pretrained → CoreAIPolicy.from_pretrained → RunnerClient → negotiation → POST → strict validation → Tensor[1, A]` with **nothing patched**. See [docs/official-lerobot-plugin.md](docs/official-lerobot-plugin.md).
v1.3.11 **closes the contracts and runs the official rollout**: it fixes a v1.3.10 flaw (native B>1 now requires `slot_isolation == 'independent'` on **both** artifact and runner — `shared`/`unknown` never enable native); adds a strict `BATCH_CONTRACT_V3_SCHEMA` (no `"false"`/`"4"` coercion); promotes `batch_contract` + `batch_contract_sha256` + `processor_stage_contract` into `plugin_artifact_manifest.json` with semantic cross-binding to the CoreAI manifest. The headline: a no-mock E2E drives the **real** `lerobot.scripts.lerobot_eval.rollout` over a deterministic `gym.vector.SyncVectorEnv` (state + front/wrist cameras + `task_description`, staggered episode termination) through the full official chain (`preprocess_observation → env/policy preprocessors → CoreAIBridgePolicy.select_action → policy/env postprocessors`), native and split, B=1/2/4, on 0.6.0 + 0.6.1-dev — proving the rollout **pipeline completes** (`Tensor[B, seq, A]`, done-masking), not eval/task-success/safety. Deferred to v1.3.12: typed processor-stage enum, `FeatureContract` v1 + real `LeRobotDatasetMetadata`, and formal signed rollout evidence. See [docs/official-lerobot-plugin.md](docs/official-lerobot-plugin.md).
v1.3.10 makes the **batch contract authoritative** and proves batching **multimodal**: `BatchContract` v3 models native and split apart (each with its own `supported`/`max_batch_size`, plus `required_slot_isolation`, `allowed_state_scopes`, `commit_semantics`, `observation_stage`), and only an authoritative v3 contract can gate B>1. The decision engine enforces native/split support, `fallback=reject` (blocks auto→split), queue layout and commit semantics — with **separate** native/split effective maxima so the runner's native max never caps split-and-stack. `slot_isolation` (`independent` required for native) is now distinct from state `scope`. Runner capabilities are **strictly typed** (a JSON `"false"` no longer becomes `True`; invalid `max_batch_size` fails) with a canonical `capabilities_sha256`; runtime observations must carry required features and a real string `task` (no silent coercion); the split batch hash is order-sensitive. A **multimodal** E2E (state + two cameras + task, B=1/2/4, native & split) runs `post(select_action(pre(batch)))` with processors **executed** (not just loaded) on 0.6.0 + 0.6.1-dev. Deferred to v1.3.11: typed processor-stage vocabulary, real `LeRobotDatasetMetadata` completeness, and official `lerobot-eval` rollout readiness. See [docs/official-lerobot-plugin.md](docs/official-lerobot-plugin.md).
v1.3.9 adds the **stateless batched runtime** (B=2/B=4) — the first version to run B>1, restricted to runners that prove requests don't mix state. `select_batch_execution_mode` fails closed on a missing/unknown `inference_state.scope`, forbids `global`, defers `session_scoped`, requires `state_isolation ∈ {stateless, request_scoped}` for native batch, and caps B at the **effective max** = `min(artifact, config, runner)`. **Native** batch = one request → `Tensor[B,H,A]`; **split-and-stack** (stateless only) = B independent requests, every sample validated before an **atomic** queue commit (failure names the sample index, queue untouched). The temporal queue follows LeRobot's chunked-policy style — a `[B,H,A]` chunk is transposed into a `deque[Tensor[B,A]]` and `select_action` returns `Tensor[B,A]`; a batch-size change while the queue is non-empty fails; B=1 stays backward-compatible. `BatchContract` v2 + capability `state_isolation` gate the whole path; the verifier now separates `semantic_consistency_verified` from `semantic_completeness_verified`. Native/split B=2/B=4 E2E runs with no mocks on 0.6.0 + 0.6.1-dev. See [docs/official-lerobot-plugin.md](docs/official-lerobot-plugin.md).
v1.3.8 **closes the artifact's semantics** and lays the batch-protocol foundation (still B=1): every embedded contract is schema-validated (processor contract v2, strict action-contract with `single⇒horizon=1`, closed `claims` with forbidden claims pinned `const:false`); `verify_artifact_semantics` cross-binds `config.json` ↔ `plugin_artifact_manifest.json` ↔ `lerobot-coreai.json` ↔ inventory ↔ processors (action contract/dim/horizon/robot-type/protocol equality, role→file mapping, step-empty processors); the inventory enforces unique paths+roles with a closed role enum and a root digest binding `role`+`size`; secrets are scanned across **all** declared JSON (sensitive key + any non-empty value, credential-URLs); provenance records an immutable `resolved_commit_sha` (external release refs require it); and verification reports are written **outside** the sealed artifact (`verify … --output-dir`), so verify is idempotent. Adds a batch-protocol contract to `RunnerCapabilities` (`action_batching.semantics`, `inference_state.{scope,supports_session_ids,reset_scope}`) and a pure `select_batch_execution_mode` (native needs native support; split allowed only for stateless/request-scoped or session-scoped-with-ids; global forbids split) — **no B>1 execution yet**. See [docs/official-lerobot-plugin.md](docs/official-lerobot-plugin.md).
v1.3.7 **hardens the artifact** before batching multiplies it: a typed `plugin_artifact_inventory.json` (`path`/`role`/`sha256`/`size_bytes` + `artifact_root_sha256`) with **exact** checksum coverage, path-traversal/symlink/undeclared-file guards (paths validated before any file is opened), strict JSON schemas (`additionalProperties:false`), and a verifier that separates **`integrity_verified`** (unsigned checksum consistency) from **`authenticity_verified`** (cryptographic signature — stays `false` until v1.3.9). Processor contract **v2**: step-empty processors are allowed only for exact `raw_lerobot_observation`→`postprocessed_environment_action` CoreAI ownership (wrong `expects`/`returns` fails), and the direct processor factory without artifact evidence now **fails closed**. Adds version binding (core/plugin lockstep + not-older-than-artifact), honest `source_coreai_artifact_reference` provenance (external needs an immutable revision; sha bound to the embedded manifest), structured secret scanning, bidirectional feature binding, and `verify … --report` evidence output. See [docs/official-lerobot-plugin.md](docs/official-lerobot-plugin.md).
v1.3.6 closes the **official-factory** loop: `ProtocolIdentifier` validates the protocol *family* + major (a newer major needs an explicit `backward_compatible_with` declaration; a foreign family fails); `runtime_binding_mode` (`strict`/`legacy`/`in_memory`) replaces the ambiguous booleans and **requires a bound runner** outside `in_memory`. `_IdentityProcessor` is replaced by real serializable `PolicyProcessorPipeline` instances; `lerobot-coreai package-lerobot-plugin-artifact` / `verify-lerobot-plugin-artifact` build and check a canonical, publishable artifact (`config.json`/`policy_preprocessor.json`/`policy_postprocessor.json`/`lerobot-coreai.json`/`plugin_artifact_manifest.json`/`checksums.json`/`README.md`) — tamper-evident, secret-free (no URL/token/local path). Feature cross-binding validates `cfg.input_features`/`output_features` against the manifest. The capstone E2E runs the **entire** official chain `register_third_party_plugins → PreTrainedConfig.from_pretrained → make_policy → make_pre_post_processors → post(select_action(pre(batch))) → Tensor[1, A]` against a real HTTP runner with **nothing patched**. Formal evidence reports + artifact-backed compat-level promotion remain v1.3.9; batching is v1.3.7. See [docs/official-lerobot-plugin.md](docs/official-lerobot-plugin.md).
Stable certified: LeRobot 0.6.0. Development snapshot tested: LeRobot main/0.6.1-dev @ `e40b58a` (non-blocking).

**Compatibility:**
- Core package: Python 3.10+ (metadata, inspect, predict, dry_run)
- LeRobot integration (`[lerobot]` extra): Python 3.12+ (LeRobot 0.6.0 requires it)
- The official plugin (`lerobot_policy_coreai_bridge`) is the sanctioned LeRobot
  path: `select_action(batch) → torch.Tensor(B, action_dim)`
- `predict_action(batch)` is the richer dict-returning helper on `CoreAIPolicy`

## Ecosystem

```
LeRobot trains.
Fabric exports.
Catalog indexes.
Runner executes.
lerobot-coreai adapts execution back into LeRobot rollout language.
Server makes Runner remote.
```

**Train with LeRobot. Export with Fabric. Run with CoreAI. Roll out with the same LeRobot workflow.**

## License

Apache-2.0
