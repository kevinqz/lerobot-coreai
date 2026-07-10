# lerobot-coreai

**Apple CoreAI runtime backend for LeRobot policies.**

`lerobot-coreai` lets LeRobot-compatible policies run as Apple CoreAI `.aimodel` artifacts while keeping the LeRobot workflow intact: same policy concepts, same robot configs, same datasets, same observation/action features, same rollout language.

Use **LeRobot** for recording, training, datasets, robots, processors, and PyTorch policy deployment.

Use **`lerobot-coreai`** when you want to export, inspect, evaluate, dry-run, shadow-run, simulate, or roll out a LeRobot policy through Apple CoreAI.

> **Same LeRobot workflow. CoreAI runtime.**

> **Current:** `inspect`, `doctor`, `predict`, `rollout --mode dry_run`, `shadow` (motor-blocked), `eval` (LeRobotDataset replay), `compare` (PyTorch vs CoreAI parity), `export`, `sim` (simulator-only egress), the safety/governance chain (`supervisor-check`, `profile-*`, `safety-gate`, `safety-regression`, `approval-request`/`approve-bundle`/`verify-approval`, `release-readiness`), and — since v1.0.0 — `real --mode guarded` (guarded real egress) with `verify-real-session`.
> Up to v0.9.3 **no robot commands are ever sent**; v1.0.0 introduces real egress **only** through `real --mode guarded`, behind every gate. This is guarded real egress, not native LeRobot robot integration, and proves nothing about physical safety.
> `select_action()` returns raw action (LeRobot 0.6.x semantics). `predict_action()` for dict+metadata.

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

# LeRobot 0.6.0 semantics: returns raw action
action = policy.select_action(batch)

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
Baseline verified: 0.6.0. Latest verified: 0.6.1.

**Compatibility:**
- Core package: Python 3.10+ (metadata, inspect, predict, dry_run)
- LeRobot integration (`[lerobot]` extra): Python 3.12+ (LeRobot 0.6.0 requires it)
- v0.3 aligns `select_action(batch)` with LeRobot semantics — returns raw action
- `predict_action(batch)` is the richer dict-returning helper

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
