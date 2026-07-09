# lerobot-coreai

**Apple CoreAI runtime backend for LeRobot policies.**

`lerobot-coreai` lets LeRobot-compatible policies run as Apple CoreAI `.aimodel` artifacts while keeping the LeRobot workflow intact: same policy concepts, same robot configs, same datasets, same observation/action features, same rollout language.

Use **LeRobot** for recording, training, datasets, robots, processors, and PyTorch policy deployment.

Use **`lerobot-coreai`** when you want to export, inspect, evaluate, dry-run, shadow-run, simulate, or roll out a LeRobot policy through Apple CoreAI.

> **Same LeRobot workflow. CoreAI runtime.**

> **Current:** `eval` (LeRobotDataset replay), `compare` (PyTorch vs CoreAI parity), `export`, and `shadow` mode (motor-blocked).
> `select_action()` returns raw action (LeRobot 0.6.0 semantics). `predict_action()` for dict+metadata.

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

## Safety model

v0.7 adds motor-blocked shadow mode.
v0.7.1 adds optional local camera observation source for shadow mode.
v0.7.2 adds observation adapters, live metrics, and run quality diagnostics.
v0.8 adds simulator-only sim mode: actions drive a simulator, never a robot.
v0.8.1 adds a gymnasium simulator adapter (`[sim]` extra) for sim mode.
v0.8.2 adds sim analytics: CSV exports, markdown summaries, failure taxonomy, and richer report sections for simulator-only runs.
v0.8.3 adds sim quality gates and a sim-regression command to compare two sim runs for regression.
v0.8.4 adds reproducibility bundles for simulator-only runs, including manifests, checksums, environment metadata, runner metadata, and audit-ready package outputs.
Shadow mode can read observations and generate actions.
Shadow mode cannot send actions to a robot, motor, simulator, or actuator.
Sim mode can send actions to a simulator.
Sim mode cannot send actions to a robot.
Neither mode ever connects to a robot or sends motor commands.
Export verification can prove numeric action fidelity only when compare passes.
It cannot prove task success or physical robot safety.

| Mode | Status | Behavior |
|------|--------|----------|
| `dry_run` | v0.3 ✅ | No physical robot. Fixture-based action generation. |
| `shadow` | v0.7 ✅ | Observations streamed/replayed, actions generated and logged, never sent. |
| `sim` | v0.8 ✅ | Actions drive a simulator; never a robot. Requires `--confirm-sim-egress`. |
| `real` | v1.0 planned | Physical robot actuation. Requires explicit confirmation. |

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
