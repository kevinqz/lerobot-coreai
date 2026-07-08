# lerobot-coreai

**Apple CoreAI runtime backend for LeRobot policies.**

`lerobot-coreai` lets LeRobot-compatible policies run as Apple CoreAI `.aimodel` artifacts while keeping the LeRobot workflow intact: same policy concepts, same robot configs, same datasets, same observation/action features, same rollout language.

Use **LeRobot** for recording, training, datasets, robots, processors, and PyTorch policy deployment.

Use **`lerobot-coreai`** when you want to export, inspect, evaluate, dry-run, shadow-run, simulate, or roll out a LeRobot policy through Apple CoreAI.

> **Same LeRobot workflow. CoreAI runtime.**

> **v0.2:** `select_action()` and `predict` command work with a running coreai-runner.
> `inspect`, `doctor`, `list`, and metadata API also work without a runner.

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

# v0.2: metadata + action inference (requires running coreai-runner)
policy = CoreAIPolicy.from_pretrained(
    "kevinqz/EVO1-SO100-CoreAI",
    runner_url="http://127.0.0.1:8710",
)

batch = {
    "observation.images.wrist": "/tmp/wrist.png",
    "observation.state": [0.0, 0.1, 0.2, 0.0, 0.0, 0.0, 0.0],
    "task": "pick up the cube",
}

result = policy.select_action(batch)
action = result["action"]
```

> **Without a runner:** `from_pretrained(repo_id)` without `runner_url` loads metadata only.
> `select_action()` will raise `RunnerNotReachableError`. Metadata access (policy_type,
> robot_type, config, parity_passed) always works.

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
| `rollout` | v0.3 planned | Dry-run rollout with fixture |
| `eval` | v0.3 planned | LeRobotDataset replay |
| `compare` | v0.3 planned | PyTorch vs CoreAI |
| `export` | v0.4 planned | Fabric wrapper |

## Safety model

v0.2 does not implement physical robot actuation. `select_action()` generates
actions but never sends motor commands.

Rollout modes are documented as future safety modes:

| Mode | Status | Behavior |
|------|--------|----------|
| `dry_run` | v0.3 planned | No physical robot. Fixture-based action generation. |
| `shadow` | v0.6 planned | Robot/cameras live, actions logged, no motor commands. |
| `sim` | v0.6 planned | Simulation receives actions. |
| `real` | v1.0 planned | Physical robot actuation. Requires explicit confirmation. |

> v0.2 does not implement physical robot actuation.
> `real`, `shadow`, and `sim` are future safety modes and are not executable yet.
> No robot commands are sent by v0.2.

## Version policy

`lerobot-coreai` 0.2.x supports LeRobot 0.6.x public APIs. If the LeRobot version is unsupported, `lerobot-coreai` warns clearly, allows metadata-only commands, and blocks rollout/eval unless `--allow-unsupported-lerobot` is passed.

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
