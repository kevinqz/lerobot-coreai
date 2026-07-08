# lerobot-coreai

**Apple CoreAI runtime backend for LeRobot policies.**

`lerobot-coreai` lets LeRobot-compatible policies run as Apple CoreAI `.aimodel` artifacts while keeping the LeRobot workflow intact: same policy concepts, same robot configs, same datasets, same observation/action features, same rollout language.

Use **LeRobot** for recording, training, datasets, robots, processors, and PyTorch policy deployment.

Use **`lerobot-coreai`** when you want to export, inspect, evaluate, dry-run, shadow-run, simulate, or roll out a LeRobot policy through Apple CoreAI.

> **Same LeRobot workflow. CoreAI runtime.**

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

policy = CoreAIPolicy.from_pretrained("kevinqz/EVO1-SO100-CoreAI")

# v0.1: metadata-only — inspect the manifest without a runner
print(policy.policy_type)       # "evo1"
print(policy.robot_type)        # "so100"
print(policy.config.observation_features)
print(policy.config.action_features)
print(policy.parity_passed)     # True if action parity verified
```

> **Note:** `select_action(batch)` is planned for v0.2, after coreai-runner action inference
> is wired. In v0.1 it raises `NotImplementedError`. The metadata API above works now.

### Doctor — check compatibility

```bash
lerobot-coreai doctor --policy.path kevinqz/EVO1-SO100-CoreAI --robot.type so100
```

## CLI commands

| Command | Status | Purpose |
|---------|--------|---------|
| `inspect` | v0.1 ✅ | Inspect a CoreAI-backed LeRobot policy |
| `doctor` | v0.1 ✅ | Metadata compatibility checks |
| `rollout` | v0.2 planned | CoreAI runner rollout |
| `serve` | v0.2 planned | Runner lifecycle |
| `eval` | v0.3 planned | LeRobotDataset replay |
| `compare` | v0.3 planned | PyTorch vs CoreAI |
| `export` | v0.4 planned | Fabric wrapper |

## Safety model

Rollout modes enforce safety through the workflow, not scary new concepts:

| Mode | Behavior |
|------|----------|
| `dry_run` | No physical robot required. No motor commands. |
| `shadow` | Robot/cameras may be live. Actions generated and logged. No motor commands. |
| `sim` | Simulation receives actions. No physical robot. |
| `real` | Physical robot receives actions. **Requires explicit confirmation.** |

```bash
# Real mode requires the confirmation flag
lerobot-coreai rollout \
  --policy.path kevinqz/ACT-SO101-CoreAI \
  --robot.type so101 \
  --mode real \
  --confirm-real-robot-actuation
```

No flag, no actuation.

## Version policy

`lerobot-coreai` 0.1.x supports LeRobot 0.6.x public APIs. If the LeRobot version is unsupported, `lerobot-coreai` warns clearly, allows metadata-only commands, and blocks rollout/eval unless `--allow-unsupported-lerobot` is passed.

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
