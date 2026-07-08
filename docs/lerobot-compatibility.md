# LeRobot Compatibility

## Core package (no LeRobot required)

- Supports Python 3.10+
- Provides: manifest loading, runner client, `select_action`, `predict_action`, `predict` CLI, fixture-based `rollout --mode dry_run`
- Does NOT import LeRobot
- Does NOT require torch, numpy, or datasets

## LeRobot-native integration (`[lerobot]` extra)

- Requires Python 3.12+ (LeRobot 0.6.0 requirement)
- Installs `lerobot>=0.6.0,<0.7.0` (which pins torch, torchvision, numpy)
- Future: LeRobotDataset-based eval, LeRobotCoreAIPolicy wrapper

## API alignment with LeRobot 0.6.0

- `select_action(batch)` returns the **raw action** (matching LeRobot's `PreTrainedPolicy.select_action`)
- `predict_action(batch)` is the richer helper returning `{"action": ..., "metadata": ...}`
- No monkey-patching or factory integration with LeRobot
- No use of LeRobot private/internal submodule paths

## Roadmap

- **v0.3** (current): `select_action` semantics aligned, fixture-based dry_run
- **v0.4**: LeRobotDataset replay/eval
- **v0.5**: Optional `LeRobotCoreAIPolicy` wrapper

## Hardware

- v0.3 has **zero** code paths for sending robot commands
- No serial, dynamixel, feetech, motor bus, or teleop imports
- Verified by automated test (`test_no_hardware_actuation.py`)
