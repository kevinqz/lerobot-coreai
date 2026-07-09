# LeRobot Compatibility

## Core package (no LeRobot required)

- Supports Python 3.10+
- Provides: manifest loading, runner client, `select_action`, `predict_action`, `predict` CLI, fixture-based `rollout --mode dry_run`, `shadow` mode
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

- **v0.1**: inspect, doctor, catalog lookup
- **v0.2**: runner client, predict_action
- **v0.3**: select_action semantics aligned, fixture-based dry_run
- **v0.4**: LeRobotDataset replay/eval
- **v0.5**: PyTorch vs CoreAI action parity compare
- **v0.6**: export/verify/package pipeline
- **v0.7** (current): motor-blocked shadow mode
- **v0.8**: sim mode (action egress to simulator only)

## Hardware

- v0.7 has **zero** code paths for sending robot commands
- Shadow mode blocks all action egress via `ActionBlocker`
- No serial, dynamixel, feetech, motor bus, or teleop imports
- Verified by automated tests (`test_no_hardware_actuation.py`, `test_shadow_no_actuation.py`)
