# LeRobot Compatibility

`lerobot-coreai` is compatible with the shape and main inference/eval workflows of
LeRobot 0.6.x, but is **not yet** a complete native integration inside the upstream
LeRobot registry/factory. Train with LeRobot; run with CoreAI.

## Compatibility levels

| Layer | Status | Notes |
|-------|--------|-------|
| Core package without LeRobot | ✅ | Python >=3.10, no torch/LeRobot imports |
| LeRobot 0.6.x dependency range | ✅ | `[lerobot]` extra installs `lerobot>=0.6.0,<0.7.0` on Python 3.12+ |
| `select_action(batch)` raw action | ✅ | Matches LeRobot 0.6 semantics |
| `LeRobotDataset` eval | ✅ | Uses public dataset constructor (since v0.4) |
| PyTorch/CoreAI compare | ⚠️ | Experimental source policy loader (since v0.5) |
| Native LeRobot policy registry | ❌ | Not yet registered as upstream policy type |
| Training | ❌ | Train with LeRobot; run with CoreAI |
| Robot hardware | ❌ until v1.0 | No real robot egress yet |

## Implemented

- LeRobot-style `select_action(batch)` raw action semantics
- `LeRobotDataset`-based eval/replay
- PyTorch vs CoreAI action parity compare, experimental policy loader
- LeRobot 0.6.x dependency range under `[lerobot]` extra

## Not yet native

- No upstream LeRobot registry/factory `policy_type="coreai"`
- No training integration
- No teleop/robot control integration
- No native LeRobot CLI replacement

## Core package (no LeRobot required)

- Supports Python 3.10+
- Provides: manifest loading, runner client, `select_action`, `predict_action`, `predict` CLI, fixture-based `rollout --mode dry_run`, `shadow` mode
- Does NOT import LeRobot
- Does NOT require torch, numpy, or datasets

## LeRobot-native integration (`[lerobot]` extra)

- Requires Python 3.12+ (LeRobot 0.6.0 requirement)
- Installs `lerobot>=0.6.0,<0.7.0` (which pins torch, torchvision, numpy)
- Enables `eval` (LeRobotDataset replay) and `compare` (PyTorch source policy loader)

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
- **v0.7**: motor-blocked shadow mode
- **v0.7.1**: optional camera observation source (`[camera]` extra)
- **v0.7.2**: observation adapters, live metrics, run quality diagnostics
- **v0.8**: sim mode (action egress to simulator only)
- **v0.8.1**: gymnasium simulator adapter (`[sim]` extra)
- **v0.8.2** (current): sim analytics — CSV exports, markdown summaries, failure taxonomy

## Hardware

- v0.8 has **zero** code paths for sending robot commands
- Shadow mode blocks all action egress via `ActionBlocker`
- Sim mode sends actions to a simulator via `SimEgress`; robot egress always raises
- No serial, dynamixel, feetech, motor bus, or teleop imports
- Verified by automated tests (`test_no_hardware_actuation.py`, `test_shadow_no_actuation.py`, `test_sim_no_robot_actuation.py`)
