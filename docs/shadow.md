# Shadow Mode

## What shadow mode does

Shadow mode runs a CoreAI-backed LeRobot policy against streamed or replayed observations,
generates the actions that *would have been* produced, validates them, logs them, and
**blocks all action egress**.

Pipeline:

```
ObservationSource → make_json_safe → CoreAIPolicy.predict_action → ActionBlocker → logs/reports
```

No action ever reaches a robot, motor, simulator, or actuator. `ActionBlocker.send()`
unconditionally raises `SafetyError`.

## What shadow mode does NOT do

- Does **not** connect to a physical robot
- Does **not** send motor commands
- Does **not** forward actions to a simulator
- Does **not** import or call any hardware APIs
- Does **not** prove task success
- Does **not** prove physical robot safety

Shadow mode is not real mode. Shadow mode is not sim mode.

## What shadow mode DOES prove

- **Runtime action generation**: the artifact runs in a loop and produces valid actions.
- **Action validation/logging**: every action is validated against the manifest and logged.
- **No-actuation logging**: all actions are blocked and audited; `actions_sent` is always 0.

## Usage

### Folder image source

```bash
lerobot-coreai shadow \
  --policy.path kevinqz/EVO1-SO100-CoreAI \
  --observation-source folder \
  --frames-dir data/shadow_frames \
  --runner.url http://127.0.0.1:8710 \
  --output-dir runs/evo1-shadow \
  --max-steps 32
```

### Fixture sequence

```bash
lerobot-coreai shadow \
  --policy.path kevinqz/EVO1-SO100-CoreAI \
  --observation-source fixtures \
  --fixtures-dir examples/shadow_sequence \
  --runner.url http://127.0.0.1:8710 \
  --fps 10 \
  --duration-seconds 10 \
  --output-dir runs/shadow-fixture-sequence
```

### Single fixture (repeated)

```bash
lerobot-coreai shadow \
  --policy.path kevinqz/EVO1-SO100-CoreAI \
  --observation-source fixture \
  --fixture examples/evo1_so100_observation.json \
  --runner.url http://127.0.0.1:8710 \
  --max-steps 16 \
  --output-dir runs/shadow-single
```

### Camera source

Camera is experimental and coming in v0.7.1. Passing `--observation-source camera` will
fail with a clear message directing you to use `folder` or `fixtures` for now.

## Output files

| File | Description |
|------|-------------|
| `shadow_report.json` | Structured report with metrics, safety invariants, claims |
| `shadow_trace.jsonl` | Event trace (one JSON per line) |
| `actions.jsonl` | Per-step action records (generated, validated, blocked) |
| `blocked_actions.jsonl` | Per-step block audit records |
| `observations.jsonl` | Per-step observation records |
| `observations/step_NNNNNN.json` | Full observation for each step |

## CLI arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--policy.path` | yes | — | HF repo id of the CoreAI artifact |
| `--observation-source` | yes | — | `fixture`, `fixtures`, `folder`, or `camera` |
| `--output-dir` | yes | — | Output directory for reports/logs |
| `--runner.url` | no | `unix:///tmp/coreai-runner.sock` | coreai-runner URL |
| `--robot.type` | no | from manifest | Override robot type |
| `--fixture` | no | — | Single fixture JSON (for `fixture` source) |
| `--fixtures-dir` | no | — | Ordered fixtures directory (for `fixtures` source) |
| `--frames-dir` | no | — | Image frames directory (for `folder` source) |
| `--image-key` | no | `observation.images.wrist` | Image observation key |
| `--state-json` | no | — | JSON array file for observation.state |
| `--state-vector` | no | — | Comma-separated floats for observation.state |
| `--task` | no | — | Task text for each observation |
| `--max-steps` | no | `32` | Maximum number of loop steps |
| `--duration-seconds` | no | — | Stop after this many seconds |
| `--fps` | no | `10.0` | Target frames per second (0 = no pacing) |
| `--warmup-steps` | no | `0` | Steps to read and discard before the loop |
| `--strict` | no | off | Strict observation key validation |
| `--fail-fast` | no | off | Stop on first step error |
| `--overwrite` | no | off | Overwrite non-empty output dir |
| `--json` | no | off | Print report as JSON |

## Safety guarantees

Enforced by `shadow-report.schema.json` (Draft 2020-12 `const` invariants):

- `safety.actions_sent` is always `0`
- `safety.action_egress` is always `"blocked"`
- `safety.physical_actuation_possible` is always `false`
- `safety.motor_commands_available` is always `false`
- `safety.actuation_device_connected` is always `false`
- `safety.robot_connected` is always `false`
- `claims.proves_task_success` is always `false`
- `claims.proves_robot_safety` is always `false`
- `claims.proves_real_world_safety` is always `false`

These hold even in failure reports.
