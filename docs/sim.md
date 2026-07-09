# Sim Mode

## What sim mode does

Sim mode runs a CoreAI-backed LeRobot policy against a **SimEnvironment**,
generates actions, and **egresses them to the simulator only**. The simulator
advances its state and returns the next observation, reward, and done signal.

Pipeline:

```
env.reset() → make_json_safe → CoreAIPolicy.predict_action → SimEgress → env.step() → logs/reports
```

Actions reach the simulator's `step()`. No action ever reaches a robot, motor,
serial device, or actuator. `SimEgress.send_to_robot()` unconditionally raises
`SafetyError`.

## What sim mode does NOT do

- Does **not** connect to a physical robot
- Does **not** send motor commands
- Does **not** import or call any hardware APIs
- Does **not** prove real-world task success
- Does **not** prove physical robot safety
- Does **not** prove real-world safety

Sim mode can prove simulator task success (when the environment reports a
success signal). Sim task success is not real-world task success.

## What sim mode DOES prove

- **Runtime policy execution in a simulator**: the artifact drives a simulated
  environment and receives feedback.
- **Simulator action egress**: actions are sent to `env.step()` and counted.
- **Simulator episode metrics**: reward, success rate, and episode length are
  recorded when the environment provides them.

## Usage

Sim mode requires `--confirm-sim-egress`. Without it, the run refuses to start.

### Fake environment (smoke test)

```bash
lerobot-coreai sim \
  --policy.path kevinqz/EVO1-SO100-CoreAI \
  --env.type fake \
  --runner.url http://127.0.0.1:8710 \
  --episodes 3 \
  --max-steps-per-episode 64 \
  --confirm-sim-egress \
  --output-dir runs/fake-sim
```

The fake environment is a deterministic stub for testing the loop end-to-end.
It reports `done` after `max-steps-per-episode` and a constant reward.

### Replay environment (deterministic sequence)

Create a replay config JSON pointing to a directory of ordered observation
files (`NNNNNN.json`):

```json
{
  "observations_dir": "examples/shadow_sequence",
  "reward_per_step": 0.0,
  "success_on_last_step": true
}
```

```bash
lerobot-coreai sim \
  --policy.path kevinqz/EVO1-SO100-CoreAI \
  --env.type replay \
  --env.config replay_config.json \
  --runner.url http://127.0.0.1:8710 \
  --episodes 1 \
  --max-steps-per-episode 300 \
  --confirm-sim-egress \
  --output-dir runs/replay-sim
```

### Gym / Gymnasium environment (v0.8.1)

The gym adapter wraps any [gymnasium](https://gymnasium.farama.org/) environment
via `gymnasium.make(env_id, **kwargs)`. It requires the optional `[sim]` extra:

```bash
pip install "lerobot-coreai[sim]"
```

gymnasium is imported lazily — only when `--env.type gym` is used — so the core
package stays free of a hard gymnasium dependency.

```bash
lerobot-coreai sim \
  --policy.path kevinqz/EVO1-SO100-CoreAI \
  --env.type gym \
  --env.id PushT-v0 \
  --runner.url http://127.0.0.1:8710 \
  --episodes 10 \
  --max-steps-per-episode 300 \
  --confirm-sim-egress \
  --output-dir runs/pusht-gym-sim
```

Optional `--env.kwargs-json` passes kwargs to `gymnasium.make()`:

```bash
  --env.kwargs-json '{"max_episode_steps": 200}'
```

**API translation.** gymnasium's `reset()` returns `(obs, info)` and `step()`
returns the 5-tuple `(obs, reward, terminated, truncated, info)`. The adapter
collapses `terminated or truncated` into the single `done` flag the sim loop
expects; the original flags are preserved in `info["terminated"]` /
`info["truncated"]`.

**Observation normalization.** Gym observations are normalized to the dict
shape the policy expects:
- `dict` → passthrough
- scalar / list / tuple / numpy-array-like → `{"observation.state": [...]}`
- anything else → a clear error (provide an env adapter or observation mapping)

### Reserved environment types

`lerobot` and `pusht` are reserved for a later release. They raise a clear
"not yet supported" error.

## Output files

| File | Description |
|------|-------------|
| `sim_report.json` | Structured report with metrics, safety invariants, claims, analytics |
| `sim_summary.md` | Human-readable markdown summary (written by default, v0.8.2) |
| `failure_taxonomy.json` | Errors classified by stage/type/category (written by default, v0.8.2) |
| `sim_trace.jsonl` | Event trace (one JSON per line) |
| `actions.jsonl` | Per-step action records (generated, egressed to simulator, with timing) |
| `episodes.jsonl` | Per-episode summaries (reward, success, action counts) |
| `observations.jsonl` | Per-step observation records |
| `observations/epNNN_stepNNNNNN.json` | Full observation for each step |
| `episode_metrics.csv` | Per-episode CSV (with `--export-csv`, v0.8.2) |
| `step_metrics.csv` | Per-step CSV (with `--export-csv`, v0.8.2) |

See [Sim Analytics](sim-analytics.md) for the analytics sections added in v0.8.2.

## CLI arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--policy.path` | yes | — | HF repo id of the CoreAI artifact |
| `--env.type` | yes | — | `fake`, `replay`, or `gym` |
| `--output-dir` | yes | — | Output directory for reports/logs |
| `--confirm-sim-egress` | yes | — | Confirm actions may be sent to the simulator |
| `--runner.url` | no | `unix:///tmp/coreai-runner.sock` | coreai-runner URL |
| `--robot.type` | no | from manifest | Override robot type |
| `--env.config` | no | — | Environment config JSON (for `replay`) |
| `--env.id` | no | — | Gymnasium env id (for `gym`, e.g. `PushT-v0`) |
| `--env.kwargs-json` | no | — | Gymnasium `make()` kwargs as JSON (for `gym`) |
| `--task` | no | — | Task text for each observation |
| `--state-vector` | no | — | Comma-separated floats for observation.state |
| `--episodes` | no | `1` | Number of episodes to run |
| `--max-steps-per-episode` | no | `300` | Maximum steps per episode |
| `--seed` | no | — | Base RNG seed (episode N uses seed+N) |
| `--fps` | no | `0.0` | Target steps per second (0 = no pacing) |
| `--strict` | no | off | Strict observation key validation |
| `--fail-fast` | no | off | Stop on first error |
| `--overwrite` | no | off | Overwrite non-empty output dir |
| `--live` | no | off | Print live metrics per step |
| `--live-every` | no | `1` | Print live metrics every N steps |
| `--json` | no | off | Print report as JSON |
| `--export-csv` | no | off | Write `episode_metrics.csv` and `step_metrics.csv` (v0.8.2) |
| `--no-summary-md` | no | off | Do not write `sim_summary.md` (v0.8.2; written by default) |
| `--no-failure-taxonomy` | no | off | Do not write `failure_taxonomy.json` (v0.8.2; written by default) |
| `--package-run` | no | off | Package the run into a reproducibility bundle after it completes (v0.8.4) |
| `--package-output-dir` | no | `<output-dir>/bundle` | Bundle output directory (v0.8.4) |
| `--package-overwrite` | no | off | Overwrite a non-empty bundle output dir (v0.8.4) |
| `--redact-runner-url` | no | off | Redact the runner URL in the bundle (v0.8.4) |
| `--no-redact-local-paths` | no | off | Keep absolute local paths in the bundle (v0.8.4; redacted by default) |
| `--include-observations-dir` | no | off | Include the full `observations/` dir in the bundle (v0.8.4) |

## Runtime safety supervisor (v0.9.0)

Every action passes a fail-closed software [safety supervisor](safety-supervisor.md)
before egress. Sim defaults to `--supervisor.mode enforce`: an action that is
non-finite, out of the profile's shape/bounds/delta/norm, or from the wrong
robot type is **blocked** and never reaches the simulator; the episode is
terminated as `safety_terminated`. Use `--safety.profile-name` /
`--safety.profile` to choose a [profile](safety-profiles.md), `report_only` to
observe without blocking, or `off` to disable. The run writes
`safety_report.jsonl` + `safety_summary.json/md` and a `safety_supervisor`
section in `sim_report.json`. The supervisor is a software layer only — it does
not prove physical robot safety.

## Reproducibility bundles (v0.8.4)

A completed sim run can be packaged into an auditable, self-contained bundle
(manifest + checksums + metadata + reproduction notes) with `--package-run`, or
after the fact with `lerobot-coreai package-sim-run`. See
[sim-bundles.md](sim-bundles.md). Packaging never alters sim results and refuses
to bundle a report that violates the no-robot-egress invariants below.

## Safety guarantees

Enforced by `sim-report.schema.json` (Draft 2020-12 `const` invariants):

- `safety.simulator_egress_enabled` is always `true`
- `safety.robot_egress_enabled` is always `false`
- `safety.physical_actuation_possible` is always `false`
- `safety.motor_commands_available` is always `false`
- `safety.robot_connected` is always `false`
- `safety.actions_sent_to_robot` is always `0`
- `safety.action_egress` is always `"simulator_only"`
- `claims.proves_real_task_success` is always `false`
- `claims.proves_robot_safety` is always `false`
- `claims.proves_real_world_safety` is always `false`

These hold even in failure reports.
