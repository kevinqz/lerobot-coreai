# Sim Report

The `sim_report.json` file is written by every sim run. It records the run
outcome, environment, metrics, claims, and safety invariants.

Schema: [`sim-report.schema.json`](../src/lerobot_coreai/schemas/sim-report.schema.json)

## Required fields

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | string (const) | `lerobot-coreai.sim_report.v0` |
| `lerobot_coreai_version` | string | The version that produced the report |
| `ok` | boolean | Whether the run completed without fatal errors |
| `mode` | string (const) | `sim` |
| `policy` | object | Policy metadata (path, type, runtime, model_id) |
| `runner` | object | Runner connection (url, reachable, supports_action) |
| `environment` | object | Environment config (type, episodes, max_steps, seed) |
| `loop` | object | Loop metadata (episodes/steps completed, duration) |
| `metrics` | object | Run metrics (see below) |
| `claims` | object | What the run proves (see below) |
| `safety` | object | Safety invariants (schema-const-locked) |
| `files` | object | Paths to output artifacts |
| `errors` | array | Per-error records |

## Safety invariants (schema-enforced `const`)

| Invariant | Const value |
|-----------|-------------|
| `safety.simulator_egress_enabled` | `true` |
| `safety.robot_egress_enabled` | `false` |
| `safety.physical_actuation_possible` | `false` |
| `safety.motor_commands_available` | `false` |
| `safety.robot_connected` | `false` |
| `safety.actions_sent_to_robot` | `0` |
| `safety.action_egress` | `"simulator_only"` |
| `claims.proves_real_task_success` | `false` |
| `claims.proves_robot_safety` | `false` |
| `claims.proves_real_world_safety` | `false` |

These hold even in failure reports — the report builder hardcodes them so they
are never a place where an unsafe claim could leak.

## Claims

`claims.proves_sim_task_success` is **conditional**, not const-locked:

```python
proves_sim_task_success = (
    episodes_completed > 0
    and success_metric_available
    and success_rate > 0
)
```

If the environment does not provide a `success` signal in its `info` dict,
`sim_success_metric_available` is `false` and `proves_sim_task_success` is
`false`. Sim task success is never conflated with real-world task success.

## Example

```json
{
  "schema_version": "lerobot-coreai.sim_report.v0",
  "lerobot_coreai_version": "0.8.0",
  "ok": true,
  "mode": "sim",
  "policy": {
    "path": "kevinqz/EVO1-SO100-CoreAI",
    "runtime": "coreai",
    "model_id": "evo1-so100-coreai"
  },
  "runner": {
    "url": "http://127.0.0.1:8710",
    "reachable": true,
    "supports_action": true
  },
  "environment": {
    "type": "fake",
    "episodes": 3,
    "max_steps_per_episode": 64,
    "seed": 42,
    "simulator_egress_enabled": true
  },
  "metrics": {
    "episodes_requested": 3,
    "episodes_completed": 3,
    "steps_completed": 192,
    "actions_generated": 192,
    "actions_sent_to_simulator": 192,
    "actions_sent_to_robot": 0,
    "mean_episode_reward": 64.0,
    "success_rate": 1.0
  },
  "claims": {
    "proves_sim_task_success": true,
    "sim_success_metric_available": true,
    "proves_real_task_success": false,
    "proves_robot_safety": false,
    "proves_real_world_safety": false
  },
  "safety": {
    "simulator_egress_enabled": true,
    "robot_egress_enabled": false,
    "physical_actuation_possible": false,
    "motor_commands_available": false,
    "robot_connected": false,
    "actions_sent_to_robot": 0,
    "action_egress": "simulator_only"
  },
  "files": {
    "actions": "actions.jsonl",
    "episodes": "episodes.jsonl",
    "observations": "observations.jsonl",
    "trace": "sim_trace.jsonl",
    "report": "sim_report.json"
  },
  "errors": []
}
```
