# Shadow Report Schema

The `shadow_report.json` is the structured output of every shadow mode run. It records
metrics, timing, safety invariants, and claims about what the run proves.

**Schema:** [`shadow-report.schema.json`](../src/lerobot_coreai/schemas/shadow-report.schema.json)
(JSON Schema Draft 2020-12)

## Required fields

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | string (const) | `"lerobot-coreai.shadow_report.v0"` |
| `lerobot_coreai_version` | string | e.g. `"0.7.0"` |
| `ok` | boolean | Whether the run completed without fatal errors |
| `mode` | string (const) | `"shadow"` |
| `policy` | object | Policy metadata (path, repo_id, type, runtime, model_id) |
| `runner` | object | Runner URL, reachable, supports_action |
| `observation_source` | object | Source type and open/closed status |
| `loop` | object | fps_target, steps_requested, steps_completed, duration_seconds, warmup_steps |
| `metrics` | object | Counters and timing (see below) |
| `claims` | object | What the run proves (see invariants) |
| `safety` | object | No-actuation invariants (see below) |
| `files` | object | Relative paths to output files |
| `errors` | array | Per-step errors |

## Metrics

| Metric | Description |
|--------|-------------|
| `observations_read` | Number of observations read from the source |
| `actions_generated` | Number of actions produced by the policy |
| `actions_blocked` | Number of actions blocked (equals `actions_generated`) |
| `actions_sent` | Always `0` (schema-enforced) |
| `observation_errors` | Errors reading/serializing observations |
| `runner_errors` | Errors calling the runner |
| `validation_errors` | Observation/action validation errors |
| `loop_errors` | Other loop errors |
| `mean_loop_ms` | Mean per-step wall time |
| `p95_loop_ms` | 95th percentile per-step wall time |
| `mean_runner_ms` | Mean runner predict time |
| `p95_runner_ms` | 95th percentile runner predict time |

## Safety invariants (schema-enforced `const`)

These values are enforced by the schema — any report violating them fails validation:

| Path | Const | Meaning |
|------|-------|---------|
| `mode` | `"shadow"` | This is always a shadow report |
| `metrics.actions_sent` | `0` | No actions were ever sent |
| `safety.physical_actuation_possible` | `false` | No physical actuation path exists |
| `safety.motor_commands_available` | `false` | No motor command path exists |
| `safety.actuation_device_connected` | `false` | No actuation device connected |
| `safety.robot_connected` | `false` | No robot connected |
| `safety.actions_sent` | `0` | Zero actions sent (redundant with metrics) |
| `safety.action_egress` | `"blocked"` | All egress blocked by ActionBlocker |

## Claims invariants (schema-enforced `const`)

| Path | Const | Why |
|------|-------|-----|
| `claims.proves_task_success` | `false` | Shadow mode cannot prove task success |
| `claims.proves_robot_safety` | `false` | Shadow mode cannot prove physical safety |
| `claims.proves_real_world_safety` | `false` | Shadow mode cannot prove real-world safety |

`claims.proves_runtime_action_generation` is `true` when `ok=true` and at least one
action was generated. This is the **only** positive claim shadow mode can make.

## Example

```json
{
  "schema_version": "lerobot-coreai.shadow_report.v0",
  "lerobot_coreai_version": "0.7.0",
  "ok": true,
  "mode": "shadow",
  "policy": {"path": "kevinqz/EVO1-SO100-CoreAI", "type": "evo1", "runtime": "coreai"},
  "runner": {"url": "http://127.0.0.1:8710", "reachable": true, "supports_action": true},
  "observation_source": {"type": "folder", "opened": true, "closed": true},
  "loop": {"fps_target": 10.0, "steps_requested": 32, "steps_completed": 32, "duration_seconds": 3.2},
  "metrics": {"observations_read": 32, "actions_generated": 32, "actions_blocked": 32, "actions_sent": 0,
              "mean_loop_ms": 15.8, "p95_loop_ms": 20.1, "mean_runner_ms": 12.3, "p95_runner_ms": 15.2},
  "claims": {"proves_runtime_action_generation": true, "proves_task_success": false,
             "proves_robot_safety": false, "proves_real_world_safety": false},
  "safety": {"physical_actuation_possible": false, "motor_commands_available": false,
             "actuation_device_connected": false, "robot_connected": false,
             "actions_sent": 0, "action_egress": "blocked", "blocker": "ActionBlocker"},
  "files": {"actions": "actions.jsonl", "blocked_actions": "blocked_actions.jsonl",
            "observations": "observations.jsonl", "trace": "shadow_trace.jsonl",
            "report": "shadow_report.json"},
  "errors": []
}
```
