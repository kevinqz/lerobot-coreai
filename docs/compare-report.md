# Compare Report

Every compare run writes `compare_report.json` to the output directory.

## Schema

- **Schema version:** `lerobot-coreai.compare_report.v0`
- **Validated against:** `src/lerobot_coreai/schemas/compare-report.schema.json`

## Key invariants (enforced by schema)

| Field | Value | Enforcement |
|-------|-------|-------------|
| `safety.actions_sent` | `0` | `const: 0` |
| `safety.physical_actuation_possible` | `false` | `const: false` |
| `safety.motor_commands_available` | `false` | `const: false` |
| `safety.robot_connected` | `false` | `const: false` |
| `claims.proves_task_success` | `false` | `const: false` |
| `claims.proves_robot_safety` | `false` | `const: false` |

## Claims

- `proves_numeric_action_fidelity`: `true` only when all frames pass tolerance and `frames_compared >= 1`
- `proves_task_success`: always `false` (numeric parity ≠ task success)
- `proves_robot_safety`: always `false`

## Output files

| File | Description |
|------|-------------|
| `compare_report.json` | Aggregate report with metrics, claims, and safety invariants |
| `compare_actions.jsonl` | Per-frame comparison records (cosine, MAE, pass/fail) |
| `compare_trace.jsonl` | Event trace (compare.started, frame.compared, etc.) |
| `manifest-evaluation-patch.json` | Manifest update suggestion (only when parity passes) |
| `actions/` | Per-frame action files (with `--save-actions`) |
