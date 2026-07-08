# Eval Report

Every eval run writes `eval_report.json` to the output directory.

## Schema

- **Schema version:** `lerobot-coreai.eval_report.v0`
- **Validated against:** `src/lerobot_coreai/schemas/eval-report.schema.json`

## Key invariants (enforced by schema)

| Field | Value | Enforcement |
|-------|-------|-------------|
| `safety.actions_sent` | `0` | `const: 0` |
| `safety.physical_actuation_possible` | `false` | `const: false` |
| `safety.motor_commands_available` | `false` | `const: false` |
| `safety.robot_connected` | `false` | `const: false` |

## Metrics

| Metric | Description |
|--------|-------------|
| `frames_requested` | Total frames selected for eval |
| `frames_processed` | Frames successfully processed |
| `actions_generated` | Actions successfully generated |
| `actions_failed` | Frames that failed |
| `shape_errors` | Action shape mismatches |
| `nan_errors` | Actions containing NaN |
| `inf_errors` | Actions containing Inf |
| `runner_errors` | Runner communication errors |
| `mean_total_ms` | Mean inference time |
| `p95_total_ms` | 95th percentile inference time |

## Output files

| File | Description |
|------|-------------|
| `eval_report.json` | Structured report with metrics and safety invariants |
| `actions.jsonl` | Per-frame action records (one JSON object per line) |
| `eval_trace.jsonl` | Event trace (rollout.started, frame.started, etc.) |
| `frames/` | Saved observation images (when dataset items contain tensors) |
