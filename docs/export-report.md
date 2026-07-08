# Export Report

Every export run writes `export_report.json` to the output directory.

## Schema

- **Schema version:** `lerobot-coreai.export_report.v0`
- **Validated against:** `src/lerobot_coreai/schemas/export-report.schema.json`

## Key invariants (enforced by schema)

| Field | Value |
|-------|-------|
| `safety.actions_sent` | `0` |
| `safety.physical_actuation_possible` | `false` |
| `safety.motor_commands_available` | `false` |
| `safety.robot_connected` | `false` |
| `claims.proves_task_success` | `false` |
| `claims.proves_robot_safety` | `false` |

## Fields

- `source`: PyTorch policy path, type, robot type
- `artifact`: format, path, model_id, manifest
- `fabric`: used/skipped, status, profile, target
- `verification`: manifest_valid, runner_checked, dry_run/eval/compare results
- `claims`: publish_ready, proves_numeric_action_fidelity
- `files`: paths to manifest, trace, report, publish_dir

## publish_ready rules

`publish_ready` is `true` only when:
- `manifest_valid == true`
- Artifact exists
- If compare requested: `compare.ok == true`
- If eval requested: `eval.ok == true`
- If dry_run requested: `dry_run.ok == true`
