# Rollout Report

Every dry-run rollout writes `rollout_report.json` to the output directory.

## Schema

- **Schema version:** `lerobot-coreai.rollout_report.v0`
- **Validated against:** `src/lerobot_coreai/schemas/rollout-report.schema.json`

## Key invariants (enforced by schema)

| Field | Value | Enforcement |
|-------|-------|-------------|
| `robot.actions_sent` | `0` | `const: 0` |
| `safety.physical_actuation_possible` | `false` | `const: false` |
| `safety.motor_commands_available` | `false` | `const: false` |

These apply to both success and failure reports.

## Success report fields

- `ok`: `true`
- `mode`: `"dry_run"`
- `policy`: `{path, repo_id, source_repo_id, type, runtime, model_id}`
- `robot`: `{type, connected: false, actions_sent: 0}`
- `runner`: `{url, reachable, supports_action, timing}`
- `manifest`: `{parity_passed, default_mode}`
- `observation`: `{source, frames, features_valid, keys}`
- `action`: `{generated, shape, contains_nan, contains_inf}`
- `safety`: `{physical_actuation_possible: false, motor_commands_available: false, ...}`
- `files`: `{observation, action, trace, report}`
- `errors`: `[]`

## Failure report fields

- `ok`: `false`
- `errors`: `[{type, message, stage, recoverable}]`

The `stage` field indicates where the failure occurred:
`policy.load`, `robot_type.validation`, `fixture.load`, `runner.predict`, etc.
