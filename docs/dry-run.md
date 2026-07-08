# Dry-Run Rollout

## What dry_run does

1. Loads the policy manifest from Hugging Face
2. Validates runner health and action support
3. Loads an observation fixture (JSON)
4. Calls `coreai-runner` with `runtime_kind=action`
5. Validates the action output (shape, NaN, Inf)
6. Writes output files:
   - `observation.json` — the observation batch used
   - `action.json` — the action + metadata
   - `trace.jsonl` — event trace
   - `rollout_report.json` — structured report

## What dry_run does NOT do

- Does **not** connect to a physical robot
- Does **not** send motor commands
- Does **not** import or call any hardware APIs

## Usage

```bash
lerobot-coreai rollout \
  --policy.path kevinqz/EVO1-SO100-CoreAI \
  --robot.type so100 \
  --mode dry_run \
  --fixture examples/evo1_so100_observation.json \
  --runner.url http://127.0.0.1:8710 \
  --output-dir runs/evo1-dry-run
```

## Safety guarantees

- `robot.actions_sent` is always `0` in the report
- `safety.physical_actuation_possible` is always `false`
- `safety.motor_commands_available` is always `false`
- These are enforced by the `rollout-report.schema.json` schema (`const: 0`, `const: false`)
