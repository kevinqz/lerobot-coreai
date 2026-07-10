# Observation Pipeline Bridge (v1.1.5)

> The most common LeRobot↔CoreAI mismatch is **not** the policy — it's
> observation normalization. This turns the v1.0.4 real-mode observation config
> into a general check: take a sample observation (a `LeRobotDataset` frame or a
> robot observation) and confirm it becomes exactly the observation dict the
> CoreAI manifest expects. Nothing is dropped silently. Proves the mapping for
> the sample only — not task success, not physical safety.

## CLI

```bash
lerobot-coreai obs-bridge-check \
  --policy.path kevinqz/EVO1-SO100-CoreAI \
  --dataset.repo_id lerobot/pusht \
  --frame-index 0 \
  --obs.config configs/so100_obs.json \
  --output-dir reports/obs-bridge
```

Writes `obs_bridge_report.json` / `.md`. Reuses the same `--obs.*` flags as
`real` (`--obs.image-key`, `--obs.state-key`, `--obs.task`, `--obs.require-state`,
`--obs.require-task`, `--obs.required-keys`, `--obs.drop-unknown-keys`).

## Checks

| Check | Meaning |
|-------|---------|
| `required_keys_present` | Adaptation succeeds — every required key resolves |
| `state_shape_compatible` | The adapted state vector length matches the manifest |
| `image_keys_resolved` | Every manifest image feature is present after mapping |
| `task_present` | `task` resolves when required (from dataset or config) |
| `no_silent_drop` | Any dropped keys are listed explicitly (info) |

The report also carries `keys_present`, `keys_missing`, `dropped_keys`, and
`warnings` so image path/tensor/list conversions and any config-supplied task are
visible rather than implicit.

## Relationship to other commands

- `real --obs.*` (v1.0.4) applies this same adapter live in guarded real mode.
- `eval-v2` (v1.1.4) maps *dataset schema* → policy features; `obs-bridge-check`
  maps a *concrete frame* → the adapted observation dict.

## What this is not

Proves the observation mapping is valid for the inspected sample only. It does not
prove task success and proves nothing about physical safety.
