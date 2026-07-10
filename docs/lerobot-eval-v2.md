# LeRobot Eval v2 — Feature Mapping (v1.1.4)

> Builds on the v0.4 `LeRobotDataset` eval by making the dataset ↔ policy
> **feature mapping** explicit and auditable. A `--strict-features` mode fails on
> missing required keys or shape mismatches. The report proves the observation
> mapping is coherent for the evaluated frames — **not** task success, **not**
> physical safety.

## CLI

```bash
lerobot-coreai eval-v2 \
  --policy.path kevinqz/EVO1-SO100-CoreAI \
  --runner.url http://127.0.0.1:8710 \
  --dataset.repo_id lerobot/pusht \
  --episodes 0,1 \
  --max-frames 100 \
  --task "push the T to the target" \
  --strict-features \
  --output-dir reports/eval-v2
```

Writes `lerobot_feature_mapping.json`, `lerobot_eval_v2_report.json`, and
`lerobot_eval_v2_report.md`.

| Flag | Meaning |
|------|---------|
| `--episodes 0,1` | Restrict to these episode indices |
| `--max-frames N` | Cap the number of frames considered |
| `--task TEXT` | Supply `task` from config when the dataset lacks it |
| `--strict-features` | Missing required key / shape mismatch → **fail** |
| `--fail-on-unknown` | In strict mode, an unexpected dataset feature also fails |

## Feature mapping

For each observation feature the policy manifest expects, the mapping records
whether the dataset provides it, whether it is provided by config (`task`), and
whether the shapes are compatible:

```json
{
  "schema_version": "lerobot-coreai.lerobot_feature_mapping.v0",
  "strict": true,
  "features": {
    "observation.state": {"policy_expected": true, "required": true,
                          "dataset_present": true, "shape_compatible": true},
    "task": {"policy_expected": true, "required": true,
             "dataset_present": false, "provided_by_config": true}
  },
  "unknown_dataset_features": [],
  "problems": [],
  "warnings": [],
  "passed": true
}
```

### Strict vs non-strict

- **strict**: any missing required key or shape mismatch goes to `problems` and
  `passed=false`.
- **non-strict**: the same issues are surfaced as `warnings` and never block —
  useful for exploration. Nothing is dropped silently; every issue is reported.

## Relationship to existing commands

`eval` (v0.4) and `compare` (v0.5) are unchanged. `eval-v2` adds the explicit
feature-mapping layer on top of the same public `LeRobotDataset` constructor.

## What this is not

Proves the observation mapping is coherent for the evaluated sample only. It does
not prove task success and proves nothing about physical safety.
