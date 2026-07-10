# Tutorial: Evaluate on a LeRobotDataset

Use `eval-v2` to audit how a `LeRobotDataset` maps onto a CoreAI policy's
expected observation features before you rely on it.

## Feature mapping (strict)

```bash
lerobot-coreai eval-v2 \
  --policy.path kevinqz/EVO1-SO100-CoreAI \
  --runner.url http://127.0.0.1:8710 \
  --dataset.repo_id lerobot/pusht \
  --episodes 0,1 --max-frames 100 \
  --task "push the T to the target" \
  --strict-features \
  --output-dir reports/eval-v2
```

Strict mode fails on a missing required key or a shape mismatch; non-strict
surfaces the same as warnings. Nothing is dropped silently.

## Check a single frame's observation

```bash
lerobot-coreai obs-bridge-check \
  --policy.path kevinqz/EVO1-SO100-CoreAI \
  --dataset.repo_id lerobot/pusht --frame-index 0 \
  --obs.config configs/so100_obs.json \
  --output-dir reports/obs-bridge
```

## From Python

```python
from lerobot_coreai.lerobot_eval_v2 import EvalV2Config, run_eval_v2

report = run_eval_v2(EvalV2Config(
    policy_path="kevinqz/EVO1-SO100-CoreAI",
    dataset_repo_id="lerobot/pusht", strict_features=True))
print(report["feature_mapping"]["passed"])
```

Both prove observation-mapping coherence for the evaluated sample — **not** task
success and nothing about physical safety. Dataset access needs the `[lerobot]`
extra (Python 3.12+).
