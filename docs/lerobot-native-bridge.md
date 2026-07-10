# LeRobot Native Bridge (v1.1.0)

> **Local bridge, not upstream-native integration.** This exposes a CoreAI-backed
> policy through the small surface LeRobot code expects from a policy object. It
> is **not** registered in LeRobot's upstream registry/factory as
> `policy_type="coreai"`, it monkeypatches nothing globally, and it does not
> import `torch` or `lerobot` at module load. Train with LeRobot, run with CoreAI.

## Why

`CoreAIPolicy` already matches LeRobot 0.6.x `select_action(batch)` semantics. The
bridge makes that usable anywhere LeRobot code expects a *policy-shaped object*
(`select_action`, `eval`/`train`/`to`, `reset`, a `.config`) — without pretending
to be part of upstream LeRobot.

## Minimal use

```python
from lerobot_coreai.lerobot_bridge import load_coreai_policy_for_lerobot

policy = load_coreai_policy_for_lerobot(
    "kevinqz/EVO1-SO100-CoreAI",
    runner_url="http://127.0.0.1:8710",
)

action = policy.select_action(batch)          # raw action (LeRobot 0.6.x semantics)
rich = policy.predict_action(batch)           # {"action": ..., "metadata": ...}
```

## Runtime-only semantics

| Method | Behavior |
|--------|----------|
| `select_action(batch)` | Forwards to CoreAI; returns the **raw action**. |
| `predict_action(batch)` | Returns the `{"action", "metadata"}` dict. |
| `eval()` | Returns `self` (already inference-only). |
| `train(False)` | Returns `self`. |
| `train(True)` | **Raises** `CoreAIPolicyError` — train with LeRobot. |
| `to(device)` | Documented **no-op** returning `self`. LeRobot loops may call `.to("cuda")`; the bridge accepts and ignores the device — inference always runs on the CoreAI runtime. |
| `reset()` | Forwards to the CoreAI policy. |

`policy_type` is `"coreai_bridge"` — deliberately **not** `"coreai"`, because no
upstream registry entry exists for either. `policy.metadata()` always reports
`training_supported=false` and `native_registry=false`.

## CLI: `lerobot-bridge-check`

Probe the bridge for a policy (sends no robot action):

```bash
lerobot-coreai lerobot-bridge-check \
  --policy.path kevinqz/EVO1-SO100-CoreAI \
  --runner.url http://127.0.0.1:8710 \
  --dataset.repo_id lerobot/pusht \
  --max-frames 5 \
  --output-dir reports/bridge-check
```

Writes `lerobot_bridge_report.json` / `.md`. Checks include: `coreai_policy_loads`,
`runner_reachable` (if `--runner.url` given), `select_action_callable`,
`predict_action_metadata_available`, `no_training_claim`, `no_native_registry_claim`,
`eval_to_safe_noops`, an optional `lerobot_version_in_range` (only when the
`[lerobot]` extra is installed), and an optional `dataset_item_to_batch` smoke
(only when both `--dataset.repo_id` and LeRobot are present).

## Optional LeRobot dependency

The bridge works **without** LeRobot installed — it never imports `lerobot` or
`torch` at load. When the `[lerobot]` extra is installed (Python 3.12+,
`lerobot>=0.6.0,<0.7.0`), `lerobot-bridge-check` additionally verifies the
version is in range and that the `PreTrainedPolicy` import path exists. The bridge
does **not** subclass `PreTrainedPolicy` — it is duck-typed on purpose, to avoid
inheriting torch/config/device expectations that don't apply to a CoreAI runtime.

## What this is not

- Not upstream-native LeRobot integration (`policy_type="coreai"` is not registered).
- Not a training policy.
- Not a robot-control or teleop path.
- Proves nothing about physical safety.
