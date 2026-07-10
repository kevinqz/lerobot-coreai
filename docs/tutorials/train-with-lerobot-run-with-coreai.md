# Tutorial: Train with LeRobot, run with CoreAI

`lerobot-coreai` is a **runtime** backend. You train policies with LeRobot as
usual; you run them through Apple CoreAI. There is no training path here — the
bridge is inference-only and refuses `train(True)`.

## The split

| Stage | Tool |
|-------|------|
| Record, train, datasets, robots | **LeRobot** |
| Export policy → CoreAI `.aimodel` | CoreAI Fabric |
| Inspect / eval / dry-run / shadow / sim | `lerobot-coreai` |
| Run inference through CoreAI | `lerobot-coreai` (+ coreai-runner) |
| Guarded real egress | `lerobot-coreai real --mode guarded` |

## Minimal run

```python
from lerobot_coreai.lerobot_bridge import load_coreai_policy_for_lerobot

policy = load_coreai_policy_for_lerobot(
    "kevinqz/EVO1-SO100-CoreAI", runner_url="http://127.0.0.1:8710")
action = policy.select_action({"observation.state": [0.0] * 7, "task": "..."})
```

## Honest boundary

- Not upstream-native LeRobot (`policy_type="coreai"` is not registered upstream).
- No training. `policy.train(True)` raises — go back to LeRobot for that.
- Nothing here proves physical safety.

See also: [coreai-policy-bridge.md](coreai-policy-bridge.md),
[lerobot-dataset-eval.md](lerobot-dataset-eval.md).
