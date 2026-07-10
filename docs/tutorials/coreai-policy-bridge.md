# Tutorial: The CoreAI policy bridge

The bridge exposes a CoreAI-backed policy through the small surface LeRobot code
expects — `select_action`, `eval`/`train`/`to`, `reset`, a `.config` — without
pretending to be upstream-native.

## Load and run

```python
from lerobot_coreai.lerobot_bridge import load_coreai_policy_for_lerobot

policy = load_coreai_policy_for_lerobot(
    "kevinqz/EVO1-SO100-CoreAI", runner_url="http://127.0.0.1:8710")

policy.eval()                 # returns self (already inference-only)
policy.to("cuda")             # documented no-op; CoreAI runs on the CoreAI device
action = policy.select_action(batch)          # raw action
rich = policy.predict_action(batch)           # {"action": ..., "metadata": ...}
# policy.train(True)          # raises — train with LeRobot
```

`policy.policy_type == "coreai_bridge"` (deliberately not `"coreai"`).

## Optional local registry

```python
from lerobot_coreai.lerobot_registry import CoreAILeRobotRegistry, local_lerobot_registry_patch

registry = CoreAILeRobotRegistry()
registry.register("coreai_bridge")
policy = registry.load("coreai_bridge", policy_path="...", runner_url="...")

with local_lerobot_registry_patch():
    from lerobot.policies.factory import get_policy_class
    get_policy_class("coreai_bridge")   # resolves inside the block; restored after
```

## Verify + publish honest metadata

```bash
lerobot-coreai lerobot-bridge-check --policy.path kevinqz/EVO1-SO100-CoreAI
lerobot-coreai hf-metadata --policy.path kevinqz/EVO1-SO100-CoreAI --output-dir meta
```

`hf-metadata` emits honest, validated metadata: `native_registry=false`,
`upstream_native=false`, `training=false`, `physical_safety_proof=false`. Local
bridge only — not upstream-native LeRobot integration.
