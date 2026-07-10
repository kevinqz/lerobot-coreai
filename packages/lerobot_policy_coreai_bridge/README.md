# lerobot_policy_coreai_bridge

Official **out-of-tree** LeRobot policy plugin backed by Apple CoreAI. Runtime-only.

LeRobot discovers installed distributions named `lerobot_policy_*` and imports
them so they self-register. Installing this package registers the
`coreai_bridge` policy type via LeRobot's official
`PreTrainedConfig.register_subclass` mechanism — no monkeypatch.

```python
import lerobot_policy_coreai_bridge  # registers "coreai_bridge"
from lerobot.configs.policies import PreTrainedConfig
assert "coreai_bridge" in PreTrainedConfig.get_known_choices()
```

- `CoreAIBridgeConfig` — registered `PreTrainedConfig` subclass (`coreai_bridge`).
- `CoreAIBridgePolicy` — a real `PreTrainedPolicy` (hence `torch.nn.Module`);
  `select_action(batch) -> torch.Tensor(B, action_dim)` (per-timestep, queued),
  `predict_action_chunk -> torch.Tensor`.
- `make_coreai_bridge_pre_post_processors(config, dataset_stats=None)`.

**Runtime-only:** `forward()` and `get_optim_params()` raise, `train(True)`
raises, `eval()`/`train(False)` work. Train with LeRobot, run with CoreAI. This
is **not** `policy_type="coreai"` — that name is not registered upstream. Nothing
here proves physical safety.
