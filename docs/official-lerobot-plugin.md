# Official Out-of-Tree LeRobot Plugin (v1.3.0)

> `packages/lerobot_policy_coreai_bridge` is a **companion package** that makes
> CoreAI policies discoverable through LeRobot's **official** out-of-tree plugin
> mechanism — a real `PreTrainedPolicy` (hence `torch.nn.Module`) registered via
> `PreTrainedConfig.register_subclass`, **no monkeypatch**. Runtime-only: it does
> not train, and it proves nothing about physical safety. This is where
> `action_semantics` / `official_*` compatibility levels finally move off
> `failed` — for the plugin, not the base package's local bridge.

## Why a separate package

LeRobot imports installed distributions named `lerobot_policy_*` so they
self-register. The base `lerobot-coreai` package is intentionally torch/lerobot-
free; the plugin lives in its own distribution that depends on both.

```bash
pip install lerobot-coreai lerobot            # base + LeRobot
pip install ./packages/lerobot_policy_coreai_bridge   # the plugin
```

```python
import lerobot_policy_coreai_bridge            # registers "coreai_bridge"
from lerobot.configs.policies import PreTrainedConfig
assert "coreai_bridge" in PreTrainedConfig.get_known_choices()
```

## What it provides

- **`CoreAIBridgeConfig`** — `@PreTrainedConfig.register_subclass("coreai_bridge")`
  dataclass. Not `"coreai"` — that name is not registered upstream. `device` stays
  a real torch device (so `make_policy`'s `policy.to(cfg.device)` works);
  `runtime_device="coreai"` records that inference runs on CoreAI.
- **`CoreAIBridgePolicy(PreTrainedPolicy)`** — a genuine `nn.Module`.
  `select_action(batch) -> torch.Tensor(B, action_dim)` (per-timestep, filled
  from a chunk queue); `predict_action_chunk -> torch.Tensor`; `reset()` clears
  the queue and resets the CoreAI policy.
- **`make_coreai_bridge_pre_post_processors(config, dataset_stats=None)`** —
  official-convention processor factory (identity by default; the CoreAI runner
  owns normalization via the manifest processor contract).

## Runtime-only boundary

`forward()` and `get_optim_params()` raise; `train(True)` raises; `eval()` /
`train(False)` work (so LeRobot eval's `policy.eval()` is fine). Train with
LeRobot; run with CoreAI.

## Deprecation

The base package's `local_lerobot_registry_patch()` (v1.1.3) is now **deprecated**
in favor of this official plugin — it emits a `DeprecationWarning`.

## CI

The stable CI job (Python 3.12, LeRobot 0.6.0) installs this package and runs its
tests: config registered under `coreai_bridge`, policy is a `PreTrainedPolicy` +
`nn.Module`, `select_action` returns `Tensor(B, A)`, `train(True)` raises,
`forward`/`get_optim_params` raise.

## Factory / runtime binding (v1.3.1)

The plugin is not just registered — it is **factory-loadable**:

- `CoreAIBridgePolicy.__init__` accepts the kwargs `make_policy` passes
  (`dataset_stats`, `dataset_meta`, `**kwargs`).
- `CoreAIBridgePolicy.from_pretrained` **binds a CoreAI runtime**: it loads the
  config, resolves the runner URL from `config.runner_url_env` (fail-closed if
  unset; the URL is read from the environment, never persisted), loads a
  `lerobot_coreai.CoreAIPolicy`, and **cross-binds** the CoreAI manifest against
  the config's `expected_action_dim` / `expected_action_horizon` /
  `expected_robot_type` — failing closed on mismatch. It never returns a policy
  with no runtime bound.
- `select_action` returns a tensor on the policy's device;
  `predict_action_chunk` returns `(B, H, A)`.
- `batch_size > 1` **fails clearly** (`batch_mode="single_only"` in v1.3.1);
  batched evaluation lands in v1.3.2.

Verified against real LeRobot 0.6.0 (no monkeypatch):
`PreTrainedConfig.get_choice_class("coreai_bridge")`, `make_policy_config`,
`get_policy_class`, and a bound `from_pretrained` all resolve.

## Compatibility profiles

`lerobot-compat-check --contract` describes the **base package's local bridge**
(its `official_*` levels stay `failed` on purpose). `lerobot-compat-check
--plugin` describes the **companion plugin** — `plugin_discovery` /
`config_registry` / `policy_class_contract` pass when installed;
`policy_factory` / `processor_pipeline` are `partial`; `official_eval` stays
`not_tested` and `official_eval_certified` stays `false` until a live end-to-end
eval proves it.

## Not yet

- Full batched evaluation (`batch_size > 1`) — v1.3.2.
- Real serializable `PolicyProcessorPipeline` + canonical Hub artifact layout
  (`config.json` / `policy_preprocessor.json` / …) — v1.3.2.
- Official `lerobot-eval` end-to-end certification (envs, success/reward
  metrics) — v1.3.3.

Guarded real egress remains a separately enforced runtime.
