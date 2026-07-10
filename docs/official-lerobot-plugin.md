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

## Data plane (v1.3.2)

The plugin now bridges a real LeRobot batch to the CoreAI observation contract:
`transport.prepare_single_coreai_observation` strips the leading batch dim,
unwraps `task: list[str]` → `task: str`, keeps **only** manifest-declared
observation features (never the ground-truth `action`/`reward`), converts
`torch`/`numpy` to JSON-safe values, and hashes the exact payload. `select_action`
now **always** routes through `predict_action_chunk` (one boundary → `(B,H,A)` →
per-timestep `(B,A)`). `batch_size > 1` fails closed (batched transport is
v1.3.3). Cross-binding is fully fail-closed: a declared expectation with an
unknown manifest value is a failure.

Real discovery is proven in a clean subprocess (`test_discovery.py`):
`register_third_party_plugins()` finds `coreai_bridge` without a manual import.
The `lerobot-dev` CI job is pinned to an exact 0.6.1-dev commit.

## v1.3.3 — local artifacts, negotiated codec, validated outputs

- **Local manifests** — `load_manifest`/`resolve_manifest` accept an HF repo id,
  a local directory, or a local `lerobot-coreai.json` file. Local sources never
  touch the network, and a local-looking path that doesn't exist **fails**
  instead of falling back to the Hub (records `source_kind` + `sha256`).
- **Observation codec** — the transport emits `nested_json_v1` (plain JSON lists)
  by default; `typed_array_envelope_v1` (`{"__array__",dtype,shape}`) is only used
  when explicitly selected. Shape is validated against the manifest **before**
  encoding, so the envelope can never mask a mismatch; the shape audit + payload
  `sha256` are recorded.
- **Strict action outputs** — `normalize_and_validate_action_chunk` rejects
  ragged / rank-4 / non-finite / wrong-dim/horizon Runner outputs;
  `predict_action_chunk` and `select_action` both route through it.
- **Fully fail-closed cross-binding** — `expected_robot_type` (like action_dim)
  now fails when the manifest declares no robot type.

## v1.3.4 — real codec negotiation + action-contract integration

- **Negotiated observation encoding** — `RunnerCapabilities` now carries
  `protocol_version` / `observation_encodings` / `supports_batch` /
  `max_batch_size`, and `negotiate_observation_encoding()` selects the encoding
  as the intersection of the config request, the plugin's supported encodings,
  and what the runner **announces**. `auto` picks the first common encoding; an
  unsupported request or empty intersection **fails closed** (legacy fallback is
  opt-in). The selected encoding + protocol version + observation `sha256` are
  sent in `request.options` (threaded through `CoreAIPolicy.predict_action`).
- **Action-contract integration** — the plugin persists the manifest action
  contract on bind and passes its `representation` / `horizon` / `action_dim`
  into the strict validator, so a single-action artifact returns `[A]` correctly
  and a wrong-horizon runner output now **fails** (v1.3.3's validator was always
  called in `chunk` mode with no horizon).

## Not yet

- Full batched evaluation (`batch_size > 1`) — v1.3.5.
- Real serializable `PolicyProcessorPipeline` + canonical Hub artifact layout
  (`config.json` / `policy_preprocessor.json` / …) — v1.3.5. LeRobot's
  `PolicyProcessorPipeline` restructures the batch (an empty pipeline is not a
  pass-through), so swapping it in changes the pre→policy→post composition and
  must be validated by the full `make_policy` E2E — its own PR.
- Live end-to-end `make_policy` → processors → fake-HTTP-runner test — v1.3.5.
- Official `lerobot-eval` end-to-end certification — v1.4.0.

Guarded real egress remains a separately enforced runtime.
