# LeRobot Compatibility Levels (v1.2.4)

> The old certificate answered one question — "shape-compatible?" — and could
> imply more than it proved. The **leveled contract report** reports each rung of
> the official LeRobot contract separately and honestly. Every level is either
> genuinely tested or marked `failed` / `not_supported` / `not_tested` — never
> assumed.

```bash
lerobot-coreai lerobot-compat-check --contract --output-dir reports/lerobot-contract
# add --strict in CI to require the stable target (LeRobot 0.6.0) to pass
```

Writes `lerobot_compatibility_report_v1.json` / `.md`.

## Targets

| Target | Version | CI | Meaning |
|--------|---------|----|---------|
| **stable** | `0.6.0` | blocking | The certified target, pinned exactly. |
| **development** | post-`0.6.0` (`0.6.1`) | non-blocking, pinned commit | A probe; never a moving `@main`. |

## Levels

| Level | Today | Why |
|-------|-------|-----|
| `base_package_import` | ✅ passed | Base package imports with no torch/lerobot. |
| `lerobot_version_supported` | ✅/– | `>=0.6.0,<0.7.0` when LeRobot installed. |
| `dataset_constructor` | ✅/– | `LeRobotDataset` module present. |
| `dataset_frame_read` | – not_tested | Real replay is Eval v3's job. |
| `action_method_name` | ✅ passed | Bridge exposes `select_action`. |
| `action_semantics` | ❌ failed | Chunk passthrough, not per-timestep. |
| `action_tensor_contract` | ❌ failed | Returns list, not `torch.Tensor (B, A)`. |
| `action_batch_contract` | ❌ failed | Runner is non-batched today. |
| `official_plugin_discovery` | ❌ failed | Package isn't `lerobot_policy_*`. |
| `official_config_registry` | ❌ failed | Config not registered upstream. |
| `official_policy_factory` | ❌ failed | Not `make_policy`-compatible. |
| `official_processor_pipeline` | ❌ failed | No pre/post processors. |
| `official_eval` | ❌ failed | Not a `PreTrainedPolicy`/`nn.Module`. |
| `official_rollout_sync` | not_supported | Not integrated. |
| `official_rollout_rtc` | not_supported | Not integrated. |
| `guarded_real_separate_runtime` | separate_runtime | Enforced separately via `real --mode guarded`. |

## Claims

The report pins these **false** (schema-enforced):
`official_plugin_compatible`, `official_eval_compatible`,
`official_rollout_compatible`, `native_upstream_registry`, `supports_training`,
`proves_physical_safety`. Only `shape_compatible` may be true, and only when the
shape-level checks pass. The report cannot mark `action_semantics=passed` while
`select_action` remains chunk-passthrough, nor `official_eval` true without a real
eval — those are honesty invariants enforced by `--contract`'s consistency check.

## Roadmap

`action_semantics` → Action Contract v2. `official_*` → the out-of-tree plugin
`lerobot_policy_coreai_bridge` (`PreTrainedPolicy`/`nn.Module`, registered config,
processor factory, official eval). Guarded real egress stays a separate runtime.
