# Action Contract v2 + Batch/Reset Semantics (v1.2.5)

> The v1.2.4 compatibility contract honestly reported that `select_action`'s
> semantics differ from LeRobot's (chunk passthrough, not per-timestep). v1.2.5
> makes the contract **explicit and machine-readable** and adds the correct
> per-timestep path — **without breaking** the historical `select_action`
> behavior. No hardware, no egress, no training or physical-safety claims.

## Two semantics, clearly separated

| Method | Returns | Semantics |
|--------|---------|-----------|
| `predict_action_chunk(batch)` | `[H, A]` | The full action chunk. |
| `select_action(batch)` | chunk `[H, A]` (chunked policy) | **Legacy** CoreAI behavior — unchanged for backward compatibility. |
| `select_next_action(batch)` | `[A]` | **LeRobot-correct** per-timestep: pops one action from an internal queue, refilling from a fresh chunk when empty. |
| `reset()` | – | Clears the queue and calls the runner's session reset when supported. |

The official out-of-tree plugin (v1.3.x) will expose LeRobot's
`select_action(batch) -> torch.Tensor(B, A)` semantics directly; the local
`CoreAIPolicy.select_action` stays legacy until a v2.0.0 major bump.

## Contracts

`parse_action_contract_from_manifest()` honors an explicit `action_contract`
block, else infers safely: a 2D action shape `[H, A]` → `chunk` of horizon H; a
1D `[A]` → `single`. Inference records shape only — it never asserts LeRobot
semantics on its own.

```json
{
  "action_contract": {
    "representation": "chunk", "horizon": 16, "action_dim": 7,
    "select_action_semantics": "next_action",
    "queue_owner": "python_bridge", "reset_clears_queue": true
  },
  "batch_contract": {"supports_batch": false, "max_batch_size": 1,
                     "fallback": "split_and_stack"}
}
```

## Batching (split-and-stack)

LeRobot eval passes batched observations (e.g. `observation["task"]` is a list
per env). The runner is non-batched today, so with `fallback="split_and_stack"`
the bridge splits a batched observation into single samples, runs each, and
stacks the actions back deterministically. Inconsistent batch sizes and
`fallback="reject"` raise clear errors.

## Queue safety

`ActionQueue` rejects ragged chunks (uneven row widths), non-finite values
(NaN/Inf), and empty chunks, and raises a clear error on exhaustion rather than
returning a stale action.

## Compatibility contract updates

`action_method_name` stays `passed`; `action_semantics` and
`action_tensor_contract` stay `failed` (the default `select_action` is still a
chunk and nothing returns a `torch.Tensor(B, A)` yet); `action_batch_contract`
moves to **`partial`** now that split-and-stack exists. These flip to `passed`
only with the official plugin.

## Migration

- **v1.2.5** (this): add `predict_action_chunk` / `select_next_action` / queue /
  batching; legacy `select_action` unchanged; docs corrected.
- **v1.3.0**: the official plugin is born with LeRobot-correct semantics.
- **v2.0.0**: `CoreAIPolicy.select_action` becomes per-timestep by default.
