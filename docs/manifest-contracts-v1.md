# Manifest Contracts v1 (v1.2.8)

> The action / batch / processor contracts previously existed only as test dicts
> or shape inference — the base manifest schema (`additionalProperties: false`)
> actually **rejected** a manifest that carried them. v1.2.8 adds an optional
> `contracts` block to the manifest so a published, schema-valid artifact can
> declare its semantics, plus a single JSON-safe observation boundary.

## The `contracts` block

```json
{
  "schema_version": "lerobot-coreai.v0",
  "...": "...",
  "contracts": {
    "action": {
      "representation": "chunk", "horizon": 16, "action_dim": 7,
      "selection_semantics": "next_action", "queue_owner": "python_bridge"
    },
    "batch": {
      "runner_supports_batch": false, "max_batch_size": 1,
      "fallback": "split_and_stack"
    },
    "processor": {
      "observation_input": {"expects": "raw_lerobot_observation",
                            "image_layout": "CHW", "image_range": [0.0, 1.0]},
      "action_output": {"returns": "postprocessed_action", "action_order": ["x", "y"]},
      "bindings": {"dataset_stats_sha256": "sha256:..."}
    }
  }
}
```

## Precedence (backward compatible)

Each parser resolves in this order:

1. **v1** — `contracts.action` / `contracts.batch` / `contracts.processor`.
2. **v0** — top-level `action_contract` / `batch_contract` / `processor_contract`
   (still accepted for existing dicts/tests).
3. **Inference** — action shape only (`[H,A]`→chunk, `[A]`→single). Inference
   never asserts LeRobot semantics or processor ownership.

`--strict-processors` still fails closed when neither a v1 nor v0 processor
contract is present.

## JSON-safe observation boundary

`coreai_observation_serialization.serialize_observation(obs)` is the single
boundary that converts a LeRobotDataset item to JSON-safe values before it
reaches the runner:

- primitives / lists / dicts pass through;
- `torch.Tensor` / `numpy.ndarray` → `{"__array__": [...], "dtype": ..., "shape": [...]}`
  (JSON-safe **and** shape/dtype auditable);
- unknown objects are **refused** (never silently coerced);
- `serialize_and_hash()` returns the payload plus a deterministic `sha256:` of
  exactly what would be sent.

compare-v2's CoreAI path now runs observations through this boundary.

## Still deferred (v1.2.9+)

Real execution of the `normalized_action` / `policy_preprocessed_observation`
modes (applying/undoing normalization), temporal alignment, and live-fixture
compares remain on the roadmap; compare-v2 stays **experimental**.
