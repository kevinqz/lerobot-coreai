# Compare v2 — Source Loader v2 + Processor-Inclusive Compare (v1.2.6)

> The v0.5 compare could compare a raw CoreAI action against a LeRobot action at
> the wrong stage — high numeric parity, operationally invalid. compare-v2 loads
> the source policy the way LeRobot itself does and compares the **final** action
> after each side's declared processing. Metrics + per-frame trace only. No
> robot/sim/real egress; no task-success or physical-safety claim.

## CLI

```bash
lerobot-coreai compare-v2 \
  --torch.policy.path lerobot/diffusion_pusht \
  --coreai.policy.path kevinqz/EVO1-Pusht-CoreAI \
  --dataset.repo_id lerobot/pusht \
  --runner.url http://127.0.0.1:8710 \
  --max-frames 32 \
  --strict-processors \
  --output-dir reports/compare-v2
```

Writes `source_policy_load_report.json`, `processor_contract_report.json`,
`compare_v2_report.json` / `.md`, and `compare_v2_actions.jsonl`.

## Source loader v2 (official API)

```python
cfg = PreTrainedConfig.from_pretrained(policy_path)
ds_meta = LeRobotDatasetMetadata(dataset_repo_id)
policy = make_policy(cfg, ds_meta=ds_meta)
pre, post = make_pre_post_processors(cfg, pretrained_path=policy_path,
                                     dataset_stats=ds_meta.stats, dataset_meta=ds_meta)
```

It never calls `make_policy` with a string `policy_type` and never instantiates
the abstract `PreTrainedPolicy` base. Failures report the exact stage
(`config` / `dataset_meta` / `policy` / `processors`).

## Processor contract

The CoreAI manifest declares who owns processing:

```json
{
  "processor_contract": {
    "observation_input": {"expects": "raw_lerobot_observation",
                          "image_layout": "CHW", "image_range": [0.0, 1.0]},
    "action_output": {"returns": "postprocessed_action", "action_order": ["x", "y"]},
    "stats": {"dataset_stats_sha256": "sha256:..."}
  }
}
```

`expects ∈ {raw_lerobot_observation, policy_preprocessed_observation}`,
`returns ∈ {postprocessed_action, normalized_action}`. Under `--strict-processors`
an ambiguous/undeclared contract **fails closed** — because comparing a
normalized action against a postprocessed action is meaningless.

## Compare flow (same final unit)

```
dataset frame -> official preprocessor -> policy.select_action() -> official postprocessor -> source final action
dataset frame -> (per processor_contract) -> CoreAI runner -> coreai final action
compare(source final, coreai final)
```

Metrics: `mae`, `max_abs_error`, `cosine_similarity`, `relative_mae`, plus
`shape_match` and `finite` gates. A shape mismatch or non-finite value fails the
comparison rather than reporting a misleading number.

## Scope

Proves action parity **on the final unit** for the compared frames — not task
success, not physical safety. The official-API loader runs live in the stable CI
job (LeRobot 0.6.0); the metrics and processor-contract logic are pure and run
everywhere.
