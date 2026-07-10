# LeRobot Eval v3 — Real Action Replay (v1.2.9)

> `eval-v2` (v1.1.4) only builds a feature mapping and evaluates **zero frames**.
> `eval-v3` actually replays frames: it serializes each observation through the
> JSON-safe boundary, calls the policy per-timestep, validates the action, records
> latency, and resets the policy at episode boundaries. It sends **no**
> robot/sim/real action and proves neither task success nor physical safety.

## CLI

```bash
lerobot-coreai eval-v3 \
  --policy.path kevinqz/EVO1-SO100-CoreAI \
  --runner.url http://127.0.0.1:8710 \
  --dataset.repo_id lerobot/pusht \
  --episodes 0,1 --stride 1 --max-frames 100 \
  --dataset.revision <sha> --policy.revision <sha> \
  --fail-fast \
  --output-dir reports/eval-v3
```

Writes `eval_v3_report.json` / `.md` and `eval_v3_trace.jsonl` (one record per
frame: index, episode, action_generated, action_valid, latency, detail).

## What it checks

- **`frames_evaluated > 0`** — a real replay ran (regression guard against the
  eval-v2 zero-frame behavior).
- **`actions_generated == frames_evaluated`** — every frame produced an action.
- **Per-action validation** — non-empty, finite, and dimension matches the
  action contract (`contracts.action.action_dim`, else inferred).
- **Latency** — p50 / p95 / max per-step.
- **Per-episode reset** — `policy.reset()` at each `episode_index` boundary
  (and at the start), so stateful/queued policies don't leak across episodes.

`ok` is true only when frames were evaluated, every frame generated an action,
and there were zero validation failures. `--fail-fast` stops on the first bad
action.

## Scope

`actions_sent_to_robot` / `actions_sent_to_simulator` are pinned to `0`
(schema-enforced). eval-v3 replays actions through the policy for numerical /
shape / latency observation only — it does not prove task success or physical
safety, and authorizes no actuation.
