# Live Metrics

Live metrics collection provides per-step timing, action diagnostics, and runtime signals
for shadow mode runs. Added in v0.7.2.

## What live metrics capture

| Metric | Description |
|--------|-------------|
| `loop_ms` | Wall time for one step (read → predict → block → log) |
| `runner_ms` | Runner predict_action call time |
| `action_shape` | Shape of the action chunk |
| `action_mean_abs` | Mean absolute value of action elements |
| `action_max_abs` | Max absolute value of action elements |
| `action_nan_count` | NaN values in the action |
| `action_inf_count` | Inf values in the action |

## Summary fields

The `live_metrics` section in `shadow_report.json`:

| Field | Description |
|-------|-------------|
| `samples` | Number of steps collected |
| `mean_loop_ms` | Mean per-step wall time |
| `p50_loop_ms` | Median per-step wall time |
| `p95_loop_ms` | 95th percentile per-step wall time |
| `max_loop_ms` | Maximum per-step wall time |
| `mean_runner_ms` | Mean runner predict time |
| `p95_runner_ms` | 95th percentile runner predict time |
| `processing_fps` | Compute-only throughput (excludes sleep/pacing) |
| `effective_fps` | Real paced FPS from wall-clock duration (includes sleep) |
| `latency_spikes` | Steps where loop_ms > 2× mean |
| `nan_actions` | Total NaN action values across run |
| `inf_actions` | Total Inf action values across run |
| `shape_changes` | Steps where action shape changed from previous |

`processing_fps` shows how fast the system can infer (read → predict → block → log).
`effective_fps` shows how close the run stayed to the target FPS, including pacing sleeps.
When `fps=0` (no pacing), `effective_fps` is `None` and only `processing_fps` is reported.

## Action diagnostics in actions.jsonl

Each line in `actions.jsonl` now includes a `diagnostics` object:

```json
{
  "step": 0,
  "diagnostics": {
    "mean_abs": 0.012,
    "max_abs": 0.08,
    "nan_count": 0,
    "inf_count": 0
  }
}
```

## Live console output

Use `--live` to print compact per-step metrics:

```bash
lerobot-coreai shadow --live --live-every 1 ...
```

Output format:
```
[shadow] step=12 obs=ok action=ok blocked=yes loop=15.8ms runner=12.3ms processing_fps=63.3 shape=[16,7]
```

Use `--live-every N` to print every N steps instead of every step.

## Important

Diagnostics are **development signals, not safety proof**. They help you understand
runtime behavior, latency, and action quality, but they do not prove task success or
physical robot safety.
