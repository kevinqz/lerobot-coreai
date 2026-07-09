# Shadow Diagnostics & Quality Gates

Shadow quality gates evaluate a completed run's metrics against configurable thresholds.
Added in v0.7.2.

## Philosophy

Quality gates are **report-only by default**. They tell you whether a run met your
quality bar, but they don't fail the run. Use `--quality.fail-on-quality` to make
quality gate failures set `result.ok=False`.

> Quality gates can fail a shadow run, but they do not prove physical robot safety.

## Available gates

| Gate | CLI arg | Description |
|------|---------|-------------|
| Runner p95 latency | `--quality.max-runner-p95-ms` | Max acceptable runner p95 time |
| Loop p95 latency | `--quality.max-loop-p95-ms` | Max acceptable loop p95 time |
| Error rate | `--quality.max-error-rate` | Max fraction of steps that errored (default: 0.0) |
| NaN actions | `max_nan_actions` (internal) | Max acceptable NaN action values |
| Inf actions | `max_inf_actions` (internal) | Max acceptable Inf action values |
| Shape changes | `allow_action_shape_changes` (internal) | Whether action shape may change mid-run |
| Min effective FPS | `--quality.min-effective-fps` | Minimum real paced FPS (wall-clock, includes sleep) |

## Usage

```bash
lerobot-coreai shadow \
  --policy.path kevinqz/EVO1-SO100-CoreAI \
  --observation-source folder \
  --frames-dir data/shadow_frames \
  --output-dir runs/shadow-quality \
  --quality.max-runner-p95-ms 50 \
  --quality.max-loop-p95-ms 100 \
  --quality.min-effective-fps 5 \
  --quality.max-error-rate 0.0 \
  --quality.fail-on-quality
```

## Report output

The `quality` section in `shadow_report.json`:

```json
{
  "quality": {
    "passed": true,
    "checks": [
      {"name": "max_runner_p95_ms", "passed": true, "value": 15.2, "threshold": 50.0},
      {"name": "max_loop_p95_ms", "passed": true, "value": 20.1, "threshold": 100.0},
      {"name": "max_error_rate", "passed": true, "value": 0.0, "threshold": 0.0},
      {"name": "max_nan_actions", "passed": true, "value": 0, "threshold": 0},
      {"name": "max_inf_actions", "passed": true, "value": 0, "threshold": 0},
      {"name": "no_action_shape_changes", "passed": true, "value": 0, "threshold": 0},
      {"name": "min_effective_fps", "passed": true, "value": 9.7, "threshold": 5.0}

> `min_effective_fps` checks the **real paced FPS** (wall-clock duration including
> sleep/pacing), not the compute-only `processing_fps`. When `fps=0` (no pacing),
> `effective_fps` is `None` and this gate is skipped.
    ]
  }
}
```

## When gates fail

- Without `--quality.fail-on-quality`: `quality.passed=false` in report, but `result.ok=true`
- With `--quality.fail-on-quality`: `quality.passed=false` and `result.ok=false`

Safety invariants (`actions_sent=0`, `action_egress=blocked`, etc.) are enforced by
schema regardless of quality gate results.
