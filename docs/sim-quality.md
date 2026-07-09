# Sim Quality Gates

v0.8.3 evaluates a sim run's analytics against configurable thresholds. Quality
gates turn a sim run into a pass/fail decision usable in CI.

## What it proves

- The policy met minimum sim performance criteria (success rate, reward, latency)

## What it does NOT prove

- Real-world task success
- Physical robot safety
- Robot readiness

Quality gates are development signals, not safety proof.

## CLI

```bash
lerobot-coreai sim \
  --policy.path kevinqz/EVO1-SO100-CoreAI \
  --env.type gym \
  --env.id PushT-v0 \
  --runner.url http://127.0.0.1:8710 \
  --episodes 10 \
  --confirm-sim-egress \
  --quality.min-success-rate 0.8 \
  --quality.max-runner-p95-ms 20.0 \
  --quality.fail-on-quality \
  --output-dir runs/evo1-sim
```

| Flag | Description |
|------|-------------|
| `--quality.min-success-rate` | Minimum fraction of successful episodes |
| `--quality.min-mean-reward` | Minimum mean episode reward |
| `--quality.max-runner-p95-ms` | Maximum runner p95 latency (ms) |
| `--quality.max-env-step-p95-ms` | Maximum env-step p95 latency (ms) |
| `--quality.max-loop-p95-ms` | Maximum loop p95 latency (ms) |
| `--quality.max-error-rate` | Maximum step error rate (default 0.0) |
| `--quality.fail-on-quality` | Set result.ok=False when gates fail |

Without `--quality.fail-on-quality`, gates are report-only (they appear in the
`quality` section of `sim_report.json` but don't fail the run).

## Report section

```json
{
  "quality": {
    "passed": false,
    "checks": [
      {"name": "min_success_rate", "passed": false, "value": 0.6, "threshold": 0.8}
    ]
  }
}
```
