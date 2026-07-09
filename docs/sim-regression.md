# Sim Regression

v0.8.3 compares a candidate sim run against a baseline to detect regressions in
success rate, reward, or latency.

## What it proves

- The candidate policy did not significantly degrade versus the baseline in sim

## What it does NOT prove

- Real-world task success
- Physical robot safety
- That the candidate is "good" in absolute terms (only relative to the baseline)

## CLI

```bash
lerobot-coreai sim-regression \
  --baseline runs/baseline/sim_report.json \
  --candidate runs/candidate/sim_report.json \
  --max-success-drop 0.05 \
  --max-reward-drop 2.0 \
  --max-runner-p95-increase-ms 5.0
```

| Flag | Description |
|------|-------------|
| `--baseline` | Path to baseline sim_report.json (required) |
| `--candidate` | Path to candidate sim_report.json (required) |
| `--max-success-drop` | Max allowed success-rate decrease before regression |
| `--max-reward-drop` | Max allowed mean-reward decrease before regression |
| `--max-runner-p95-increase-ms` | Max allowed runner-p95 latency increase (ms) |
| `--json` | Print result as JSON |

Exit code is 0 (passed) or 1 (regression detected).

## Output

```json
{
  "passed": true,
  "deltas": {
    "success_rate_delta": -0.02,
    "mean_reward_delta": -1.1,
    "runner_p95_delta_ms": 2.0
  },
  "checks": [
    {"name": "max_success_drop", "passed": true, "value": 0.02, "threshold": 0.05}
  ]
}
```

A typical CI workflow: run `sim` on the baseline policy, run `sim` on the
candidate, then run `sim-regression` as a gate.
