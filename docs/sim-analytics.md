# Sim Analytics

v0.8.2 turns simulator-only runs into auditable evaluation artifacts.

> The analytics artifacts described here (report, CSVs, summary, taxonomy) can
> be packaged into a self-contained, checksummed reproducibility bundle — see
> [sim-bundles.md](sim-bundles.md) (v0.8.4).

## What it proves

- Simulator episode metrics (reward, success rate, episode length)
- Simulator latency metrics (runner / loop / env-step p50/p95/max)
- Action diagnostics (mean/max abs, NaN/Inf counts, shape changes)
- Failure taxonomy (errors classified by stage/type/category)

## What it does NOT prove

- Real-world task success
- Physical robot safety
- Robot readiness

These are simulator analytics only. They do not prove real-world task success
or physical robot safety.

## Artifacts

By default, a sim run writes:

| Artifact | Description |
|----------|-------------|
| `sim_report.json` | Full structured report including the analytics sections below |
| `sim_summary.md` | Human-readable markdown summary (results, timing, actions, failures, safety, claims) |
| `failure_taxonomy.json` | Errors classified by stage, type, and coarse category |

With `--export-csv`, a sim run also writes:

| Artifact | Description |
|----------|-------------|
| `episode_metrics.csv` | One row per episode (reward, success, steps, action counts) |
| `step_metrics.csv` | One row per step (timing, action diagnostics, error) |

## CLI

```bash
lerobot-coreai sim \
  --policy.path kevinqz/EVO1-SO100-CoreAI \
  --env.type fake \
  --runner.url http://127.0.0.1:8710 \
  --episodes 10 \
  --max-steps-per-episode 300 \
  --confirm-sim-egress \
  --export-csv \
  --output-dir runs/evo1-sim
```

| Flag | Default | Description |
|------|---------|-------------|
| `--export-csv` | off | Write `episode_metrics.csv` and `step_metrics.csv` |
| `--no-summary-md` | off (summary on) | Do not write `sim_summary.md` |
| `--no-failure-taxonomy` | off (taxonomy on) | Do not write `failure_taxonomy.json` |

## Report sections (v0.8.2)

`sim_report.json` gains four optional sections:

- **`episode_metrics`**: `episodes_completed`, `mean_reward`, `median_reward`,
  `min_reward`, `max_reward`, `success_rate`, `mean_steps`, `median_steps`,
  `terminated_episodes`, `truncated_episodes`.
- **`latency_metrics`**: `runner_p50_ms` / `runner_p95_ms` / `runner_max_ms`,
  `loop_p50_ms` / `loop_p95_ms` / `loop_max_ms`,
  `env_step_p50_ms` / `env_step_p95_ms` / `env_step_max_ms`.
- **`action_metrics`**: `mean_abs_action`, `max_abs_action`,
  `nan_action_steps`, `inf_action_steps`, `unique_action_shapes`,
  `shape_changes`.
- **`failure_metrics`**: `total_errors`, `runner_errors`, `env_errors`,
  `validation_errors`, `episodes_failed`, `error_rate`.

These are optional — older v0.8.0/v0.8.1 reports without them remain valid. The
safety `const` invariants (`actions_sent_to_robot = 0`,
`robot_egress_enabled = false`, `action_egress = "simulator_only"`) are unchanged.

## Failure taxonomy

`failure_taxonomy.json` classifies each error record:

```json
{
  "total_failures": 3,
  "by_stage": {"action.generate": 2, "simulator.step": 1},
  "by_type": {"RunnerTimeoutError": 1, "CoreAIPolicyError": 2},
  "classified": {"runner": 1, "environment": 1, "validation": 1, "unknown": 0},
  "first_failure": {"episode": 0, "step": 12, "stage": "action.generate", ...}
}
```
