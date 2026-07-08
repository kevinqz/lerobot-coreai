# Compare — PyTorch vs CoreAI Action Parity

## What compare does

1. Loads a CoreAI-backed LeRobot policy (from HF artifact)
2. Loads a source PyTorch LeRobot policy
3. Resets both policies
4. Loads a LeRobotDataset
5. For each frame: runs both policies on identical observations → computes metrics
6. Writes `compare_actions.jsonl`, `compare_trace.jsonl`, `compare_report.json`

## What compare does NOT do

- Does **not** connect to a physical robot
- Does **not** send motor commands
- Does **not** claim task success
- Does **not** claim physical safety
- Numeric parity is a conversion-fidelity metric, not a behavioral metric

## Usage

```bash
lerobot-coreai compare \
  --torch.policy.path lerobot/evo1_so100 \
  --coreai.policy.path kevinqz/EVO1-SO100-CoreAI \
  --dataset.repo_id lerobot/evo1_so100_eval \
  --runner.url http://127.0.0.1:8710 \
  --max-frames 32 \
  --output-dir runs/evo1-compare
```

## Metrics

| Metric | Description |
|--------|-------------|
| `cosine_similarity` | Cosine similarity between flattened actions (1.0 = identical) |
| `mean_absolute_error` | Mean per-element absolute difference |
| `max_absolute_error` | Maximum per-element absolute difference |
| `relative_mae` | Scale-invariant relative MAE |

## Tolerances

Default pass thresholds (configurable):

| Tolerance | Default | Flag |
|-----------|---------|------|
| Min cosine similarity | 0.999 | `--tolerance.cosine` |
| Max absolute error | 1e-4 | `--tolerance.max-mae` |
| Max mean absolute error | 1e-5 | `--tolerance.mean-mae` |

## Notes

Source PyTorch policy loading is experimental in v0.5 alpha and depends on LeRobot 0.6.x
public policy loading APIs. Use `--torch.policy.type` when inference fails.
