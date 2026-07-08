# Eval — LeRobotDataset Replay

## What eval does

1. Loads the policy manifest from Hugging Face
2. Validates runner health and action support
3. Loads a LeRobotDataset via the public `LeRobotDataset(repo_id, ...)` constructor
4. Selects frames by `start_index`, `stride`, `max_frames`
5. For each frame: extracts observation → calls `predict_action` → validates action
6. Writes output files:
   - `actions.jsonl` — per-frame action records
   - `eval_trace.jsonl` — event trace
   - `eval_report.json` — structured report with metrics

## What eval does NOT do

- Does **not** connect to a physical robot
- Does **not** send motor commands
- Does **not** calculate task success or action parity (v0.5)
- Does **not** compare with PyTorch (v0.5)

## Usage

```bash
lerobot-coreai eval \
  --policy.path kevinqz/EVO1-SO100-CoreAI \
  --dataset.repo_id lerobot/evo1_so100_eval \
  --runner.url http://127.0.0.1:8710 \
  --max-frames 32 \
  --output-dir runs/evo1-eval
```

## Requirements

- `pip install "lerobot-coreai[lerobot]"` (requires Python 3.12+)
- A running coreai-runner with action support
- A LeRobotDataset accessible via HuggingFace (or local `--dataset.root`)

## Metrics

| Metric | Description |
|--------|-------------|
| `frames_requested` | Number of frames selected |
| `frames_processed` | Frames successfully processed |
| `actions_generated` | Actions successfully generated |
| `actions_failed` | Frames that failed (validation, runner error, etc.) |
| `shape_errors` | Action shape mismatches |
| `nan_errors` | Actions containing NaN |
| `inf_errors` | Actions containing Inf |
| `runner_errors` | Runner communication errors |
| `mean_total_ms` | Mean total inference time |
| `p95_total_ms` | 95th percentile inference time |
