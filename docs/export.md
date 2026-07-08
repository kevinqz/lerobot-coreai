# Export — Verify/Package Pipeline

## What export does

Orchestrates the full artifact lifecycle:

1. **Export** a LeRobot PyTorch policy to CoreAI `.aimodel` via `coreai-fabric`
2. **Generate** `lerobot-coreai.json` manifest
3. **Verify** the manifest against the schema
4. Optionally: **runner smoke test**, **dry_run**, **dataset eval**, **compare/parity**
5. **Package** a publish-ready folder with all artifacts and reports

## What export does NOT do

- Does **not** connect to a physical robot
- Does **not** send motor commands
- Does **not** claim task success or physical safety

## Usage

### Minimal (skip fabric, use existing artifact)

```bash
lerobot-coreai export \
  --torch.policy.path lerobot/evo1_so100 \
  --skip-fabric \
  --existing-artifact /path/to/model.aimodel \
  --output-dir runs/export
```

### Full pipeline

```bash
lerobot-coreai export \
  --torch.policy.path lerobot/evo1_so100 \
  --policy.type evo1 \
  --robot.type so100 \
  --dataset.repo_id lerobot/evo1_so100_eval \
  --runner.url http://127.0.0.1:8710 \
  --output-dir runs/evo1-export \
  --verify-runner \
  --eval-max-frames 32 \
  --compare-max-frames 32 \
  --publish-ready
```

## Requirements

- `pip install "lerobot-coreai[fabric]"` for fabric export
- `pip install "lerobot-coreai[lerobot,fabric]"` for eval/compare verification
- A running `coreai-runner` for runner-dependent verification steps

## Output files

| File | Description |
|------|-------------|
| `export_report.json` | Full pipeline report with claims and safety invariants |
| `export_trace.jsonl` | Event trace |
| `lerobot-coreai.json` | Manifest |
| `publish/` | Publish-ready folder (with `--publish-ready`) |
