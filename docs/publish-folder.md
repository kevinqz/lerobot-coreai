# Publish Folder

When `--publish-ready` is passed and the export succeeds, a `publish/` folder is created.

## Structure

```
publish/
├── lerobot-coreai.json      # Manifest
├── model.aimodel/           # CoreAI artifact bundle
├── README.md                # Artifact card with verification summary
├── reports/
│   ├── export_report.json   # Full export pipeline report
│   ├── rollout_report.json  # Dry-run report (if run)
│   ├── eval_report.json     # Dataset eval report (if run)
│   └── compare_report.json  # Parity compare report (if run)
└── traces/
    ├── export_trace.jsonl   # Export event trace
    ├── eval_trace.jsonl     # Eval event trace (if run)
    └── compare_trace.jsonl  # Compare event trace (if run)
```

## What is NOT proven

- Task success — numeric parity does not prove the robot achieves the task
- Physical robot safety — no hardware was connected during verification
- No motor commands were sent during any verification step
