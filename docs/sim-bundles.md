# Sim Bundles

v0.8.4 packages a simulator-only run into a reproducibility bundle: a
self-contained, auditable artifact you can archive, compare, or publish.

## What it is

A local directory containing:

- the source run's report, traces, actions, episodes, and CSVs
- derived `policy.json`, `environment.json`, `runner.json` metadata
- a `bundle_manifest.json` describing the run, results, safety, and claims
- a `checksums.json` with a SHA256 for every bundled file
- a human `README.md` and `reproducibility.md`

## What it does NOT prove

- Real-world task success
- Physical robot safety
- Robot readiness

A bundle records simulator behavior. It is not evidence of anything beyond the
simulator. The packager refuses to bundle a report that violates the
no-robot-egress invariants (see below).

## Create a bundle

Standalone, from an existing run directory:

```bash
lerobot-coreai package-sim-run \
  --run-dir runs/evo1-pusht-sim \
  --output-dir publish/evo1-pusht-sim-bundle
```

Or as part of a sim run:

```bash
lerobot-coreai sim \
  --policy.path kevinqz/EVO1-SO100-CoreAI \
  --env.type gym \
  --env.id PushT-v0 \
  --runner.url http://127.0.0.1:8710 \
  --episodes 10 \
  --confirm-sim-egress \
  --export-csv \
  --package-run \
  --package-output-dir publish/evo1-pusht-sim-bundle \
  --output-dir runs/evo1-pusht-sim
```

Packaging runs *after* `sim_report.json` is written and never alters the sim
results. If `--package-run` is set without `--package-output-dir`, the bundle
is written to `<output-dir>/bundle`. A packaging failure during `sim` is
recorded as a warning in the report; the sim result itself stays `ok`.

### package-sim-run flags

| Flag | Default | Description |
|------|---------|-------------|
| `--run-dir` | required | Source run directory (must contain `sim_report.json`) |
| `--output-dir` | required | Bundle output directory |
| `--overwrite` | off | Replace a non-empty output directory |
| `--redact-runner-url` | off | Replace the runner URL with `<redacted>` |
| `--no-redact-local-paths` | (paths redacted) | Keep absolute local paths |
| `--include-observations-dir` | off | Include the full `observations/` dir (may be large) |
| `--no-actions` | (included) | Exclude `actions.jsonl` / `episodes.jsonl` |
| `--no-traces` | (included) | Exclude `sim_trace.jsonl` |
| `--no-csv` | (included) | Exclude `episode_metrics.csv` / `step_metrics.csv` |
| `--no-summary` | (included) | Exclude `sim_summary.md` |
| `--no-failure-taxonomy` | (included) | Exclude `failure_taxonomy.json` |
| `--json` | off | Emit a JSON result |

Only `sim_report.json` is required. Any missing optional file produces a
warning, not a failure.

## Verify a bundle

```bash
lerobot-coreai verify-sim-bundle \
  --bundle-dir publish/evo1-pusht-sim-bundle
```

Verification checks, in order:

- `bundle_manifest.json` is present
- the manifest validates against `schemas/sim-bundle.schema.json` (which pins
  every safety/claims invariant)
- explicit invariant checks on `schema_version`, `bundle_type`, `mode`, the
  full `safety` block, and the `claims` block
- every checksum in `checksums.json` matches the file on disk
- `source_run/sim_report.json` is present and itself passes the
  no-robot-egress / no-overclaim invariants

```
✓ manifest schema valid
✓ checksums valid (11 files)
✓ source report valid
✓ no-robot-egress invariants preserved
Bundle verification passed.
```

A tampered file (any byte changed after packaging) fails checksum verification;
a tampered manifest fails schema/invariant verification. Either returns a
nonzero exit code.

## Redaction

- `--redact-local-paths` is **on by default**: absolute local paths are not
  copied into metadata. Pass `--no-redact-local-paths` to keep them.
- `--redact-runner-url` is **off by default** (localhost is usually useful).
  Enable it when the runner URL exposes a private IP or hostname.

## File layout

```
publish/evo1-pusht-sim-bundle/
├── bundle_manifest.json
├── checksums.json
├── README.md
├── reproducibility.md
├── environment.json
├── runner.json
├── policy.json
├── source_run/
│   ├── sim_report.json
│   ├── sim_summary.md
│   ├── failure_taxonomy.json
│   ├── sim_trace.jsonl
│   ├── actions.jsonl
│   ├── observations.jsonl
│   ├── episodes.jsonl
│   ├── episode_metrics.csv
│   └── step_metrics.csv
└── metadata/
    ├── package_info.json
    └── files.json
```

`source_run/observations/` is included only with `--include-observations-dir`.

## No-robot-egress invariants

The packager rejects a source `sim_report.json` (raising an error, writing
nothing) unless all of these hold:

- `mode == "sim"`
- `safety.robot_egress_enabled == false`
- `safety.actions_sent_to_robot == 0`
- `safety.action_egress == "simulator_only"`
- `safety.physical_actuation_possible == false`
- a `claims` block is present, with `proves_real_task_success == false`,
  `proves_robot_safety == false`, and `proves_real_world_safety == false`

The generated `bundle_manifest.json` is validated against
`schemas/sim-bundle.schema.json`, which pins the same invariants.
`verify-sim-bundle` re-checks all of them (via the schema and explicit
assertions) *and* re-validates the bundled `source_run/sim_report.json`, so a
bundle can be audited long after it leaves the machine that produced it.
