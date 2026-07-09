# Safety Quality Gates

v0.9.2 turns [safety supervisor](safety-supervisor.md) findings into enforceable
CI gates.

> If unsafe actions appear, the run fails before it can graduate.

## What safety gates prove

Safety gates can prove that a **specific artifact satisfied configured software
thresholds**. They do **not** prove:

- physical robot safety
- real-world task success
- future policy safety
- hardware safety
- absence of all unsafe behavior

## Gate a safety summary

```bash
lerobot-coreai safety-gate \
  --run-dir runs/pusht-supervised \
  --max-actions-blocked 0 \
  --max-critical-findings 0 \
  --max-would-block-actions 0 \
  --fail-on-safety-quality
```

Inputs (resolved in this priority): `--safety-summary`, `--sim-report`,
`--profile-fit`, `--run-dir` (looks for `safety_summary.json` then
`sim_report.json`), `--bundle-dir` (looks under `source_run/`).

| Flag | Default | Description |
|------|---------|-------------|
| `--max-actions-blocked` | `0` | Max blocked actions |
| `--max-block-rate` | `0.0` | Max blocked / supervised |
| `--max-critical-failures` | `0` | Max critical failures (blocked) |
| `--max-critical-findings` | `0` | Max critical findings (incl. report_only would-block) |
| `--max-would-block-actions` | `0` | Max report_only would-block actions |
| `--max-would-block-rate` | `0.0` | Max would-block rate |
| `--max-actions-modified` | (off) | Max clipped/modified actions |
| `--max-modification-rate` | (off) | Max modification rate |
| `--max-clip-rate` | (off) | Max clip rate |
| `--max-delta-failures` | `0` | Max delta-bound failures |
| `--max-shape-failures` | `0` | Max shape failures |
| `--max-nonfinite-failures` | `0` | Max NaN/Inf failures |
| `--min-actions-supervised` | (off) | Require at least N supervised actions |
| `--allow-summary-failed` | off | Do not require `summary.passed` |
| `--allow-parse-errors` | off | Do not require zero parse errors |
| `--fail-on-safety-quality` | off | Return rc 1 when gates fail |

Without `--fail-on-safety-quality`, `safety-gate` is **report-only** (always rc 0
unless the input is missing/unreadable) — useful for exploration. The defaults
are zero-tolerance: fail-closed is the right posture pre-real-mode.

## Gate inside sim

```bash
lerobot-coreai sim \
  --policy.path kevinqz/EVO1-SO100-CoreAI \
  --runner.url http://127.0.0.1:8710 \
  --env.type gym --env.id PushT-v0 --confirm-sim-egress \
  --safety.profile-name so100-sim-default \
  --supervisor.mode enforce \
  --safety.max-actions-blocked 0 \
  --safety.max-critical-findings 0 \
  --safety.fail-on-safety-quality \
  --output-dir runs/pusht-gated
```

> **Flag surface:** `safety-gate` exposes the full gate surface (all
> `--max-*` thresholds, `--min-actions-supervised`, `--allow-*`). The integrated
> `sim` / `supervisor-check` `--safety.*` flags expose the common gates only —
> pipe a run through `safety-gate` for the complete set. Safety gates also
> require the supervisor to be enabled: `--supervisor.mode off` together with a
> safety gate flag is rejected (fail-closed).

A failed gate with `--safety.fail-on-safety-quality` sets `SimResult.ok = false`
and returns rc 1, while preserving the no-robot-egress invariants
(`actions_sent_to_robot = 0`, `robot_egress_enabled = false`). The run writes
`safety_quality_report.json/md` and a `safety_quality` section in
`sim_report.json`. `supervisor-check` accepts the same `--safety.*` flags.

## Report

`safety_quality_report.json` carries the derived metrics, per-check results, and
honest claims:

```json
{
  "proves_software_safety_quality": true,
  "proves_physical_safety": false,
  "proves_real_world_safety": false,
  "proves_real_task_success": false
}
```

See also [safety regression](safety-regression.md) for baseline-vs-candidate
comparison.
