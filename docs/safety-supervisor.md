# Runtime Safety Supervisor

v0.9.0 adds a **software** action supervisor that validates, bounds, clips,
blocks, and audits every action **before egress**.

> Generate actions freely. Egress only under supervision.

## What it does

- Validates action presence, finiteness (no NaN/Inf), and shape
- Enforces bounds (min/max, max abs), step delta, and L2 norm
- Optionally clips out-of-bound actions to the profile bounds
- Blocks unsafe actions (fail-closed) and records an auditable decision
- Writes `safety_report.jsonl`, `safety_summary.json`, and `safety_summary.md`

## What it does NOT do

- Prove physical robot safety
- Replace a hardware emergency stop
- Certify a robot
- Enable unrestricted real-world actuation

This is a software layer for simulator and future *guarded* real-mode
workflows. It never connects to a robot and never sends motor commands.

## The rule

**No supervised decision, no egress.** The supervisor is fail-closed: a missing
profile, a missing action, a NaN/Inf, an unknown/mismatched shape, an exceeded
bound/delta/norm, a robot-type mismatch, or an internal supervisor error all
result in a **blocked** action in enforce mode.

## Modes

| Mode | Behavior |
|------|----------|
| `off` | Supervisor does not run. Allowed only for backward compatibility. |
| `report_only` | Supervisor runs and records decisions but never blocks or modifies egress. Pure diagnosis (calibrate profiles). |
| `enforce` | Supervisor runs and blocks egress when a decision is not allowed. **Default for sim.** |

**For any current or future egress path, `enforce` is the only
production-safe supervisor mode.** `report_only` and `off` are for diagnosis and
backward compatibility only. A `report_only` run that encounters an unsafe
action does **not** pass: its `safety_summary.json` records `would_block_actions`
/ `critical_findings` and sets `passed: false`, so findings are never silently
masked.

Defaults:

- **sim**: `enforce`
- **shadow**: `off` (shadow already blocks everything via the ActionBlocker; the
  supervisor is opt-in diagnostic there)

When an action is blocked in sim `enforce`, the episode is terminated as
`safety_terminated` (`terminated_by: "safety_supervisor"`). No no-op action is
invented — the run stays honest.

## Profiles

A [safety profile](safety-profiles.md) is a conservative software contract for
action bounds. Pass one with `--safety.profile <path>` or `--safety.profile-name
<builtin>`. If none is given, the conservative built-in `default-sim-safe`
(finite-only, no bounds) is used.

The v0.9.1 profile toolkit lets you choose and tune profiles without touching
hardware: `profile-list` / `profile-show` / `profile-validate`,
[`profile-recommend`](profile-recommendation.md) to pick one from a policy or
actions log, [`profile-calibrate`](profile-calibration.md) to fit bounds to
observed actions, and `profile-compare` to diff two profiles over the same
actions.

## CLI

Run sim with the supervisor:

```bash
lerobot-coreai sim \
  --policy.path kevinqz/EVO1-SO100-CoreAI \
  --env.type gym --env.id PushT-v0 \
  --runner.url http://127.0.0.1:8710 \
  --episodes 10 --confirm-sim-egress \
  --safety.profile-name so100-sim-default \
  --supervisor.mode enforce \
  --output-dir runs/pusht-supervised
```

| Flag | Default | Description |
|------|---------|-------------|
| `--supervisor.mode` | `enforce` | `off` / `report_only` / `enforce` |
| `--safety.profile` | — | Path to a profile JSON |
| `--safety.profile-name` | `default-sim-safe` | Built-in profile name |
| `--no-safety-report` | (written) | Skip safety artifacts |

Check an actions file offline:

```bash
lerobot-coreai supervisor-check \
  --actions runs/pusht/actions.jsonl \
  --safety.profile-name so100-sim-default \
  --output-dir runs/pusht-safety-check \
  --fail-on-block
```

`supervisor-check` returns rc `1` when `--fail-on-block` is set and any action is
blocked, otherwise rc `0`.

## Reports

`safety_report.jsonl` — one auditable decision per action:

```json
{"episode": 0, "step": 12, "allowed": false, "severity": "critical",
 "reasons": ["finite"], "checks": [{"name": "finite", "passed": false}],
 "profile": "so100-sim-default"}
```

`safety_summary.json` / `safety_summary.md` — aggregate counts and top reasons.
The summary always carries honest claims:

```json
{"proves_software_supervision": true, "proves_physical_safety": false,
 "proves_real_world_safety": false, "proves_real_task_success": false}
```

`sim_report.json` gains a `safety_supervisor` section (mode, profile, counts,
top reasons), and reproducibility bundles copy the safety artifacts and record
the section in `bundle_manifest.json`. `verify-sim-bundle` rejects a bundle
whose safety summary overclaims physical/real-world safety.

## Roadmap

- v0.9.1 — curated robot-family safety profiles
- v0.9.2 — supervisor quality gates (block rate, modification rate, critical failures)
- v0.9.3 — operator approval artifacts
- v1.0.0 — guarded real mode (supervisor enforced, explicit confirmation, no bypass)
