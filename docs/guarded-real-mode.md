# Guarded Real Mode

v1.0.0 introduces the **first guarded real-egress path** — the first time
`actions_sent_to_robot` can exceed zero, and only inside `real --mode guarded`,
only after every gate passes, only through the `RealEgressGuard`.

> No verified readiness, no real egress.

This is **guarded real egress for CoreAI-backed, LeRobot-shaped policies**. It is
**not** unrestricted autonomy, **not** proof of physical safety, and **not** a
full native LeRobot integration.

## Required evidence (all mandatory in guarded mode)

- a verified sim [bundle](sim-bundles.md)
- a [release readiness](release-readiness.md) report with `ready=true`
- a valid, unexpired [operator approval](operator-approval.md) bound to the bundle
- a [safety profile](safety-profiles.md) matching the robot type
- the [safety supervisor](safety-supervisor.md) in `enforce` (hardcoded)
- a bounded session (`--max-steps`, bounded `--fps`, optional bounded `--duration-seconds`)
- three explicit operator attestations
- an adapter that passes preflight

Preflight is strict about the readiness report: it must **validate against the
schema**, be `ready=true` with `bundle.verified` / `approval.valid` /
`evidence.safety_quality_passed` / `evidence.safety_regression_passed` all true,
carry no physical/real-world/actuation overclaim, and its `bundle`/`approval`
paths must reference the **same** `--bundle-dir` and `--approval` you passed. The
safety profile must be **intended for real** (its `intended_modes` must include a
guarded-real mode) — a sim/shadow-only profile is refused.

**Evidence cross-binding (v1.0.4):** the policy and robot type you run must be
the **same** the bundle evidence was produced for — preflight reads the bundled
`sim_report.json` and refuses a `--policy.path` / `--robot.type` that disagrees
with it.

**Observation config (v1.0.4):** a non-mock (real) adapter must declare its
observation config — `--obs.config <json>` or explicit `--obs.*` flags
(`--obs.image-key`, `--obs.state-key`, `--obs.task`, `--obs.require-state`,
`--obs.require-task`, `--obs.required-keys`, `--obs.drop-unknown-keys`).
Real observations vary and must not fall back to defaults; the mock adapter is
exempt.

## Modes

| Mode | Behavior |
|------|----------|
| `preflight` | Runs every gate, sends **zero** actions, writes `real_preflight_report.json`. |
| `guarded` | Runs the bounded session; a supervised action may reach the adapter. |

## Example

```bash
# 1. Preflight — verifies everything, sends nothing.
lerobot-coreai real --mode preflight \
  --policy.path kevinqz/EVO1-SO100-CoreAI --runner.url http://127.0.0.1:8710 \
  --robot.adapter mock --robot.type so100 \
  --safety.profile profiles/so100-real-guarded.json \
  --readiness-report readiness/evo1/release_readiness_report.json \
  --approval approvals/evo1/approval_manifest.json \
  --bundle-dir publish/evo1-bundle \
  --output-dir runs/evo1-real-preflight

# 2. Guarded — bounded session; requires operator + explicit attestations.
lerobot-coreai real --mode guarded \
  --policy.path kevinqz/EVO1-SO100-CoreAI --runner.url http://127.0.0.1:8710 \
  --robot.adapter mock --robot.type so100 \
  --safety.profile profiles/so100-real-guarded.json \
  --readiness-report readiness/evo1/release_readiness_report.json \
  --approval approvals/evo1/approval_manifest.json \
  --bundle-dir publish/evo1-bundle \
  --operator "Kevin Saltarelli" --max-steps 10 --fps 2 \
  --i-understand-this-may-move-real-hardware \
  --i-have-physical-emergency-stop-ready \
  --i-confirm-robot-workspace-is-clear \
  --output-dir runs/evo1-real-guarded
```

## Adapters

The only thing that touches a robot is a [robot adapter](real-adapters.md). The
built-in `mock` adapter touches no hardware (for testing the full gated flow
safely). Real hardware runs through an operator-provided `external-http` adapter,
behind every gate. There is **no** bundled motor/serial driver and **no** hidden
fallback to any robot API.

## What this does not prove

- physical robot safety
- real-world task success
- authorization for unrestricted real-world actuation
- future policy safety

Every real report pins `proves_physical_safety=false`,
`proves_real_world_safety=false`, and
`authorizes_unrestricted_real_world_actuation=false`. See
[real-mode-safety.md](real-mode-safety.md).
