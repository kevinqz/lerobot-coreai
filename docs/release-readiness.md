# Release Readiness

v0.9.3's `release-readiness` is the final go/no-go: it combines bundle
verification, a valid [operator approval](operator-approval.md), and the
required evidence into a single decision.

```bash
lerobot-coreai release-readiness \
  --bundle-dir publish/evo1-pusht-bundle \
  --approval approvals/evo1-pusht/approval_manifest.json \
  --output-dir readiness/evo1-pusht
```

`ready = true` only when **all** hold:

- the bundle verifies (`verify-sim-bundle`)
- the approval verifies (schema, hashes, attestation) and is **not expired**
- `sim_report.json` and `safety_summary.json` are present
- `safety_quality_report.json` is present and `passed`
- checksums are valid

It writes `release_readiness_report.json` / `.md`. Exit code is `0` when ready,
`1` otherwise.

Release readiness is **stricter than `approve-bundle`**: a missing/failed safety
regression blocks readiness by **default**, even if the operator waived it at
approval time. It downgrades to a warning only with `--allow-missing-regression`
here **and** a matching waiver in the approval.

## What readiness proves

Release readiness is scoped to **software evidence**. For v0.9.3, the strongest
readiness scope is preparation for a future *guarded* real mode. The report
carries honest claims:

```json
{
  "proves_release_readiness_for_scope": true,
  "proves_physical_safety": false,
  "proves_real_world_safety": false,
  "authorizes_unrestricted_real_world_actuation": false
}
```

`proves_release_readiness_for_scope` is `true` only when `ready == true`. **This
release does not enable real-world actuation.**

## approval vs readiness

- `approval_manifest` — a **named operator** reviewed and accepted the evidence.
- `release_readiness_report` — the **system** verified the bundle + approval +
  required evidence and produced a go/no-go.

## Pre-v1.0 workflow

```
sim → safety-gate → safety-regression → package-sim-run → verify-sim-bundle
    → approval-request → approve-bundle → verify-approval → release-readiness
```

A `ready=true` report is the entry ticket to **[guarded real mode](guarded-real-mode.md)**
(v1.0.0): `real --mode guarded` requires this readiness report plus a valid
approval, an enforced supervisor, a bounded session, and explicit operator
attestations. No verified readiness report → no real-mode egress.
