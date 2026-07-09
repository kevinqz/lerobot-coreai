# Operator Approval Protocol

v0.9.3 requires explicit human approval of a simulator-only evidence bundle
before it can be marked release-ready.

> Approve the evidence, not just the code.

## What approval means

Approval means a **named operator reviewed a specific software evidence bundle**
and accepted it for the declared scope. It does **not** mean:

- physical robot safety is proven
- real-world task success is proven
- unrestricted robot actuation is authorized
- future policies are safe
- hardware safety systems can be skipped

## Workflow

```bash
# 1. Build the checklist (no approval yet).
lerobot-coreai approval-request \
  --bundle-dir publish/evo1-pusht-bundle \
  --output-dir approvals/evo1-pusht

# 2. Approve — requires explicit attestation flags.
lerobot-coreai approve-bundle \
  --bundle-dir publish/evo1-pusht-bundle \
  --operator "Kevin Saltarelli" \
  --approval-scope sim_to_guarded_real_readiness \
  --expires-days 30 \
  --output-dir approvals/evo1-pusht \
  --i-understand-this-does-not-prove-physical-safety \
  --i-understand-this-does-not-authorize-unrestricted-real-world-actuation

# 3. Verify the approval against the bundle.
lerobot-coreai verify-approval \
  --bundle-dir publish/evo1-pusht-bundle \
  --approval approvals/evo1-pusht/approval_manifest.json
```

## Required checks

`approve-bundle` refuses (rc 1) unless all required checks pass:

- bundle verifies (`verify-sim-bundle`)
- `sim_report.json` present, `mode == sim`, no robot egress, `actions_sent_to_robot == 0`
- `safety_summary.json` present
- `safety_quality_report.json` present **and** `passed`
- `safety_regression_report.json` present **and** `passed` (unless `--allow-missing-regression`)
- a safety profile / calibration artifact present (unless `--allow-missing-calibration`)
- no artifact overclaims physical / real-world safety

It also requires **both** attestation flags and an `--operator`.

## Approval manifest

`approve-bundle` writes `approval_manifest.json`, bound to artifact SHA256
hashes so tampering is detectable, with an expiry (default 30 days), the
operator, the scope, an attestation, and honest claims:

```json
{
  "proves_operator_reviewed_evidence": true,
  "proves_physical_safety": false,
  "proves_real_world_safety": false,
  "authorizes_unrestricted_real_world_actuation": false
}
```

The approval **references** the bundle; it does not mutate it (so the bundle's
own checksums stay valid). `verify-approval` re-checks the schema, expiry, every
bound hash, the attestation, and that the bundle still verifies.

## Scopes

`sim_only`, `sim_to_guarded_real_readiness` (default), `guarded_real_dry_run`,
`guarded_real_single_session`. v0.9.3 only **prepares** approval artifacts — it
does not enable real-world actuation. The strongest scope here is preparation
for a future guarded real mode.

See [release-readiness](release-readiness.md) for the final go/no-go report.
