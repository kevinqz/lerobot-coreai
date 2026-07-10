# Bridge Benchmark Packs (v1.1.7)

> Bundle the software bridge reports — compatibility, bridge, registry, eval-v2
> feature mapping, observation bridge — into one reproducibility pack with
> per-file SHA256 checksums, an auto-generated README, and a verifier that
> detects tampering. **Fail-closed on overclaim**: a report claiming physical
> safety, task success, or actuation authorization is refused at packaging time
> and flagged at verify time. Software artifacts only.

## Package

```bash
lerobot-coreai package-bridge-benchmark \
  --compat-report reports/compat/lerobot_compatibility_report.json \
  --bridge-report reports/bridge/lerobot_bridge_report.json \
  --registry-report reports/registry/lerobot_registry_report.json \
  --obs-bridge-report reports/obs-bridge/obs_bridge_report.json \
  --eval-v2-dir reports/eval-v2 \
  --output-dir publish/evo1-pusht-bridge-benchmark
```

`--eval-v2-dir` is scanned for `lerobot_eval_v2_report.json` and
`lerobot_feature_mapping.json`. Any subset of reports may be provided (at least
one). `policy_path`/`dataset_repo_id` are inferred from the reports if not given.

### Bundle layout

```
benchmark_manifest.json
checksums.json
README.md
reports/
  lerobot_compatibility_report.json
  lerobot_bridge_report.json
  lerobot_registry_report.json
  feature_mapping.json
  eval_v2_report.json
  obs_bridge_report.json
```

## Verify

```bash
lerobot-coreai verify-bridge-benchmark --bundle-dir publish/evo1-pusht-bridge-benchmark
```

Checks: manifest + checksums present, manifest schema-valid, every listed report
present, **checksums match** (tamper detection over every file including the
manifest), no bundled report overclaims, and the manifest's own claims are honest.

## Honesty guarantees

The manifest pins `proves_task_success=false`, `proves_physical_safety=false`,
and `authorizes_robot_actuation=false` (schema `const false`). Packaging scans
every included report and refuses any that sets a banned claim
(`proves_physical_safety`, `authorizes_robot_actuation`, `native_registry`, …) to
true. A pack proves **reproducibility of the bundled software reports** — nothing
about task success or physical safety.
