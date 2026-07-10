# Policy Card Generator (v1.2.3)

> A policy card is the human-readable layer over the evidence graph. It is
> **generated deterministically from verified artifacts**, never written by hand.
> Sources are verified before use and scanned for overclaims — either failure
> aborts generation. Every card carries the mandatory non-claims. Proves the card
> was generated from verified evidence — never physical safety, task success,
> training, native registry, or actuation.

## From an artifact-index entry

```bash
lerobot-coreai policy-card \
  --artifact-index .lerobot-coreai/index \
  --artifact-id evo1-pusht-bridge-benchmark@2026-07-10T03-20-00Z \
  --output README.md \
  --output-report policy_card_report.json
```

The indexed artifact must still verify; otherwise generation aborts.

## From direct paths

```bash
lerobot-coreai policy-card \
  --benchmark-bundle publish/evo1-pusht-bridge-benchmark \
  --provenance provenance.json --signature signature.json \
  --release-check release_check_report.json \
  --output README.md --output-report policy_card_report.json
```

Or pass individual reports (`--compat-report`, `--bridge-report`,
`--registry-report`, `--eval-v2-report`, `--obs-bridge-report`).

## Sections

What this policy is · Runtime & ecosystem · How to run (safely) · Compatibility
evidence · Bridge evidence · Registry evidence · Feature-mapping summary ·
Eval-v2 summary · Observation-pipeline summary · Benchmark-pack summary ·
Provenance/signature/release status · Known limitations · **Non-claims**.

Sections render only when their evidence is present; the card is **deterministic**
(no timestamps/randomness), so the same evidence always yields the same card.

## Mandatory non-claims (always present)

- This does not prove physical safety.
- This does not prove real-world task success.
- This does not authorize unrestricted robot actuation.
- This is not upstream-native LeRobot registry integration.
- This does not support training inside `lerobot-coreai`.

## Fail-closed

- A benchmark bundle that fails `verify-bridge-benchmark` aborts generation.
- An indexed artifact that fails `artifact-index verify` aborts generation.
- Any source report setting a forbidden claim (`proves_physical_safety`,
  `authorizes_robot_actuation`, `native_upstream_registry`, `supports_training`, …)
  true aborts generation.

`policy_card_report.json` records `source_verified`, `source_mode`,
`sections_written`, and honest claims (all safety/registry/training claims false).
