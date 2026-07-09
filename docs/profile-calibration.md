# Profile Calibration

v0.9.1 fits a [safety profile](safety-profiles.md) to actions observed in
sim/shadow logs. Calibration proves **whether a profile fits the actions it was
calibrated on** — nothing more. It does not prove future action safety or
physical robot safety.

## Calibrate

```bash
lerobot-coreai profile-calibrate \
  --actions runs/pusht/actions.jsonl \
  --base-profile-name so100-sim-default \
  --output-profile profiles/so100-pusht-calibrated.json \
  --output-dir runs/pusht-profile-calibration \
  --quantile 0.995 \
  --margin 0.10 \
  --conservative
```

| Flag | Default | Description |
|------|---------|-------------|
| `--actions` | required | Actions JSONL to calibrate from |
| `--base-profile` / `--base-profile-name` | — | Profile to start from (bounds, robot type, shape) |
| `--output-profile` | — | Where to write the calibrated profile JSON |
| `--output-dir` | — | Where to write the calibration report |
| `--quantile` | `0.995` | Quantile of observed values used as the bound (supported: `0.50`, `0.95`, `0.99`, `0.995`) |
| `--margin` | `0.10` | Fractional margin added to the quantile |
| `--min-samples` | `10` | Fail if fewer valid actions than this |
| `--conservative` | off | Never exceed the base profile's bounds |

## How bounds are computed

For each of `max_abs_action`, `max_delta`, `max_l2_norm`:

```
recommended = max(p995(observed) * (1 + margin), floor)
```

with floors `0.05` / `0.01` / `0.05` respectively. In `--conservative` mode the
recommended bound is clamped to `min(recommended, base_bound)`, so calibration
can only tighten, never loosen. When a recommended bound would exceed the base
profile, a warning is recorded in the report.

## Reports

`profile_calibration_report.json` records the action statistics (abs/delta/L2
quantiles, dominant shape, NaN/Inf counts), the recommended bounds, the method,
and honest claims:

```json
{
  "proves_profile_fit_to_observed_actions": true,
  "proves_future_action_safety": false,
  "proves_physical_safety": false,
  "proves_real_world_safety": false
}
```

The generated profile always validates against `safety-profile.schema.json` and
carries `calibrated_from` / `calibration_method` metadata plus limitations.

## Fit check

`supervisor-check --output-dir <dir>` also writes `profile_fit.json` / `.md`
with allowed/blocked/modified/would-block rates — a quick read on whether a
profile is too strict or too loose for a given actions log.
