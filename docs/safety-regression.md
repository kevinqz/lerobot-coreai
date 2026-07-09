# Safety Regression

v0.9.2 compares a **baseline** vs **candidate** safety summary to detect whether
the candidate introduced more unsafe behavior — the gate that stops a regressing
policy or profile from graduating.

> A candidate policy cannot graduate if it introduces new unsafe actions.

## Compare

```bash
lerobot-coreai safety-regression \
  --baseline-run-dir runs/pusht-baseline \
  --candidate-run-dir runs/pusht-candidate \
  --max-block-rate-increase 0.0 \
  --max-critical-findings-increase 0 \
  --fail-on-regression
```

Each side accepts a summary file (`--baseline` / `--candidate`), a run dir
(`--baseline-run-dir` / `--candidate-run-dir`), or a bundle
(`--baseline-bundle-dir` / `--candidate-bundle-dir`).

| Flag | Default | Description |
|------|---------|-------------|
| `--max-blocked-increase` | `0` | Max increase in blocked actions |
| `--max-block-rate-increase` | `0.0` | Max increase in block rate |
| `--max-critical-failures-increase` | `0` | Max increase in critical failures |
| `--max-critical-findings-increase` | `0` | Max increase in critical findings |
| `--max-would-block-increase` | `0` | Max increase in would-block actions |
| `--max-would-block-rate-increase` | `0.0` | Max increase in would-block rate |
| `--max-modified-increase` | (off) | Max increase in modified actions |
| `--max-modification-rate-increase` | (off) | Max increase in modification rate |
| `--require-same-profile` | off | Fail if baseline/candidate profiles differ |
| `--no-require-candidate-passed` | (required) | Do not require the candidate summary to pass |
| `--fail-on-regression` | off | Return rc 1 on regression |

Deltas are computed on **normalized rates** as well as counts, so runs of
different sizes compare fairly. A candidate with fewer supervised actions than
the baseline emits a warning. Malformed summaries fail closed.

## What a passed report means

A passed safety regression report **only** means the candidate did not exceed
the configured regression thresholds **on the compared artifacts**. It does not
prove physical robot safety or real-world safety. The positive claim
`proves_no_safety_regression_on_compared_artifacts` is `true` only when the
report passed, and is scoped strictly to those artifacts.

## Relation to profile-compare

- [`profile-compare`](safety-profiles.md) — profile A vs profile B on the **same** actions log.
- `safety-regression` — baseline run vs candidate run (different artifacts).

They complement each other: one isolates the effect of the profile, the other
tracks a policy/run over time.
