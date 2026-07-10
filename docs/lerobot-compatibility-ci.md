# LeRobot Compatibility CI + Certificate (v1.1.2)

> Turns the "compatible with LeRobot 0.6.x shape" claim into a **tested artifact**,
> not just documentation. A dedicated CI job installs the `[lerobot]` extra on
> Python 3.12 and produces a compatibility certificate.

## Certificate CLI

```bash
lerobot-coreai lerobot-compat-check --output-dir reports/lerobot-compat --json
```

Writes `lerobot_compatibility_report.json` / `.md`. Checks:

| Check | Meaning |
|-------|---------|
| `python_version_compatible` | Python ≥ 3.10 (base package) |
| `python_supports_lerobot` | Python ≥ 3.12 (info — LeRobot's floor) |
| `lerobot_installed` | `[lerobot]` extra present (required under `--strict`) |
| `lerobot_version_in_range` | `>=0.6.0,<0.7.0` |
| `pretrained_policy_importable` | LeRobot's `PreTrainedPolicy` import path exists |
| `lerobot_dataset_importable` | `LeRobotDataset` import path exists |
| `coreai_bridge_importable` | The local bridge imports (no torch/lerobot needed) |
| `native_registry_claim_false` / `training_claim_false` / `physical_safety_claim_false` | Honest-claim invariants |

`--strict` makes the LeRobot-dependent checks **required** (used by the compat CI
job); without it they are informational so the certificate is still produced on a
base install.

## CI

The `lerobot-compat` job (Python 3.12, `pip install -e ".[lerobot,test]"`) runs
`lerobot-compat-check --strict` and the compat/bridge tests, then uploads the
certificate as a build artifact. The main `test` matrix (3.10/3.11/3.12) keeps
running on the base `[test]` install and must pass **without** LeRobot — the base
package never imports `torch` or `lerobot`.

## Honest boundary

Tested compatibility is with the LeRobot 0.6.x **shape** only. `policy_type="coreai"`
is not registered upstream, training remains LeRobot's job, and nothing here
proves physical safety.
