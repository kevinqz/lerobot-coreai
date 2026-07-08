# AGENTS.md — Operating Manual for lerobot-coreai

> Rules for coding agents working in this repository. Read this before making changes.

## What is lerobot-coreai?

`lerobot-coreai` is a **runtime backend adapter**, not a robotics framework. It connects
LeRobot-shaped policies and observations/actions to Apple CoreAI `.aimodel` artifacts executed
by `coreai-runner`. LeRobot remains the source of truth for all robot learning concepts.

**Canonical sentence:** *Same LeRobot workflow. CoreAI runtime.*

## Language rules

### Forbidden public framing (never use in code, docs, or user-facing strings)

- "CoreAI Robot framework"
- "new robotics stack"
- "robot-policy contract system"
- "embodiment harness"
- "safe host-loop platform"

### Required public framing

- "LeRobot CoreAI backend"
- "CoreAI policy runtime"
- `policy.type = coreai`, `runtime = coreai`
- `policy.path`, `robot.type`, `dataset.repo_id`
- `observation_features`, `action_features`
- "rollout mode"

## Architecture (spec §7)

```
LeRobot policy/checkpoint
    → lerobot-coreai export
    → coreai-fabric (convert/verify/publish/register)
    → Hugging Face CoreAI artifact (.aimodel + lerobot-coreai.json)
    → coreai-catalog (indexes artifact)
    → coreai-runner (executes .aimodel)
    → lerobot-coreai (maps LeRobot batch → CoreAI predict → LeRobot action)
    → LeRobot/LeLab rollout
```

## Ecosystem boundaries (spec §4, §8)

| Component | Owns | Does NOT own |
|-----------|------|-------------|
| LeRobot | recording, datasets, training, policy definitions, PyTorch inference | CoreAI |
| LeLab | GUI over LeRobot | CoreAI internals |
| **lerobot-coreai** | CoreAI runtime backend for LeRobot policies | training, datasets, robot configs |
| coreai-fabric | export / convert / verify / publish / register | runtime, rollout |
| coreai-catalog | registry, metadata, provenance, compatibility | execution |
| coreai-runner | local execution of .aimodel | serial ports, motors, safety gates |
| coreai-server | remote/network transport for runner | robots, teleop, safety |

## Coding rules

1. **LeRobot-first language.** User-facing API is `policy.select_action(batch) -> {"action": ...}`.
   No CoreAI graph names leak into the default Python API.
2. **Inference-only.** `CoreAIPolicy.train()` must raise `NotImplementedError`.
3. **Safety through modes.** `dry_run`, `shadow`, `sim`, `real`. Real mode requires `--confirm-real-robot-actuation`. No flag, no actuation.
4. **Version pinning.** 0.1.x supports LeRobot 0.6.x. Warn on unsupported versions; block rollout unless `--allow-unsupported-lerobot`.
5. **No private LeRobot submodule paths.** Use canonical public imports only (spec §27).
6. **Hardcode nothing.** Action dims, robot joint order, camera keys — all derived from checkpoint/features/manifest, never guessed.
7. **Progressive complexity.** `inspect` and `doctor` always work without a runner. `eval`/`rollout` need the runner. `export` needs fabric.

## Testing

```bash
# Run tests (no LeRobot or runner required for MVP v0.1 tests)
pip install -e ".[test]"
pytest
```

## Current version: 0.1.0 (MVP v0.1 — spec §28)

Scope: metadata + inspect only. No actuation, no runner integration, no export.

- `CoreAIPolicy.from_pretrained(repo_id)` — downloads `lerobot-coreai.json`, validates, loads metadata
- `lerobot-coreai inspect` — pretty-print the manifest
- `lerobot-coreai doctor` — check LeRobot version + artifact presence + manifest validity
- Catalog lookup (query `dist/lerobot-coreai.json`)
