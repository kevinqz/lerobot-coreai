# Compare v2 — Source Loader v2 + Processor-Inclusive Compare (v1.2.6, hardened v1.2.7)

> **Experimental — not release evidence yet.** compare-v2 loads the source policy
> the way LeRobot does and compares the **final** action after each side's
> processing. v1.2.7 hardened the evidence integrity so a result can only be
> called "parity" when explicit numeric tolerances are met. Software-only; no
> robot/sim/real egress; no task-success/physical-safety claim.

## CLI

```bash
lerobot-coreai compare-v2 \
  --torch.policy.path lerobot/diffusion_pusht \
  --coreai.policy.path kevinqz/EVO1-Pusht-CoreAI \
  --dataset.repo_id lerobot/pusht \
  --runner.url http://127.0.0.1:8710 \
  --compare-target next_action \
  --policy.revision <sha-or-tag> --dataset.revision <sha-or-tag> \
  --max-frames 32 --min-frames 32 \
  --tolerance.mean-mae 1e-5 --tolerance.max-abs-error 1e-4 \
  --tolerance.min-cosine 0.999 --tolerance.max-relative-mae 0.01 \
  --strict-processors \
  --output-dir reports/compare-v2
```

## What v1.2.7 fixed (evidence integrity)

- **Numeric gates decide parity.** `proves_action_parity_on_final_unit` is true
  **only** when tolerance gates are configured *and* pass. Without tolerances a
  perfect-looking match is reported but parity is **not** claimed; a large finite
  error fails the gate (previously it could read as "parity").
- **Structural shape.** Actions must match nested shape, not just flattened
  length — `[[1,2],[3,4]]` no longer "matches" `[1,2,3,4]`.
- **Explicit compare target.** `--compare-target next_action|action_chunk` —
  a per-timestep action is never compared against a full chunk. `next_action`
  uses source `select_action` vs CoreAI `select_next_action`; `action_chunk` uses
  `predict_action_chunk` on both sides.
- **Source weights bound.** The loader sets `cfg.pretrained_path` (+ revision)
  before `make_policy`, so the trained checkpoint is loaded — not a randomly
  initialized policy. The report proves `weights.pretrained_path_bound`.

## Processor contract

`--strict-processors` fails closed when the CoreAI manifest doesn't declare
observation/action ownership (`expects` ∈ raw/preprocessed, `returns` ∈
postprocessed/normalized).

## Still deferred (do not treat compare-v2 as strong release evidence yet)

The following are tracked for v1.2.8+ and are **not** implemented yet:

- **Manifest v1 contracts** — the base manifest schema does not yet carry
  `processor_contract`/`action_contract`/`batch_contract`; today they're passed
  as dicts / inferred. A published, schema-valid manifest carrying them is v1.2.8.
- **Real processor execution** — the CoreAI side does not yet execute the
  declared preprocessing/normalization path; `normalized_action` mode is declared
  but not applied. Until then, prefer artifacts whose runner returns the final
  postprocessed action.
- **Temporal alignment** — delta timestamps, episode boundaries, and per-episode
  reset are not yet wired into the frame iteration.
- **Live unmocked fixtures** — the stable CI job exercises the official loader
  imports and the mocked flow; a tiny-checkpoint + tiny-dataset live compare is
  v1.2.8.

Because of these, `release-check`/approval should **not** rely on a compare-v2
report as strong parity evidence yet.
