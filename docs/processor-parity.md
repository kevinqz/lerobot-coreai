# Nontrivial Processor Parity (v1.3.26)

> Prove that the reference processor pipeline and the CoreAI execution boundary
> produce semantically equivalent tensors.

Empty pipelines prove wiring, not equivalence. v1.3.26 runs **non-trivial** transforms
and compares each stage between an independent **reference** and a **candidate**.

## Transform contract

`processor_transform_contract` declares an ordered op list (`permute` / `cast` /
`scale` / `normalize` / `denormalize`) between two canonical stages, bound to
input/output FeatureContract hashes. An **independent reference implementation**
applies the declared ops on nested Python lists (no numpy/torch) — so the reference
and the candidate runtime path have genuinely separate code, the redteam mitigation
against a shared buggy implementation (RFC §18.3).

## Parity levels

- **Exact structural parity** (rename / reshape / permute / batching / stacking):
  canonical-hash equality of reference vs candidate.
- **Numeric parity** (cast / scale / normalize / denormalize): `max_abs_error`,
  `mean_abs_error`, `relative_mae`, `cosine_similarity`, non-finite counts. Thresholds
  are **carried explicitly in the case** — a numeric case with no thresholds **fails**
  (no silent permissive tolerance). NaN/Inf and shape mismatch always fail.

## ProcessorParityReport v1

`build_processor_parity_report` evaluates each `ParityCase` and promotes
`processor_parity_verified` only when every case passes; `model_output_parity_verified`
stays false (model/policy output parity is a later, separate proof).

### Independent replay (v1.3.26.10, P1.1)

The report carries an `evidence_grade`. In **certificate** grade (the default) each case
persists its **raw reference/candidate arrays**, and `verify_processor_parity_report`
**re-derives** every case's metrics, array hashes, and pass/fail from those arrays — an
independent replay, not merely an internal-consistency check. A report that is
self-consistent (its `passed` flag agrees with its `reasons`) but whose recorded
**metrics** were forged is now caught, because a fresh recomputation from the raw arrays
won't match. A tampered raw value breaks the array hash. **diagnostic** grade omits the
arrays and enforces consistency only.

## Save/reload

The transform contract IS the serializable processor spec: serialize → reload →
re-apply must reproduce the output (config-hash parity + output parity), so parity is
never certified on in-memory objects only.

## CLI

```bash
lerobot-coreai processor-parity run    --spec cases.json --output report.json
lerobot-coreai processor-parity verify --report report.json
```

## Not yet

- **Reference = the real LeRobot `NormalizerProcessor` loaded from serialized config**
  (vs the independent reference used here) + full processor artifact save/reload
  against lerobot's own serialization — the parity framework, metrics, gates, report
  and an independent reference are in; wiring lerobot's actual processor objects as
  the reference (and driving the candidate through the plugin transport in a
  lerobot CI job) is the integration step.
- `model_output_parity_verified`, official-eval CLI (v1.3.27), signed evidence
  (v1.3.28), Apple runtime (v1.4.0) — each promoted only by its own proof.
