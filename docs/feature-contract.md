# FeatureContract v1 (v1.3.24)

> Every tensor must declare what it means, where it exists in the pipeline, and who
> owns its transformation.

`FeatureContract v1` turns the ad-hoc, fixture-specific shape/transport checks of
earlier versions into a **declarative, stage-bound, versioned** contract. It answers
not just "`observation.state` has shape `[7]`" but *at which stage* that shape holds,
what dtype is allowed, the component names + order, units, value range, whether the
values are normalized and **who owns** that normalization, how batching/temporal axes
appear, and whether absence / NaN / Inf are permitted.

## Model

- **Feature identity** — `feature_id = <role>:<canonical_key>@<stage>`, e.g.
  `observation:observation.images.front@coreai_runner_input.v1`. Stages come from the
  canonical `ObservationStage`/`ActionStage` vocabulary (v1.3.20+).
- **`FeatureSpec`** — role, modality, stage, requiredness, dtype, **symbolic** shape
  (only declared symbols `B/T/H/A/C/IH/IW/S`), axes, layout, value domain (finite +
  min/max + closed-interval), units, coordinate frame, component names+order, and a
  `NormalizationContract` (state ∈ none/raw/normalized/unknown + single owner).
- **`FeatureContract`** — observations + actions + context, bound to the
  `processor_stage_contract_sha256` and `runtime_support_profile_sha256`. Its
  `sha256()` hashes the semantics (not the claim flags).

## Fail-closed validation

`validate_contract_structure` rejects: duplicate ids, non-canonical stages,
undeclared shape symbols, axis/shape rank mismatch, `H` on a selected/environment
action, names-vs-concrete-dim mismatch, an unknown/owner-less normalization, and
**double normalization** (two owners for one key).

`validate_payload_against_feature_contract(payload, contract, stage, symbols)`
resolves the symbolic shape, then rejects: missing required features, unexpected
features (closed mode), ragged payloads, shape mismatches, non-finite values, and
out-of-range values. `feature_contract_verified` is promoted **only** when the
contract is structurally valid and every payload validated cleanly — never by version.

## Breaking-change policy

`diff_feature_contracts` classifies a candidate vs a baseline. **Breaking**: removed
required feature, dtype / axis / layout / units / normalization-owner / names / shape
change, or an incompatible range reduction. **Non-breaking**: a new optional feature,
an explicit range expansion, a new alias.

## CLI

```bash
lerobot-coreai feature-contract validate \
  --contract feature_contract.json --input observation.json \
  --stage coreai_runner_input.v1 --symbols B=4,A=7 --output report.json

lerobot-coreai feature-contract diff \
  --baseline baseline.json --candidate candidate.json --fail-on-breaking
```

## Phase 0 — canonical stage/contract groundwork (this release)

- `build_processor_stage_contract(expects, returns)` maps the **legacy** processor
  ownership strings (`raw_lerobot_observation` / `postprocessed_action`) onto a
  canonical `ProcessorStageContract v1` (`observation`/`action` `source`→`target`
  `transform`) + a hash. The legacy strings remain accepted **only in the reader**;
  new writers emit the enum forms.

## v1.3.24a — Certification Binding Closure

The stop-the-line closure of the RFC's Phase 0: the FeatureContract now **governs the
certified runtime evidence** instead of sitting beside it.

- **Hash-bound into the artifact root** — the plugin artifact manifest (already inside
  the artifact-root inventory) carries `runtime_backend`, the canonical
  `canonical_processor_stage_contract` (+ its sha), and `feature_contract_sha256`
  derived from the CoreAI manifest. `verify_artifact_semantics` **recomputes** both
  from the embedded manifest and fails if they don't match — so a tamper of the CoreAI
  manifest features/ownership breaks the artifact (both via the byte-integrity root
  and the semantic recompute).
- **Hash-bound into the rollout/matrix root** — each rollout case's readiness report
  records `feature_contract_sha256` + `processor_stage_contract_sha256` in its
  `contracts` block; the report is checksummed into the bundle root, which is bound
  into the matrix root. Tamper breaks the case and the matrix.
- **Backend-neutral now** — a `RuntimeProviderStage` canonical layer
  (`runtime_provider_input.v1`/`output.v1`) + `runtime_backend` +
  `canonical_source`/`canonical_target` annotations let MLX / PyTorch-reference
  providers plug in later without a destructive contract migration. The legacy
  `coreai_*` stages remain valid (they are the coreai `backend_stage` values).

## Not yet

- **Full `raw_lerobot_observation` string removal** across every consumer +
  companion BatchContract schema dedup — the ownership strings are still accepted in
  the READER (diagnostic), while writers emit the canonical backend-neutral form; the
  final removal is isolated so it can land as its own green change.
- **Processor parity** — v1.3.24 may *declare* cross-stage transforms; equivalence is
  not proven until v1.3.26.
- `dataset_metadata_verified` (v1.3.25), `processor_parity_verified` (v1.3.26),
  `official_eval_certified` (v1.3.27), `authenticity_verified` (v1.3.28),
  `apple_runtime_certified` (v1.4.0) — each promoted only by its own proof.
