# Official Out-of-Tree LeRobot Plugin (v1.3.0)

> `packages/lerobot_policy_coreai_bridge` is a **companion package** that makes
> CoreAI policies discoverable through LeRobot's **official** out-of-tree plugin
> mechanism — a real `PreTrainedPolicy` (hence `torch.nn.Module`) registered via
> `PreTrainedConfig.register_subclass`, **no monkeypatch**. Runtime-only: it does
> not train, and it proves nothing about physical safety. This is where
> `action_semantics` / `official_*` compatibility levels finally move off
> `failed` — for the plugin, not the base package's local bridge.

## Why a separate package

LeRobot imports installed distributions named `lerobot_policy_*` so they
self-register. The base `lerobot-coreai` package is intentionally torch/lerobot-
free; the plugin lives in its own distribution that depends on both.

```bash
pip install lerobot-coreai lerobot            # base + LeRobot
pip install ./packages/lerobot_policy_coreai_bridge   # the plugin
```

```python
import lerobot_policy_coreai_bridge            # registers "coreai_bridge"
from lerobot.configs.policies import PreTrainedConfig
assert "coreai_bridge" in PreTrainedConfig.get_known_choices()
```

## What it provides

- **`CoreAIBridgeConfig`** — `@PreTrainedConfig.register_subclass("coreai_bridge")`
  dataclass. Not `"coreai"` — that name is not registered upstream. `device` stays
  a real torch device (so `make_policy`'s `policy.to(cfg.device)` works);
  `runtime_device="coreai"` records that inference runs on CoreAI.
- **`CoreAIBridgePolicy(PreTrainedPolicy)`** — a genuine `nn.Module`.
  `select_action(batch) -> torch.Tensor(B, action_dim)` (per-timestep, filled
  from a chunk queue); `predict_action_chunk -> torch.Tensor`; `reset()` clears
  the queue and resets the CoreAI policy.
- **`make_coreai_bridge_pre_post_processors(config, dataset_stats=None)`** —
  official-convention processor factory (identity by default; the CoreAI runner
  owns normalization via the manifest processor contract).

## Runtime-only boundary

`forward()` and `get_optim_params()` raise; `train(True)` raises; `eval()` /
`train(False)` work (so LeRobot eval's `policy.eval()` is fine). Train with
LeRobot; run with CoreAI.

## Deprecation

The base package's `local_lerobot_registry_patch()` (v1.1.3) is now **deprecated**
in favor of this official plugin — it emits a `DeprecationWarning`.

## CI

The stable CI job (Python 3.12, LeRobot 0.6.0) installs this package and runs its
tests: config registered under `coreai_bridge`, policy is a `PreTrainedPolicy` +
`nn.Module`, `select_action` returns `Tensor(B, A)`, `train(True)` raises,
`forward`/`get_optim_params` raise.

## Factory / runtime binding (v1.3.1)

The plugin is not just registered — it is **factory-loadable**:

- `CoreAIBridgePolicy.__init__` accepts the kwargs `make_policy` passes
  (`dataset_stats`, `dataset_meta`, `**kwargs`).
- `CoreAIBridgePolicy.from_pretrained` **binds a CoreAI runtime**: it loads the
  config, resolves the runner URL from `config.runner_url_env` (fail-closed if
  unset; the URL is read from the environment, never persisted), loads a
  `lerobot_coreai.CoreAIPolicy`, and **cross-binds** the CoreAI manifest against
  the config's `expected_action_dim` / `expected_action_horizon` /
  `expected_robot_type` — failing closed on mismatch. It never returns a policy
  with no runtime bound.
- `select_action` returns a tensor on the policy's device;
  `predict_action_chunk` returns `(B, H, A)`.
- `batch_size > 1` **fails clearly** (`batch_mode="single_only"` in v1.3.1);
  batched evaluation lands in v1.3.2.

Verified against real LeRobot 0.6.0 (no monkeypatch):
`PreTrainedConfig.get_choice_class("coreai_bridge")`, `make_policy_config`,
`get_policy_class`, and a bound `from_pretrained` all resolve.

## Compatibility profiles

`lerobot-compat-check --contract` describes the **base package's local bridge**
(its `official_*` levels stay `failed` on purpose). `lerobot-compat-check
--plugin` describes the **companion plugin** — `plugin_discovery` /
`config_registry` / `policy_class_contract` pass when installed;
`policy_factory` / `processor_pipeline` are `partial`; `official_eval` stays
`not_tested` and `official_eval_certified` stays `false` until a live end-to-end
eval proves it.

## Data plane (v1.3.2)

The plugin now bridges a real LeRobot batch to the CoreAI observation contract:
`transport.prepare_single_coreai_observation` strips the leading batch dim,
unwraps `task: list[str]` → `task: str`, keeps **only** manifest-declared
observation features (never the ground-truth `action`/`reward`), converts
`torch`/`numpy` to JSON-safe values, and hashes the exact payload. `select_action`
now **always** routes through `predict_action_chunk` (one boundary → `(B,H,A)` →
per-timestep `(B,A)`). `batch_size > 1` fails closed (batched transport is
v1.3.3). Cross-binding is fully fail-closed: a declared expectation with an
unknown manifest value is a failure.

Real discovery is proven in a clean subprocess (`test_discovery.py`):
`register_third_party_plugins()` finds `coreai_bridge` without a manual import.
The `lerobot-dev` CI job is pinned to an exact 0.6.1-dev commit.

## v1.3.3 — local artifacts, negotiated codec, validated outputs

- **Local manifests** — `load_manifest`/`resolve_manifest` accept an HF repo id,
  a local directory, or a local `lerobot-coreai.json` file. Local sources never
  touch the network, and a local-looking path that doesn't exist **fails**
  instead of falling back to the Hub (records `source_kind` + `sha256`).
- **Observation codec** — the transport emits `nested_json_v1` (plain JSON lists)
  by default; `typed_array_envelope_v1` (`{"__array__",dtype,shape}`) is only used
  when explicitly selected. Shape is validated against the manifest **before**
  encoding, so the envelope can never mask a mismatch; the shape audit + payload
  `sha256` are recorded.
- **Strict action outputs** — `normalize_and_validate_action_chunk` rejects
  ragged / rank-4 / non-finite / wrong-dim/horizon Runner outputs;
  `predict_action_chunk` and `select_action` both route through it.
- **Fully fail-closed cross-binding** — `expected_robot_type` (like action_dim)
  now fails when the manifest declares no robot type.

## v1.3.4 — real codec negotiation + action-contract integration

- **Negotiated observation encoding** — `RunnerCapabilities` now carries
  `protocol_version` / `observation_encodings` / `supports_batch` /
  `max_batch_size`, and `negotiate_observation_encoding()` selects the encoding
  as the intersection of the config request, the plugin's supported encodings,
  and what the runner **announces**. `auto` picks the first common encoding; an
  unsupported request or empty intersection **fails closed** (legacy fallback is
  opt-in). The selected encoding + protocol version + observation `sha256` are
  sent in `request.options` (threaded through `CoreAIPolicy.predict_action`).
- **Action-contract integration** — the plugin persists the manifest action
  contract on bind and passes its `representation` / `horizon` / `action_dim`
  into the strict validator, so a single-action artifact returns `[A]` correctly
  and a wrong-horizon runner output now **fails** (v1.3.3's validator was always
  called in `chunk` mode with no horizon).

## v1.3.5 — strict protocol binding + hermetic B=1 E2E (no mocks)

- **`NegotiatedRunnerProtocol`** — negotiation now returns a structured result
  (`protocol_version` / `observation_encoding` / `supports_batch` /
  `max_batch_size` / `legacy`). The runner's announced `protocol_version` is
  validated against `config.minimum_runner_protocol` (default `coreai-runner.v2`):
  an absent (without opt-in), unknown, or lower protocol **fails closed**. The
  wire request carries the *negotiated* version, never a hardcoded constant.
- **No silent legacy** — a capabilities/transport/protocol failure now
  **propagates**; the old catch-all that turned any failure into
  `nested_json_v1` is gone. Legacy is opt-in via
  `allow_legacy_runner_protocol=True`. Config gains `require_protocol_negotiation`,
  `allow_legacy_runner_protocol`, `minimum_runner_protocol`.
- **No `TypeError` fallback** — the second `predict_action_chunk` call (which
  could double-advance a stateful runner) is removed; the companion requires
  `lerobot-coreai>=1.3.5`, whose signature accepts `runner_options`.
- **Action contract fails closed** — a malformed contract raises
  `PluginBindingError` instead of degrading to chunk/no-horizon; the base parser
  enforces `representation=single ⇒ horizon=1` and `chunk ⇒ horizon≥1`.
- **`reset()` re-negotiates** — the cached protocol is invalidated so a restarted
  runner with different capabilities is re-negotiated on the next inference.
- **Hermetic B=1 E2E, no mocks** (`tests/test_e2e_http_no_mocks.py`) — a real
  `ThreadingHTTPServer` implements `/v1/health`, `/v1/capabilities`,
  `/v1/predict`; a real `CoreAIBridgePolicy.from_pretrained` loads a local
  canonical artifact, opens a real `RunnerClient`, negotiates over HTTP, POSTs a
  real observation, and the strict validator yields `Tensor[1, A]` — nothing is
  patched. Asserts the wire payload (batch dim stripped, `task` list→str,
  no `action`/`index`/`reward`/`timestamp` leakage), negotiated
  protocol/encoding/hash in `options`, no secret persisted, and fail-closed
  failure paths (lower/unknown/missing protocol, no common encoding, wrong
  horizon). Runs on both LeRobot 0.6.0 (blocking) and 0.6.1-dev.

## v1.3.6 — canonical artifact + serializable processors + full factory E2E

- **Structured protocol identity** — `ProtocolIdentifier(family, major)` replaces
  the suffix-only `.vN` compare: the family must match (`coreai-runner.v3` ≠
  `malicious-runner.v3`), a lower major fails, and a **higher** major is accepted
  only when capabilities declare `backward_compatible_with` the minimum. A newer
  major without that declaration fails closed (it may be breaking).
- **Explicit binding modes** — `runtime_binding_mode` (`strict` | `legacy` |
  `in_memory`) replaces the ambiguous boolean pair. `strict`/`legacy` **require a
  bound runner** (no runner → error); `in_memory` is the only mode that skips the
  wire, for local/in-process binding. `require_protocol_negotiation` is now
  expressed by the mode.
- **Real serializable processors** — `_IdentityProcessor` is gone;
  `make_coreai_bridge_pre_post_processors` returns real
  `lerobot.processor.PolicyProcessorPipeline` instances (step-empty, with the
  official transition converters) that the factory loads from
  `policy_preprocessor.json` / `policy_postprocessor.json`. Identity is allowed
  only when the manifest declares `contracts.processor` CoreAI ownership;
  packaging/binding fails closed otherwise.
- **Canonical artifact** — `lerobot-coreai package-lerobot-plugin-artifact`
  builds `config.json` / `policy_preprocessor.json` / `policy_postprocessor.json` /
  `lerobot-coreai.json` / `plugin_artifact_manifest.json` / `checksums.json` /
  `README.md`. `config.json` carries `coreai_artifact=""` (the root) and only the
  runner env-var **name** — never a URL, token, or local path.
  `verify-lerobot-plugin-artifact` checks schemas, checksums (tamper), versions,
  processor reload, ownership, protocol floor, external-ref pinning
  (revision+sha256 required), forbidden claims, secrets, and symlink escape.
- **Feature cross-binding** — after `make_policy` fills
  `cfg.input_features`/`output_features`, `from_pretrained` validates them against
  the manifest: every input feature must be a declared observation feature (with
  matching per-frame shape), and the ACTION output feature's last dim must equal
  the manifest action dim (horizon lives in the action contract, not the
  per-timestep feature). `validate_features()` is implemented; contradictory
  `action_horizon`/`expected_action_horizon` fail.
- **Full official-factory E2E, no mocks** (`tests/test_factory_e2e.py`) —
  `register_third_party_plugins → PreTrainedConfig.from_pretrained(artifact) →
  make_policy(cfg, ds_meta=…) → make_pre_post_processors(cfg, pretrained_path=…) →
  post(policy.select_action(pre(batch))) → Tensor[1, A]` against a real
  `ThreadingHTTPServer`. Nothing patched (not `CoreAIPolicy`, `RunnerClient`,
  `make_policy`, or the processors). Asserts one predict request, no label
  leakage on the wire (the `pre` pipeline injects `action`/`next.*` placeholders
  that the transport allowlist drops), negotiated protocol/encoding/hash in
  `options`, and fail-closed feature mismatch. Runs on 0.6.0 (blocking) + 0.6.1-dev.

## v1.3.7 — artifact integrity + processor-contract hardening

Before batching multiplies the artifact, the object being replicated/certified is
made structurally, semantically, and (partly) cryptographically trustworthy:

- **Typed inventory + exact checksums** — `plugin_artifact_inventory.json` lists
  every content file (`path`/`role`/`sha256`/`size_bytes`) plus an
  `artifact_root_sha256`. Verification requires `checksums.json` to **exactly**
  cover the inventory (missing/extra/malformed entries fail), every declared file
  to match digest+size, no **undeclared** file in the dir, and no symlink.
- **Path-traversal guard** — inventory paths must be simple basenames; `..`,
  absolute paths, and symlinks are rejected **before** any file is opened.
- **Integrity ≠ authenticity** — the verifier reports `integrity_verified`
  (unsigned checksum consistency) separately from `authenticity_verified`, which
  stays **false** until a trusted signature is issued (signing lands in v1.3.10).
- **Strict JSON schemas** (`artifact_schemas.py`, `additionalProperties:false`)
  for the plugin manifest, inventory, processor contract, and verification report.
- **Processor contract v2** — step-empty (identity) processors are permitted
  **only** when the manifest declares `owner=coreai_runner` on both ends AND
  `observation_input.expects=="raw_lerobot_observation"` AND
  `action_output.returns=="postprocessed_environment_action"`. Owner-correct but
  wrong `expects`/`returns` fails. The direct
  `make_coreai_bridge_pre_post_processors(config)` (no artifact evidence) now
  **fails closed** instead of silently synthesizing identity processors.
- **Version binding** — `check_version_compatibility` enforces artifact core/plugin
  **lockstep**, installed core/plugin not older than the artifact and same major,
  and (deep) installed LeRobot sharing the recorded major.minor.
- **Honest provenance** — `source_coreai_artifact_reference` (renamed) records the
  embedded manifest `sha256`; `external` mode requires an immutable `revision`, and
  a supplied `external_sha256` that disagrees with the embedded manifest fails.
- **Structured secret scan** — recursively refuses non-empty values under
  `token`/`secret`/`password`/`api_key`/`authorization`/`bearer` keys and
  credential-bearing URLs (`scheme://user:pass@…`); public repo URLs pass.
- **Bidirectional feature binding** — additionally, every **required** manifest
  observation feature must be present in `cfg.input_features`.
- **Evidence report** — `verify … --report` writes
  `plugin_artifact_verification_report.json/md` (schema-valid), separating
  integrity/authenticity/processor-contract claims; `factory_b1_certified` stays
  `false` (promoted only by the signed certificate, v1.3.10).

## v1.3.8 — artifact semantic closure + batch protocol foundation

Before batching multiplies the artifact, its **contracts** are cross-bound and its
**state semantics** are pinned (still B=1):

- **Schemas enforced** — the embedded processor contract is validated against
  `PROCESSOR_CONTRACT_SCHEMA` (previously defined but unused); the action contract
  has a strict schema (`representation` enum, `horizon≥1`, `single⇒horizon=1`); and
  `claims` is closed (`additionalProperties:false`, forbidden claims `const:false`,
  `official_plugin_factory_compatible` ∈ {null, false}).
- **Cross-file semantic closure** — `verify_artifact_semantics` reconciles
  `config.json` ↔ `plugin_artifact_manifest.json` ↔ `lerobot-coreai.json` ↔
  inventory ↔ processors: action contract/dim/horizon equality, robot type, runner
  minimum protocol, role→file mapping, and step-empty processor structure vs the
  identity contract. Unknowable properties (dtype/names/units/layout, which config
  `PolicyFeature`s don't carry) are reported **`not_verified`**, never silently
  `passed`.
- **Canonical inventory** — unique paths and roles, a closed role enum, exact
  role→filename mapping, and a root digest that binds `path`+`role`+`sha256`+
  `size_bytes` + `artifact_root_algorithm` (so a future signature authenticates
  role semantics, not just bytes).
- **Reports outside the artifact** — `verify … --output-dir` writes the report
  externally; the sealed artifact is never modified, so verification is
  **idempotent** (root digest unchanged across repeated runs).
- **Full secret scan** — every declared JSON file is scanned; a sensitive key with
  **any** non-empty value (including dict/list) fails, plus credential-bearing
  URLs; normal public repo URLs pass.
- **Immutable provenance** — `source_coreai_artifact_reference` records
  `requested_ref` + `resolved_commit_sha` (40-hex) + `embedded_manifest_sha256`;
  external release references require a resolved commit (a mutable `main` alone
  fails).
- **Batch protocol foundation (no B>1)** — `RunnerCapabilities` gains
  `action_batching.semantics` and `inference_state.{scope,supports_session_ids,
  reset_scope}`; a pure `select_batch_execution_mode(config, capabilities)` encodes
  the rules (native requires native support; split allowed for
  stateless/request-scoped, or session-scoped **with** session ids; **global
  forbids split**; unknown scope fails). No batched inference is implemented — this
  stabilizes the contract so v1.3.9 batching is mechanical and cannot mix sessions.

## v1.3.9 — stateless batched runtime + atomic temporal queue

Batching arrives, restricted to runners that can prove requests don't mix state
(`stateless` / `request_scoped`); `session_scoped` and `global` fail closed.

- **Decision engine** — `select_batch_execution_mode(config, artifact_batch_contract,
  capabilities, requested_batch_size)`: B=1 always single; for B>1 a **missing or
  unknown** `inference_state.scope` fails (never assume stateless), `global` and
  `session_scoped` fail (session batching deferred), native requires
  `state_isolation ∈ {stateless, request_scoped}`, and B is capped by the
  **effective max** = `min(artifact, config, runner)`. Capabilities gain
  `action_batching.state_isolation`; a `BatchContract` v2 (`policy_supports_batch`,
  `supported_client_modes`, `queue_layout`, `requires_atomic_commit`) is parsed and
  strictly validated.
- **Strict batch-size boundary** — `infer_and_validate_batch_size` requires every
  manifest observation feature's leading dim AND the `task` length to be exactly
  equal (a ragged `state=4/task=2` batch fails).
- **Native** — `prepare_batched_coreai_observation` → `CoreAIPolicy.predict_action_batch`
  (**one** request) → `normalize_and_validate_batched_action_chunk` → `Tensor[B,H,A]`;
  wire carries `[B,…]` features, a `task` list of length B, `batch_size`, and per-sample
  hashes; no label leakage.
- **Split-and-stack** (stateless/request-scoped only) — B independent single
  requests, **every** sample validated before anything commits; on sample *i*
  failure the queue is untouched and the error names `sample index i` (atomic).
- **LeRobot-style temporal queue** — a chunk `[B,H,A]` is `transpose(0,1)`'d into a
  `deque[Tensor[B,A]]`; `select_action` returns `Tensor[B,A]`; the active batch size
  is tracked and a size change while the queue is non-empty fails; `reset()` clears
  the queue, active size, protocol, and caches. B=1 stays backward-compatible.
- **Evidence honesty** — the verifier now reports `semantic_consistency_verified`
  (nothing failed) separately from `semantic_completeness_verified` (everything
  passed); `not_verified` aspects keep completeness false without failing consistency.
- **E2E, no mocks** (`tests/test_e2e_batched.py`) — native B=2/B=4 (one request each)
  and split B=2/B=4 (B requests) through `make_policy`/`make_pre_post_processors`
  against a real batch-capable HTTP runner, on 0.6.0 + 0.6.1-dev.

## v1.3.10 — authoritative batch contracts + multimodal readiness

The batch contract becomes the authority, the runner's capabilities are strictly
typed, and batching is proven **multimodal** with processors actually executed.

- **`BatchContract` v3** — native and split modeled **apart** (`native_batch` /
  `client_split` each with their own `supported` + `max_batch_size`;
  `required_slot_isolation`, `allowed_state_scopes`, `queue.commit_semantics`,
  `observation_stage`). Only an **authoritative** v3 contract can gate B>1; legacy
  v0/v2 blocks stay readable for B=1 but cannot certify batching.
- **Contract is authoritative** — `select_batch_execution_mode` now enforces
  `native_supported` / `split_supported` / `fallback` (`reject` blocks auto→split)
  / queue layout / commit semantics, not just `max_batch_size`.
- **Separate mode limits** — native effective max = `min(artifact native, config,
  runner native)`; split effective max = `min(artifact split, config,
  max_split_requests)`. The runner's **native** max never caps split-and-stack.
- **Slot isolation ≠ state scope** — native B>1 requires
  `slot_isolation == 'independent'` (per-slot), distinct from
  `inference_state.scope` lifetime.
- **Strict capabilities** — the parser rejects a JSON string `"false"`
  (`bool("false")` is `True`!), an invalid `max_batch_size`, and unknown types at
  the earliest boundary; a canonical `capabilities_sha256` fingerprint is exposed.
- **Runtime observation validation** — required manifest features must be present
  in each request; a `task` must already **be** a string (never coerce
  `None`/`int`/`dict`).
- **Order-sensitive split hash** — the split path's batch hash is
  `canonical_batch_sha256(batch_size, ordered_sample_hashes, mode)`, not the first
  sample's hash.
- **Multimodal E2E, processors executed** (`tests/test_e2e_multimodal.py`) —
  `observation.state` + two cameras (`front`/`wrist`) + `task`, B=1/2/4, native and
  split, run as `post(policy.select_action(pre(batch)))` (the processors are
  **executed**, not just loaded) against a real batch-capable HTTP runner on
  0.6.0 + 0.6.1-dev.

## v1.3.11 — contract closure + official rollout readiness

- **Native slot-isolation is fail-closed** (fixes a v1.3.10 flaw) — native B>1 now
  requires `slot_isolation == 'independent'` on **both** the artifact contract and
  the runner; `shared`/`unknown` never enable native, even if both sides agree.
- **`BatchContract` v3 schema** — `BATCH_CONTRACT_V3_SCHEMA`
  (`additionalProperties:false`, `required_slot_isolation` pinned `independent`,
  closed scopes/layout/commit); the parser no longer coerces (`"false"`/`"4"` fail).
- **Batch contract in the artifact** — `plugin_artifact_manifest.json` now carries
  `batch_contract` + `batch_contract_sha256` + `processor_stage_contract`;
  `verify_artifact_semantics` cross-binds the plugin manifest's batch contract to
  the CoreAI manifest's (equality + hash) and the processor input stage.
- **Official rollout readiness, no mocks** (`tests/test_e2e_official_rollout.py`) —
  drives the **real** `lerobot.scripts.lerobot_eval.rollout` over a deterministic
  `gym.vector.SyncVectorEnv` (state + front/wrist cameras + `task_description`,
  staggered episode termination) through the official chain
  `preprocess_observation → env_preprocessor → policy_preprocessor →
  CoreAIBridgePolicy.select_action → policy_postprocessor → env_postprocessor`,
  native and split, B=1/2/4, on 0.6.0 + 0.6.1-dev. This proves the rollout
  **pipeline completes** (`Tensor[B, seq, A]`, done-masking) — NOT lerobot-eval
  certification, task success, or safety.

## v1.3.12 — mandatory rollout gate + evidence integrity

- **Rollout is now a mandatory CI gate** — a dedicated `lerobot-rollout-stable` job
  installs LeRobot's media stack (`datasets`/`av`) **isolated** from the compat
  jobs (which must not see the dataset stack) and runs the rollout E2E with
  `COREAI_REQUIRE_ROLLOUT=1`, so a missing dependency **fails** instead of skipping.
- **Rollout actually exercises the loop** — a common `_max_episode_steps=8` with
  **staggered** `terminate_at` (`[2,4,6,8]`) and `HORIZON=3` means the temporal
  queue genuinely drains and refills. The test asserts **exact** request counts
  (`native = ceil(seq/H) = 3`; `split = B×3`), **cumulative** done masks
  (`first_done[i] == terminate_at[i]-1`, every env reaches done), and the **wire
  payload** (batched obs, both cameras, `task` list/str, `batch_size`, and **no**
  `action`/`reward`/`done`/`success`/`index`/`timestamp` leakage).
- **Readiness evidence** — `rollout_evidence.build_rollout_readiness_report`
  emits a schema-valid report (`official_rollout_pipeline_smoke_passed` only when
  every check passes; `official_eval_certified` etc. stay false).
- **Task requiredness** — `extract_observation` no longer forwards an **undeclared**
  task; a declared-**required** task that is absent fails; declared-optional may be
  absent.
- **Capabilities enum + alias hardening** — `capabilities()` validates
  `semantics`/`slot_isolation`/`scope`/`reset_scope` enums and `observation_encodings`
  as `list[str]`, and **fails on conflicting** `slot_isolation`/`state_isolation`.
- **Schema convergence** — the public `action-contract` batch schema now pins
  native `required_slot_isolation` to `independent` (matching the runtime), with
  regressions for artifact `shared`/`unknown` and string `"false"`/`"4"`.

## v1.3.13 — real rollout evidence + dual-target gate

- **Measurements → evaluator → checks** (not caller booleans) — `RolloutMeasurements`
  captures the run (request bodies, done mask, terminate_at, required keys);
  `evaluate_rollout_measurements` **derives** every check (exact request count,
  cumulative done, `first_done == terminate_at-1`, queue refill, wire validity, no
  leakage). The report builder consumes a `RolloutEvaluation`, so a caller can no
  longer hand-write a passing report.
- **Real, required hashes** — the readiness report (schema **v2**) binds real
  `artifact_root` / `batch_contract` / `runner_capabilities` / `preprocessor` /
  `postprocessor` sha256 (each `^sha256:[0-9a-f]{64}$`, placeholders rejected) plus
  order-sensitive per-request hashes.
- **Persisted evidence bundle** — `write_evidence_bundle` writes
  `official_rollout_readiness_report.json/md`, `official_rollout_trace.jsonl`, and
  `checksums.json` per case (single B=1, native B=2/4, split B=2/4), **even on
  failure** (`failed_stage` + `errors`). CI uploads the bundles.
- **Dual-target gate** — `lerobot-rollout-stable` (0.6.0, blocking) and
  `lerobot-rollout-dev` (pinned 0.6.1-dev, `continue-on-error` but the rollout may
  **not skip** inside the job, `COREAI_REQUIRE_ROLLOUT=1`); both upload evidence
  with `if: always()`.
- **RunnerCapabilities v2 object-typing** — `capabilities()` rejects non-object
  `supports`/`action_batching`/`inference_state`, a non-string `protocol_version`,
  and a `backward_compatible_with` that isn't `list[str]` (a bare string no longer
  becomes a char tuple).
- **Fixture feature semantics** — the wire check asserts exact `state [.,A]` and
  image `[.,C,H,W]` shapes; the report marks `fixture_feature_semantics_verified`
  true and `universal_feature_contract_verified` **false** (honest).

## v1.3.14 — independently verifiable rollout evidence

Evidence becomes **self-sufficient and third-party verifiable offline** — the
producer is no longer trusted.

- **Offline verifier** — `lerobot-coreai verify-official-rollout-evidence --bundle …
  --require-complete-matrix` (base package, no lerobot) recomputes every checksum,
  validates report/bundle/matrix schemas, recomputes bundle + matrix roots, requires
  the full 5-case matrix, and **refuses** any promoted/forbidden claim or secret.
  Detects report tamper, missing cases, and root mismatches.
- **Exact environment identity** — the report binds LeRobot version/source/commit,
  Python/Torch/NumPy/platform, core + companion versions, repo head SHA, workflow
  run id + job, and target (from env + `importlib.metadata`).
- **Request AND response/action hashes** — ordered request + response hashes, plus
  `final_action_sha256` and `done_mask_sha256`; the evaluator derives
  `response_action_chain_valid`.
- **Canonical hashing** — `canonical-json-sha256.v1` (JSON types only; non-finite /
  non-JSON rejected) replaces `json.dumps(default=str)`.
- **Derived checks, no fallbacks** — `fixture_feature_semantics_verified` is derived
  from measured request shapes (not caller-supplied); missing negotiated
  capabilities raise `EvidenceBindingError` (no synthetic hash); artifact hashes are
  bound only after `verify_plugin_artifact(deep).ok`.
- **Closed schemas** — the readiness report's `checks` are a required, closed set
  and forbidden claims are `const:false`.
- **Time/slot fixture** — observations depend on timestep + slot, so ordered request
  hashes are genuinely distinct (`distinct_request_hashes`); reorder/tamper change
  the roots.
- **Bundle + matrix + failure evidence** — per-case `bundle_manifest.json` +
  `checksums.json`, an aggregate `official_rollout_matrix_manifest.json` with a
  recomputable root, and `write_failure_evidence` for stage exceptions. Both CI
  rollout jobs generate, **verify offline**, and upload the bundle.

## v1.3.15 — semantic evidence replay + matrix/bundle hardening

The verifier now **re-derives** every passing claim from recorded raw data instead
of trusting the report that contains it.

- **Semantic replay** — each case bundle persists `measurements.json` (canonical raw
  requests/responses/done-mask/final-action/fixture); `lerobot_coreai.rollout_replay.
  replay_rollout_evidence` re-derives the full check set (lerobot-free) and matches
  it to the report's checks/claims and recomputed request/response/final/done hashes.
  The plugin evaluator and the verifier share ONE `derive_checks` engine.
- **Response → final-action chain** — the fake runner returns deterministic,
  index-dependent actions; the replay reconstructs `final_action` from the responses
  via the temporal-queue transpose and requires an exact, finite match (a response
  reorder or chain break fails).
- **Restored wire validation** — the derived `wire_payload_valid` again requires the
  exact observation key set, **no** `action`/`reward`/`done`/`success`/`index`/
  `timestamp` leakage, correct `task` type/length, `batch_size`, and encoding/hash
  options — plus per-feature fixture shapes.
- **Target identity fixed** — `capture_environment_identity`/`write_matrix_manifest`
  take the target from `COREAI_ROLLOUT_TARGET` (the dev bundle is no longer
  mislabeled `stable`); the readiness environment records it.
- **Matrix ↔ case binding** — the verifier compares each matrix entry's
  `bundle_root_sha256` + `passed` to the **independently** verified case root/result,
  requires the matrix case set to equal the verified bundles, and recomputes the
  matrix root.
- **Bundle security** — path-traversal / absolute / symlink rejection before opening
  any manifest/checksum path, exact checksum coverage (`report`+`md`+`trace`+
  `measurements`+`bundle_manifest`), exact bundle-manifest content coverage, and a
  recursive secret scan over the report + measurements.

## v1.3.16 — evidence closure

Every byte the offline verifier reads is now schema-checked before access and
cross-bound; the verifier cannot crash on a malformed bundle.

- **Closed raw schema + no-crash** — `measurements.json` (and trace events) have
  closed schemas (`MEASUREMENTS_SCHEMA`); `replay_rollout_evidence` validates the
  raw against it and wraps derivation in try/except, so a malformed bundle returns
  a **structured failure**, never an uncaught exception.
- **`observation_sha256` recomputed** — the wire check re-derives each request's
  `options.observation_sha256` from the sent observation and requires it to match;
  it also validates `observation_encoding`, `protocol_version`, and the
  single/split-must-not-send-`batch_size` rule.
- **Strict response validation** — `_response_valid` requires exactly `[H,A]` /
  `[B,H,A]`, exact horizon/dim, finiteness, non-ragged, and a closed `{action}` key
  set **before** the chain reconstruction (which is additionally index-guarded).
- **Trace is a verified source** — the verifier parses `official_rollout_trace.jsonl`
  and requires `index==0..N-1` and `trace hashes == report hashes == recomputed raw
  hashes` (a trace that contradicts the report/measurements fails).
- **Matrix v2 target binding** — the matrix root binds `target` + `passed` +
  bundle root (a target/pass flip changes it), and the verifier requires
  `matrix.target` to equal **every** case's report environment target.
- **Exact actual-file coverage** — the case directory's real file set must equal the
  expected set (report/md/trace/measurements/bundle_manifest/checksums) — no extra,
  hidden, nested, or symlinked files.
- **Full-bundle secret scan** — report + measurements are scanned for sensitive
  keys / credential URLs.

## v1.3.17 — queue lifecycle evidence + verifier closure

The queue's internal state transitions are now proven by recorded events, and the
remaining cheap verifier gaps are closed.

- **Queue lifecycle observer** — `CoreAIBridgePolicy` records a typed event stream
  (`queue.reset`/`empty`/`refill_requested`/`chunk.validated`/`chunk.committed`/
  `action.popped`) when `record_queue_events=True` (off by default, trivial
  overhead). Persisted in `measurements.json` and validated offline as a **state
  machine**: reset first, monotonic indices, refill only when empty, commit only
  after validation, pop only after commit, ≤H pops per chunk. The `queue_refilled`
  proxy is replaced by derived `queue_lifecycle_valid` + `queue_refill_count_exact`
  (refills == commits == predictions, pops == sequence length).
- **`TRACE_EVENT_SCHEMA` enforced** — the verifier now `jsonschema.validate`s each
  trace event (was defined but unapplied), and the trace check recomputes
  `raw_resp` too so it is self-sufficient (not merely transitively bound).
- **Strict rectangular arrays** — `_rect_shape` validates that **every** branch of a
  nested array matches (a ragged response/final/fixture with an extra horizon or dim
  on a later slot fails, not just the first branch).
- **Tighter measurements schema** — `done` values are `enum [0,1]`,
  `single_only ⇒ batch_size==1` (if/then), `required_obs_keys` unique.

## v1.3.18 — Runtime Evidence Protocol v2

The queue event stream is upgraded from a loose transition log to a **causal,
hash-bound protocol** that a formal offline state machine replays and refuses to
accept when tampered.

- **Evidence sessions** — `begin_evidence_session(run_id)` /
  `end_evidence_session()` bracket a rollout with `execution.started` /
  `execution.completed`. `reset()` now emits `policy.reset` **without** clearing the
  buffer, so a rollout that calls `reset()` once (as `lerobot_eval.rollout` does)
  no longer contaminates or truncates the evidence (P1.7).
- **Causal identity** — every chunk carries a `prediction_id` (monotonic, never
  reused) and a `chunk_id`; each `action.popped` is attributed to the **committing**
  chunk's `prediction_id`, fixing the v1.3.17 off-by-one where a pop was tagged with
  the *next* prediction index (P1.1). The state machine rejects reused ids and pops
  attributed to a non-active prediction.
- **Typed, closed event schema** — `QUEUE_EVENT_SCHEMA` (base) with
  `additionalProperties:false` over the closed `QUEUE_EVENT_TYPES`; each event is
  `jsonschema`-validated offline (P1.2). Adds `runner.request_started` /
  `runner.response_received` (with `response_sha256`) and `queue.exhausted`.
- **Queue arithmetic proven** — `chunk.committed` must satisfy `after == before + H`
  and `action.popped` must satisfy `after == before - 1`, with the empty→refill
  precondition (`queue.empty` before any `queue.refill_requested`) required (P1.3,
  P1.4, P1.9).
- **Per-event hashes + atomic commit** — the committed `chunk_sha256` must equal the
  validated one, the popped `selected_action_sha256` is recorded, and the commit is a
  single atomic queue extend (a `pending` tuple, one `extend`) so no partial chunk is
  ever observable (P1.5, P1.6, P1.8).
- **Single engine** — the base `_queue_lifecycle` is one state machine shared by the
  producer's derived checks and the offline verifier; `reqs_per_refill` is `B` in
  `split_and_stack` (B requests per refill) and `1` otherwise.

## v1.3.19 — Execution Envelope + Negotiation Binding + Failure Evidence

Every execution must be complete, negotiated, terminal and independently explainable
— even when it fails. v1.3.18 proved the queue's *happy path* causally; v1.3.19
closes the execution envelope: its start, negotiation, termination and failure path.

- **Session state machine** — `begin_evidence_session(run_id)` /
  `end_evidence_session()` enforce IDLE→ACTIVE→IDLE. A double-begin, an
  end-without-begin and a double-end all raise, so an incomplete execution can never
  be silently overwritten or double-sealed (P1.8, P1.9). Completion declares any
  actions still cached in the queue (`unused_action_count` + hashes +
  `termination_reason`), so a truncated rollout is explicit, not a silent drop.
- **Discriminated trace-event schema (v3)** — `EXECUTION_EVENT_SCHEMA` is a
  `oneOf` keyed on `event`: each event declares exactly the fields it must and may
  carry (`additionalProperties:false` per branch), so
  `{"event":"execution.started","chunk_sha256":null}` is no longer structurally valid
  (P1.1). The permissive v2 schema (every field optional for every event) is gone.
- **Full causal identity** — `request_id`, `sample_index`, `action_id` and
  `rollout_step` join `prediction_id`/`chunk_id`. Request ids are unique with exactly
  one response per request; split issues `B` slot-bound requests per refill; action
  ids never repeat and `rollout_step`/`chunk_timestep` are ordered (P1.2).
- **Constant id + monotonic clock** — `execution_id` must be identical on every
  event, and `relative_monotonic_ns` (relative to session start) must be
  non-decreasing (P1.6, P1.7).
- **Response→chunk binding** — `chunk.validated` carries `ordered_response_sha256s`,
  which the verifier binds to the **recomputed** response bodies of that prediction
  (native: 1; split: the ordered `B`), and the committed chunk hash still equals the
  validated one (P1.3, P1.4).
- **Selected-action binding** — each `action.popped.selected_action_sha256` must
  equal the canonical hash of the `final_action` slice at its `rollout_step`, binding
  the trace's popped action to the rollout's recorded action (P1.5).
- **Terminal queue semantics** — a pop that empties the queue forces
  `AWAITING_EXHAUSTED`; only `queue.exhausted` may follow (P1.11). Normal vs abort
  reset is distinguished — an `abort` must record `discarded_action_count` +
  `discarded_queue_sha256`; a normal reset is only valid on an already-empty queue
  (P1.10).
- **NegotiationRecord v1** — the negotiated protocol/encoding + runner-capabilities
  hash are persisted (`measurements.negotiation`) and self-hashed. Wire validation
  now requires every request's `observation_encoding`/`protocol_version` to **equal
  the negotiated result** (no hardcoded allowlist when a record is present), and the
  report binds the record hash (P1.14).
- **FailureEvidence v2** — `write_failure_evidence` emits a schema-valid bundle
  (`failure_report.json` with all claims pinned false, `execution_envelope.json`,
  `environment_identity.json`, `partial_trace.jsonl`, manifest + checksums) that the
  offline verifier re-proves exactly like a success bundle. Success and failure cases
  are representable together in one matrix (`passed` bound per case) (P1.13).
- **EnvironmentIdentity v2** — separates the provenance SHAs a PR merge conflates
  (`source_head_sha` / `base_sha` / `merge_sha` / `workflow_sha`) plus run attempt +
  runner image.

## v1.3.20 — Negotiation/Failure Closure + Canonical Stage Foundation

Before describing every tensor, close the protocol, failure and stage boundaries
that give those tensors meaning. v1.3.19 introduced negotiation binding + structural
failure bundles; v1.3.20 makes them **certificate-grade** and lays the canonical
stage vocabulary a `FeatureContract` will need.

- **NegotiationRecord v2 + offline re-run** — the record now carries
  `selection_policy` + the runner's `runner_backward_compatible_with`, and the
  normalized announced capabilities are persisted as `runner_capabilities.json`.
  The offline verifier **recomputes** the capabilities hash from that file and
  **re-runs the exact negotiation algorithm** (a lerobot-free
  `negotiation_algorithm.py` shared by intent), requiring the recomputed protocol +
  encoding to equal the persisted result — so a self-consistent but semantically
  invalid record (e.g. negotiated below the minimum, an undeclared downgrade, an
  unannounced encoding) is rejected, not merely hashed (P1.2, P1.3, P1.4).
  Negotiation is **required** for certificate-grade success evidence (legacy
  unnegotiated bundles only pass with an explicit `require_negotiation=False`) (P1.1).
- **Failure closure** — closed `FAILURE_BUNDLE_MANIFEST_SCHEMA`; exact
  checksum/manifest/actual-file coverage; `environment_identity.json` validated
  against the full environment schema (P1.5, P1.6, P1.7); a partial trace must end
  with a **terminal** `execution.failed`/`execution.aborted` event whose
  `execution_id` + `failed_stage` are **cross-bound** to the report and envelope
  (P1.8). Discriminated failure event types (`negotiation.failed`,
  `runner.request_failed`, `chunk.commit_failed`, …) are in the schema.
- **Producer terminal validation** — `end_evidence_session` fail-fasts on an open
  request (emits `execution.failed`, never an invalid `execution.completed`), and
  completion **always** declares `termination_reason` + `unused_action_count` +
  `unused_action_sha256s`, even when zero (P1.10, P1.11).
- **Exact provenance** — the CI workflow resolves `source_head_sha` / `base_sha` /
  `merge_sha` / `workflow_sha` via dedicated `COREAI_*` env vars, since `GITHUB_SHA`
  is a synthetic merge commit on PRs and `GITHUB_BASE_REF` is a branch name (P1.12).
- **Canonical stage vocabulary** — base `ObservationStage`/`ActionStage` enums +
  `ProcessorStageContract v1` schema (with an identity-transform hash rule) + a
  **single** canonical `schemas/batch-contract-v3.schema.json` the parser validates
  against — the typed language a `FeatureContract` will bind features to.

## v1.3.21 — Negotiation Semantics + Failure E2E + capabilities normalization

Remove every semantic alias, execute every important failure, and make negotiation
mean exactly what it declares. v1.3.20 created the negotiation/failure/stage formats;
v1.3.21 unifies the negotiation primitive, normalizes capabilities, and runs the
failure paths through the real stack.

- **One negotiation primitive** — a single `negotiate_runner_contract()` in the base
  package is the source of truth for what "negotiated" means; the record builder and
  the offline verifier both go through it (P1.3). `selection_policy` is pinned to the
  only implemented value, `minimum_compatible` — the schema rejects a policy whose
  semantics the verifier would otherwise ignore (P1.1, P1.2).
- **Capabilities normalization** — the bundle persists **both**
  `runner_capabilities_raw.json` (audit) and `runner_capabilities_normalized.json`
  (canonical: explicit defaults, sorted unordered lists, duplicates rejected). The
  verifier re-runs normalization from the raw payload OFFLINE and requires it to equal
  the persisted normalized file AND the record's `normalized_capabilities_sha256`;
  the normalized file is schema-validated. Decisions bind the normalized hash
  (P1.4, P1.5).
- **Failure paths through the real stack** — a real local HTTP runner is driven to
  fail at negotiation, request (HTTP 500), and validation (non-finite / wrong-shape
  action); the policy's `abort_evidence_session(failed_stage)` emits a **runtime**
  stage-specific failure event + a terminal `execution.failed`, and the bundle is
  re-proved offline. `failure_report.terminal_event_origin` distinguishes a
  **runtime** terminal (certificate-grade) from a **writer-synthesized** one
  (diagnostic, e.g. a bind-time failure with no session) (P1.6, P1.8, L).
- **Producer terminal validation** — `end_evidence_session` still fail-fasts on an
  open request; `abort_evidence_session` seals a FAILED execution without emitting an
  invalid `completed`.
- **RuntimeSupportProfile v1** — persisted beside the matrix and verified against the
  canonical frozen profile: native slot isolation is `independent`-only and split
  scopes are `stateless`/`request_scoped`-only, separating the *expressible*
  BatchContract language from the *supported* subset (P1.14). A tamper adding native
  `shared` or split `global` fails.

## v1.3.22 — Semantic Authority Closure

One negotiation engine, one normalized capability truth, one stage language, and no
post-hoc failure claims. v1.3.21 persisted and revalidated negotiation/capabilities/
failures; v1.3.22 makes those representations the runtime's real authority.

- **One negotiation engine** — the plugin's `negotiate_runner_protocol` is now a thin
  adapter: the protocol + encoding **decision** lives only in the base
  `negotiate_runner_contract`, which the offline verifier also runs. No parallel
  selection logic remains (P1.1). A parity test asserts runtime and base agree.
- **Fail-closed capabilities normalization** — normalization **never coerces**: a JSON
  string where a bool/int is required, or an unknown enum, now **raises** instead of
  being silently promoted (the old `bool("false") is True` / `int("4") == 4` traps are
  gone) (P1.2). The raw payload is validated against a strict `RawCapabilitiesContract`
  first (P1.3), and the normalized form uses **closed enums** for
  semantics/slot-isolation/scope/reset-scope (P1.4). A typed, immutable
  `NormalizedRunnerCapabilities` is the single authority object (P1.5).
- **Automatic failure-stage classification** — an `_evidence_stage(...)` context
  manager wraps each boundary (negotiate / request / validation); on an exception the
  **runtime** emits the stage-specific failure event + terminal `execution.failed`
  tagged `runtime_exception_boundary` — the caller no longer chooses `failed_stage`
  (P1.6, P1.7). `abort_evidence_session` remains as an explicit, post-hoc
  (`runtime_api_posthoc`) diagnostic path.
- **Evidence grades** — `verify-official-rollout-evidence --evidence-grade
  certificate` requires negotiation, a `runtime_exception_boundary` failure terminal,
  and the RuntimeSupportProfile; `diagnostic` (default) accepts writer-synthesized
  terminals (P1.8). A writer-synthesized bundle passes diagnostic but fails
  certificate.
- **RuntimeSupportProfile mandatory in certificate grade** — a certificate-grade
  matrix must declare it; the canonical profile is verified byte-for-byte (P1.10).

## v1.3.23 — Capabilities Authority + conditional contract + failure taxonomy

Normalized capabilities now drive execution, not just evidence; the failure taxonomy
is precise; and the RuntimeSupportProfile is cryptographically bound to the matrix.

- **Normalized capabilities are the runtime authority** — the typed, fail-closed
  `NormalizedRunnerCapabilities` (duck-typed for the batch decision) is what
  `select_batch_execution_mode` reads; the runtime no longer decides batching from the
  raw/parsed object (P1.1, P1.4). A binding also requires `supports_action` (P1.3).
- **Conditional capabilities contract + alias resolution** — normalization now
  enforces: `action_batching.supported` ⇒ a `semantics` + `slot_isolation` +
  `max_batch_size ≥ 2`; `native` ⇒ `slot_isolation`; `session_scoped` /
  `reset_scope=session` ⇒ `supports_session_ids`. The `state_isolation` ↔
  `slot_isolation` alias is resolved and a **conflict fails** (P1.2, P1.3).
- **Precise failure taxonomy** — the runner boundary is classified by exception TYPE:
  a transport error is `request`, a response-contract violation (non-finite / wrong
  shape / malformed) is `response` — no longer mislabeled as a generic `request`
  (P1.6).
- **RuntimeSupportProfile matrix-root bound** — the profile hash is folded into the
  matrix root and covered by the matrix checksums, and the verifier recomputes it from
  the on-disk profile; removing or tampering with the profile now fails the root, not
  just the sidecar (P1.7).

## Not yet

- **Full `raw_lerobot_observation` removal + canonical stage migration** — the
  `ObservationStage`/`ActionStage` vocabulary, `ProcessorStageContract` and
  `RuntimeSupportProfile` are in and enforced; migrating the manifest + processor
  contract + artifact builder/verifier + every consumer off the legacy `expects`
  ownership strings (a semantic mapping, not a pure rename, across ~10 files incl. the
  E2E fixtures) + making `ProcessorStageContract` mandatory & hash-bound + removing the
  duplicate companion BatchContract schema is its own isolated PR — v1.3.24.
- **Failure injection through the official `lerobot_eval.rollout` CI matrix** — the
  failure paths run E2E through the real policy/transport/negotiation stack with
  precise runtime-boundary classification + certificate-grade offline verification;
  driving them *inside* the stable/dev rollout jobs (incl. preprocessor / env.step /
  batched failures) is v1.3.24.
- **Stable wheel-distribution digest** — `lerobot_distribution_sha256` for the stable
  target — v1.3.24.
- **Chunk happy-path sub-stage events** (`chunk.assembly_started`/`assembled`/
  `validation_started`/`commit_started`) + commit-fault rollback fixture (the
  failed-refill/no-partial-commit case is in) — v1.3.24.
- **`observation_stage` closure in BatchContract** (required + enum, or moved to the
  stage contract) — v1.3.24.
- **`FeatureContract` v1 + real `LeRobotDatasetMetadata`** (→
  `universal_feature_contract_verified` true) — v1.3.24+.
- **`FeatureContract` v1 + real `LeRobotDatasetMetadata`** — dtype/names/layout/
  value_range/units in the manifest schema and an on-disk official dataset fixture
  to close `feature_dtype` / `action_names_order` / `image_layout_range` so
  `semantic_completeness_verified` can be `true` — v1.3.12.
- **Formal rollout evidence** — `official_rollout_readiness_report.json/md` +
  `batch_execution_report`; the rollout E2E is the evidence today (run in CI on
  both targets), the signed report is v1.3.12/v1.3.13.
- **Session-scoped / global batching** — requires a per-slot session lifecycle
  (create / session-ids / reset / close) + transaction/rollback protocol; deferred
  until that contract exists. Only stateless/request-scoped B>1 is supported.
- Signed **compatibility certificate v2** that promotes `plugin_compat` levels
  (`policy_factory` / `processor_pipeline`) from hash-bound, signed evidence —
  v1.3.12. `authenticity_verified` stays `false` until then.
- Compare/eval temporal evidence hardening — v1.3.13.
- Official `lerobot-eval` end-to-end certification — v1.4.0; Apple CoreAI runtime
  certification — v1.4.1.

Guarded real egress remains a separately enforced runtime. `official_eval_certified`
stays `false` until a real official-eval E2E.

## v1.3.26.9 — official-eval promotion authority (no self-certification, P0.1)

The prior builder derived `official_eval_certified` from a caller-supplied `execution`
dict, so a hand-built `argv=["/usr/bin/lerobot-eval", …]` + all-true `checks` could
certify. That self-certification path is closed:

- The public builder is now `build_diagnostic_official_eval_report` — `evidence_grade:
  "diagnostic"`, `official_eval_certified` **always false**.
- A **true** certificate is produced only by `promote_official_eval_certificate(receipt=…)`,
  which accepts **only** a `VerifiedOfficialEvalExecutionReceipt` (a dict/bool raises
  `TypeError`). The receipt is minted by `authority.verify_official_eval_execution_receipt`,
  which requires a **real subprocess** (no `fake_executor`), the **official entrypoint**
  argv, an executable resolved to an **installed path** (a `/tmp/lerobot-eval` shim is
  refused), the **coreai env instantiated**, the **full case matrix**, and a **clean
  exit**. The certificate's checks are derived from the receipt, never supplied.
- The verifier is grade-aware: a diagnostic report forged to `true` is rejected; a
  certificate grade must equal the gate.

The fully-live executor (running the real `lerobot-eval` subprocess against a
registered `coreai_cert_env` and deriving the receipt from captured outputs) lands in
v1.3.27; this ships the receipt type + gate + anti-forgery closure.

## v1.3.26.10 — full replay + root closure (P1.1 / P1.3)

Certificate-grade evidence is now **independently replayable**, not merely
self-consistent:

- **Processor parity (P1.1)** — a `certificate`-grade `ProcessorParityReport` persists
  each case's **raw reference/candidate arrays**; the verifier **re-derives** metrics,
  hashes, and pass/fail from them. A report with forged metrics (but a consistent
  `passed`/`reasons` pair) is caught by the fresh recomputation.
- **Model conversion (P1.3)** — a `certificate`-grade `ModelConversionEvidence` must
  carry a **`replay_bundle`** (raw reference/candidate outputs); the verifier recomputes
  numeric parity from it offline. `model_conversion_verified` is promoted **only** in
  certificate grade with passing parity; a certificate evidence missing the bundle, or
  whose bundle doesn't reproduce the record, is rejected. `diagnostic` grade omits the
  bundle and never promotes the claim.

**Honestly deferred (interdependent, not above authority closure):** the reference =
real LeRobot `NormalizerProcessor` loaded from serialized config (P1.2, needs a lerobot
rollout CI job), and the OfficialEvalCertificate root-set expansion (policy-execution /
model-conversion / processor-stage / runtime-support / negotiation / runner-capabilities
/ rollout-matrix roots) — both land with the v1.3.27 live-run work.
