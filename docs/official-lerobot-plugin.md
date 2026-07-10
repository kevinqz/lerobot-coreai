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
  stays **false** until a trusted signature is issued (signing lands in v1.3.9).
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
  `false` (promoted only by the signed certificate, v1.3.9).

## Not yet

- Signed **compatibility certificate v2** that promotes `plugin_compat` levels
  (`policy_factory` / `processor_pipeline`) from hash-bound, signed evidence —
  v1.3.9. `authenticity_verified` stays `false` until then.
- Full batched evaluation (`batch_size > 1`) — v1.3.8.
- Compare/eval temporal evidence hardening — v1.3.10.
- Official `lerobot-eval` end-to-end certification — v1.4.0; Apple CoreAI runtime
  certification — v1.4.1.

Guarded real egress remains a separately enforced runtime. `official_eval_certified`
stays `false` until a real official-eval E2E.
