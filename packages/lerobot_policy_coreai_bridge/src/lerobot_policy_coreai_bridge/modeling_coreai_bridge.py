# modeling_coreai_bridge.py — official LeRobot policy for the CoreAI bridge (v1.3.1).
#
# A real PreTrainedPolicy (hence torch.nn.Module) that the official LeRobot
# factory can construct AND that binds a CoreAI runtime via from_pretrained:
#   - __init__ accepts the kwargs make_policy passes (dataset_stats/dataset_meta);
#   - from_pretrained loads the config, resolves the runner URL from an env var
#     (fail-closed, no secret persisted), binds a lerobot_coreai.CoreAIPolicy,
#     and cross-binds the CoreAI manifest to the config expectations;
#   - select_action returns torch.Tensor(B, action_dim) on the policy's device;
#   - B>1 fails clearly under batch_mode="single_only" (v1.3.1).
# Runtime-only: forward/get_optim_params/train(True) raise. No hardware, no egress.

from __future__ import annotations

import os
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

from lerobot.policies.pretrained import PreTrainedPolicy

from .configuration_coreai_bridge import POLICY_TYPE, CoreAIBridgeConfig


class PluginBindingError(RuntimeError):
    """Raised when the CoreAI runtime cannot be bound or fails cross-binding."""


def _action_sha256(action: Any) -> str:
    """Canonical hash of an action tensor (materialized to a JSON-safe list)."""
    from lerobot_coreai.rollout_evidence_schema import canonical_json_sha256
    return canonical_json_sha256(action.detach().to("cpu").tolist())


def _infer_batch_size(batch: dict[str, Any]) -> int:
    """Best-effort batch size from a LeRobot observation batch (leading dim)."""
    sizes = set()
    for k, v in batch.items():
        if k == "task":
            if isinstance(v, (list, tuple)):
                sizes.add(len(v))
            continue
        shape = getattr(v, "shape", None)
        if shape is not None and len(shape) >= 1:
            sizes.add(int(shape[0]))
    if not sizes:
        return 1
    return max(sizes)


class CoreAIBridgePolicy(PreTrainedPolicy):
    """LeRobot-shaped policy backed by a CoreAI runtime. Runtime-only."""

    config_class = CoreAIBridgeConfig
    name = POLICY_TYPE

    def __init__(self, config: CoreAIBridgeConfig, coreai_policy: Any = None,
                 dataset_stats: Any = None, dataset_meta: Any = None, **kwargs: Any):
        super().__init__(config)
        self.config = config
        self.coreai_policy = coreai_policy
        self.dataset_stats = dataset_stats
        self.dataset_meta = dataset_meta
        self._queue: deque = deque()          # deque[Tensor[B, A]] (temporal)
        self._active_batch_size: int | None = None
        # v1.3.18: queue evidence protocol v2. OFF by default; an explicit
        # begin_evidence_session(run_id) opens a causal, typed event stream the
        # offline verifier replays as a formal state machine. Each pop is bound to
        # the prediction/chunk that produced it; no cross-run contamination.
        self.record_queue_events: bool = False
        self.queue_events: list[dict[str, Any]] = []
        self._event_index: int = 0
        self._prediction_index: int = 0
        self._execution_id: str | None = None
        self._active_prediction_id: int | None = None
        self._active_chunk_id: str | None = None
        # v1.3.19 Execution Trace Protocol v3: session state machine, monotonic
        # clock, causal request/action ids, and per-prediction response hashes.
        self._session_state: str = "IDLE"          # IDLE | ACTIVE
        self._clock_base_ns: int = 0
        self._rollout_step: int = 0
        self._pops_since_commit: int = 0
        self._pending_responses: list[str] = []
        self._open_requests: int = 0
        self.negotiation_record: dict[str, Any] | None = None
        self.runner_capabilities_raw: dict[str, Any] | None = None
        self.runner_capabilities_normalized: dict[str, Any] | None = None
        self._normalized_capabilities: Any = None   # typed authority for decisions
        self.last_observation_sha256: str | None = None
        self.last_observation_audit: dict[str, Any] = {}
        self.last_batch_mode: str | None = None
        self._action_contract = None
        self._batch_contract = None
        self._protocol = None
        self._capabilities = None
        if coreai_policy is not None:
            self._bind_action_contract(coreai_policy)
        self.register_buffer("_sentinel", torch.zeros(1), persistent=False)

    def _bind_action_contract(self, coreai_policy: Any) -> None:
        """Parse the manifest action + batch contracts, fail-closed (v1.3.5/1.3.9).

        A malformed contract must not silently degrade to chunk/no-horizon
        validation; it is a binding error.
        """
        manifest = getattr(coreai_policy, "manifest", None)
        if manifest is None:
            # No manifest bound (local/test path): nothing to fail closed on.
            self._action_contract = None
            return
        from lerobot_coreai.action_contract import (
            parse_action_contract_from_manifest, parse_batch_contract_from_manifest)
        try:
            self._action_contract = parse_action_contract_from_manifest(manifest)
            self._batch_contract = parse_batch_contract_from_manifest(manifest)
        except Exception as exc:
            raise PluginBindingError(
                f"invalid CoreAI action/batch contract: {exc}") from exc

    def _resolve_protocol(self):
        """Negotiate the runner protocol once, honoring runtime_binding_mode (v1.3.6).

        - strict/legacy: a runner is REQUIRED. Capabilities are fetched and any
          capabilities/transport/protocol failure PROPAGATES (never silent legacy).
          ``legacy`` additionally accepts a runner that announces no protocol.
        - in_memory: NO wire boundary — for local/test binding against an
          in-process CoreAI policy with no RunnerClient. Uses the config default
          encoding and the minimum protocol.

        require_protocol_negotiation is now expressed by the mode: only in_memory
        skips negotiation, and it must be chosen explicitly.
        """
        if self._protocol is not None:
            return self._protocol
        from .negotiation import NegotiatedRunnerProtocol, negotiate_runner_protocol
        mode = self.config.runtime_binding_mode
        runner = getattr(self.coreai_policy, "runner", None)
        has_runner = runner is not None and hasattr(runner, "capabilities")

        if not has_runner:
            if mode != "in_memory":
                raise PluginBindingError(
                    "no runner is bound but runtime_binding_mode is "
                    f"{mode!r}; a runner is required outside 'in_memory' mode.")
            self._protocol = NegotiatedRunnerProtocol(
                protocol_version=self.config.minimum_runner_protocol,
                observation_encoding=self.config.observation_encoding_or_default(),
                supports_batch=False, max_batch_size=1, legacy=True)
            return self._protocol

        if mode == "in_memory":
            raise PluginBindingError(
                "runtime_binding_mode='in_memory' must not be used with a bound "
                "runner; use 'strict' (or 'legacy') for a real wire.")
        caps = runner.capabilities()  # propagate failures — no silent legacy
        self._capabilities = caps      # cached for the batch-execution decision
        self._protocol = negotiate_runner_protocol(
            requested_encoding=self.config.observation_encoding,
            capabilities=caps,
            minimum_protocol=self.config.minimum_runner_protocol,
            allow_legacy=(mode == "legacy"))
        self._build_negotiation_record(caps, self._protocol)
        return self._protocol

    def _build_negotiation_record(self, caps: Any, proto: Any) -> None:
        """Persist the NegotiationRecord v2 (v1.3.20, P1.2/P1.4): the negotiated
        protocol + encoding, the selection policy, and the runner's declared
        backward-compatibility list + announced capabilities. The offline verifier
        RE-RUNS the negotiation algorithm from these inputs and requires equality."""
        from lerobot_coreai.capabilities_normalize import (
            normalize_capabilities, normalized_capabilities_sha256,
            typed_normalized_capabilities)
        from lerobot_coreai.rollout_evidence_schema import (
            NEGOTIATION_SCHEMA_VERSION, canonical_json_sha256)
        from lerobot_coreai.runner import capabilities_sha256
        raw = dict(getattr(caps, "raw", {}) or {})
        # v1.3.23 (P1.1/P1.3): the typed NORMALIZED capabilities are the runtime
        # authority; a fail-closed normalization is the ONLY gate that produces them.
        typed = typed_normalized_capabilities(raw)
        if not typed.supports_action:
            raise PluginBindingError(
                "runner does not announce supports.action; refusing to bind for "
                "policy inference.")
        self._normalized_capabilities = typed
        rec = {
            "schema_version": NEGOTIATION_SCHEMA_VERSION,
            "selection_policy": "minimum_compatible",
            "requested_protocol": None,
            "minimum_protocol": self.config.minimum_runner_protocol,
            "runner_protocol": getattr(caps, "protocol_version", None)
            or self.config.minimum_runner_protocol,
            "runner_backward_compatible_with": list(
                getattr(caps, "backward_compatible_with", ()) or ()),
            "negotiated_protocol": proto.protocol_version,
            "requested_encoding": self.config.observation_encoding,
            "runner_encodings": list(getattr(caps, "observation_encodings", ())
                                     or [proto.observation_encoding]),
            "negotiated_encoding": proto.observation_encoding,
            "runner_capabilities_sha256": capabilities_sha256(caps),
            "normalized_capabilities_sha256": normalized_capabilities_sha256(raw),
        }
        rec["record_sha256"] = canonical_json_sha256(rec)
        self.negotiation_record = rec
        self.runner_capabilities_raw = raw
        self.runner_capabilities_normalized = normalize_capabilities(raw)

    # MARK: - Official runtime binding

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, *, config=None, revision=None,
                        dataset_stats=None, dataset_meta=None, **kwargs):
        """Bind a CoreAI runtime instead of loading PyTorch weights.

        Resolves the config, the CoreAI artifact, and the runner URL (from
        ``config.runner_url_env``; fail-closed if unset), loads a
        ``lerobot_coreai.CoreAIPolicy``, cross-binds its manifest to the config,
        and returns a ready-to-run policy. Never returns coreai_policy=None.
        """
        if config is None:
            config = cls.config_class.from_pretrained(
                pretrained_name_or_path, revision=revision, **kwargs)

        artifact = config.coreai_artifact or str(pretrained_name_or_path)
        runner_url = os.environ.get(config.runner_url_env)
        if not runner_url:
            raise PluginBindingError(
                f"{config.runner_url_env} must be set for coreai_bridge inference "
                "(the runner URL is read from the environment, never persisted).")

        from lerobot_coreai.policy import CoreAIPolicy
        coreai_policy = CoreAIPolicy.from_pretrained(
            artifact, runner_url=runner_url, revision=config.coreai_revision or "main",
            validate_runner=True)

        cls._cross_bind_manifest(config, coreai_policy)
        return cls(config, coreai_policy=coreai_policy, dataset_stats=dataset_stats,
                   dataset_meta=dataset_meta)

    @staticmethod
    def _cross_bind_manifest(config: CoreAIBridgeConfig, coreai_policy: Any) -> None:
        """Fail closed if the CoreAI manifest contradicts declared expectations."""
        from lerobot_coreai.action_contract import parse_action_contract_from_manifest
        contract = parse_action_contract_from_manifest(coreai_policy.manifest)
        # Fail-closed: a declared expectation with an unknown artifact value is a
        # failure, not a pass.
        if config.expected_action_dim is not None:
            if contract.action_dim is None:
                raise PluginBindingError(
                    "expected_action_dim declared but the manifest declares no "
                    "action dimension; refusing to bind.")
            if contract.action_dim != config.expected_action_dim:
                raise PluginBindingError(
                    f"action_dim mismatch: manifest {contract.action_dim} != expected "
                    f"{config.expected_action_dim}.")
        exp_horizon = config.effective_action_horizon()
        if exp_horizon is not None and contract.horizon != exp_horizon:
            raise PluginBindingError(
                f"action_horizon mismatch: manifest {contract.horizon} != expected "
                f"{exp_horizon}.")
        rt = getattr(coreai_policy, "robot_type", None)
        if config.expected_robot_type is not None:
            if rt is None:
                raise PluginBindingError(
                    "expected_robot_type declared but the manifest declares no "
                    "robot type; refusing to bind.")
            if rt != config.expected_robot_type:
                raise PluginBindingError(
                    f"robot_type mismatch: manifest {rt} != expected "
                    f"{config.expected_robot_type}.")
        CoreAIBridgePolicy._cross_bind_features(config, coreai_policy, contract)

    @staticmethod
    def _cross_bind_features(config: CoreAIBridgeConfig, coreai_policy: Any,
                             contract: Any) -> None:
        """Validate make_policy-populated input/output features vs the manifest.

        make_policy fills ``cfg.input_features``/``output_features`` from the
        dataset/env before ``from_pretrained``. Here we hold them against the
        CoreAI manifest, fail-closed:
          - every declared input (observation) feature must exist in the manifest,
            with a matching per-frame shape when the manifest declares one;
          - the ACTION output feature's last dim must equal the manifest action dim
            (horizon lives in the action contract, NOT the per-timestep feature).
        Nothing to check if make_policy has not populated features (standalone use).
        """
        from lerobot.configs.types import FeatureType
        manifest = coreai_policy.manifest
        obs_feats = getattr(manifest, "observation_features", {}) or {}

        def _manifest_shape(name: str):
            spec = obs_feats.get(name)
            if spec is None:
                return None
            shape = getattr(spec, "shape", None)
            if shape is None and isinstance(spec, dict):
                shape = spec.get("shape")
            return tuple(shape) if shape else None

        cfg_inputs = config.input_features or {}
        # Direction 1: every config input feature must be a declared observation.
        for key, feat in cfg_inputs.items():
            if getattr(feat, "type", None) == FeatureType.ENV:
                continue
            if key not in obs_feats:
                raise PluginBindingError(
                    f"input feature {key!r} is not declared in the CoreAI manifest "
                    f"observation features {sorted(obs_feats)}.")
            m_shape = _manifest_shape(key)
            f_shape = tuple(getattr(feat, "shape", ()) or ())
            if m_shape is not None and f_shape and tuple(f_shape) != tuple(m_shape):
                raise PluginBindingError(
                    f"input feature {key!r} shape {f_shape} != manifest {m_shape}.")
        # Direction 2 (v1.3.7): every REQUIRED manifest observation feature must be
        # present in the config inputs — a dataset/env that omits a required
        # observation would silently under-feed the runner.
        if cfg_inputs:
            for name, spec in obs_feats.items():
                if name == "task":
                    continue
                required = getattr(spec, "required", None)
                if required is None and isinstance(spec, dict):
                    required = spec.get("required", True)
                if required and name not in cfg_inputs:
                    raise PluginBindingError(
                        f"required manifest observation {name!r} is missing from the "
                        f"config input features {sorted(cfg_inputs)}.")

        action_feats = {k: f for k, f in (config.output_features or {}).items()
                        if getattr(f, "type", None) == FeatureType.ACTION}
        if config.output_features and not action_feats:
            raise PluginBindingError(
                "output_features present but no ACTION feature; refusing to bind.")
        for key, feat in action_feats.items():
            shape = tuple(getattr(feat, "shape", ()) or ())
            if shape and contract.action_dim is not None and \
                    shape[-1] != contract.action_dim:
                raise PluginBindingError(
                    f"output action feature {key!r} last-dim {shape[-1]} != manifest "
                    f"action_dim {contract.action_dim}.")

    # MARK: - Inference (LeRobot contract)

    def _emit(self, event: str, **fields: Any) -> None:
        """Append a v3 trace event. Only the 4 common fields are implicit; every
        causal id (prediction/chunk/request/action) is passed EXPLICITLY by the
        event that carries it, so the discriminated schema stays exact."""
        if not self.record_queue_events:
            return
        rec = {"event_index": self._event_index, "event": event,
               "execution_id": self._execution_id,
               "relative_monotonic_ns": max(0, time.monotonic_ns() - self._clock_base_ns)}
        rec.update(fields)
        self.queue_events.append(rec)
        self._event_index += 1

    def begin_evidence_session(self, run_id: str) -> None:
        """Open a causal evidence session (v1.3.19). Rejects a double-begin so an
        incomplete execution can never be silently overwritten (P1.9)."""
        if not run_id:
            raise PluginBindingError("begin_evidence_session requires a non-empty run_id.")
        if self._session_state == "ACTIVE":
            raise PluginBindingError(
                "an evidence session is already active; end (or abort) it first.")
        self.record_queue_events = True
        self._session_state = "ACTIVE"
        self.queue_events = []
        self._event_index = 0
        self._prediction_index = 0
        self._rollout_step = 0
        self._pops_since_commit = 0
        self._pending_responses = []
        self._open_requests = 0
        self._failed_sealed = False
        self.terminal_event_origin = None
        self._active_prediction_id = None
        self._active_chunk_id = None
        self._execution_id = run_id
        self._clock_base_ns = time.monotonic_ns()
        self._emit("execution.started")

    def end_evidence_session(self) -> None:
        """Seal the session. Rejects end-without-begin / double-end (P1.8).

        v1.3.20: the producer now fail-fasts on an inconsistent terminal state — an
        open request or a pending (validated-but-uncommitted) chunk emits
        ``execution.failed`` and raises, and NEVER an invalid ``execution.completed``
        (P1.10). Completion ALWAYS declares termination_reason + unused_action_count
        + unused_action_sha256s, even when zero (P1.11)."""
        if self._session_state != "ACTIVE":
            raise PluginBindingError(
                "end_evidence_session called without an active session.")
        if self._open_requests:
            self._emit("execution.failed", failed_stage="response",
                       detail=f"{self._open_requests} request(s) without a response")
            self.record_queue_events = False
            self._session_state = "IDLE"
            raise PluginBindingError(
                "cannot seal an execution with open runner requests.")
        cached = len(self._queue)
        hashes = [_action_sha256(a) for a in self._queue] if cached else []
        reason = ("rollout_complete_with_cached_actions" if cached
                  else "rollout_completed_queue_empty")
        self._emit("execution.completed", termination_reason=reason,
                   unused_action_count=cached, unused_action_sha256s=hashes)
        self.record_queue_events = False
        self._session_state = "IDLE"

    _STAGE_FAILURE_EVENT = {
        "runner_negotiate": "negotiation.failed",
        "request": "runner.request_failed",
        "response": "runner.response_invalid",
        "chunk_assembly": "chunk.assembly_failed",
        "validation": "chunk.validation_failed",
        "queue_commit": "chunk.commit_failed",
    }

    def _seal_failed(self, failed_stage: str, detail: str, origin: str) -> None:
        """Emit the stage-specific failure event + a terminal execution.failed once,
        record the terminal origin, and seal the session."""
        if getattr(self, "_failed_sealed", False):
            return
        stage_event = self._STAGE_FAILURE_EVENT.get(failed_stage)
        if stage_event:
            self._emit(stage_event, failed_stage=failed_stage, detail=detail[:500])
        self._emit("execution.failed", failed_stage=failed_stage, detail=detail[:500])
        self._failed_sealed = True
        self.terminal_event_origin = origin
        self.record_queue_events = False
        self._session_state = "IDLE"

    @contextmanager
    def _evidence_stage(self, failed_stage: str, classify=None):
        """Classify a failing boundary AUTOMATICALLY (v1.3.22, P1.6/P1.7).

        If the wrapped boundary raises during an active evidence session, the RUNTIME
        emits the stage-specific failure event + terminal ``execution.failed`` tagged
        ``runtime_exception_boundary`` — the caller does NOT choose the stage. Inner
        boundaries win (``_failed_sealed`` guards against a re-tag while unwinding).
        ``classify`` (v1.3.23, P1.6) may refine the stage from the exception TYPE
        (e.g. a transport error is ``request``; a response-contract error is
        ``response``) so a non-finite response is not mislabeled as a generic request."""
        try:
            yield
        except Exception as exc:  # noqa: BLE001
            if self.record_queue_events and self._session_state == "ACTIVE":
                stage = classify(exc) if classify is not None else failed_stage
                self._seal_failed(stage, str(exc), "runtime_exception_boundary")
            raise

    @staticmethod
    def _classify_runner_exc(exc: Exception) -> str:
        """request (transport) vs response (the runner's response violated the
        contract). A non-finite / wrong-shape / malformed response is ``response``."""
        from lerobot_coreai.errors import (
            ActionValidationError, RunnerProtocolError,
        )
        if isinstance(exc, (RunnerProtocolError, ActionValidationError)):
            return "response"
        return "request"

    def abort_evidence_session(self, failed_stage: str, detail: str = "") -> list:
        """Explicit, POST-HOC failure seal (diagnostic). Prefer ``_evidence_stage``,
        which classifies the stage at the boundary. Returns the sealed event stream."""
        if self._session_state != "ACTIVE":
            raise PluginBindingError("abort_evidence_session without an active session.")
        self._seal_failed(failed_stage, detail or "", "runtime_api_posthoc")
        return list(self.queue_events)

    def reset(self) -> None:
        n = len(self._queue)
        if n:
            # abort: a partially drained queue is being discarded (P1.10). Record the
            # discarded count + order-sensitive hash so it is not a silent drop.
            discarded = [a.detach().to("cpu").tolist() for a in self._queue]
            self._queue.clear()
            from lerobot_coreai.rollout_evidence_schema import canonical_json_sha256
            self._emit("policy.reset", reset_kind="abort", queue_size_after=0,
                       discarded_action_count=n,
                       discarded_queue_sha256=(canonical_json_sha256(discarded)
                                               if self.record_queue_events else
                                               "sha256:" + "0" * 64))
        else:
            self._emit("policy.reset", reset_kind="normal", queue_size_after=0)
        self._active_batch_size = None
        self._pops_since_commit = 0
        # Invalidate the negotiated protocol + cached caps: a runner may restart
        # with different capabilities, so re-negotiate on the next inference.
        self._protocol = None
        self._capabilities = None
        self._normalized_capabilities = None
        if self.coreai_policy is not None and hasattr(self.coreai_policy, "reset"):
            self.coreai_policy.reset()

    def _batch_size(self, batch: dict[str, Any]) -> int:
        from .transport import infer_and_validate_batch_size
        manifest = getattr(self.coreai_policy, "manifest", None)
        return infer_and_validate_batch_size(batch, manifest)

    def _contract_shapes(self) -> tuple[str, int | None, int | None]:
        c = self._action_contract
        rep = c.representation if c is not None else "chunk"
        horizon = c.horizon if (c is not None and c.representation == "chunk") else None
        dim = self.config.expected_action_dim or (c.action_dim if c is not None else None)
        return rep, horizon, dim

    def _runner_options(self, proto, sha: str, *, batch_size: int | None = None) -> dict:
        opts = {
            "protocol_version": proto.protocol_version,
            "observation_encoding": proto.observation_encoding,
            "observation_schema_version": "coreai-observation.v1",
            "observation_sha256": sha,
        }
        if batch_size is not None:
            opts["batch_size"] = batch_size
        return opts

    def _runner_call(self, fn, *args, sample_index=None, **kwargs):
        """Invoke a runner call, emitting request_started/response_received with a
        unique request_id bound to the active chunk (v1.3.19, P1.2)."""
        cid = self._active_chunk_id
        rid = (f"{cid}/sample-{sample_index}" if sample_index is not None
               else f"{cid}/native")
        self._emit("runner.request_started", prediction_id=self._active_prediction_id,
                   chunk_id=cid, request_id=rid, sample_index=sample_index)
        self._open_requests += 1
        with self._evidence_stage("request", classify=self._classify_runner_exc):
            raw = fn(*args, **kwargs)
        if self.record_queue_events:
            from lerobot_coreai.rollout_evidence_schema import canonical_json_sha256
            sha = canonical_json_sha256({"action": raw})
            self._pending_responses.append(sha)
            self._emit("runner.response_received",
                       prediction_id=self._active_prediction_id, chunk_id=cid,
                       request_id=rid, sample_index=sample_index, response_sha256=sha)
        self._open_requests -= 1
        return raw

    def _predict_single_chunk(self, batch, proto) -> torch.Tensor:
        from .action_validation import normalize_and_validate_action_chunk
        from .transport import prepare_single_coreai_observation
        manifest = getattr(self.coreai_policy, "manifest", None)
        audit: dict[str, Any] = {}
        obs, sha = prepare_single_coreai_observation(
            batch, manifest, encoding=proto.observation_encoding, audit=audit)
        self.last_observation_sha256 = sha
        self.last_observation_audit = audit
        raw = self._runner_call(
            self.coreai_policy.predict_action_chunk, obs,
            runner_options=self._runner_options(proto, sha))
        rep, horizon, dim = self._contract_shapes()
        with self._evidence_stage("validation"):
            return normalize_and_validate_action_chunk(
                raw, representation=rep, expected_batch_size=1,
                expected_horizon=horizon, expected_action_dim=dim,
                device=self._sentinel.device)

    def _predict_native_chunk(self, batch, proto, b: int) -> torch.Tensor:
        from .action_validation import normalize_and_validate_batched_action_chunk
        from .transport import prepare_batched_coreai_observation
        manifest = getattr(self.coreai_policy, "manifest", None)
        audit: dict[str, Any] = {}
        payload, batch_sha, sample_shas = prepare_batched_coreai_observation(
            batch, manifest, batch_size=b, encoding=proto.observation_encoding,
            audit=audit)
        self.last_observation_sha256 = batch_sha
        self.last_observation_audit = {"batch_size": b, "sample_sha256": sample_shas,
                                       **audit}
        raw = self._runner_call(
            self.coreai_policy.predict_action_batch, payload, batch_size=b,
            runner_options=self._runner_options(proto, batch_sha, batch_size=b))
        rep, horizon, dim = self._contract_shapes()
        with self._evidence_stage("validation"):
            return normalize_and_validate_batched_action_chunk(
                raw, representation=rep, expected_batch_size=b,
                expected_horizon=horizon, expected_action_dim=dim,
                device=self._sentinel.device)

    def _predict_split_chunk(self, batch, proto, b: int) -> torch.Tensor:
        # Split-and-stack (stateless/request-scoped only): B independent requests,
        # validate EVERY sample before committing anything (atomic).
        from .action_validation import normalize_and_validate_action_chunk
        from .transport import split_coreai_observations
        manifest = getattr(self.coreai_policy, "manifest", None)
        samples = split_coreai_observations(
            batch, manifest, batch_size=b, encoding=proto.observation_encoding)
        rep, horizon, dim = self._contract_shapes()
        chunks: list[torch.Tensor] = []
        sample_shas: list[str] = []
        for i, (obs, sha) in enumerate(samples):
            try:
                raw = self._runner_call(
                    self.coreai_policy.predict_action_chunk, obs,
                    sample_index=i, runner_options=self._runner_options(proto, sha))
                with self._evidence_stage("validation"):
                    chunk = normalize_and_validate_action_chunk(
                        raw, representation=rep, expected_batch_size=1,
                        expected_horizon=horizon, expected_action_dim=dim,
                        device=self._sentinel.device)      # [1, H, A]
            except Exception as exc:  # noqa: BLE001
                raise PluginBindingError(
                    f"split-and-stack failed at sample index {i}: {exc}") from exc
            chunks.append(chunk[0])                     # [H, A]
            sample_shas.append(sha)
        stacked = torch.stack(chunks, dim=0)            # [B, H, A]
        from .transport import canonical_batch_sha256
        # Order-sensitive batch hash (P1.11) — NOT just the first sample's hash.
        self.last_observation_sha256 = canonical_batch_sha256(
            b, sample_shas, "split_and_stack")
        self.last_observation_audit = {"batch_size": b, "mode": "split_and_stack",
                                       "sample_sha256": sample_shas}
        return stacked

    def predict_action_chunk(self, batch: dict[str, Any], **kwargs: Any) -> torch.Tensor:
        if self.coreai_policy is None:
            raise PluginBindingError("coreai_bridge has no CoreAI policy bound.")
        from .batch_protocol import (
            MODE_NATIVE, MODE_SINGLE, select_batch_execution_mode,
        )
        b = self._batch_size(batch)
        with self._evidence_stage("runner_negotiate"):
            proto = self._resolve_protocol()  # also caches self._capabilities
        # v1.3.23 (P1.1/P1.4): the batch decision reads capability facts from the
        # typed NORMALIZED authority, not the raw/parsed object.
        decision = select_batch_execution_mode(
            self.config, self._batch_contract,
            self._normalized_capabilities or self._capabilities, b)
        self.last_batch_mode = decision.mode
        if decision.mode == MODE_SINGLE:
            return self._predict_single_chunk(batch, proto)          # [1, H, A]
        if decision.mode == MODE_NATIVE:
            return self._predict_native_chunk(batch, proto, b)       # [B, H, A]
        return self._predict_split_chunk(batch, proto, b)            # [B, H, A]

    @torch.no_grad()
    def select_action(self, batch: dict[str, Any], **kwargs: Any) -> torch.Tensor:
        # Temporal queue in the LeRobot chunked-policy style: a chunk [B, H, A] is
        # transposed to a deque of per-timestep [B, A] tensors; each call pops one.
        b = self._batch_size(batch)
        if self._queue and self._active_batch_size is not None and b != self._active_batch_size:
            raise PluginBindingError(
                f"batch size changed ({self._active_batch_size} -> {b}) while the "
                "action queue is non-empty; drain or reset() first.")
        if not self._queue:
            pid = self._prediction_index
            cid = f"chunk-{pid}"
            self._emit("queue.empty", queue_size_after=0)
            # Allocate the prediction/chunk id BEFORE the refill and keep it on every
            # request/response/validate/commit/pop of this chunk (P1.1).
            self._active_prediction_id = pid
            self._active_chunk_id = cid
            self._pending_responses = []
            self._emit("queue.refill_requested", prediction_id=pid, chunk_id=cid,
                       queue_size_before=0, queue_size_after=0)
            chunk = self.predict_action_chunk(batch, **kwargs)       # [B, H, A]
            self._active_batch_size = int(chunk.shape[0])
            h = int(chunk.shape[1])
            chunk_hash = _action_sha256(chunk) if self.record_queue_events else None
            self._emit("chunk.validated", prediction_id=pid, chunk_id=cid,
                       chunk_sha256=chunk_hash, horizon=h,
                       ordered_response_sha256s=list(self._pending_responses))
            before = len(self._queue)
            # atomic commit: materialize the whole chunk, then a single extend.
            pending = tuple(chunk.transpose(0, 1))                   # each [B, A]
            self._queue.extend(pending)
            self._emit("chunk.committed", prediction_id=pid, chunk_id=cid,
                       chunk_sha256=chunk_hash, committed=h,
                       queue_size_before=before, queue_size_after=len(self._queue))
            self._prediction_index += 1
            self._pops_since_commit = 0
        before = len(self._queue)
        action = self._queue.popleft()                               # [B, A]
        step = self._rollout_step
        self._emit("action.popped", prediction_id=self._active_prediction_id,
                   chunk_id=self._active_chunk_id,
                   action_id=f"{self._active_chunk_id}/t{self._pops_since_commit}",
                   rollout_step=step, chunk_timestep=self._pops_since_commit,
                   queue_size_before=before, queue_size_after=len(self._queue),
                   selected_action_sha256=(_action_sha256(action)
                                           if self.record_queue_events else
                                           "sha256:" + "0" * 64))
        self._rollout_step += 1
        self._pops_since_commit += 1
        if not self._queue:
            self._emit("queue.exhausted", queue_size_after=0)
        return action

    # MARK: - Training boundary (runtime-only)

    def forward(self, batch: dict[str, Any]) -> Any:
        raise RuntimeError(
            "coreai_bridge is runtime-only; it has no training forward. "
            "Train with LeRobot, run with CoreAI.")

    def get_optim_params(self) -> Any:
        raise RuntimeError("coreai_bridge is runtime-only; it exposes no optimizer params.")

    def train(self, mode: bool = True):
        if mode:
            raise RuntimeError(
                "coreai_bridge is runtime-only; it cannot enter training mode.")
        return super().train(False)
