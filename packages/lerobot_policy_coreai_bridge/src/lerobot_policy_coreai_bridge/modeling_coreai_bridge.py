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
from collections import deque
from pathlib import Path
from typing import Any

import torch

from lerobot.policies.pretrained import PreTrainedPolicy

from .configuration_coreai_bridge import POLICY_TYPE, CoreAIBridgeConfig


class PluginBindingError(RuntimeError):
    """Raised when the CoreAI runtime cannot be bound or fails cross-binding."""


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
        self._queue: deque = deque()
        self.last_observation_sha256: str | None = None
        self.last_observation_audit: dict[str, Any] = {}
        self._action_contract = None
        self._protocol = None
        if coreai_policy is not None:
            self._bind_action_contract(coreai_policy)
        self.register_buffer("_sentinel", torch.zeros(1), persistent=False)

    def _bind_action_contract(self, coreai_policy: Any) -> None:
        """Parse the manifest action contract, fail-closed (v1.3.5).

        A malformed contract must not silently degrade to chunk/no-horizon
        validation; it is a binding error.
        """
        manifest = getattr(coreai_policy, "manifest", None)
        if manifest is None:
            # No manifest bound (local/test path): nothing to fail closed on.
            self._action_contract = None
            return
        from lerobot_coreai.action_contract import parse_action_contract_from_manifest
        try:
            self._action_contract = parse_action_contract_from_manifest(manifest)
        except Exception as exc:
            raise PluginBindingError(
                f"invalid CoreAI action contract: {exc}") from exc

    def _resolve_protocol(self):
        """Negotiate the full runner protocol with the bound runner (once).

        Real path (runner present): capabilities are fetched and any
        capabilities/transport/protocol failure PROPAGATES — it is never turned
        into a silent legacy fallback. Legacy is used only when the config opts in
        via ``allow_legacy_runner_protocol``.

        Local/test path (no runner bound): there is no wire to negotiate, so the
        config default encoding and the minimum protocol are used. This is not a
        capabilities failure.
        """
        if self._protocol is not None:
            return self._protocol
        from .negotiation import NegotiatedRunnerProtocol, negotiate_runner_protocol
        runner = getattr(self.coreai_policy, "runner", None)
        if runner is None or not hasattr(runner, "capabilities"):
            self._protocol = NegotiatedRunnerProtocol(
                protocol_version=self.config.minimum_runner_protocol,
                observation_encoding=self.config.observation_encoding_or_default(),
                supports_batch=False, max_batch_size=1, legacy=True)
            return self._protocol
        caps = runner.capabilities()  # propagate failures — no silent legacy
        self._protocol = negotiate_runner_protocol(
            requested_encoding=self.config.observation_encoding,
            capabilities=caps,
            minimum_protocol=self.config.minimum_runner_protocol,
            allow_legacy=self.config.allow_legacy_runner_protocol)
        return self._protocol

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
        if config.expected_action_horizon is not None and \
                contract.horizon != config.expected_action_horizon:
            raise PluginBindingError(
                f"action_horizon mismatch: manifest {contract.horizon} != expected "
                f"{config.expected_action_horizon}.")
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

    # MARK: - Inference (LeRobot contract)

    def reset(self) -> None:
        self._queue.clear()
        # Invalidate the negotiated protocol: a runner may restart with different
        # capabilities, so re-negotiate on the next inference (v1.3.5).
        self._protocol = None
        if self.coreai_policy is not None and hasattr(self.coreai_policy, "reset"):
            self.coreai_policy.reset()

    def _require_single_batch(self, batch: dict[str, Any]) -> None:
        b = _infer_batch_size(batch)
        if b > 1 and self.config.batch_mode == "single_only":
            raise PluginBindingError(
                f"coreai_bridge supports only batch_size=1 (got {b}); "
                "batched evaluation lands in v1.3.6.")

    def _tensor(self, action: Any) -> torch.Tensor:
        t = torch.as_tensor(action, dtype=torch.float32, device=self._sentinel.device)
        if t.ndim == 1:
            t = t.unsqueeze(0)  # (action_dim,) -> (1, action_dim)
        return t

    def _coreai_observation(self, batch: dict[str, Any]) -> tuple[dict[str, Any], str]:
        """Bridge a LeRobot batch to a single JSON-safe CoreAI observation."""
        from .transport import prepare_single_coreai_observation
        manifest = getattr(self.coreai_policy, "manifest", None)
        audit: dict[str, Any] = {}
        proto = self._resolve_protocol()
        obs, sha = prepare_single_coreai_observation(
            batch, manifest, encoding=proto.observation_encoding, audit=audit)
        self.last_observation_sha256 = sha
        self.last_observation_audit = audit
        return obs, sha

    def predict_action_chunk(self, batch: dict[str, Any], **kwargs: Any) -> torch.Tensor:
        if self.coreai_policy is None:
            raise PluginBindingError("coreai_bridge has no CoreAI policy bound.")
        self._require_single_batch(batch)
        obs, sha = self._coreai_observation(batch)
        proto = self._resolve_protocol()
        runner_options = {
            # The negotiated protocol/encoding — never a hardcoded constant.
            "protocol_version": proto.protocol_version,
            "observation_encoding": proto.observation_encoding,
            "observation_schema_version": "coreai-observation.v1",
            "observation_sha256": sha,
        }
        # No TypeError fallback (v1.3.5): the companion requires
        # lerobot-coreai>=1.3.5, whose predict_action_chunk accepts runner_options.
        # A TypeError raised inside inference must not be mistaken for a signature
        # mismatch and silently re-run (which could double-advance a stateful runner).
        raw = self.coreai_policy.predict_action_chunk(obs, runner_options=runner_options)

        # Strict normalization + validation, using the manifest action contract
        # for representation and horizon when known.
        from .action_validation import normalize_and_validate_action_chunk
        c = self._action_contract
        rep = c.representation if c is not None else "chunk"
        horizon = c.horizon if (c is not None and c.representation == "chunk") else None
        dim = self.config.expected_action_dim or (c.action_dim if c is not None else None)
        return normalize_and_validate_action_chunk(
            raw, representation=rep, expected_batch_size=1,
            expected_horizon=horizon, expected_action_dim=dim,
            device=self._sentinel.device)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Any], **kwargs: Any) -> torch.Tensor:
        # Single boundary: select_action ALWAYS goes through predict_action_chunk.
        if not self._queue:
            chunk = self.predict_action_chunk(batch, **kwargs)  # (B, H, A), B=1
            for row in chunk[0]:  # enqueue each timestep [A]
                self._queue.append(row)
        return self._tensor(self._queue.popleft())

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
