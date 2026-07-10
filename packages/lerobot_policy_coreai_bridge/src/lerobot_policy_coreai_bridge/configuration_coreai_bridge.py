# configuration_coreai_bridge.py — official LeRobot config for the CoreAI bridge.
#
# Registered under policy_type "coreai_bridge" via the official
# PreTrainedConfig.register_subclass mechanism — NOT "coreai" (that name is not,
# and should not appear to be, an upstream policy). Runtime-only: it declares no
# training presets and no temporal delta indices.

from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig

POLICY_TYPE = "coreai_bridge"


@PreTrainedConfig.register_subclass(POLICY_TYPE)
@dataclass
class CoreAIBridgeConfig(PreTrainedConfig):
    """LeRobot config for a CoreAI-backed bridge policy (runtime-only)."""

    coreai_artifact: str = ""
    coreai_revision: str | None = None
    runner_url_env: str = "COREAI_RUNNER_URL"
    # Cross-binding expectations verified against the loaded CoreAI manifest.
    expected_robot_type: str | None = None
    expected_action_dim: int | None = None
    expected_action_horizon: int | None = None
    # Deprecated alias for expected_action_horizon (v1.3.6). Kept for back-compat;
    # a contradictory explicit value fails in __post_init__. Prefer
    # expected_action_horizon as the single source of truth.
    action_horizon: int = 1
    # Batch handling (v1.3.9): "single_only" | "native_batch" | "split_and_stack"
    # | "auto". B>1 requires a batch-capable, safely-scoped runner.
    batch_mode: str = "single_only"
    # Optional client-side cap; the effective max is min(artifact, config, runner).
    max_batch_size: int | None = None
    # Observation transport encoding: "auto" (default -> nested_json_v1),
    # "nested_json_v1", or "typed_array_envelope_v1" (only if the runner announces it).
    observation_encoding: str = "auto"
    # Runner protocol binding mode (v1.3.6), replacing the ambiguous boolean pair:
    #   "strict"    — a runner is REQUIRED; capabilities are fetched and the
    #                 protocol is negotiated fail-closed (any failure propagates).
    #   "legacy"    — a runner is required, but a runner that announces no
    #                 protocol_version is accepted as legacy nested_json_v1.
    #   "in_memory" — NO wire boundary; for local/test binding against an
    #                 in-process CoreAI policy with no RunnerClient. Never used by
    #                 the official from_pretrained path (which always binds a runner).
    runtime_binding_mode: str = "strict"
    minimum_runner_protocol: str = "coreai-runner.v2"
    # The CoreAI runtime device is separate from the torch host device. The
    # inherited `device` (default from PreTrainedConfig) stays a real torch
    # device so make_policy's `policy.to(cfg.device)` works; runtime_device
    # records that inference actually runs on CoreAI.
    runtime_device: str = "coreai"

    _VALID_BINDING_MODES = ("strict", "legacy", "in_memory")

    def __post_init__(self):
        parent_post = getattr(super(), "__post_init__", None)
        if callable(parent_post):
            parent_post()
        if self.runtime_binding_mode not in self._VALID_BINDING_MODES:
            raise ValueError(
                f"runtime_binding_mode must be one of {self._VALID_BINDING_MODES}, "
                f"got {self.runtime_binding_mode!r}.")
        # Reconcile the deprecated action_horizon alias with the source of truth.
        if (self.expected_action_horizon is not None
                and self.action_horizon != 1
                and self.action_horizon != self.expected_action_horizon):
            raise ValueError(
                f"contradictory horizons: action_horizon={self.action_horizon} != "
                f"expected_action_horizon={self.expected_action_horizon}; set only "
                "expected_action_horizon.")

    def effective_action_horizon(self) -> int | None:
        """The single horizon source of truth (expected_action_horizon wins)."""
        if self.expected_action_horizon is not None:
            return self.expected_action_horizon
        return self.action_horizon if self.action_horizon != 1 else None

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> None:
        return None

    @property
    def reward_delta_indices(self) -> None:
        return None

    def validate_features(self) -> None:
        """Validate populated input/output features against declared expectations.

        make_policy fills ``input_features``/``output_features`` from the dataset
        or env before ``from_pretrained``. The full manifest cross-binding lives in
        the policy (which has the loaded manifest); here we enforce the invariants
        that need only the config: an action output feature must exist, and if
        ``expected_action_dim`` is declared it must match the per-timestep action
        feature's last dimension (horizon is NOT part of the per-timestep feature).
        """
        if not self.output_features:
            return  # not yet populated (e.g. constructed standalone) — nothing to check
        from lerobot.configs.types import FeatureType
        action_feats = {k: f for k, f in self.output_features.items()
                        if getattr(f, "type", None) == FeatureType.ACTION}
        if not action_feats:
            raise ValueError(
                "coreai_bridge requires an ACTION output feature; none present.")
        if self.expected_action_dim is not None:
            for k, f in action_feats.items():
                shape = tuple(getattr(f, "shape", ()) or ())
                if shape and shape[-1] != self.expected_action_dim:
                    raise ValueError(
                        f"output feature {k!r} action dim {shape[-1]} != "
                        f"expected_action_dim {self.expected_action_dim}.")

    def get_optimizer_preset(self):
        raise NotImplementedError(
            "coreai_bridge is runtime-only; train with LeRobot, run with CoreAI.")

    def get_scheduler_preset(self):
        return None

    def observation_encoding_or_default(self) -> str:
        """Resolve "auto" to the safe default (nested_json_v1)."""
        return "nested_json_v1" if self.observation_encoding == "auto" \
            else self.observation_encoding
