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
    action_horizon: int = 1
    # Cross-binding expectations verified against the loaded CoreAI manifest.
    expected_robot_type: str | None = None
    expected_action_dim: int | None = None
    expected_action_horizon: int | None = None
    # Batch handling: "single_only" (v1.3.1) raises clearly on B>1;
    # "split_and_stack" is reserved for v1.3.2.
    batch_mode: str = "single_only"
    # The CoreAI runtime device is separate from the torch host device. The
    # inherited `device` (default from PreTrainedConfig) stays a real torch
    # device so make_policy's `policy.to(cfg.device)` works; runtime_device
    # records that inference actually runs on CoreAI.
    runtime_device: str = "coreai"

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
        # Feature validation is delegated to the CoreAI manifest / obs-bridge; the
        # official plugin does not re-impose LeRobot feature constraints here.
        return None

    def get_optimizer_preset(self):
        raise NotImplementedError(
            "coreai_bridge is runtime-only; train with LeRobot, run with CoreAI.")

    def get_scheduler_preset(self):
        return None
