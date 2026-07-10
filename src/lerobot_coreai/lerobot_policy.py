# lerobot_policy.py — LeRobot-shaped policy bridge over CoreAIPolicy (v1.1.0).
#
# A duck-typed, runtime-only wrapper that exposes a CoreAI-backed policy through
# the small surface LeRobot code expects from a policy object: select_action,
# eval/train/to, reset, and a `.config`. It does NOT subclass
# lerobot PreTrainedPolicy, does NOT import torch or lerobot, and is NOT
# registered in the upstream LeRobot policy registry/factory. Inference is
# runtime-only; training must be done with LeRobot.

from __future__ import annotations

from typing import Any

from .errors import CoreAIPolicyError
from .lerobot_config import BRIDGE_POLICY_TYPE, CoreAIBridgeConfig
from .policy import CoreAIPolicy


class CoreAILeRobotPolicyBridge:
    """LeRobot-shaped CoreAI policy bridge.

    Not an upstream LeRobot policy registry entry. Not a training policy.
    Runtime-only. Wraps a :class:`CoreAIPolicy` and forwards inference to it.
    """

    #: Deliberately not "coreai"; there is no upstream registry entry.
    policy_type = BRIDGE_POLICY_TYPE

    def __init__(self, coreai_policy: CoreAIPolicy):
        self.coreai_policy = coreai_policy
        self.config = CoreAIBridgeConfig.from_manifest(coreai_policy.manifest)
        self.device = "coreai"

    # MARK: - Inference (LeRobot-shaped)

    def select_action(self, batch: dict[str, Any], **kwargs: Any) -> Any:
        """Return the raw action for ``batch`` (LeRobot 0.6.x semantics)."""
        return self.coreai_policy.select_action(batch, **kwargs)

    def predict_action(self, batch: dict[str, Any], *, return_metadata: bool = True,
                       **kwargs: Any) -> dict[str, Any]:
        """Return the richer ``{"action": ..., "metadata": ...}`` dict helper."""
        return self.coreai_policy.predict_action(
            batch, return_metadata=return_metadata, **kwargs)

    # MARK: - Lifecycle (LeRobot-shaped no-ops / guards)

    def reset(self) -> None:
        """Reset internal state. Forwards to the CoreAI policy (usually a no-op)."""
        self.coreai_policy.reset()

    def eval(self) -> "CoreAILeRobotPolicyBridge":
        """Inference mode. Always returns self (already inference-only)."""
        return self

    def train(self, mode: bool = True) -> "CoreAILeRobotPolicyBridge":
        """Runtime-only bridge: entering train mode is an error.

        Accepts LeRobot's ``train(mode: bool)`` signature. ``train(False)`` (i.e.
        eval) is allowed; ``train(True)`` raises — train with LeRobot.
        """
        if mode:
            raise CoreAIPolicyError(
                "CoreAI bridge is runtime-only; it cannot enter training mode. "
                "Train with LeRobot, then run with CoreAI.")
        return self

    def to(self, *args: Any, **kwargs: Any) -> "CoreAILeRobotPolicyBridge":
        """Documented no-op. CoreAI always runs on the CoreAI device.

        LeRobot code commonly calls ``policy.to(device)`` with a torch device
        (e.g. "cuda"/"cpu"). The bridge accepts and ignores any device argument
        and returns self, so LeRobot-shaped loops don't break; inference still
        runs on the CoreAI runtime regardless.
        """
        return self

    # MARK: - Accessors

    @property
    def manifest(self):
        return self.coreai_policy.manifest

    @property
    def robot_type(self) -> str:
        return self.coreai_policy.robot_type

    def metadata(self) -> dict[str, Any]:
        """Honest bridge metadata — never claims native registry or training."""
        return {
            "runtime": "coreai",
            "policy_type": self.policy_type,
            "robot_type": self.robot_type,
            "training_supported": False,
            "native_registry": False,
            "bridge": "local",
        }

    def __repr__(self) -> str:
        return (f"CoreAILeRobotPolicyBridge(policy_type={self.policy_type!r}, "
                f"robot={self.robot_type!r}, runtime='coreai')")
