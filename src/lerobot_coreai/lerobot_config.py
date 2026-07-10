# lerobot_config.py — LeRobot-shaped config for the CoreAI bridge (v1.1.0).
#
# A small, honest config object shaped like the attributes LeRobot code tends to
# read off a policy's `.config` (policy_type, device, input/output features). It
# is derived entirely from the CoreAI manifest. It is NOT a LeRobot
# `PreTrainedConfig` subclass and is not registered upstream.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .manifest import LeRobotCoreAIManifest

# The bridge's policy type. Deliberately NOT "coreai" — there is no upstream
# LeRobot registry entry for either name; the "_bridge" suffix makes the local,
# non-native nature explicit anywhere this string surfaces.
BRIDGE_POLICY_TYPE = "coreai_bridge"


@dataclass
class CoreAIBridgeConfig:
    """LeRobot-shaped, read-only config derived from a CoreAI manifest.

    Runtime-only. `training_supported` and `native_registry` are always False and
    are part of the object so no caller can mistake this for a trainable,
    upstream-registered policy config.
    """

    policy_type: str = BRIDGE_POLICY_TYPE
    robot_type: str | None = None
    device: str = "coreai"
    input_features: dict[str, Any] = field(default_factory=dict)
    output_features: dict[str, Any] = field(default_factory=dict)
    training_supported: bool = False
    native_registry: bool = False

    @classmethod
    def from_manifest(cls, manifest: LeRobotCoreAIManifest) -> "CoreAIBridgeConfig":
        input_features = {
            name: {"dtype": f.dtype,
                   "shape": list(f.shape) if f.shape is not None else None,
                   "required": f.required}
            for name, f in manifest.observation_features.items()
        }
        output_features = {
            name: {"dtype": f.dtype,
                   "shape": list(f.shape) if f.shape is not None else None}
            for name, f in manifest.action_features.items()
        }
        return cls(
            policy_type=BRIDGE_POLICY_TYPE,
            robot_type=manifest.robot_type,
            device="coreai",
            input_features=input_features,
            output_features=output_features,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_type": self.policy_type,
            "robot_type": self.robot_type,
            "device": self.device,
            "input_features": self.input_features,
            "output_features": self.output_features,
            "training_supported": self.training_supported,
            "native_registry": self.native_registry,
        }
