# config.py — runtime and policy configuration dataclasses (spec §10.3, §10.4).

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class CoreAIRuntimeConfig:
    """Runtime configuration for connecting to coreai-runner.

    Default mode is dry_run (no actuation). Real mode requires explicit confirmation
    in the rollout command (--confirm-real-robot-actuation).
    """
    type: Literal["coreai"] = "coreai"
    runner_url: str = "unix:///tmp/coreai-runner.sock"
    endpoint: str | None = None  # remote runner (e.g. http://mac-studio.local:8710)
    download: Literal["auto", "never", "force"] = "auto"
    mode: Literal["dry_run", "shadow", "sim", "real"] = "dry_run"
    timeout_s: float = 30.0


@dataclass
class CoreAIPolicyConfig:
    """Configuration for a CoreAI-backed LeRobot policy.

    Derived from the lerobot-coreai.json manifest at load time.
    """
    type: Literal["coreai"] = "coreai"
    path: str = ""  # HF repo id of the artifact
    policy_type: str = ""  # e.g. 'act', 'pi0', 'evo1'
    robot_type: str = ""  # e.g. 'so100', 'so101'
    observation_features: dict = field(default_factory=dict)
    action_features: dict = field(default_factory=dict)
    runtime: CoreAIRuntimeConfig = field(default_factory=CoreAIRuntimeConfig)
