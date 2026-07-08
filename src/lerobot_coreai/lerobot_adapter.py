# lerobot_adapter.py — isolated adapter for loading PyTorch LeRobot policies (v0.5).
#
# All LeRobot/torch imports are isolated here. The core package never imports this module
# at the top level. This module is only used by compare.py when the [lerobot] extra is installed.

from __future__ import annotations

from typing import Any

from .dataset import require_lerobot
from .errors import CoreAIPolicyError


def load_lerobot_policy(policy_path: str, *, policy_type: str | None = None) -> Any:
    """Load a source PyTorch LeRobot policy from a Hugging Face repo.

    Uses LeRobot 0.6.x public APIs. The loading strategy tries the canonical factory
    path first, falling back to direct PreTrainedPolicy loading.

    Args:
        policy_path: HF repo id (e.g. 'lerobot/evo1_so100') or local path.
        policy_type: Optional policy type override (e.g. 'act', 'pi0').

    Returns:
        A LeRobot policy object with a ``select_action(batch)`` method.

    Raises:
        CoreAIPolicyError: If LeRobot is not installed or the policy can't be loaded.
    """
    require_lerobot()

    # Strategy 1: LeRobot 0.6.x factory API (if available).
    try:
        from lerobot.common.utils.utils import get_global_random_state  # type: ignore[import-not-found]  # noqa: F401
        # Try the factory path that LeRobot uses for pretrained policies.
        try:
            from lerobot.policies.factory import make_policy  # type: ignore[import-not-found]
            from lerobot.common.configs.parser import load_hf_config  # type: ignore[import-not-found]

            # Load policy config from HF and instantiate.
            policy = make_policy(policy_type or "act", dataset_stats=None, pretrained_path=policy_path)
            return policy
        except (ImportError, AttributeError, Exception):
            pass
    except ImportError:
        pass

    # Strategy 2: Direct PreTrainedPolicy loading.
    try:
        from lerobot.common.policies.pretrained import PreTrainedPolicy  # type: ignore[import-not-found]
        return PreTrainedPolicy.from_pretrained(policy_path)
    except ImportError:
        pass

    # Strategy 3: Try lerobot.policies import path.
    try:
        import lerobot  # type: ignore[import-not-found]
        # Some versions have a top-level helper.
        if hasattr(lerobot, "load_policy"):
            return lerobot.load_policy(policy_path)
    except (ImportError, AttributeError):
        pass

    raise CoreAIPolicyError(
        f"Could not load source LeRobot policy from '{policy_path}'. "
        f"v0.5 compare requires a LeRobot 0.6.x-compatible PyTorch policy loader. "
        f"Install `pip install \"lerobot-coreai[lerobot]\"` with Python 3.12+. "
        f"If the policy type is known, pass --torch.policy.type.\n"
        f"No robot commands were sent."
    )


def make_torch_policy_batch(batch: dict[str, Any]) -> dict[str, Any]:
    """Convert a raw dataset batch for PyTorch policy consumption.

    Preserves tensors and images from the dataset item (not the JSON-safe version).
    Converts numeric lists to tensors when needed by the PyTorch policy.

    Args:
        batch: Raw observation batch from dataset_item_to_observation_batch.

    Returns:
        A batch suitable for ``torch_policy.select_action()``.
    """
    result: dict[str, Any] = {}

    for key, value in batch.items():
        # Strings (task, image paths) — preserve.
        if isinstance(value, str):
            result[key] = value
            continue

        # Already a tensor — preserve.
        if hasattr(value, "detach"):
            result[key] = value
            continue

        # Numeric lists — preserve (the PyTorch policy's preprocessor will handle conversion).
        result[key] = value

    return result
