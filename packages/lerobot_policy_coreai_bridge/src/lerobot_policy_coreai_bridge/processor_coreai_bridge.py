# processor_coreai_bridge.py — official processor factory for the CoreAI bridge.
#
# Follows LeRobot's make_<name>_pre_post_processors naming so the official
# factory can construct pre/post processors for this policy type. The CoreAI
# runner owns its own normalization by default, so the bridge's default
# processors are identity pass-throughs; a real artifact can bind normalization
# stats via its processor_contract (see lerobot-coreai manifest contracts v1).

from __future__ import annotations

from typing import Any


class _IdentityProcessor:
    """A minimal identity processor: returns its input unchanged."""

    def __init__(self, name: str):
        self.name = name

    def __call__(self, data: Any) -> Any:
        return data

    def reset(self) -> None:  # stateful-processor interface parity
        pass


def make_coreai_bridge_pre_post_processors(config: Any, dataset_stats: Any = None):
    """Return (preprocessor, postprocessor) for the CoreAI bridge policy.

    Identity by default — the CoreAI runner is declared to own normalization via
    the manifest processor_contract. Signature matches the official factory
    convention ``make_<name>_pre_post_processors(config, dataset_stats=None)``.
    """
    return _IdentityProcessor("coreai_bridge_pre"), _IdentityProcessor("coreai_bridge_post")
