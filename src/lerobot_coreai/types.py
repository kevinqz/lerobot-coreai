# types.py — typed request/response dataclasses for the runner protocol (v0.2).

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class RunnerHealth:
    """Response from GET /v1/health."""
    status: str
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class RunnerCapabilities:
    """Response from GET /v1/capabilities."""
    runtime: str = "coreai-runner"
    supports_action: bool = False
    supports_llm: bool = False
    supports_vlm: bool = False
    supports_host_loop: bool = False
    supports_multi_graph: bool = False
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class ActionPredictRequest:
    """Request for POST /v1/predict with runtime_kind=action.

    The observation dict preserves LeRobot feature names as keys:
        observation.images.wrist, observation.state, task, etc.
    """
    model_id: str
    runtime_kind: Literal["action"] = "action"
    observation: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionPredictResponse:
    """Response from a successful action predict call.

    ``action`` is typically a list of lists (action chunk [[float]]).
    """
    action: Any
    action_features: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, float] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)
