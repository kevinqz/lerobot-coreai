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
    # v1.3.4: protocol + observation encoding negotiation.
    protocol_version: str | None = None
    observation_encodings: tuple[str, ...] = ()
    supports_batch: bool = False
    max_batch_size: int | None = None
    # v1.3.6: protocols the runner declares itself backward-compatible with, so a
    # newer major (e.g. coreai-runner.v3) can still be accepted by a v2 plugin.
    backward_compatible_with: tuple[str, ...] = ()
    # v1.3.8: batch protocol foundation. v1.3.9: state_isolation gates native B>1.
    # None means "not announced" (fail-closed for B>1).
    action_batching_semantics: str | None = None   # "native" | "split_and_stack"
    action_batching_state_isolation: str | None = None  # stateless|request_scoped|per_slot|shared|unknown
    inference_state_scope: str | None = None        # stateless|request_scoped|session_scoped|global
    supports_session_ids: bool = False
    reset_scope: str | None = None                   # none|request|session|all_sessions|global
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
