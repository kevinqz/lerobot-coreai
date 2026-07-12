# app_conformance.py — Apple-app conformance reference (RFC-0700 §19.1 / RFC-0900 §11.3,
# LR12). `lerobot-coreai` publishes the Python REFERENCE + fixtures that the Swift
# `LeRobotCoreAIKit` (in lerobot-coreai-apple) must reproduce — it does NOT contain the
# SwiftUI app. Two semantics are pinned here so Swift and Python agree:
#
#   1. ActionQueue: a Core AI policy emits a CHUNK, but LeRobot per-timestep semantics are
#      preserved — the queue hands out one action per step and INVALIDATES the chunk on any
#      identity/contract change, gateway reset, deadline expiry, stop or fault.
#   2. Planner→policy handoff: a validated SkillPlan becomes a task string for the policy;
#      the language model NEVER emits actions/motor values (RFC-0900 §14.3).
#
# Pure Python; no torch/lerobot.

from __future__ import annotations

from dataclasses import dataclass


class ActionQueueEmpty(Exception):
    """next() with no valid action — the caller must not actuate."""


class ChunkInvalidated(Exception):
    """The current chunk was invalidated; a fresh inference is required."""


# the exact set of reasons a chunk MUST be invalidated (RFC-0900 §11.3).
INVALIDATION_REASONS = (
    "session_change", "gateway_reset", "observation_contract_change",
    "policy_artifact_change", "deadline_expired", "stop", "fault")


@dataclass(frozen=True)
class QueueContext:
    session_id: str
    policy_artifact_root: str
    observation_contract_sha: str


class ReferenceActionQueue:
    """Reference for LeRobotCoreAIKit's Swift `ActionQueue`. Same inputs → same outputs."""

    def __init__(self) -> None:
        self._actions: list = []
        self._i = 0
        self._ctx: QueueContext | None = None
        self._deadline_ns: int | None = None
        self._invalidated: str | None = None

    def replace(self, chunk_actions: list, ctx: QueueContext, deadline_ns: int) -> None:
        self._actions = list(chunk_actions)
        self._i = 0
        self._ctx = ctx
        self._deadline_ns = deadline_ns
        self._invalidated = None

    def clear(self, reason: str) -> None:
        if reason not in INVALIDATION_REASONS:
            raise ValueError(f"unknown invalidation reason {reason!r}")
        self._actions = []
        self._i = 0
        self._invalidated = reason

    def observe(self, ctx: QueueContext) -> None:
        """A new context that differs from the chunk's context invalidates it."""
        if self._ctx is None:
            return
        if ctx.session_id != self._ctx.session_id:
            self.clear("session_change")
        elif ctx.policy_artifact_root != self._ctx.policy_artifact_root:
            self.clear("policy_artifact_change")
        elif ctx.observation_contract_sha != self._ctx.observation_contract_sha:
            self.clear("observation_contract_change")

    def next(self, now_ns: int):
        if self._invalidated:
            raise ChunkInvalidated(self._invalidated)
        if self._deadline_ns is not None and now_ns > self._deadline_ns:
            self.clear("deadline_expired")
            raise ChunkInvalidated("deadline_expired")
        if self._i >= len(self._actions):
            raise ActionQueueEmpty("no action available; refill required")
        a = self._actions[self._i]
        self._i += 1
        return a


# --- planner → policy handoff (Gemma boundary, RFC-0900 §14.3) ---

_MOTOR_KEYS = ("joint", "torque", "pwm", "position", "velocity", "action", "motor")


def skill_plan_to_policy_tasks(skill_plan: dict) -> list[str]:
    """Turn a validated SkillPlan into per-step task strings for the LeRobot policy. The
    planner supplies GOALS, never motor commands — any motor-ish argument key is refused
    (the LM must not steer actuators)."""
    if skill_plan.get("schema") != "org.lerobot.robot-brain.skill-plan.v1":
        raise ValueError("not an org.lerobot.robot-brain.skill-plan.v1 plan")
    tasks: list[str] = []
    for step in skill_plan.get("steps", []):
        args = step.get("arguments", {})
        for k in args:
            if any(m in str(k).lower() for m in _MOTOR_KEYS):
                raise ValueError(
                    f"planner emitted a motor-control argument {k!r} — the language "
                    "model must produce skills/goals only, never actuator commands")
        goal = ", ".join(f"{k}={v}" for k, v in sorted(args.items()))
        tasks.append(f"{step['skill']}: {goal}" if goal else step["skill"])
    return tasks
