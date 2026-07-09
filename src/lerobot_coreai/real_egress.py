# real_egress.py — the guarded real-egress gate (v1.0.0).
#
# RealEgressGuard is the ONLY path from a generated action to a RobotAdapter.
# It is fail-closed: session state, deadman, e-stop, rate limit, adapter
# readiness, and the enforced SafetySupervisor must ALL pass before an action is
# handed to the adapter. A blocked action never reaches the adapter.

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

from .safety_supervisor import SafetyContext, SafetySupervisor, safe_evaluate


@dataclass
class DeadmanSwitch:
    """Software deadman. NOT a substitute for a physical emergency stop."""

    timeout_s: float = 1.0
    last_heartbeat: float | None = None
    enabled: bool = True

    def heartbeat(self, now: float | None = None) -> None:
        self.last_heartbeat = time.monotonic() if now is None else now

    def healthy(self, now: float | None = None) -> bool:
        if not self.enabled:
            # A disabled deadman is only ever allowed for the mock adapter; the
            # guard enforces that. Disabled here means "not a live safety layer".
            return True
        if self.last_heartbeat is None:
            return False
        t = time.monotonic() if now is None else now
        return (t - self.last_heartbeat) <= self.timeout_s


@dataclass
class RateLimiter:
    fps: float
    last_send_time: float | None = None

    def allow(self, now: float | None = None) -> bool:
        t = time.monotonic() if now is None else now
        if self.fps <= 0:
            return False
        if self.last_send_time is None:
            self.last_send_time = t
            return True
        if (t - self.last_send_time) >= (1.0 / self.fps):
            self.last_send_time = t
            return True
        return False


@dataclass
class RealEgressDecision:
    allowed: bool
    reason: str
    step: int
    supervisor_decision: dict[str, Any] | None = None
    estop_state: str = "clear"
    action_hash: str | None = None
    sent: bool = False
    adapter_result: dict[str, Any] | None = None


def _action_hash(action: Any) -> str:
    try:
        blob = json.dumps(action, sort_keys=True, default=str).encode()
    except Exception:
        blob = repr(action).encode()
    return "sha256:" + hashlib.sha256(blob).hexdigest()


class RealEgressGuard:
    """Fail-closed gate between a supervised action and the robot adapter."""

    def __init__(self, supervisor: SafetySupervisor, adapter, session,
                 deadman: DeadmanSwitch, rate_limiter: RateLimiter, trace=None,
                 allow_disabled_deadman: bool = False):
        self.supervisor = supervisor
        self.adapter = adapter
        self.session = session
        self.deadman = deadman
        self.rate_limiter = rate_limiter
        self.trace = trace
        # Defence-in-depth: a disabled deadman is only ever tolerated when the
        # caller explicitly authorized it (mock only, per real_mode). Otherwise
        # the guard blocks even if a disabled deadman is handed in.
        self.allow_disabled_deadman = allow_disabled_deadman
        self.estop_triggered = False
        self.deadman_lost = False
        self.adapter_error: str | None = None

    def trigger_estop(self, reason: str = "operator") -> None:
        self.estop_triggered = True
        self._trace("real.estop.triggered", {"reason": reason})

    def _trace(self, event: str, data: dict[str, Any] | None = None) -> None:
        if self.trace is not None:
            self.trace.write(event, data or {})

    def send_action(self, raw_action: Any, context: SafetyContext,
                    now: float | None = None) -> RealEgressDecision:
        step = context.step if context and context.step is not None else -1
        ah = _action_hash(raw_action)

        def _block(reason: str, sup: dict | None = None) -> RealEgressDecision:
            self._trace("real.egress.blocked", {"step": step, "reason": reason})
            return RealEgressDecision(
                allowed=False, reason=reason, step=step, supervisor_decision=sup,
                estop_state="triggered" if self.estop_triggered else "clear",
                action_hash=ah, sent=False)

        # 1. session must be running (session may be a dict or an object).
        status = (self.session.get("status") if isinstance(self.session, dict)
                  else getattr(self.session, "status", None))
        if status != "running":
            return _block("session_not_running")
        # 2. e-stop must not be triggered.
        if self.estop_triggered:
            return _block("estop_triggered")
        # 3. deadman: a disabled deadman is only allowed when explicitly
        #    authorized (mock); otherwise it must be enabled AND healthy.
        if not self.deadman.enabled:
            if not self.allow_disabled_deadman:
                return _block("deadman_disabled_not_allowed")
        elif not self.deadman.healthy(now):
            self.deadman_lost = True
            self._trace("real.deadman.lost", {"step": step})
            return _block("deadman_unhealthy")
        # 4. rate limiter must allow.
        if not self.rate_limiter.allow(now):
            return _block("rate_limited")
        # 5. adapter must be ready.
        try:
            ready = self.adapter.is_ready()
        except Exception as e:  # fail-closed
            return _block(f"adapter_readiness_error:{e}")
        if not ready:
            return _block("adapter_not_ready")
        # 6. supervisor must allow (fail-closed on any internal error).
        supervised = safe_evaluate(self.supervisor, raw_action, context=context)
        sup_dict = supervised.decision.to_dict()
        self._trace("real.supervisor.decision", {
            "step": step, "allowed": supervised.decision.allowed,
            "reasons": supervised.decision.reasons, "action_hash": ah})
        if not supervised.decision.allowed:
            # CRITICAL: a blocked action never reaches the adapter.
            return _block("supervisor_blocked", sup=sup_dict)

        # 7. Allowed → hand the supervised action to the adapter.
        try:
            result = self.adapter.send_action(supervised.action)
        except Exception as e:  # any adapter error stops the session, fail-closed
            self.adapter_error = str(e)
            self._trace("real.egress.adapter_error", {"step": step, "error": str(e)})
            return _block(f"adapter_error:{e}", sup=sup_dict)

        self._trace("real.egress.sent", {"step": step, "action_hash": ah})
        return RealEgressDecision(
            allowed=True, reason="sent", step=step, supervisor_decision=sup_dict,
            estop_state="clear", action_hash=ah, sent=True, adapter_result=result)
