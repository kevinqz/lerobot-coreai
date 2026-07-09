# action_blocker.py — explicit action egress blocking for shadow mode (v0.7).
#
# Shadow mode generates actions, validates them, logs them, and blocks all egress.
# This module makes that contract explicit: ActionBlocker.send() always raises.
# No object here may forward an action to a robot, motor, simulator, or actuator.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .errors import SafetyError


@dataclass(frozen=True)
class BlockedAction:
    """An action that was generated, validated, and blocked from egress.

    Invariants:
        - sent is always False.
        - destination is always "none".
        - The action value is preserved for logging/audit.
    """

    action: Any
    reason: str
    mode: str = "shadow"
    sent: bool = False
    destination: str = "none"


@dataclass
class ActionBlocker:
    """Records actions and blocks all egress.

    The shadow loop calls block() for every generated action. send() is the only
    egress path and it unconditionally raises SafetyError — it exists so that
    *if* anyone ever wires up an actuation device, the call fails loudly.

    Invariants:
        - blocked_count increments on every block().
        - sent_count is always 0 (send() never returns).
        - actions_sent is always 0.
    """

    mode: str = "shadow"
    blocked_count: int = 0
    sent_count: int = 0
    reasons: list[str] = field(default_factory=list)

    def block(self, action: Any, *, reason: str = "shadow_mode_no_actuation") -> BlockedAction:
        """Record an action as blocked. Returns a BlockedAction (sent=False)."""
        self.blocked_count += 1
        if reason not in self.reasons:
            self.reasons.append(reason)
        return BlockedAction(action=action, reason=reason, mode=self.mode)

    def send(self, action: Any) -> None:
        """Egress path — always disabled in shadow mode.

        Raises:
            SafetyError: Always. No robot commands were sent.
        """
        raise SafetyError(
            "Action egress is disabled in shadow mode. No robot commands were sent."
        )

    @property
    def actions_sent(self) -> int:
        """Always 0. Shadow mode never sends."""
        return 0
