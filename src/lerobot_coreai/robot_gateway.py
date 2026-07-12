# robot_gateway.py — Robot Gateway reference (RFC-0700 §19.1 / RFC-0900 §13, LR10).
#
# The gateway is the final SOFTWARE authority before hardware egress. It is DISTINCT from
# the CoreAI Runner (inference) and the Server (LAN exposure): those never own motor
# egress. This is the Python reference implementation of the `org.lerobot.robot-gateway.v1`
# protocol (defined in coreai-interop) — it accepts only authenticated, ordered, unexpired,
# identity-matched, in-bounds action chunks; stops on heartbeat loss (watchdog); records
# accepted/rejected/executed; and emits a session receipt that proves NOTHING about
# physical safety.
#
# Every gate is fail-closed. The actual robot driver is an INJECTED egress; the default
# `DryRunEgress` sends nothing (dry-run is the safe default — no hardware here). The real
# upstream-LeRobot `Robot` egress is supplied by the caller in a guarded session on real
# hardware. Pure Python; no torch/lerobot needed for the protocol layer.

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol

# actuation-bearing fields the protocol REQUIRES on every action chunk (RFC-0900 §13.1).
REQUIRED_ACTION_CHUNK_FIELDS = (
    "type", "session_id", "sequence", "policy_artifact_root", "action_representation",
    "action_version", "issued_monotonic_ns", "deadline_monotonic_ns", "robot_identity",
    "actions")


class GatewayReject(Exception):
    """A protocol/safety violation — the chunk is refused and NEVER actuates."""


@dataclass(frozen=True)
class SafetyProfile:
    """Software safety bounds (never a mechanical/physical-safety guarantee)."""
    max_abs_action: float = 1.0        # reject any |action| beyond this
    allow_nonfinite: bool = False      # NaN/Inf always rejected by default
    watchdog_ns: int = 500_000_000     # stop if no accepted message within this window
    expected_action_dim: int | None = None


class RobotEgress(Protocol):
    def send(self, actions: Any) -> None: ...


@dataclass
class DryRunEgress:
    """The safe default: sends NOTHING to any hardware; only counts calls."""
    sent: int = 0

    def send(self, actions: Any) -> None:
        self.sent += 1


def _flatten(v):
    if isinstance(v, list):
        for x in v:
            yield from _flatten(x)
    else:
        yield v


@dataclass
class ReferenceRobotGateway:
    robot_identity: str
    policy_artifact_root: str
    safety: SafetyProfile = field(default_factory=SafetyProfile)
    egress: RobotEgress = field(default_factory=DryRunEgress)
    mode: str = "dry_run"              # "dry_run" | "guarded"
    authenticated: bool = False
    session_id: str | None = None
    _next_seq: int = 0
    _last_accept_ns: int = 0
    stopped: bool = False
    faulted: bool = False
    accepted: int = 0
    rejected: int = 0
    executed: int = 0
    rejections: list = field(default_factory=list)

    # --- session lifecycle ---
    def authenticate(self, ok: bool) -> None:
        self.authenticated = bool(ok)

    def open_session(self, msg: dict) -> None:
        if not self.authenticated:
            raise GatewayReject("unauthenticated app cannot open a session")
        if msg.get("robot_identity") != self.robot_identity:
            raise GatewayReject("robot identity mismatch")
        if msg.get("policy_artifact_root") != self.policy_artifact_root:
            raise GatewayReject("policy artifact root mismatch")
        if msg.get("mode") not in ("dry_run", "guarded"):
            raise GatewayReject("invalid session mode")
        self.session_id = msg["session_id"]
        self.mode = msg["mode"]
        self._next_seq = 0
        self.stopped = self.faulted = False

    def heartbeat(self, now_ns: int) -> None:
        if not self.stopped and not self.faulted:
            self._last_accept_ns = now_ns

    def stop(self, reason: str) -> None:
        self.stopped = True
        self._stopped_reason = reason

    def fault(self, reason: str) -> None:
        self.faulted = True
        self._stopped_reason = reason

    # --- the actuation gate (fail-closed) ---
    def submit(self, chunk: dict, now_ns: int) -> str:
        def reject(why: str) -> None:
            self.rejected += 1
            self.rejections.append(why)
            raise GatewayReject(why)

        if self.stopped or self.faulted:
            reject("session is stopped/faulted")
        missing = [f for f in REQUIRED_ACTION_CHUNK_FIELDS if f not in chunk]
        if missing:
            reject(f"action chunk missing required fields: {missing}")
        if not self.authenticated or chunk["session_id"] != self.session_id:
            reject("unauthenticated or wrong session")
        if chunk["robot_identity"] != self.robot_identity or \
                chunk["policy_artifact_root"] != self.policy_artifact_root:
            reject("robot/policy identity mismatch")
        # watchdog: too long since the last accepted message → stop, fail closed.
        if self._last_accept_ns and (now_ns - self._last_accept_ns) > self.safety.watchdog_ns:
            self.stop("watchdog")
            reject("watchdog: message gap exceeded")
        if chunk["sequence"] != self._next_seq:            # replay / out-of-order
            reject(f"sequence {chunk['sequence']} != expected {self._next_seq}")
        if now_ns > chunk["deadline_monotonic_ns"]:        # expired
            reject("action chunk expired past its deadline")
        # action bounds: finite + within |max_abs_action| + (optional) dim.
        actions = chunk["actions"]
        if not (isinstance(actions, dict) and "__array__" in actions):
            reject("actions must be a NamedTensor {__array__,dtype,shape}")
        vals = list(_flatten(actions["__array__"]))
        if not self.safety.allow_nonfinite and any(
                isinstance(v, float) and not math.isfinite(v) for v in vals):
            reject("non-finite action value")
        if any(abs(float(v)) > self.safety.max_abs_action for v in vals):
            reject(f"action exceeds |{self.safety.max_abs_action}| bound")
        dim = self.safety.expected_action_dim
        if dim is not None and actions.get("shape", [None])[-1] != dim:
            reject(f"action dim != expected {dim}")

        # all gates passed → accept + (only in guarded mode) actuate via the egress.
        self._next_seq += 1
        self._last_accept_ns = now_ns
        self.accepted += 1
        if self.mode == "guarded":
            self.egress.send(actions)
            self.executed += 1
            return "executed"
        return "accepted"                                  # dry_run: nothing sent

    def receipt(self) -> dict:
        return {
            "type": "session-receipt", "session_id": self.session_id or "",
            "robot_identity": self.robot_identity,
            "policy_artifact_root": self.policy_artifact_root, "mode": self.mode,
            "accepted": self.accepted, "rejected": self.rejected,
            "executed": self.executed,
            "stopped_reason": getattr(self, "_stopped_reason", None)
            if (self.stopped or self.faulted) else None,
            "proves_physical_safety": False,
        }
