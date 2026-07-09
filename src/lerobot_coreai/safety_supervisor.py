# safety_supervisor.py — runtime safety supervisor for action egress (v0.9.0).
#
# Every action must pass an explicit, auditable supervisor before egress. The
# supervisor validates shape/finiteness/bounds/delta/norm/robot-type against a
# SafetyProfile and returns an auditable SafetyDecision. It is FAIL-CLOSED: any
# uncertain critical condition (including an internal error) blocks the action.
#
# This is a SOFTWARE safety layer. It does not prove physical robot safety, does
# not replace a hardware emergency stop, and does not enable real-world actuation.

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .metrics import action_to_flat_float_list, infer_shape
from .safety_profiles import SafetyProfile

# Operational modes (distinct from the profile's static fail_closed mode).
MODE_OFF = "off"
MODE_REPORT_ONLY = "report_only"
MODE_ENFORCE = "enforce"
SUPERVISOR_MODES = (MODE_OFF, MODE_REPORT_ONLY, MODE_ENFORCE)


@dataclass
class SafetyContext:
    """Runtime context for a supervised action."""

    mode: str = "sim"
    step: int | None = None
    episode: int | None = None
    robot_type: str | None = None
    policy_type: str | None = None
    env_type: str | None = None
    action_source: str = "policy"


@dataclass
class SafetyDecision:
    """An auditable decision about a single action."""

    allowed: bool
    action_modified: bool
    reasons: list[str]
    checks: list[dict[str, Any]]
    profile: str
    mode: str
    severity: str
    original_action_shape: list[int] | None = None
    supervised_action_shape: list[int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "action_modified": self.action_modified,
            "original_action_shape": self.original_action_shape,
            "supervised_action_shape": self.supervised_action_shape,
            "reasons": self.reasons,
            "checks": self.checks,
            "profile": self.profile,
            "mode": self.mode,
            "severity": self.severity,
        }


@dataclass
class SupervisedAction:
    """The result of supervising one action.

    Invariant: decision.allowed is False  => action is None.
               decision.allowed is True   => action may egress.
    """

    original_action: Any
    action: Any | None
    decision: SafetyDecision


# MARK: - Action normalization helpers

def normalize_action(action: Any) -> tuple[list[float] | None, list[int] | None]:
    """Return (flat_float_list, shape). Either may be None if not derivable."""
    if action is None:
        return None, None
    if isinstance(action, dict) and "action" in action:
        action = action["action"]
    try:
        flat = action_to_flat_float_list(action)
    except Exception:
        flat = None
    shape = infer_shape(action)
    return flat, shape


def reshape(flat: list[float], shape: list[int] | None) -> Any:
    """Rebuild a nested list from a flat list and a shape. Falls back to flat."""
    if shape is None or len(shape) <= 1:
        return list(flat)
    def _build(idx: int, dims: list[int]) -> Any:
        if len(dims) == 1:
            return flat[idx:idx + dims[0]]
        block = 1
        for d in dims[1:]:
            block *= d
        return [_build(idx + i * block, dims[1:]) for i in range(dims[0])]
    return _build(0, shape)


# MARK: - Supervisor

class SafetySupervisor:
    """Evaluates actions against a SafetyProfile before egress."""

    def __init__(self, profile: SafetyProfile, mode: str = MODE_ENFORCE):
        if mode not in SUPERVISOR_MODES:
            raise ValueError(f"Unknown supervisor mode: {mode}")
        self.profile = profile
        self.mode = mode
        self.previous_action: list[float] | None = None

    # -- individual checks (each returns a check dict) --

    def _check_present(self, action: Any) -> dict[str, Any]:
        return {"name": "action_present", "passed": action is not None,
                "severity": "critical"}

    def _check_finite(self, flat: list[float] | None) -> dict[str, Any]:
        p = self.profile
        if flat is None:
            return {"name": "finite", "passed": False, "severity": "critical",
                    "value": "unparseable"}
        nan = sum(1 for v in flat if math.isnan(v))
        inf = sum(1 for v in flat if math.isinf(v))
        passed = (p.allow_nan or nan == 0) and (p.allow_inf or inf == 0)
        return {"name": "finite", "passed": passed, "severity": "critical",
                "nan_count": nan, "inf_count": inf}

    def _check_shape(self, shape: list[int] | None) -> dict[str, Any]:
        p = self.profile
        if shape is None:
            return {"name": "shape", "passed": not p.require_known_shape,
                    "severity": "critical", "value": None}
        if p.action_shape is not None and shape != p.action_shape:
            passed = p.allow_shape_change
            return {"name": "shape", "passed": passed, "severity": "critical",
                    "value": shape, "expected": p.action_shape}
        return {"name": "shape", "passed": True, "severity": "critical", "value": shape}

    def _bounds_arrays(self, n: int) -> tuple[list[float], list[float], bool]:
        """Compute per-element (lo, hi) bounds; ok=False if a list length mismatches."""
        p = self.profile
        lo = [-math.inf] * n
        hi = [math.inf] * n
        ok = True

        def _apply(bound, is_min):
            nonlocal ok
            if bound is None:
                return
            if isinstance(bound, list):
                if len(bound) != n:
                    ok = False
                    return
                for i in range(n):
                    if is_min:
                        lo[i] = max(lo[i], float(bound[i]))
                    else:
                        hi[i] = min(hi[i], float(bound[i]))
            else:
                for i in range(n):
                    if is_min:
                        lo[i] = max(lo[i], float(bound))
                    else:
                        hi[i] = min(hi[i], float(bound))

        _apply(p.min_action, True)
        _apply(p.max_action, False)
        if p.max_abs_action is not None:
            m = float(p.max_abs_action)
            for i in range(n):
                lo[i] = max(lo[i], -m)
                hi[i] = min(hi[i], m)
        return lo, hi, ok

    def _check_bounds(self, flat: list[float] | None) -> dict[str, Any]:
        p = self.profile
        if flat is None:
            return {"name": "bounds", "passed": True, "severity": "critical",
                    "reason": "no_values"}
        if p.min_action is None and p.max_action is None and p.max_abs_action is None:
            return {"name": "bounds", "passed": True, "severity": "critical",
                    "reason": "no_bounds_configured"}
        lo, hi, ok = self._bounds_arrays(len(flat))
        if not ok:
            return {"name": "bounds", "passed": False, "severity": "critical",
                    "reason": "bound_length_mismatch"}
        violations = sum(1 for i, v in enumerate(flat) if v < lo[i] or v > hi[i])
        return {"name": "bounds", "passed": violations == 0, "severity": "critical",
                "violations": violations}

    def _check_delta(self, flat: list[float] | None) -> dict[str, Any]:
        p = self.profile
        if p.max_delta is None:
            return {"name": "delta", "passed": True, "severity": "critical",
                    "reason": "no_max_delta"}
        if self.previous_action is None:
            return {"name": "delta", "passed": True, "severity": "critical",
                    "reason": "first_action_no_previous"}
        if flat is None or len(flat) != len(self.previous_action):
            # v0.9.1: a max_delta constraint cannot be verified across a shape
            # change. Fail-closed rather than silently skip — if you asked for
            # bounded motion, an unverifiable step is unsafe (previously this
            # passed, letting allow_shape_change bypass the delta bound).
            return {"name": "delta", "passed": False, "severity": "critical",
                    "reason": "delta_unverifiable_shape_changed"}
        max_d = max((abs(flat[i] - self.previous_action[i]) for i in range(len(flat))),
                    default=0.0)
        return {"name": "delta", "passed": max_d <= p.max_delta, "severity": "critical",
                "value": max_d, "threshold": p.max_delta}

    def _check_l2(self, flat: list[float] | None) -> dict[str, Any]:
        p = self.profile
        if p.max_l2_norm is None:
            return {"name": "l2_norm", "passed": True, "severity": "critical",
                    "reason": "no_max_l2_norm"}
        if flat is None:
            return {"name": "l2_norm", "passed": True, "severity": "critical",
                    "reason": "no_values"}
        finite = [v for v in flat if math.isfinite(v)]
        norm = math.sqrt(sum(v * v for v in finite))
        return {"name": "l2_norm", "passed": norm <= p.max_l2_norm, "severity": "critical",
                "value": norm, "threshold": p.max_l2_norm}

    def _check_robot_type(self, context: SafetyContext | None) -> dict[str, Any]:
        p = self.profile
        if not p.require_robot_type_match or p.robot_type is None:
            return {"name": "robot_type", "passed": True, "severity": "critical",
                    "reason": "not_required"}
        ctx_rt = context.robot_type if context else None
        return {"name": "robot_type", "passed": ctx_rt == p.robot_type,
                "severity": "critical", "value": ctx_rt, "expected": p.robot_type}

    def evaluate(self, action: Any, *, context: SafetyContext | None = None) -> SupervisedAction:
        """Evaluate one action against the profile. Honors the operational mode."""
        p = self.profile
        flat, shape = normalize_action(action)

        checks = [
            self._check_present(action),
            self._check_finite(flat),
            self._check_shape(shape),
            self._check_bounds(flat),
            self._check_delta(flat),
            self._check_l2(flat),
            self._check_robot_type(context),
        ]
        reasons: list[str] = [c["name"] for c in checks if not c["passed"]]

        bounds_failed = not next(c for c in checks if c["name"] == "bounds")["passed"]
        finite_ok = next(c for c in checks if c["name"] == "finite")["passed"]

        # Clipping (only meaningful when finite and bounds failed and enabled).
        supervised = action
        supervised_shape = shape
        action_modified = False
        clipped = False
        if bounds_failed and p.clip_to_bounds and finite_ok and flat is not None:
            lo, hi, ok = self._bounds_arrays(len(flat))
            if ok:
                clipped_flat = [min(max(flat[i], lo[i]), hi[i]) for i in range(len(flat))]
                supervised = reshape(clipped_flat, shape)
                supervised_shape = infer_shape(supervised)
                clipped = True
                action_modified = True
                reasons.append("action_clipped_to_bounds")
                # After clipping, bounds no longer fail.
                for c in checks:
                    if c["name"] == "bounds":
                        c["passed"] = True
                        c["clipped"] = True

        # An action is unsafe if any critical check still fails.
        unsafe = any(c["severity"] == "critical" and not c["passed"] for c in checks)
        if clipped and p.block_on_clip:
            unsafe = True
            reasons.append("blocked_due_to_clip")

        severity = "critical" if unsafe else ("warning" if action_modified else "info")

        # Apply operational mode.
        if self.mode == MODE_ENFORCE:
            allowed = not unsafe
        else:  # off / report_only never block egress at the decision level
            allowed = True
            if self.mode == MODE_REPORT_ONLY:
                # Report-only is pure observation: never modifies the action.
                supervised = action
                supervised_shape = shape
                action_modified = False
                if "action_clipped_to_bounds" in reasons:
                    reasons.remove("action_clipped_to_bounds")
                if unsafe:
                    reasons.append("report_only_would_block")

        decision = SafetyDecision(
            allowed=allowed,
            action_modified=action_modified,
            reasons=reasons,
            checks=checks,
            profile=p.name,
            mode=self.mode,
            severity=severity,
            original_action_shape=shape,
            supervised_action_shape=supervised_shape,
        )

        if allowed:
            # Track the (possibly clipped) egressed action for the next delta check.
            egress_flat, _ = normalize_action(supervised)
            if egress_flat is not None:
                self.previous_action = egress_flat
            return SupervisedAction(original_action=action, action=supervised, decision=decision)
        return SupervisedAction(original_action=action, action=None, decision=decision)


def safe_evaluate(
    supervisor: SafetySupervisor, action: Any, *, context: SafetyContext | None = None,
) -> SupervisedAction:
    """Fail-closed wrapper: any internal error blocks the action."""
    try:
        return supervisor.evaluate(action, context=context)
    except Exception as e:  # never let a supervisor bug open the gate
        decision = SafetyDecision(
            allowed=False,
            action_modified=False,
            reasons=["supervisor_internal_error"],
            checks=[{"name": "supervisor_internal_error", "passed": False,
                     "severity": "critical", "error": str(e)}],
            profile=getattr(supervisor.profile, "name", "unknown"),
            mode=getattr(supervisor, "mode", MODE_ENFORCE),
            severity="critical",
        )
        return SupervisedAction(original_action=action, action=None, decision=decision)
