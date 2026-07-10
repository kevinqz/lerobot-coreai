# batch_protocol.py — batch execution decision (v1.3.8 foundation, no B>1 yet).
#
# Batching is not just [B,...]; the runner may be stateless, session-scoped, or
# globally stateful, and may batch natively or not at all. This module pins the
# state/batch contract and provides a PURE decision function so the actual batched
# runtime (v1.3.9) is a mechanical, auditable implementation — never a new source
# of session-mixing ambiguity. Nothing here performs inference.

from __future__ import annotations

from dataclasses import dataclass

from lerobot_coreai.errors import CoreAIPolicyError

INFERENCE_STATE_SCOPES = ("stateless", "request_scoped", "session_scoped", "global")
BATCH_SEMANTICS = ("native", "split_and_stack")
BATCH_MODES = ("single_only", "native_batch", "split_and_stack", "auto")

# Execution modes the decision resolves to (auto never survives as an outcome).
MODE_SINGLE = "single_only"
MODE_NATIVE = "native_batch"
MODE_SPLIT = "split_and_stack"


@dataclass(frozen=True)
class BatchDecision:
    mode: str            # single_only | native_batch | split_and_stack
    reason: str
    max_batch_size: int = 1
    requires_session_ids: bool = False


def _validate_scope(scope: str | None) -> None:
    if scope is not None and scope not in INFERENCE_STATE_SCOPES:
        raise CoreAIPolicyError(
            f"unknown inference_state.scope {scope!r}; expected one of "
            f"{INFERENCE_STATE_SCOPES}.")


def _split_allowed(scope: str | None, supports_session_ids: bool) -> tuple[bool, bool, str]:
    """Return (allowed, requires_session_ids, reason) for split-and-stack."""
    if scope in (None, "stateless", "request_scoped"):
        return True, False, "stateless/request-scoped runner: split is safe"
    if scope == "session_scoped":
        if supports_session_ids:
            return True, True, "session-scoped runner with per-slot session ids"
        return False, False, ("session-scoped runner without session id support: "
                              "split would mix sessions")
    if scope == "global":
        return False, False, ("globally stateful runner: split-and-stack is "
                              "forbidden (calls would corrupt shared state)")
    return False, False, f"unhandled scope {scope!r}"


def select_batch_execution_mode(config, capabilities) -> BatchDecision:
    """Decide how a B-observation batch would execute, fail-closed (no B>1 here).

    ``config.batch_mode`` is the request; ``capabilities`` (RunnerCapabilities or
    None) describes the runner. Rules:
      - single_only              -> single (B=1 only)
      - native_batch             -> requires supports_batch AND semantics 'native'
      - split_and_stack          -> allowed for stateless/request-scoped; session-
                                    scoped needs session ids; global is forbidden
      - auto                     -> native if supported, else split if allowed,
                                    else single_only
    An unknown scope raises.
    """
    mode = getattr(config, "batch_mode", "single_only")
    if mode not in BATCH_MODES:
        raise CoreAIPolicyError(f"unknown batch_mode {mode!r}; expected {BATCH_MODES}.")

    scope = getattr(capabilities, "inference_state_scope", None)
    _validate_scope(scope)
    supports_batch = bool(getattr(capabilities, "supports_batch", False))
    semantics = getattr(capabilities, "action_batching_semantics", None)
    max_bs = int(getattr(capabilities, "max_batch_size", None) or 1)
    session_ids = bool(getattr(capabilities, "supports_session_ids", False))
    native_ok = supports_batch and semantics == "native"

    if mode == "single_only":
        return BatchDecision(MODE_SINGLE, "single_only requested", 1)

    if mode == "native_batch":
        if not native_ok:
            raise CoreAIPolicyError(
                "native_batch requested but the runner does not announce "
                "action_batching.supported with semantics 'native'.")
        return BatchDecision(MODE_NATIVE, "native batch announced", max_bs)

    if mode == "split_and_stack":
        allowed, needs_sid, reason = _split_allowed(scope, session_ids)
        if not allowed:
            raise CoreAIPolicyError(f"split_and_stack refused: {reason}.")
        return BatchDecision(MODE_SPLIT, reason, max_bs or 1,
                             requires_session_ids=needs_sid)

    # auto
    if native_ok:
        return BatchDecision(MODE_NATIVE, "auto -> native (announced)", max_bs)
    allowed, needs_sid, reason = _split_allowed(scope, session_ids)
    if allowed:
        return BatchDecision(MODE_SPLIT, f"auto -> split ({reason})", max_bs or 1,
                             requires_session_ids=needs_sid)
    return BatchDecision(MODE_SINGLE, f"auto -> single ({reason})", 1)
