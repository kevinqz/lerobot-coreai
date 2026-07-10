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
# state_isolation values a native B>1 batch is allowed to run under (v1.3.9).
SAFE_ISOLATION = ("stateless", "request_scoped")
# scopes a split B>1 batch is allowed to run under (v1.3.9); session/global deferred.
SAFE_SCOPES = ("stateless", "request_scoped")

MODE_SINGLE = "single_only"
MODE_NATIVE = "native_batch"
MODE_SPLIT = "split_and_stack"


@dataclass(frozen=True)
class BatchDecision:
    mode: str            # single_only | native_batch | split_and_stack
    reason: str
    batch_size: int = 1
    effective_max_batch_size: int = 1


def _effective_max(config, artifact_batch_contract, capabilities) -> int:
    """Minimum of the artifact/config/runner maxima (each optional)."""
    limits = []
    for src, attr in ((artifact_batch_contract, "max_batch_size"),
                      (config, "max_batch_size"),
                      (capabilities, "max_batch_size")):
        v = getattr(src, attr, None)
        if v is not None:
            limits.append(int(v))
    return min(limits) if limits else 1


def _validate_batch_capabilities(capabilities) -> None:
    """Fail-closed capability validation used only when B>1 is requested."""
    if capabilities is None:
        raise CoreAIPolicyError(
            "no runner capabilities available; B>1 requires an announced runner.")
    scope = getattr(capabilities, "inference_state_scope", None)
    if scope is None:
        raise CoreAIPolicyError(
            "inference_state.scope is not announced; B>1 refuses to assume "
            "stateless (a missing scope is not safe).")
    if scope not in INFERENCE_STATE_SCOPES:
        raise CoreAIPolicyError(f"unknown inference_state.scope {scope!r}.")
    if getattr(capabilities, "supports_batch", False):
        mbs = getattr(capabilities, "max_batch_size", None)
        if mbs is None or int(mbs) < 1:
            raise CoreAIPolicyError(
                f"runner announces batch support but max_batch_size is invalid: {mbs!r}.")


def select_batch_execution_mode(config, artifact_batch_contract, capabilities,
                                requested_batch_size: int) -> BatchDecision:
    """Decide how a B-observation request executes, fail-closed (v1.3.9).

    B=1 always resolves to single_only (backward compatible, no capability
    requirement). For B>1 the runner MUST announce a known, safe state scope:
      - scope missing/unknown       -> fail (never assume stateless)
      - global / session_scoped     -> fail (session batching is deferred)
      - native_batch                -> needs native semantics AND state_isolation
                                       in {stateless, request_scoped}
      - split_and_stack / auto      -> needs scope in {stateless, request_scoped}
      - B > effective max           -> fail (min of artifact/config/runner limits)
    """
    mode = getattr(config, "batch_mode", "single_only")
    if mode not in BATCH_MODES:
        raise CoreAIPolicyError(f"unknown batch_mode {mode!r}; expected {BATCH_MODES}.")
    b = int(requested_batch_size)
    if b < 1:
        raise CoreAIPolicyError(f"requested_batch_size must be >= 1, got {b}.")

    if b == 1:
        return BatchDecision(MODE_SINGLE, "B=1", 1, 1)

    if mode == "single_only":
        raise CoreAIPolicyError(
            f"batch_mode='single_only' cannot serve B={b}; set native_batch/"
            "split_and_stack/auto and use a batch-capable runner.")

    _validate_batch_capabilities(capabilities)
    scope = getattr(capabilities, "inference_state_scope", None)
    if scope == "global":
        raise CoreAIPolicyError("global-state runner: B>1 is forbidden.")
    if scope == "session_scoped":
        raise CoreAIPolicyError(
            "session_scoped batching is deferred (needs a per-slot session "
            "lifecycle/transaction protocol); refusing B>1.")

    eff_max = _effective_max(config, artifact_batch_contract, capabilities)
    if b > eff_max:
        raise CoreAIPolicyError(
            f"requested batch {b} exceeds the effective max {eff_max} "
            "(min of artifact/config/runner limits).")

    supports_batch = bool(getattr(capabilities, "supports_batch", False))
    semantics = getattr(capabilities, "action_batching_semantics", None)
    isolation = getattr(capabilities, "action_batching_state_isolation", None)
    native_ok = (supports_batch and semantics == "native"
                 and isolation in SAFE_ISOLATION)

    if mode == "native_batch":
        if not native_ok:
            raise CoreAIPolicyError(
                "native_batch requires action_batching.supported + semantics "
                f"'native' + state_isolation in {SAFE_ISOLATION}; got "
                f"supported={supports_batch} semantics={semantics!r} "
                f"isolation={isolation!r}.")
        return BatchDecision(MODE_NATIVE, "native batch (isolated)", b, eff_max)

    if mode == "split_and_stack":
        if scope not in SAFE_SCOPES:
            raise CoreAIPolicyError(
                f"split_and_stack requires scope in {SAFE_SCOPES}; got {scope!r}.")
        return BatchDecision(MODE_SPLIT, "split-and-stack (stateless/request)", b, eff_max)

    # auto
    if native_ok:
        return BatchDecision(MODE_NATIVE, "auto -> native (isolated)", b, eff_max)
    if scope in SAFE_SCOPES:
        return BatchDecision(MODE_SPLIT, "auto -> split (stateless/request)", b, eff_max)
    raise CoreAIPolicyError(
        f"auto: no safe B>1 mode for scope={scope!r}, semantics={semantics!r}, "
        f"isolation={isolation!r}.")
