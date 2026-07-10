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


def _min_present(*vals) -> int:
    present = [int(v) for v in vals if v is not None]
    return min(present) if present else 1


def _native_effective_max(config, contract, capabilities) -> int:
    # Native = ONE batched request -> the runner's native max applies.
    return _min_present(getattr(contract, "native_max_batch_size", None),
                        getattr(config, "max_batch_size", None),
                        getattr(capabilities, "max_batch_size", None))


def _split_effective_max(config, contract) -> int:
    # Split = B single requests -> the runner's NATIVE max must NOT cap it (P1.3).
    return _min_present(getattr(contract, "split_max_batch_size", None),
                        getattr(config, "max_batch_size", None),
                        getattr(config, "max_split_requests", None))


def _validate_batch_capabilities(capabilities) -> None:
    if capabilities is None:
        raise CoreAIPolicyError(
            "no runner capabilities available; B>1 requires an announced runner.")
    scope = getattr(capabilities, "inference_state_scope", None)
    if scope is None:
        raise CoreAIPolicyError(
            "inference_state.scope is not announced; B>1 refuses to assume stateless.")
    if scope not in INFERENCE_STATE_SCOPES:
        raise CoreAIPolicyError(f"unknown inference_state.scope {scope!r}.")


def select_batch_execution_mode(config, artifact_batch_contract, capabilities,
                                requested_batch_size: int) -> BatchDecision:
    """Decide how a B-observation request executes, fail-closed (v1.3.10).

    B=1 is always single. For B>1 the ARTIFACT BATCH CONTRACT is authoritative:
      - non-authoritative (legacy v0/v2) contract         -> fail (needs v3)
      - native requires contract.native_supported + runner native semantics +
        slot_isolation == contract.native_slot_isolation (default 'independent')
      - split requires contract.split_supported + scope in the contract's allowed
        scopes (global/session deferred)
      - fallback == 'reject' forbids auto from choosing split
      - native/split effective maxima are computed SEPARATELY; the runner's native
        max never caps split
      - queue.layout / commit_semantics must match what this runtime implements
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
            f"batch_mode='single_only' cannot serve B={b}.")

    c = artifact_batch_contract
    if c is None or not getattr(c, "authoritative", False):
        raise CoreAIPolicyError(
            "B>1 requires an authoritative coreai-batch-contract.v3 in the manifest; "
            "the artifact does not declare one.")
    if getattr(c, "queue_layout", None) != "time_major_batched":
        raise CoreAIPolicyError(
            f"unsupported queue layout {c.queue_layout!r} (this runtime is "
            "time_major_batched only).")
    if getattr(c, "commit_semantics", None) != "atomic_queue_commit":
        raise CoreAIPolicyError(
            f"unsupported commit semantics {c.commit_semantics!r}.")

    _validate_batch_capabilities(capabilities)
    scope = getattr(capabilities, "inference_state_scope", None)
    if scope == "global":
        raise CoreAIPolicyError("global-state runner: B>1 is forbidden.")
    if scope == "session_scoped":
        raise CoreAIPolicyError("session_scoped batching is deferred; refusing B>1.")

    semantics = getattr(capabilities, "action_batching_semantics", None)
    slot_iso = (getattr(capabilities, "action_batching_slot_isolation", None)
                or getattr(capabilities, "action_batching_state_isolation", None))
    runner_supports_batch = bool(getattr(capabilities, "supports_batch", False))
    native_ok = (c.native_supported and runner_supports_batch and semantics == "native"
                 and slot_iso == c.native_slot_isolation)
    split_ok = c.split_supported and scope in c.split_allowed_scopes

    def _native():
        if not native_ok:
            raise CoreAIPolicyError(
                "native_batch refused: needs contract.native_supported + runner "
                f"native semantics + slot_isolation=={c.native_slot_isolation!r}; got "
                f"contract={c.native_supported} runner_supported={runner_supports_batch} "
                f"semantics={semantics!r} slot_isolation={slot_iso!r}.")
        eff = _native_effective_max(config, c, capabilities)
        if b > eff:
            raise CoreAIPolicyError(f"native batch {b} exceeds effective max {eff}.")
        return BatchDecision(MODE_NATIVE, "native batch (isolated)", b, eff)

    def _split():
        if not split_ok:
            raise CoreAIPolicyError(
                "split_and_stack refused: needs contract.split_supported + scope in "
                f"{c.split_allowed_scopes}; got contract={c.split_supported} "
                f"scope={scope!r}.")
        eff = _split_effective_max(config, c)
        if b > eff:
            raise CoreAIPolicyError(f"split batch {b} exceeds effective max {eff}.")
        return BatchDecision(MODE_SPLIT, "split-and-stack", b, eff)

    if mode == "native_batch":
        return _native()
    if mode == "split_and_stack":
        return _split()
    # auto
    if native_ok:
        return _native()
    if split_ok and c.fallback != "reject":
        return _split()
    raise CoreAIPolicyError(
        f"auto: no permitted B>1 mode (native_ok={native_ok}, split_ok={split_ok}, "
        f"fallback={c.fallback!r}).")
