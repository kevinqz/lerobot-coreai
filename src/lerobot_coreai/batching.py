# batching.py — split-and-stack fallback for a non-batched runner (v1.2.5).
#
# LeRobot's eval passes batched observations (e.g. observation["task"] is a list,
# one per env). The CoreAI runner is non-batched today. When the batch contract
# says supports_batch=false with fallback=split_and_stack, we split a batched
# observation into single-sample observations, run each, and stack the actions
# back deterministically. Unsupported shapes are rejected with a clear error. No
# hardware, no egress.

from __future__ import annotations

from typing import Any, Callable

from .errors import CoreAIPolicyError

# Keys that never carry a per-sample batch dimension.
_SCALAR_KEYS = {"task"}


def detect_batch_size(observation: dict[str, Any]) -> int | None:
    """Return the batch size if the observation looks batched, else None.

    Batched signals: a list-valued ``task`` (one string per env), or a leading
    batch dimension that is consistent across batched keys. Returns None for a
    plain single observation.
    """
    sizes: set[int] = set()
    task = observation.get("task")
    if isinstance(task, (list, tuple)):
        sizes.add(len(task))
    for k, v in observation.items():
        if k in _SCALAR_KEYS:
            continue
        # A batched vector/image is a list whose elements are themselves
        # sequences (list-of-rows) — the outer length is the batch size.
        if isinstance(v, (list, tuple)) and v and isinstance(v[0], (list, tuple)):
            sizes.add(len(v))
    if not sizes:
        return None
    if len(sizes) > 1:
        raise CoreAIPolicyError(
            f"inconsistent batch sizes across observation keys: {sorted(sizes)}.")
    (n,) = tuple(sizes)
    return n if n > 0 else None


def split_observation(observation: dict[str, Any], n: int) -> list[dict[str, Any]]:
    """Split a batched observation into ``n`` single-sample observations."""
    out: list[dict[str, Any]] = []
    for i in range(n):
        sample: dict[str, Any] = {}
        for k, v in observation.items():
            if k == "task" and isinstance(v, (list, tuple)):
                sample[k] = v[i]
            elif isinstance(v, (list, tuple)) and v and isinstance(v[0], (list, tuple)):
                sample[k] = v[i]
            else:
                sample[k] = v  # shared scalar/config value
        out.append(sample)
    return out


def stack_actions(actions: list[Any]) -> list[Any]:
    """Stack per-sample actions back into a batch (a list of actions)."""
    return list(actions)


def run_batched_with_fallback(
    observation: dict[str, Any], batch_contract, run_single: Callable[[dict], Any],
) -> Any:
    """Run a possibly-batched observation using the batch contract's policy.

    - Not batched: call run_single directly.
    - Batched + supports_batch: caller should have handled it; we still split.
    - Batched + fallback=split_and_stack: split → run each → stack.
    - Batched + fallback=reject (or over max_batch_size): raise.
    """
    n = detect_batch_size(observation)
    if n is None:
        return run_single(observation)
    if batch_contract.supports_batch and n <= batch_contract.max_batch_size:
        return run_single(observation)
    if batch_contract.fallback != "split_and_stack":
        raise CoreAIPolicyError(
            f"batched observation (n={n}) but batch fallback is "
            f"{batch_contract.fallback!r}.")
    return stack_actions([run_single(s) for s in split_observation(observation, n)])
