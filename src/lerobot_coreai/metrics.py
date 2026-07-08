# metrics.py — action comparison metrics for PyTorch vs CoreAI parity (v0.5).

from __future__ import annotations

import math
from typing import Any

from .errors import ActionParityError


def action_to_flat_float_list(action: Any) -> list[float]:
    """Flatten any action representation to a single list of floats.

    Handles nested lists/tuples, numpy arrays (.tolist()), torch tensors
    (.detach().cpu().tolist()), and scalar numbers.
    """
    # Duck-type torch tensor.
    if hasattr(action, "detach"):
        action = action.detach()
    if hasattr(action, "cpu"):
        action = action.cpu()
    if hasattr(action, "tolist"):
        action = action.tolist()

    result: list[float] = []

    def _flatten(value: Any) -> None:
        if isinstance(value, (list, tuple)):
            for v in value:
                _flatten(v)
        elif isinstance(value, (int, float)):
            result.append(float(value))
        elif value is None:
            pass
        else:
            # Last resort: try float conversion.
            try:
                result.append(float(value))
            except (TypeError, ValueError):
                raise ActionParityError(
                    f"Cannot convert action element to float: {type(value).__name__}"
                ) from None

    _flatten(action)
    return result


def infer_shape(value: Any) -> list[int] | None:
    """Infer the shape of a nested list or tensor-like."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "shape"):
        try:
            return list(value.shape)
        except Exception:
            pass
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        return None
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return [0]
        child_shapes = [infer_shape(v) for v in value]
        first = child_shapes[0]
        if any(s != first for s in child_shapes):
            return None  # ragged
        if first is not None:
            return [len(value)] + first
        return [len(value)]
    return None


def _check_shapes_match(a: Any, b: Any) -> None:
    """Verify two actions have the same shape."""
    sa = infer_shape(a)
    sb = infer_shape(b)
    if sa is not None and sb is not None and sa != sb:
        raise ActionParityError(
            f"Action shapes differ: torch={sa}, coreai={sb}"
        )


def mean_absolute_error(a: Any, b: Any) -> float:
    """Mean absolute error between two actions."""
    _check_shapes_match(a, b)
    fa = action_to_flat_float_list(a)
    fb = action_to_flat_float_list(b)
    if len(fa) != len(fb):
        raise ActionParityError(
            f"Action sizes differ after flatten: torch={len(fa)}, coreai={len(fb)}"
        )
    if len(fa) == 0:
        return 0.0
    return sum(abs(x - y) for x, y in zip(fa, fb)) / len(fa)


def max_absolute_error(a: Any, b: Any) -> float:
    """Max absolute error between two actions."""
    _check_shapes_match(a, b)
    fa = action_to_flat_float_list(a)
    fb = action_to_flat_float_list(b)
    if len(fa) != len(fb):
        raise ActionParityError(
            f"Action sizes differ after flatten: torch={len(fa)}, coreai={len(fb)}"
        )
    if len(fa) == 0:
        return 0.0
    return max(abs(x - y) for x, y in zip(fa, fb))


def cosine_similarity(a: Any, b: Any) -> float:
    """Cosine similarity between two flattened actions. Returns 1.0 for identical."""
    _check_shapes_match(a, b)
    fa = action_to_flat_float_list(a)
    fb = action_to_flat_float_list(b)
    if len(fa) != len(fb):
        raise ActionParityError(
            f"Action sizes differ after flatten: torch={len(fa)}, coreai={len(fb)}"
        )
    if len(fa) == 0:
        return 1.0
    dot = sum(x * y for x, y in zip(fa, fb))
    norm_a = math.sqrt(sum(x * x for x in fa))
    norm_b = math.sqrt(sum(y * y for y in fb))
    if norm_a < 1e-12 or norm_b < 1e-12:
        # Near-zero actions — cosine is ill-conditioned. Return 1.0 if both are near zero.
        if norm_a < 1e-12 and norm_b < 1e-12:
            return 1.0
        return 0.0
    return dot / (norm_a * norm_b)


def relative_mae(a: Any, b: Any, eps: float = 1e-8) -> float:
    """Scale-invariant relative MAE: mean(|a - b|) / (mean(|a|) + eps)."""
    _check_shapes_match(a, b)
    fa = action_to_flat_float_list(a)
    fb = action_to_flat_float_list(b)
    if len(fa) != len(fb):
        raise ActionParityError(
            f"Action sizes differ after flatten: torch={len(fa)}, coreai={len(fb)}"
        )
    if len(fa) == 0:
        return 0.0
    mae = sum(abs(x - y) for x, y in zip(fa, fb)) / len(fa)
    scale = sum(abs(x) for x in fa) / len(fa)
    return mae / (scale + eps)
