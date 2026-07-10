# coreai_observation_serialization.py — JSON-safe observation boundary (v1.2.8).
#
# LeRobotDataset items contain torch tensors / numpy arrays, which are not JSON
# serializable and cannot be sent to the CoreAI runner as-is. This is the single
# boundary that converts an observation to JSON-safe values, preserves shape/
# dtype metadata, hashes the exact payload sent, and REFUSES unknown objects
# rather than silently coercing them. No hardware, no egress.

from __future__ import annotations

import hashlib
import json
from typing import Any

from .errors import CoreAIPolicyError

_PRIMITIVES = (str, int, float, bool, type(None))


def _tensorish_to_list(value: Any):
    """Convert a torch.Tensor / numpy.ndarray to (nested list, dtype, shape) or None."""
    # numpy
    tolist = getattr(value, "tolist", None)
    dtype = getattr(value, "dtype", None)
    shape = getattr(value, "shape", None)
    # torch tensors need detach().cpu() before tolist in general; guard for it.
    if type(value).__module__.split(".")[0] == "torch":
        try:  # pragma: no cover - only when torch present
            v = value.detach().cpu()
            return v.tolist(), str(v.dtype), tuple(v.shape)
        except Exception:
            return None
    if tolist is not None and shape is not None:
        try:
            return tolist(), (str(dtype) if dtype is not None else None), tuple(shape)
        except Exception:
            return None
    return None


def serialize_value(value: Any) -> Any:
    """Return a JSON-safe representation of a single value. Fail-closed on unknown.

    Tensors/arrays become ``{"__array__": [...], "dtype": ..., "shape": [...]}`` so
    the payload is JSON-safe *and* the shape/dtype are auditable.
    """
    if isinstance(value, _PRIMITIVES):
        return value
    if isinstance(value, dict):
        return {str(k): serialize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_value(v) for v in value]

    conv = _tensorish_to_list(value)
    if conv is not None:
        data, dtype, shape = conv
        return {"__array__": data, "dtype": dtype, "shape": list(shape)}

    raise CoreAIPolicyError(
        f"cannot serialize observation value of type {type(value).__name__!r} to "
        "JSON-safe form; refusing to send an unknown object to the runner.")


def serialize_observation(observation: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe copy of an observation dict. Fail-closed on unknowns."""
    if not isinstance(observation, dict):
        raise CoreAIPolicyError("observation must be a dict.")
    return {str(k): serialize_value(v) for k, v in observation.items()}


def observation_sha256(payload: dict[str, Any]) -> str:
    """Deterministic sha256 of a JSON-safe observation payload."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(blob).hexdigest()}"


def serialize_and_hash(observation: dict[str, Any]) -> tuple[dict[str, Any], str]:
    payload = serialize_observation(observation)
    return payload, observation_sha256(payload)


# Keys a dataset frame may carry that must NOT be fed to a policy as input —
# feeding the ground-truth action/reward/index would contaminate evaluation.
_NON_OBSERVATION_KEYS = {
    "action", "reward", "index", "frame_index", "episode_index", "timestamp",
    "task_index", "dataset_index", "next.reward", "next.done", "next.success",
    "success", "done",
}


def extract_observation(item: dict[str, Any], manifest: Any = None) -> dict[str, Any]:
    """Return only the policy's declared observation inputs (+ task).

    Prevents label leakage: a LeRobotDataset frame carries the ground-truth
    action/reward/index, which must never reach the policy. With a manifest, keep
    exactly the declared observation features (plus ``task``); without one, keep
    ``observation.*`` and ``task`` and drop known non-observation keys.
    """
    if not isinstance(item, dict):
        return item
    obs: dict[str, Any] = {}
    features = getattr(manifest, "observation_features", None) if manifest else None
    if features:
        for k in features:
            if k in item:
                obs[k] = item[k]
        if "task" in item:
            obs["task"] = item["task"]
        return obs
    for k, v in item.items():
        if k in _NON_OBSERVATION_KEYS:
            continue
        if k.startswith("observation") or k == "task":
            obs[k] = v
    return obs
