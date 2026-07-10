# transport.py — LeRobot batch -> CoreAI observation boundary (v1.3.2).
#
# LeRobot passes a BATCHED observation: observation.state is [B, A], images are
# [B, C, H, W], and task is a list[str] (one per env). The CoreAI runner expects
# a SINGLE observation matching the manifest: observation.state [A], task a
# string. This is the single boundary that bridges the two for B=1 — it strips
# the leading batch dim, unwraps the task list, keeps ONLY manifest-declared
# observation features (never the ground-truth action/reward), converts
# torch/numpy to JSON-safe values, and hashes the exact payload. B>1 fails
# closed (batched transport is v1.3.3).

from __future__ import annotations

from typing import Any

from lerobot_coreai.coreai_observation_serialization import (
    extract_observation, observation_sha256, serialize_value,
)
from lerobot_coreai.errors import CoreAIPolicyError


def infer_batch_size(batch: dict[str, Any]) -> int:
    sizes: set[int] = set()
    for k, v in batch.items():
        if k == "task":
            if isinstance(v, (list, tuple)):
                sizes.add(len(v))
            continue
        shape = getattr(v, "shape", None)
        if shape is not None and len(shape) >= 1:
            sizes.add(int(shape[0]))
        elif isinstance(v, (list, tuple)) and v and isinstance(v[0], (list, tuple)):
            sizes.add(len(v))
    if len(sizes) > 1:
        raise CoreAIPolicyError(
            f"inconsistent batch sizes across observation keys: {sorted(sizes)}.")
    return next(iter(sizes)) if sizes else 1


def _strip_leading_batch(value: Any) -> Any:
    """Drop a leading batch dim of size 1 (tensor/ndarray/list)."""
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) >= 1 and int(shape[0]) == 1:
        return value[0]
    if isinstance(value, (list, tuple)) and len(value) == 1 and \
            isinstance(value[0], (list, tuple)):
        return value[0]
    return value


def prepare_single_coreai_observation(
    batch: dict[str, Any], manifest: Any, *, require_single: bool = True,
) -> tuple[dict[str, Any], str]:
    """Convert a LeRobot B=1 batch into a JSON-safe CoreAI observation + hash.

    Fail-closed: B>1 is rejected (v1.3.3), unknown values are refused by the
    serializer, and only manifest-declared observation features (plus task) pass.
    """
    b = infer_batch_size(batch)
    if require_single and b > 1:
        raise CoreAIPolicyError(
            f"coreai_bridge transport supports only batch_size=1 (got {b}); "
            "batched transport lands in v1.3.3.")

    # Keep only declared observation inputs (+ task) — never the ground-truth action.
    obs = extract_observation(dict(batch), manifest)

    single: dict[str, Any] = {}
    for k, v in obs.items():
        if k == "task":
            single[k] = v[0] if isinstance(v, (list, tuple)) and v else v
            continue
        single[k] = serialize_value(_strip_leading_batch(v))
    return single, observation_sha256(single)
