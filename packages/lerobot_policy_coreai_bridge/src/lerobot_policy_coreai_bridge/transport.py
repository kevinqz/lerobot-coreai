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

# Observation transport encodings the plugin can emit. nested_json_v1 sends the
# model plain nested JSON arrays (the default, safe for a runner that expects
# JSON); typed_array_envelope_v1 sends {"__array__",dtype,shape} and may ONLY be
# used when the runner announces it.
NESTED_JSON_V1 = "nested_json_v1"
TYPED_ARRAY_ENVELOPE_V1 = "typed_array_envelope_v1"
VALID_ENCODINGS = (NESTED_JSON_V1, TYPED_ARRAY_ENVELOPE_V1)


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


def _plain_and_shape(value: Any) -> tuple[Any, tuple[int, ...] | None]:
    """Return (plain nested list / scalar, shape) for a tensor/ndarray/list/scalar."""
    tolist = getattr(value, "tolist", None)
    shape = getattr(value, "shape", None)
    if tolist is not None and shape is not None:
        try:
            return tolist(), tuple(int(s) for s in shape)
        except Exception:
            pass
    if isinstance(value, (list, tuple)):
        v = list(value)
        return v, (len(v),) if (not v or not isinstance(v[0], (list, tuple))) else None
    return value, ()


def prepare_single_coreai_observation(
    batch: dict[str, Any], manifest: Any, *, require_single: bool = True,
    encoding: str = NESTED_JSON_V1, audit: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    """Convert a LeRobot B=1 batch into a JSON-safe CoreAI observation + hash.

    Fail-closed: B>1 is rejected (batched transport = v1.3.4), unknown values are
    refused, and only manifest-declared observation features (plus task) pass.
    Shape is validated (recorded in ``audit``) BEFORE encoding, so the envelope
    can never mask a shape mismatch. ``encoding`` selects the wire format:
    ``nested_json_v1`` (plain lists, default) or ``typed_array_envelope_v1``.
    """
    if encoding not in VALID_ENCODINGS:
        raise CoreAIPolicyError(f"unknown observation encoding {encoding!r}.")
    b = infer_batch_size(batch)
    if require_single and b > 1:
        raise CoreAIPolicyError(
            f"coreai_bridge transport supports only batch_size=1 (got {b}); "
            "batched transport lands in v1.3.4.")

    # Keep only declared observation inputs (+ task) — never the ground-truth action.
    obs = extract_observation(dict(batch), manifest)
    expected = getattr(manifest, "observation_features", None) if manifest else None

    single: dict[str, Any] = {}
    for k, v in obs.items():
        if k == "task":
            single[k] = v[0] if isinstance(v, (list, tuple)) and v else v
            continue
        stripped = _strip_leading_batch(v)
        values, shape = _plain_and_shape(stripped)
        # Validate shape against the manifest BEFORE encoding.
        if expected and k in expected:
            want = getattr(expected[k], "shape", None)
            if want is not None and shape is not None and list(shape) != list(want):
                raise CoreAIPolicyError(
                    f"observation {k!r} shape {list(shape)} != manifest {list(want)}.")
        if audit is not None:
            audit[k] = {"shape": list(shape) if shape is not None else None}
        if encoding == NESTED_JSON_V1:
            single[k] = values          # plain nested JSON — no typed envelope
        else:
            single[k] = serialize_value(stripped)  # typed_array_envelope_v1
    return single, observation_sha256(single)
