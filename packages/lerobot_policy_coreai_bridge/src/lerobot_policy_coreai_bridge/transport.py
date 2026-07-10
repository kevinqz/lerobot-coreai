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

import hashlib
import json
from typing import Any

from lerobot_coreai.coreai_observation_serialization import (
    extract_observation, observation_sha256, serialize_value,
)
from lerobot_coreai.errors import CoreAIPolicyError


def _require_declared_features(obs: dict[str, Any], manifest: Any) -> None:
    """Fail if a REQUIRED manifest observation feature is missing at runtime (v1.3.10).

    The factory cross-binding proves the dataset declares the feature; this guards
    the per-request observation actually carrying it.
    """
    expected = getattr(manifest, "observation_features", None) if manifest else None
    if not expected:
        return
    for name, spec in expected.items():
        required = getattr(spec, "required", None)
        if required is None and isinstance(spec, dict):
            required = spec.get("required", True)
        # v1.3.12: task requiredness is enforced like any declared feature — a
        # required task that is absent fails; an optional one may be absent.
        if required and name not in obs:
            raise CoreAIPolicyError(
                f"required observation feature {name!r} is missing from the runtime "
                f"observation (present: {sorted(obs)}).")


def _require_str_task(value: Any) -> str:
    """A task must already BE a string — never silently coerce None/int/dict (P1.7)."""
    if not isinstance(value, str):
        raise CoreAIPolicyError(
            f"task must be a string; got {type(value).__name__} {value!r}.")
    return value


def canonical_batch_sha256(batch_size: int, sample_shas: list[str], mode: str) -> str:
    """Order-sensitive batch hash over (batch_size, mode, ordered sample hashes)."""
    canon = json.dumps({"batch_size": int(batch_size), "mode": mode,
                        "ordered_sample_sha256s": list(sample_shas)},
                       separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canon.encode()).hexdigest()

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


def infer_and_validate_batch_size(batch: dict[str, Any], manifest: Any = None) -> int:
    """Single strict batch-size boundary (v1.3.9).

    Every leading dimension of a manifest-declared observation feature — and the
    ``task`` list length, if present — must be EXACTLY equal. A ragged batch
    (state B=4 but task length 2) fails closed rather than defaulting to max.
    """
    expected = getattr(manifest, "observation_features", None) if manifest else None
    sizes: dict[str, int] = {}
    for k, v in batch.items():
        if k == "task":
            if isinstance(v, (list, tuple)):
                sizes["task"] = len(v)
            continue
        if expected is not None and k not in expected:
            continue  # ignore stray label keys (action/reward/index/…)
        shape = getattr(v, "shape", None)
        if shape is not None and len(shape) >= 1:
            sizes[k] = int(shape[0])
        elif isinstance(v, (list, tuple)) and v and isinstance(v[0], (list, tuple)):
            sizes[k] = len(v)
    if not sizes:
        return 1
    distinct = set(sizes.values())
    if len(distinct) > 1:
        raise CoreAIPolicyError(
            f"inconsistent batch sizes across observation keys: {sizes}.")
    return next(iter(distinct))


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
    _require_declared_features(obs, manifest)

    single: dict[str, Any] = {}
    for k, v in obs.items():
        if k == "task":
            unwrapped = v[0] if isinstance(v, (list, tuple)) and v else v
            single[k] = _require_str_task(unwrapped)
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


def _sample_slice(value: Any, i: int) -> Any:
    """Return sample ``i`` keeping a leading singleton dim ([B,...] -> [1,...])."""
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) >= 1:
        return value[i:i + 1]
    if isinstance(value, (list, tuple)):
        return [value[i]]
    return value


def split_coreai_observations(
    batch: dict[str, Any], manifest: Any, *, batch_size: int,
    encoding: str = NESTED_JSON_V1,
) -> list[tuple[dict[str, Any], str]]:
    """Split a B-observation batch into B independent single observations (v1.3.9).

    Each sample reuses the strict B=1 boundary, so every per-sample payload keeps
    only manifest-declared features, drops labels, unwraps its task, is shape-
    validated, and carries its own sha256. Used by split-and-stack (stateless only).
    """
    out: list[tuple[dict[str, Any], str]] = []
    for i in range(batch_size):
        sample: dict[str, Any] = {}
        for k, v in batch.items():
            if k == "task" and isinstance(v, (list, tuple)):
                sample[k] = [v[i]]
            else:
                sample[k] = _sample_slice(v, i)
        obs, sha = prepare_single_coreai_observation(
            sample, manifest, require_single=True, encoding=encoding)
        out.append((obs, sha))
    return out


def prepare_batched_coreai_observation(
    batch: dict[str, Any], manifest: Any, *, batch_size: int,
    encoding: str = NESTED_JSON_V1, audit: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str, list[str]]:
    """Build a native [B, ...] CoreAI observation payload (v1.3.9).

    Keeps only manifest-declared features, drops labels, requires ``task`` to be a
    list[str] of length B, validates each feature's leading dim == B and per-sample
    shape against the manifest, and returns (payload, batch_sha256, sample_sha256s).
    """
    if encoding not in VALID_ENCODINGS:
        raise CoreAIPolicyError(f"unknown observation encoding {encoding!r}.")
    b = infer_and_validate_batch_size(batch, manifest)
    if b != batch_size:
        raise CoreAIPolicyError(
            f"batched observation size {b} != requested {batch_size}.")

    obs = extract_observation(dict(batch), manifest)
    expected = getattr(manifest, "observation_features", None) if manifest else None
    _require_declared_features(obs, manifest)

    payload: dict[str, Any] = {}
    for k, v in obs.items():
        if k == "task":
            if not isinstance(v, (list, tuple)) or len(v) != batch_size:
                raise CoreAIPolicyError(
                    f"task must be a list[str] of length {batch_size}; got {v!r}.")
            payload[k] = [_require_str_task(t) for t in v]   # no silent coercion
            continue
        values, shape = _plain_and_shape(v)
        if shape is None or len(shape) < 1 or int(shape[0]) != batch_size:
            raise CoreAIPolicyError(
                f"batched feature {k!r} must have leading dim {batch_size}; "
                f"got shape {list(shape) if shape else None}.")
        if expected and k in expected:
            want = getattr(expected[k], "shape", None)
            if want is not None and list(shape[1:]) != list(want):
                raise CoreAIPolicyError(
                    f"batched {k!r} per-sample shape {list(shape[1:])} != manifest "
                    f"{list(want)}.")
        if audit is not None:
            audit[k] = {"shape": list(shape)}
        payload[k] = values if encoding == NESTED_JSON_V1 else serialize_value(v)

    # Per-sample hashes (over the same single-obs form the runner would see).
    sample_hashes: list[str] = []
    for obs_i, sha_i in split_coreai_observations(
            batch, manifest, batch_size=batch_size, encoding=encoding):
        sample_hashes.append(sha_i)
    return payload, observation_sha256(payload), sample_hashes
