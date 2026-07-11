# processor_transform_contract.py — declared, replayable processor transforms (v1.3.26).
#
# A TransformContract declares an ordered op list (permute / cast / scale /
# normalize / denormalize) between two canonical stages, bound to input/output
# FeatureContract hashes. An INDEPENDENT reference applies the declared ops; the
# candidate is the runtime path. Comparing both against known-expected fixtures (with
# independent code paths) is the redteam mitigation against a shared buggy impl. Pure
# Python (nested lists) — no numpy/torch.

from __future__ import annotations

from .rollout_evidence_schema import canonical_json_sha256
from .stages import ACTION_STAGES, OBSERVATION_STAGES

PROCESSOR_TRANSFORM_SCHEMA_VERSION = "lerobot-coreai.processor-transform.v1"
_OPS = ("permute", "cast", "scale", "normalize", "denormalize")
_ALL_STAGES = tuple(OBSERVATION_STAGES) + tuple(ACTION_STAGES)

PROCESSOR_TRANSFORM_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object", "additionalProperties": False,
    "required": ["schema_version", "transform_id", "owner", "source_stage",
                 "target_stage", "operations"],
    "properties": {
        "schema_version": {"const": PROCESSOR_TRANSFORM_SCHEMA_VERSION},
        "transform_id": {"type": "string", "minLength": 1},
        "owner": {"type": "string", "minLength": 1},
        "source_stage": {"enum": list(_ALL_STAGES)},
        "target_stage": {"enum": list(_ALL_STAGES)},
        "input_feature_contract_sha256": {"type": ["string", "null"]},
        "output_feature_contract_sha256": {"type": ["string", "null"]},
        "operations": {"type": "array", "items": {
            "type": "object", "additionalProperties": True,
            "required": ["op"], "properties": {"op": {"enum": list(_OPS)}}}},
    },
}


class TransformError(ValueError):
    """Raised when a transform op is malformed or inapplicable."""


def _permute(v, order):
    # order: list of source-axis indices; v is a nested list. Build via index walk.
    def shape(x):
        s = []
        while isinstance(x, list):
            s.append(len(x)); x = x[0]
        return s
    src_shape = shape(v)
    if len(order) != len(src_shape):
        raise TransformError("permute order rank != input rank")

    def get(idx):
        cur = v
        for i in idx:
            cur = cur[i]
        return cur
    out_shape = [src_shape[order[i]] for i in range(len(order))]

    def build(prefix):
        d = len(prefix)
        if d == len(out_shape):
            src_idx = [0] * len(order)
            for out_ax, val in enumerate(prefix):
                src_idx[order[out_ax]] = val
            return get(src_idx)
        return [build(prefix + [i]) for i in range(out_shape[d])]
    return build([])


def _map_leaves(v, fn):
    if isinstance(v, list):
        return [_map_leaves(x, fn) for x in v]
    return fn(v)


def apply_operations(value, operations, *, stats=None):
    """Apply the declared ops in order (independent reference implementation)."""
    out = value
    for op in operations:
        kind = op["op"]
        if kind == "permute":
            out = _permute(out, list(op["order"]))
        elif kind == "cast":
            out = _map_leaves(out, float) if op.get("to", "float32").startswith("float") \
                else _map_leaves(out, lambda x: int(x))
        elif kind == "scale":
            f = float(op["factor"])
            out = _map_leaves(out, lambda x: x * f)
        elif kind == "normalize":
            m = op.get("mean"); s = op.get("std")
            if m is None or s is None:
                if not stats:
                    raise TransformError("normalize requires mean/std or stats")
                m, s = stats["mean"], stats["std"]
            out = _affine(out, m, s, invert=False)
        elif kind == "denormalize":
            m, s = op.get("mean"), op.get("std")
            if m is None or s is None:
                if not stats:
                    raise TransformError("denormalize requires mean/std or stats")
                m, s = stats["mean"], stats["std"]
            out = _affine(out, m, s, invert=True)
        else:  # pragma: no cover - schema-guarded
            raise TransformError(f"unknown op {kind}")
    return out


def _affine(v, mean, std, *, invert: bool):
    """Per-last-axis affine: normalize (x-mean)/std or denormalize x*std+mean."""
    def rec(x, depth):
        if isinstance(x[0], list):
            return [rec(sub, depth) for sub in x]
        out = []
        for i, val in enumerate(x):
            m = mean[i] if isinstance(mean, list) else mean
            s = std[i] if isinstance(std, list) else std
            out.append((val * s + m) if invert else ((val - m) / s))
        return out
    if not isinstance(v, list):
        raise TransformError("affine requires a list")
    return rec(v, 0)


def transform_contract_sha256(contract: dict) -> str:
    return canonical_json_sha256(contract)
