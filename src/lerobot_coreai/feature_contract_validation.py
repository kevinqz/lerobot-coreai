# feature_contract_validation.py — fail-closed FeatureContract v1 validation (v1.3.24).
#
# Two layers: (1) STRUCTURAL validation of the contract itself (canonical stages,
# declared symbols, single normalization owner, names/order coherence); (2) PAYLOAD
# validation of an observed tensor tree against the contract at a specific stage
# (presence, shape with symbol resolution, finiteness, value domain, closed key set).
# Pure Python — no numpy/torch.

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from .feature_contract import (
    FeatureContract, FeatureSpec, MODALITIES, NORMALIZATION_OWNERS,
    NORMALIZATION_STATES, ROLES, SHAPE_SYMBOLS, TEMPORAL_KINDS,
)
from .stages import ACTION_STAGES, OBSERVATION_STAGES


@dataclass
class FeatureValidationResult:
    ok: bool
    stage: str
    validated_feature_ids: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    observed: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "stage": self.stage,
                "validated_feature_ids": self.validated_feature_ids,
                "failures": self.failures, "observed": self.observed}


def _rect_shape(v: Any) -> list[int] | None:
    """Strict rectangular shape; None if ragged (a ragged payload never matches)."""
    if not isinstance(v, list):
        return []
    child = [_rect_shape(x) for x in v]
    if any(c is None for c in child):
        return None
    if child and any(c != child[0] for c in child):
        return None
    return [len(v)] + (child[0] if child else [])


def _finite(v: Any) -> bool:
    if isinstance(v, bool):
        return True
    if isinstance(v, (int, float)):
        return math.isfinite(v)
    if isinstance(v, list):
        return all(_finite(x) for x in v)
    return False


def _numeric_leaves(v: Any):
    if isinstance(v, list):
        for x in v:
            yield from _numeric_leaves(x)
    elif isinstance(v, (int, float)) and not isinstance(v, bool):
        yield float(v)


def validate_contract_structure(contract: FeatureContract) -> list[str]:
    """Return a list of structural errors (empty = structurally valid)."""
    errs: list[str] = []
    seen_ids: set[str] = set()
    # a key may be `normalized` by only ONE owner (no double normalization).
    norm_owner_by_key: dict[str, str] = {}
    for spec in contract.all_specs():
        if spec.feature_id in seen_ids:
            errs.append(f"duplicate feature_id {spec.feature_id}")
        seen_ids.add(spec.feature_id)
        if spec.role not in ROLES:
            errs.append(f"{spec.feature_id}: unknown role {spec.role!r}")
        if spec.modality not in MODALITIES:
            errs.append(f"{spec.feature_id}: unknown modality {spec.modality!r}")
        valid_stages = OBSERVATION_STAGES if spec.role != "action" else ACTION_STAGES
        # context features live on observation-side stages.
        if spec.role == "context":
            valid_stages = OBSERVATION_STAGES
        if spec.stage not in valid_stages:
            errs.append(f"{spec.feature_id}: non-canonical stage {spec.stage!r}")
        for dim in spec.shape:
            if isinstance(dim, str) and dim not in SHAPE_SYMBOLS:
                errs.append(f"{spec.feature_id}: undeclared shape symbol {dim!r}")
            if isinstance(dim, int) and dim < 0:
                errs.append(f"{spec.feature_id}: negative shape dim {dim}")
        if len(spec.axes) != len(spec.shape):
            errs.append(f"{spec.feature_id}: axes/shape rank mismatch")
        # H only in chunk stages, not on a selected/environment action.
        if "H" in spec.shape and spec.stage in (
                "lerobot_selected_policy_action.v1", "environment_action.v1"):
            errs.append(f"{spec.feature_id}: horizon H not allowed at stage {spec.stage}")
        if spec.names is not None:
            # names must match the last non-symbolic dim (the component dimension).
            comp = next((d for d in reversed(spec.shape) if isinstance(d, int)), None)
            if comp is not None and len(spec.names) != comp:
                errs.append(f"{spec.feature_id}: {len(spec.names)} names != dim {comp}")
        nz = spec.normalization
        if nz.state not in NORMALIZATION_STATES:
            errs.append(f"{spec.feature_id}: bad normalization state {nz.state!r}")
        if nz.owner is not None and nz.owner not in NORMALIZATION_OWNERS:
            errs.append(f"{spec.feature_id}: bad normalization owner {nz.owner!r}")
        if nz.state == "normalized" and nz.owner is None:
            errs.append(f"{spec.feature_id}: normalized without an owner")
        if nz.state == "normalized" and nz.owner is not None:
            prev = norm_owner_by_key.get(spec.key)
            if prev is not None and prev != nz.owner:
                errs.append(f"{spec.key}: double normalization owners "
                            f"{prev!r} vs {nz.owner!r}")
            norm_owner_by_key[spec.key] = nz.owner
    return errs


def _resolve_shape(shape, symbols: Mapping[str, int]) -> list[int] | None:
    out = []
    for d in shape:
        if isinstance(d, int):
            out.append(d)
        elif d in symbols:
            out.append(int(symbols[d]))
        else:
            return None                       # unresolved symbol
    return out


def validate_payload_against_feature_contract(
    payload: Mapping[str, Any], contract: FeatureContract, *, stage: str,
    symbols: Mapping[str, int], closed: bool = True,
) -> FeatureValidationResult:
    """Validate an observed tensor tree against the contract at ``stage``."""
    res = FeatureValidationResult(ok=True, stage=stage)

    def fail(msg: str):
        res.failures.append(msg)
        res.ok = False

    specs = [s for s in contract.all_specs() if s.stage == stage]
    contract_keys = {s.key for s in specs}
    # closed mode: no key outside the contract (task is always permitted context).
    if closed:
        for k in payload:
            if k not in contract_keys and k != "task":
                fail(f"unexpected feature {k!r} not in contract at stage {stage}")

    for spec in specs:
        before = len(res.failures)
        if spec.key not in payload:
            if spec.required:
                fail(f"missing required feature {spec.key}")
            continue
        value = payload[spec.key]
        # task/text: type by batch, not shape.
        if spec.modality == "text":
            res.validated_feature_ids.append(spec.feature_id)
            continue
        want = _resolve_shape(spec.shape, symbols)
        if want is None:
            fail(f"{spec.feature_id}: unresolved shape symbol in {spec.shape}")
            continue
        got = _rect_shape(value)
        if got is None:
            fail(f"{spec.feature_id}: ragged payload")
            continue
        if got != want:
            fail(f"{spec.feature_id}: shape {got} != expected {want}")
            continue
        if spec.value_domain.finite and not _finite(value):
            fail(f"{spec.feature_id}: non-finite value")
            continue
        leaves = list(_numeric_leaves(value))
        lo, hi = spec.value_domain.minimum, spec.value_domain.maximum
        if leaves and lo is not None and min(leaves) < lo:
            fail(f"{spec.feature_id}: value {min(leaves)} < minimum {lo}")
        if leaves and hi is not None and max(leaves) > hi:
            fail(f"{spec.feature_id}: value {max(leaves)} > maximum {hi}")
        res.observed[spec.feature_id] = {
            "shape": got, "finite": _finite(value),
            "min": min(leaves) if leaves else None,
            "max": max(leaves) if leaves else None}
        if len(res.failures) == before:       # this spec validated cleanly
            res.validated_feature_ids.append(spec.feature_id)
    return res
