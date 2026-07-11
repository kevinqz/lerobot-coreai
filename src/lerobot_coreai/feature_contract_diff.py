# feature_contract_diff.py — breaking-change detection for FeatureContract v1 (v1.3.24).
#
# A contract is a compatibility surface. This classifies changes between a baseline
# and a candidate as BREAKING (removed required feature, dtype/axis/layout/units/
# normalization-owner/names/stage change, incompatible range reduction) or
# NON-BREAKING (new optional feature, explicit range expansion, new alias, added
# descriptive metadata). Pure Python.

from __future__ import annotations

from dataclasses import dataclass, field

from .feature_contract import FeatureContract, FeatureSpec


@dataclass
class FeatureContractDiff:
    breaking: list[str] = field(default_factory=list)
    non_breaking: list[str] = field(default_factory=list)

    @property
    def is_breaking(self) -> bool:
        return bool(self.breaking)

    def to_dict(self) -> dict:
        return {"breaking": self.breaking, "non_breaking": self.non_breaking,
                "is_breaking": self.is_breaking}


def _by_id(c: FeatureContract) -> dict[str, FeatureSpec]:
    return {s.feature_id: s for s in c.all_specs()}


def diff_feature_contracts(baseline: FeatureContract,
                           candidate: FeatureContract) -> FeatureContractDiff:
    d = FeatureContractDiff()
    base, cand = _by_id(baseline), _by_id(candidate)

    for fid, b in base.items():
        c = cand.get(fid)
        if c is None:
            (d.breaking if b.required else d.non_breaking).append(
                f"removed {'required' if b.required else 'optional'} feature {fid}")
            continue
        if b.dtype != c.dtype:
            d.breaking.append(f"{fid}: dtype {b.dtype} -> {c.dtype}")
        if b.axes != c.axes or b.layout != c.layout:
            d.breaking.append(f"{fid}: axis/layout change")
        if tuple(b.shape) != tuple(c.shape):
            d.breaking.append(f"{fid}: shape {b.shape} -> {c.shape}")
        if b.units != c.units:
            d.breaking.append(f"{fid}: units {b.units} -> {c.units}")
        if b.names != c.names:
            d.breaking.append(f"{fid}: names/order change")
        if b.normalization.owner != c.normalization.owner:
            d.breaking.append(
                f"{fid}: normalization owner {b.normalization.owner} -> "
                f"{c.normalization.owner}")
        # a range reduction is breaking; an explicit expansion is non-breaking.
        bl, bh = b.value_domain.minimum, b.value_domain.maximum
        cl, ch = c.value_domain.minimum, c.value_domain.maximum
        if (bl is not None and cl is not None and cl > bl) or \
           (bh is not None and ch is not None and ch < bh):
            d.breaking.append(f"{fid}: incompatible range reduction")
        elif (bl is not None and cl is not None and cl < bl) or \
             (bh is not None and ch is not None and ch > bh):
            d.non_breaking.append(f"{fid}: range expansion")

    for fid, c in cand.items():
        if fid not in base:
            (d.non_breaking if not c.required else d.breaking).append(
                f"added {'required' if c.required else 'optional'} feature {fid}")
    return d
