# processor_parity_metrics.py — parity metrics over nested numeric arrays (v1.3.26).
#
# Exact structural parity uses canonical-hash / exact equality; numeric parity uses
# max/mean absolute error, relative MAE, cosine similarity and non-finite counts. All
# thresholds live in the contract/report, never hardcoded implicitly. Pure Python.

from __future__ import annotations

import math
from dataclasses import dataclass


def _flatten(v):
    if isinstance(v, list):
        for x in v:
            yield from _flatten(x)
    else:
        yield v


def _shape(v):
    s = []
    cur = v
    while isinstance(cur, list):
        s.append(len(cur))
        cur = cur[0] if cur else None
    return tuple(s)


@dataclass(frozen=True)
class ParityMetrics:
    shape_ref: tuple
    shape_cand: tuple
    shape_match: bool
    count: int
    nonfinite_ref: int
    nonfinite_cand: int
    max_abs_error: float
    mean_abs_error: float
    relative_mae: float
    cosine_similarity: float

    def to_dict(self) -> dict:
        return {"shape_ref": list(self.shape_ref), "shape_cand": list(self.shape_cand),
                "shape_match": self.shape_match, "count": self.count,
                "nonfinite_ref": self.nonfinite_ref, "nonfinite_cand": self.nonfinite_cand,
                "max_abs_error": self.max_abs_error, "mean_abs_error": self.mean_abs_error,
                "relative_mae": self.relative_mae,
                "cosine_similarity": self.cosine_similarity}


def compute_parity_metrics(reference, candidate) -> ParityMetrics:
    """Element-wise parity metrics between two nested numeric arrays.

    A shape mismatch yields shape_match=False and infinite errors (never silently
    broadcast/pad). Non-finite values are counted and force infinite error."""
    sr, sc = _shape(reference), _shape(candidate)
    ref = [float(x) for x in _flatten(reference)]
    cand = [float(x) for x in _flatten(candidate)]
    nf_ref = sum(0 if math.isfinite(x) else 1 for x in ref)
    nf_cand = sum(0 if math.isfinite(x) else 1 for x in cand)
    if sr != sc or len(ref) != len(cand):
        return ParityMetrics(sr, sc, False, max(len(ref), len(cand)), nf_ref, nf_cand,
                             math.inf, math.inf, math.inf, 0.0)
    if nf_ref or nf_cand:
        return ParityMetrics(sr, sc, True, len(ref), nf_ref, nf_cand,
                             math.inf, math.inf, math.inf, 0.0)
    if not ref:
        return ParityMetrics(sr, sc, True, 0, 0, 0, 0.0, 0.0, 0.0, 1.0)
    diffs = [abs(a - b) for a, b in zip(ref, cand)]
    max_abs = max(diffs)
    mae = sum(diffs) / len(diffs)
    denom = sum(abs(a) for a in ref) / len(ref)
    rel_mae = (mae / denom) if denom > 0 else (0.0 if mae == 0 else math.inf)
    dot = sum(a * b for a, b in zip(ref, cand))
    nr = math.sqrt(sum(a * a for a in ref))
    nc = math.sqrt(sum(b * b for b in cand))
    cos = (dot / (nr * nc)) if nr > 0 and nc > 0 else (1.0 if nr == nc else 0.0)
    return ParityMetrics(sr, sc, True, len(ref), 0, 0, max_abs, mae, rel_mae, cos)
