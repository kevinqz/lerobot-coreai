# live_metrics.py — live metrics collection for shadow mode (v0.7.2).
#
# Collects per-step timing, action diagnostics, and runtime signals. Summarizes them
# for the shadow report and quality gates. No actuation — metrics only.

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .metrics import action_to_flat_float_list, infer_shape


@dataclass
class LiveMetricSample:
    """A single step's metrics."""

    step: int
    ts: str
    loop_ms: float | None = None
    runner_ms: float | None = None
    observation_ms: float | None = None
    serialization_ms: float | None = None
    action_shape: list[int] | None = None
    action_mean_abs: float | None = None
    action_max_abs: float | None = None
    action_nan_count: int = 0
    action_inf_count: int = 0
    ok: bool = True
    error_type: str | None = None


@dataclass
class LiveMetricsCollector:
    """Accumulates per-step samples and produces a summary."""

    samples: list[LiveMetricSample] = field(default_factory=list)

    def add(self, sample: LiveMetricSample) -> None:
        self.samples.append(sample)

    def summary(self) -> dict[str, Any]:
        """Produce a summary dict for the shadow report."""
        if not self.samples:
            return {
                "samples": 0,
                "mean_loop_ms": None,
                "p50_loop_ms": None,
                "p95_loop_ms": None,
                "max_loop_ms": None,
                "mean_runner_ms": None,
                "p95_runner_ms": None,
                "effective_fps": None,
                "latency_spikes": 0,
                "nan_actions": 0,
                "inf_actions": 0,
                "shape_changes": 0,
            }

        loop_times = [s.loop_ms for s in self.samples if s.loop_ms is not None]
        runner_times = [s.runner_ms for s in self.samples if s.runner_ms is not None]
        total_duration_s = self._total_duration()
        nan_count = sum(s.action_nan_count for s in self.samples)
        inf_count = sum(s.action_inf_count for s in self.samples)

        # Shape changes: count steps where shape differs from previous step.
        shape_changes = self._count_shape_changes()

        # Latency spikes: steps where loop_ms > 2x mean.
        latency_spikes = 0
        if loop_times:
            mean_loop = sum(loop_times) / len(loop_times)
            latency_spikes = sum(1 for t in loop_times if t > 2 * mean_loop)

        effective_fps = None
        if total_duration_s and total_duration_s > 0:
            effective_fps = len(self.samples) / total_duration_s

        return {
            "samples": len(self.samples),
            "mean_loop_ms": _mean(loop_times),
            "p50_loop_ms": _percentile(loop_times, 50),
            "p95_loop_ms": _percentile(loop_times, 95),
            "max_loop_ms": max(loop_times) if loop_times else None,
            "mean_runner_ms": _mean(runner_times),
            "p95_runner_ms": _percentile(runner_times, 95),
            "effective_fps": effective_fps,
            "latency_spikes": latency_spikes,
            "nan_actions": nan_count,
            "inf_actions": inf_count,
            "shape_changes": shape_changes,
        }

    def _total_duration(self) -> float | None:
        """Estimate total duration from loop_ms samples."""
        loop_times = [s.loop_ms for s in self.samples if s.loop_ms is not None]
        if not loop_times:
            return None
        return sum(loop_times) / 1000.0  # ms → s

    def _count_shape_changes(self) -> int:
        changes = 0
        prev_shape = None
        for s in self.samples:
            if s.action_shape is not None:
                if prev_shape is not None and s.action_shape != prev_shape:
                    changes += 1
                prev_shape = s.action_shape
        return changes


# MARK: - Action diagnostics

def summarize_action(action: Any) -> dict[str, Any]:
    """Compute diagnostics for a single action.

    Returns shape, mean_abs, max_abs, nan_count, inf_count.
    """
    shape = infer_shape(action)
    try:
        flat = action_to_flat_float_list(action)
    except Exception:
        flat = []

    if flat:
        mean_abs = sum(abs(v) for v in flat) / len(flat)
        max_abs = max(abs(v) for v in flat)
        nan_count = sum(1 for v in flat if math.isnan(v))
        inf_count = sum(1 for v in flat if math.isinf(v))
    else:
        mean_abs = None
        max_abs = None
        nan_count = 0
        inf_count = 0

    return {
        "shape": shape,
        "mean_abs": mean_abs,
        "max_abs": max_abs,
        "nan_count": nan_count,
        "inf_count": inf_count,
    }


# MARK: - Helpers

def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] + (s[hi] - s[lo]) * frac
