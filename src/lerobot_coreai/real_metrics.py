# real_metrics.py — per-step runtime metrics for guarded real sessions (v1.0.5).
#
# Observability only. These numbers describe loop timing; they do not prove
# physical safety or real-world success.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 4)


@dataclass
class RealMetricsCollector:
    """Accumulates per-step timing for a guarded real session."""

    fps: float
    steps: list[dict[str, float]] = field(default_factory=list)

    def add(self, *, observation_ms: float, policy_ms: float, egress_ms: float,
            loop_ms: float) -> None:
        self.steps.append({
            "observation_ms": round(observation_ms, 4),
            "policy_ms": round(policy_ms, 4),
            "egress_ms": round(egress_ms, 4),
            "loop_ms": round(loop_ms, 4),
        })

    def _col(self, key: str) -> list[float]:
        return [s[key] for s in self.steps]

    def summary(self, *, wall_seconds: float | None = None) -> dict[str, Any]:
        n = len(self.steps)
        deadline_ms = (1000.0 / self.fps) if self.fps and self.fps > 0 else None
        missed = 0
        if deadline_ms is not None:
            missed = sum(1 for s in self.steps if s["loop_ms"] > deadline_ms)
        effective_fps = round(n / wall_seconds, 4) if wall_seconds and wall_seconds > 0 else None

        def _agg(key):
            vals = self._col(key)
            return {"p50": _percentile(vals, 0.5), "p95": _percentile(vals, 0.95),
                    "max": max(vals) if vals else None}

        return {
            "steps": n,
            "fps_target": self.fps,
            "effective_fps": effective_fps,
            "missed_deadline_count": missed,
            "observation_latency_ms": _agg("observation_ms"),
            "policy_latency_ms": _agg("policy_ms"),
            "egress_latency_ms": _agg("egress_ms"),
            "loop_ms": _agg("loop_ms"),
        }


def build_real_metrics_report(collector: RealMetricsCollector, *,
                              wall_seconds: float | None = None) -> dict[str, Any]:
    return {
        "schema_version": "lerobot-coreai.real_metrics.v0",
        "summary": collector.summary(wall_seconds=wall_seconds),
        "per_step": collector.steps,
    }


def build_real_metrics_markdown(report: dict[str, Any]) -> str:
    s = report.get("summary", {})
    def _line(name, agg):
        agg = agg or {}
        return f"- {name}: p50={agg.get('p50')} p95={agg.get('p95')} max={agg.get('max')}"
    return (
        "# Real Session Metrics\n\n"
        f"- Steps: {s.get('steps')}\n"
        f"- Target fps: {s.get('fps_target')}\n"
        f"- Effective fps: {s.get('effective_fps')}\n"
        f"- Missed deadlines: {s.get('missed_deadline_count')}\n"
        f"{_line('observation_latency_ms', s.get('observation_latency_ms'))}\n"
        f"{_line('policy_latency_ms', s.get('policy_latency_ms'))}\n"
        f"{_line('egress_latency_ms', s.get('egress_latency_ms'))}\n"
        f"{_line('loop_ms', s.get('loop_ms'))}\n\n"
        "Observability only — does not prove physical safety or real-world success.\n"
    )


def write_real_metrics_csv(path, steps: list[dict[str, float]]) -> None:
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "observation_ms", "policy_ms", "egress_ms", "loop_ms"])
        for i, s in enumerate(steps):
            w.writerow([i, s["observation_ms"], s["policy_ms"], s["egress_ms"], s["loop_ms"]])
