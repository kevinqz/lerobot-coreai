# shadow_quality.py — run quality gates for shadow mode (v0.7.2).
#
# Evaluates a live metrics summary against configurable thresholds. By default,
# quality gates are report-only (they don't fail the run). With --quality.fail-on-quality,
# a failed gate sets result.ok=False.
#
# Quality gates are development signals, not safety proof. They do not prove
# physical robot safety.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ShadowQualityConfig:
    """Thresholds for shadow run quality evaluation.

    None means the check is disabled.
    """

    max_runner_p95_ms: float | None = None
    max_loop_p95_ms: float | None = None
    max_error_rate: float = 0.0
    max_nan_actions: int = 0
    max_inf_actions: int = 0
    allow_action_shape_changes: bool = False
    min_effective_fps: float | None = None


@dataclass
class ShadowQualityResult:
    """Result of quality evaluation."""

    passed: bool
    checks: list[dict[str, Any]] = field(default_factory=list)


def evaluate_shadow_quality(
    summary: dict[str, Any],
    config: ShadowQualityConfig,
    *,
    error_rate: float = 0.0,
) -> ShadowQualityResult:
    """Evaluate a live metrics summary against quality thresholds.

    Args:
        summary: The LiveMetricsCollector.summary() dict.
        config: Threshold configuration.
        error_rate: Fraction of steps that errored (0.0 = all passed).

    Returns:
        ShadowQualityResult with passed flag and per-check details.
    """
    checks: list[dict[str, Any]] = []

    # Runner p95 latency.
    if config.max_runner_p95_ms is not None:
        value = summary.get("p95_runner_ms")
        passed = value is not None and value <= config.max_runner_p95_ms
        checks.append({
            "name": "max_runner_p95_ms",
            "passed": passed,
            "value": value,
            "threshold": config.max_runner_p95_ms,
        })

    # Loop p95 latency.
    if config.max_loop_p95_ms is not None:
        value = summary.get("p95_loop_ms")
        passed = value is not None and value <= config.max_loop_p95_ms
        checks.append({
            "name": "max_loop_p95_ms",
            "passed": passed,
            "value": value,
            "threshold": config.max_loop_p95_ms,
        })

    # Error rate.
    checks.append({
        "name": "max_error_rate",
        "passed": error_rate <= config.max_error_rate,
        "value": error_rate,
        "threshold": config.max_error_rate,
    })

    # NaN actions.
    nan_count = summary.get("nan_actions", 0)
    checks.append({
        "name": "max_nan_actions",
        "passed": nan_count <= config.max_nan_actions,
        "value": nan_count,
        "threshold": config.max_nan_actions,
    })

    # Inf actions.
    inf_count = summary.get("inf_actions", 0)
    checks.append({
        "name": "max_inf_actions",
        "passed": inf_count <= config.max_inf_actions,
        "value": inf_count,
        "threshold": config.max_inf_actions,
    })

    # Shape changes.
    if not config.allow_action_shape_changes:
        shape_changes = summary.get("shape_changes", 0)
        checks.append({
            "name": "no_action_shape_changes",
            "passed": shape_changes == 0,
            "value": shape_changes,
            "threshold": 0,
        })

    # Min effective FPS.
    if config.min_effective_fps is not None:
        fps = summary.get("effective_fps")
        passed = fps is not None and fps >= config.min_effective_fps
        checks.append({
            "name": "min_effective_fps",
            "passed": passed,
            "value": fps,
            "threshold": config.min_effective_fps,
        })

    overall_passed = all(c["passed"] for c in checks)
    return ShadowQualityResult(passed=overall_passed, checks=checks)
