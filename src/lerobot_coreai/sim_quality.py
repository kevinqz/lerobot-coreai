# sim_quality.py — run quality gates for sim mode (v0.8.3).
#
# Evaluates a sim run's analytics against configurable thresholds. By default,
# quality gates are report-only (they don't fail the run). With
# --quality.fail-on-quality, a failed gate sets result.ok=False.
#
# Quality gates are development signals, not safety proof. They do not prove
# physical robot safety or real-world task success.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SimQualityConfig:
    """Thresholds for sim run quality evaluation.

    None means the check is disabled.
    """

    min_success_rate: float | None = None
    min_mean_reward: float | None = None
    max_runner_p95_ms: float | None = None
    max_env_step_p95_ms: float | None = None
    max_loop_p95_ms: float | None = None
    max_error_rate: float = 0.0
    max_nan_action_steps: int = 0
    max_inf_action_steps: int = 0
    allow_action_shape_changes: bool = False


@dataclass
class SimQualityResult:
    """Result of sim quality evaluation."""

    passed: bool
    checks: list[dict[str, Any]] = field(default_factory=list)


def evaluate_sim_quality(
    analytics: dict[str, Any],
    config: SimQualityConfig,
    *,
    error_rate: float = 0.0,
) -> SimQualityResult:
    """Evaluate a sim run's analytics against quality thresholds.

    Args:
        analytics: The analytics block (episode_metrics/latency_metrics/
            action_metrics/failure_metrics) from sim_report.json.
        config: Threshold configuration.
        error_rate: Fraction of steps that errored (0.0 = all passed).

    Returns:
        SimQualityResult with passed flag and per-check details.
    """
    checks: list[dict[str, Any]] = []
    episode_m = analytics.get("episode_metrics", {}) or {}
    latency_m = analytics.get("latency_metrics", {}) or {}
    action_m = analytics.get("action_metrics", {}) or {}

    # Min success rate.
    if config.min_success_rate is not None:
        value = episode_m.get("success_rate")
        passed = value is not None and value >= config.min_success_rate
        checks.append({
            "name": "min_success_rate", "passed": passed,
            "value": value, "threshold": config.min_success_rate,
        })

    # Min mean reward.
    if config.min_mean_reward is not None:
        value = episode_m.get("mean_reward")
        passed = value is not None and value >= config.min_mean_reward
        checks.append({
            "name": "min_mean_reward", "passed": passed,
            "value": value, "threshold": config.min_mean_reward,
        })

    # Runner p95 latency.
    if config.max_runner_p95_ms is not None:
        value = latency_m.get("runner_p95_ms")
        passed = value is not None and value <= config.max_runner_p95_ms
        checks.append({
            "name": "max_runner_p95_ms", "passed": passed,
            "value": value, "threshold": config.max_runner_p95_ms,
        })

    # Env step p95 latency.
    if config.max_env_step_p95_ms is not None:
        value = latency_m.get("env_step_p95_ms")
        passed = value is not None and value <= config.max_env_step_p95_ms
        checks.append({
            "name": "max_env_step_p95_ms", "passed": passed,
            "value": value, "threshold": config.max_env_step_p95_ms,
        })

    # Loop p95 latency.
    if config.max_loop_p95_ms is not None:
        value = latency_m.get("loop_p95_ms")
        passed = value is not None and value <= config.max_loop_p95_ms
        checks.append({
            "name": "max_loop_p95_ms", "passed": passed,
            "value": value, "threshold": config.max_loop_p95_ms,
        })

    # Error rate.
    checks.append({
        "name": "max_error_rate", "passed": error_rate <= config.max_error_rate,
        "value": error_rate, "threshold": config.max_error_rate,
    })

    # NaN action steps.
    nan_steps = action_m.get("nan_action_steps", 0)
    checks.append({
        "name": "max_nan_action_steps", "passed": nan_steps <= config.max_nan_action_steps,
        "value": nan_steps, "threshold": config.max_nan_action_steps,
    })

    # Inf action steps.
    inf_steps = action_m.get("inf_action_steps", 0)
    checks.append({
        "name": "max_inf_action_steps", "passed": inf_steps <= config.max_inf_action_steps,
        "value": inf_steps, "threshold": config.max_inf_action_steps,
    })

    # Shape changes.
    if not config.allow_action_shape_changes:
        shape_changes = action_m.get("shape_changes", 0)
        checks.append({
            "name": "no_action_shape_changes", "passed": shape_changes == 0,
            "value": shape_changes, "threshold": 0,
        })

    overall_passed = all(c["passed"] for c in checks)
    return SimQualityResult(passed=overall_passed, checks=checks)
