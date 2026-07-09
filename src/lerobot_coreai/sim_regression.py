# sim_regression.py — compare two sim runs for regression (v0.8.3).
#
# Loads a baseline and a candidate sim_report.json, computes deltas on the key
# metrics (success rate, mean reward, latency), and decides whether the
# candidate regressed against configurable thresholds.
#
# Regression checks are development signals. They do not prove physical robot
# safety or real-world task success.

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import CoreAIPolicyError


@dataclass
class SimRegressionConfig:
    """Thresholds for sim regression evaluation.

    A regression is detected when the candidate degrades beyond the allowed
    drop/tolerance. None means the check is disabled.
    """

    max_success_drop: float | None = None
    max_reward_drop: float | None = None
    max_runner_p95_increase_ms: float | None = None


@dataclass
class SimRegressionResult:
    """Result of a sim regression comparison."""

    passed: bool
    deltas: dict[str, Any] = field(default_factory=dict)
    checks: list[dict[str, Any]] = field(default_factory=list)


def _load_report(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise CoreAIPolicyError(f"sim_report.json not found: {path}")
    try:
        report = json.loads(path.read_text())
    except Exception as e:
        raise CoreAIPolicyError(f"Failed to read report {path}: {e}") from e
    if report.get("mode") != "sim":
        raise CoreAIPolicyError(
            f"Report {path} is not a sim report (mode != 'sim')."
        )
    return report


def _delta(candidate: Any, baseline: Any) -> Any:
    """Compute candidate - baseline for numeric values; None if either is None."""
    if candidate is None or baseline is None:
        return None
    if isinstance(candidate, (int, float)) and isinstance(baseline, (int, float)):
        return candidate - baseline
    return None


def run_sim_regression(
    baseline_path: Path,
    candidate_path: Path,
    config: SimRegressionConfig,
) -> SimRegressionResult:
    """Compare a candidate sim report against a baseline.

    Args:
        baseline_path: Path to the baseline sim_report.json.
        candidate_path: Path to the candidate sim_report.json.
        config: Regression thresholds.

    Returns:
        SimRegressionResult with deltas and per-check pass/fail.
    """
    baseline = _load_report(baseline_path)
    candidate = _load_report(candidate_path)

    b_episode = baseline.get("episode_metrics", {}) or {}
    c_episode = candidate.get("episode_metrics", {}) or {}
    b_latency = baseline.get("latency_metrics", {}) or {}
    c_latency = candidate.get("latency_metrics", {}) or {}

    success_delta = _delta(c_episode.get("success_rate"), b_episode.get("success_rate"))
    reward_delta = _delta(c_episode.get("mean_reward"), b_episode.get("mean_reward"))
    runner_p95_delta = _delta(
        c_latency.get("runner_p95_ms"), b_latency.get("runner_p95_ms")
    )

    deltas = {
        "success_rate_delta": success_delta,
        "mean_reward_delta": reward_delta,
        "runner_p95_delta_ms": runner_p95_delta,
    }

    checks: list[dict[str, Any]] = []

    # Success rate regression (a drop larger than max_success_drop fails).
    if config.max_success_drop is not None:
        drop = (-success_delta) if success_delta is not None else None
        passed = drop is not None and drop <= config.max_success_drop
        checks.append({
            "name": "max_success_drop", "passed": passed,
            "value": drop, "threshold": config.max_success_drop,
        })

    # Mean reward regression.
    if config.max_reward_drop is not None:
        drop = (-reward_delta) if reward_delta is not None else None
        passed = drop is not None and drop <= config.max_reward_drop
        checks.append({
            "name": "max_reward_drop", "passed": passed,
            "value": drop, "threshold": config.max_reward_drop,
        })

    # Runner p95 latency regression (an increase beyond tolerance fails).
    if config.max_runner_p95_increase_ms is not None:
        increase = runner_p95_delta
        passed = increase is not None and increase <= config.max_runner_p95_increase_ms
        checks.append({
            "name": "max_runner_p95_increase_ms", "passed": passed,
            "value": increase, "threshold": config.max_runner_p95_increase_ms,
        })

    overall_passed = all(c["passed"] for c in checks) if checks else True
    return SimRegressionResult(passed=overall_passed, deltas=deltas, checks=checks)
