# sim_analytics.py — aggregate sim run data into comparable metrics (v0.8.2).
#
# Reads the JSONL artifacts written by a sim run (actions.jsonl, episodes.jsonl)
# and the accumulated errors list, then produces episode/latency/action/failure
# metrics suitable for the report, CSV exports, and markdown summaries.
#
# These are simulator analytics only. They do not prove real-world task success
# or physical robot safety.

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


# MARK: - JSONL loading

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of dicts. Empty/nonexistent -> []."""
    path = Path(path)
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


# MARK: - Statistics helpers (stdlib only — no numpy dependency)

def percentile(values: list[float], p: float) -> float | None:
    """Linear-interpolation percentile (p in 0-100). None for empty input."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    k = (len(ordered) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return ordered[int(k)]
    return ordered[f] * (c - k) + ordered[c] * (k - f)


def mean(values: list[float]) -> float | None:
    """Arithmetic mean. None for empty input."""
    return (sum(values) / len(values)) if values else None


def median(values: list[float]) -> float | None:
    """Median (50th percentile). None for empty input."""
    return percentile(values, 50)


def _round_ms(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 3)


# MARK: - Episode metrics

def aggregate_episode_metrics(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-episode summaries into episode-level metrics.

    Rules:
        - episodes_completed == 0 -> success_rate/mean_reward/median_reward are None.
        - success_rate = successful_episodes / episodes_completed.
    """
    episodes_completed = len(episodes)
    rewards = [float(e.get("total_reward", 0.0)) for e in episodes]
    steps = [int(e.get("steps", 0)) for e in episodes]
    successful = sum(1 for e in episodes if e.get("success"))
    terminated = sum(1 for e in episodes if e.get("terminated"))
    truncated = sum(1 for e in episodes if e.get("truncated"))

    if episodes_completed > 0:
        success_rate = successful / episodes_completed
        mean_reward = mean(rewards)
        median_reward = median(rewards)
    else:
        success_rate = None
        mean_reward = None
        median_reward = None

    return {
        "episodes": episodes_completed,
        "episodes_completed": episodes_completed,
        "mean_reward": mean_reward,
        "median_reward": median_reward,
        "min_reward": min(rewards) if rewards else None,
        "max_reward": max(rewards) if rewards else None,
        "success_rate": success_rate,
        "mean_steps": mean([float(s) for s in steps]) if steps else None,
        "median_steps": median([float(s) for s in steps]) if steps else None,
        "min_steps": min(steps) if steps else None,
        "max_steps": max(steps) if steps else None,
        "terminated_episodes": terminated,
        "truncated_episodes": truncated,
    }


# MARK: - Step (latency + action) metrics

def aggregate_step_metrics(actions: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Aggregate per-step action records into latency and action metrics.

    Returns (latency_metrics, action_metrics).
    """
    runner_ms = [float(r["timing"]["runner_total_ms"]) for r in actions
                 if r.get("ok") and r.get("timing", {}).get("runner_total_ms") is not None]
    loop_ms = [float(r["timing"]["loop_total_ms"]) for r in actions
               if r.get("ok") and r.get("timing", {}).get("loop_total_ms") is not None]
    env_ms = [float(r["timing"]["env_step_ms"]) for r in actions
              if r.get("ok") and r.get("timing", {}).get("env_step_ms") is not None]

    latency = {
        "runner_p50_ms": _round_ms(percentile(runner_ms, 50)),
        "runner_p95_ms": _round_ms(percentile(runner_ms, 95)),
        "runner_max_ms": _round_ms(max(runner_ms) if runner_ms else None),
        "runner_mean_ms": _round_ms(mean(runner_ms)),
        "loop_p50_ms": _round_ms(percentile(loop_ms, 50)),
        "loop_p95_ms": _round_ms(percentile(loop_ms, 95)),
        "loop_max_ms": _round_ms(max(loop_ms) if loop_ms else None),
        "loop_mean_ms": _round_ms(mean(loop_ms)),
        "env_step_p50_ms": _round_ms(percentile(env_ms, 50)),
        "env_step_p95_ms": _round_ms(percentile(env_ms, 95)),
        "env_step_max_ms": _round_ms(max(env_ms) if env_ms else None),
        "env_step_mean_ms": _round_ms(mean(env_ms)),
    }

    # Action diagnostics (only from successful records).
    diag = [r.get("diagnostics", {}) for r in actions if r.get("ok")]
    mean_abs_values = [float(d["mean_abs"]) for d in diag if d.get("mean_abs") is not None]
    max_abs_values = [float(d["max_abs"]) for d in diag if d.get("max_abs") is not None]
    nan_steps = sum(1 for d in diag if int(d.get("nan_count", 0) or 0) > 0)
    inf_steps = sum(1 for d in diag if int(d.get("inf_count", 0) or 0) > 0)

    shapes = [tuple(r.get("action_shape") or []) for r in actions if r.get("ok")]
    unique_shapes = [list(s) for s in _unique_preserve_order(shapes)]
    shape_changes = max(0, len(unique_shapes) - 1) if shapes else 0

    action_metrics = {
        "mean_abs_action": mean(mean_abs_values),
        "max_abs_action": max(max_abs_values) if max_abs_values else None,
        "nan_action_steps": nan_steps,
        "inf_action_steps": inf_steps,
        "unique_action_shapes": unique_shapes,
        "shape_changes": shape_changes,
    }

    return latency, action_metrics


def _unique_preserve_order(items: list) -> list:
    """Return unique items preserving first-seen order (items must be hashable)."""
    seen: set = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


# MARK: - Failure metrics

def aggregate_failure_metrics(
    errors: list[dict[str, Any]],
    *,
    episodes_completed: int,
    total_steps: int,
) -> dict[str, Any]:
    """Aggregate errors into failure metrics."""
    total_errors = len(errors)
    runner_errors = sum(1 for e in errors if e.get("stage", "").startswith(("action.generate", "policy")))
    env_errors = sum(1 for e in errors if "env" in e.get("stage", "") or "simulator" in e.get("stage", ""))
    validation_errors = sum(1 for e in errors if "Validation" in e.get("type", ""))

    # episodes_failed: episodes that hit an env/simulator error and did not complete.
    # We approximate from env/simulator-stage errors at the episode scope.
    episodes_failed = sum(
        1 for e in errors
        if "env" in e.get("stage", "") or "simulator" in e.get("stage", "")
    )

    denominator = max(1, total_steps)
    error_rate = total_errors / denominator if total_errors else 0.0

    return {
        "total_errors": total_errors,
        "runner_errors": runner_errors,
        "env_errors": env_errors,
        "validation_errors": validation_errors,
        "episodes_failed": episodes_failed,
        "error_rate": error_rate,
    }


# MARK: - Top-level analytics builder

def build_sim_analytics(
    *,
    actions_path: Path,
    episodes_path: Path,
    errors: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the full analytics block from a sim run's JSONL artifacts.

    Reads actions.jsonl and episodes.jsonl from disk. errors is the in-memory
    error list from the run (may be None).
    """
    actions = load_jsonl(actions_path)
    episodes = load_jsonl(episodes_path)
    errors = errors or []

    episode_metrics = aggregate_episode_metrics(episodes)
    latency_metrics, action_metrics = aggregate_step_metrics(actions)
    failure_metrics = aggregate_failure_metrics(
        errors,
        episodes_completed=episode_metrics["episodes_completed"],
        total_steps=len(actions),
    )

    return {
        "episode_metrics": episode_metrics,
        "latency_metrics": latency_metrics,
        "action_metrics": action_metrics,
        "failure_metrics": failure_metrics,
    }


# MARK: - CSV exports

_EPISODE_CSV_FIELDS = [
    "episode", "steps", "total_reward", "success", "terminated", "truncated",
    "actions_sent_to_simulator", "actions_sent_to_robot",
]

_STEP_CSV_FIELDS = [
    "episode", "step", "ok", "reward", "done",
    "runner_total_ms", "env_step_ms", "loop_total_ms",
    "action_shape", "action_mean_abs", "action_max_abs",
    "action_nan_count", "action_inf_count", "error",
]


def write_episode_metrics_csv(path: Path, episodes: list[dict[str, Any]]) -> None:
    """Write per-episode metrics to a CSV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_EPISODE_CSV_FIELDS)
        writer.writeheader()
        for ep in episodes:
            row = {k: ep.get(k, "") for k in _EPISODE_CSV_FIELDS}
            writer.writerow(row)


def write_step_metrics_csv(path: Path, actions: list[dict[str, Any]]) -> None:
    """Write per-step metrics to a CSV file.

    Flattens the nested timing/diagnostics/error fields from actions.jsonl.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_STEP_CSV_FIELDS)
        writer.writeheader()
        for rec in actions:
            timing = rec.get("timing") or {}
            diag = rec.get("diagnostics") or {}
            shape = rec.get("action_shape")
            row = {
                "episode": rec.get("episode", ""),
                "step": rec.get("step", ""),
                "ok": rec.get("ok", ""),
                "reward": rec.get("reward", ""),
                "done": rec.get("done", ""),
                "runner_total_ms": timing.get("runner_total_ms", ""),
                "env_step_ms": timing.get("env_step_ms", ""),
                "loop_total_ms": timing.get("loop_total_ms", ""),
                "action_shape": json.dumps(shape) if shape is not None else "",
                "action_mean_abs": diag.get("mean_abs", ""),
                "action_max_abs": diag.get("max_abs", ""),
                "action_nan_count": diag.get("nan_count", ""),
                "action_inf_count": diag.get("inf_count", ""),
                "error": rec.get("error") or "",
            }
            writer.writerow(row)
