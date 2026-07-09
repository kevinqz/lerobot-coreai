# sim_summary.py — build a human-readable markdown summary of a sim run (v0.8.2).
#
# The summary is an audit artifact: it surfaces results, timing, action stats,
# failures, safety invariants, and claims. Null metrics render as "n/a". It
# never claims real-world success or physical robot safety.

from __future__ import annotations

from typing import Any


def _fmt(value: Any, suffix: str = "") -> str:
    """Render a metric value; None -> 'n/a'."""
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value}{suffix}"
    return f"{value}{suffix}"


def _fmt_pct(value: Any) -> str:
    """Render a 0-1 ratio as a percentage; None -> 'n/a'."""
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def build_sim_summary_markdown(report: dict[str, Any]) -> str:
    """Build the sim_summary.md content from a sim_report dict.

    The report is expected to include the v0.8.2 analytics sections
    (episode_metrics, latency_metrics, action_metrics, failure_metrics), the
    policy/environment metadata, and the safety/claims blocks.
    """
    policy = report.get("policy", {})
    env = report.get("environment", {})
    metrics = report.get("metrics", {})
    episode_m = report.get("episode_metrics", {})
    latency_m = report.get("latency_metrics", {})
    action_m = report.get("action_metrics", {})
    failure_m = report.get("failure_metrics", {})
    safety = report.get("safety", {})
    claims = report.get("claims", {})
    files = report.get("files", {})

    lines: list[str] = []
    lines.append("# Sim Summary")
    lines.append(f"Policy: {policy.get('path', 'n/a')}")
    lines.append(f"Runtime: {policy.get('runtime', 'n/a')}")
    env_id = env.get("id") or env.get("type", "n/a")
    lines.append(f"Environment: {env_id}")
    lines.append(f"Mode: {report.get('mode', 'sim')}")
    lines.append(f"Version: {report.get('lerobot_coreai_version', 'n/a')}")

    # Results
    lines.append("## Results")
    episodes_completed = metrics.get("episodes_completed", episode_m.get("episodes_completed"))
    episodes_requested = metrics.get("episodes_requested", episode_m.get("episodes"))
    lines.append(f"Episodes completed: {_fmt(episodes_completed)} / {_fmt(episodes_requested)}")
    lines.append(f"Steps completed: {_fmt(metrics.get('steps_completed'))}")
    lines.append(f"Success rate: {_fmt_pct(episode_m.get('success_rate'))}")
    lines.append(f"Mean reward: {_fmt(episode_m.get('mean_reward'))}")
    lines.append(f"Median reward: {_fmt(episode_m.get('median_reward'))}")

    # Timing
    lines.append("## Timing")
    lines.append(f"Runner p50: {_fmt(latency_m.get('runner_p50_ms'), ' ms')}")
    lines.append(f"Runner p95: {_fmt(latency_m.get('runner_p95_ms'), ' ms')}")
    lines.append(f"Env step p95: {_fmt(latency_m.get('env_step_p95_ms'), ' ms')}")
    lines.append(f"Loop p95: {_fmt(latency_m.get('loop_p95_ms'), ' ms')}")

    # Actions
    lines.append("## Actions")
    lines.append(f"Actions generated: {_fmt(metrics.get('actions_generated'))}")
    lines.append(f"Actions sent to simulator: {_fmt(metrics.get('actions_sent_to_simulator'))}")
    lines.append(f"Actions sent to robot: {_fmt(safety.get('actions_sent_to_robot'))}")
    lines.append(f"Action egress: {safety.get('action_egress', 'n/a')}")
    lines.append(f"NaN action steps: {_fmt(action_m.get('nan_action_steps'))}")
    lines.append(f"Inf action steps: {_fmt(action_m.get('inf_action_steps'))}")
    lines.append(f"Shape changes: {_fmt(action_m.get('shape_changes'))}")

    # Failures
    lines.append("## Failures")
    lines.append(f"Total errors: {_fmt(failure_m.get('total_errors'))}")
    lines.append(f"Runner errors: {_fmt(failure_m.get('runner_errors'))}")
    lines.append(f"Environment errors: {_fmt(failure_m.get('env_errors'))}")
    lines.append(f"Validation errors: {_fmt(failure_m.get('validation_errors'))}")

    # Safety
    lines.append("## Safety")
    lines.append(f"Simulator egress enabled: {safety.get('simulator_egress_enabled', 'n/a')}")
    lines.append(f"Robot egress enabled: {safety.get('robot_egress_enabled', 'n/a')}")
    lines.append(f"Physical actuation possible: {safety.get('physical_actuation_possible', 'n/a')}")
    lines.append(f"Motor commands available: {safety.get('motor_commands_available', 'n/a')}")
    lines.append(f"Robot connected: {safety.get('robot_connected', 'n/a')}")

    # Claims
    lines.append("## Claims")
    lines.append(f"Proves sim task success: {claims.get('proves_sim_task_success', 'n/a')}")
    lines.append(f"Proves real task success: {claims.get('proves_real_task_success', 'n/a')}")
    lines.append(f"Proves robot safety: {claims.get('proves_robot_safety', 'n/a')}")
    lines.append(f"Proves real-world safety: {claims.get('proves_real_world_safety', 'n/a')}")

    # Files
    lines.append("## Files")
    for key in ("report", "summary", "failure_taxonomy", "trace", "actions",
                "observations", "episodes", "episode_metrics_csv", "step_metrics_csv"):
        fname = files.get(key)
        if fname:
            lines.append(f"- {fname}")

    return "\n".join(lines) + "\n"
