# sim.py — simulator-only sim mode runtime (v0.8).
#
# Sim mode runs a CoreAI-backed LeRobot policy against a SimEnvironment, generates
# actions, and egresses them to the simulator. No robot connection. No motor
# commands. The only egress destination is the simulator's step().
#
# Pipeline:
#   env.reset() → make_json_safe → CoreAIPolicy.predict_action → SimEgress → env.step()
#
# Safety invariants (schema-enforced):
#   - simulator_egress_enabled = true
#   - robot_egress_enabled     = false
#   - actions_sent_to_robot    = 0  (always)
# SimEgress.send_to_robot() always raises.

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import __version__
from .errors import CoreAIPolicyError
from .failure_taxonomy import build_failure_taxonomy
from .live_metrics import LiveMetricSample, LiveMetricsCollector, summarize_action
from .observation_adapters import ObservationAdapterConfig, adapt_observation
from .policy import CoreAIPolicy
from .reports import infer_shape, now_iso, save_json
from .serialization import make_json_safe_observation
from .sim_analytics import build_sim_analytics, write_episode_metrics_csv, write_step_metrics_csv
from .sim_egress import SimEgress
from .sim_envs import SimEnvConfig, SimEnvironment, build_sim_environment
from .sim_summary import build_sim_summary_markdown
from .trace import TraceWriter
from .validation import validate_robot_type


@dataclass
class SimConfig:
    """Configuration for a sim mode run."""

    policy_path: str
    output_dir: Path
    env_type: str
    runner_url: str = "unix:///tmp/coreai-runner.sock"
    episodes: int = 1
    max_steps_per_episode: int = 300
    seed: int | None = None
    fps: float = 0.0
    robot_type: str | None = None
    task: str | None = None
    state_vector: list[float] | None = None
    env_config: Path | None = None
    env_render: bool = False
    env_record_video: bool = False
    env_video_dir: Path | None = None
    # Gym/gymnasium adapter options (v0.8.1).
    env_id: str | None = None
    env_kwargs: dict[str, Any] | None = None
    image_key: str = "observation.images.wrist"
    strict_observation_keys: bool = False
    fail_fast: bool = False
    overwrite: bool = False
    live: bool = False
    live_every: int = 1
    confirm_sim_egress: bool = False
    # Analytics artifacts (v0.8.2).
    export_csv: bool = False
    summary_md: bool = True
    failure_taxonomy: bool = True


@dataclass
class SimResult:
    """Result of a sim mode run."""

    ok: bool
    output_dir: Path
    report_path: Path
    trace_path: Path
    actions_path: Path
    observations_path: Path
    episodes_path: Path
    report: dict[str, Any] = field(default_factory=dict)


# MARK: - FPS helper

def sleep_to_maintain_fps(step_started_at: float, fps: float) -> None:
    """Sleep the remainder of the step interval to maintain target fps.

    fps <= 0 → no sleep (run as fast as possible).
    """
    if fps <= 0:
        return
    elapsed = time.monotonic() - step_started_at
    target = 1.0 / fps
    remaining = target - elapsed
    if remaining > 0:
        time.sleep(remaining)


# MARK: - Sim report builder

def build_sim_report(
    *,
    ok: bool,
    policy: CoreAIPolicy,
    runner_url: str,
    env_meta: dict[str, Any],
    loop: dict[str, Any],
    metrics: dict[str, Any],
    claims: dict[str, Any],
    files: dict[str, str],
    errors: list[dict[str, Any]],
    live_metrics: dict[str, Any] | None = None,
    episode_metrics: dict[str, Any] | None = None,
    latency_metrics: dict[str, Any] | None = None,
    action_metrics: dict[str, Any] | None = None,
    failure_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the sim_report.json dict.

    Safety invariants are hardcoded — this function is never a place where a
    robot egress could leak. The only egress is the simulator.
    """
    report = {
        "schema_version": "lerobot-coreai.sim_report.v0",
        "lerobot_coreai_version": __version__,
        "ok": ok,
        "mode": "sim",
        "policy": {
            "path": policy.policy_repo_id,
            "repo_id": policy.policy_repo_id,
            "source_repo_id": policy.manifest.policy_source_repo_id,
            "type": policy.policy_type,
            "runtime": "coreai",
            "model_id": policy.manifest.model_id,
        },
        "runner": {
            "url": runner_url,
            "reachable": True,
            "supports_action": True,
        },
        "environment": {
            **env_meta,
            "simulator_egress_enabled": True,
        },
        "loop": loop,
        "metrics": metrics,
        "claims": claims,
        "safety": {
            "simulator_egress_enabled": True,
            "robot_egress_enabled": False,
            "physical_actuation_possible": False,
            "motor_commands_available": False,
            "robot_connected": False,
            "actions_sent_to_robot": 0,
            "action_egress": "simulator_only",
        },
        "files": files,
        "errors": errors,
    }
    if live_metrics is not None:
        report["live_metrics"] = live_metrics
    # v0.8.2 analytics sections (only included when present).
    if episode_metrics is not None:
        report["episode_metrics"] = episode_metrics
    if latency_metrics is not None:
        report["latency_metrics"] = latency_metrics
    if action_metrics is not None:
        report["action_metrics"] = action_metrics
    if failure_metrics is not None:
        report["failure_metrics"] = failure_metrics
    return report


def _build_claims(
    *,
    episodes_completed: int,
    success_metric_available: bool,
    success_rate: float,
) -> dict[str, Any]:
    """Compute the claims block.

    proves_sim_task_success is conditional on a success metric existing and a
    non-zero success rate. The three real-world claims are always False.
    """
    return {
        "proves_sim_task_success": (
            episodes_completed > 0 and success_metric_available and success_rate > 0
        ),
        "sim_success_metric_available": success_metric_available,
        "proves_real_task_success": False,
        "proves_robot_safety": False,
        "proves_real_world_safety": False,
    }


# MARK: - Main entry point

def run_sim_mode(config: SimConfig) -> SimResult:
    """Execute a simulator-only sim mode run.

    Flow:
    1. Require --confirm-sim-egress (before any policy load)
    2. Prepare output_dir
    3. Load CoreAIPolicy (with runner validation)
    4. Validate robot type
    5. Build SimEnvironment + SimEgress
    6. Loop episodes × steps: reset → predict → egress to simulator
    7. Close environment (always)
    8. Write sim_report.json

    Never sends robot commands. SimEgress.send_to_robot() always raises.
    """
    # Confirm gate — enforce before any work.
    if not config.confirm_sim_egress:
        raise CoreAIPolicyError(
            "Sim mode sends actions to a simulator. "
            "Re-run with --confirm-sim-egress.\n"
            "No robot commands were sent."
        )

    output_dir = Path(config.output_dir)
    report_path = output_dir / "sim_report.json"
    trace_path = output_dir / "sim_trace.jsonl"
    actions_path = output_dir / "actions.jsonl"
    observations_path = output_dir / "observations.jsonl"
    episodes_path = output_dir / "episodes.jsonl"
    obs_dir = output_dir / "observations"

    # Prepare output dir (same overwrite semantics as shadow.py / rollout.py).
    if output_dir.exists() and any(output_dir.iterdir()):
        if not config.overwrite:
            raise CoreAIPolicyError(
                f"Output directory not empty: {output_dir}. Use --overwrite to replace."
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    trace = TraceWriter(trace_path)
    egress = SimEgress(destination="simulator")

    loop_times_ms: list[float] = []
    runner_times_ms: list[float] = []
    metrics = {
        "episodes_requested": config.episodes,
        "episodes_completed": 0,
        "steps_completed": 0,
        "actions_generated": 0,
        "actions_sent_to_simulator": 0,
        "actions_sent_to_robot": 0,
        "runner_errors": 0,
        "env_errors": 0,
        "validation_errors": 0,
        "loop_errors": 0,
    }
    errors: list[dict[str, Any]] = []

    live_collector = LiveMetricsCollector()
    adapter_config = ObservationAdapterConfig(
        image_key=config.image_key,
        state_vector=config.state_vector,
        task=config.task,
    )
    adapter_warnings: list[str] = []

    # Per-episode summaries.
    episode_summaries: list[dict[str, Any]] = []
    # Track whether any episode reported a success signal.
    success_metric_available = False
    successful_episodes = 0

    trace.write("sim.started", {
        "mode": "sim",
        "policy": config.policy_path,
        "env_type": config.env_type,
        "episodes": config.episodes,
        "max_steps_per_episode": config.max_steps_per_episode,
        "fps": config.fps,
    })

    stage = "init"
    policy: CoreAIPolicy | None = None
    env: SimEnvironment | None = None
    env_built = False
    env_closed = False
    fatal_error: Exception | None = None
    loop_start = time.monotonic()

    try:
        # Load policy.
        stage = "policy.load"
        trace.write("policy.loading")
        policy = CoreAIPolicy.from_pretrained(
            config.policy_path,
            runner_url=config.runner_url,
            validate_runner=True,
            return_metadata=True,
            strict_observation_keys=config.strict_observation_keys,
        )
        trace.write("policy.loaded", {
            "policy_type": policy.policy_type,
            "robot_type": policy.robot_type,
            "model_id": policy.manifest.model_id,
        })
        trace.write("runner.checked", {"url": config.runner_url})

        # Validate robot type.
        stage = "robot_type.validation"
        validate_robot_type(config.robot_type, policy.manifest)
        trace.write("robot_type.validated", {"requested": config.robot_type})

        # Build environment.
        stage = "environment.build"
        trace.write("environment.building", {"type": config.env_type})
        env = build_sim_environment(SimEnvConfig(
            env_type=config.env_type,
            env_config=config.env_config,
            seed=config.seed,
            render=config.env_render,
            record_video=config.env_record_video,
            video_dir=config.env_video_dir,
            task=config.task,
            state_vector=config.state_vector,
            max_steps=config.max_steps_per_episode,
            env_id=config.env_id,
            env_kwargs=config.env_kwargs,
        ))
        env_built = True
        trace.write("environment.built", {"type": config.env_type})

        obs_dir.mkdir(parents=True, exist_ok=True)

        # Episode loop.
        stage = "loop"
        trace.write("loop.started", {
            "episodes": config.episodes,
            "max_steps_per_episode": config.max_steps_per_episode,
        })

        for episode in range(config.episodes):
            episode_seed = (config.seed + episode) if config.seed is not None else None
            trace.write("episode.started", {"episode": episode, "seed": episode_seed})

            # Reset environment.
            try:
                obs = env.reset(seed=episode_seed)
            except Exception as e:
                metrics["env_errors"] += 1
                err = {
                    "episode": episode,
                    "type": type(e).__name__,
                    "message": str(e),
                    "stage": "environment.reset",
                }
                errors.append(err)
                trace.write("episode.failed", err)
                if config.fail_fast:
                    raise
                continue

            trace.write("environment.reset", {"episode": episode, "keys": list(obs.keys())})

            episode_reward = 0.0
            episode_steps = 0
            episode_success = False
            episode_terminated = False
            episode_truncated = False

            for step in range(config.max_steps_per_episode):
                step_started = time.monotonic()
                trace.write("step.started", {"episode": episode, "step": step})

                # Adapt observation (inject task/state, map keys).
                try:
                    adapted = adapt_observation(obs, adapter_config, manifest=policy.manifest)
                    obs = adapted.observation
                    if adapted.warnings:
                        adapter_warnings.extend(adapted.warnings)
                except Exception as e:
                    metrics["env_errors"] += 1
                    err = {
                        "episode": episode, "step": step,
                        "type": type(e).__name__, "message": str(e),
                        "stage": "observation.adapt",
                    }
                    errors.append(err)
                    trace.write("step.failed", err)
                    if config.fail_fast:
                        raise
                    break

                # Log observation index record.
                _append_jsonl(observations_path, {
                    "episode": episode,
                    "step": step,
                    "timestamp": now_iso(),
                    "keys": list(obs.keys()),
                })
                # Save full observation.
                # Use a global step index for frame_index so per-step artifacts
                # (e.g. saved image frames) don't collide across episodes.
                global_step = metrics["steps_completed"]
                obs_file = obs_dir / f"ep{episode:03d}_step{step:06d}.json"
                try:
                    safe_obs = make_json_safe_observation(
                        obs, output_dir=output_dir, frame_index=global_step
                    )
                    save_json(obs_file, safe_obs)
                except Exception as e:
                    metrics["env_errors"] += 1
                    err = {
                        "episode": episode, "step": step,
                        "type": type(e).__name__, "message": str(e),
                        "stage": "observation.serialize",
                    }
                    errors.append(err)
                    trace.write("step.failed", err)
                    if config.fail_fast:
                        raise
                    break

                # Predict action.
                runner_total_ms: float | None = None
                try:
                    result = policy.predict_action(safe_obs, return_metadata=True)
                    action = result["action"]
                    meta = result.get("metadata", {})
                    timing = meta.get("timing") or {}
                    runner_total_ms = timing.get("total_ms") if isinstance(timing, dict) else None
                    if runner_total_ms is not None:
                        runner_times_ms.append(float(runner_total_ms))
                    trace.write("action.generated", {
                        "episode": episode, "step": step, "shape": infer_shape(action),
                    })
                except Exception as e:
                    etype = type(e).__name__
                    if "Validation" in etype:
                        metrics["validation_errors"] += 1
                    else:
                        metrics["runner_errors"] += 1
                    err = {
                        "episode": episode, "step": step,
                        "type": etype, "message": str(e), "stage": "action.generate",
                    }
                    errors.append(err)
                    _append_jsonl(actions_path, {
                        "episode": episode, "step": step, "timestamp": now_iso(),
                        "ok": False, "action": None, "action_shape": None,
                        "egress": {
                            "sent_to_simulator": False,
                            "sent_to_robot": False,
                            "destination": "none",
                        },
                        "reward": None, "done": None,
                        "timing": {"runner_total_ms": runner_total_ms}, "error": str(e),
                    })
                    trace.write("step.failed", err)
                    if config.fail_fast:
                        raise
                    break

                # Egress action to the simulator (never to a robot).
                metrics["actions_generated"] += 1
                env_step_ms: float | None = None
                try:
                    env_step_started = time.monotonic()
                    egress_result, next_obs, reward, done, info = egress.send_to_simulator(env, action)
                    env_step_ms = (time.monotonic() - env_step_started) * 1000.0
                    metrics["actions_sent_to_simulator"] += 1
                except Exception as e:
                    metrics["env_errors"] += 1
                    err = {
                        "episode": episode, "step": step,
                        "type": type(e).__name__, "message": str(e),
                        "stage": "simulator.step",
                    }
                    errors.append(err)
                    trace.write("step.failed", err)
                    if config.fail_fast:
                        raise
                    break

                episode_reward += reward
                episode_steps += 1
                metrics["steps_completed"] += 1

                # Track success signal if the environment provides one.
                if isinstance(info, dict) and "success" in info:
                    success_metric_available = True
                    if info.get("success"):
                        episode_success = True

                loop_total_ms = (time.monotonic() - step_started) * 1000.0
                loop_times_ms.append(loop_total_ms)
                action_diag = summarize_action(action)

                # Write action record.
                _append_jsonl(actions_path, {
                    "episode": episode,
                    "step": step,
                    "timestamp": now_iso(),
                    "ok": True,
                    "action": action,
                    "action_shape": infer_shape(action),
                    "egress": {
                        "sent_to_simulator": True,
                        "sent_to_robot": False,
                        "destination": "simulator",
                    },
                    "reward": reward,
                    "done": done,
                    "timing": {
                        "runner_total_ms": runner_total_ms,
                        "loop_total_ms": loop_total_ms,
                        "env_step_ms": env_step_ms,
                    },
                    "diagnostics": {
                        "mean_abs": action_diag["mean_abs"],
                        "max_abs": action_diag["max_abs"],
                        "nan_count": action_diag["nan_count"],
                        "inf_count": action_diag["inf_count"],
                    },
                    "error": None,
                })

                # Live metrics sample.
                live_collector.add(LiveMetricSample(
                    step=metrics["steps_completed"] - 1,
                    ts=now_iso(),
                    loop_ms=loop_total_ms,
                    runner_ms=runner_total_ms,
                    action_shape=action_diag["shape"],
                    action_mean_abs=action_diag["mean_abs"],
                    action_max_abs=action_diag["max_abs"],
                    action_nan_count=action_diag["nan_count"],
                    action_inf_count=action_diag["inf_count"],
                    ok=True,
                ))

                if config.live and (step % config.live_every == 0):
                    proc_fps = 1000.0 / loop_total_ms if loop_total_ms > 0 else 0
                    print(
                        f"[sim] ep={episode} step={step} action=ok sent=sim "
                        f"loop={loop_total_ms:.1f}ms runner={runner_total_ms or 0:.1f}ms "
                        f"processing_fps={proc_fps:.1f} reward={reward:.2f} done={done} "
                        f"shape={action_diag['shape']}"
                    )

                trace.write("step.completed", {
                    "episode": episode, "step": step,
                    "reward": reward, "done": done, "loop_total_ms": loop_total_ms,
                })

                obs = next_obs
                if done:
                    episode_terminated = True
                    trace.write("episode.terminated", {"episode": episode, "steps": episode_steps})
                    break

                # Pace the loop — sleep per step to maintain target fps (matches shadow).
                sleep_to_maintain_fps(step_started, config.fps)

            # Truncated if we hit max steps without done.
            if not episode_terminated and episode_steps >= config.max_steps_per_episode:
                episode_truncated = True

            if episode_success:
                successful_episodes += 1
            metrics["episodes_completed"] += 1

            episode_summary = {
                "episode": episode,
                "steps": episode_steps,
                "total_reward": episode_reward,
                "success": episode_success,
                "terminated": episode_terminated,
                "truncated": episode_truncated,
                "actions_sent_to_simulator": episode_steps,
                "actions_sent_to_robot": 0,
            }
            episode_summaries.append(episode_summary)
            _append_jsonl(episodes_path, episode_summary)
            trace.write("episode.completed", {"episode": episode, "summary": episode_summary})

        trace.write("loop.completed", {
            "episodes_completed": metrics["episodes_completed"],
            "steps_completed": metrics["steps_completed"],
        })

    except Exception as e:
        fatal_error = e
        errors.append({"type": type(e).__name__, "message": str(e), "stage": stage})
        metrics["loop_errors"] += 1
        trace.write("sim.failed", {"error": type(e).__name__, "message": str(e)})

    finally:
        if env is not None:
            try:
                env.close()
                env_closed = True
                trace.write("environment.closed")
            except Exception:
                pass  # best-effort

    # Compute aggregates.
    loop_total_s = time.monotonic() - loop_start
    live_summary = live_collector.summary(wall_duration_s=loop_total_s)
    episodes_completed = metrics["episodes_completed"]
    success_rate = (
        successful_episodes / episodes_completed if episodes_completed > 0 else 0.0
    )
    mean_episode_reward = (
        sum(e["total_reward"] for e in episode_summaries) / len(episode_summaries)
        if episode_summaries else 0.0
    )
    claims = _build_claims(
        episodes_completed=episodes_completed,
        success_metric_available=success_metric_available,
        success_rate=success_rate,
    )

    final_metrics = _finalize_metrics(metrics, loop_times_ms, runner_times_ms)
    final_metrics["mean_episode_reward"] = mean_episode_reward
    final_metrics["success_rate"] = success_rate

    # v0.8.2: build analytics from the JSONL artifacts written so far, and
    # generate the audit artifacts (CSV/summary/taxonomy). Re-reading the JSONL
    # files is simpler than threading every record through memory and is fine
    # for the run sizes sim mode targets.
    analytics = build_sim_analytics(
        actions_path=actions_path,
        episodes_path=episodes_path,
        errors=errors,
    )
    episode_analytics = analytics["episode_metrics"]
    latency_analytics = analytics["latency_metrics"]
    action_analytics = analytics["action_metrics"]
    failure_analytics = analytics["failure_metrics"]

    files_map = {
        "actions": "actions.jsonl",
        "episodes": "episodes.jsonl",
        "observations": "observations.jsonl",
        "trace": "sim_trace.jsonl",
        "report": "sim_report.json",
    }

    # Generate audit artifacts.
    if config.export_csv:
        episode_csv_path = output_dir / "episode_metrics.csv"
        step_csv_path = output_dir / "step_metrics.csv"
        try:
            from .sim_analytics import load_jsonl
            write_episode_metrics_csv(episode_csv_path, episode_summaries)
            write_step_metrics_csv(step_csv_path, load_jsonl(actions_path))
            files_map["episode_metrics_csv"] = "episode_metrics.csv"
            files_map["step_metrics_csv"] = "step_metrics.csv"
        except Exception:
            pass  # best-effort; CSV is optional
    if config.failure_taxonomy:
        taxonomy_path = output_dir / "failure_taxonomy.json"
        try:
            save_json(taxonomy_path, build_failure_taxonomy(errors))
            files_map["failure_taxonomy"] = "failure_taxonomy.json"
        except Exception:
            pass  # best-effort
    summary_path: Path | None = None
    if config.summary_md:
        summary_path = output_dir / "sim_summary.md"
        files_map["summary"] = "sim_summary.md"
        # Content is written after the report is built (it reads the report dict).

    env_meta = {
        "type": config.env_type,
        "episodes": config.episodes,
        "max_steps_per_episode": config.max_steps_per_episode,
        "seed": config.seed,
        "built": env_built,
        "closed": env_closed,
    }
    loop_meta = {
        "fps_target": config.fps,
        "episodes_requested": config.episodes,
        "episodes_completed": episodes_completed,
        "steps_completed": metrics["steps_completed"],
        "duration_seconds": loop_total_s,
    }

    # Failure path — write the failure report.
    if fatal_error is not None:
        if policy is not None:
            fail_report = build_sim_report(
                ok=False,
                policy=policy,
                runner_url=config.runner_url,
                env_meta=env_meta,
                loop=loop_meta,
                metrics=final_metrics,
                claims=claims,
                files=files_map,
                errors=errors,
                live_metrics=live_summary,
                episode_metrics=episode_analytics,
                latency_metrics=latency_analytics,
                action_metrics=action_analytics,
                failure_metrics=failure_analytics,
            )
            try:
                save_json(report_path, fail_report)
                if summary_path is not None:
                    summary_path.write_text(build_sim_summary_markdown(fail_report))
            except Exception:
                pass  # best-effort
        trace.close()
        raise fatal_error

    # Success path.
    report = build_sim_report(
        ok=True,
        policy=policy,  # type: ignore[arg-type]
        runner_url=config.runner_url,
        env_meta=env_meta,
        loop=loop_meta,
        metrics=final_metrics,
        claims=claims,
        files=files_map,
        errors=errors,
        live_metrics=live_summary,
        episode_metrics=episode_analytics,
        latency_metrics=latency_analytics,
        action_metrics=action_analytics,
        failure_metrics=failure_analytics,
    )
    save_json(report_path, report)
    if summary_path is not None:
        try:
            summary_path.write_text(build_sim_summary_markdown(report))
        except Exception:
            pass  # best-effort
    trace.write("sim.completed", {
        "ok": True,
        "episodes": episodes_completed,
        "steps": metrics["steps_completed"],
    })
    trace.close()

    return SimResult(
        ok=True,
        output_dir=output_dir,
        report_path=report_path,
        trace_path=trace_path,
        actions_path=actions_path,
        observations_path=observations_path,
        episodes_path=episodes_path,
        report=report,
    )


# MARK: - Helpers

def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append a single JSON record as a line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def _percentile(values: list[float], p: float) -> float | None:
    """Simple percentile (p in 0-100). Returns None for empty input."""
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def _finalize_metrics(
    base: dict[str, Any],
    loop_times: list[float],
    runner_times: list[float],
) -> dict[str, Any]:
    """Compute timing aggregates on top of the running counters."""
    out = dict(base)
    out["mean_loop_ms"] = (sum(loop_times) / len(loop_times)) if loop_times else None
    out["p95_loop_ms"] = _percentile(loop_times, 95)
    out["mean_runner_ms"] = (sum(runner_times) / len(runner_times)) if runner_times else None
    out["p95_runner_ms"] = _percentile(runner_times, 95)
    return out
