# shadow.py — motor-blocked shadow mode runtime (v0.7).
#
# Shadow mode runs a CoreAI-backed LeRobot policy against streamed or replayed
# observations, generates actions, validates them, logs them, and blocks all action
# egress. No robot connection. No motor commands. No simulator egress.
#
# Pipeline:
#   ObservationSource → make_json_safe → CoreAIPolicy.predict_action → ActionBlocker → logs

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import __version__
from .action_blocker import ActionBlocker
from .errors import CoreAIPolicyError
from .live_metrics import LiveMetricSample, LiveMetricsCollector, summarize_action
from .observation_adapters import ObservationAdapterConfig, adapt_observation
from .observation_sources import ObservationSource, build_observation_source
from .policy import CoreAIPolicy
from .reports import infer_shape, now_iso, save_json
from .serialization import make_json_safe_observation
from .shadow_quality import ShadowQualityConfig, ShadowQualityResult, evaluate_shadow_quality
from .safety_profiles import resolve_safety_profile
from .safety_supervisor import SafetyContext, SafetySupervisor, safe_evaluate
from .safety_reports import (
    SafetyAccumulator,
    append_safety_decision,
    build_safety_summary,
    build_safety_summary_markdown,
)
from .trace import TraceWriter
from .validation import validate_robot_type


@dataclass
class ShadowConfig:
    """Configuration for a shadow mode run."""

    policy_path: str
    runner_url: str = "unix:///tmp/coreai-runner.sock"
    output_dir: Path = Path("runs/shadow")
    observation_source: str = "folder"  # fixture | fixtures | folder | camera
    robot_type: str | None = None
    fixture: Path | None = None
    fixtures_dir: Path | None = None
    frames_dir: Path | None = None
    camera_index: int | None = None
    image_key: str = "observation.images.wrist"
    state_json: Path | None = None
    state_vector: list[float] | None = None
    task: str | None = None
    max_steps: int = 32
    duration_seconds: float | None = None
    fps: float = 10.0
    warmup_steps: int = 0
    strict_observation_keys: bool = False
    fail_fast: bool = False
    overwrite: bool = False
    include_metadata: bool = True
    # Camera source options (v0.7.1).
    camera_width: int | None = None
    camera_height: int | None = None
    camera_fps: float | None = None
    # Observation adapter options (v0.7.2).
    adapter_image_keys: dict[str, str] | None = None
    require_task: bool = False
    require_state: bool = False
    required_keys: list[str] | None = None
    drop_unknown_keys: bool = False
    # Live metrics / quality options (v0.7.2).
    live: bool = False
    live_every: int = 1
    quality_config: ShadowQualityConfig | None = None
    fail_on_quality: bool = False
    # Runtime safety supervisor (v0.9.0) — diagnostic only.
    # Shadow ALWAYS blocks every action via ActionBlocker regardless of the
    # supervisor. Enabling the supervisor only adds auditable decisions so
    # safety profiles can be calibrated with zero egress risk. Default off.
    supervisor_mode: str = "off"
    safety_profile: Path | None = None
    safety_profile_name: str | None = None
    safety_report: bool = True


@dataclass
class ShadowResult:
    """Result of a shadow mode run."""

    ok: bool
    output_dir: Path
    report_path: Path
    trace_path: Path
    actions_path: Path
    observations_path: Path
    blocked_actions_path: Path
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


# MARK: - Shadow report builder

def build_shadow_report(
    *,
    ok: bool,
    policy: CoreAIPolicy,
    runner_url: str,
    source_type: str,
    source_meta: dict[str, Any],
    loop: dict[str, Any],
    metrics: dict[str, Any],
    files: dict[str, str],
    errors: list[dict[str, Any]],
    live_metrics: dict[str, Any] | None = None,
    quality: dict[str, Any] | None = None,
    adapter: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the shadow_report.json dict.

    Safety invariants are hardcoded — this function is never a place where an
    action could leak.
    """
    report = {
        "schema_version": "lerobot-coreai.shadow_report.v0",
        "lerobot_coreai_version": __version__,
        "ok": ok,
        "mode": "shadow",
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
        "observation_source": {
            "type": source_type,
            **source_meta,
        },
        "loop": loop,
        "metrics": metrics,
        "claims": {
            "proves_runtime_action_generation": ok and metrics.get("actions_generated", 0) > 0,
            "proves_task_success": False,
            "proves_robot_safety": False,
            "proves_real_world_safety": False,
        },
        "safety": {
            "physical_actuation_possible": False,
            "motor_commands_available": False,
            "actuation_device_connected": False,
            "robot_connected": False,
            "actions_sent": 0,
            "action_egress": "blocked",
            "blocker": "ActionBlocker",
            "block_reason": "shadow_mode_no_actuation",
        },
        "files": files,
        "errors": errors,
    }
    # Optional v0.7.2 sections (only included when present, for backward compat).
    if live_metrics is not None:
        report["live_metrics"] = live_metrics
    if quality is not None:
        report["quality"] = quality
    if adapter is not None:
        report["adapter"] = adapter
    return report


# MARK: - Main entry point

def run_shadow_mode(config: ShadowConfig) -> ShadowResult:
    """Execute a motor-blocked shadow mode run.

    Flow (spec §9):
    1. Prepare output_dir
    2. Create ActionBlocker + TraceWriter
    3. Load CoreAIPolicy (with runner validation)
    4. Validate robot type
    5. Open ObservationSource
    6. Loop: read obs → predict action → block → log
    7. Close source (always)
    8. Write shadow_report.json

    Never sends robot commands. ActionBlocker.send() always raises.
    """
    output_dir = Path(config.output_dir)
    report_path = output_dir / "shadow_report.json"
    trace_path = output_dir / "shadow_trace.jsonl"
    actions_path = output_dir / "actions.jsonl"
    observations_path = output_dir / "observations.jsonl"
    blocked_actions_path = output_dir / "blocked_actions.jsonl"
    obs_dir = output_dir / "observations"

    # Prepare output dir (same overwrite semantics as rollout.py).
    if output_dir.exists() and any(output_dir.iterdir()):
        if not config.overwrite:
            raise CoreAIPolicyError(
                f"Output directory not empty: {output_dir}. Use --overwrite to replace."
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    trace = TraceWriter(trace_path)
    blocker = ActionBlocker(mode="shadow")

    loop_times_ms: list[float] = []
    runner_times_ms: list[float] = []
    metrics = {
        "observations_read": 0,
        "actions_generated": 0,
        "actions_blocked": 0,
        "actions_sent": 0,
        "observation_errors": 0,
        "runner_errors": 0,
        "validation_errors": 0,
        "loop_errors": 0,
    }
    errors: list[dict[str, Any]] = []
    steps_completed = 0

    # v0.9.0: optional diagnostic safety supervisor (shadow still blocks all).
    supervisor_mode = config.supervisor_mode
    supervisor: SafetySupervisor | None = None
    safety_profile_obj = None
    safety_acc: SafetyAccumulator | None = None
    safety_report_path: Path | None = None
    if supervisor_mode != "off":
        safety_profile_obj = resolve_safety_profile(
            path=config.safety_profile, name=config.safety_profile_name,
        )
        supervisor = SafetySupervisor(safety_profile_obj, mode=supervisor_mode)
        safety_acc = SafetyAccumulator(profile=safety_profile_obj.name, mode=supervisor_mode)
        if config.safety_report:
            safety_report_path = output_dir / "safety_report.jsonl"

    # v0.7.2: live metrics collector + adapter config.
    live_collector = LiveMetricsCollector()
    adapter_config = ObservationAdapterConfig(
        image_key=config.image_key,
        image_keys=config.adapter_image_keys,
        state_vector=config.state_vector,
        state_json=config.state_json,
        task=config.task,
        require_task=config.require_task,
        require_state=config.require_state,
        required_keys=config.required_keys or [],
        drop_unknown_keys=config.drop_unknown_keys,
    )
    adapter_warnings: list[str] = []

    trace.write("shadow.started", {
        "mode": "shadow",
        "policy": config.policy_path,
        "observation_source": config.observation_source,
        "max_steps": config.max_steps,
        "fps": config.fps,
    })

    stage = "init"
    policy: CoreAIPolicy | None = None
    source: ObservationSource | None = None
    source_opened = False
    source_closed = False
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

        # Build + open observation source.
        stage = "observation_source.open"
        trace.write("observation_source.opening", {"type": config.observation_source})
        source = build_observation_source(
            config.observation_source,
            fixture=config.fixture,
            fixtures_dir=config.fixtures_dir,
            frames_dir=config.frames_dir,
            image_key=config.image_key,
            state_json=config.state_json,
            state_vector=config.state_vector,
            task=config.task,
            camera_index=config.camera_index,
            camera_width=config.camera_width,
            camera_height=config.camera_height,
            camera_fps=config.camera_fps,
            output_dir=output_dir,
        )
        source.open()
        source_opened = True
        trace.write("observation_source.opened", {"type": config.observation_source})

        # Create observations/ subdir for per-step observation files.
        obs_dir.mkdir(parents=True, exist_ok=True)

        # Warmup steps: read and discard.
        if config.warmup_steps > 0:
            for w in range(config.warmup_steps):
                warmup_obs = source.read()
                if warmup_obs is None:
                    break
                trace.write("warmup.step", {"index": w})

        # Main loop.
        stage = "loop"
        trace.write("loop.started", {
            "max_steps": config.max_steps,
            "fps": config.fps,
            "duration_seconds": config.duration_seconds,
        })

        for step in range(config.max_steps):
            # Duration cap.
            if config.duration_seconds is not None:
                elapsed = time.monotonic() - loop_start
                if elapsed >= config.duration_seconds:
                    trace.write("loop.duration_reached", {"elapsed": elapsed})
                    break

            step_started = time.monotonic()
            trace.write("step.started", {"step": step})

            # Read observation.
            try:
                raw_obs = source.read()
            except Exception as e:
                metrics["observation_errors"] += 1
                err = {"step": step, "type": type(e).__name__, "message": str(e), "stage": "observation.read"}
                errors.append(err)
                trace.write("step.failed", err)
                if config.fail_fast:
                    raise
                continue

            if raw_obs is None:
                trace.write("loop.source_exhausted", {"step": step})
                break

            metrics["observations_read"] += 1
            trace.write("observation.read", {"step": step, "keys": list(raw_obs.keys())})

            # v0.7.2: adapt observation (inject task/state, map keys, check required).
            try:
                adapted = adapt_observation(raw_obs, adapter_config, manifest=policy.manifest)
                raw_obs = adapted.observation
                if adapted.warnings:
                    adapter_warnings.extend(adapted.warnings)
            except Exception as e:
                metrics["observation_errors"] += 1
                err = {"step": step, "type": type(e).__name__, "message": str(e), "stage": "observation.adapt"}
                errors.append(err)
                trace.write("step.failed", err)
                if config.fail_fast:
                    raise
                continue

            # Save observation record.
            _append_jsonl(observations_path, {
                "step": step,
                "timestamp": now_iso(),
                "source": config.observation_source,
                "keys": list(raw_obs.keys()),
            })
            # Save full observation to observations/ dir.
            obs_file = obs_dir / f"step_{step:06d}.json"
            try:
                safe_obs = make_json_safe_observation(raw_obs, output_dir=output_dir, frame_index=step)
                save_json(obs_file, safe_obs)
                trace.write("observation.serialized", {"step": step, "path": str(obs_file)})
            except Exception as e:
                metrics["observation_errors"] += 1
                err = {"step": step, "type": type(e).__name__, "message": str(e), "stage": "observation.serialize"}
                errors.append(err)
                trace.write("step.failed", err)
                if config.fail_fast:
                    raise
                continue

            # Predict action using the JSON-safe observation (not raw_obs).
            # The runner receives JSON over HTTP; raw objects (tensors/PIL/arrays)
            # from real sources would fail to serialize. This mirrors the v0.4
            # eval hardening fix.
            runner_total_ms: float | None = None
            try:
                result = policy.predict_action(safe_obs, return_metadata=True)
                action = result["action"]
                meta = result.get("metadata", {})
                timing = meta.get("timing") or {}
                runner_total_ms = timing.get("total_ms") if isinstance(timing, dict) else None
                if runner_total_ms is not None:
                    runner_times_ms.append(float(runner_total_ms))
                trace.write("action.generated", {"step": step, "shape": infer_shape(action)})
            except Exception as e:
                # Distinguish validation vs runner errors loosely by type name.
                etype = type(e).__name__
                if "Validation" in etype:
                    metrics["validation_errors"] += 1
                else:
                    metrics["runner_errors"] += 1
                err = {"step": step, "type": etype, "message": str(e), "stage": "action.generate"}
                errors.append(err)
                _append_jsonl(actions_path, {
                    "step": step, "observation_index": step, "timestamp": now_iso(),
                    "ok": False, "action": None, "action_shape": None,
                    "blocked": False, "sent": False, "block_reason": None,
                    "timing": {"runner_total_ms": runner_total_ms}, "error": str(e),
                })
                trace.write("step.failed", err)
                if config.fail_fast:
                    raise
                continue

            # v0.9.0: optional diagnostic supervision (never affects egress —
            # ActionBlocker below is the final, unconditional block).
            if supervisor is not None:
                ctx = SafetyContext(
                    mode="shadow", step=step,
                    robot_type=policy.robot_type, policy_type=policy.policy_type,
                )
                supervised = safe_evaluate(supervisor, action, context=ctx)
                if safety_report_path is not None:
                    append_safety_decision(safety_report_path, supervised.decision, context=ctx)
                if safety_acc is not None:
                    safety_acc.add(supervised.decision)
                trace.write("safety.decision", {
                    "step": step, "allowed": supervised.decision.allowed,
                    "reasons": supervised.decision.reasons,
                })

            # Block the action (never send).
            blocked = blocker.block(action)
            metrics["actions_generated"] += 1
            metrics["actions_blocked"] += 1
            trace.write("action.blocked", {"step": step, "reason": blocked.reason})

            # Write action + blocked records.
            loop_total_ms = (time.monotonic() - step_started) * 1000.0
            loop_times_ms.append(loop_total_ms)
            # v0.7.2: action diagnostics.
            action_diag = summarize_action(action)
            action_record = {
                "step": step,
                "observation_index": step,
                "timestamp": now_iso(),
                "ok": True,
                "action": action,
                "action_shape": infer_shape(action),
                "blocked": True,
                "sent": False,
                "block_reason": blocked.reason,
                "timing": {
                    "runner_total_ms": runner_total_ms,
                    "loop_total_ms": loop_total_ms,
                },
                "diagnostics": {
                    "mean_abs": action_diag["mean_abs"],
                    "max_abs": action_diag["max_abs"],
                    "nan_count": action_diag["nan_count"],
                    "inf_count": action_diag["inf_count"],
                },
                "error": None,
            }
            _append_jsonl(actions_path, action_record)
            _append_jsonl(blocked_actions_path, {
                "step": step,
                "timestamp": now_iso(),
                "reason": blocked.reason,
                "sent": False,
                "destination": "none",
            })

            # v0.7.2: collect live metric sample.
            live_collector.add(LiveMetricSample(
                step=step,
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

            # v0.7.2: optional live console output.
            if config.live and (step % config.live_every == 0):
                proc_fps = 1000.0 / loop_total_ms if loop_total_ms > 0 else 0
                print(
                    f"[shadow] step={step} obs=ok action=ok blocked=yes "
                    f"loop={loop_total_ms:.1f}ms runner={runner_total_ms or 0:.1f}ms "
                    f"processing_fps={proc_fps:.1f} shape={action_diag['shape']}"
                )

            steps_completed += 1
            trace.write("step.completed", {"step": step, "loop_total_ms": loop_total_ms})

            # Pace the loop.
            sleep_to_maintain_fps(step_started, config.fps)

        trace.write("loop.completed", {"steps_completed": steps_completed})

    except Exception as e:
        # Capture the error; the failure report is written after the finally
        # block so that source_closed reflects the actual close status.
        fatal_error = e
        errors.append({"type": type(e).__name__, "message": str(e), "stage": stage})
        metrics["loop_errors"] += 1
        trace.write("shadow.failed", {"error": type(e).__name__, "message": str(e)})

    finally:
        if source is not None:
            try:
                source.close()
                source_closed = True
                trace.write("observation_source.closed")
            except Exception:
                pass  # best-effort

    # v0.7.2: compute live metrics summary + quality evaluation for the report.
    # Wall-clock duration includes setup, loop, and pacing sleeps — used for
    # effective_fps (real paced rate). processing_fps comes from loop_ms sum.
    loop_total_s = time.monotonic() - loop_start
    live_summary = live_collector.summary(wall_duration_s=loop_total_s)
    total_steps_attempted = steps_completed + metrics["observation_errors"] + metrics["runner_errors"] + metrics["validation_errors"]
    error_rate = 0.0
    if total_steps_attempted > 0:
        error_rate = (metrics["observation_errors"] + metrics["runner_errors"] + metrics["validation_errors"]) / total_steps_attempted
    quality_result: ShadowQualityResult | None = None
    if config.quality_config is not None:
        quality_result = evaluate_shadow_quality(live_summary, config.quality_config, error_rate=error_rate)

    adapter_section = {
        "image_key": config.image_key,
        "required_keys": config.required_keys or [],
        "warnings": adapter_warnings,
    }
    quality_section = None
    if quality_result is not None:
        quality_section = {
            "passed": quality_result.passed,
            "checks": quality_result.checks,
        }

    # Failure path — write the failure report (after finally, so source_closed is correct).
    if fatal_error is not None:
        if policy is not None:
            fail_report = build_shadow_report(
                ok=False,
                policy=policy,
                runner_url=config.runner_url,
                source_type=config.observation_source,
                source_meta={"opened": source_opened, "closed": source_closed},
                loop={
                    "fps_target": config.fps,
                    "steps_requested": config.max_steps,
                    "steps_completed": steps_completed,
                    "duration_seconds": loop_total_s,
                    "warmup_steps": config.warmup_steps,
                },
                metrics=_finalize_metrics(metrics, loop_times_ms, runner_times_ms),
                files={
                    "actions": "actions.jsonl",
                    "blocked_actions": "blocked_actions.jsonl",
                    "observations": "observations.jsonl",
                    "trace": "shadow_trace.jsonl",
                    "report": "shadow_report.json",
                },
                errors=errors,
                live_metrics=live_summary,
                quality=quality_section,
                adapter=adapter_section,
            )
            try:
                save_json(report_path, fail_report)
            except Exception:
                pass  # best-effort
        trace.close()
        raise fatal_error

    # v0.7.2: if fail_on_quality and quality failed, mark ok=False.
    quality_failed = quality_result is not None and not quality_result.passed and config.fail_on_quality

    # Success path — build the report.
    final_metrics = _finalize_metrics(metrics, loop_times_ms, runner_times_ms)
    files_map = {
        "actions": "actions.jsonl",
        "blocked_actions": "blocked_actions.jsonl",
        "observations": "observations.jsonl",
        "trace": "shadow_trace.jsonl",
        "report": "shadow_report.json",
    }
    report = build_shadow_report(
        ok=not quality_failed,
        policy=policy,  # type: ignore[arg-type]
        runner_url=config.runner_url,
        source_type=config.observation_source,
        source_meta={"opened": source_opened, "closed": source_closed},
        loop={
            "fps_target": config.fps,
            "steps_requested": config.max_steps,
            "steps_completed": steps_completed,
            "duration_seconds": loop_total_s,
            "warmup_steps": config.warmup_steps,
        },
        metrics=final_metrics,
        files=files_map,
        errors=errors,
        live_metrics=live_summary,
        quality=quality_section,
        adapter=adapter_section,
    )
    # v0.9.0: attach diagnostic safety supervisor section (shadow never egresses).
    if supervisor is not None and safety_acc is not None:
        safety_summary = build_safety_summary(safety_acc)
        if config.safety_report:
            try:
                save_json(output_dir / "safety_summary.json", safety_summary)
                (output_dir / "safety_summary.md").write_text(
                    build_safety_summary_markdown(safety_summary))
            except Exception:
                pass  # best-effort
        report["safety_supervisor"] = {
            "enabled": True,
            "mode": supervisor_mode,
            "profile": safety_acc.profile,
            "actions_supervised": safety_acc.actions_supervised,
            "actions_allowed": safety_acc.actions_allowed,
            "actions_blocked": safety_acc.actions_blocked,
            "actions_modified": safety_acc.actions_modified,
            "critical_failures": safety_acc.critical_failures,
            "top_reasons": safety_acc.top_reasons(),
            "note": "Shadow mode blocks all actions via ActionBlocker; supervisor is diagnostic only.",
        }
    save_json(report_path, report)
    trace.write("shadow.completed", {"ok": not quality_failed, "steps": steps_completed})
    trace.close()

    return ShadowResult(
        ok=not quality_failed,
        output_dir=output_dir,
        report_path=report_path,
        trace_path=trace_path,
        actions_path=actions_path,
        observations_path=observations_path,
        blocked_actions_path=blocked_actions_path,
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
