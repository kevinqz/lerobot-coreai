# real_mode.py — guarded real mode orchestrator (v1.0.0).
#
# The first path where actions_sent_to_robot may exceed zero — and only inside
# `real --mode guarded`, only after every gate passes, only through the
# RealEgressGuard. preflight sends nothing. A blocked action never reaches the
# adapter. Any exception stops and disconnects the adapter.

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import signal

from . import __version__
from .errors import CoreAIPolicyError
from .observation_adapters import ObservationAdapterConfig, adapt_observation
from .policy import CoreAIPolicy
from .reports import now_iso, save_json
from .real_arming import build_arming_manifest, build_arming_markdown
from .real_egress import DeadmanSwitch, RateLimiter, RealEgressGuard
from .real_metrics import (
    RealMetricsCollector, build_real_metrics_markdown, build_real_metrics_report,
    write_real_metrics_csv,
)
from .real_preflight import RealPreflightConfig, evaluate_real_preflight
from .real_reports import (
    build_real_report, build_real_report_markdown, build_real_session,
)
from .robot_adapters import build_robot_adapter
from .safety_profiles import resolve_safety_profile
from .safety_reports import SafetyAccumulator, append_safety_decision, build_safety_summary
from .safety_supervisor import SafetyContext, SafetySupervisor
from .trace import TraceWriter


@dataclass
class RealModeConfig:
    mode: str
    policy_path: str
    runner_url: str
    robot_adapter: str
    robot_type: str
    safety_profile: Path
    readiness_report: Path
    approval: Path
    bundle_dir: Path
    output_dir: Path
    robot_config: Path | None = None
    robot_endpoint: str | None = None
    robot_token: str | None = None
    robot_auth_token_env: str | None = None
    operator: str | None = None
    max_steps: int | None = None
    duration_seconds: float | None = None
    fps: float = 2.0
    deadman_timeout_s: float = 1.0
    deadman_disable_for_mock_only: bool = False
    attest_real_hardware: bool = False
    attest_physical_estop: bool = False
    attest_workspace_clear: bool = False
    # Observation config (v1.0.4).
    observation_config: Path | None = None
    obs_image_key: str | None = None
    obs_state_key: str | None = None
    obs_task: str | None = None
    obs_require_state: bool = False
    obs_require_task: bool = False
    obs_required_keys: list[str] | None = None
    obs_drop_unknown_keys: bool = False
    # Redaction (v1.0.5).
    redact_paths: bool = False
    redact_runner_url: bool = False
    redact_robot_endpoint: bool = False
    redact_operator: bool = False
    # Abort controls (v1.0.6).
    abort_file: Path | None = None

    def has_observation_config(self) -> bool:
        return bool(
            self.observation_config or self.obs_image_key or self.obs_state_key
            or self.obs_task or self.obs_require_state or self.obs_require_task
            or self.obs_required_keys or self.obs_drop_unknown_keys)


@dataclass
class RealModeResult:
    ok: bool
    output_dir: Path
    report: dict[str, Any]
    actions_sent_to_robot: int = 0
    actions_blocked_by_supervisor: int = 0
    stopped_reason: str | None = None


def _preflight_config(config: RealModeConfig) -> RealPreflightConfig:
    return RealPreflightConfig(
        mode=config.mode, policy_path=config.policy_path, runner_url=config.runner_url,
        robot_adapter=config.robot_adapter, robot_type=config.robot_type,
        safety_profile=config.safety_profile, readiness_report=config.readiness_report,
        approval=config.approval, bundle_dir=config.bundle_dir,
        robot_config=config.robot_config, robot_endpoint=config.robot_endpoint,
        robot_token=config.robot_token,
        robot_auth_token_env=config.robot_auth_token_env,
        operator=config.operator, max_steps=config.max_steps,
        duration_seconds=config.duration_seconds, fps=config.fps,
        attest_real_hardware=config.attest_real_hardware,
        attest_physical_estop=config.attest_physical_estop,
        attest_workspace_clear=config.attest_workspace_clear,
        has_observation_config=config.has_observation_config(),
    )


def _redact(obj: dict[str, Any], config: RealModeConfig) -> dict[str, Any]:
    """Return a redacted copy of a session/report dict per the config flags.

    Endpoints are never persisted, so --redact-robot-endpoint is a documented
    no-op on the current artifacts (nothing to leak).
    """
    if not (config.redact_paths or config.redact_runner_url
            or config.redact_operator or config.redact_robot_endpoint):
        return obj
    import copy
    d = copy.deepcopy(obj)

    def _basename(v):
        return Path(v).name if isinstance(v, str) and v else v

    if config.redact_operator:
        if d.get("operator"):
            d["operator"] = "<redacted>"
    if config.redact_runner_url:
        if d.get("runner_url"):
            d["runner_url"] = "<redacted>"
    if config.redact_paths:
        for k in ("safety_profile", "readiness_report", "approval"):
            if d.get(k):
                d[k] = _basename(d[k])
        # Report-shaped nested paths.
        if isinstance(d.get("readiness"), dict) and d["readiness"].get("report"):
            d["readiness"]["report"] = _basename(d["readiness"]["report"])
        if isinstance(d.get("approval"), dict) and d["approval"].get("path"):
            d["approval"]["path"] = _basename(d["approval"]["path"])
        # Arming-manifest bindings paths (hashes are kept — they leak nothing).
        if isinstance(d.get("bindings"), dict):
            for k in ("safety_profile", "readiness_report", "approval"):
                if d["bindings"].get(k):
                    d["bindings"][k] = _basename(d["bindings"][k])
        if isinstance(d.get("abort_controls"), dict) and d["abort_controls"].get("abort_file"):
            d["abort_controls"]["abort_file"] = _basename(d["abort_controls"]["abort_file"])
    return d


def _build_obs_config(config: RealModeConfig) -> ObservationAdapterConfig:
    data: dict[str, Any] = {}
    if config.observation_config:
        data = json.loads(Path(config.observation_config).read_text())
    kwargs: dict[str, Any] = {}
    for k in ("image_key", "state_key", "task", "require_task", "require_state",
              "required_keys", "drop_unknown_keys"):
        if k in data:
            kwargs[k] = data[k]
    # Individual flags override the config file.
    if config.obs_image_key is not None:
        kwargs["image_key"] = config.obs_image_key
    if config.obs_state_key is not None:
        kwargs["state_key"] = config.obs_state_key
    if config.obs_task is not None:
        kwargs["task"] = config.obs_task
    if config.obs_require_state:
        kwargs["require_state"] = True
    if config.obs_require_task:
        kwargs["require_task"] = True
    if config.obs_required_keys is not None:
        kwargs["required_keys"] = config.obs_required_keys
    if config.obs_drop_unknown_keys:
        kwargs["drop_unknown_keys"] = True
    return ObservationAdapterConfig(**kwargs)


def run_real_mode(config: RealModeConfig) -> RealModeResult:
    """Run guarded real mode. Fail-closed at every gate."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Preflight — always runs, never sends an action.
    preflight = evaluate_real_preflight(_preflight_config(config))
    save_json(output_dir / "real_preflight_report.json", preflight.report)

    def _finalize(ok, stop_reason, session, sup_summary, sent, blocked,
                  estop=False, deadman_lost=False, ready=None, approval_valid=None,
                  approval_id=None, robot_ready=False):
        report = build_real_report(
            ok=ok, mode=config.mode, session=session, robot_ready=robot_ready,
            supervisor_summary=sup_summary, actions_sent_to_robot=sent,
            actions_blocked_before_robot=blocked, readiness_ready=ready,
            approval_valid=approval_valid, approval_id=approval_id,
            stop_reason=stop_reason, estop_triggered=estop, deadman_lost=deadman_lost)
        report = _redact(report, config)
        save_json(output_dir / "real_report.json", report)
        (output_dir / "real_report.md").write_text(build_real_report_markdown(report))
        return RealModeResult(ok=ok, output_dir=output_dir, report=report,
                              actions_sent_to_robot=sent,
                              actions_blocked_by_supervisor=blocked,
                              stopped_reason=stop_reason)

    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    sid_seed = f"{config.policy_path}{created}{config.robot_adapter}".encode()
    session_id = f"real_{datetime.now(timezone.utc).strftime('%Y%m%d')}_" \
                 f"{hashlib.sha256(sid_seed).hexdigest()[:8]}"
    session = build_real_session(
        session_id=session_id, created_at=created, operator=config.operator,
        mode=config.mode, robot_adapter=config.robot_adapter,
        robot_type=config.robot_type, policy_path=config.policy_path,
        runner_url=config.runner_url, safety_profile=config.safety_profile,
        readiness_report=config.readiness_report, approval=config.approval,
        max_steps=config.max_steps, fps=config.fps,
        duration_seconds=config.duration_seconds, status="preflight_passed")

    empty_sup = {"mode": "enforce", "actions_supervised": 0, "actions_allowed": 0,
                 "actions_blocked": 0}

    if not preflight.ok:
        failed = [c.name for c in preflight.checks if c.severity == "required" and not c.passed]
        session["status"] = "preflight_failed"
        save_json(output_dir / "real_session.json", session)
        return _finalize(False, "preflight_failed: " + ", ".join(failed), session,
                         empty_sup, 0, 0)

    # 2. Preflight-only mode: stop here. Zero egress.
    if config.mode != "guarded":
        save_json(output_dir / "real_session.json", session)
        return _finalize(True, "preflight_only", session, empty_sup, 0, 0)

    # 3. Guarded: deadman may be disabled ONLY for the mock adapter.
    deadman_enabled = True
    if config.deadman_disable_for_mock_only:
        if config.robot_adapter != "mock":
            save_json(output_dir / "real_session.json", session)
            return _finalize(False, "deadman_cannot_be_disabled_for_non_mock", session,
                             empty_sup, 0, 0)
        deadman_enabled = False

    # Build the guarded session components.
    profile = resolve_safety_profile(path=Path(config.safety_profile))
    supervisor = SafetySupervisor(profile, mode="enforce")
    acc = SafetyAccumulator(profile=profile.name, mode="enforce")

    # Readiness / approval facts for the report — read before building the
    # adapter so the controller can carry the approval id in its request headers.
    ready = None
    approval_valid = None
    approval_id = None
    try:
        ap = json.loads(Path(config.approval).read_text())
        approval_id = ap.get("approval_id")
        approval_valid = True  # preflight already verified it
    except Exception:
        pass

    adapter = build_robot_adapter(
        config.robot_adapter, config.robot_type,
        endpoint=config.robot_endpoint, config=config.robot_config,
        token=config.robot_token, session_id=session_id, approval_id=approval_id)
    deadman = DeadmanSwitch(timeout_s=config.deadman_timeout_s, enabled=deadman_enabled)
    rate_limiter = RateLimiter(fps=config.fps)
    trace = TraceWriter(output_dir / "real_trace.jsonl")
    safety_report_path = output_dir / "safety_report.jsonl"
    actions_path = output_dir / "real_actions.jsonl"

    guard = RealEgressGuard(supervisor, adapter, session, deadman, rate_limiter, trace,
                            allow_disabled_deadman=not deadman_enabled)

    try:
        rr = json.loads(Path(config.readiness_report).read_text())
        ready = rr.get("ready")
    except Exception:
        pass

    trace.write("real.preflight.completed", {"ok": True})
    trace.write("real.session.created", {"session_id": session_id})

    policy: CoreAIPolicy | None = None
    sent = 0
    stop_reason = None
    fatal: Exception | None = None
    adapter_connected = False
    metrics = RealMetricsCollector(fps=config.fps)
    loop_start = time.monotonic()

    # v1.0.6: operator abort controls. SIGINT (Ctrl-C) and/or the appearance of
    # an --abort-file both trigger an e-stop and stop the session cleanly.
    abort = {"triggered": False, "reason": None}

    def _on_sigint(signum, frame):  # pragma: no cover - exercised via handler ref
        abort["triggered"] = True
        abort["reason"] = "operator_abort_sigint"

    prev_sigint = None
    try:
        prev_sigint = signal.signal(signal.SIGINT, _on_sigint)
    except (ValueError, OSError):
        prev_sigint = None  # not in main thread; abort-file still works

    def _abort_reason() -> str | None:
        if abort["triggered"]:
            return abort["reason"] or "operator_abort"
        if config.abort_file and Path(config.abort_file).exists():
            return "operator_abort_file"
        return None

    try:
        policy = CoreAIPolicy.from_pretrained(
            config.policy_path, runner_url=config.runner_url,
            validate_runner=True, return_metadata=True)
        adapter_config = _build_obs_config(config)
        adapter.connect()
        adapter_connected = True
        session["status"] = "running"
        save_json(output_dir / "real_session.json", session)
        deadman.heartbeat()
        trace.write("real.session.armed", {"session_id": session_id})
        # v1.0.6: record the armed envelope before the first action.
        arming = build_arming_manifest(
            session_id=session_id, created_at=created, operator=config.operator,
            robot_adapter=config.robot_adapter, robot_type=config.robot_type,
            policy_path=config.policy_path, safety_profile=config.safety_profile,
            readiness_report=config.readiness_report, approval=config.approval,
            approval_id=approval_id, max_steps=config.max_steps,
            duration_seconds=config.duration_seconds, fps=config.fps,
            deadman_timeout_s=config.deadman_timeout_s, deadman_enabled=deadman_enabled,
            attestations={
                "real_hardware": config.attest_real_hardware,
                "physical_estop": config.attest_physical_estop,
                "workspace_clear": config.attest_workspace_clear,
            },
            abort_file=config.abort_file)
        save_json(output_dir / "real_arming_manifest.json", _redact(arming, config))
        (output_dir / "real_arming_manifest.md").write_text(build_arming_markdown(arming))
        loop_start = time.monotonic()

        for step in range(config.max_steps or 0):
            step_start = time.monotonic()
            # v1.0.6: operator abort takes precedence over everything else.
            _ar = _abort_reason()
            if _ar is not None:
                guard.trigger_estop(_ar)
                stop_reason = "operator_abort"
                break
            # Bounded by duration as well as step count.
            if config.duration_seconds is not None and \
                    (time.monotonic() - loop_start) >= config.duration_seconds:
                stop_reason = "duration_seconds_reached"
                break
            deadman.heartbeat()  # software liveness for this bounded session
            _t = time.monotonic()
            obs = adapter.get_observation()
            trace.write("real.observation.read", {"step": step})
            safe_obs = adapt_observation(obs, adapter_config, manifest=policy.manifest)
            safe_obs = safe_obs.observation
            observation_ms = (time.monotonic() - _t) * 1000.0
            _t = time.monotonic()
            pred = policy.predict_action(safe_obs, return_metadata=True)
            raw_action = pred["action"]
            policy_ms = (time.monotonic() - _t) * 1000.0
            trace.write("real.action.generated", {"step": step})

            ctx = SafetyContext(mode="real", step=step, robot_type=config.robot_type,
                                policy_type=policy.policy_type, action_source="policy")
            _t = time.monotonic()
            decision = guard.send_action(raw_action, ctx)
            egress_ms = (time.monotonic() - _t) * 1000.0
            metrics.add(observation_ms=observation_ms, policy_ms=policy_ms,
                        egress_ms=egress_ms, loop_ms=(time.monotonic() - step_start) * 1000.0)
            acc.add(_decision_obj(decision))
            append_safety_decision(safety_report_path,
                                   _decision_obj(decision), context=ctx)
            _append_jsonl(actions_path, {
                "step": step, "timestamp": now_iso(), "allowed": decision.allowed,
                "reason": decision.reason, "action_hash": decision.action_hash,
                "sent_to_robot": decision.sent,
            })

            if not decision.allowed:
                # Real mode stops on the first non-egress decision.
                stop_reason = decision.reason
                break
            sent += 1
            _sleep_to_fps(config.fps)
        else:
            stop_reason = "max_steps_reached"

        adapter.stop()
        trace.write("real.adapter.stop", {"reason": stop_reason})
        adapter.disconnect()
        trace.write("real.adapter.disconnect", {})
        adapter_connected = False
        session["status"] = "stopped"
    except Exception as e:  # fail-closed cleanup
        fatal = e
        stop_reason = f"exception: {type(e).__name__}: {e}"
        trace.write("real.session.failed", {"error": str(e)})
        try:
            adapter.stop()
            trace.write("real.adapter.stop", {"reason": "exception"})
        finally:
            if adapter_connected:
                adapter.disconnect()
                trace.write("real.adapter.disconnect", {})
        session["status"] = "failed"
    finally:
        if prev_sigint is not None:
            try:
                signal.signal(signal.SIGINT, prev_sigint)
            except (ValueError, OSError):
                pass

    # v1.0.5: per-step runtime metrics.
    if metrics.steps:
        wall = time.monotonic() - loop_start
        metrics_report = build_real_metrics_report(metrics, wall_seconds=wall)
        save_json(output_dir / "real_metrics.json", metrics_report)
        (output_dir / "real_metrics.md").write_text(build_real_metrics_markdown(metrics_report))
        write_real_metrics_csv(output_dir / "real_metrics.csv", metrics.steps)

    save_json(output_dir / "real_session.json", _redact(session, config))
    summary = build_safety_summary(acc)
    save_json(output_dir / "safety_summary.json", summary)
    # v1.0.2: post-session safety quality gate (real-strict: zero blocks/findings).
    try:
        from .safety_quality import (
            SafetyQualityConfig, build_safety_quality_markdown,
            build_safety_quality_report, evaluate_safety_quality,
        )
        if summary.get("actions_supervised", 0):
            sq = evaluate_safety_quality(summary, SafetyQualityConfig())
            sq_report = build_safety_quality_report(
                sq, source={"type": "real_session", "path": str(output_dir)})
            save_json(output_dir / "real_safety_quality_report.json", sq_report)
            (output_dir / "real_safety_quality_report.md").write_text(
                build_safety_quality_markdown(sq_report))
    except Exception:
        pass  # best-effort auxiliary artifact
    sup_summary = {
        "mode": "enforce", "profile": profile.name,
        "actions_supervised": acc.actions_supervised,
        "actions_allowed": acc.actions_allowed,
        "actions_blocked": acc.actions_blocked,
    }
    trace.write("real.session.completed", {"ok": fatal is None, "sent": sent})
    trace.close()

    ok = fatal is None and stop_reason in ("max_steps_reached", "duration_seconds_reached")
    return _finalize(
        ok, stop_reason, session, sup_summary, sent, acc.actions_blocked,
        estop=guard.estop_triggered, deadman_lost=guard.deadman_lost,
        ready=ready, approval_valid=approval_valid, approval_id=approval_id,
        robot_ready=False)


def _decision_obj(decision):
    """Wrap a RealEgressDecision's supervisor decision for the accumulator.

    Uses the supervisor's own decision when present; otherwise synthesizes a
    blocked/critical decision (deadman/estop/rate/adapter block)."""
    from .safety_supervisor import SafetyDecision
    sd = decision.supervisor_decision
    if sd is not None:
        return SafetyDecision(
            allowed=sd.get("allowed", decision.allowed),
            action_modified=sd.get("action_modified", False),
            reasons=sd.get("reasons", []), checks=sd.get("checks", []),
            profile=sd.get("profile", "?"), mode=sd.get("mode", "enforce"),
            severity=sd.get("severity", "info"))
    return SafetyDecision(
        allowed=decision.allowed, action_modified=False,
        reasons=[decision.reason], checks=[], profile="?", mode="enforce",
        severity="info" if decision.allowed else "critical")


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def _sleep_to_fps(fps: float) -> None:
    # Pace egress to the bounded fps. This is the same interval the egress
    # guard's rate limiter enforces, so the two agree instead of fighting.
    if fps and fps > 0:
        time.sleep(1.0 / fps)
