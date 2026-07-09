# real_reports.py — session manifest + real report builders (v1.0.0).
#
# These record a guarded real session. They never claim physical safety,
# real-world success, or authorization for unrestricted actuation.

from __future__ import annotations

from typing import Any

from . import __version__

REAL_SESSION_SCHEMA_VERSION = "lerobot-coreai.real_session.v0"
REAL_REPORT_SCHEMA_VERSION = "lerobot-coreai.real_report.v0"


def build_real_session(*, session_id, created_at, operator, mode, robot_adapter,
                       robot_type, policy_path, runner_url, safety_profile,
                       readiness_report, approval, max_steps, fps,
                       duration_seconds=None, status="initialized") -> dict[str, Any]:
    return {
        "schema_version": REAL_SESSION_SCHEMA_VERSION,
        "session_id": session_id,
        "created_at": created_at,
        "operator": operator,
        "mode": mode,
        "robot_adapter": robot_adapter,
        "robot_type": robot_type,
        "policy_path": policy_path,
        "runner_url": runner_url,
        "safety_profile": str(safety_profile) if safety_profile else None,
        "readiness_report": str(readiness_report) if readiness_report else None,
        "approval": str(approval) if approval else None,
        "max_steps": max_steps,
        "fps": fps,
        "duration_seconds": duration_seconds,
        "status": status,
    }


def build_real_report(*, ok, mode, session, robot_ready, supervisor_summary,
                      actions_sent_to_robot, actions_blocked_before_robot,
                      readiness_ready, approval_valid, approval_id,
                      stop_reason, estop_triggered, deadman_lost) -> dict[str, Any]:
    guarded = mode == "guarded"
    report_mode = "guarded_real" if guarded else "preflight"
    egress_enabled = guarded and actions_sent_to_robot > 0
    return {
        "schema_version": REAL_REPORT_SCHEMA_VERSION,
        "lerobot_coreai_version": __version__,
        "ok": ok,
        "mode": report_mode,
        "session_id": session.get("session_id"),
        "operator": session.get("operator"),
        "robot": {
            "adapter": session.get("robot_adapter"),
            "type": session.get("robot_type"),
            "ready": robot_ready,
        },
        "policy": {"path": session.get("policy_path"), "runtime": "coreai"},
        "readiness": {"report": session.get("readiness_report"), "ready": readiness_ready},
        "approval": {"path": session.get("approval"), "valid": approval_valid,
                     "approval_id": approval_id},
        "limits": {
            "max_steps": session.get("max_steps"),
            "fps": session.get("fps"),
            "duration_seconds": session.get("duration_seconds"),
        },
        "safety_supervisor": supervisor_summary,
        "egress": {
            # path_enabled: the guarded real egress path was armed this session.
            # robot_egress_enabled: at least one action actually reached the robot.
            "robot_egress_path_enabled": guarded,
            "robot_egress_enabled": egress_enabled,
            "action_egress": "guarded_real" if guarded else "none",
            "actions_sent_to_robot": actions_sent_to_robot,
            "actions_blocked_before_robot": actions_blocked_before_robot,
        },
        "stop": {
            "reason": stop_reason,
            "estop_triggered": estop_triggered,
            "deadman_lost": deadman_lost,
        },
        "claims": {
            "proves_guarded_real_session_executed": bool(ok and guarded),
            "proves_physical_safety": False,
            "proves_real_world_safety": False,
            "authorizes_unrestricted_real_world_actuation": False,
        },
    }


def build_real_report_markdown(report: dict[str, Any]) -> str:
    eg = report.get("egress", {})
    stop = report.get("stop", {})
    ss = report.get("safety_supervisor", {})
    return (
        "# Guarded Real Session Report\n\n"
        f"- OK: {report.get('ok')}\n"
        f"- Mode: {report.get('mode')}\n"
        f"- Session: {report.get('session_id')}\n"
        f"- Operator: {report.get('operator')}\n"
        f"- Robot adapter: {report.get('robot', {}).get('adapter')}\n"
        f"- Robot egress enabled: {eg.get('robot_egress_enabled')}\n"
        f"- Actions sent to robot: {eg.get('actions_sent_to_robot')}\n"
        f"- Actions blocked before robot: {eg.get('actions_blocked_before_robot')}\n"
        f"- Supervisor blocked: {ss.get('actions_blocked')}\n"
        f"- Stop reason: {stop.get('reason')}\n"
        f"- E-stop triggered: {stop.get('estop_triggered')}\n"
        f"- Deadman lost: {stop.get('deadman_lost')}\n\n"
        "## Claims\n\n"
        "This run executed a bounded guarded real session under verified software "
        "readiness, operator approval, and enforced safety supervision. "
        "It does not prove physical robot safety. It does not prove real-world "
        "task success. It does not authorize unrestricted real-world actuation.\n"
    )
