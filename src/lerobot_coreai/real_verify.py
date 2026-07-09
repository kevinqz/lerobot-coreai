# real_verify.py — offline audit verifier for a guarded real session (v1.0.2).
#
# Re-checks, after the fact, that a real session's artifacts are internally
# consistent and honest: schema-valid, no overclaim, action accounting matches,
# every sent action was supervisor-allowed, and the trace has the required order.
# It can also re-verify the readiness/approval/bundle the session claimed.

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any


@dataclass
class RealSessionVerifyResult:
    ok: bool
    checks: list[dict[str, Any]] = field(default_factory=list)


def _schema(name: str) -> dict[str, Any]:
    return json.loads(files("lerobot_coreai.schemas").joinpath(name).read_text())


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def verify_real_session(
    run_dir: Path, *, bundle_dir: Path | None = None,
    approval: Path | None = None, readiness_report: Path | None = None,
) -> RealSessionVerifyResult:
    """Verify a completed real session directory. Read-only, fail-closed."""
    run_dir = Path(run_dir)
    checks: list[dict[str, Any]] = []

    def _c(name, passed, message=""):
        checks.append({"name": name, "passed": bool(passed), "message": message})

    report_path = run_dir / "real_report.json"
    session_path = run_dir / "real_session.json"
    if not report_path.is_file():
        _c("real_report_exists", False, str(report_path))
        return RealSessionVerifyResult(ok=False, checks=checks)
    _c("real_report_exists", True)
    report = _read_json(report_path)

    # Schema validity.
    try:
        import jsonschema
        jsonschema.validate(report, _schema("real-report.schema.json"))
        _c("real_report_schema_valid", True)
    except Exception as e:
        _c("real_report_schema_valid", False, getattr(e, "message", str(e)))

    if session_path.is_file():
        _c("real_session_exists", True)
        try:
            import jsonschema
            jsonschema.validate(_read_json(session_path),
                                _schema("real-session.schema.json"))
            _c("real_session_schema_valid", True)
        except Exception as e:
            _c("real_session_schema_valid", False, getattr(e, "message", str(e)))
    else:
        _c("real_session_exists", False)

    # No overclaim.
    claims = report.get("claims", {}) or {}
    _c("no_overclaim",
       claims.get("proves_physical_safety") is False
       and claims.get("proves_real_world_safety") is False
       and claims.get("authorizes_unrestricted_real_world_actuation") is False)

    egress = report.get("egress", {}) or {}
    reported_sent = egress.get("actions_sent_to_robot", 0)

    # Action accounting from real_actions.jsonl.
    actions_path = run_dir / "real_actions.jsonl"
    if actions_path.is_file():
        actions = _read_jsonl(actions_path)
        sent_records = [a for a in actions if a.get("sent_to_robot")]
        _c("actions_sent_count_matches", len(sent_records) == reported_sent,
           f"records={len(sent_records)} report={reported_sent}")
        # Every sent action must have been allowed; no blocked action was sent.
        _c("every_sent_action_allowed",
           all(a.get("allowed") for a in sent_records))
        _c("no_blocked_action_sent",
           all(not a.get("sent_to_robot") for a in actions if not a.get("allowed")))
    else:
        _c("real_actions_exists", False)

    # Trace order invariants.
    trace_path = run_dir / "real_trace.jsonl"
    if trace_path.is_file():
        events = [e.get("event") for e in _read_jsonl(trace_path)]
        _c("trace_has_terminal_event",
           "real.session.completed" in events or "real.session.failed" in events)

        def _first(ev):
            return events.index(ev) if ev in events else None

        armed = _first("real.session.armed")
        first_gen = _first("real.action.generated")
        first_sent = _first("real.egress.sent")
        first_decision = _first("real.supervisor.decision")
        # No action generated before the session is armed.
        _c("no_action_before_armed",
           first_gen is None or (armed is not None and armed < first_gen))
        # No egress sent before a supervisor decision.
        _c("no_egress_before_supervisor",
           first_sent is None or (first_decision is not None and first_decision < first_sent))
        # If any egress happened, the adapter was stopped afterwards.
        if "real.egress.sent" in events:
            _c("adapter_stopped_after_egress", "real.adapter.stop" in events)
    else:
        _c("real_trace_exists", False)

    # Optional: re-verify the readiness/approval/bundle the session claimed.
    if bundle_dir is not None and approval is not None:
        try:
            from .operator_approval import verify_approval
            av = verify_approval(Path(bundle_dir), Path(approval))
            _c("approval_still_valid", av.approval_valid)
            _c("approval_not_expired", not av.expired)
        except Exception as e:
            _c("approval_still_valid", False, str(e))
    if readiness_report is not None:
        rp = Path(readiness_report)
        if rp.is_file():
            try:
                import jsonschema
                rr = _read_json(rp)
                jsonschema.validate(rr, _schema("release-readiness-report.schema.json"))
                _c("readiness_still_ready", rr.get("ready") is True)
            except Exception as e:
                _c("readiness_still_ready", False, getattr(e, "message", str(e)))
        else:
            _c("readiness_report_exists", False, str(rp))

    ok = all(c["passed"] for c in checks)
    return RealSessionVerifyResult(ok=ok, checks=checks)
