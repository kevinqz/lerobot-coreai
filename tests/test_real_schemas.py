# test_real_schemas.py — schema validation for real session/report/preflight (v1.0.0).

import json
from importlib.resources import files

import jsonschema
import pytest


def _schema(name):
    return json.loads(files("lerobot_coreai.schemas").joinpath(name).read_text())


def _session():
    return {
        "schema_version": "lerobot-coreai.real_session.v0", "session_id": "real_x",
        "created_at": "2026-07-09T00:00:00Z", "operator": "K", "mode": "guarded",
        "robot_adapter": "mock", "robot_type": "so100", "policy_path": "p",
        "safety_profile": "prof.json", "readiness_report": "r.json",
        "approval": "a.json", "max_steps": 10, "fps": 2, "status": "running",
    }


def test_valid_session_passes():
    jsonschema.validate(_session(), _schema("real-session.schema.json"))


def test_session_bad_mode_fails():
    s = _session(); s["mode"] = "autonomous"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(s, _schema("real-session.schema.json"))


def _report(mode="guarded_real", sent=5, egress=True):
    return {
        "schema_version": "lerobot-coreai.real_report.v0",
        "lerobot_coreai_version": "1.0.0", "ok": True, "mode": mode,
        "session_id": "real_x",
        "robot": {"adapter": "mock", "type": "so100"}, "policy": {},
        "readiness": {"ready": True}, "approval": {"valid": True},
        "limits": {"max_steps": 10, "fps": 2}, "safety_supervisor": {"mode": "enforce"},
        "egress": {"robot_egress_enabled": egress, "action_egress":
                   "guarded_real" if mode == "guarded_real" else "none",
                   "actions_sent_to_robot": sent, "actions_blocked_before_robot": 0},
        "stop": {},
        "claims": {
            "proves_guarded_real_session_executed": True,
            "proves_physical_safety": False, "proves_real_world_safety": False,
            "authorizes_unrestricted_real_world_actuation": False,
        },
    }


def test_valid_report_passes():
    jsonschema.validate(_report(), _schema("real-report.schema.json"))


def test_report_physical_overclaim_fails():
    r = _report(); r["claims"]["proves_physical_safety"] = True
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(r, _schema("real-report.schema.json"))


def test_report_actuation_overclaim_fails():
    r = _report(); r["claims"]["authorizes_unrestricted_real_world_actuation"] = True
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(r, _schema("real-report.schema.json"))


def test_preflight_mode_report_must_have_zero_egress():
    # A preflight-mode report claiming robot egress must fail schema.
    r = _report(mode="preflight", sent=3, egress=True)
    r["egress"]["action_egress"] = "guarded_real"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(r, _schema("real-report.schema.json"))


def test_preflight_mode_zero_egress_passes():
    r = _report(mode="preflight", sent=0, egress=False)
    r["egress"]["action_egress"] = "none"
    jsonschema.validate(r, _schema("real-report.schema.json"))


def _preflight():
    return {
        "schema_version": "lerobot-coreai.real_preflight.v0",
        "lerobot_coreai_version": "1.0.0", "ok": True, "mode": "guarded",
        "actions_sent_to_robot": 0,
        "checks": [{"name": "x", "passed": True}],
        "claims": {"proves_preflight_passed": True, "proves_physical_safety": False,
                   "proves_real_world_safety": False,
                   "authorizes_unrestricted_real_world_actuation": False},
    }


def test_valid_preflight_passes():
    jsonschema.validate(_preflight(), _schema("real-preflight.schema.json"))


def test_preflight_nonzero_actions_fails():
    p = _preflight(); p["actions_sent_to_robot"] = 1
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(p, _schema("real-preflight.schema.json"))
