# test_shadow_reports.py — schema validation for shadow reports.

import json
import pytest
from importlib.resources import files
import jsonschema


def _load_shadow_schema():
    schema_text = files("lerobot_coreai.schemas").joinpath("shadow-report.schema.json").read_text()
    return json.loads(schema_text)


def _valid_report():
    """A minimal valid shadow report satisfying all invariants."""
    return {
        "schema_version": "lerobot-coreai.shadow_report.v0",
        "lerobot_coreai_version": "0.7.0",
        "ok": True,
        "mode": "shadow",
        "policy": {"path": "kevinqz/EVO1-SO100-CoreAI"},
        "runner": {"url": "http://localhost:8710", "reachable": True, "supports_action": True},
        "observation_source": {"type": "folder", "opened": True, "closed": True},
        "loop": {"fps_target": 10.0, "steps_requested": 32, "steps_completed": 32},
        "metrics": {
            "observations_read": 32,
            "actions_generated": 32,
            "actions_blocked": 32,
            "actions_sent": 0,
        },
        "claims": {
            "proves_runtime_action_generation": True,
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
        },
        "files": {"actions": "actions.jsonl"},
        "errors": [],
    }


class TestShadowReportSchema:
    def test_schema_exists(self):
        s = _load_shadow_schema()
        assert s["title"] == "lerobot-coreai shadow report"

    def test_valid_report_validates(self):
        s = _load_shadow_schema()
        jsonschema.validate(instance=_valid_report(), schema=s)

    def test_rejects_actions_sent_nonzero_in_safety(self):
        s = _load_shadow_schema()
        report = _valid_report()
        report["safety"]["actions_sent"] = 1
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=report, schema=s)

    def test_rejects_actions_sent_nonzero_in_metrics(self):
        s = _load_shadow_schema()
        report = _valid_report()
        report["metrics"]["actions_sent"] = 5
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=report, schema=s)

    def test_rejects_action_egress_not_blocked(self):
        s = _load_shadow_schema()
        report = _valid_report()
        report["safety"]["action_egress"] = "open"
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=report, schema=s)

    def test_rejects_proves_task_success_true(self):
        s = _load_shadow_schema()
        report = _valid_report()
        report["claims"]["proves_task_success"] = True
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=report, schema=s)

    def test_rejects_proves_robot_safety_true(self):
        s = _load_shadow_schema()
        report = _valid_report()
        report["claims"]["proves_robot_safety"] = True
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=report, schema=s)

    def test_rejects_actuation_device_connected_true(self):
        s = _load_shadow_schema()
        report = _valid_report()
        report["safety"]["actuation_device_connected"] = True
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=report, schema=s)

    def test_rejects_robot_connected_true(self):
        s = _load_shadow_schema()
        report = _valid_report()
        report["safety"]["robot_connected"] = True
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=report, schema=s)

    def test_rejects_motor_commands_available_true(self):
        s = _load_shadow_schema()
        report = _valid_report()
        report["safety"]["motor_commands_available"] = True
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=report, schema=s)

    def test_rejects_mode_not_shadow(self):
        s = _load_shadow_schema()
        report = _valid_report()
        report["mode"] = "real"
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=report, schema=s)
