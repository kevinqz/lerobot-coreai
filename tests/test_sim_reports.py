# test_sim_reports.py — schema validation for sim_report.json (v0.8).

import json
import pytest
from importlib.resources import files

import jsonschema


def _load_sim_schema():
    schema_text = files("lerobot_coreai.schemas").joinpath("sim-report.schema.json").read_text()
    return json.loads(schema_text)


def _valid_report():
    """A minimal valid sim report."""
    return {
        "schema_version": "lerobot-coreai.sim_report.v0",
        "lerobot_coreai_version": "0.8.0",
        "ok": True,
        "mode": "sim",
        "policy": {"path": "test/policy", "runtime": "coreai"},
        "runner": {"url": "http://localhost:8710", "reachable": True},
        "environment": {
            "type": "fake",
            "episodes": 1,
            "max_steps_per_episode": 10,
            "seed": 42,
            "simulator_egress_enabled": True,
        },
        "loop": {"episodes_completed": 1, "steps_completed": 10},
        "metrics": {
            "episodes_requested": 1,
            "episodes_completed": 1,
            "steps_completed": 10,
            "actions_generated": 10,
            "actions_sent_to_simulator": 10,
            "actions_sent_to_robot": 0,
            "success_rate": 1.0,
            "mean_episode_reward": 10.0,
            "mean_loop_ms": 5.0,
            "p95_loop_ms": 8.0,
        },
        "claims": {
            "proves_sim_task_success": True,
            "sim_success_metric_available": True,
            "proves_real_task_success": False,
            "proves_robot_safety": False,
            "proves_real_world_safety": False,
        },
        "safety": {
            "simulator_egress_enabled": True,
            "robot_egress_enabled": False,
            "physical_actuation_possible": False,
            "motor_commands_available": False,
            "robot_connected": False,
            "actions_sent_to_robot": 0,
            "action_egress": "simulator_only",
        },
        "files": {
            "actions": "actions.jsonl",
            "episodes": "episodes.jsonl",
            "observations": "observations.jsonl",
            "trace": "sim_trace.jsonl",
            "report": "sim_report.json",
        },
        "errors": [],
    }


class TestSimReportSchema:
    def test_valid_report_validates(self):
        s = _load_sim_schema()
        jsonschema.validate(instance=_valid_report(), schema=s)

    def test_rejects_wrong_schema_version(self):
        s = _load_sim_schema()
        report = _valid_report()
        report["schema_version"] = "lerobot-coreai.shadow_report.v0"
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=report, schema=s)

    def test_rejects_wrong_mode(self):
        s = _load_sim_schema()
        report = _valid_report()
        report["mode"] = "shadow"
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=report, schema=s)

    def test_rejects_actions_sent_to_robot_nonzero_in_metrics(self):
        s = _load_sim_schema()
        report = _valid_report()
        report["metrics"]["actions_sent_to_robot"] = 1
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=report, schema=s)

    def test_rejects_actions_sent_to_robot_nonzero_in_safety(self):
        s = _load_sim_schema()
        report = _valid_report()
        report["safety"]["actions_sent_to_robot"] = 1
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=report, schema=s)

    def test_rejects_robot_egress_enabled_true(self):
        s = _load_sim_schema()
        report = _valid_report()
        report["safety"]["robot_egress_enabled"] = True
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=report, schema=s)

    def test_rejects_simulator_egress_disabled(self):
        s = _load_sim_schema()
        report = _valid_report()
        report["safety"]["simulator_egress_enabled"] = False
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=report, schema=s)

    def test_rejects_robot_connected_true(self):
        s = _load_sim_schema()
        report = _valid_report()
        report["safety"]["robot_connected"] = True
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=report, schema=s)

    def test_rejects_physical_actuation_possible_true(self):
        s = _load_sim_schema()
        report = _valid_report()
        report["safety"]["physical_actuation_possible"] = True
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=report, schema=s)

    def test_rejects_motor_commands_available_true(self):
        s = _load_sim_schema()
        report = _valid_report()
        report["safety"]["motor_commands_available"] = True
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=report, schema=s)

    def test_rejects_wrong_action_egress(self):
        s = _load_sim_schema()
        report = _valid_report()
        report["safety"]["action_egress"] = "blocked"
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=report, schema=s)

    def test_rejects_proves_real_task_success_true(self):
        s = _load_sim_schema()
        report = _valid_report()
        report["claims"]["proves_real_task_success"] = True
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=report, schema=s)

    def test_rejects_proves_robot_safety_true(self):
        s = _load_sim_schema()
        report = _valid_report()
        report["claims"]["proves_robot_safety"] = True
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=report, schema=s)

    def test_rejects_proves_real_world_safety_true(self):
        s = _load_sim_schema()
        report = _valid_report()
        report["claims"]["proves_real_world_safety"] = True
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=report, schema=s)

    def test_proves_sim_task_success_can_be_false(self):
        """proves_sim_task_success is conditional — False is also valid."""
        s = _load_sim_schema()
        report = _valid_report()
        report["claims"]["proves_sim_task_success"] = False
        jsonschema.validate(instance=report, schema=s)
