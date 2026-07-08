# test_compare_reports.py — schema validation for compare reports.

import json
import pytest
from importlib.resources import files
import jsonschema


def _load_compare_schema():
    schema_text = files("lerobot_coreai.schemas").joinpath("compare-report.schema.json").read_text()
    return json.loads(schema_text)


class TestCompareReportSchema:
    def test_schema_exists(self):
        s = _load_compare_schema()
        assert s["title"] == "lerobot-coreai compare report"

    def test_valid_report_validates(self):
        s = _load_compare_schema()
        report = {
            "schema_version": "lerobot-coreai.compare_report.v0",
            "lerobot_coreai_version": "0.5.0",
            "ok": True,
            "mode": "dataset_compare",
            "policy": {"torch": {}, "coreai": {}},
            "dataset": {},
            "runner": {},
            "metrics": {},
            "claims": {"proves_numeric_action_fidelity": True, "proves_task_success": False, "proves_robot_safety": False},
            "safety": {"physical_actuation_possible": False, "motor_commands_available": False,
                       "robot_connected": False, "actions_sent": 0},
            "errors": [],
        }
        jsonschema.validate(instance=report, schema=s)

    def test_rejects_proves_task_success_true(self):
        s = _load_compare_schema()
        report = {
            "schema_version": "lerobot-coreai.compare_report.v0",
            "lerobot_coreai_version": "0.5.0",
            "ok": True,
            "mode": "dataset_compare",
            "policy": {}, "dataset": {}, "runner": {},
            "metrics": {},
            "claims": {"proves_numeric_action_fidelity": True, "proves_task_success": True, "proves_robot_safety": False},
            "safety": {"physical_actuation_possible": False, "motor_commands_available": False,
                       "robot_connected": False, "actions_sent": 0},
            "errors": [],
        }
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=report, schema=s)

    def test_rejects_actions_sent_nonzero(self):
        s = _load_compare_schema()
        report = {
            "schema_version": "lerobot-coreai.compare_report.v0",
            "lerobot_coreai_version": "0.5.0",
            "ok": True,
            "mode": "dataset_compare",
            "policy": {}, "dataset": {}, "runner": {},
            "metrics": {},
            "claims": {"proves_numeric_action_fidelity": True, "proves_task_success": False, "proves_robot_safety": False},
            "safety": {"physical_actuation_possible": False, "motor_commands_available": False,
                       "robot_connected": False, "actions_sent": 5},
            "errors": [],
        }
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=report, schema=s)
