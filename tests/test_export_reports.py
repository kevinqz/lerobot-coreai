# test_export_reports.py — schema validation for export reports.

import json
import pytest
from importlib.resources import files
import jsonschema


def _load_export_schema():
    schema_text = files("lerobot_coreai.schemas").joinpath("export-report.schema.json").read_text()
    return json.loads(schema_text)


class TestExportReportSchema:
    def test_schema_exists(self):
        s = _load_export_schema()
        assert s["title"] == "lerobot-coreai export report"

    def test_valid_report_validates(self):
        s = _load_export_schema()
        report = {
            "schema_version": "lerobot-coreai.export_report.v0",
            "lerobot_coreai_version": "0.6.0",
            "ok": True,
            "mode": "export_verify_package",
            "source": {}, "artifact": {}, "fabric": {},
            "verification": {},
            "claims": {"proves_numeric_action_fidelity": True, "proves_task_success": False, "proves_robot_safety": False, "publish_ready": True},
            "safety": {"physical_actuation_possible": False, "motor_commands_available": False, "robot_connected": False, "actions_sent": 0},
            "errors": [],
        }
        jsonschema.validate(instance=report, schema=s)

    def test_rejects_proves_task_success_true(self):
        s = _load_export_schema()
        report = {
            "schema_version": "lerobot-coreai.export_report.v0",
            "lerobot_coreai_version": "0.6.0",
            "ok": True, "mode": "export_verify_package",
            "source": {}, "artifact": {}, "fabric": {}, "verification": {},
            "claims": {"proves_numeric_action_fidelity": True, "proves_task_success": True, "proves_robot_safety": False, "publish_ready": True},
            "safety": {"physical_actuation_possible": False, "motor_commands_available": False, "robot_connected": False, "actions_sent": 0},
            "errors": [],
        }
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=report, schema=s)

    def test_rejects_actions_sent_nonzero(self):
        s = _load_export_schema()
        report = {
            "schema_version": "lerobot-coreai.export_report.v0",
            "lerobot_coreai_version": "0.6.0",
            "ok": True, "mode": "export_verify_package",
            "source": {}, "artifact": {}, "fabric": {}, "verification": {},
            "claims": {"proves_numeric_action_fidelity": True, "proves_task_success": False, "proves_robot_safety": False, "publish_ready": True},
            "safety": {"physical_actuation_possible": False, "motor_commands_available": False, "robot_connected": False, "actions_sent": 5},
            "errors": [],
        }
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=report, schema=s)
