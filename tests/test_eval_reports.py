# test_eval_reports.py — schema validation for eval reports.

import json
import pytest
from pathlib import Path
from importlib.resources import files
import jsonschema

from lerobot_coreai.reports import build_success_report


def _load_eval_schema():
    schema_text = files("lerobot_coreai.schemas").joinpath("eval-report.schema.json").read_text()
    return json.loads(schema_text)


class TestEvalReportSchema:
    def test_eval_report_schema_exists(self):
        schema = _load_eval_schema()
        assert schema["title"] == "lerobot-coreai eval report"

    def test_eval_report_validates(self):
        """A well-formed eval report should validate against the schema."""
        schema = _load_eval_schema()
        report = {
            "schema_version": "lerobot-coreai.eval_report.v0",
            "lerobot_coreai_version": "0.4.0",
            "ok": True,
            "mode": "dataset_eval",
            "policy": {"path": "test", "type": "evo1"},
            "dataset": {"repo_id": "test"},
            "runner": {"url": "http://x", "reachable": True, "supports_action": True},
            "metrics": {
                "frames_requested": 5,
                "frames_processed": 5,
                "actions_generated": 5,
                "actions_failed": 0,
                "shape_errors": 0,
                "nan_errors": 0,
                "inf_errors": 0,
                "runner_errors": 0,
                "mean_total_ms": 12.0,
                "p95_total_ms": 15.0,
            },
            "safety": {
                "physical_actuation_possible": False,
                "motor_commands_available": False,
                "robot_connected": False,
                "actions_sent": 0,
            },
            "errors": [],
        }
        jsonschema.validate(instance=report, schema=schema)

    def test_eval_report_rejects_actions_sent_nonzero(self):
        """Schema should reject actions_sent != 0."""
        schema = _load_eval_schema()
        report = {
            "schema_version": "lerobot-coreai.eval_report.v0",
            "lerobot_coreai_version": "0.4.0",
            "ok": True,
            "mode": "dataset_eval",
            "policy": {},
            "dataset": {},
            "runner": {},
            "metrics": {"frames_requested": 0, "frames_processed": 0, "actions_generated": 0, "actions_failed": 0},
            "safety": {"physical_actuation_possible": False, "motor_commands_available": False,
                       "robot_connected": False, "actions_sent": 1},
            "errors": [],
        }
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=report, schema=schema)
