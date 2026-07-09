# test_safety_quality_schemas.py — schema validation for safety gate/regression (v0.9.2).

import json
from importlib.resources import files

import jsonschema
import pytest


def _schema(name):
    return json.loads(files("lerobot_coreai.schemas").joinpath(name).read_text())


def _valid_quality():
    return {
        "schema_version": "lerobot-coreai.safety_quality_report.v0",
        "lerobot_coreai_version": "0.9.2", "passed": False,
        "summary": {"actions_blocked": 1}, "checks": [],
        "claims": {
            "proves_software_safety_quality": True,
            "proves_physical_safety": False,
            "proves_real_world_safety": False,
            "proves_real_task_success": False,
        },
    }


def test_valid_quality_report_passes():
    jsonschema.validate(_valid_quality(), _schema("safety-quality-report.schema.json"))


def test_quality_physical_overclaim_fails():
    r = _valid_quality()
    r["claims"]["proves_physical_safety"] = True
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(r, _schema("safety-quality-report.schema.json"))


def test_quality_real_world_overclaim_fails():
    r = _valid_quality()
    r["claims"]["proves_real_world_safety"] = True
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(r, _schema("safety-quality-report.schema.json"))


def _valid_regression():
    return {
        "schema_version": "lerobot-coreai.safety_regression_report.v0",
        "passed": True, "baseline": {}, "candidate": {}, "deltas": {}, "checks": [],
        "claims": {
            "proves_no_safety_regression_on_compared_artifacts": True,
            "proves_physical_safety": False,
            "proves_real_world_safety": False,
        },
    }


def test_valid_regression_report_passes():
    jsonschema.validate(_valid_regression(), _schema("safety-regression-report.schema.json"))


def test_regression_physical_overclaim_fails():
    r = _valid_regression()
    r["claims"]["proves_physical_safety"] = True
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(r, _schema("safety-regression-report.schema.json"))


def test_regression_real_world_overclaim_fails():
    r = _valid_regression()
    r["claims"]["proves_real_world_safety"] = True
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(r, _schema("safety-regression-report.schema.json"))
