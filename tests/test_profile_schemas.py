# test_profile_schemas.py — schema validation for profile reports (v0.9.1).

import json
from importlib.resources import files

import jsonschema
import pytest


def _schema(name):
    return json.loads(files("lerobot_coreai.schemas").joinpath(name).read_text())


def _valid_calibration():
    return {
        "schema_version": "lerobot-coreai.profile_calibration_report.v0",
        "ok": True, "actions_path": "runs/x/actions.jsonl", "samples": 100,
        "statistics": {"abs": {}, "delta": {}, "l2_norm": {}},
        "recommended_bounds": {"max_abs_action": 0.7},
        "claims": {
            "proves_profile_fit_to_observed_actions": True,
            "proves_future_action_safety": False,
            "proves_physical_safety": False,
            "proves_real_world_safety": False,
            "proves_real_task_success": False,
        },
    }


def test_valid_calibration_report_passes():
    jsonschema.validate(_valid_calibration(), _schema("profile-calibration-report.schema.json"))


def test_calibration_physical_safety_overclaim_fails():
    r = _valid_calibration()
    r["claims"]["proves_physical_safety"] = True
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(r, _schema("profile-calibration-report.schema.json"))


def test_calibration_future_safety_overclaim_fails():
    r = _valid_calibration()
    r["claims"]["proves_future_action_safety"] = True
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(r, _schema("profile-calibration-report.schema.json"))


def _valid_comparison():
    return {
        "schema_version": "lerobot-coreai.profile_comparison_report.v0",
        "profile_a": "a", "profile_b": "b", "actions_supervised": 10,
        "agreement_rate": 0.9,
        "claims": {
            "proves_profile_equivalence": False,
            "proves_physical_safety": False,
            "proves_real_world_safety": False,
        },
    }


def test_valid_comparison_report_passes():
    jsonschema.validate(_valid_comparison(), _schema("profile-comparison-report.schema.json"))


def test_comparison_equivalence_overclaim_fails():
    r = _valid_comparison()
    r["claims"]["proves_profile_equivalence"] = True
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(r, _schema("profile-comparison-report.schema.json"))


def test_comparison_physical_safety_overclaim_fails():
    r = _valid_comparison()
    r["claims"]["proves_physical_safety"] = True
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(r, _schema("profile-comparison-report.schema.json"))
