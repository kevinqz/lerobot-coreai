# test_safety_schemas.py — schema validation for safety artifacts (v0.9.0).

import json
from importlib.resources import files

import jsonschema
import pytest


def _schema(name):
    return json.loads(files("lerobot_coreai.schemas").joinpath(name).read_text())


# -- profile --

def _valid_profile():
    return {
        "schema_version": "lerobot-coreai.safety_profile.v0",
        "name": "p", "mode": "fail_closed", "profile_type": "software_bounds",
        "allow_nan": False, "allow_inf": False,
    }


def test_valid_profile_passes():
    jsonschema.validate(_valid_profile(), _schema("safety-profile.schema.json"))


def test_profile_bad_mode_fails():
    p = _valid_profile()
    p["mode"] = "fail_open"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(p, _schema("safety-profile.schema.json"))


def test_profile_bad_type_fails():
    p = _valid_profile()
    p["profile_type"] = "hardware_certified"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(p, _schema("safety-profile.schema.json"))


def test_profile_missing_type_fails():
    p = _valid_profile()
    del p["profile_type"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(p, _schema("safety-profile.schema.json"))


def test_profile_missing_name_fails():
    p = _valid_profile()
    del p["name"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(p, _schema("safety-profile.schema.json"))


def test_builtin_profiles_validate():
    for name in ("default-sim-safe.json", "so100-sim-default.json",
                 "so101-sim-default.json", "generic-7dof-sim-default.json",
                 "pusht-sim-default.json"):
        data = json.loads(files("lerobot_coreai.profiles").joinpath(name).read_text())
        jsonschema.validate(data, _schema("safety-profile.schema.json"))


# -- decision --

def _valid_decision():
    return {
        "allowed": True, "action_modified": False, "reasons": [],
        "checks": [{"name": "finite", "passed": True}],
        "profile": "p", "mode": "enforce", "severity": "info",
    }


def test_valid_decision_passes():
    jsonschema.validate(_valid_decision(), _schema("safety-decision.schema.json"))


def test_decision_bad_mode_fails():
    d = _valid_decision()
    d["mode"] = "bogus"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(d, _schema("safety-decision.schema.json"))


def test_decision_bad_severity_fails():
    d = _valid_decision()
    d["severity"] = "meh"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(d, _schema("safety-decision.schema.json"))


# -- report/summary --

def _valid_summary():
    return {
        "schema_version": "lerobot-coreai.safety_summary.v0",
        "profile": "p", "mode": "enforce",
        "actions_supervised": 10, "actions_allowed": 9, "actions_blocked": 1,
        "actions_modified": 2, "critical_failures": 1,
        "claims": {
            "proves_software_supervision": True,
            "proves_physical_safety": False,
            "proves_real_world_safety": False,
            "proves_real_task_success": False,
        },
    }


def test_valid_summary_passes():
    jsonschema.validate(_valid_summary(), _schema("safety-report.schema.json"))


def test_summary_physical_safety_true_fails():
    s = _valid_summary()
    s["claims"]["proves_physical_safety"] = True
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(s, _schema("safety-report.schema.json"))


def test_summary_real_world_safety_true_fails():
    s = _valid_summary()
    s["claims"]["proves_real_world_safety"] = True
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(s, _schema("safety-report.schema.json"))


def test_summary_missing_claims_fails():
    s = _valid_summary()
    del s["claims"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(s, _schema("safety-report.schema.json"))
