# test_sim_bundle_schema.py — schema validation for bundle_manifest.json (v0.8.4).

import json
from importlib.resources import files

import jsonschema
import pytest


def _load_schema():
    schema_text = files("lerobot_coreai.schemas").joinpath("sim-bundle.schema.json").read_text()
    return json.loads(schema_text)


def _valid_manifest():
    return {
        "schema_version": "lerobot-coreai.sim_bundle.v0",
        "lerobot_coreai_version": "0.8.4",
        "created_at": "2026-07-09T12:00:00Z",
        "created_by": "lerobot-coreai",
        "bundle_type": "sim_run",
        "mode": "sim",
        "policy": {"path": "kevinqz/EVO1-SO100-CoreAI"},
        "environment": {"type": "gym", "id": "PushT-v0"},
        "runner": {"url": "http://127.0.0.1:8710", "redacted": False},
        "results": {"ok": True},
        "analytics": {"has_episode_metrics": True},
        "safety": {
            "simulator_egress_enabled": True,
            "robot_egress_enabled": False,
            "actions_sent_to_robot": 0,
            "action_egress": "simulator_only",
            "physical_actuation_possible": False,
        },
        "claims": {
            "proves_sim_task_success": True,
            "proves_real_task_success": False,
            "proves_robot_safety": False,
            "proves_real_world_safety": False,
        },
        "files": {"report": "source_run/sim_report.json"},
        "warnings": [],
    }


def test_valid_manifest_passes():
    jsonschema.validate(_valid_manifest(), _load_schema())


def test_mode_not_sim_fails():
    m = _valid_manifest()
    m["mode"] = "real"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(m, _load_schema())


def test_bundle_type_wrong_fails():
    m = _valid_manifest()
    m["bundle_type"] = "real_run"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(m, _load_schema())


def test_robot_egress_enabled_fails():
    m = _valid_manifest()
    m["safety"]["robot_egress_enabled"] = True
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(m, _load_schema())


def test_actions_sent_to_robot_nonzero_fails():
    m = _valid_manifest()
    m["safety"]["actions_sent_to_robot"] = 5
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(m, _load_schema())


def test_action_egress_wrong_fails():
    m = _valid_manifest()
    m["safety"]["action_egress"] = "robot"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(m, _load_schema())


def test_physical_actuation_possible_fails():
    m = _valid_manifest()
    m["safety"]["physical_actuation_possible"] = True
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(m, _load_schema())


def test_proves_real_task_success_fails():
    m = _valid_manifest()
    m["claims"]["proves_real_task_success"] = True
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(m, _load_schema())


def test_proves_robot_safety_fails():
    m = _valid_manifest()
    m["claims"]["proves_robot_safety"] = True
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(m, _load_schema())


def test_proves_real_world_safety_fails():
    m = _valid_manifest()
    m["claims"]["proves_real_world_safety"] = True
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(m, _load_schema())


def test_missing_required_field_fails():
    m = _valid_manifest()
    del m["safety"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(m, _load_schema())


def _minimal_sim_report():
    """A minimal invariant-valid sim_report.json for end-to-end packaging."""
    return {
        "schema_version": "lerobot-coreai.sim_report.v0",
        "lerobot_coreai_version": "0.8.4",
        "ok": True,
        "mode": "sim",
        "policy": {"path": "test/policy", "runtime": "coreai", "type": "evo1"},
        "runner": {"url": "http://localhost:8710", "reachable": True, "supports_action": True},
        "environment": {"type": "gym", "id": "PushT-v0", "episodes": 1,
                        "max_steps_per_episode": 10, "seed": 42,
                        "simulator_egress_enabled": True},
        "loop": {"fps_target": 0, "episodes_completed": 1, "steps_completed": 10},
        "metrics": {"episodes_completed": 1, "steps_completed": 10, "mean_episode_reward": 1.0},
        "episode_metrics": {"success_rate": 1.0, "mean_reward": 1.0},
        "claims": {
            "proves_sim_task_success": True,
            "proves_real_task_success": False,
            "proves_robot_safety": False,
            "proves_real_world_safety": False,
        },
        "safety": {
            "simulator_egress_enabled": True,
            "robot_egress_enabled": False,
            "physical_actuation_possible": False,
            "actions_sent_to_robot": 0,
            "action_egress": "simulator_only",
        },
        "files": {"report": "sim_report.json"},
        "errors": [],
    }


def test_packaged_manifest_validates_against_schema(tmp_path):
    # End-to-end: a manifest produced by package_sim_run must validate.
    import json as _json

    from lerobot_coreai.sim_bundle import SimBundleConfig, package_sim_run

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "sim_report.json").write_text(_json.dumps(_minimal_sim_report()))
    out = tmp_path / "bundle"
    package_sim_run(SimBundleConfig(run_dir=run_dir, output_dir=out))
    manifest = _json.loads((out / "bundle_manifest.json").read_text())
    jsonschema.validate(manifest, _load_schema())
