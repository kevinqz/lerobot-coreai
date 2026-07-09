# test_external_http_contract.py — external-http controller capability contract (v1.0.3).

import json
from importlib.resources import files

import jsonschema
import pytest

from lerobot_coreai.external_http_contract import validate_controller_preflight


def _pf(**over):
    base = {
        "ok": True,
        "controller_schema_version": "lerobot-coreai.external_http.v0",
        "adapter": "external-http", "robot_type": "so100",
        "action_shape": [16, 7], "observation_keys": ["observation.state", "task"],
        "supports_stop": True, "supports_ready": True,
        "supports_safety_state": True, "physical_estop_required": True,
        "max_fps": 5.0,
    }
    base.update(over)
    return base


def _failed(checks):
    return {n for n, ok, _ in checks if not ok}


def test_valid_controller_passes():
    checks = validate_controller_preflight(
        _pf(), robot_type="so100", profile_action_shape=[16, 7], requested_fps=2.0)
    assert not _failed(checks)


def test_schema_valid_against_file():
    jsonschema.validate(_pf(), json.loads(files("lerobot_coreai.schemas").joinpath(
        "external-http-preflight.schema.json").read_text()))


def test_robot_type_mismatch_fails():
    checks = validate_controller_preflight(
        _pf(robot_type="so101"), robot_type="so100",
        profile_action_shape=[16, 7], requested_fps=2.0)
    assert "external_controller_robot_type_matches" in _failed(checks)


def test_action_shape_mismatch_fails():
    checks = validate_controller_preflight(
        _pf(action_shape=[8, 7]), robot_type="so100",
        profile_action_shape=[16, 7], requested_fps=2.0)
    assert "external_controller_action_shape_matches_profile" in _failed(checks)


def test_requested_fps_above_controller_max_fails():
    checks = validate_controller_preflight(
        _pf(max_fps=1.0), robot_type="so100",
        profile_action_shape=[16, 7], requested_fps=5.0)
    assert "external_controller_max_fps_allows_requested_fps" in _failed(checks)


def test_missing_supports_stop_fails_schema_and_check():
    pf = _pf()
    del pf["supports_stop"]
    checks = validate_controller_preflight(
        pf, robot_type="so100", profile_action_shape=[16, 7], requested_fps=2.0)
    # Schema requires supports_stop → schema check fails and short-circuits.
    assert "external_controller_schema_valid" in _failed(checks)


def test_wrong_schema_version_fails():
    checks = validate_controller_preflight(
        _pf(controller_schema_version="bogus"), robot_type="so100",
        profile_action_shape=[16, 7], requested_fps=2.0)
    assert "external_controller_schema_valid" in _failed(checks)


def test_real_preflight_validates_external_controller(real_ready_scenario):
    # Wire it end-to-end through real_preflight with a fake external adapter.
    from unittest.mock import patch
    from lerobot_coreai.real_preflight import RealPreflightConfig, evaluate_real_preflight

    sc = real_ready_scenario()

    class _FakeExternal:
        name = "external-http"
        robot_type = "so100"
        def preflight(self):
            return _pf(action_shape=[8, 7])  # mismatched shape

    cfg = RealPreflightConfig(
        mode="guarded", policy_path="p", runner_url="http://127.0.0.1:8710",
        robot_adapter="external-http", robot_type="so100",
        safety_profile=sc["profile"], readiness_report=sc["readiness"],
        approval=sc["approval"], bundle_dir=sc["bundle_dir"],
        robot_endpoint="http://127.0.0.1:8765", operator="K", max_steps=5, fps=2.0,
        attest_real_hardware=True, attest_physical_estop=True, attest_workspace_clear=True)
    with patch("lerobot_coreai.robot_adapters.build_robot_adapter", return_value=_FakeExternal()):
        result = evaluate_real_preflight(cfg)
    assert not result.ok
    failed = {c.name for c in result.checks if not c.passed}
    assert "external_controller_action_shape_matches_profile" in failed
