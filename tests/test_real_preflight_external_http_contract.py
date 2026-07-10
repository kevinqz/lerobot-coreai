# test_real_preflight_external_http_contract.py — end-to-end external-http
# capability + safety-state gating through real preflight (v1.1.1).

import json
from unittest.mock import patch

from lerobot_coreai.real_preflight import RealPreflightConfig, evaluate_real_preflight


def _pf(**over):
    base = {
        "ok": True, "controller_schema_version": "lerobot-coreai.external_http.v0",
        "adapter": "external-http", "robot_type": "so100",
        "action_shape": [16, 7], "observation_keys": ["observation.state", "task"],
        "supports_stop": True, "supports_ready": True, "supports_observation": True,
        "supports_safety_state": True, "physical_estop_required": True, "max_fps": 5.0,
    }
    base.update(over)
    return base


def _ss(**over):
    base = {
        "ok": True, "ready": True, "robot_type": "so100",
        "controller_connected": True, "physical_estop_state": "armed",
        "workspace_state": "clear", "motors_powered": True, "faults": [],
    }
    base.update(over)
    return base


def _fake(pf=None, ss=None):
    class _FakeExternal:
        name = "external-http"
        robot_type = "so100"
        def preflight(self):
            return pf if pf is not None else _pf()
        def safety_state(self):
            return ss if ss is not None else _ss()
    return _FakeExternal()


def _cfg(sc, **over):
    base = dict(
        mode="guarded", policy_path="kevinqz/EVO1-SO100-CoreAI",
        runner_url="http://127.0.0.1:8710", robot_adapter="external-http",
        robot_type="so100", safety_profile=sc["profile"], readiness_report=sc["readiness"],
        approval=sc["approval"], bundle_dir=sc["bundle_dir"],
        robot_endpoint="http://127.0.0.1:8765", operator="K", max_steps=5, fps=2.0,
        attest_real_hardware=True, attest_physical_estop=True, attest_workspace_clear=True,
        has_observation_config=True)
    base.update(over)
    return RealPreflightConfig(**base)


def _run(sc, fake, **over):
    with patch("lerobot_coreai.robot_adapters.build_robot_adapter", return_value=fake):
        return evaluate_real_preflight(_cfg(sc, **over))


def _failed(result):
    return {c.name for c in result.checks if not c.passed}


def test_valid_controller_and_safety_state_pass(real_ready_scenario):
    sc = real_ready_scenario()
    result = _run(sc, _fake())
    f = _failed(result)
    # No external-controller check should be among failures.
    assert not {n for n in f if n.startswith("external_controller_")}


def test_missing_attestations_skips_contacting_controller(real_ready_scenario):
    sc = real_ready_scenario()

    contacted = {"preflight": False}

    class _Spy:
        name = "external-http"
        robot_type = "so100"
        def preflight(self):
            contacted["preflight"] = True
            return _pf()
        def safety_state(self):
            return _ss()

    result = _run(sc, _Spy(), attest_real_hardware=False)
    assert contacted["preflight"] is False
    assert not result.ok


def test_estop_triggered_blocks(real_ready_scenario):
    sc = real_ready_scenario()
    result = _run(sc, _fake(ss=_ss(physical_estop_state="triggered")))
    assert not result.ok
    assert "external_controller_estop_armed" in _failed(result)


def test_workspace_unknown_blocks(real_ready_scenario):
    sc = real_ready_scenario()
    result = _run(sc, _fake(ss=_ss(workspace_state="unknown")))
    assert not result.ok
    assert "external_controller_workspace_clear" in _failed(result)


def test_faults_nonempty_blocks(real_ready_scenario):
    sc = real_ready_scenario()
    result = _run(sc, _fake(ss=_ss(faults=["gripper_fault"])))
    assert not result.ok
    assert "external_controller_faults_empty" in _failed(result)


def test_max_fps_too_low_blocks(real_ready_scenario):
    sc = real_ready_scenario()
    result = _run(sc, _fake(pf=_pf(max_fps=1.0)), fps=5.0)
    assert not result.ok
    assert "external_controller_max_fps_allows_requested_fps" in _failed(result)


def test_report_has_external_http_section_and_no_raw_token(real_ready_scenario):
    sc = real_ready_scenario()
    result = _run(sc, _fake(), robot_token="super-secret-token")
    section = result.report.get("external_http")
    assert section is not None
    assert section["loopback_only"] is True
    assert section["capabilities"]["controller_schema_version"] == \
        "lerobot-coreai.external_http.v0"
    assert section["safety_state"]["physical_estop_state"] == "armed"
    # token appears only as a sha256 prefix, never raw.
    assert section["auth"]["token_sha256_prefix"].startswith("sha256:")
    assert "super-secret-token" not in json.dumps(result.report)
