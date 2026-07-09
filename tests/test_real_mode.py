# test_real_mode.py — guarded real mode orchestration + no-bypass (v1.0.0).

import json
from importlib.resources import files
from unittest.mock import MagicMock, patch

import jsonschema

from lerobot_coreai.real_mode import RealModeConfig, run_real_mode


def _mock_policy(valid_manifest_dict, action=None):
    from lerobot_coreai.manifest import LeRobotCoreAIManifest
    m = MagicMock()
    m.predict_action.return_value = {"action": action if action is not None else [[0.0] * 7] * 16,
                                     "metadata": {"timing": {"total_ms": 5.0}}}
    m.manifest = LeRobotCoreAIManifest.from_dict(valid_manifest_dict)
    m.policy_type = "evo1"
    m.robot_type = "so100"
    m.policy_repo_id = "test/policy"
    return m


def _cfg(sc, tmp_path, mode="guarded", **over):
    base = dict(
        mode=mode, policy_path="test/p", runner_url="http://127.0.0.1:8710",
        robot_adapter="mock", robot_type=sc["robot_type"],
        safety_profile=sc["profile"], readiness_report=sc["readiness"],
        approval=sc["approval"], bundle_dir=sc["bundle_dir"],
        output_dir=tmp_path / "real_out", operator="Kevin Saltarelli",
        max_steps=5, fps=10.0, attest_real_hardware=True,
        attest_physical_estop=True, attest_workspace_clear=True,
    )
    base.update(over)
    return RealModeConfig(**base)


def _real_schema():
    return json.loads(files("lerobot_coreai.schemas").joinpath(
        "real-report.schema.json").read_text())


class TestPreflightMode:
    def test_preflight_sends_zero_actions(self, real_ready_scenario, tmp_path, valid_manifest_dict):
        sc = real_ready_scenario()
        # In preflight mode the policy/adapter loop never runs.
        with patch("lerobot_coreai.real_mode.build_robot_adapter") as mk:
            result = run_real_mode(_cfg(sc, tmp_path, mode="preflight"))
        assert result.ok
        assert result.actions_sent_to_robot == 0
        assert result.report["egress"]["actions_sent_to_robot"] == 0
        assert result.report["egress"]["robot_egress_enabled"] is False
        mk.assert_not_called()  # no adapter built in preflight
        assert (tmp_path / "real_out" / "real_preflight_report.json").is_file()


class TestGuardedMode:
    def test_happy_path_sends_actions_to_mock(self, real_ready_scenario, tmp_path,
                                              valid_manifest_dict):
        sc = real_ready_scenario()
        with patch("lerobot_coreai.real_mode.CoreAIPolicy.from_pretrained",
                   return_value=_mock_policy(valid_manifest_dict)):
            result = run_real_mode(_cfg(sc, tmp_path))
        assert result.ok
        assert result.stopped_reason == "max_steps_reached"
        assert result.actions_sent_to_robot == 5
        eg = result.report["egress"]
        assert eg["action_egress"] == "guarded_real"
        assert eg["robot_egress_enabled"] is True
        assert eg["robot_egress_path_enabled"] is True
        assert eg["actions_sent_to_robot"] == 5
        # Report + session + trace written.
        out = tmp_path / "real_out"
        assert (out / "real_report.json").is_file()
        assert (out / "real_session.json").is_file()
        assert (out / "real_trace.jsonl").is_file()
        jsonschema.validate(result.report, _real_schema())

    def test_report_claims_never_overclaim(self, real_ready_scenario, tmp_path,
                                           valid_manifest_dict):
        sc = real_ready_scenario()
        with patch("lerobot_coreai.real_mode.CoreAIPolicy.from_pretrained",
                   return_value=_mock_policy(valid_manifest_dict)):
            result = run_real_mode(_cfg(sc, tmp_path))
        claims = result.report["claims"]
        assert claims["proves_physical_safety"] is False
        assert claims["proves_real_world_safety"] is False
        assert claims["authorizes_unrestricted_real_world_actuation"] is False
        assert claims["proves_guarded_real_session_executed"] is True

    def test_refuses_when_readiness_not_ready(self, real_ready_scenario, tmp_path,
                                              valid_manifest_dict):
        sc = real_ready_scenario(ready=False)
        with patch("lerobot_coreai.real_mode.CoreAIPolicy.from_pretrained",
                   return_value=_mock_policy(valid_manifest_dict)) as mk:
            result = run_real_mode(_cfg(sc, tmp_path))
        assert not result.ok
        assert result.actions_sent_to_robot == 0
        assert "preflight_failed" in result.stopped_reason
        mk.assert_not_called()  # policy never loaded — gated out before the loop

    def test_refuses_missing_attestation(self, real_ready_scenario, tmp_path,
                                         valid_manifest_dict):
        sc = real_ready_scenario()
        result = run_real_mode(_cfg(sc, tmp_path, attest_physical_estop=False))
        assert not result.ok
        assert result.actions_sent_to_robot == 0

    def test_duration_seconds_stops_the_session(self, real_ready_scenario, tmp_path,
                                                valid_manifest_dict):
        # A tiny duration bound stops the session before max_steps.
        sc = real_ready_scenario()
        with patch("lerobot_coreai.real_mode.CoreAIPolicy.from_pretrained",
                   return_value=_mock_policy(valid_manifest_dict)):
            result = run_real_mode(_cfg(sc, tmp_path, max_steps=100000,
                                        duration_seconds=0.01))
        assert result.ok
        assert result.stopped_reason == "duration_seconds_reached"
        assert result.report["limits"]["duration_seconds"] == 0.01

    def test_deadman_cannot_be_disabled_for_non_mock(self, real_ready_scenario, tmp_path):
        sc = real_ready_scenario()
        # external-http adapter with deadman disabled → refused before any egress.
        result = run_real_mode(_cfg(
            sc, tmp_path, robot_adapter="external-http",
            robot_endpoint="http://127.0.0.1:9",
            deadman_disable_for_mock_only=True))
        assert not result.ok
        assert result.actions_sent_to_robot == 0


class TestNoBypass:
    def test_blocked_action_never_reaches_adapter(self, real_ready_scenario, tmp_path,
                                                  valid_manifest_dict):
        # NaN action → supervisor blocks → session stops, zero actions sent.
        sc = real_ready_scenario()
        bad = [[float("nan")] * 7] * 16
        captured = {}
        real_build = None
        from lerobot_coreai import robot_adapters

        def _spy_build(name, robot_type, **kw):
            a = robot_adapters.MockRobotAdapter(robot_type=robot_type)
            captured["adapter"] = a
            return a

        with patch("lerobot_coreai.real_mode.CoreAIPolicy.from_pretrained",
                   return_value=_mock_policy(valid_manifest_dict, action=bad)), \
             patch("lerobot_coreai.real_mode.build_robot_adapter", side_effect=_spy_build):
            result = run_real_mode(_cfg(sc, tmp_path))
        assert not result.ok
        assert result.stopped_reason == "supervisor_blocked"
        assert result.actions_sent_to_robot == 0
        assert captured["adapter"].actions_sent == []  # never reached the adapter

    def test_exception_triggers_stop_and_disconnect(self, real_ready_scenario, tmp_path,
                                                    valid_manifest_dict):
        sc = real_ready_scenario()
        from lerobot_coreai import robot_adapters
        adapter = robot_adapters.MockRobotAdapter(robot_type="so100")
        adapter.stop = MagicMock(wraps=adapter.stop)
        adapter.disconnect = MagicMock(wraps=adapter.disconnect)

        policy = _mock_policy(valid_manifest_dict)
        policy.predict_action.side_effect = RuntimeError("runner exploded")
        with patch("lerobot_coreai.real_mode.CoreAIPolicy.from_pretrained", return_value=policy), \
             patch("lerobot_coreai.real_mode.build_robot_adapter", return_value=adapter):
            result = run_real_mode(_cfg(sc, tmp_path))
        assert not result.ok
        adapter.stop.assert_called()
        adapter.disconnect.assert_called()
        assert result.actions_sent_to_robot == 0
