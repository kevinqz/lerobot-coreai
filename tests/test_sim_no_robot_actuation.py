# test_sim_no_robot_actuation.py — guarantees that sim mode never sends actions to a robot.

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from lerobot_coreai.errors import CoreAIPolicyError, SafetyError
from lerobot_coreai.sim import SimConfig, run_sim_mode
from lerobot_coreai.sim_egress import SimEgress


def _make_mock_policy(manifest_dict):
    from lerobot_coreai.manifest import LeRobotCoreAIManifest
    mock = MagicMock()
    mock.predict_action.return_value = {"action": [[0.0] * 7] * 16, "metadata": {}}
    mock.manifest = LeRobotCoreAIManifest.from_dict(manifest_dict)
    mock.policy_type = "evo1"
    mock.robot_type = "so100"
    mock.parity_passed = True
    mock.policy_repo_id = "test/policy"
    return mock


# Tokens that must never appear in executable sim source.
SIM_FORBIDDEN_TOKENS = [
    "dynamixel", "feetech", "serial.serial", "motor_bus", "teleop",
    "pypot", "dynamixel_sdk", "write_position", "write_goal_position",
]


class TestSimNoRobotActuation:
    def test_sim_egress_send_to_robot_always_raises(self):
        """The robot egress path must always raise."""
        e = SimEgress()
        with pytest.raises(SafetyError, match="No robot commands"):
            e.send_to_robot([0.0] * 7)

    def test_sim_egress_actions_sent_to_robot_always_zero(self):
        e = SimEgress()
        env = MagicMock()
        env.step.return_value = ({}, 0.0, False, {})
        for _ in range(100):
            e.send_to_simulator(env, [0.0] * 7)
        assert e.actions_sent_to_robot == 0

    def test_sim_report_robot_invariants_even_on_failure(self, tmp_path, valid_manifest_dict):
        """A failure report must still show actions_sent_to_robot=0 and all safety invariants."""
        mock_policy = _make_mock_policy(valid_manifest_dict)
        # Fail on every predict to trigger the error path.
        mock_policy.predict_action.side_effect = RuntimeError("runner down")

        with patch("lerobot_coreai.sim.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            config = SimConfig(
                policy_path="test",
                output_dir=tmp_path / "run",
                env_type="fake",
                confirm_sim_egress=True,
                max_steps_per_episode=2,
                episodes=1,
                fail_fast=True,
            )
            with pytest.raises(RuntimeError):
                run_sim_mode(config)

        report_path = tmp_path / "run" / "sim_report.json"
        assert report_path.exists()
        report = json.loads(report_path.read_text())
        assert report["safety"]["actions_sent_to_robot"] == 0
        assert report["safety"]["robot_egress_enabled"] is False
        assert report["safety"]["physical_actuation_possible"] is False
        assert report["safety"]["motor_commands_available"] is False
        assert report["safety"]["robot_connected"] is False
        assert report["safety"]["simulator_egress_enabled"] is True
        assert report["safety"]["action_egress"] == "simulator_only"

    def test_no_hardware_imports_in_sim_module(self):
        sim_src = Path(__file__).parent.parent / "src" / "lerobot_coreai" / "sim.py"
        content = sim_src.read_text().lower()
        for token in SIM_FORBIDDEN_TOKENS:
            assert token.lower() not in content, f"sim.py contains forbidden token: {token}"

    def test_no_hardware_imports_in_sim_egress_module(self):
        src = Path(__file__).parent.parent / "src" / "lerobot_coreai" / "sim_egress.py"
        content = src.read_text().lower()
        for token in SIM_FORBIDDEN_TOKENS:
            assert token.lower() not in content, f"sim_egress.py contains forbidden token: {token}"

    def test_no_hardware_imports_in_sim_envs_module(self):
        src = Path(__file__).parent.parent / "src" / "lerobot_coreai" / "sim_envs.py"
        content = src.read_text().lower()
        for token in SIM_FORBIDDEN_TOKENS:
            assert token.lower() not in content, f"sim_envs.py contains forbidden token: {token}"
