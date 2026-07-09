# test_shadow_no_actuation.py — guarantees that shadow mode never sends actions.

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from lerobot_coreai.action_blocker import ActionBlocker
from lerobot_coreai.errors import SafetyError
from lerobot_coreai.shadow import ShadowConfig, run_shadow_mode


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


class TestShadowNoActuation:
    def test_action_blocker_send_always_raises(self):
        """The only egress path (ActionBlocker.send) must always raise."""
        b = ActionBlocker()
        with pytest.raises(SafetyError, match="No robot commands"):
            b.send([0.0] * 7)

    def test_action_blocker_actions_sent_always_zero(self):
        b = ActionBlocker()
        for _ in range(100):
            b.block([0.0] * 7)
        assert b.actions_sent == 0

    def test_shadow_report_actions_sent_zero_even_on_failure(self, tmp_path, valid_manifest_dict):
        """A failure report must still show actions_sent=0 and all safety invariants."""
        fixtures_dir = tmp_path / "fx"
        fixtures_dir.mkdir()
        (fixtures_dir / "000000.json").write_text(json.dumps({"observation.state": [0.0] * 7}))

        mock_policy = _make_mock_policy(valid_manifest_dict)
        # Fail on every predict to trigger error path.
        mock_policy.predict_action.side_effect = RuntimeError("runner down")

        with patch("lerobot_coreai.shadow.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            config = ShadowConfig(
                policy_path="test",
                observation_source="fixtures",
                fixtures_dir=fixtures_dir,
                output_dir=tmp_path / "run",
                max_steps=2,
                fps=0,
                fail_fast=True,
            )
            with pytest.raises(RuntimeError):
                run_shadow_mode(config)

        report_path = tmp_path / "run" / "shadow_report.json"
        assert report_path.exists()
        report = json.loads(report_path.read_text())
        assert report["safety"]["actions_sent"] == 0
        assert report["safety"]["physical_actuation_possible"] is False
        assert report["safety"]["motor_commands_available"] is False
        assert report["safety"]["actuation_device_connected"] is False
        assert report["safety"]["robot_connected"] is False

    def test_no_hardware_imports_in_shadow_module(self):
        """shadow.py must not import any robot hardware modules."""
        shadow_src = Path(__file__).parent.parent / "src" / "lerobot_coreai" / "shadow.py"
        content = shadow_src.read_text().lower()
        forbidden = ["dynamixel", "feetech", "serial.serial", "motor_bus", "teleop",
                     "pypot", "dynamixel_sdk", "write_position", "write_goal_position"]
        for token in forbidden:
            assert token.lower() not in content, f"shadow.py contains forbidden token: {token}"

    def test_no_hardware_imports_in_observation_sources_module(self):
        """observation_sources.py must not import any robot hardware modules."""
        src = Path(__file__).parent.parent / "src" / "lerobot_coreai" / "observation_sources.py"
        content = src.read_text().lower()
        forbidden = ["dynamixel", "feetech", "serial.serial", "motor_bus", "teleop",
                     "pypot", "dynamixel_sdk"]
        for token in forbidden:
            assert token.lower() not in content, f"observation_sources.py contains forbidden token: {token}"

    def test_no_hardware_imports_in_action_blocker_module(self):
        """action_blocker.py must not import any robot hardware modules."""
        src = Path(__file__).parent.parent / "src" / "lerobot_coreai" / "action_blocker.py"
        content = src.read_text().lower()
        forbidden = ["dynamixel", "feetech", "serial.serial", "motor_bus", "teleop",
                     "pypot", "dynamixel_sdk"]
        for token in forbidden:
            assert token.lower() not in content, f"action_blocker.py contains forbidden token: {token}"
