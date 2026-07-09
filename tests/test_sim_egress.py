# test_sim_egress.py — unit tests for simulator-only egress (v0.8).

import pytest
from unittest.mock import MagicMock

from lerobot_coreai.errors import SafetyError
from lerobot_coreai.sim_egress import SimEgress, SimEgressResult


class TestSimEgress:
    def test_send_to_simulator_calls_env_step(self):
        env = MagicMock()
        env.step.return_value = ({"observation.state": [0.0] * 7}, 1.0, False, {"sim_step": 1})
        egress = SimEgress()

        result, obs, reward, done, info = egress.send_to_simulator(env, [0.0] * 7)

        env.step.assert_called_once_with([0.0] * 7)
        assert isinstance(result, SimEgressResult)
        assert result.sent_to_simulator is True
        assert result.sent_to_robot is False
        assert result.destination == "simulator"
        assert result.action == [0.0] * 7
        assert reward == 1.0
        assert done is False
        assert info == {"sim_step": 1}

    def test_actions_sent_to_simulator_increments(self):
        env = MagicMock()
        env.step.return_value = ({}, 0.0, False, {})
        egress = SimEgress()

        for _ in range(5):
            egress.send_to_simulator(env, [0.0] * 7)

        assert egress.actions_sent_to_simulator == 5

    def test_actions_sent_to_robot_always_zero(self):
        env = MagicMock()
        env.step.return_value = ({}, 0.0, False, {})
        egress = SimEgress()
        for _ in range(100):
            egress.send_to_simulator(env, [0.0] * 7)
        assert egress.actions_sent_to_robot == 0

    def test_send_to_robot_raises_safety_error(self):
        egress = SimEgress()
        with pytest.raises(SafetyError, match="No robot commands"):
            egress.send_to_robot([0.0] * 7)

    def test_send_to_robot_message_mentions_sim_mode(self):
        egress = SimEgress()
        with pytest.raises(SafetyError, match="disabled in sim mode"):
            egress.send_to_robot([0.0] * 7)

    def test_egress_result_is_frozen(self):
        env = MagicMock()
        env.step.return_value = ({}, 0.0, False, {})
        egress = SimEgress()
        result, *_ = egress.send_to_simulator(env, [0.0] * 7)
        with pytest.raises(Exception):
            result.sent_to_robot = True
