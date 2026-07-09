# test_safety_no_bypass.py — prove blocked actions never reach egress (v0.9.0).
#
# The single most important property of the supervisor: when it blocks an
# action in enforce mode, SimEgress.send_to_simulator must NOT be called for it.

from unittest.mock import MagicMock, patch

from lerobot_coreai.sim import SimConfig, run_sim_mode


def _make_mock_policy(manifest_dict, action):
    from lerobot_coreai.manifest import LeRobotCoreAIManifest
    mock = MagicMock()
    mock.predict_action.return_value = {
        "action": action, "metadata": {"timing": {"total_ms": 5.0}},
    }
    mock.manifest = LeRobotCoreAIManifest.from_dict(manifest_dict)
    mock.policy_type = "evo1"
    mock.robot_type = "so100"
    mock.parity_passed = True
    mock.policy_repo_id = "test/policy"
    return mock


def test_sim_egress_not_called_when_supervisor_blocks(tmp_path, valid_manifest_dict):
    bad = [[float("nan")] * 7] * 16
    mock_policy = _make_mock_policy(valid_manifest_dict, bad)
    config = SimConfig(
        policy_path="test/p", output_dir=tmp_path / "run", env_type="fake",
        confirm_sim_egress=True, episodes=1, max_steps_per_episode=5, fps=0,
        supervisor_mode="enforce", safety_profile_name="default-sim-safe",
    )
    with patch("lerobot_coreai.sim.CoreAIPolicy.from_pretrained", return_value=mock_policy), \
         patch("lerobot_coreai.sim.SimEgress.send_to_simulator") as send:
        result = run_sim_mode(config)
    send.assert_not_called()
    assert result.report["metrics"]["actions_blocked_by_supervisor"] >= 1
    assert result.report["metrics"]["actions_sent_to_simulator"] == 0


def test_valid_action_does_reach_egress(tmp_path, valid_manifest_dict):
    # Control: a valid action IS sent (so the block test above isn't vacuous).
    good = [[0.0] * 7] * 16
    mock_policy = _make_mock_policy(valid_manifest_dict, good)
    config = SimConfig(
        policy_path="test/p", output_dir=tmp_path / "run", env_type="fake",
        confirm_sim_egress=True, episodes=1, max_steps_per_episode=2, fps=0,
        supervisor_mode="enforce",
    )
    real_env_return = (None, {"pixels": [[0]]}, 1.0, True, {})
    with patch("lerobot_coreai.sim.CoreAIPolicy.from_pretrained", return_value=mock_policy), \
         patch("lerobot_coreai.sim.SimEgress.send_to_simulator", return_value=real_env_return) as send:
        run_sim_mode(config)
    assert send.called


def test_robot_egress_invariants_preserved_under_supervisor(tmp_path, valid_manifest_dict):
    good = [[0.0] * 7] * 16
    mock_policy = _make_mock_policy(valid_manifest_dict, good)
    config = SimConfig(
        policy_path="test/p", output_dir=tmp_path / "run", env_type="fake",
        confirm_sim_egress=True, episodes=1, max_steps_per_episode=2, fps=0,
        supervisor_mode="enforce",
    )
    with patch("lerobot_coreai.sim.CoreAIPolicy.from_pretrained", return_value=mock_policy):
        result = run_sim_mode(config)
    safety = result.report["safety"]
    assert safety["robot_egress_enabled"] is False
    assert safety["actions_sent_to_robot"] == 0
    assert safety["action_egress"] == "simulator_only"
    assert safety["physical_actuation_possible"] is False
