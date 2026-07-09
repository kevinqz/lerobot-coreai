# test_sim_envs.py — unit tests for simulator environments (v0.8).

import json
import pytest
from pathlib import Path

from lerobot_coreai.errors import CoreAIPolicyError
from lerobot_coreai.sim_envs import (
    FakeSimEnvironment,
    ReplaySimEnvironment,
    SimEnvConfig,
    build_sim_environment,
)


class TestFakeSimEnvironment:
    def test_reset_returns_observation(self):
        env = FakeSimEnvironment(max_steps=3, action_size=7)
        obs = env.reset()
        assert "observation.state" in obs
        assert len(obs["observation.state"]) == 7
        assert "task" in obs

    def test_reset_with_state_vector(self):
        env = FakeSimEnvironment(max_steps=3, state_vector=[1.0, 2.0, 3.0])
        obs = env.reset()
        assert obs["observation.state"] == [1.0, 2.0, 3.0]

    def test_step_returns_obs_reward_done_info(self):
        env = FakeSimEnvironment(max_steps=3, action_size=7)
        env.reset()
        obs, reward, done, info = env.step([0.0] * 7)
        assert "observation.state" in obs
        assert reward == 1.0
        assert isinstance(done, bool)
        assert "sim_step" in info
        assert "success" in info

    def test_done_after_max_steps(self):
        env = FakeSimEnvironment(max_steps=2, action_size=7)
        env.reset()
        _, _, done1, _ = env.step([0.0] * 7)
        assert done1 is False
        _, _, done2, info = env.step([0.0] * 7)
        assert done2 is True
        assert info["success"] is True

    def test_reset_resets_step_index(self):
        env = FakeSimEnvironment(max_steps=2, action_size=7)
        env.reset()
        env.step([0.0] * 7)
        env.reset()
        assert env.step_index == 0

    def test_close_is_noop(self):
        env = FakeSimEnvironment()
        env.close()  # should not raise


class TestReplaySimEnvironment:
    def _make_obs_dir(self, tmp_path, n=3):
        d = tmp_path / "replay_obs"
        d.mkdir()
        for i in range(n):
            (d / f"{i:06d}.json").write_text(json.dumps({
                "observation.state": [float(i)] * 7,
            }))
        return d

    def test_loads_observations(self, tmp_path):
        obs_dir = self._make_obs_dir(tmp_path, n=3)
        env = ReplaySimEnvironment(observations_dir=obs_dir)
        assert len(env.observations) == 3

    def test_reset_returns_first_obs(self, tmp_path):
        obs_dir = self._make_obs_dir(tmp_path, n=3)
        env = ReplaySimEnvironment(observations_dir=obs_dir)
        obs = env.reset()
        assert obs["observation.state"] == [0.0] * 7

    def test_step_advances_deterministically(self, tmp_path):
        obs_dir = self._make_obs_dir(tmp_path, n=3)
        env = ReplaySimEnvironment(
            observations_dir=obs_dir, reward_per_step=0.5, success_on_last_step=True,
        )
        env.reset()
        # 3 observations: reset→obs[0], step→obs[1], step→obs[2], step→exhausted.
        obs1, r1, d1, _ = env.step([0.0] * 7)
        assert obs1["observation.state"] == [1.0] * 7
        assert r1 == 0.5
        assert d1 is False
        obs2, r2, d2, _ = env.step([0.0] * 7)
        assert obs2["observation.state"] == [2.0] * 7
        assert d2 is False
        obs3, r3, d3, info3 = env.step([0.0] * 7)
        assert d3 is True
        assert info3["success"] is True
        assert r3 == 0.5

    def test_empty_dir_raises(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        with pytest.raises(CoreAIPolicyError, match="no observation"):
            ReplaySimEnvironment(observations_dir=d)

    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(CoreAIPolicyError, match="does not exist"):
            ReplaySimEnvironment(observations_dir=tmp_path / "nope")


class TestBuildSimEnvironment:
    def test_build_fake(self):
        env = build_sim_environment(SimEnvConfig(
            env_type="fake", max_steps=5, task="test",
        ))
        assert isinstance(env, FakeSimEnvironment)
        obs = env.reset()
        assert obs["task"] == "test"

    def test_build_replay(self, tmp_path):
        obs_dir = tmp_path / "obs"
        obs_dir.mkdir()
        (obs_dir / "000000.json").write_text(json.dumps({"observation.state": [0.0] * 7}))
        cfg_path = tmp_path / "replay.json"
        cfg_path.write_text(json.dumps({
            "observations_dir": str(obs_dir),
            "reward_per_step": 1.0,
        }))
        env = build_sim_environment(SimEnvConfig(env_type="replay", env_config=cfg_path))
        assert isinstance(env, ReplaySimEnvironment)

    def test_replay_without_config_raises(self):
        with pytest.raises(CoreAIPolicyError, match="requires --env.config"):
            build_sim_environment(SimEnvConfig(env_type="replay"))

    def test_reserved_env_types_raise(self):
        for reserved in ("gym", "lerobot", "pusht"):
            with pytest.raises(CoreAIPolicyError, match="not yet supported"):
                build_sim_environment(SimEnvConfig(env_type=reserved))

    def test_unknown_env_type_raises(self):
        with pytest.raises(CoreAIPolicyError, match="not a known simulator"):
            build_sim_environment(SimEnvConfig(env_type="nonexistent"))
