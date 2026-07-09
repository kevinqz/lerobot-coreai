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
        # 3 observations: reset→obs[0], step 1→obs[1] done=False, step 2→obs[2] done=True.
        obs1, r1, d1, _ = env.step([0.0] * 7)
        assert obs1["observation.state"] == [1.0] * 7
        assert r1 == 0.5
        assert d1 is False
        obs2, r2, d2, info2 = env.step([0.0] * 7)
        assert obs2["observation.state"] == [2.0] * 7
        assert d2 is True
        assert info2["success"] is True
        assert r2 == 0.5

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
        # gym is supported as of v0.8.1; lerobot/pusht remain reserved.
        for reserved in ("lerobot", "pusht"):
            with pytest.raises(CoreAIPolicyError, match="not yet supported"):
                build_sim_environment(SimEnvConfig(env_type=reserved))

    def test_unknown_env_type_raises(self):
        with pytest.raises(CoreAIPolicyError, match="not a known simulator"):
            build_sim_environment(SimEnvConfig(env_type="nonexistent"))


# MARK: - Gym/gymnasium adapter (v0.8.1)

import sys

from lerobot_coreai.sim_envs import (
    GymSimEnvironment,
    normalize_gym_observation,
)


class FakeGymEnv:
    """Minimal fake gymnasium env producing deterministic reset/step output."""

    def __init__(self, env_id, **kwargs):
        self.env_id = env_id
        self.kwargs = kwargs
        self.closed = False
        self._step = 0

    def reset(self, *, seed=None):
        self._step = 0
        return [0.0, 0.0, 0.0], {"reset_seed": seed}

    def step(self, action):
        self._step += 1
        terminated = self._step >= 3
        return [float(self._step)] * 3, 0.5, terminated, False, {"sim_step": self._step}

    def close(self):
        self.closed = True


class FakeGymnasium:
    """Minimal fake gymnasium module: make() returns a FakeGymEnv."""

    def __init__(self):
        self.made = []

    def make(self, env_id, **kwargs):
        env = FakeGymEnv(env_id, **kwargs)
        self.made.append(env)
        return env


@pytest.fixture
def fake_gymnasium(monkeypatch):
    """Inject a fake gymnasium module into sys.modules."""
    gym = FakeGymnasium()
    monkeypatch.setitem(sys.modules, "gymnasium", gym)
    return gym


class TestNormalizeGymObservation:
    def test_dict_passthrough(self):
        obs = {"observation.state": [1.0, 2.0], "task": "x"}
        out = normalize_gym_observation(obs)
        assert out == {"observation.state": [1.0, 2.0], "task": "x"}
        # Must be a copy, not the same object.
        assert out is not obs

    def test_list_to_state(self):
        out = normalize_gym_observation([1.0, 2.0, 3.0])
        assert out == {"observation.state": [1.0, 2.0, 3.0]}

    def test_tuple_to_state(self):
        out = normalize_gym_observation((1, 2))
        assert out == {"observation.state": [1, 2]}

    def test_scalar_to_state(self):
        out = normalize_gym_observation(3.5)
        assert out == {"observation.state": [3.5]}

    def test_numpy_like_to_state(self):
        class ArrayLike:
            def tolist(self):
                return [1.0, 2.0]

        out = normalize_gym_observation(ArrayLike())
        assert out == {"observation.state": [1.0, 2.0]}

    def test_unsupported_raises(self):
        with pytest.raises(CoreAIPolicyError, match="could not be normalized"):
            normalize_gym_observation("a string")


class TestGymSimEnvironment:
    def test_reset_returns_normalized_dict(self, fake_gymnasium):
        env = GymSimEnvironment(env_id="FakeEnv-v0")
        obs = env.reset(seed=42)
        assert isinstance(obs, dict)
        assert obs["observation.state"] == [0.0, 0.0, 0.0]

    def test_step_collapses_terminated_truncated_to_done(self, fake_gymnasium):
        env = GymSimEnvironment(env_id="FakeEnv-v0")
        env.reset()
        obs, reward, done, info = env.step([0.1])
        assert isinstance(obs, dict)
        assert reward == 0.5
        assert done is False
        assert info["terminated"] is False
        assert info["truncated"] is False

    def test_step_done_when_terminated(self, fake_gymnasium):
        env = GymSimEnvironment(env_id="FakeEnv-v0")
        env.reset()
        env.step([0.1])
        env.step([0.1])
        _, _, done, info = env.step([0.1])
        assert done is True
        assert info["terminated"] is True

    def test_close_called(self, fake_gymnasium):
        env = GymSimEnvironment(env_id="FakeEnv-v0")
        env.close()
        assert env._env.closed is True

    def test_task_and_state_vector_injected(self, fake_gymnasium):
        env = GymSimEnvironment(
            env_id="FakeEnv-v0", task="pick", state_vector=[1.0, 2.0],
        )
        obs = env.reset()
        assert obs["task"] == "pick"
        assert obs["observation.state"] == [0.0, 0.0, 0.0]

    def test_kwargs_passed_to_make(self, fake_gymnasium):
        env = GymSimEnvironment(env_id="FakeEnv-v0", env_kwargs={"max_steps": 5})
        assert env._env.kwargs == {"max_steps": 5}

    def test_missing_gymnasium_raises_install_hint(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "gymnasium", None)
        with pytest.raises(CoreAIPolicyError, match="lerobot-coreai\\[sim\\]"):
            GymSimEnvironment(env_id="FakeEnv-v0")


class TestBuildGymEnvironment:
    def test_build_gym_returns_gym_env(self, fake_gymnasium):
        env = build_sim_environment(SimEnvConfig(
            env_type="gym", env_id="FakeEnv-v0",
        ))
        assert isinstance(env, GymSimEnvironment)
        assert env.env_id == "FakeEnv-v0"

    def test_build_gym_without_env_id_raises(self, fake_gymnasium):
        with pytest.raises(CoreAIPolicyError, match="requires --env.id"):
            build_sim_environment(SimEnvConfig(env_type="gym"))

    def test_build_gym_passes_kwargs(self, fake_gymnasium):
        env = build_sim_environment(SimEnvConfig(
            env_type="gym", env_id="FakeEnv-v0", env_kwargs={"foo": 1},
        ))
        assert env._env.kwargs == {"foo": 1}
