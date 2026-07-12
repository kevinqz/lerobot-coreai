# coreai_cert_env.py — a minimal, dependency-light certification environment for the
# official lerobot-eval path (v1.3.27.1, WS1 env half).
#
# The certification run drives the OFFICIAL `lerobot-eval` CLI. That needs a real,
# registered environment lerobot can build via `EnvConfig.create_envs()` → `gym.make()`.
# This is that environment: a self-contained gymnasium env (no mujoco / no simulator
# download) with a deterministic episode, plus the `EnvConfig` subclass registered as
# the draccus choice "coreai_cert_env". Because lerobot auto-imports installed
# `lerobot_policy_*` plugins at startup, importing this plugin registers the choice, so
# `lerobot-eval --env.type=coreai_cert_env` resolves it in a real subprocess.
#
# It is intentionally NOT a physics benchmark — it exists to exercise the official
# eval/rollout machinery end-to-end (env build → reset → step → success signal), not to
# claim task competence. The executor's challenge nonce is echoed into `info` so a real
# run can prove the env was actually instantiated (the nonce round-trip).

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

_STATE_DIM = 6
_ACTION_DIM = 6
_GYM_ID = "coreai_cert/CoreAICert-v0"
_CHALLENGE_ENV = "COREAI_OFFICIAL_EVAL_CHALLENGE"


def _make_env(**kwargs):
    import gymnasium as gym
    from gymnasium import spaces

    class CoreAICertEnv(gym.Env):
        metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

        def __init__(self, max_episode_steps: int = 10, render_mode: str | None = None,
                     **_):
            self._max = int(max_episode_steps)
            self._t = 0
            self.render_mode = render_mode
            self.action_space = spaces.Box(-1.0, 1.0, (_ACTION_DIM,), dtype=np.float32)
            self.observation_space = spaces.Dict({
                "agent_pos": spaces.Box(-1.0, 1.0, (_STATE_DIM,), dtype=np.float32)})

        def render(self):
            # a deterministic tiny RGB frame so the official eval's video/render path
            # works (this env is a machinery exerciser, not a visual benchmark).
            shade = int(255 * self._t / max(self._max, 1))
            return np.full((16, 16, 3), shade, dtype=np.uint8)

        def _obs(self):
            # deterministic ramp so a run is reproducible (no RNG dependence).
            v = np.full((_STATE_DIM,), self._t / max(self._max, 1), dtype=np.float32)
            return {"agent_pos": v}

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self._t = 0
            info = {"coreai_challenge": os.environ.get(_CHALLENGE_ENV, "")}
            return self._obs(), info

        def step(self, action):
            self._t += 1
            terminated = self._t >= self._max
            reward = 1.0 if terminated else 0.0
            info = {"is_success": bool(terminated),
                    "coreai_challenge": os.environ.get(_CHALLENGE_ENV, "")}
            return self._obs(), reward, terminated, False, info

    return CoreAICertEnv(**kwargs)


def _register_gym_env() -> None:
    import gymnasium as gym
    if _GYM_ID not in gym.registry:
        gym.register(id=_GYM_ID, entry_point=_make_env, disable_env_checker=True)


_register_gym_env()


def _register_env_config():
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.envs.configs import ACTION, OBS_STATE, EnvConfig

    @EnvConfig.register_subclass("coreai_cert_env")
    @dataclass
    class CoreAICertEnvConfig(EnvConfig):
        task: str | None = "CoreAICert-v0"
        fps: int = 10
        episode_length: int = 10
        features: dict = field(default_factory=lambda: {
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(_ACTION_DIM,)),
            "agent_pos": PolicyFeature(type=FeatureType.STATE, shape=(_STATE_DIM,))})
        features_map: dict = field(default_factory=lambda: {
            ACTION: ACTION, "agent_pos": OBS_STATE})

        @property
        def gym_id(self) -> str:
            return _GYM_ID

        @property
        def package_name(self) -> str:
            return "lerobot_policy_coreai_bridge.coreai_cert_env"

        @property
        def gym_kwargs(self) -> dict:
            return {"max_episode_steps": self.episode_length,
                    "render_mode": "rgb_array"}

    return CoreAICertEnvConfig


CoreAICertEnvConfig = _register_env_config()
