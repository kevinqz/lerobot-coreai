# test_coreai_cert_env.py — the certification environment for the official lerobot-eval
# path (v1.3.27.1, WS1 env half). These run wherever lerobot is installed (the rollout
# CI jobs run `pytest packages/lerobot_policy_coreai_bridge/tests`).

import os

import numpy as np
import pytest

pytest.importorskip("lerobot")

import lerobot_policy_coreai_bridge  # noqa: F401,E402  (registers the env + choice)
from lerobot_policy_coreai_bridge.coreai_cert_env import (  # noqa: E402
    CoreAICertEnvConfig,
)

_CHALLENGE_ENV = "COREAI_OFFICIAL_EVAL_CHALLENGE"


def test_env_config_registered_as_choice():
    from lerobot.envs.configs import EnvConfig
    # the draccus choice must exist so `lerobot-eval --env.type=coreai_cert_env` parses.
    cfg = EnvConfig.get_choice_class("coreai_cert_env")
    assert cfg is CoreAICertEnvConfig
    assert CoreAICertEnvConfig().type == "coreai_cert_env"


def test_create_envs_builds_and_steps():
    cfg = CoreAICertEnvConfig(episode_length=5)
    envs = cfg.create_envs(n_envs=1)
    vec = envs[cfg.type][0]
    obs, _info = vec.reset()
    assert "agent_pos" in obs
    for _ in range(5):                     # steps without error through autoreset
        obs, reward, term, trunc, info = vec.step(vec.action_space.sample())
    assert "agent_pos" in obs
    vec.close()


def test_raw_env_reaches_terminal_success():
    # on the RAW env (no gym.make TimeLimit wrapper, no vector autoreset), the episode
    # reaches a terminal success signal at its own horizon.
    from lerobot_policy_coreai_bridge.coreai_cert_env import _make_env
    env = _make_env(max_episode_steps=3)
    env.reset()
    term = False
    reward = 0.0
    info = {}
    for _ in range(3):
        _obs, reward, term, _trunc, info = env.step(env.action_space.sample())
        if term:
            break
    assert term and reward == 1.0 and info["is_success"] is True


def test_challenge_nonce_round_trips_through_env():
    # the executor's challenge nonce, injected via env var, appears in the env's info —
    # the round-trip that proves the env was actually instantiated by the run.
    prev = os.environ.get(_CHALLENGE_ENV)
    os.environ[_CHALLENGE_ENV] = "nonce-cert-1234"
    try:
        vec = CoreAICertEnvConfig(episode_length=3).create_envs(n_envs=1)["coreai_cert_env"][0]
        _obs, info = vec.reset()
        assert "nonce-cert-1234" in np.asarray(info["coreai_challenge"]).ravel().tolist()
        vec.close()
    finally:
        if prev is None:
            os.environ.pop(_CHALLENGE_ENV, None)
        else:
            os.environ[_CHALLENGE_ENV] = prev


def test_real_lerobot_eval_subprocess_accepts_env_choice():
    # the decisive integration: a REAL `lerobot-eval` subprocess auto-imports the plugin
    # and accepts --env.type=coreai_cert_env (it then fails only on the absent policy,
    # which is the v1.3.27.2 half). No "invalid choice" for the env.
    from lerobot_coreai.official_eval_executor import run_official_eval
    run = run_official_eval(
        ["--env.type=coreai_cert_env", "--eval.n_episodes=1", "--eval.batch_size=1"],
        challenge_nonce="nonce-cert-1234", timeout=300)
    combined = run["stdout"] + run["stderr"]
    assert "lerobot_policy_coreai_bridge" in combined      # plugin auto-imported
    assert "coreai_cert_env" in combined                   # env choice recognized
    assert "not a valid" not in combined.lower()
    assert "invalid choice" not in combined.lower()
