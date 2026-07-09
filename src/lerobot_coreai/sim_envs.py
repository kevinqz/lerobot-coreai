# sim_envs.py — simulator environments for sim mode (v0.8).
#
# Sim mode sends actions to a SimEnvironment. The environment may advance its
# internal state and return observations/reward/done. A SimEnvironment is a
# *simulator* — never a robot, motor, serial device, or actuator.
#
# v0.8.0 ships two in-process environments:
#   - fake:  a deterministic stub for testing the loop end-to-end
#   - replay: replays a stored observation sequence deterministically
#
# Real simulator adapters (gym/lerobot/pusht) are reserved for v0.8.1.

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .errors import CoreAIPolicyError


@runtime_checkable
class SimEnvironment(Protocol):
    """A simulator environment that accepts actions and yields observations.

    reset() returns the first observation dict. step() applies an action and
    returns (observation, reward, done, info). close() releases resources.
    """

    def reset(self, *, seed: int | None = None) -> dict[str, Any]: ...
    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, dict[str, Any]]: ...
    def close(self) -> None: ...


@dataclass
class SimEnvConfig:
    """Configuration for building a SimEnvironment."""

    env_type: str
    env_config: Path | None = None
    seed: int | None = None
    render: bool = False
    record_video: bool = False
    video_dir: Path | None = None
    task: str | None = None
    state_vector: list[float] | None = None
    max_steps: int = 300


@dataclass
class FakeSimEnvironment:
    """A deterministic stub simulator for testing the sim loop.

    reset() returns a fixed observation. step() advances an internal counter and
    reports done after max_steps. reward is constant. This is the default
    environment for smoke-testing sim mode without a real simulator.
    """

    max_steps: int = 10
    action_size: int = 7
    task: str | None = None
    state_vector: list[float] | None = None
    step_index: int = 0

    def reset(self, *, seed: int | None = None) -> dict[str, Any]:
        self.step_index = 0
        return {
            "observation.state": list(self.state_vector) if self.state_vector else [0.0] * self.action_size,
            "task": self.task or "fake sim task",
        }

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        self.step_index += 1
        done = self.step_index >= self.max_steps
        obs: dict[str, Any] = {
            "observation.state": [float(self.step_index)] * self.action_size,
            "task": self.task or "fake sim task",
        }
        reward = 1.0
        info: dict[str, Any] = {
            "sim_step": self.step_index,
            "success": done,
        }
        return obs, reward, done, info

    def close(self) -> None:
        pass


@dataclass
class ReplaySimEnvironment:
    """A deterministic environment that replays a stored observation sequence.

    Observations are loaded from a directory of ordered JSON files (NNNNNN.json
    or lexicographic order). Each step advances to the next stored observation
    and returns a deterministic reward/done based on the replay config.
    """

    observations_dir: Path
    reward_per_step: float = 0.0
    success_on_last_step: bool = True
    task: str | None = None
    state_vector: list[float] | None = None
    observations: list[dict[str, Any]] = field(default_factory=list)
    step_index: int = 0

    def __post_init__(self) -> None:
        self.observations = self._load_observations(self.observations_dir)
        if not self.observations:
            raise CoreAIPolicyError(
                f"ReplaySimEnvironment: no observation JSONs found in {self.observations_dir}"
            )
        self.step_index = 0

    @staticmethod
    def _load_observations(observations_dir: Path) -> list[dict[str, Any]]:
        """Load ordered observation JSONs from a directory."""
        observations_dir = Path(observations_dir)
        if not observations_dir.is_dir():
            raise CoreAIPolicyError(
                f"ReplaySimEnvironment: observations dir does not exist: {observations_dir}"
            )
        # Prefer NNNNNN.json ordering, fall back to lexicographic.
        jsons = sorted(observations_dir.glob("*.json"))
        obs_list: list[dict[str, Any]] = []
        for jf in jsons:
            try:
                obs_list.append(json.loads(jf.read_text()))
            except Exception as e:
                raise CoreAIPolicyError(
                    f"ReplaySimEnvironment: failed to read {jf}: {e}"
                ) from e
        return obs_list

    def reset(self, *, seed: int | None = None) -> dict[str, Any]:
        self.step_index = 0
        obs = dict(self.observations[0])
        if self.task:
            obs["task"] = self.task
        if self.state_vector:
            obs.setdefault("observation.state", list(self.state_vector))
        return obs

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        # Advance to the next observation. With N observations:
        #   reset → obs[0]
        #   step 1 → obs[1], done=False
        #   ...
        #   step N-1 → obs[N-1] (last), done=True
        # No extra terminal step is required — the step that returns the last
        # observation is itself terminal.
        self.step_index += 1
        last_step = self.step_index >= len(self.observations) - 1
        idx = min(self.step_index, len(self.observations) - 1)
        obs = dict(self.observations[idx])
        if self.task:
            obs["task"] = self.task
        if self.state_vector:
            obs.setdefault("observation.state", list(self.state_vector))
        done = last_step
        reward = self.reward_per_step
        info: dict[str, Any] = {
            "sim_step": self.step_index,
            "success": self.success_on_last_step and last_step,
        }
        return obs, reward, done, info

    def close(self) -> None:
        pass


def build_sim_environment(config: SimEnvConfig) -> SimEnvironment:
    """Build a SimEnvironment from config.

    v0.8.0 supports: fake, replay.
    Reserved for v0.8.1: gym, lerobot, pusht.
    """
    env_type = config.env_type

    if env_type == "fake":
        return FakeSimEnvironment(
            max_steps=config.max_steps,
            task=config.task,
            state_vector=config.state_vector,
        )

    if env_type == "replay":
        if config.env_config is None:
            raise CoreAIPolicyError(
                "ReplaySimEnvironment requires --env.config pointing to a replay config JSON."
            )
        cfg_path = Path(config.env_config)
        if not cfg_path.is_file():
            raise CoreAIPolicyError(
                f"ReplaySimEnvironment config not found: {cfg_path}"
            )
        try:
            cfg = json.loads(cfg_path.read_text())
        except Exception as e:
            raise CoreAIPolicyError(
                f"ReplaySimEnvironment: invalid config JSON {cfg_path}: {e}"
            ) from e
        observations_dir = cfg.get("observations_dir")
        if not observations_dir:
            raise CoreAIPolicyError(
                "ReplaySimEnvironment config must include 'observations_dir'."
            )
        return ReplaySimEnvironment(
            observations_dir=Path(observations_dir),
            reward_per_step=float(cfg.get("reward_per_step", 0.0)),
            success_on_last_step=bool(cfg.get("success_on_last_step", True)),
            task=config.task,
            state_vector=config.state_vector,
        )

    # Reserved names — real simulator adapters land in v0.8.1.
    if env_type in ("gym", "lerobot", "pusht"):
        raise CoreAIPolicyError(
            f"env.type='{env_type}' is not yet supported in lerobot-coreai v0.8.0. "
            f"Real simulator adapters are planned for v0.8.1. "
            f"Use --env.type fake or --env.type replay for now."
        )

    raise CoreAIPolicyError(
        f"env.type='{env_type}' is not a known simulator environment. "
        f"Supported in v0.8.0: fake, replay."
    )
