# sim_egress.py — simulator-only action egress for sim mode (v0.8).
#
# Sim mode sends actions to a SimEnvironment. This module makes that contract
# explicit: SimEgress.send_to_simulator() forwards an action to a simulator's
# step(); SimEgress.send_to_robot() is the egress that must never fire — it
# always raises SafetyError.
#
# No object here may forward an action to a robot, motor, serial device, or
# actuator. The only permitted destination is a SimEnvironment.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .errors import SafetyError
from .sim_envs import SimEnvironment


@dataclass(frozen=True)
class SimEgressResult:
    """The outcome of egressing an action.

    Invariants:
        - sent_to_simulator is True after send_to_simulator.
        - sent_to_robot is always False (send_to_robot raises).
        - destination is always "simulator".
        - The action value is preserved for logging/audit.
    """

    sent_to_simulator: bool
    sent_to_robot: bool
    destination: str
    action: Any


@dataclass
class SimEgress:
    """Forwards actions to a simulator and blocks all robot egress.

    The sim loop calls send_to_simulator() for every generated action, which
    advances the environment. send_to_robot() is the robot egress path and it
    unconditionally raises SafetyError — it exists so that *if* anyone ever
    wires up a robot, the call fails loudly.

    Invariants:
        - actions_sent_to_simulator increments on every send_to_simulator().
        - actions_sent_to_robot is always 0 (send_to_robot never returns).
    """

    destination: str = "simulator"
    actions_sent_to_simulator: int = 0

    def send_to_simulator(
        self,
        env: SimEnvironment,
        action: Any,
    ) -> tuple[SimEgressResult, dict[str, Any], float, bool, dict[str, Any]]:
        """Forward an action to the simulator's step().

        Returns:
            (result, observation, reward, done, info)
        """
        obs, reward, done, info = env.step(action)
        self.actions_sent_to_simulator += 1
        result = SimEgressResult(
            sent_to_simulator=True,
            sent_to_robot=False,
            destination=self.destination,
            action=action,
        )
        return result, obs, reward, done, info

    def send_to_robot(self, action: Any) -> None:
        """Robot egress path — always disabled in sim mode.

        Raises:
            SafetyError: Always. No robot commands were sent.
        """
        raise SafetyError(
            "Robot egress is disabled in sim mode. No robot commands were sent."
        )

    @property
    def actions_sent_to_robot(self) -> int:
        """Always 0. Sim mode never sends to a robot."""
        return 0
