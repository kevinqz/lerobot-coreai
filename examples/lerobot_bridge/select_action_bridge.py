"""Golden example: use a CoreAI policy where LeRobot expects a policy object.

Loads a CoreAI policy through the local LeRobot bridge and calls
``select_action(batch)`` — the same shape LeRobot uses. Requires a reachable
coreai-runner to actually run inference; without one, this prints how to wire it.
No hardware is touched.

Usage:
    python examples/lerobot_bridge/select_action_bridge.py POLICY_PATH RUNNER_URL
"""

from __future__ import annotations

import sys

from lerobot_coreai.lerobot_bridge import load_coreai_policy_for_lerobot


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: select_action_bridge.py POLICY_PATH RUNNER_URL")
        print("example: ... kevinqz/EVO1-SO100-CoreAI http://127.0.0.1:8710")
        return 0
    policy_path, runner_url = sys.argv[1], sys.argv[2]

    # A LeRobot-shaped, runtime-only bridge. Train with LeRobot, run with CoreAI.
    policy = load_coreai_policy_for_lerobot(policy_path, runner_url=runner_url)
    policy.eval()  # already inference-only; returns self

    # Build a batch the way LeRobot would (values depend on your policy manifest).
    batch = {"observation.state": [0.0] * 7, "task": "example task"}
    action = policy.select_action(batch)  # raw action (LeRobot 0.6.x semantics)
    print("action:", action)

    # policy.train(True) would raise — training is LeRobot's job.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
