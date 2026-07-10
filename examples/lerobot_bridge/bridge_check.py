"""Golden example: check the local LeRobot bridge for a CoreAI policy.

Runs the same checks as `lerobot-coreai lerobot-bridge-check`, in-process. Safe
to run without hardware. Without a reachable runner it still verifies the bridge
shape and honest claims; the policy-load check reflects your environment.

Usage:
    python examples/lerobot_bridge/bridge_check.py [POLICY_PATH] [RUNNER_URL]
"""

from __future__ import annotations

import json
import sys

from lerobot_coreai.lerobot_bridge import evaluate_bridge_check


def main() -> int:
    policy_path = sys.argv[1] if len(sys.argv) > 1 else "kevinqz/EVO1-SO100-CoreAI"
    runner_url = sys.argv[2] if len(sys.argv) > 2 else None
    report = evaluate_bridge_check(policy_path, runner_url=runner_url)
    print(json.dumps(report, indent=2))
    print("\nLocal bridge only — not upstream-native LeRobot integration.")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
