"""Golden example: audit a LeRobotDataset against a CoreAI policy (eval-v2).

Builds the dataset<->policy feature mapping with strict checks. Dataset loading
needs the ``[lerobot]`` extra; the feature-mapping logic itself is pure. No
hardware is touched and no robot action is sent.

Usage:
    python examples/lerobot_bridge/dataset_eval_bridge.py POLICY_PATH DATASET_REPO_ID [RUNNER_URL]
"""

from __future__ import annotations

import json
import sys

from lerobot_coreai.lerobot_eval_v2 import EvalV2Config, run_eval_v2


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: dataset_eval_bridge.py POLICY_PATH DATASET_REPO_ID [RUNNER_URL]")
        print("example: ... kevinqz/EVO1-SO100-CoreAI lerobot/pusht")
        return 0
    policy_path, dataset_repo_id = sys.argv[1], sys.argv[2]
    runner_url = sys.argv[3] if len(sys.argv) > 3 else None

    report = run_eval_v2(EvalV2Config(
        policy_path=policy_path, dataset_repo_id=dataset_repo_id,
        runner_url=runner_url, strict_features=True))
    print(json.dumps(report["feature_mapping"], indent=2))
    print("\nProves observation mapping only — not task success or physical safety.")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
