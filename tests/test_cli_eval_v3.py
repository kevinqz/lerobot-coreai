# test_cli_eval_v3.py — eval-v3 CLI (v1.2.9).

import json
from unittest.mock import patch

from lerobot_coreai import cli
from lerobot_coreai import lerobot_eval_v3 as ev3
from lerobot_coreai.manifest import LeRobotCoreAIManifest
from lerobot_coreai.policy import CoreAIPolicy


def _policy(valid_manifest_dict, actions):
    m = LeRobotCoreAIManifest.from_dict(valid_manifest_dict)
    p = CoreAIPolicy(m, validate_io=False)
    it = iter(actions)
    p.select_next_action = lambda obs, **kw: next(it)  # type: ignore
    p.reset = lambda: None  # type: ignore
    return p


def test_cli_eval_v3_rc0(valid_manifest_dict, tmp_path):
    actions = [[0.0] * 7, [0.0] * 7]
    frames = [{"episode_index": 0}, {"episode_index": 0}]
    with patch.object(CoreAIPolicy, "from_pretrained",
                      return_value=_policy(valid_manifest_dict, actions)), \
         patch.object(ev3, "_load_frames", return_value=frames):
        rc = cli.main(["eval-v3", "--policy.path", "p", "--dataset.repo_id", "d",
                       "--output-dir", str(tmp_path / "o"), "--json"])
    assert rc == 0
    report = json.loads((tmp_path / "o" / "eval_v3_report.json").read_text())
    assert report["summary"]["frames_evaluated"] == 2


def test_cli_eval_v3_rc1_on_failure(valid_manifest_dict):
    actions = [[1.0]]  # wrong dim
    frames = [{"episode_index": 0}]
    with patch.object(CoreAIPolicy, "from_pretrained",
                      return_value=_policy(valid_manifest_dict, actions)), \
         patch.object(ev3, "_load_frames", return_value=frames):
        rc = cli.main(["eval-v3", "--policy.path", "p", "--dataset.repo_id", "d", "--json"])
    assert rc == 1
