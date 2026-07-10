# test_lerobot_eval_v3.py — real action replay (v1.2.9).

import json
from importlib.resources import files
from unittest.mock import patch

import jsonschema

from lerobot_coreai import lerobot_eval_v3 as ev3
from lerobot_coreai.lerobot_eval_v3 import (
    EVAL_V3_SCHEMA_VERSION, EvalV3Config, build_eval_v3_report, run_eval_v3,
    summarize_eval, validate_action,
)
from lerobot_coreai.manifest import LeRobotCoreAIManifest
from lerobot_coreai.policy import CoreAIPolicy


def test_validate_action_ok():
    ok, _ = validate_action([1.0, 2.0, 3.0], expected_dim=3)
    assert ok is True


def test_validate_action_wrong_dim():
    ok, detail = validate_action([1.0, 2.0], expected_dim=3)
    assert ok is False and "dim" in detail


def test_validate_action_non_finite():
    ok, detail = validate_action([1.0, float("inf")], expected_dim=2)
    assert ok is False


def test_validate_action_not_sequence():
    ok, _ = validate_action(5.0, expected_dim=None)
    assert ok is False


def test_summarize_counts():
    recs = [
        {"action_generated": True, "action_valid": True, "latency_ms": 1.0},
        {"action_generated": True, "action_valid": False, "latency_ms": 2.0},
    ]
    s = summarize_eval(recs)
    assert s["frames_evaluated"] == 2
    assert s["actions_generated"] == 2
    assert s["actions_valid"] == 1
    assert s["failures"] == 1
    assert s["actions_sent_to_robot"] == 0


def _policy(valid_manifest_dict, action_seq):
    m = LeRobotCoreAIManifest.from_dict(valid_manifest_dict)
    p = CoreAIPolicy(m, validate_io=False)
    it = iter(action_seq)
    p.select_next_action = lambda obs, **kw: next(it)  # type: ignore
    p.reset = lambda: None  # type: ignore
    return p


def test_run_eval_v3_replays_frames(valid_manifest_dict, tmp_path):
    # action_dim from valid_manifest is 7; return valid 7-dim actions.
    actions = [[float(i)] * 7 for i in range(3)]
    frames = [{"episode_index": 0}, {"episode_index": 0}, {"episode_index": 1}]
    with patch.object(CoreAIPolicy, "from_pretrained",
                      return_value=_policy(valid_manifest_dict, actions)), \
         patch.object(ev3, "_load_frames", return_value=frames):
        report = run_eval_v3(EvalV3Config(policy_path="p", dataset_repo_id="lerobot/pusht",
                                          output_dir=tmp_path / "out"))
    assert report["ok"] is True
    assert report["summary"]["frames_evaluated"] == 3      # regression: not zero!
    assert report["summary"]["actions_generated"] == 3
    assert report["summary"]["failures"] == 0
    assert (tmp_path / "out" / "eval_v3_report.json").is_file()
    assert (tmp_path / "out" / "eval_v3_trace.jsonl").is_file()


def test_run_eval_v3_invalid_action_fails(valid_manifest_dict):
    # Wrong dim → failure, ok=False.
    actions = [[1.0, 2.0]]  # dim 2 != 7
    frames = [{"episode_index": 0}]
    with patch.object(CoreAIPolicy, "from_pretrained",
                      return_value=_policy(valid_manifest_dict, actions)), \
         patch.object(ev3, "_load_frames", return_value=frames):
        report = run_eval_v3(EvalV3Config(policy_path="p", dataset_repo_id="d"))
    assert report["ok"] is False
    assert report["summary"]["failures"] == 1


def test_report_schema_valid_and_honest(valid_manifest_dict):
    report = build_eval_v3_report(
        EvalV3Config(policy_path="p", dataset_repo_id="d"),
        ok=True, summary=summarize_eval([]), action_contract={})
    assert report["schema_version"] == EVAL_V3_SCHEMA_VERSION
    for k in ("proves_task_success", "proves_physical_safety", "authorizes_robot_actuation"):
        assert report["claims"][k] is False
    schema = json.loads(files("lerobot_coreai.schemas").joinpath(
        "lerobot-eval-v3-report.schema.json").read_text())
    jsonschema.validate(report, schema)


def test_ground_truth_action_not_fed_to_policy(valid_manifest_dict):
    # Label-leakage guard: the frame carries an "action", which must never reach
    # the policy's observation.
    seen = {}

    def _capture(obs, **kw):
        seen.update(obs)
        return [0.0] * 7

    m = LeRobotCoreAIManifest.from_dict(valid_manifest_dict)
    p = CoreAIPolicy(m, validate_io=False)
    p.select_next_action = _capture  # type: ignore
    p.reset = lambda: None  # type: ignore
    obs_key = next(iter(m.observation_features))
    frame = {"episode_index": 0, obs_key: [0.0] * 7, "action": [9.0] * 7,
             "reward": 1.0, "timestamp": 0.1}
    with patch.object(CoreAIPolicy, "from_pretrained", return_value=p), \
         patch.object(ev3, "_load_frames", return_value=[frame]):
        run_eval_v3(EvalV3Config(policy_path="p", dataset_repo_id="d"))
    assert "action" not in seen
    assert "reward" not in seen


def test_first_reset_when_no_episode_index(valid_manifest_dict):
    actions = [[0.0] * 7, [0.0] * 7]
    frames = [{"observation.state": [0.0] * 7}, {"observation.state": [0.0] * 7}]
    p = _policy(valid_manifest_dict, actions)
    resets = {"n": 0}
    p.reset = lambda: resets.__setitem__("n", resets["n"] + 1)  # type: ignore
    with patch.object(CoreAIPolicy, "from_pretrained", return_value=p), \
         patch.object(ev3, "_load_frames", return_value=frames):
        run_eval_v3(EvalV3Config(policy_path="p", dataset_repo_id="d"))
    # No episode_index anywhere → reset must still fire once at the start.
    assert resets["n"] == 1


def test_reset_called_per_episode(valid_manifest_dict):
    actions = [[0.0] * 7, [0.0] * 7, [0.0] * 7]
    frames = [{"episode_index": 0}, {"episode_index": 1}, {"episode_index": 1}]
    p = _policy(valid_manifest_dict, actions)
    resets = {"n": 0}
    p.reset = lambda: resets.__setitem__("n", resets["n"] + 1)  # type: ignore
    with patch.object(CoreAIPolicy, "from_pretrained", return_value=p), \
         patch.object(ev3, "_load_frames", return_value=frames):
        run_eval_v3(EvalV3Config(policy_path="p", dataset_repo_id="d"))
    # Reset at start (ep 0) + at boundary to ep 1 = 2 resets.
    assert resets["n"] == 2
