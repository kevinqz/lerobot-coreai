# test_lerobot_eval_v2.py — eval-v2 report + orchestrator (v1.1.4).

import json
from importlib.resources import files
from unittest.mock import patch

import jsonschema

from lerobot_coreai.lerobot_eval_v2 import (
    EVAL_V2_SCHEMA_VERSION, EvalV2Config, build_eval_v2_report, run_eval_v2,
)
from lerobot_coreai.lerobot_features import build_feature_mapping
from lerobot_coreai.manifest import LeRobotCoreAIManifest
from lerobot_coreai.policy import CoreAIPolicy


_POLICY = {"observation.state": {"shape": [7], "required": True}}


def _fm(passed=True, **kw):
    ds = {"observation.state": [7]} if passed else {}
    return build_feature_mapping(dataset_features=ds, policy_obs_features=_POLICY,
                                 strict=True, **kw)


def test_report_schema_valid():
    report = build_eval_v2_report(
        policy_path="p", dataset_repo_id="lerobot/pusht",
        feature_mapping=_fm(True), frames_evaluated=0, strict=True)
    assert report["schema_version"] == EVAL_V2_SCHEMA_VERSION
    assert report["ok"] is True
    schema = json.loads(files("lerobot_coreai.schemas").joinpath(
        "lerobot-eval-v2-report.schema.json").read_text())
    jsonschema.validate(report, schema)


def test_report_claims_no_task_success():
    report = build_eval_v2_report(
        policy_path="p", dataset_repo_id="d", feature_mapping=_fm(True),
        frames_evaluated=5, strict=True)
    assert report["claims"]["proves_task_success"] is False
    assert report["claims"]["proves_physical_safety"] is False


def test_strict_failure_makes_report_not_ok():
    report = build_eval_v2_report(
        policy_path="p", dataset_repo_id="d", feature_mapping=_fm(False),
        frames_evaluated=0, strict=True)
    assert report["ok"] is False


def test_non_strict_failure_is_info_not_blocking():
    fm = build_feature_mapping(dataset_features={}, policy_obs_features=_POLICY,
                               strict=False)
    report = build_eval_v2_report(
        policy_path="p", dataset_repo_id="d", feature_mapping=fm,
        frames_evaluated=0, strict=False)
    assert report["ok"] is True  # non-strict never blocks


def _fake_policy(valid_manifest_dict):
    m = LeRobotCoreAIManifest.from_dict(valid_manifest_dict)
    p = CoreAIPolicy(m, validate_io=False)
    p.predict_action = lambda batch, **kw: {"action": [[0.0] * 7] * 16, "metadata": {}}  # type: ignore
    return p


def test_run_eval_v2_writes_reports(valid_manifest_dict, tmp_path):
    # Mock the dataset feature loader and the policy load so no network is needed.
    from lerobot_coreai.manifest import LeRobotCoreAIManifest
    m = LeRobotCoreAIManifest.from_dict(valid_manifest_dict)
    obs_keys = list(m.observation_features.keys())
    dataset_features = {k: (list(m.observation_features[k].shape)
                            if m.observation_features[k].shape is not None else None)
                        for k in obs_keys}

    with patch.object(CoreAIPolicy, "from_pretrained",
                      return_value=_fake_policy(valid_manifest_dict)), \
         patch("lerobot_coreai.lerobot_eval_v2._load_dataset_features",
               return_value=(dataset_features, object())):
        report = run_eval_v2(EvalV2Config(
            policy_path="kevinqz/EVO1-SO100-CoreAI", dataset_repo_id="lerobot/pusht",
            strict_features=True, output_dir=tmp_path / "ev2"))

    assert report["ok"] is True
    assert (tmp_path / "ev2" / "lerobot_feature_mapping.json").is_file()
    assert (tmp_path / "ev2" / "lerobot_eval_v2_report.json").is_file()
    assert (tmp_path / "ev2" / "lerobot_eval_v2_report.md").is_file()
