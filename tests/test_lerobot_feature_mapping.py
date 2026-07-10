# test_lerobot_feature_mapping.py — pure dataset↔policy feature mapping (v1.1.4).

from lerobot_coreai.lerobot_features import (
    FEATURE_MAPPING_SCHEMA_VERSION, build_feature_mapping, manifest_obs_features,
)
from lerobot_coreai.manifest import LeRobotCoreAIManifest


_POLICY = {
    "observation.state": {"shape": [7], "required": True},
    "task": {"shape": None, "required": True},
}


def test_all_present_passes_strict():
    fm = build_feature_mapping(
        dataset_features={"observation.state": [7], "task": None},
        policy_obs_features=_POLICY, strict=True)
    assert fm["passed"] is True
    assert fm["schema_version"] == FEATURE_MAPPING_SCHEMA_VERSION


def test_missing_required_fails_strict():
    fm = build_feature_mapping(
        dataset_features={"observation.state": [7]},  # no task
        policy_obs_features=_POLICY, strict=True)
    assert fm["passed"] is False
    assert any("task" in p for p in fm["problems"])


def test_task_from_config_satisfies_requirement():
    fm = build_feature_mapping(
        dataset_features={"observation.state": [7]},
        policy_obs_features=_POLICY, task_in_config=True, strict=True)
    assert fm["passed"] is True
    assert fm["features"]["task"]["provided_by_config"] is True


def test_shape_mismatch_fails_strict():
    fm = build_feature_mapping(
        dataset_features={"observation.state": [9], "task": None},
        policy_obs_features=_POLICY, strict=True)
    assert fm["passed"] is False
    assert any("shape mismatch" in p for p in fm["problems"])
    assert fm["features"]["observation.state"]["shape_compatible"] is False


def test_non_strict_downgrades_problems_to_warnings():
    fm = build_feature_mapping(
        dataset_features={"observation.state": [9]},  # mismatch + missing task
        policy_obs_features=_POLICY, strict=False)
    assert fm["passed"] is True  # non-strict never fails
    assert fm["problems"] == []
    assert fm["warnings"]  # issues surfaced as warnings


def test_unknown_feature_warns_but_can_fail_with_flag():
    ds = {"observation.state": [7], "task": None, "observation.extra": [3]}
    warn = build_feature_mapping(dataset_features=ds, policy_obs_features=_POLICY,
                                 strict=True)
    assert warn["passed"] is True
    assert "observation.extra" in warn["unknown_dataset_features"]
    fail = build_feature_mapping(dataset_features=ds, policy_obs_features=_POLICY,
                                 strict=True, fail_on_unknown=True)
    assert fail["passed"] is False


def test_manifest_obs_features(valid_manifest_dict):
    m = LeRobotCoreAIManifest.from_dict(valid_manifest_dict)
    feats = manifest_obs_features(m)
    assert feats  # non-empty
    for spec in feats.values():
        assert "shape" in spec and "required" in spec
