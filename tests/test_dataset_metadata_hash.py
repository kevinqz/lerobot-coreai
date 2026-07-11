# test_dataset_metadata_hash.py — canonical metadata-tree hash + evidence + binding
# (v1.3.25), pure base (no lerobot). Uses a hand-built on-disk meta/ tree.

import json
import os
from types import SimpleNamespace

import pytest

from lerobot_coreai.dataset_metadata_evidence import (
    capture_dataset_metadata_evidence, verify_dataset_metadata_evidence,
)
from lerobot_coreai.dataset_metadata_hash import (
    collect_metadata_files, compute_metadata_tree, metadata_tree_sha256,
)


def _build_tree(root, *, info_extra=""):
    meta = root / "meta"
    (meta / "episodes" / "chunk-000").mkdir(parents=True)
    (meta / "info.json").write_text(json.dumps({
        "codebase_version": "v3.0", "robot_type": "so100-fixture", "fps": 30,
        "features": {"observation.state": {"dtype": "float32", "shape": [7]}},
        "extra": info_extra}))
    (meta / "stats.json").write_text(json.dumps({"observation.state": {"mean": [0.0]}}))
    (meta / "tasks.parquet").write_bytes(b"PAR1-fake-tasks")
    (meta / "episodes" / "chunk-000" / "file-000.parquet").write_bytes(b"PAR1-fake-eps")
    return root


def _meta_obj():
    return SimpleNamespace(
        robot_type="so100-fixture", fps=30,
        features={"observation.state": {"dtype": "float32", "shape": [7],
                                        "names": ["j1", "j2", "j3", "j4", "j5", "j6", "g"]},
                  "action": {"dtype": "float32", "shape": [7],
                             "names": ["j1", "j2", "j3", "j4", "j5", "j6", "g"]}},
        camera_keys=[], names={"action": ["j1", "j2", "j3", "j4", "j5", "j6", "g"]},
        shapes={"observation.state": [7], "action": [7]},
        total_tasks=1, total_episodes=2, _version="v3.0")


def test_tree_hash_deterministic(tmp_path):
    _build_tree(tmp_path)
    _, a = compute_metadata_tree(tmp_path)
    _, b = compute_metadata_tree(tmp_path)
    assert a == b and a.startswith("sha256:")


def test_tree_hash_mtime_independent(tmp_path):
    _build_tree(tmp_path)
    _, a = compute_metadata_tree(tmp_path)
    os.utime(tmp_path / "meta" / "info.json", (0, 0))     # change mtime, not content
    _, b = compute_metadata_tree(tmp_path)
    assert a == b


def test_content_change_changes_root(tmp_path):
    _build_tree(tmp_path)
    _, a = compute_metadata_tree(tmp_path)
    (tmp_path / "meta" / "stats.json").write_text(json.dumps({"changed": True}))
    _, b = compute_metadata_tree(tmp_path)
    assert a != b


def test_missing_info_fails(tmp_path):
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "stats.json").write_text("{}")
    with pytest.raises(FileNotFoundError):
        collect_metadata_files(tmp_path)


def test_symlink_rejected(tmp_path):
    _build_tree(tmp_path)
    target = tmp_path / "outside.json"; target.write_text("{}")
    link = tmp_path / "meta" / "tasks.parquet"
    link.unlink(); link.symlink_to(target)
    with pytest.raises(ValueError):
        collect_metadata_files(tmp_path)


def test_capture_and_verify_evidence(tmp_path):
    _build_tree(tmp_path)
    ev = capture_dataset_metadata_evidence(
        _meta_obj(), root=str(tmp_path), repo_id="local/fixture", revision="fixture-v1")
    assert ev["claims"]["dataset_metadata_verified"] is True
    assert ev["claims"]["dataset_content_verified"] is False
    ok, errs = verify_dataset_metadata_evidence(ev, str(tmp_path))
    assert ok, errs


def test_evidence_tamper_detected(tmp_path):
    _build_tree(tmp_path)
    ev = capture_dataset_metadata_evidence(
        _meta_obj(), root=str(tmp_path), repo_id="local/fixture", revision="fixture-v1")
    (tmp_path / "meta" / "info.json").write_text(json.dumps({"tampered": True}))
    ok, errs = verify_dataset_metadata_evidence(ev, str(tmp_path))
    assert not ok


def test_certificate_grade_rejects_duck_typed_loader(tmp_path):
    # a SimpleNamespace stand-in is fine for diagnostic, refused for certificate grade.
    _build_tree(tmp_path)
    with pytest.raises(ValueError):
        capture_dataset_metadata_evidence(
            _meta_obj(), root=str(tmp_path), repo_id="local/fixture", revision="v1",
            evidence_grade="certificate")


def test_certificate_grade_requires_official_loader_identity(tmp_path):
    _build_tree(tmp_path)
    ev = capture_dataset_metadata_evidence(
        _meta_obj(), root=str(tmp_path), repo_id="local/fixture", revision="v1")
    # forge a certificate grade with a bogus loader -> verifier rejects.
    ev["evidence_grade"] = "certificate"
    ev["loader_identity"] = {"module": "impostor", "class_name": "Fake",
                             "lerobot_version": "0.6.0"}
    ok, errs = verify_dataset_metadata_evidence(ev, str(tmp_path))
    assert not ok and any("official LeRobotDatasetMetadata" in e for e in errs)


def test_hub_snapshot_without_commit_not_certificate(tmp_path):
    _build_tree(tmp_path)
    ev = capture_dataset_metadata_evidence(
        _meta_obj(), root=str(tmp_path), repo_id="hf/x", revision="main",
        root_kind="hub_snapshot")
    ok, errs = verify_dataset_metadata_evidence(ev, str(tmp_path))
    assert not ok and any("resolved_commit" in e for e in errs)


# --- binding to FeatureContract ---

def _contract(action_names=("j1", "j2", "j3", "j4", "j5", "j6", "g")):
    from lerobot_coreai.feature_contract import (
        FeatureContract, FeatureSpec, NormalizationContract, ValueDomain,
        make_feature_id,
    )
    state = FeatureSpec(
        feature_id=make_feature_id("observation", "observation.state",
                                   "coreai_runner_input.v1"),
        key="observation.state", role="observation", modality="vector",
        stage="coreai_runner_input.v1", required=True, dtype="float32",
        shape=("S",), axes=("state",), layout=None, value_domain=ValueDomain(),
        names=("j1", "j2", "j3", "j4", "j5", "j6", "g"),
        normalization=NormalizationContract("normalized", None, "policy_preprocessor"))
    action = FeatureSpec(
        feature_id=make_feature_id("action", "action", "coreai_runner_output.v1"),
        key="action", role="action", modality="vector",
        stage="coreai_runner_output.v1", required=True, dtype="float32",
        shape=("H", "A"), axes=("horizon", "action"), layout=None,
        value_domain=ValueDomain(), names=action_names,
        normalization=NormalizationContract("normalized", None, "coreai_model"))
    return FeatureContract(contract_id="c", robot_type="so100-fixture",
                           policy_path="k/E", observations=(state,), actions=(action,))


def test_binding_passes(tmp_path):
    from lerobot_coreai.dataset_metadata_validation import (
        bind_metadata_to_feature_contract,
    )
    _build_tree(tmp_path)
    ev = capture_dataset_metadata_evidence(
        _meta_obj(), root=str(tmp_path), repo_id="local/fixture", revision="v1")
    r = bind_metadata_to_feature_contract(ev, _contract())
    assert r.ok, r.failures


def test_binding_action_names_mismatch_fails(tmp_path):
    from lerobot_coreai.dataset_metadata_validation import (
        bind_metadata_to_feature_contract,
    )
    _build_tree(tmp_path)
    ev = capture_dataset_metadata_evidence(
        _meta_obj(), root=str(tmp_path), repo_id="local/fixture", revision="v1")
    r = bind_metadata_to_feature_contract(ev, _contract(action_names=("x", "y")))
    assert not r.ok and any("names" in f for f in r.failures)


def test_binding_robot_type_mismatch_fails(tmp_path):
    from lerobot_coreai.dataset_metadata_validation import (
        bind_metadata_to_feature_contract,
    )
    _build_tree(tmp_path)
    ev = capture_dataset_metadata_evidence(
        _meta_obj(), root=str(tmp_path), repo_id="local/fixture", revision="v1")
    ev["robot_type"] = "aloha"
    r = bind_metadata_to_feature_contract(ev, _contract(), robot_type_compatible=False)
    assert not r.ok and any("robot_type" in f for f in r.failures)
