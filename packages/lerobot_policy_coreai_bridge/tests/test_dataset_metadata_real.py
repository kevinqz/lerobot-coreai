# test_dataset_metadata_real.py — v1.3.25: load a REAL LeRobotDatasetMetadata from a
# deterministic, video-free on-disk dataset built by lerobot's own writer, capture
# DatasetMetadataEvidence, verify it offline, and bind it to a FeatureContract.
# No SimpleNamespace in this path — the real upstream class is loaded. No network.

import os
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("lerobot")
import numpy as np  # noqa: E402

# LeRobotDataset needs the dataset/media extra (`lerobot[dataset]`), which only the
# isolated rollout CI jobs install — the plain lerobot-stable/dev jobs do not. Guard
# the heavy import with a module-level skip (pytest 8 importorskip re-raises a
# mismatched-name ImportError, so use an explicit try/except). COREAI_REQUIRE_ROLLOUT=1
# (set in the rollout jobs) makes it mandatory there.
_REQUIRE = os.environ.get("COREAI_REQUIRE_ROLLOUT") == "1"
try:
    from lerobot.datasets.lerobot_dataset import (  # noqa: E402
        LeRobotDataset, LeRobotDatasetMetadata,
    )
except ImportError as _exc:  # pragma: no cover
    if _REQUIRE:
        raise
    pytest.skip(f"lerobot dataset stack unavailable ({_exc})", allow_module_level=True)

from lerobot_coreai.dataset_metadata_evidence import (  # noqa: E402
    capture_dataset_metadata_evidence, verify_dataset_metadata_evidence,
)
from lerobot_coreai.dataset_metadata_validation import (  # noqa: E402
    bind_metadata_to_feature_contract,
)
from lerobot_coreai.feature_contract import (  # noqa: E402
    FeatureContract, FeatureSpec, NormalizationContract, ValueDomain, make_feature_id,
)

_NAMES = ("j1", "j2", "j3", "j4", "j5", "j6", "grip")


def _build_real_dataset(root):
    """Build a deterministic, video-free LeRobot dataset (state+action) on disk."""
    feats = {
        "observation.state": {"dtype": "float32", "shape": [7], "names": list(_NAMES)},
        "action": {"dtype": "float32", "shape": [7], "names": list(_NAMES)},
    }
    ds = LeRobotDataset.create(
        repo_id="local/coreai-cert-fixture", fps=30, features=feats,
        robot_type="so100-fixture", root=str(root), use_videos=False)
    for _ep in range(2):
        for t in range(6):
            ds.add_frame({"observation.state": np.full(7, float(t), np.float32),
                          "action": np.full(7, float(t), np.float32),
                          "task": "push the T"})
        ds.save_episode()
    ds.finalize()
    return LeRobotDatasetMetadata(repo_id="local/coreai-cert-fixture", root=str(root))


def _contract(action_names=_NAMES):
    state = FeatureSpec(
        feature_id=make_feature_id("observation", "observation.state",
                                   "coreai_runner_input.v1"),
        key="observation.state", role="observation", modality="vector",
        stage="coreai_runner_input.v1", required=True, dtype="float32", shape=("S",),
        axes=("state",), layout=None, value_domain=ValueDomain(), names=_NAMES,
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


def test_real_metadata_loads_and_evidence_verifies(tmp_path):
    root = tmp_path / "ds"
    meta = _build_real_dataset(root)
    # the REAL upstream class, loaded from disk with no network.
    assert type(meta).__name__ == "LeRobotDatasetMetadata"
    assert meta.robot_type == "so100-fixture" and meta.fps == 30
    assert meta.total_episodes == 2 and meta.total_tasks == 1

    ev = capture_dataset_metadata_evidence(
        meta, root=str(root), repo_id="local/coreai-cert-fixture", revision="fixture-v1",
        root_kind="local_fixture")
    assert ev["claims"]["dataset_metadata_verified"] is True
    assert "meta/info.json" in ev["files"]
    assert ev["metadata_tree_sha256"].startswith("sha256:")
    ok, errs = verify_dataset_metadata_evidence(ev, str(root))
    assert ok, errs


def test_real_metadata_tamper_fails_offline(tmp_path):
    root = tmp_path / "ds"
    meta = _build_real_dataset(root)
    ev = capture_dataset_metadata_evidence(
        meta, root=str(root), repo_id="local/coreai-cert-fixture", revision="v1")
    (root / "meta" / "info.json").write_text('{"tampered": true}')
    ok, _ = verify_dataset_metadata_evidence(ev, str(root))
    assert not ok


def test_real_metadata_binds_feature_contract(tmp_path):
    root = tmp_path / "ds"
    meta = _build_real_dataset(root)
    ev = capture_dataset_metadata_evidence(
        meta, root=str(root), repo_id="local/coreai-cert-fixture", revision="v1")
    r = bind_metadata_to_feature_contract(ev, _contract())
    assert r.ok, r.failures
    # a contract with mismatched action names must fail the binding.
    bad = bind_metadata_to_feature_contract(ev, _contract(action_names=("x", "y")))
    assert not bad.ok


# --- v1.3.26.3: static, byte-exact, multimodal, cross-version fixture ---

# committed under tests/fixtures/lerobot_dataset_v3_multimodal (generated with 0.6.0).
_FIXTURE_ROOT = (Path(__file__).resolve().parents[3]
                 / "tests" / "fixtures" / "lerobot_dataset_v3_multimodal")
_PINNED_TREE_SHA256 = "sha256:07cefa4af2846794960732b4e9033bcef60baf501cdf35eda2365076066a2937"


def test_static_multimodal_fixture_certificate_grade():
    # BOTH stable (0.6.0) and dev CI read the SAME committed bytes through the REAL
    # LeRobotDatasetMetadata and must produce the SAME semantic evidence + tree hash.
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata as MetaCls
    from lerobot_coreai.dataset_metadata_evidence import (
        verify_dataset_metadata_evidence,
    )
    meta = MetaCls(repo_id="local/coreai-cert-fixture-mm", root=str(_FIXTURE_ROOT))
    assert set(meta.camera_keys) == {"observation.images.front",
                                     "observation.images.wrist"}
    ev = capture_dataset_metadata_evidence(
        meta, root=str(_FIXTURE_ROOT), repo_id="local/coreai-cert-fixture-mm",
        revision="fixture-v1", evidence_grade="certificate")
    # byte-exact cross-version pin: identical bytes -> identical root on every version.
    assert ev["metadata_tree_sha256"] == _PINNED_TREE_SHA256
    assert ev["evidence_grade"] == "certificate"
    assert ev["loader_identity"]["class_name"] == "LeRobotDatasetMetadata"
    ok, errs = verify_dataset_metadata_evidence(ev, str(_FIXTURE_ROOT))
    assert ok, errs
