# test_cli_compare_v2.py — compare-v2 CLI end-to-end with mocked stages (v1.2.7).

import json
from unittest.mock import MagicMock, patch

from lerobot_coreai import cli
from lerobot_coreai import compare_v2 as cv2
from lerobot_coreai import source_loader_v2 as sl
from lerobot_coreai.policy import CoreAIPolicy


def _fake_coreai(processor_contract=True):
    m = MagicMock()
    manifest = {}
    if processor_contract:
        manifest["processor_contract"] = {
            "observation_input": {"expects": "raw_lerobot_observation"},
            "action_output": {"returns": "postprocessed_action"}}
    m.manifest = manifest
    return m


def _fake_bundle():
    # pretrained_path_bound must be True for a valid compare.
    return sl.SourcePolicyBundle(
        policy=object(), preprocessor=object(), postprocessor=object(),
        dataset_metadata=object(), config=object(), pretrained_path_bound=True,
        policy_revision=None, policy_class="FakePolicy")


def _base_args(tmp_path, extra):
    return (["compare-v2", "--torch.policy.path", "t", "--coreai.policy.path", "c",
             "--dataset.repo_id", "lerobot/pusht"] + extra)


def test_cli_rc0_ran_but_parity_not_proven_without_gates(tmp_path):
    with patch.object(sl, "load_source_policy", return_value=_fake_bundle()), \
         patch.object(cv2, "_load_frames", return_value=[{"i": 0}, {"i": 1}]), \
         patch.object(cv2, "_source_action", side_effect=[[1.0, 2.0], [3.0, 4.0]]), \
         patch.object(cv2, "_coreai_action", side_effect=[[1.0, 2.0], [3.0, 4.0]]), \
         patch.object(CoreAIPolicy, "from_pretrained", return_value=_fake_coreai()):
        rc = cli.main(_base_args(tmp_path, ["--strict-processors",
                      "--output-dir", str(tmp_path / "out"), "--json"]))
    assert rc == 0
    report = json.loads((tmp_path / "out" / "compare_v2_report.json").read_text())
    assert report["ok"] is True
    # No tolerances passed → parity is NOT proven even with a perfect match.
    assert report["claims"]["proves_action_parity_on_final_unit"] is False


def test_cli_parity_proven_with_gates(tmp_path):
    with patch.object(sl, "load_source_policy", return_value=_fake_bundle()), \
         patch.object(cv2, "_load_frames", return_value=[{"i": 0}]), \
         patch.object(cv2, "_source_action", side_effect=[[1.0, 2.0]]), \
         patch.object(cv2, "_coreai_action", side_effect=[[1.0, 2.0]]), \
         patch.object(CoreAIPolicy, "from_pretrained", return_value=_fake_coreai()):
        rc = cli.main(_base_args(tmp_path, [
            "--strict-processors", "--tolerance.mean-mae", "1e-6",
            "--tolerance.min-cosine", "0.999", "--json"]))
    assert rc == 0


def test_cli_huge_error_fails_gates(tmp_path):
    with patch.object(sl, "load_source_policy", return_value=_fake_bundle()), \
         patch.object(cv2, "_load_frames", return_value=[{"i": 0}]), \
         patch.object(cv2, "_source_action", side_effect=[[0.0, 0.0]]), \
         patch.object(cv2, "_coreai_action", side_effect=[[1000.0, 1000.0]]), \
         patch.object(CoreAIPolicy, "from_pretrained", return_value=_fake_coreai()):
        rc = cli.main(_base_args(tmp_path, [
            "--strict-processors", "--tolerance.mean-mae", "1e-6", "--json"]))
    assert rc == 1  # gate fails → not ok


def test_cli_rc1_structural_shape_mismatch(tmp_path):
    with patch.object(sl, "load_source_policy", return_value=_fake_bundle()), \
         patch.object(cv2, "_load_frames", return_value=[{"i": 0}]), \
         patch.object(cv2, "_source_action", side_effect=[[[1.0, 2.0]]]), \
         patch.object(cv2, "_coreai_action", side_effect=[[1.0, 2.0]]), \
         patch.object(CoreAIPolicy, "from_pretrained", return_value=_fake_coreai()):
        rc = cli.main(_base_args(tmp_path, ["--strict-processors", "--json"]))
    assert rc == 1


def test_cli_rc1_ambiguous_processors_strict(tmp_path):
    with patch.object(CoreAIPolicy, "from_pretrained",
                      return_value=_fake_coreai(processor_contract=False)), \
         patch.object(sl, "load_source_policy", return_value=_fake_bundle()):
        rc = cli.main(_base_args(tmp_path, ["--strict-processors"]))
    assert rc == 1


def test_cli_rc1_unbound_weights(tmp_path):
    unbound = sl.SourcePolicyBundle(
        policy=object(), preprocessor=object(), postprocessor=object(),
        dataset_metadata=object(), config=object(), pretrained_path_bound=False)
    with patch.object(sl, "load_source_policy", return_value=unbound), \
         patch.object(cv2, "_load_frames", return_value=[{"i": 0}]), \
         patch.object(cv2, "_source_action", side_effect=[[1.0]]), \
         patch.object(cv2, "_coreai_action", side_effect=[[1.0]]), \
         patch.object(CoreAIPolicy, "from_pretrained", return_value=_fake_coreai()):
        rc = cli.main(_base_args(tmp_path, ["--strict-processors",
                      "--tolerance.mean-mae", "1e-6", "--json"]))
    assert rc == 1  # weights not bound → cannot be a valid compare
