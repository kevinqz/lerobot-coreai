# test_cli_compare_v2.py — compare-v2 CLI end-to-end with mocked stages (v1.2.6).

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
    return sl.SourcePolicyBundle(policy=object(), preprocessor=object(),
                                 postprocessor=object(), dataset_metadata=object(),
                                 config=object())


def test_cli_compare_v2_rc0_matching(tmp_path):
    with patch.object(sl, "load_source_policy", return_value=_fake_bundle()), \
         patch.object(cv2, "_load_frames", return_value=[{"i": 0}, {"i": 1}]), \
         patch.object(cv2, "_source_final_action", side_effect=[[1.0, 2.0], [3.0, 4.0]]), \
         patch.object(cv2, "_coreai_final_action", side_effect=[[1.0, 2.0], [3.0, 4.0]]), \
         patch.object(CoreAIPolicy, "from_pretrained", return_value=_fake_coreai()):
        rc = cli.main(["compare-v2", "--torch.policy.path", "t",
                       "--coreai.policy.path", "c", "--dataset.repo_id", "lerobot/pusht",
                       "--strict-processors", "--output-dir", str(tmp_path / "out"),
                       "--json"])
    assert rc == 0
    report = json.loads((tmp_path / "out" / "compare_v2_report.json").read_text())
    assert report["ok"] is True
    assert report["metrics"]["mae"] == 0.0
    assert (tmp_path / "out" / "source_policy_load_report.json").is_file()
    assert (tmp_path / "out" / "processor_contract_report.json").is_file()


def test_cli_compare_v2_rc1_shape_mismatch(tmp_path):
    with patch.object(sl, "load_source_policy", return_value=_fake_bundle()), \
         patch.object(cv2, "_load_frames", return_value=[{"i": 0}]), \
         patch.object(cv2, "_source_final_action", side_effect=[[1.0, 2.0]]), \
         patch.object(cv2, "_coreai_final_action", side_effect=[[1.0, 2.0, 3.0]]), \
         patch.object(CoreAIPolicy, "from_pretrained", return_value=_fake_coreai()):
        rc = cli.main(["compare-v2", "--torch.policy.path", "t",
                       "--coreai.policy.path", "c", "--dataset.repo_id", "d", "--json"])
    assert rc == 1


def test_cli_compare_v2_rc1_ambiguous_processors_strict():
    with patch.object(CoreAIPolicy, "from_pretrained",
                      return_value=_fake_coreai(processor_contract=False)), \
         patch.object(sl, "load_source_policy", return_value=_fake_bundle()):
        rc = cli.main(["compare-v2", "--torch.policy.path", "t",
                       "--coreai.policy.path", "c", "--dataset.repo_id", "d",
                       "--strict-processors"])
    assert rc == 1
