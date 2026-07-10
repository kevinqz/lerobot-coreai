# test_cli_hf_metadata.py — CLI for hf-metadata + example scripts run offline (v1.1.6).

import json
import runpy
import sys
from pathlib import Path

from lerobot_coreai import cli

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "lerobot_bridge"


def test_cli_hf_metadata_writes_and_validates(tmp_path):
    rc = cli.main(["hf-metadata", "--policy.path", "kevinqz/EVO1-SO100-CoreAI",
                   "--robot.type", "so100", "--output-dir", str(tmp_path / "meta")])
    assert rc == 0
    m = json.loads((tmp_path / "meta" / "lerobot_coreai_metadata.json").read_text())
    assert m["bridge"]["native_registry"] is False
    assert m["safety"]["physical_safety_proof"] is False
    assert (tmp_path / "meta" / "lerobot_coreai_metadata.md").is_file()


def test_cli_hf_metadata_json_rc0():
    assert cli.main(["hf-metadata", "--json"]) == 0


def test_example_scripts_are_importable_and_print_usage(capsys, monkeypatch):
    # The example scripts must run without hardware. With no args they print
    # usage and exit 0 (no network, no runner needed).
    for name in ("select_action_bridge.py", "dataset_eval_bridge.py"):
        monkeypatch.setattr(sys, "argv", [str(EXAMPLES / name)])
        try:
            runpy.run_path(str(EXAMPLES / name), run_name="__main__")
        except SystemExit as e:
            assert e.code == 0
