# test_obs_pipeline_report.py — obs bridge report writers (v1.1.5).

import json

from lerobot_coreai.manifest import LeRobotCoreAIManifest
from lerobot_coreai.obs_bridge import evaluate_obs_bridge
from lerobot_coreai.obs_pipeline_report import (
    build_obs_bridge_markdown, write_obs_bridge_report,
)
from lerobot_coreai.observation_adapters import ObservationAdapterConfig


def _report(valid_manifest_dict):
    m = LeRobotCoreAIManifest.from_dict(valid_manifest_dict)
    sk = next((n for n in m.observation_features if "state" in n),
              next(iter(m.observation_features)))
    raw = {}
    for name, spec in m.observation_features.items():
        raw[name] = "img" if "image" in name else (
            [0.0] * int(spec.shape[-1]) if spec.shape is not None else 0.0)
    if "task" in m.observation_features:
        raw["task"] = "t"
    return evaluate_obs_bridge(raw, ObservationAdapterConfig(state_key=sk), manifest=m)


def test_markdown_renders(valid_manifest_dict):
    md = build_obs_bridge_markdown(_report(valid_manifest_dict))
    assert "Observation Pipeline Bridge Check" in md
    assert "not task success" in md


def test_write_report_files(valid_manifest_dict, tmp_path):
    write_obs_bridge_report(tmp_path / "obs", _report(valid_manifest_dict))
    assert (tmp_path / "obs" / "obs_bridge_report.json").is_file()
    assert (tmp_path / "obs" / "obs_bridge_report.md").is_file()
    data = json.loads((tmp_path / "obs" / "obs_bridge_report.json").read_text())
    assert data["claims"]["proves_physical_safety"] is False
