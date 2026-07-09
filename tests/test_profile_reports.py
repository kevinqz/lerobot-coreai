# test_profile_reports.py — profile comparison + fit reports (v0.9.1).

import json
from importlib.resources import files

import jsonschema

from lerobot_coreai.profile_reports import (
    build_comparison_markdown,
    build_profile_fit,
    compare_profiles,
)
from lerobot_coreai.safety_profiles import SafetyProfile


def _write_actions(tmp_path, vals):
    path = tmp_path / "actions.jsonl"
    with open(path, "w") as f:
        for i, v in enumerate(vals):
            f.write(json.dumps({"step": i, "action": [v]}) + "\n")
    return path


def _prof(name, max_abs):
    return SafetyProfile(name=name, profile_type="software_bounds",
                         max_abs_action=max_abs, require_robot_type_match=False,
                         require_known_shape=False)


def test_stricter_profile_blocks_more(tmp_path):
    # actions with abs up to 0.9; profile B (max_abs 0.5) blocks the big ones.
    path = _write_actions(tmp_path, [0.1, 0.3, 0.6, 0.9])
    report = compare_profiles(_prof("a", 1.0), _prof("b", 0.5), path)
    assert report["actions_supervised"] == 4
    assert report["a"]["blocked"] == 0
    assert report["b"]["blocked"] == 2      # 0.6 and 0.9 exceed 0.5
    assert report["breakdown"]["b_only_blocks"] == 2


def test_identical_profiles_full_agreement(tmp_path):
    path = _write_actions(tmp_path, [0.1, 0.2, 0.3])
    report = compare_profiles(_prof("a", 1.0), _prof("b", 1.0), path)
    assert report["agreement_rate"] == 1.0


def test_comparison_report_validates_schema(tmp_path):
    path = _write_actions(tmp_path, [0.1, 0.9])
    report = compare_profiles(_prof("a", 1.0), _prof("b", 0.5), path)
    schema = json.loads(files("lerobot_coreai.schemas").joinpath(
        "profile-comparison-report.schema.json").read_text())
    jsonschema.validate(report, schema)


def test_comparison_markdown_no_overclaim(tmp_path):
    path = _write_actions(tmp_path, [0.1])
    md = build_comparison_markdown(compare_profiles(_prof("a", 1.0), _prof("b", 1.0), path))
    assert "does not prove profile equivalence or physical safety" in md.lower()


def test_profile_fit_rates():
    summary = {
        "profile": "so100-sim-default", "actions_supervised": 100,
        "actions_allowed": 95, "actions_blocked": 5, "actions_modified": 10,
        "would_block_actions": 0,
    }
    fit = build_profile_fit(summary)
    assert fit["fit"]["allowed_rate"] == 0.95
    assert fit["fit"]["blocked_rate"] == 0.05
    assert fit["claims"]["proves_physical_safety"] is False
