# test_compare_v2.py — compare-v2 metrics + report (v1.2.6).

import json
from importlib.resources import files

import jsonschema

from lerobot_coreai.compare_v2 import (
    COMPARE_V2_SCHEMA_VERSION, CompareV2Config, build_compare_v2_report,
    compute_compare_metrics,
)


def test_identical_actions_perfect_parity():
    src = [[1.0, 2.0], [3.0, 4.0]]
    m = compute_compare_metrics(src, [list(x) for x in src])
    assert m["shape_match"] is True and m["finite"] is True
    assert m["mae"] == 0.0
    assert m["max_abs_error"] == 0.0
    assert m["cosine_similarity"] == 1.0


def test_small_difference_metrics():
    m = compute_compare_metrics([[1.0, 1.0]], [[1.0, 2.0]])
    assert m["max_abs_error"] == 1.0
    assert m["mae"] == 0.5
    assert 0.0 < m["cosine_similarity"] < 1.0


def test_shape_mismatch_detected():
    m = compute_compare_metrics([[1.0, 2.0]], [[1.0, 2.0, 3.0]])
    assert m["shape_match"] is False
    assert m["mae"] is None


def test_frame_count_mismatch_detected():
    m = compute_compare_metrics([[1.0]], [[1.0], [2.0]])
    assert m["shape_match"] is False


def test_non_finite_detected():
    m = compute_compare_metrics([[1.0]], [[float("nan")]])
    assert m["finite"] is False
    assert m["mae"] is None


def test_report_schema_valid():
    cfg = CompareV2Config(torch_policy_path="t", coreai_policy_path="c",
                          dataset_repo_id="lerobot/pusht")
    report = build_compare_v2_report(
        cfg, ok=True, source_report={"ok": True}, contract_report={"ambiguous": False},
        metrics=compute_compare_metrics([[1.0]], [[1.0]]))
    assert report["schema_version"] == COMPARE_V2_SCHEMA_VERSION
    assert report["claims"]["proves_task_success"] is False
    schema = json.loads(files("lerobot_coreai.schemas").joinpath(
        "compare-v2-report.schema.json").read_text())
    jsonschema.validate(report, schema)
