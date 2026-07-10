# test_compare_v2.py — compare-v2 metrics, gates, structural shape (v1.2.6/1.2.7).

import json
from importlib.resources import files

import jsonschema

from lerobot_coreai.compare_v2 import (
    COMPARE_V2_SCHEMA_VERSION, CompareV2Config, build_compare_v2_report,
    compute_compare_metrics, evaluate_gates,
)


def test_identical_actions_perfect_parity():
    src = [[1.0, 2.0], [3.0, 4.0]]
    m = compute_compare_metrics(src, [list(x) for x in src])
    assert m["shape_match"] is True and m["finite"] is True
    assert m["mae"] == 0.0 and m["max_abs_error"] == 0.0
    assert m["cosine_similarity"] == 1.0


def test_structural_shape_mismatch_detected_even_with_same_flat_len():
    # [[1,2],[3,4]] flattens to the same 4 values as [1,2,3,4] but is a
    # different structure — must be caught (regression of the v1.2.6 bug).
    m = compute_compare_metrics([[[1.0, 2.0], [3.0, 4.0]]], [[1.0, 2.0, 3.0, 4.0]])
    assert m["shape_match"] is False
    assert m["mae"] is None


def test_non_finite_detected():
    m = compute_compare_metrics([[1.0]], [[float("nan")]])
    assert m["finite"] is False


def test_frame_count_mismatch_detected():
    m = compute_compare_metrics([[1.0]], [[1.0], [2.0]])
    assert m["shape_match"] is False


def test_gates_pass_and_fail():
    good = compute_compare_metrics([[1.0, 1.0]], [[1.0, 1.0]])
    cfg = CompareV2Config(torch_policy_path="t", coreai_policy_path="c",
                          dataset_repo_id="d", max_mean_mae=1e-6,
                          min_cosine_similarity=0.999)
    gates = evaluate_gates(good, cfg)
    assert gates["mean_mae"]["passed"] is True
    assert gates["min_cosine_similarity"]["passed"] is True

    bad = compute_compare_metrics([[0.0, 0.0]], [[1000.0, 1000.0]])
    gates_bad = evaluate_gates(bad, cfg)
    assert gates_bad["mean_mae"]["passed"] is False


def test_configured_gate_with_missing_metric_fails_not_omitted():
    # cosine is None when norms are zero; a configured gate must FAIL, not vanish.
    metrics = compute_compare_metrics([[0.0, 0.0]], [[0.0, 0.0]])
    cfg = CompareV2Config(torch_policy_path="t", coreai_policy_path="c",
                          dataset_repo_id="d", min_cosine_similarity=0.999)
    gates = evaluate_gates(metrics, cfg)
    assert "min_cosine_similarity" in gates
    assert gates["min_cosine_similarity"]["passed"] is False
    assert gates["min_cosine_similarity"]["reason"] == "metric_unavailable"


def test_report_schema_valid_and_parity_requires_gates():
    cfg = CompareV2Config(torch_policy_path="t", coreai_policy_path="c",
                          dataset_repo_id="d")
    # No gates configured → parity must NOT be proven even with a perfect match.
    report = build_compare_v2_report(
        cfg, ok=True, parity_proven=False, source_report={"ok": True},
        contract_report={"ambiguous": False},
        metrics=compute_compare_metrics([[1.0]], [[1.0]]), gates={},
        gates_configured=False)
    assert report["schema_version"] == COMPARE_V2_SCHEMA_VERSION
    assert report["experimental"] is True
    assert report["claims"]["proves_action_parity_on_final_unit"] is False
    assert report["claims"]["proves_physical_safety"] is False
    schema = json.loads(files("lerobot_coreai.schemas").joinpath(
        "compare-v2-report.schema.json").read_text())
    jsonschema.validate(report, schema)
