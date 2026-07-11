# test_processor_parity.py — v1.3.26: transform contract, parity metrics, exact +
# numeric parity gates, report + offline verification. Pure base (no lerobot/torch).
# Reference and candidate use INDEPENDENT code paths (redteam mitigation).

import jsonschema
import pytest

from lerobot_coreai.processor_parity import (
    PROCESSOR_PARITY_REPORT_SCHEMA, ParityCase, build_processor_parity_report,
    verify_processor_parity_report,
)
from lerobot_coreai.processor_parity_metrics import compute_parity_metrics
from lerobot_coreai.processor_transform_contract import (
    PROCESSOR_TRANSFORM_SCHEMA, TransformError, apply_operations,
    transform_contract_sha256,
)

# a 2x2x3 HWC image with distinct values.
_HWC = [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]


def _reference_hwc_to_chw(hwc):
    """Independent reference: HWC -> CHW via explicit index walk."""
    H, W, C = len(hwc), len(hwc[0]), len(hwc[0][0])
    return [[[hwc[h][w][c] for w in range(W)] for h in range(H)] for c in range(C)]


def _candidate_hwc_to_chw_scaled(hwc):
    """Independent candidate: HWC uint8 -> CHW float [0,1]."""
    H, W, C = len(hwc), len(hwc[0]), len(hwc[0][0])
    return [[[hwc[h][w][c] / 255.0 for w in range(W)] for h in range(H)]
            for c in range(C)]


# --- transform ops (reference implementation) ---

def test_permute_hwc_to_chw():
    got = apply_operations(_HWC, [{"op": "permute", "order": [2, 0, 1]}])
    assert got == _reference_hwc_to_chw(_HWC)


def test_scale_and_cast():
    got = apply_operations([[255, 0]], [{"op": "cast", "to": "float32"},
                                        {"op": "scale", "factor": 1 / 255}])
    assert got == [[1.0, 0.0]]


def test_normalize_denormalize_roundtrip():
    x = [[2.0, 4.0, 6.0]]
    norm = apply_operations(x, [{"op": "normalize", "mean": [1.0, 1.0, 1.0],
                                 "std": [2.0, 2.0, 2.0]}])
    back = apply_operations(norm, [{"op": "denormalize", "mean": [1.0, 1.0, 1.0],
                                    "std": [2.0, 2.0, 2.0]}])
    assert back == x


def test_permute_bad_rank_fails():
    with pytest.raises(TransformError):
        apply_operations(_HWC, [{"op": "permute", "order": [0, 1]}])


def test_transform_contract_schema_and_hash():
    c = {"schema_version": "lerobot-coreai.processor-transform.v1",
         "transform_id": "front-hwc-u8-to-chw-f32-01", "owner": "policy_preprocessor",
         "source_stage": "lerobot_preprocess_observation_output.v1",
         "target_stage": "lerobot_policy_preprocessor_output.v1",
         "operations": [{"op": "permute", "order": [2, 0, 1]},
                        {"op": "cast", "to": "float32"},
                        {"op": "scale", "factor": 0.00392156862745098}]}
    jsonschema.validate(c, PROCESSOR_TRANSFORM_SCHEMA)
    assert transform_contract_sha256(c).startswith("sha256:")


# --- exact parity ---

def test_exact_parity_permute_passes():
    ref = apply_operations(_HWC, [{"op": "permute", "order": [2, 0, 1]}])
    cand = _reference_hwc_to_chw(_HWC)          # independent path, same ints
    case = ParityCase("observation:front@...", "raw", "chw", "exact", ref, cand)
    assert case.evaluate()["passed"]


def test_exact_parity_hwc_chw_swap_fails():
    ref = apply_operations(_HWC, [{"op": "permute", "order": [2, 0, 1]}])  # CHW
    cand = _HWC                                  # candidate forgot to permute
    case = ParityCase("observation:front@...", "raw", "chw", "exact", ref, cand)
    r = case.evaluate()
    assert not r["passed"]


# --- numeric parity ---

def _numeric_case(ref, cand, thresholds=None):
    return ParityCase("observation:front@...", "raw", "chw", "numeric", ref, cand,
                      thresholds=thresholds if thresholds is not None
                      else {"max_abs_error": 1e-6, "min_cosine_similarity": 0.999999})


def test_numeric_parity_passes():
    ref = apply_operations(_HWC, [{"op": "permute", "order": [2, 0, 1]},
                                  {"op": "cast", "to": "float32"},
                                  {"op": "scale", "factor": 1 / 255}])
    cand = _candidate_hwc_to_chw_scaled(_HWC)
    assert _numeric_case(ref, cand).evaluate()["passed"]


def test_numeric_parity_missing_divide_fails():
    ref = apply_operations(_HWC, [{"op": "permute", "order": [2, 0, 1]},
                                  {"op": "cast", "to": "float32"},
                                  {"op": "scale", "factor": 1 / 255}])
    # candidate forgot the /255 scale.
    cand = _reference_hwc_to_chw(_HWC)
    assert not _numeric_case(ref, cand).evaluate()["passed"]


def test_numeric_parity_double_normalization_fails():
    x = [[2.0, 4.0]]
    ref = apply_operations(x, [{"op": "normalize", "mean": [0.0, 0.0], "std": [2.0, 2.0]}])
    cand = apply_operations(ref, [{"op": "normalize", "mean": [0.0, 0.0],
                                   "std": [2.0, 2.0]}])  # normalized twice
    assert not _numeric_case(ref, cand).evaluate()["passed"]


def test_numeric_parity_requires_explicit_thresholds():
    r = ParityCase("x", "a", "b", "numeric", [[1.0]], [[1.0]], thresholds={}).evaluate()
    assert not r["passed"] and any("threshold" in x for x in r["reasons"])


def test_nonfinite_always_fails():
    r = _numeric_case([[float("inf")]], [[float("inf")]]).evaluate()
    assert not r["passed"]


# --- report + offline verification ---

def test_report_passes_and_verifies():
    ref = apply_operations(_HWC, [{"op": "permute", "order": [2, 0, 1]}])
    cand = _reference_hwc_to_chw(_HWC)
    report = build_processor_parity_report(
        [ParityCase("f", "a", "b", "exact", ref, cand)],
        feature_contract_sha256="sha256:" + "a" * 64)
    jsonschema.validate(report, PROCESSOR_PARITY_REPORT_SCHEMA)
    assert report["claims"]["processor_parity_verified"] is True
    ok, errs = verify_processor_parity_report(report)
    assert ok, errs


def test_report_tamper_detected():
    ref = apply_operations(_HWC, [{"op": "permute", "order": [2, 0, 1]}])
    report = build_processor_parity_report(
        [ParityCase("f", "a", "b", "exact", ref, _HWC)])       # a failing case
    assert report["passed"] is False
    report["cases"][0]["passed"] = True                        # forge the flag
    ok, _ = verify_processor_parity_report(report)
    assert not ok


# --- P1.1: certificate-grade independent replay ---

def test_certificate_grade_persists_arrays_and_replays():
    ref = apply_operations(_HWC, [{"op": "permute", "order": [2, 0, 1]}])
    cand = _reference_hwc_to_chw(_HWC)
    report = build_processor_parity_report(
        [ParityCase("f", "a", "b", "exact", ref, cand)])       # default = certificate
    assert report["evidence_grade"] == "certificate"
    c0 = report["cases"][0]
    assert c0["reference"] == ref and c0["candidate"] == cand   # raw arrays persisted
    ok, errs = verify_processor_parity_report(report)
    assert ok, errs


def test_certificate_grade_metrics_tamper_detected_by_replay():
    # a report can be self-consistent (passed matches reasons) yet have FORGED metrics;
    # certificate-grade replay recomputes from the raw arrays and catches it.
    ref = [[0.0, 0.0]]
    cand = [[10.0, 10.0]]                                       # clearly far apart
    report = build_processor_parity_report(
        [ParityCase("f", "a", "b", "numeric", ref, cand,
                    thresholds={"max_abs_error": 1e-6})])
    assert report["passed"] is False
    # forge the case to look passing AND self-consistent (empty reasons).
    report["cases"][0]["passed"] = True
    report["cases"][0]["reasons"] = []
    report["passed"] = True
    report["claims"]["processor_parity_verified"] = True
    ok, errs = verify_processor_parity_report(report)
    assert not ok and any("verdict" in e or "metrics" in e for e in errs)


def test_certificate_grade_array_tamper_detected_by_replay():
    ref = apply_operations(_HWC, [{"op": "permute", "order": [2, 0, 1]}])
    cand = _reference_hwc_to_chw(_HWC)
    report = build_processor_parity_report([ParityCase("f", "a", "b", "exact", ref, cand)])
    report["cases"][0]["candidate"][0][0][0] = 999             # tamper a raw value
    ok, errs = verify_processor_parity_report(report)
    assert not ok and any("hash mismatch" in e or "verdict" in e for e in errs)


def test_diagnostic_grade_omits_arrays():
    ref = apply_operations(_HWC, [{"op": "permute", "order": [2, 0, 1]}])
    cand = _reference_hwc_to_chw(_HWC)
    report = build_processor_parity_report(
        [ParityCase("f", "a", "b", "exact", ref, cand)], evidence_grade="diagnostic")
    assert report["evidence_grade"] == "diagnostic"
    assert "reference" not in report["cases"][0]
    ok, errs = verify_processor_parity_report(report)      # consistency-only, still ok
    assert ok, errs


def test_shape_mismatch_metrics():
    m = compute_parity_metrics([[1.0, 2.0]], [[1.0, 2.0, 3.0]])
    assert not m.shape_match and m.max_abs_error == float("inf")


# --- P1.4: ragged arrays never match ---

def test_ragged_array_never_matches():
    ragged = [[1.0], [2.0, 3.0]]                     # not rectangular
    m = compute_parity_metrics(ragged, ragged)       # identical ragged inputs
    assert not m.shape_match and m.max_abs_error == float("inf")
    r = _numeric_case(ragged, ragged, {"max_abs_error": 1e-6}).evaluate()
    assert not r["passed"]


# --- P1.5: closed case schema + mode enum ---

def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        ParityCase("f", "a", "b", "fuzzy", [[1.0]], [[1.0]]).evaluate()


def test_malformed_case_fails_schema():
    ref = apply_operations(_HWC, [{"op": "permute", "order": [2, 0, 1]}])
    cand = _reference_hwc_to_chw(_HWC)
    report = build_processor_parity_report([ParityCase("f", "a", "b", "exact", ref, cand)])
    report["cases"][0]["mode"] = "sideways"          # not in the closed enum
    ok, errs = verify_processor_parity_report(report)
    assert not ok and any("schema" in e for e in errs)


# --- save/reload parity (the transform contract IS the serializable processor spec) ---

def test_transform_save_reload_output_parity(tmp_path):
    import json
    ops = [{"op": "permute", "order": [2, 0, 1]}, {"op": "cast", "to": "float32"},
           {"op": "scale", "factor": 1 / 255}]
    contract = {"schema_version": "lerobot-coreai.processor-transform.v1",
                "transform_id": "t", "owner": "policy_preprocessor",
                "source_stage": "lerobot_preprocess_observation_output.v1",
                "target_stage": "lerobot_policy_preprocessor_output.v1",
                "operations": ops}
    before = apply_operations(_HWC, contract["operations"])
    # serialize -> reload (config parity) -> re-apply (output parity).
    (tmp_path / "t.json").write_text(json.dumps(contract))
    reloaded = json.loads((tmp_path / "t.json").read_text())
    assert transform_contract_sha256(reloaded) == transform_contract_sha256(contract)
    after = apply_operations(_HWC, reloaded["operations"])
    case = _numeric_case(before, after)
    assert case.evaluate()["passed"]
