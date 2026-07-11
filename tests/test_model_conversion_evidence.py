# test_model_conversion_evidence.py — ModelConversionEvidence v1 (v1.3.26.2).
# Pure base. Numeric parity uses synthetic reference/candidate arrays (the real
# exporter/.aimodel runs Apple-side); the identity-chain + parity gate are proven here.

import pytest

from lerobot_coreai.model_conversion_evidence import (
    build_model_conversion_evidence, verify_model_conversion_evidence,
)

_H = "sha256:" + "a" * 64


def _source():
    return {"repository": "kevinqz/EVO1-SO100", "revision": "e40b58a", "weights_sha256": _H}


def _exporter():
    return {"name": "coreai-fabric", "build": "fabric-2026.7.0"}


def _artifact():
    return {"aimodel_sha256": _H, "aimodel_schema_version": "aimodel.v1",
            "manifest_sha256": _H}


def _build(ref, cand, tol):
    return build_model_conversion_evidence(
        source=_source(), exporter=_exporter(),
        export_configuration={"opset": 17, "fp16": True}, artifact=_artifact(),
        reference_outputs=ref, candidate_outputs=cand, tolerance=tol,
        operator_coverage=["conv", "matmul", "layernorm"],
        quantization={"scheme": "int8_dynamic", "parameters": {}})


_TOL = {"max_abs_error": 1e-3, "min_cosine_similarity": 0.999}


def test_matching_outputs_verify():
    ref = [[1.0, 2.0, 3.0]]
    cand = [[1.0005, 2.0004, 2.9996]]
    ev = _build(ref, cand, _TOL)
    assert ev["claims"]["model_conversion_verified"] is True
    ok, errs = verify_model_conversion_evidence(
        ev, reference_outputs=ref, candidate_outputs=cand)
    assert ok, errs


def test_tolerance_exceeded_not_verified():
    ref = [[1.0, 2.0, 3.0]]
    cand = [[1.5, 2.5, 3.5]]                     # well outside 1e-3
    ev = _build(ref, cand, _TOL)
    assert ev["claims"]["model_conversion_verified"] is False
    ok, _ = verify_model_conversion_evidence(
        ev, reference_outputs=ref, candidate_outputs=cand)
    assert ok        # evidence is internally consistent (honestly records the failure)


def test_missing_tolerance_fails():
    with pytest.raises(Exception):
        _build([[1.0]], [[1.0]], {})            # no explicit tolerance


def test_nonfinite_candidate_not_verified():
    ev = _build([[1.0]], [[float("inf")]], _TOL)
    assert ev["claims"]["model_conversion_verified"] is False


def test_shape_mismatch_not_verified():
    ev = _build([[1.0, 2.0]], [[1.0, 2.0, 3.0]], _TOL)
    assert ev["claims"]["model_conversion_verified"] is False


def test_output_hash_tamper_detected():
    ref, cand = [[1.0]], [[1.0]]
    ev = _build(ref, cand, _TOL)
    ev["numeric_parity"]["candidate_outputs_sha256"] = "sha256:" + "b" * 64
    ok, errs = verify_model_conversion_evidence(
        ev, reference_outputs=ref, candidate_outputs=cand)
    assert not ok and any("candidate_outputs hash" in e for e in errs)


def test_claim_forgery_detected():
    ref, cand = [[1.0]], [[9.0]]                # parity fails
    ev = _build(ref, cand, _TOL)
    ev["claims"]["model_conversion_verified"] = True    # forge the claim
    ok, errs = verify_model_conversion_evidence(ev)
    assert not ok


def test_missing_weights_digest_fails_schema():
    ref, cand = [[1.0]], [[1.0]]
    ev = _build(ref, cand, _TOL)
    del ev["source"]["weights_sha256"]
    ok, _ = verify_model_conversion_evidence(ev)
    assert not ok        # schema requires the source weights digest
