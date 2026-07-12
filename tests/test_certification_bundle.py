# test_certification_bundle.py — CertificationBundle v1 (v1.3.26.13). Every root is
# DERIVED by content-addressing its leaf (no bare-hash inputs) and the self-contained
# leaves are re-verified.

import copy

import pytest

from lerobot_coreai.authority import AuthorityError, verify_certification_bundle
from lerobot_coreai.certification_bundle import (
    CERTIFICATION_BUNDLE_SCHEMA_VERSION, verify_certification_bundle as _verify,
)
from lerobot_coreai.official_eval_certificate import _ROOT_KEYS
from lerobot_coreai.rollout_evidence_schema import canonical_json_sha256

_H = "sha256:" + "a" * 64


def _parity_leaf():
    from lerobot_coreai.processor_parity import ParityCase, build_processor_parity_report
    return build_processor_parity_report(
        [ParityCase("f", "a", "b", "exact", [[1, 2]], [[1, 2]])])


def _conversion_leaf():
    from lerobot_coreai.model_conversion_evidence import build_model_conversion_evidence
    return build_model_conversion_evidence(
        source={"repository": "org/p", "revision": "r1", "weights_sha256": _H},
        exporter={"name": "coreai", "build": "2.0.0"},
        export_configuration={"opset": 18},
        artifact={"aimodel_sha256": _H, "aimodel_schema_version": "aimodel.v1",
                  "manifest_sha256": _H},
        reference_outputs=[[1.0, 2.0]], candidate_outputs=[[1.0, 2.0]],
        tolerance={"max_abs_error": 0.0})


def _leaves():
    leaves = {k: {"root_kind": k} for k in _ROOT_KEYS}
    leaves["processor_parity_sha256"] = _parity_leaf()
    leaves["model_conversion_sha256"] = _conversion_leaf()
    leaves["artifact_root_sha256"] = {"aimodel_sha256": _H}
    return leaves


def _bundle(leaves=None):
    return {"schema_version": CERTIFICATION_BUNDLE_SCHEMA_VERSION,
            "leaves": leaves if leaves is not None else _leaves()}


def test_valid_bundle_verifies_and_derives_roots():
    ok, reasons, result = _verify(_bundle())
    assert ok, reasons
    # every root is the content hash of its leaf — not a caller-supplied string.
    for name in _ROOT_KEYS:
        assert result["roots"][name].startswith("sha256:")
    assert result["levels"]["processor_parity_sha256"] == "reverified"
    assert result["levels"]["model_conversion_sha256"] == "reverified"
    assert result["levels"]["runner_capabilities_sha256"] == "addressed"


def test_roots_are_content_addresses_of_leaves():
    leaves = _leaves()
    ok, _, result = _verify(_bundle(leaves))
    assert ok
    assert result["roots"]["runner_capabilities_sha256"] == \
        canonical_json_sha256(leaves["runner_capabilities_sha256"])


def test_missing_leaf_rejected():
    leaves = _leaves(); del leaves["rollout_matrix_sha256"]
    ok, reasons, _ = _verify(_bundle(leaves))
    assert not ok and any("schema" in r for r in reasons)


def test_failing_parity_leaf_rejects_bundle():
    # a parity report whose verdict was forged inconsistent fails re-verification.
    leaves = _leaves()
    bad = copy.deepcopy(leaves["processor_parity_sha256"])
    bad["cases"][0]["passed"] = False        # inconsistent with empty reasons
    leaves["processor_parity_sha256"] = bad
    ok, reasons, _ = _verify(_bundle(leaves))
    assert not ok and any("processor_parity" in r for r in reasons)


def test_failing_conversion_leaf_rejects_bundle():
    leaves = _leaves()
    bad = copy.deepcopy(leaves["model_conversion_sha256"])
    bad["numeric_parity"]["metrics"]["max_abs_error"] = 999.0   # forged metric
    leaves["model_conversion_sha256"] = bad
    ok, reasons, _ = _verify(_bundle(leaves))
    assert not ok and any("model_conversion" in r for r in reasons)


def test_conversion_aimodel_cross_binding_enforced():
    leaves = _leaves()
    leaves["artifact_root_sha256"] = {"aimodel_sha256": "sha256:" + "d" * 64}  # != _H
    ok, reasons, _ = _verify(_bundle(leaves))
    assert not ok and any("aimodel" in r for r in reasons)


def test_authority_mint_raises_on_invalid_bundle():
    leaves = _leaves(); del leaves["negotiation_record_sha256"]
    with pytest.raises(AuthorityError):
        verify_certification_bundle(_bundle(leaves))


def test_authority_mint_carries_roots_and_levels():
    vb = verify_certification_bundle(_bundle())
    assert set(vb.payload["roots"]) == set(_ROOT_KEYS)
    assert vb.payload["levels"]["model_conversion_sha256"] == "reverified"
