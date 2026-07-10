# test_rollout_verify.py — canonical hashing + offline evidence verifier (v1.3.14).

import json

import pytest

from lerobot_coreai.rollout_evidence_schema import (
    CanonicalJSONError, canonical_json_sha256,
)
from lerobot_coreai.rollout_verify import verify_official_rollout_evidence


def test_canonical_hash_stable_and_order_independent_for_dicts():
    a = canonical_json_sha256({"x": 1, "y": [1, 2]})
    b = canonical_json_sha256({"y": [1, 2], "x": 1})
    assert a == b and a.startswith("sha256:")


def test_canonical_hash_rejects_non_json():
    with pytest.raises(CanonicalJSONError):
        canonical_json_sha256({"x": object()})


def test_canonical_hash_rejects_nonfinite():
    with pytest.raises(CanonicalJSONError):
        canonical_json_sha256([float("inf")])


def test_verify_missing_bundle_dir_fails(tmp_path):
    assert not verify_official_rollout_evidence(str(tmp_path / "nope")).ok


def test_verify_empty_dir_requires_matrix(tmp_path):
    (tmp_path).mkdir(exist_ok=True)
    res = verify_official_rollout_evidence(str(tmp_path), require_complete_matrix=True)
    assert not res.ok            # no cases, no matrix
    assert any(v.startswith("failed") for v in res.checks.values())
