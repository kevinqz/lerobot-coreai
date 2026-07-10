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


# --- v1.3.16: replay robustness on malformed raw (no crash) ---

def test_replay_malformed_measurements_fails_not_crashes(tmp_path):
    from lerobot_coreai.rollout_replay import replay_rollout_evidence
    case = tmp_path / "single_only-b1"; case.mkdir()
    (case / "measurements.json").write_text(json.dumps({"batch_size": 1}))  # missing fields
    (case / "official_rollout_readiness_report.json").write_text(json.dumps({"checks": {}}))
    res = replay_rollout_evidence(str(case))
    assert res.ok is False and res.errors        # structured failure, no exception


def test_derive_checks_bad_shapes_do_not_crash():
    from lerobot_coreai.rollout_replay import derive_checks
    raw = {"batch_size": 2, "mode": "native_batch", "sequence_length": 3, "horizon": 3,
           "action_dim": 7, "terminate_at": [3, 3],
           "request_bodies": [{"observation": {}, "options": {}}],
           "response_bodies": [{"action": "not-a-list"}],
           "done_mask": [[1, 1, 1], [1, 1, 1]], "final_action": [],
           "required_obs_keys": ["observation.state"], "fixture_contract": {}}
    checks = derive_checks(raw)          # must not raise
    assert checks["response_action_chain_valid"] is False
    assert checks["wire_payload_valid"] is False
