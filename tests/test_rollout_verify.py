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


# --- v1.3.17: queue lifecycle state-machine derivation ---

def _raw_with_events(events):
    return {"batch_size": 1, "mode": "single_only", "sequence_length": 3, "horizon": 3,
            "action_dim": 7, "terminate_at": [3],
            "request_bodies": [{"observation": {}, "options": {}}],
            "response_bodies": [{"action": [[0.0] * 7 for _ in range(3)]}],
            "done_mask": [[1, 1, 1]], "final_action": [[[0.0] * 7 for _ in range(3)]],
            "required_obs_keys": [], "fixture_contract": {}, "queue_events": events}


def _valid_events():
    """A minimal valid protocol-v2 stream: single_only B=1, seq=3, H=3 (1 chunk)."""
    ch = "sha256:" + "a" * 64
    ev, i = [], 0

    def add(name, **kw):
        nonlocal i
        e = {"event_index": i, "event": name, "execution_id": "x",
             "prediction_id": kw.pop("pid", None), "chunk_id": kw.pop("cid", None)}
        e.update(kw)
        ev.append(e)
        i += 1
    add("execution.started")
    add("policy.reset", queue_size_after=0)
    add("queue.empty", queue_size_after=0)
    add("queue.refill_requested", pid=0, cid="chunk-0", queue_size_before=0, queue_size_after=0)
    add("runner.request_started", pid=0, cid="chunk-0")
    add("runner.response_received", pid=0, cid="chunk-0", response_sha256=ch)
    add("chunk.validated", pid=0, cid="chunk-0", chunk_sha256=ch, horizon=3)
    add("chunk.committed", pid=0, cid="chunk-0", chunk_sha256=ch,
        queue_size_before=0, queue_size_after=3)
    for b in range(3):
        add("action.popped", pid=0, cid="chunk-0",
            queue_size_before=3 - b, queue_size_after=2 - b)
    add("queue.exhausted", pid=0, cid="chunk-0", queue_size_after=0)
    add("execution.completed")
    return ev


def _reindex(ev):
    for j, e in enumerate(ev):
        e["event_index"] = j
    return ev


def test_queue_lifecycle_valid_sequence():
    from lerobot_coreai.rollout_replay import derive_checks
    c = derive_checks(_raw_with_events(_valid_events()))
    assert c["queue_lifecycle_valid"] and c["queue_refill_count_exact"]


def test_queue_commit_before_validation_fails():
    from lerobot_coreai.rollout_replay import derive_checks
    ev = [e for e in _valid_events() if e["event"] != "chunk.validated"]
    assert derive_checks(_raw_with_events(_reindex(ev)))["queue_lifecycle_valid"] is False


def test_queue_pop_wrong_prediction_id_fails():
    from lerobot_coreai.rollout_replay import derive_checks
    ev = _valid_events()
    for e in ev:
        if e["event"] == "action.popped":
            e["prediction_id"] = 99          # not the committing chunk's id (P1.1)
            break
    assert derive_checks(_raw_with_events(ev))["queue_lifecycle_valid"] is False


def test_queue_bad_commit_arithmetic_fails():
    from lerobot_coreai.rollout_replay import derive_checks
    ev = _valid_events()
    for e in ev:
        if e["event"] == "chunk.committed":
            e["queue_size_after"] = 99       # not before + H
    assert derive_checks(_raw_with_events(ev))["queue_lifecycle_valid"] is False


def test_no_queue_events_fails_closed():
    from lerobot_coreai.rollout_replay import derive_checks
    c = derive_checks(_raw_with_events([]))
    assert c["queue_lifecycle_valid"] is False and c["queue_refill_count_exact"] is False
