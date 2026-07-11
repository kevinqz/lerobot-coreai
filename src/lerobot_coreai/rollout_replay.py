# rollout_replay.py — SEMANTIC replay of rollout evidence, offline (v1.3.15).
#
# v1.3.14's verifier proved file integrity. This re-DERIVES every semantic check
# from the recorded raw data (requests, responses, done mask, final action) and the
# response -> temporal-queue -> final-action chain, WITHOUT trusting the report's
# own check booleans. Pure Python + JSON — no lerobot, no torch, no NumPy.

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .coreai_observation_serialization import observation_sha256
from .rollout_evidence_schema import (
    MEASUREMENTS_SCHEMA, REQUIRED_CHECKS, canonical_json_sha256,
)

_LEAK_KEYS = ("action", "reward", "done", "success", "index", "episode_index",
              "frame_index", "timestamp", "next.reward", "next.done", "next.truncated")


def _rect_shape(v: Any) -> list[int] | None:
    """Strict rectangular shape: EVERY branch must match (P1.4). None if ragged."""
    if not isinstance(v, list):
        return []                       # scalar leaf
    child_shapes = [_rect_shape(x) for x in v]
    if any(cs is None for cs in child_shapes):
        return None
    if child_shapes and any(cs != child_shapes[0] for cs in child_shapes):
        return None                     # ragged: sibling shapes differ
    return [len(v)] + (child_shapes[0] if child_shapes else [])


def _per_sample_shape(v: Any) -> list[int]:
    s = _rect_shape(v)
    return s if s is not None else [-1]   # ragged -> never matches an expected shape


def _finite(v: Any) -> bool:
    if isinstance(v, bool):
        return True
    if isinstance(v, (int, float)):
        return math.isfinite(v)
    if isinstance(v, list):
        return all(_finite(x) for x in v)
    return False


def _done_cumulative(done) -> bool:
    for row in done:
        seen = False
        for v in row:
            if v:
                seen = True
            elif seen:
                return False
    return True


def _first_done_matches(done, terminate_at) -> bool:
    if len(done) != len(terminate_at):
        return False
    for i, ta in enumerate(terminate_at):
        row, fd = done[i], ta - 1
        if any(row[:fd]) or not all(row[fd:]):
            return False
    return True


def _wire_ok(raw) -> bool:
    B, native = raw["batch_size"], raw["mode"] == "native_batch"
    exact = set(raw["required_obs_keys"]) | {"task"}
    for body in raw["request_bodies"]:
        obs = body.get("observation", {})
        if set(obs.keys()) != exact:
            return False
        if any(k in obs for k in _LEAK_KEYS):
            return False
        task = obs.get("task")
        if native and B > 1:
            if not (isinstance(task, list) and len(task) == B
                    and all(isinstance(t, str) for t in task)):
                return False
        elif not isinstance(task, str):
            return False
        for key, per_sample in raw["fixture_contract"].items():
            shape = _per_sample_shape(obs.get(key))
            expected = ([B] + list(per_sample)) if (native and B > 1) else list(per_sample)
            if shape != expected:
                return False
        opts = body.get("options", {})
        if native and B > 1:
            if opts.get("batch_size") != B:
                return False
        elif "batch_size" in opts:
            return False                      # single/split must NOT send batch_size
        # v1.3.16: recompute observation_sha256 from the sent observation (P1.3).
        if opts.get("observation_sha256") != observation_sha256(obs):
            return False
        if opts.get("observation_encoding") not in ("nested_json_v1",
                                                     "typed_array_envelope_v1"):
            return False
        if not isinstance(opts.get("protocol_version"), str):
            return False
    return True


def _response_valid(raw) -> bool:
    """Strict, exception-free response-shape/finiteness validation (P1.4)."""
    B, H, A = raw["batch_size"], raw["horizon"], raw["action_dim"]
    native = raw["mode"] == "native_batch"
    responses = raw["response_bodies"]
    if len(responses) != len(raw["request_bodies"]) or not responses:
        return False
    for r in responses:
        if set(r.keys()) - {"action"}:            # closed key set
            return False
        act = r.get("action")
        shape = _per_sample_shape(act)
        want = [B, H, A] if (native and B > 1) else [H, A]
        if shape != want or not _finite(act):
            return False
    return True


def _chain_ok(raw) -> bool:
    """Reconstruct final_action from responses via the temporal-queue transpose.

    Assumes responses already passed ``_response_valid`` (called first in
    derive_checks); still index-guarded so malformed data can never raise.
    """
    if not _response_valid(raw):
        return False
    B, H, A = raw["batch_size"], raw["horizon"], raw["action_dim"]
    seq, native = raw["sequence_length"], raw["mode"] == "native_batch"
    final = raw["final_action"]                       # [B, seq, A]
    responses = [r.get("action") for r in raw["response_bodies"]]
    if _per_sample_shape(final) != [B, seq, A] or not _finite(final):
        return False
    try:
        for t in range(seq):
            pred, hidx = t // H, t % H
            for b in range(B):
                if native and B > 1:
                    expected = responses[pred][b][hidx]        # [B,H,A]
                else:                                          # single/split: [H,A]
                    idx = pred * B + b if raw["mode"] == "split_and_stack" else pred
                    expected = responses[idx][hidx]
                if final[b][t] != expected:
                    return False
    except (IndexError, TypeError, KeyError):
        return False
    return True


def _fixture_ok(raw) -> bool:
    return _wire_ok(raw)   # fixture shape validation is part of the wire check


def _queue_lifecycle(events, predictions: int, seq: int, H: int,
                     reqs_per_refill: int) -> tuple[bool, bool]:
    """Formal state-machine replay of the queue evidence protocol v2 (v1.3.18).

    Proves causal ordering, per-chunk prediction/chunk-id attribution, queue
    before/after arithmetic, validated==committed chunk identity, atomic commit
    (before -> before+H), and exact refill/commit/pop counts. Fail-closed.
    Returns (lifecycle_valid, refill_count_exact).
    """
    from .rollout_evidence_schema import QUEUE_EVENT_SCHEMA
    import jsonschema
    if not events:
        return False, False
    try:
        for e in events:
            jsonschema.validate(e, QUEUE_EVENT_SCHEMA)
    except Exception:  # noqa: BLE001
        return False, False
    if [e["event_index"] for e in events] != list(range(len(events))):
        return False, False
    if events[0]["event"] != "execution.started" or events[-1]["event"] != "execution.completed":
        return False, False

    st = "STARTED"
    committed = popped = refills = resets = 0
    reqs = 0
    qsize = 0
    validated_hash = None
    active_pid = None
    active_cid = None
    seen_pids: set = set()
    pops_since_commit = 0

    def bad():
        return False, False

    for e in events[1:-1]:
        ev = e["event"]
        if ev == "policy.reset":
            if st not in ("STARTED", "EMPTY"):
                return bad()
            resets += 1
            qsize = 0
            st = "EMPTY"
        elif ev == "queue.empty":
            if e.get("queue_size_after", 0) != 0 or qsize != 0:
                return bad()
            st = "EMPTY"
        elif ev == "queue.refill_requested":
            if st != "EMPTY" or qsize != 0:      # refill ONLY when empty (P1.3)
                return bad()
            pid, cid = e.get("prediction_id"), e.get("chunk_id")
            if pid is None or cid is None or pid in seen_pids:   # no reuse (P1.1)
                return bad()
            active_pid, active_cid = pid, cid
            seen_pids.add(pid)
            refills += 1
            reqs = 0
            validated_hash = None
            st = "REFILL"
        elif ev == "runner.request_started":
            # allowed after the refill (native/single) or after a prior response
            # (split issues B requests per refill).
            if st not in ("REFILL", "RESP") or e.get("prediction_id") != active_pid:
                return bad()
            st = "REQ"
        elif ev == "runner.response_received":
            if st != "REQ" or e.get("prediction_id") != active_pid:
                return bad()
            reqs += 1
            st = "RESP"
        elif ev == "chunk.validated":
            if st != "RESP" or reqs != reqs_per_refill:   # request(s) before validate
                return bad()
            validated_hash = e.get("chunk_sha256")
            st = "VALIDATED"
        elif ev == "chunk.committed":
            if st != "VALIDATED" or e.get("prediction_id") != active_pid:
                return bad()
            if e.get("chunk_sha256") != validated_hash:  # committed == validated (identity)
                return bad()
            if e.get("queue_size_before") != qsize or e.get("queue_size_after") != qsize + H:
                return bad()                     # atomic commit before->before+H (P1.4/P1.8)
            qsize += H
            committed += 1
            pops_since_commit = 0
            st = "READY"
        elif ev == "action.popped":
            if st not in ("READY", "DRAINING") or committed == 0:
                return bad()
            if e.get("prediction_id") != active_pid or e.get("chunk_id") != active_cid:
                return bad()                     # pop attributed to its own chunk (P1.1)
            if e.get("queue_size_before") != qsize or e.get("queue_size_after") != qsize - 1:
                return bad()
            qsize -= 1
            popped += 1
            pops_since_commit += 1
            if pops_since_commit > H:
                return bad()
            st = "EMPTY" if qsize == 0 else "DRAINING"
        elif ev == "queue.exhausted":
            if qsize != 0:
                return bad()
            st = "EMPTY"
        else:
            return bad()

    lifecycle_ok = (resets >= 1 and committed == refills and st in ("EMPTY", "READY", "DRAINING"))
    refill_exact = (refills == predictions and committed == predictions and popped == seq)
    return lifecycle_ok, refill_exact


def derive_checks(raw: dict) -> dict[str, bool]:
    """Re-derive the closed check set from raw recorded data (single source)."""
    predictions = math.ceil(raw["sequence_length"] / raw["horizon"])
    expected = (raw["batch_size"] * predictions) if raw["mode"] == "split_and_stack" \
        else predictions
    req_count = len(raw["request_bodies"])
    done = raw["done_mask"]
    reqs_per_refill = raw["batch_size"] if raw["mode"] == "split_and_stack" else 1
    ql, rc = _queue_lifecycle(raw.get("queue_events", []), predictions,
                             raw["sequence_length"], raw["horizon"], reqs_per_refill)
    checks = {
        "official_rollout_called": req_count > 0,
        "all_environments_reached_done": all(any(r) for r in done) and bool(done),
        "done_mask_cumulative": _done_cumulative(done),
        "done_mask_matches_terminate_at": _first_done_matches(done, raw["terminate_at"]),
        "queue_lifecycle_valid": ql,
        "queue_refill_count_exact": rc,
        "wire_payload_valid": _wire_ok(raw),
        "request_count_exact": req_count == expected,
        "response_action_chain_valid": _chain_ok(raw),
        "fixture_feature_semantics_verified": _fixture_ok(raw),
    }
    assert set(checks) == set(REQUIRED_CHECKS)
    return checks


@dataclass
class ReplayResult:
    ok: bool
    derived_checks: dict[str, bool] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def replay_rollout_evidence(case_dir: str) -> ReplayResult:
    """Re-derive checks from a case bundle's raw records and match them to the report.

    Requires ``measurements.json`` (the canonical raw data) + the readiness report.
    Fails if the derived checks/claim or recomputed hashes disagree with the report.
    """
    d = Path(case_dir)
    errors: list[str] = []
    try:
        raw = json.loads((d / "measurements.json").read_text())
        report = json.loads((d / "official_rollout_readiness_report.json").read_text())
    except Exception as exc:  # noqa: BLE001
        return ReplayResult(False, {}, [f"load: {exc}"])

    # v1.3.16 (P1.7): validate the raw schema BEFORE field access, so a malformed
    # measurements file fails the verifier instead of crashing it.
    import jsonschema
    try:
        jsonschema.validate(raw, MEASUREMENTS_SCHEMA)
    except Exception as exc:  # noqa: BLE001
        return ReplayResult(False, {}, [f"measurements schema: {exc}"])

    try:
        derived = derive_checks(raw)
    except Exception as exc:  # noqa: BLE001
        return ReplayResult(False, {}, [f"derive: {exc}"])
    reported = report.get("checks", {})
    if derived != reported:
        for k, v in derived.items():
            if reported.get(k) != v:
                errors.append(f"check {k}: report={reported.get(k)} derived={v}")
    smoke = report["claims"]["official_rollout_pipeline_smoke_passed"]
    if smoke != all(derived.values()):
        errors.append(f"smoke claim {smoke} != derived {all(derived.values())}")

    # recompute the report's request/response/final/done hashes from raw records.
    req_h = [canonical_json_sha256(b) for b in raw["request_bodies"]]
    resp_h = [canonical_json_sha256(b) for b in raw["response_bodies"]]
    if report["observation"]["ordered_request_sha256s"] != req_h:
        errors.append("ordered_request_sha256s mismatch")
    if report["action"]["ordered_response_sha256s"] != resp_h:
        errors.append("ordered_response_sha256s mismatch")
    if report["action"]["final_action_sha256"] != canonical_json_sha256(raw["final_action"]):
        errors.append("final_action_sha256 mismatch")
    if report["action"]["done_mask_sha256"] != canonical_json_sha256(
            [list(r) for r in raw["done_mask"]]):
        errors.append("done_mask_sha256 mismatch")

    return ReplayResult(not errors, derived, errors)
