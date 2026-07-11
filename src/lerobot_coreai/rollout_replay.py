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
    MEASUREMENTS_SCHEMA, NEGOTIATION_SCHEMA, REQUIRED_CHECKS, canonical_json_sha256,
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
    neg = raw.get("negotiation")
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
        if not isinstance(opts.get("protocol_version"), str):
            return False
        # v1.3.19 (P1.14): if a NegotiationRecord is persisted, the request options
        # MUST equal the negotiated result — no hardcoded allowlist. Otherwise fall
        # back to the known-encoding allowlist (legacy bundles).
        if neg is not None:
            if opts.get("observation_encoding") != neg["negotiated_encoding"]:
                return False
            if opts.get("protocol_version") != neg["negotiated_protocol"]:
                return False
        elif opts.get("observation_encoding") not in ("nested_json_v1",
                                                       "typed_array_envelope_v1"):
            return False
    return True


def _negotiation_ok(raw) -> bool:
    """Validate the persisted NegotiationRecord is schema-valid + self-consistent."""
    neg = raw.get("negotiation")
    if neg is None:
        return True                          # optional (legacy bundles)
    import jsonschema
    try:
        jsonschema.validate(neg, NEGOTIATION_SCHEMA)
    except Exception:  # noqa: BLE001
        return False
    # negotiated protocol/encoding must be an option the runner announced.
    if neg["negotiated_encoding"] not in neg["runner_encodings"]:
        return False
    # record_sha256 must recompute from the negotiated content (tamper-evident).
    body = {k: v for k, v in neg.items() if k not in ("record_sha256",)}
    return neg["record_sha256"] == canonical_json_sha256(body)


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


def _pred_response_hashes(raw, pred: int) -> list[str]:
    """The response-body hashes the runner returned for prediction ``pred``."""
    B = raw["batch_size"]
    if raw["mode"] == "split_and_stack":
        idxs = range(pred * B, pred * B + B)
    else:
        idxs = [pred]
    bodies = raw["response_bodies"]
    return [canonical_json_sha256(bodies[i]) for i in idxs if 0 <= i < len(bodies)]


def _selected_action_hash(raw, step: int) -> str | None:
    """canonical hash of the [B, A] action recorded at rollout timestep ``step``."""
    final = raw["final_action"]                 # [B, seq, A]
    B, seq = raw["batch_size"], raw["sequence_length"]
    if _per_sample_shape(final) != [B, seq, raw["action_dim"]] or step >= seq:
        return None
    try:
        return canonical_json_sha256([final[b][step] for b in range(B)])
    except (IndexError, TypeError):
        return None


def _execution_lifecycle(raw) -> tuple[bool, bool]:
    """State-machine replay of Execution Trace Protocol v3 (v1.3.19). Fail-closed.

    Proves, over and above v1.3.18: a discriminated per-event schema, monotonic
    timestamps, constant execution_id, unique request/action IDs with exact
    request->response pairing, ordered_response_sha256s bound to the recomputed
    response bodies of that prediction (P1.3/P1.4), each popped action's hash bound
    to the final_action slice at its rollout_step (P1.5), a mandatory queue.exhausted
    after the queue empties (P1.11), and normal/abort reset semantics (P1.10).
    Returns (lifecycle_valid, refill_count_exact).
    """
    from .rollout_evidence_schema import EXECUTION_EVENT_SCHEMA
    import jsonschema
    events = raw.get("queue_events", [])
    H = raw["horizon"]
    seq = raw["sequence_length"]
    predictions = math.ceil(seq / H)
    reqs_per_refill = raw["batch_size"] if raw["mode"] == "split_and_stack" else 1
    if not events:
        return False, False
    try:
        for e in events:
            jsonschema.validate(e, EXECUTION_EVENT_SCHEMA)
    except Exception:  # noqa: BLE001
        return False, False
    if [e["event_index"] for e in events] != list(range(len(events))):
        return False, False
    if events[0]["event"] != "execution.started" \
            or events[-1]["event"] != "execution.completed":
        return False, False
    exec_id = events[0]["execution_id"]
    if any(e["execution_id"] != exec_id for e in events):     # constancy (P1.6)
        return False, False
    ts = [e["relative_monotonic_ns"] for e in events]
    if any(b < a for a, b in zip(ts, ts[1:])):                # monotonic (P1.7)
        return False, False

    st = "STARTED"
    committed = popped = refills = 0
    qsize = 0
    validated_hash = None
    active_pid = active_cid = None
    seen_pids: set = set()
    seen_reqs: set = set()
    open_reqs: set = set()
    seen_actions: set = set()
    resp_hashes: list = []
    sample_seen: set = set()
    pops_since_commit = 0

    def bad():
        return False, False

    for e in events[1:-1]:
        ev = e["event"]
        if ev == "policy.reset":
            kind = e["reset_kind"]
            if kind == "normal":
                if st not in ("STARTED", "EMPTY") or qsize != 0:
                    return bad()               # normal reset only when already empty
            else:                               # abort: may discard a partial queue
                if e.get("discarded_action_count") != qsize \
                        or "discarded_queue_sha256" not in e:
                    return bad()
            qsize = 0
            st = "EMPTY"
        elif ev == "queue.empty":
            if e["queue_size_after"] != 0 or qsize != 0:
                return bad()
            st = "EMPTY"
        elif ev == "queue.refill_requested":
            if st != "EMPTY" or qsize != 0 \
                    or e["queue_size_before"] != 0 or e["queue_size_after"] != 0:
                return bad()
            pid, cid = e["prediction_id"], e["chunk_id"]
            if pid in seen_pids:                # never reused (P1.1)
                return bad()
            active_pid, active_cid = pid, cid
            seen_pids.add(pid)
            refills += 1
            resp_hashes = []
            sample_seen = set()
            validated_hash = None
            st = "REFILL"
        elif ev == "runner.request_started":
            if st not in ("REFILL", "RESP") \
                    or e["prediction_id"] != active_pid or e["chunk_id"] != active_cid:
                return bad()
            rid = e["request_id"]
            if rid in seen_reqs:                # unique request id (P1.2)
                return bad()
            si = e.get("sample_index")
            if reqs_per_refill > 1:             # split: distinct 0..B-1
                if si is None or si in sample_seen:
                    return bad()
                sample_seen.add(si)
            elif si is not None:                # native/single: no sample index
                return bad()
            seen_reqs.add(rid)
            open_reqs.add(rid)
            st = "REQ"
        elif ev == "runner.response_received":
            rid = e["request_id"]
            if st != "REQ" or e["prediction_id"] != active_pid \
                    or rid not in open_reqs:    # exactly one response per open request
                return bad()
            open_reqs.discard(rid)
            resp_hashes.append(e["response_sha256"])
            st = "RESP"
        elif ev == "chunk.validated":
            if st != "RESP" or e["prediction_id"] != active_pid \
                    or len(resp_hashes) != reqs_per_refill or open_reqs:
                return bad()
            if e["horizon"] != H:
                return bad()
            # ordered response hashes bound to the recomputed response bodies (F).
            if e["ordered_response_sha256s"] != resp_hashes \
                    or resp_hashes != _pred_response_hashes(raw, active_pid):
                return bad()
            validated_hash = e["chunk_sha256"]
            st = "VALIDATED"
        elif ev == "chunk.committed":
            if st != "VALIDATED" or e["prediction_id"] != active_pid \
                    or e["chunk_sha256"] != validated_hash:  # committed == validated
                return bad()
            if e["queue_size_before"] != qsize or e["queue_size_after"] != qsize + H \
                    or e["committed"] != H:
                return bad()                     # atomic commit before->before+H
            qsize += H
            committed += 1
            pops_since_commit = 0
            st = "READY"
        elif ev == "action.popped":
            if st not in ("READY", "DRAINING") or committed == 0:
                return bad()
            if e["prediction_id"] != active_pid or e["chunk_id"] != active_cid:
                return bad()                     # pop attributed to its own chunk
            if e["queue_size_before"] != qsize or e["queue_size_after"] != qsize - 1:
                return bad()
            step = e["rollout_step"]
            if step != popped or e["chunk_timestep"] != pops_since_commit:
                return bad()                     # global + within-chunk step order
            aid = e["action_id"]
            if aid in seen_actions:              # action id never reused (P1.2)
                return bad()
            seen_actions.add(aid)
            # selected action hash bound to the final_action slice (P1.5).
            if e["selected_action_sha256"] != _selected_action_hash(raw, step):
                return bad()
            qsize -= 1
            popped += 1
            pops_since_commit += 1
            if pops_since_commit > H:
                return bad()
            st = "AWAITING_EXHAUSTED" if qsize == 0 else "DRAINING"
        elif ev == "queue.exhausted":
            if st != "AWAITING_EXHAUSTED" or qsize != 0 or e["queue_size_after"] != 0:
                return bad()                     # mandatory after emptying (P1.11)
            st = "EMPTY"
        else:
            return bad()

    # completion (execution.completed) terminal-state + cached-action accounting.
    done = events[-1]
    if st not in ("EMPTY", "DRAINING"):
        return bad()
    unused = done.get("unused_action_count")
    if st == "DRAINING":                          # actions left in the queue
        if unused != qsize or qsize == 0 or "termination_reason" not in done:
            return bad()
    elif unused not in (None, 0):
        return bad()

    lifecycle_ok = committed == refills and open_reqs == set()
    refill_exact = (refills == predictions and committed == predictions
                    and popped == seq)
    return lifecycle_ok, refill_exact


def derive_checks(raw: dict) -> dict[str, bool]:
    """Re-derive the closed check set from raw recorded data (single source)."""
    predictions = math.ceil(raw["sequence_length"] / raw["horizon"])
    expected = (raw["batch_size"] * predictions) if raw["mode"] == "split_and_stack" \
        else predictions
    req_count = len(raw["request_bodies"])
    done = raw["done_mask"]
    ql, rc = _execution_lifecycle(raw)
    checks = {
        "official_rollout_called": req_count > 0,
        "all_environments_reached_done": all(any(r) for r in done) and bool(done),
        "done_mask_cumulative": _done_cumulative(done),
        "done_mask_matches_terminate_at": _first_done_matches(done, raw["terminate_at"]),
        "queue_lifecycle_valid": ql and _negotiation_ok(raw),
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

    # v1.3.19: the report's execution envelope must bind the persisted negotiation
    # record + execution id (fail if the report claims a different negotiation).
    neg = raw.get("negotiation")
    execu = report.get("execution", {})
    if neg is not None and execu.get("negotiation_sha256") != neg.get("record_sha256"):
        errors.append("execution.negotiation_sha256 != persisted negotiation record")
    events = raw.get("queue_events", [])
    if events and execu.get("execution_id") != events[0].get("execution_id"):
        errors.append("execution.execution_id != trace execution id")

    return ReplayResult(not errors, derived, errors)
