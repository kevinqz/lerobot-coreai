# test_robot_gateway.py — Robot Gateway reference (RFC-0700 §19.1 / RFC-0900 §13, LR10).
# Every safety gate fails closed; dry-run sends nothing; the receipt proves no physical
# safety. Pure base package.

import pytest

from lerobot_coreai.robot_gateway import (
    DryRunEgress, GatewayReject, ReferenceRobotGateway, SafetyProfile,
)

_H = "sha256:" + "a" * 64


def _session_open(mode="dry_run"):
    return {"type": "session-open", "session_id": "S1", "robot_identity": "so101-A",
            "policy_artifact_root": _H, "mode": mode}


def _chunk(seq=0, deadline=2_000_000_000, actions=None):
    return {"type": "action-chunk", "session_id": "S1", "sequence": seq,
            "policy_artifact_root": _H, "action_representation": "joint_position_delta",
            "action_version": "v1", "issued_monotonic_ns": 1000,
            "deadline_monotonic_ns": deadline, "robot_identity": "so101-A",
            "actions": actions or {"__array__": [[0.0, 0.1, -0.1, 0.2, -0.2, 0.0]],
                                   "dtype": "float32", "shape": [1, 6]}}


def _gw(mode="dry_run", **safety):
    g = ReferenceRobotGateway("so101-A", _H, safety=SafetyProfile(**safety),
                              egress=DryRunEgress())
    g.authenticate(True)
    g.open_session(_session_open(mode))
    return g


def test_dry_run_accepts_but_sends_nothing():
    g = _gw("dry_run")
    assert g.submit(_chunk(0), now_ns=1000) == "accepted"
    assert g.submit(_chunk(1), now_ns=2000) == "accepted"
    assert g.accepted == 2 and g.executed == 0 and g.egress.sent == 0   # nothing egressed


def test_guarded_mode_executes_via_egress():
    g = _gw("guarded")
    assert g.submit(_chunk(0), now_ns=1000) == "executed"
    assert g.executed == 1 and g.egress.sent == 1


def test_unauthenticated_open_rejected():
    g = ReferenceRobotGateway("so101-A", _H)
    with pytest.raises(GatewayReject):
        g.open_session(_session_open())


def test_replay_and_out_of_order_rejected():
    g = _gw()
    g.submit(_chunk(0), now_ns=1000)
    with pytest.raises(GatewayReject):
        g.submit(_chunk(0), now_ns=1100)          # replay
    with pytest.raises(GatewayReject):
        g.submit(_chunk(9), now_ns=1200)          # out of order


def test_expired_chunk_rejected():
    g = _gw()
    with pytest.raises(GatewayReject):
        g.submit(_chunk(0, deadline=500), now_ns=1000)


def test_identity_and_policy_mismatch_rejected():
    g = _gw()
    bad = _chunk(0); bad["robot_identity"] = "other"
    with pytest.raises(GatewayReject):
        g.submit(bad, now_ns=1000)
    bad2 = _chunk(0); bad2["policy_artifact_root"] = "sha256:" + "b" * 64
    with pytest.raises(GatewayReject):
        g.submit(bad2, now_ns=1000)


def test_missing_required_field_rejected():
    g = _gw()
    bad = _chunk(0); del bad["sequence"]
    with pytest.raises(GatewayReject):
        g.submit(bad, now_ns=1000)


def test_nonfinite_and_out_of_bounds_actions_rejected():
    g = _gw(max_abs_action=1.0)
    inf = _chunk(0, actions={"__array__": [[float("inf")]], "dtype": "float32", "shape": [1, 1]})
    with pytest.raises(GatewayReject):
        g.submit(inf, now_ns=1000)
    big = _chunk(0, actions={"__array__": [[5.0]], "dtype": "float32", "shape": [1, 1]})
    with pytest.raises(GatewayReject):
        g.submit(big, now_ns=1000)


def test_watchdog_stops_on_gap():
    g = _gw(watchdog_ns=1000)
    g.submit(_chunk(0), now_ns=1000)
    with pytest.raises(GatewayReject):
        g.submit(_chunk(1), now_ns=1_000_000)     # huge gap → watchdog stop
    assert g.stopped


def test_stop_then_no_actuation_and_receipt_is_honest():
    g = _gw("guarded")
    g.submit(_chunk(0), now_ns=1000)
    g.stop("operator_stop")
    with pytest.raises(GatewayReject):
        g.submit(_chunk(1), now_ns=1100)
    r = g.receipt()
    assert r["proves_physical_safety"] is False
    assert r["stopped_reason"] == "operator_stop"
    assert r["executed"] == 1 and r["rejected"] >= 1
