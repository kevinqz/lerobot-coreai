# test_app_conformance.py — Apple-app conformance reference (RFC-0700 §19.1 / RFC-0900
# §11.3, LR12). Drives the published conformance/app fixtures through the reference
# ActionQueue + planner handoff; the Swift LeRobotCoreAIKit must reproduce these. Base pkg.

import json
from pathlib import Path

import pytest

from lerobot_coreai.app_conformance import (
    INVALIDATION_REASONS, ActionQueueEmpty, ChunkInvalidated, QueueContext,
    ReferenceActionQueue, skill_plan_to_policy_tasks,
)

_FIX = Path(__file__).resolve().parents[1] / "conformance" / "app"
_H = "sha256:" + "a" * 64
_CTX = QueueContext("S1", _H, "sha256:" + "c" * 64)


def _load(name):
    return json.loads((_FIX / name).read_text())


def test_published_fixtures_exist():
    for f in ("policy-manifest.json", "skill-plan.json", "action-queue-cases.json"):
        assert (_FIX / f).exists()


def test_action_queue_in_order_then_empty():
    cases = _load("action-queue-cases.json")
    q = ReferenceActionQueue()
    q.replace(cases["chunk"], _CTX, deadline_ns=cases["deadline_ns"])
    got = [q.next(now_ns=0), q.next(now_ns=0), q.next(now_ns=0)]
    assert got[-1] == cases["chunk"][-1]
    with pytest.raises(ActionQueueEmpty):
        q.next(now_ns=0)


def test_action_queue_deadline_invalidates():
    q = ReferenceActionQueue()
    q.replace([[0.0]], _CTX, deadline_ns=1000)
    with pytest.raises(ChunkInvalidated):
        q.next(now_ns=2000)                       # now > deadline


@pytest.mark.parametrize("field,reason", [
    ("session_id", "session_change"),
    ("policy_artifact_root", "policy_artifact_change"),
    ("observation_contract_sha", "observation_contract_change"),
])
def test_context_change_invalidates_chunk(field, reason):
    q = ReferenceActionQueue()
    q.replace([[0.0], [0.1]], _CTX, deadline_ns=10_000)
    changed = {"session_id": _CTX.session_id, "policy_artifact_root": _CTX.policy_artifact_root,
               "observation_contract_sha": _CTX.observation_contract_sha}
    changed[field] = "DIFFERENT"
    q.observe(QueueContext(**changed))
    with pytest.raises(ChunkInvalidated) as e:
        q.next(now_ns=0)
    assert reason in str(e.value)


def test_stop_and_fault_are_valid_invalidations():
    for reason in ("stop", "fault", "gateway_reset"):
        q = ReferenceActionQueue()
        q.replace([[0.0]], _CTX, deadline_ns=10_000)
        q.clear(reason)
        with pytest.raises(ChunkInvalidated):
            q.next(now_ns=0)
    assert set(("stop", "fault", "gateway_reset")) <= set(INVALIDATION_REASONS)


def test_skill_plan_handoff_produces_tasks_no_motor_commands():
    plan = _load("skill-plan.json")
    tasks = skill_plan_to_policy_tasks(plan)
    assert tasks[0].startswith("pick_and_place")
    assert "red_cube" in tasks[0] and "tray" in tasks[0]
    assert tasks[1] == "home"


def test_planner_emitting_motor_argument_is_refused():
    bad = {"schema": "org.lerobot.robot-brain.skill-plan.v1", "plan_id": "p",
           "steps": [{"skill": "move", "arguments": {"joint_positions": [0, 1, 2]}}],
           "requires_operator_confirmation": False}
    with pytest.raises(ValueError):
        skill_plan_to_policy_tasks(bad)


def test_handoff_rejects_wrong_schema():
    with pytest.raises(ValueError):
        skill_plan_to_policy_tasks({"schema": "something.else", "steps": []})
