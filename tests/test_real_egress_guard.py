# test_real_egress_guard.py — the guarded real-egress gate (v1.0.0).

from dataclasses import dataclass

from lerobot_coreai.real_egress import DeadmanSwitch, RateLimiter, RealEgressGuard
from lerobot_coreai.robot_adapters import MockRobotAdapter
from lerobot_coreai.safety_profiles import SafetyProfile
from lerobot_coreai.safety_supervisor import SafetyContext, SafetySupervisor


@dataclass
class _Session:
    status: str = "running"


def _profile(**over):
    base = dict(name="real", require_robot_type_match=False, require_known_shape=False)
    base.update(over)
    return SafetyProfile(**base)


def _guard(mode_profile=None, deadman_ok=True, fps=1000.0, adapter=None,
           session_status="running"):
    adapter = adapter or MockRobotAdapter()
    adapter.connect()
    sup = SafetySupervisor(mode_profile or _profile(), mode="enforce")
    deadman = DeadmanSwitch(timeout_s=1.0, enabled=True)
    if deadman_ok:
        deadman.heartbeat(now=100.0)
    rl = RateLimiter(fps=fps)
    guard = RealEgressGuard(sup, adapter, _Session(status=session_status), deadman, rl)
    return guard, adapter


def _ctx(step=0):
    return SafetyContext(mode="real", step=step, robot_type="so100")


def test_allowed_action_reaches_adapter():
    guard, adapter = _guard()
    d = guard.send_action([0.0, 0.1], _ctx(), now=100.0)
    assert d.allowed and d.sent
    assert adapter.actions_sent == [[0.0, 0.1]]


def test_blocked_action_never_reaches_adapter():
    # NaN → supervisor blocks → adapter.send_action must NOT be called.
    guard, adapter = _guard()
    d = guard.send_action([float("nan")], _ctx(), now=100.0)
    assert not d.allowed
    assert d.reason == "supervisor_blocked"
    assert adapter.actions_sent == []


def test_session_not_running_blocks():
    guard, adapter = _guard(session_status="stopped")
    d = guard.send_action([0.0], _ctx(), now=100.0)
    assert not d.allowed and d.reason == "session_not_running"
    assert adapter.actions_sent == []


def test_estop_blocks():
    guard, adapter = _guard()
    guard.trigger_estop("test")
    d = guard.send_action([0.0], _ctx(), now=100.0)
    assert not d.allowed and d.reason == "estop_triggered"
    assert adapter.actions_sent == []


def test_deadman_unhealthy_blocks():
    guard, adapter = _guard()
    # now far beyond the last heartbeat (100.0) + timeout (1.0)
    d = guard.send_action([0.0], _ctx(), now=200.0)
    assert not d.allowed and d.reason == "deadman_unhealthy"
    assert guard.deadman_lost is True
    assert adapter.actions_sent == []


def test_rate_limit_blocks_second_immediate_send():
    guard, adapter = _guard(fps=1.0)
    d1 = guard.send_action([0.0], _ctx(0), now=100.0)
    d2 = guard.send_action([0.0], _ctx(1), now=100.1)  # < 1s later
    assert d1.allowed
    assert not d2.allowed and d2.reason == "rate_limited"
    assert len(adapter.actions_sent) == 1


def test_disabled_deadman_blocks_without_authorization():
    # A disabled deadman handed to the guard without explicit authorization blocks.
    adapter = MockRobotAdapter(); adapter.connect()
    sup = SafetySupervisor(_profile(), mode="enforce")
    deadman = DeadmanSwitch(timeout_s=1.0, enabled=False)
    guard = RealEgressGuard(sup, adapter, _Session(), deadman, RateLimiter(1000.0))
    d = guard.send_action([0.0], _ctx(), now=100.0)
    assert not d.allowed and d.reason == "deadman_disabled_not_allowed"
    assert adapter.actions_sent == []


def test_disabled_deadman_allowed_when_authorized():
    adapter = MockRobotAdapter(); adapter.connect()
    sup = SafetySupervisor(_profile(), mode="enforce")
    deadman = DeadmanSwitch(timeout_s=1.0, enabled=False)
    guard = RealEgressGuard(sup, adapter, _Session(), deadman, RateLimiter(1000.0),
                            allow_disabled_deadman=True)
    d = guard.send_action([0.0], _ctx(), now=100.0)
    assert d.allowed and d.sent


def test_adapter_not_ready_blocks():
    adapter = MockRobotAdapter()
    adapter.connect()
    adapter.stop()  # not ready
    sup = SafetySupervisor(_profile(), mode="enforce")
    deadman = DeadmanSwitch(timeout_s=1.0)
    deadman.heartbeat(now=100.0)
    guard = RealEgressGuard(sup, adapter, _Session(), deadman, RateLimiter(1000.0))
    d = guard.send_action([0.0], _ctx(), now=100.0)
    assert not d.allowed and d.reason == "adapter_not_ready"


def test_adapter_exception_blocks_and_records_error():
    class Boom(MockRobotAdapter):
        def send_action(self, action):
            raise RuntimeError("hardware fault")
    adapter = Boom()
    adapter.connect()
    sup = SafetySupervisor(_profile(), mode="enforce")
    deadman = DeadmanSwitch(timeout_s=1.0)
    deadman.heartbeat(now=100.0)
    guard = RealEgressGuard(sup, adapter, _Session(), deadman, RateLimiter(1000.0))
    d = guard.send_action([0.0], _ctx(), now=100.0)
    assert not d.allowed
    assert "adapter_error" in d.reason
    assert guard.adapter_error is not None
