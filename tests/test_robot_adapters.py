# test_robot_adapters.py — robot adapter protocol + mock (v1.0.0).

import pytest

from lerobot_coreai.errors import CoreAIPolicyError
from lerobot_coreai.robot_adapters import (
    MockRobotAdapter,
    RobotAdapter,
    build_robot_adapter,
)


def test_mock_adapter_lifecycle():
    a = MockRobotAdapter(robot_type="so100")
    assert isinstance(a, RobotAdapter)
    assert a.preflight()["ok"] is True
    assert a.preflight()["safe_mock"] is True
    assert not a.is_ready()          # not connected
    a.connect()
    assert a.is_ready()
    assert a.send_action([0.0])["sent"] is True
    assert len(a.actions_sent) == 1
    a.stop()
    assert not a.is_ready()          # stopped
    a.disconnect()
    assert not a.connected


def test_mock_send_action_refused_when_not_ready():
    a = MockRobotAdapter()
    with pytest.raises(CoreAIPolicyError):
        a.send_action([0.0])         # never connected


def test_build_mock():
    a = build_robot_adapter("mock", "so100")
    assert a.name == "mock"
    assert a.robot_type == "so100"


def test_build_unknown_adapter_fails_closed():
    # No hidden fallback to any robot API.
    with pytest.raises(CoreAIPolicyError, match="Unknown or unimplemented"):
        build_robot_adapter("so100", "so100")
    with pytest.raises(CoreAIPolicyError):
        build_robot_adapter("custom", "so100")


def test_external_http_requires_endpoint():
    with pytest.raises(CoreAIPolicyError, match="requires --robot.endpoint"):
        build_robot_adapter("external-http", "so100")


def test_external_http_loopback_only():
    # A remote endpoint is refused in v1.0.0.
    with pytest.raises(CoreAIPolicyError, match="loopback"):
        build_robot_adapter("external-http", "so100", endpoint="http://10.0.0.5:8765")
    # Loopback endpoints are accepted.
    a = build_robot_adapter("external-http", "so100", endpoint="http://127.0.0.1:8765")
    assert a.name == "external-http"
