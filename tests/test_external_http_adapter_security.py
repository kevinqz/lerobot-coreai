# test_external_http_adapter_security.py — external-http URL + auth hardening (v1.1.1).

import pytest

from lerobot_coreai.errors import CoreAIPolicyError
from lerobot_coreai.robot_adapters import (
    ExternalHttpRobotAdapter, token_sha256_prefix, validate_loopback_http_endpoint,
)


@pytest.mark.parametrize("endpoint", [
    "http://127.0.0.1:8765",
    "http://localhost:8765",
    "http://[::1]:8765",
    "http://127.0.0.1:80",
])
def test_valid_loopback_endpoints_pass(endpoint):
    assert validate_loopback_http_endpoint(endpoint).startswith("http://")


@pytest.mark.parametrize("endpoint,reason", [
    ("http://192.168.0.10:8765", "remote host"),
    ("http://example.com:8765", "remote host"),
    ("https://127.0.0.1:8765", "https scheme"),
    ("file:///tmp/x", "file scheme"),
    ("http://127.0.0.1", "missing port"),
    ("http://localhost", "missing port"),
    ("http://0.0.0.0:8765", "not loopback"),
    ("http://2130706433:8765", "obfuscated numeric host"),
    ("http://:8765", "empty host"),
    ("", "empty endpoint"),
])
def test_non_loopback_or_malformed_endpoints_fail(endpoint, reason):
    with pytest.raises(CoreAIPolicyError):
        validate_loopback_http_endpoint(endpoint)


def test_adapter_rejects_remote_endpoint():
    with pytest.raises(CoreAIPolicyError):
        ExternalHttpRobotAdapter("so100", endpoint="http://192.168.1.5:8765")


def test_adapter_requires_explicit_port():
    with pytest.raises(CoreAIPolicyError):
        ExternalHttpRobotAdapter("so100", endpoint="http://127.0.0.1")


def test_headers_include_token_session_approval():
    a = ExternalHttpRobotAdapter(
        "so100", endpoint="http://127.0.0.1:8765", token="secret-token",
        session_id="real_abc", approval_id="appr_123")
    h = a._headers()
    assert h["Authorization"] == "Bearer secret-token"
    assert h["X-LeRobot-CoreAI-Token"] == "secret-token"
    assert h["X-LeRobot-CoreAI-Session"] == "real_abc"
    assert h["X-LeRobot-CoreAI-Approval"] == "appr_123"


def test_preflight_intent_header():
    a = ExternalHttpRobotAdapter("so100", endpoint="http://127.0.0.1:8765")
    assert a._headers(intent="preflight")["X-LeRobot-CoreAI-Intent"] == "preflight"


def test_no_token_means_no_auth_headers():
    a = ExternalHttpRobotAdapter("so100", endpoint="http://127.0.0.1:8765")
    h = a._headers()
    assert "Authorization" not in h
    assert "X-LeRobot-CoreAI-Token" not in h


def test_token_sha256_prefix_never_reveals_token():
    prefix = token_sha256_prefix("super-secret-token")
    assert prefix.startswith("sha256:")
    assert "super-secret-token" not in prefix
    assert token_sha256_prefix(None) is None


def test_env_token_picked_up(monkeypatch):
    monkeypatch.setenv("LEROBOT_COREAI_ROBOT_TOKEN", "env-token")
    a = ExternalHttpRobotAdapter("so100", endpoint="http://127.0.0.1:8765")
    assert a.token == "env-token"
    assert a._headers()["X-LeRobot-CoreAI-Token"] == "env-token"
