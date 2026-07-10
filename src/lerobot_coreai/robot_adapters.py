# robot_adapters.py — robot adapter protocol + implementations (v1.0.0).
#
# An adapter is the ONLY thing that can touch a robot. It is invoked exclusively
# by RealEgressGuard, only in guarded real mode, only after every gate passes.
# The built-in MockRobotAdapter touches no hardware. Real hardware adapters must
# be explicit and gated; there is no hidden fallback to any robot API.

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .errors import CoreAIPolicyError


def validate_loopback_http_endpoint(endpoint: str | None) -> str:
    """Return a normalized loopback http:// endpoint, or fail closed.

    Requires an explicit ``http://`` scheme, an explicit loopback host
    (``localhost`` or a loopback IP), and an explicit port. Rejects remote
    hosts, non-http schemes (``https``/``file``/unix sockets), ``0.0.0.0``,
    obfuscated numeric hosts, and port-less endpoints. This is the single
    external real-egress boundary — it must be unambiguous.
    """
    import ipaddress
    import urllib.parse
    if not endpoint:
        raise CoreAIPolicyError("external-http adapter requires --robot.endpoint.")
    parsed = urllib.parse.urlparse(endpoint)
    if parsed.scheme != "http":
        raise CoreAIPolicyError(
            f"external-http endpoint must be an http:// loopback URL; got scheme "
            f"{parsed.scheme!r}.")
    host = (parsed.hostname or "").lower()
    if not host:
        raise CoreAIPolicyError(
            "external-http endpoint must include an explicit loopback hostname.")
    # Explicit port required — no implicit :80. urlparse raises on a malformed
    # port, which we also treat as fail-closed.
    try:
        port = parsed.port
    except ValueError:
        raise CoreAIPolicyError(
            "external-http endpoint has a malformed port.")
    if port is None:
        raise CoreAIPolicyError(
            "external-http endpoint must include an explicit port "
            "(e.g. http://127.0.0.1:8765).")
    # Accept the literal 'localhost'; otherwise the host must parse to a loopback
    # IP (this also catches obfuscated forms like 2130706433 or 127.000.000.001
    # and rejects 0.0.0.0, which is not loopback).
    loopback = host == "localhost"
    if not loopback:
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = False
    if not loopback:
        raise CoreAIPolicyError(
            f"external-http endpoint must be loopback (127.0.0.1/localhost); "
            f"refusing host {host!r}. Run the controller locally.")
    return endpoint.rstrip("/")


def token_sha256_prefix(token: str | None, *, length: int = 8) -> str | None:
    """Return ``sha256:<prefix>`` for a token, or None. Never returns the token."""
    if not token:
        return None
    import hashlib
    return f"sha256:{hashlib.sha256(token.encode()).hexdigest()[:length]}"


@runtime_checkable
class RobotAdapter(Protocol):
    name: str
    robot_type: str

    def preflight(self) -> dict[str, Any]: ...
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def is_ready(self) -> bool: ...
    def get_observation(self) -> dict[str, Any]: ...
    def send_action(self, action: Any) -> dict[str, Any]: ...
    def stop(self) -> None: ...


class MockRobotAdapter:
    """A hardware-free adapter. Records actions; never touches a robot.

    Used for tests and for exercising the full guarded flow safely. Its
    preflight advertises `safe_mock=True` so the deadman may be disabled only
    for the mock (never for real adapters).
    """

    name = "mock"

    def __init__(self, robot_type: str = "mock", **_ignored: Any):
        self.robot_type = robot_type
        self.connected = False
        self.ready = True
        self.actions_sent: list[Any] = []
        self.stopped = False

    def preflight(self) -> dict[str, Any]:
        return {"ok": True, "adapter": "mock", "safe_mock": True,
                "robot_type": self.robot_type}

    def connect(self) -> None:
        self.connected = True
        self.stopped = False

    def disconnect(self) -> None:
        self.connected = False

    def is_ready(self) -> bool:
        return self.connected and self.ready and not self.stopped

    def get_observation(self) -> dict[str, Any]:
        return {"observation.state": [0.0], "task": "mock guarded real session"}

    def send_action(self, action: Any) -> dict[str, Any]:
        if not self.is_ready():
            raise CoreAIPolicyError("mock adapter not ready")
        self.actions_sent.append(action)
        return {"sent": True, "count": len(self.actions_sent)}

    def stop(self) -> None:
        self.stopped = True


class ExternalHttpRobotAdapter:
    """Delegates egress to an external, operator-controlled HTTP controller.

    This IS real egress and only runs behind every real-mode gate. Endpoints:
    GET /preflight, POST /connect, POST /disconnect, GET /ready,
    GET /observation, POST /action, POST /stop.
    """

    name = "external-http"

    # Loopback-only: real egress must be to a controller the operator runs on
    # the same machine, not a remote host.

    def __init__(self, robot_type: str, endpoint: str | None = None,
                 token: str | None = None, session_id: str | None = None,
                 approval_id: str | None = None, **_ignored: Any):
        import os
        self.endpoint = validate_loopback_http_endpoint(endpoint)
        self.robot_type = robot_type
        # Optional bearer token: explicit arg, else LEROBOT_COREAI_ROBOT_TOKEN.
        # Kept out of logs/reports — it is only ever sent as request headers.
        self.token = token or os.environ.get("LEROBOT_COREAI_ROBOT_TOKEN") or None
        self.session_id = session_id
        self.approval_id = approval_id

    def _headers(self, *, intent: str | None = None) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.token:
            # Send both the standard bearer header and the namespaced token
            # header the controller contract documents.
            headers["Authorization"] = f"Bearer {self.token}"
            headers["X-LeRobot-CoreAI-Token"] = self.token
        if self.session_id:
            headers["X-LeRobot-CoreAI-Session"] = self.session_id
        if self.approval_id:
            headers["X-LeRobot-CoreAI-Approval"] = self.approval_id
        if intent:
            headers["X-LeRobot-CoreAI-Intent"] = intent
        return headers

    def _req(self, method: str, path: str, payload: dict | None = None,
             *, intent: str | None = None) -> dict[str, Any]:
        import json
        import urllib.request
        url = f"{self.endpoint}{path}"
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(url, data=data, method=method,
                                     headers=self._headers(intent=intent))
        with urllib.request.urlopen(req, timeout=5.0) as resp:  # noqa: S310 (operator-controlled, loopback)
            body = resp.read().decode()
            return json.loads(body) if body else {}

    def preflight(self) -> dict[str, Any]:
        try:
            return {"ok": True, **self._req("GET", "/preflight", intent="preflight")}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def safety_state(self) -> dict[str, Any]:
        """Query the controller's /safety-state. Never sends an action."""
        try:
            return {"ok": True, **self._req("GET", "/safety-state", intent="safety-state")}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def connect(self) -> None:
        self._req("POST", "/connect")

    def disconnect(self) -> None:
        try:
            self._req("POST", "/disconnect")
        except Exception:
            pass  # best-effort on teardown

    def is_ready(self) -> bool:
        try:
            return bool(self._req("GET", "/ready").get("ready"))
        except Exception:
            return False

    def get_observation(self) -> dict[str, Any]:
        return self._req("GET", "/observation")

    def send_action(self, action: Any) -> dict[str, Any]:
        return self._req("POST", "/action", {"action": action})

    def stop(self) -> None:
        try:
            self._req("POST", "/stop")
        except Exception:
            pass  # best-effort


KNOWN_ADAPTERS = ("mock", "external-http")


def build_robot_adapter(
    name: str, robot_type: str, *, endpoint: str | None = None,
    config: Path | None = None, token: str | None = None,
    session_id: str | None = None, approval_id: str | None = None,
) -> RobotAdapter:
    """Build a robot adapter by name. Fail-closed on unknown names — there is no
    hidden fallback to any robot API."""
    if name == "mock":
        return MockRobotAdapter(robot_type=robot_type)
    if name == "external-http":
        return ExternalHttpRobotAdapter(robot_type=robot_type, endpoint=endpoint,
                                        token=token, session_id=session_id,
                                        approval_id=approval_id)
    raise CoreAIPolicyError(
        f"Unknown or unimplemented robot adapter: {name!r}. "
        f"Available: {', '.join(KNOWN_ADAPTERS)}. Native hardware adapters "
        "(e.g. so100/so101) are not built in — provide an external-http "
        "controller you operate, behind all real-mode gates."
    )
