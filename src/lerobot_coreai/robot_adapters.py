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

    # Loopback-only in v1.0.0: real egress must be to a controller the operator
    # runs on the same machine, not a remote host.
    _LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", ""}

    def __init__(self, robot_type: str, endpoint: str | None = None,
                 token: str | None = None, **_ignored: Any):
        if not endpoint:
            raise CoreAIPolicyError(
                "external-http adapter requires --robot.endpoint.")
        import os
        import urllib.parse
        host = urllib.parse.urlparse(endpoint).hostname or ""
        if host.lower() not in self._LOOPBACK_HOSTS:
            raise CoreAIPolicyError(
                f"external-http endpoint must be loopback (127.0.0.1/localhost) in "
                f"v1.0.x; refusing remote host {host!r}. Run the controller locally."
            )
        self.robot_type = robot_type
        self.endpoint = endpoint.rstrip("/")
        # Optional bearer token: explicit arg, else LEROBOT_COREAI_ROBOT_TOKEN.
        # Kept out of logs/reports — it is only ever sent as an Authorization header.
        self.token = token or os.environ.get("LEROBOT_COREAI_ROBOT_TOKEN") or None

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _req(self, method: str, path: str, payload: dict | None = None) -> dict[str, Any]:
        import json
        import urllib.request
        url = f"{self.endpoint}{path}"
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(url, data=data, method=method, headers=self._headers())
        with urllib.request.urlopen(req, timeout=5.0) as resp:  # noqa: S310 (operator-controlled, loopback)
            body = resp.read().decode()
            return json.loads(body) if body else {}

    def preflight(self) -> dict[str, Any]:
        try:
            return {"ok": True, **self._req("GET", "/preflight")}
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
) -> RobotAdapter:
    """Build a robot adapter by name. Fail-closed on unknown names — there is no
    hidden fallback to any robot API."""
    if name == "mock":
        return MockRobotAdapter(robot_type=robot_type)
    if name == "external-http":
        return ExternalHttpRobotAdapter(robot_type=robot_type, endpoint=endpoint,
                                        token=token)
    raise CoreAIPolicyError(
        f"Unknown or unimplemented robot adapter: {name!r}. "
        f"Available: {', '.join(KNOWN_ADAPTERS)}. Native hardware adapters "
        "(e.g. so100/so101) are not built in — provide an external-http "
        "controller you operate, behind all real-mode gates."
    )
