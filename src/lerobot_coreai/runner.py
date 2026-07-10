# runner.py — RunnerClient for coreai-runner action inference (v0.2).
#
# Communicates with coreai-runner over HTTP (or Unix domain socket via httpx transport).
# The runner executes .aimodel graphs; this client never touches robot hardware.
#
# Error mapping (spec §12.3):
#   connection failed → RunnerNotReachableError
#   timeout           → RunnerTimeoutError
#   HTTP 400          → RunnerRequestError
#   HTTP 404          → RunnerProtocolError
#   HTTP 500          → RunnerExecutionError
#   invalid JSON      → RunnerProtocolError
#   missing action    → RunnerProtocolError

from __future__ import annotations

import json
from typing import Any

import httpx

from .errors import (
    RunnerCapabilityError,
    RunnerExecutionError,
    RunnerNotReachableError,
    RunnerProtocolError,
    RunnerRequestError,
    RunnerTimeoutError,
)
from .types import ActionPredictRequest, ActionPredictResponse, RunnerCapabilities, RunnerHealth


class RunnerClient:
    """Client for coreai-runner HTTP API.

    Supports both HTTP URLs (``http://127.0.0.1:8710``) and Unix domain sockets
    (``unix:///tmp/coreai-runner.sock``) via httpx's UDS transport.
    """

    def __init__(
        self,
        runner_url: str = "unix:///tmp/coreai-runner.sock",
        *,
        timeout_s: float = 30.0,
    ):
        self.runner_url = runner_url
        self.timeout_s = timeout_s
        self._client = self._make_client()

    def _make_client(self) -> httpx.Client:
        if self.runner_url.startswith("unix://"):
            socket_path = self.runner_url.removeprefix("unix://")
            transport = httpx.HTTPTransport(uds=socket_path)
            return httpx.Client(
                transport=transport,
                base_url="http://coreai-runner",
                timeout=self.timeout_s,
            )
        return httpx.Client(
            base_url=self.runner_url.rstrip("/"),
            timeout=self.timeout_s,
        )

    # MARK: - Health

    def health(self) -> RunnerHealth:
        """GET /v1/health — check if the runner is alive."""
        resp = self._get("v1/health")
        data = self._parse_json(resp)
        return RunnerHealth(
            status=data.get("status", "unknown"),
            raw=data,
        )

    # MARK: - Capabilities

    def capabilities(self) -> RunnerCapabilities:
        """GET /v1/capabilities — discover what the runner supports."""
        resp = self._get("v1/capabilities")
        data = self._parse_json(resp)
        supports = data.get("supports", {})
        batching = data.get("action_batching", {}) or {}
        encodings = data.get("observation_encodings") or ()
        return RunnerCapabilities(
            runtime=data.get("runtime", "coreai-runner"),
            supports_action=supports.get("action", False),
            supports_llm=supports.get("llm", False),
            supports_vlm=supports.get("vlm", False),
            supports_host_loop=supports.get("host_loop", False),
            supports_multi_graph=supports.get("multi_graph", False),
            protocol_version=data.get("protocol_version"),
            observation_encodings=tuple(encodings),
            supports_batch=bool(batching.get("supported", False)),
            max_batch_size=batching.get("max_batch_size"),
            raw=data,
        )

    def supports_action(self) -> bool:
        """Check if the runner supports runtime_kind=action."""
        caps = self.capabilities()
        if not caps.supports_action:
            raise RunnerCapabilityError(
                "coreai-runner does not support runtime_kind='action'. "
                "Ensure the runner is built with action inference support."
            )
        return True

    # MARK: - Predict (action)

    def predict_action(self, request: ActionPredictRequest) -> ActionPredictResponse:
        """POST /v1/predict with runtime_kind=action.

        Sends a LeRobot-shaped observation and returns an action chunk.
        """
        payload: dict[str, Any] = {
            "model_id": request.model_id,
            "runtime_kind": "action",
            "observation": request.observation,
            "options": request.options,
        }

        resp = self._post("v1/predict", payload)
        data = self._parse_json(resp)

        if "action" not in data:
            raise RunnerProtocolError(
                f"coreai-runner response missing 'action' field. "
                f"Got keys: {list(data.keys())}"
            )

        return ActionPredictResponse(
            action=data["action"],
            action_features=data.get("action_features", {}),
            timing=data.get("timing", {}),
            raw=data,
        )

    # MARK: - HTTP helpers

    def _get(self, path: str) -> httpx.Response:
        try:
            resp = self._client.get(path)
        except httpx.TimeoutException as e:
            raise RunnerTimeoutError(
                f"coreai-runner timed out after {self.timeout_s}s: {e}"
            ) from e
        except (httpx.ConnectError, httpx.TransportError) as e:
            raise RunnerNotReachableError(
                f"coreai-runner not reachable at {self.runner_url}: {e}"
            ) from e
        self._check_status(resp)
        return resp

    def _post(self, path: str, json_body: dict[str, Any]) -> httpx.Response:
        try:
            resp = self._client.post(path, json=json_body)
        except httpx.TimeoutException as e:
            raise RunnerTimeoutError(
                f"coreai-runner timed out after {self.timeout_s}s: {e}"
            ) from e
        except (httpx.ConnectError, httpx.TransportError) as e:
            raise RunnerNotReachableError(
                f"coreai-runner not reachable at {self.runner_url}: {e}"
            ) from e
        self._check_status(resp)
        return resp

    def _check_status(self, resp: httpx.Response) -> None:
        if resp.status_code == 400:
            detail = self._try_error_message(resp)
            raise RunnerRequestError(detail)
        if resp.status_code == 404:
            raise RunnerProtocolError(f"Endpoint not found (HTTP 404). Is the runner running?")
        if resp.status_code == 501:
            detail = self._try_error_message(resp)
            raise RunnerCapabilityError(
                f"coreai-runner does not support this operation (HTTP 501): {detail}"
            )
        if resp.status_code >= 500:
            detail = self._try_error_message(resp)
            raise RunnerExecutionError(
                f"coreai-runner execution failed (HTTP {resp.status_code}): {detail}"
            )
        if resp.status_code >= 400:
            raise RunnerProtocolError(
                f"Unexpected HTTP {resp.status_code}: {resp.text[:200]}"
            )

    def _parse_json(self, resp: httpx.Response) -> dict[str, Any]:
        try:
            return resp.json()
        except (json.JSONDecodeError, ValueError) as e:
            raise RunnerProtocolError(
                f"coreai-runner returned invalid JSON: {e}"
            ) from e

    def _try_error_message(self, resp: httpx.Response) -> str:
        try:
            data = resp.json()
            err = data.get("error", data)
            if isinstance(err, dict):
                return err.get("message", str(err))
            return str(err)
        except Exception:
            return resp.text[:200]

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
