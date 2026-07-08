# test_runner_client.py — tests for RunnerClient with mocked httpx.

import pytest
import httpx
from unittest.mock import patch, MagicMock

from lerobot_coreai.runner import RunnerClient
from lerobot_coreai.types import ActionPredictRequest
from lerobot_coreai.errors import (
    RunnerNotReachableError,
    RunnerTimeoutError,
    RunnerRequestError,
    RunnerExecutionError,
    RunnerProtocolError,
    RunnerCapabilityError,
)


def _mock_response(status_code=200, json_data=None):
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.text = str(json_data or {})
    return resp


class TestRunnerHealth:
    def test_health_success(self):
        client = RunnerClient("http://localhost:8710")
        with patch.object(client._client, "get", return_value=_mock_response(200, {"status": "healthy"})):
            health = client.health()
            assert health.status == "healthy"

    def test_health_connection_error(self):
        client = RunnerClient("http://localhost:8710")
        with patch.object(client._client, "get", side_effect=httpx.ConnectError("refused")):
            with pytest.raises(RunnerNotReachableError):
                client.health()

    def test_health_timeout(self):
        client = RunnerClient("http://localhost:8710", timeout_s=0.1)
        with patch.object(client._client, "get", side_effect=httpx.TimeoutException("timed out")):
            with pytest.raises(RunnerTimeoutError):
                client.health()


class TestRunnerCapabilities:
    def test_capabilities_supports_action(self):
        client = RunnerClient("http://localhost:8710")
        data = {"runtime": "coreai-runner", "supports": {"action": True, "llm": True}}
        with patch.object(client._client, "get", return_value=_mock_response(200, data)):
            caps = client.capabilities()
            assert caps.supports_action is True

    def test_capabilities_missing_action(self):
        client = RunnerClient("http://localhost:8710")
        data = {"runtime": "coreai-runner", "supports": {"action": False}}
        with patch.object(client._client, "get", return_value=_mock_response(200, data)):
            with pytest.raises(RunnerCapabilityError):
                client.supports_action()

    def test_supports_action_true(self):
        client = RunnerClient("http://localhost:8710")
        data = {"supports": {"action": True}}
        with patch.object(client._client, "get", return_value=_mock_response(200, data)):
            assert client.supports_action() is True


class TestRunnerPredictAction:
    def test_predict_action_success(self):
        client = RunnerClient("http://localhost:8710")
        action = [[0.01, 0.02, 0.03, 0.0, 0.0, 0.0, 0.0]]
        data = {"model_id": "evo1-so100", "action": action, "action_features": {"shape": [16, 7]}}
        with patch.object(client._client, "post", return_value=_mock_response(200, data)):
            req = ActionPredictRequest(model_id="evo1-so100", observation={"task": "test"})
            resp = client.predict_action(req)
            assert resp.action == action

    def test_predict_action_missing_action_field(self):
        client = RunnerClient("http://localhost:8710")
        data = {"model_id": "evo1-so100"}  # no "action" key
        with patch.object(client._client, "post", return_value=_mock_response(200, data)):
            req = ActionPredictRequest(model_id="evo1-so100")
            with pytest.raises(RunnerProtocolError, match="missing 'action'"):
                client.predict_action(req)

    def test_predict_action_http_400(self):
        client = RunnerClient("http://localhost:8710")
        err_data = {"error": {"message": "bad model id"}}
        with patch.object(client._client, "post", return_value=_mock_response(400, err_data)):
            req = ActionPredictRequest(model_id="bad-id")
            with pytest.raises(RunnerRequestError, match="bad model id"):
                client.predict_action(req)

    def test_predict_action_http_500(self):
        client = RunnerClient("http://localhost:8710")
        err_data = {"error": {"message": "model crashed"}}
        with patch.object(client._client, "post", return_value=_mock_response(500, err_data)):
            req = ActionPredictRequest(model_id="evo1-so100")
            with pytest.raises(RunnerExecutionError):
                client.predict_action(req)

    def test_predict_action_http_501_capability(self):
        client = RunnerClient("http://localhost:8710")
        err_data = {"error": {"message": "action runtime not available"}}
        with patch.object(client._client, "post", return_value=_mock_response(501, err_data)):
            req = ActionPredictRequest(model_id="evo1-so100")
            with pytest.raises(RunnerCapabilityError):
                client.predict_action(req)

    def test_predict_action_connection_error(self):
        client = RunnerClient("http://localhost:8710")
        with patch.object(client._client, "post", side_effect=httpx.ConnectError("refused")):
            req = ActionPredictRequest(model_id="evo1-so100")
            with pytest.raises(RunnerNotReachableError):
                client.predict_action(req)

    def test_predict_action_timeout(self):
        client = RunnerClient("http://localhost:8710", timeout_s=0.1)
        with patch.object(client._client, "post", side_effect=httpx.TimeoutException("timed out")):
            req = ActionPredictRequest(model_id="evo1-so100")
            with pytest.raises(RunnerTimeoutError):
                client.predict_action(req)

    def test_predict_action_timing_included(self):
        client = RunnerClient("http://localhost:8710")
        timing = {"inference_ms": 12.5, "total_ms": 14.0}
        data = {"action": [[0.0]], "timing": timing}
        with patch.object(client._client, "post", return_value=_mock_response(200, data)):
            req = ActionPredictRequest(model_id="evo1-so100")
            resp = client.predict_action(req)
            assert resp.timing["inference_ms"] == 12.5
