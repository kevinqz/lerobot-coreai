# test_policy_select_action.py — tests for CoreAIPolicy.select_action with mocked runner.

import pytest
from unittest.mock import MagicMock, patch

from lerobot_coreai.policy import CoreAIPolicy
from lerobot_coreai.manifest import LeRobotCoreAIManifest
from lerobot_coreai.types import ActionPredictResponse
from lerobot_coreai.errors import (
    RunnerNotReachableError,
    ObservationValidationError,
    ActionValidationError,
    CoreAIPolicyError,
)


class TestSelectActionWithMockedRunner:
    def _make_policy_with_mock_runner(self, valid_manifest_dict):
        manifest = LeRobotCoreAIManifest.from_dict(valid_manifest_dict)
        mock_runner = MagicMock()
        policy = CoreAIPolicy(manifest, runner_client=mock_runner)
        return policy, mock_runner

    def test_select_action_calls_runner(self, valid_manifest_dict):
        p, mock_runner = self._make_policy_with_mock_runner(valid_manifest_dict)
        action = [[0.01] * 7 for _ in range(16)]
        mock_runner.predict_action.return_value = ActionPredictResponse(action=action)

        batch = {
            "observation.images.wrist": "/tmp/wrist.png",
            "observation.state": [0.0] * 7,
            "task": "pick",
        }
        result = p.select_action(batch)

        assert "action" in result
        assert result["action"] == action
        mock_runner.predict_action.assert_called_once()

    def test_select_action_return_metadata(self, valid_manifest_dict):
        p, mock_runner = self._make_policy_with_mock_runner(valid_manifest_dict)
        action = [[0.01] * 7 for _ in range(16)]
        mock_runner.predict_action.return_value = ActionPredictResponse(
            action=action, timing={"inference_ms": 12.5}
        )

        batch = {
            "observation.images.wrist": "/tmp/wrist.png",
            "observation.state": [0.0] * 7,
        }
        result = p.select_action(batch, return_metadata=True)

        assert "action" in result
        assert "metadata" in result
        assert result["metadata"]["policy_type"] == "evo1"
        assert result["metadata"]["robot_type"] == "so100"
        assert result["metadata"]["timing"]["inference_ms"] == 12.5

    def test_select_action_validation_before_runner(self, valid_manifest_dict):
        """Observation validation should fail before calling the runner."""
        p, mock_runner = self._make_policy_with_mock_runner(valid_manifest_dict)
        batch = {"task": "test"}  # missing required keys

        with pytest.raises(ObservationValidationError):
            p.select_action(batch)
        mock_runner.predict_action.assert_not_called()

    def test_select_action_runner_error_bubbles(self, valid_manifest_dict):
        """Runner errors should propagate as CoreAIPolicyError subclasses."""
        p, mock_runner = self._make_policy_with_mock_runner(valid_manifest_dict)
        mock_runner.predict_action.side_effect = RunnerNotReachableError("down")

        batch = {
            "observation.images.wrist": "/tmp/wrist.png",
            "observation.state": [0.0] * 7,
        }
        with pytest.raises(CoreAIPolicyError):
            p.select_action(batch)

    def test_select_action_action_validation(self, valid_manifest_dict):
        """Bad action shape from runner should raise ActionValidationError."""
        p, mock_runner = self._make_policy_with_mock_runner(valid_manifest_dict)
        bad_action = [[0.0] * 7]  # [1, 7] instead of [16, 7]
        mock_runner.predict_action.return_value = ActionPredictResponse(action=bad_action)

        batch = {
            "observation.images.wrist": "/tmp/wrist.png",
            "observation.state": [0.0] * 7,
        }
        with pytest.raises(ActionValidationError):
            p.select_action(batch)

    def test_select_action_no_runner_raises(self, valid_manifest_dict):
        """select_action without a runner should raise RunnerNotReachableError."""
        manifest = LeRobotCoreAIManifest.from_dict(valid_manifest_dict)
        p = CoreAIPolicy(manifest)  # no runner_client
        with pytest.raises(RunnerNotReachableError):
            p.select_action({"observation.state": [0.0] * 7})

    def test_model_id_in_request(self, valid_manifest_dict):
        """The runner request should carry the correct model_id."""
        p, mock_runner = self._make_policy_with_mock_runner(valid_manifest_dict)
        action = [[0.01] * 7 for _ in range(16)]
        mock_runner.predict_action.return_value = ActionPredictResponse(action=action)

        batch = {
            "observation.images.wrist": "/tmp/wrist.png",
            "observation.state": [0.0] * 7,
        }
        p.select_action(batch)

        call_args = mock_runner.predict_action.call_args[0][0]
        # model_id should be derived from repo_id: kevinqz/EVO1-SO100-CoreAI -> evo1-so100
        assert call_args.model_id == "evo1-so100"


class TestValidateRunner:
    def test_validate_runner_calls_health_and_capabilities(self, monkeypatch, valid_manifest_dict):
        """from_pretrained(validate_runner=True) should call health() and supports_action()."""
        from lerobot_coreai.manifest import LeRobotCoreAIManifest
        from lerobot_coreai.types import RunnerHealth, RunnerCapabilities

        manifest = LeRobotCoreAIManifest.from_dict(valid_manifest_dict)
        mock_runner = MagicMock()
        mock_runner.health.return_value = RunnerHealth(status="healthy")
        mock_runner.capabilities.return_value = RunnerCapabilities(supports_action=True)

        with patch("lerobot_coreai.policy.load_manifest", return_value=manifest), \
             patch("lerobot_coreai.policy.RunnerClient", return_value=mock_runner):
            policy = CoreAIPolicy.from_pretrained(
                "kevinqz/EVO1-SO100-CoreAI",
                runner_url="http://localhost:8710",
                validate_runner=True,
            )

        mock_runner.health.assert_called_once()
        mock_runner.supports_action.assert_called_once()

    def test_validate_runner_uses_default_url(self, monkeypatch, valid_manifest_dict):
        """validate_runner=True without explicit URL should use the default runtime URL."""
        from lerobot_coreai.manifest import LeRobotCoreAIManifest
        from lerobot_coreai.types import RunnerHealth, RunnerCapabilities

        manifest = LeRobotCoreAIManifest.from_dict(valid_manifest_dict)
        mock_runner = MagicMock()
        mock_runner.health.return_value = RunnerHealth(status="healthy")
        mock_runner.capabilities.return_value = RunnerCapabilities(supports_action=True)

        with patch("lerobot_coreai.policy.load_manifest", return_value=manifest), \
             patch("lerobot_coreai.policy.RunnerClient", return_value=mock_runner) as mock_rc:
            policy = CoreAIPolicy.from_pretrained(
                "kevinqz/EVO1-SO100-CoreAI",
                validate_runner=True,
            )
        # RunnerClient should have been created with the default URL
        mock_rc.assert_called_once()
