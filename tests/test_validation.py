# test_validation.py — tests for manifest-based observation/action validation.

import math
import pytest

from lerobot_coreai.manifest import LeRobotCoreAIManifest
from lerobot_coreai.validation import (
    validate_observation_batch,
    validate_action_output,
    validate_robot_type,
)
from lerobot_coreai.errors import ObservationValidationError, ActionValidationError


@pytest.fixture
def manifest(valid_manifest_dict):
    return LeRobotCoreAIManifest.from_dict(valid_manifest_dict)


class TestObservationValidation:
    def test_valid_observation_passes(self, manifest):
        batch = {
            "observation.images.wrist": "/tmp/wrist.png",
            "observation.state": [0.0, 0.1, 0.2, 0.0, 0.0, 0.0, 0.0],
            "task": "pick up the cube",
        }
        validate_observation_batch(batch, manifest)

    def test_missing_required_key_fails(self, manifest):
        batch = {
            "observation.state": [0.0, 0.1, 0.2, 0.0, 0.0, 0.0, 0.0],
            # missing observation.images.wrist
        }
        with pytest.raises(ObservationValidationError, match="Missing required"):
            validate_observation_batch(batch, manifest)

    def test_optional_task_missing_passes(self, manifest):
        batch = {
            "observation.images.wrist": "/tmp/wrist.png",
            "observation.state": [0.0] * 7,
        }
        validate_observation_batch(batch, manifest)  # should not raise

    def test_wrong_state_shape_fails(self, manifest):
        batch = {
            "observation.images.wrist": "/tmp/wrist.png",
            "observation.state": [0.0] * 8,  # expected 7
        }
        with pytest.raises(ObservationValidationError, match="shape mismatch"):
            validate_observation_batch(batch, manifest)

    def test_unknown_key_non_strict_passes(self, manifest):
        batch = {
            "observation.images.wrist": "/tmp/wrist.png",
            "observation.state": [0.0] * 7,
            "observation.images.front": "/tmp/front.png",  # unknown
        }
        validate_observation_batch(batch, manifest)  # should not raise

    def test_unknown_key_strict_fails(self, manifest):
        batch = {
            "observation.images.wrist": "/tmp/wrist.png",
            "observation.state": [0.0] * 7,
            "observation.images.front": "/tmp/front.png",
        }
        with pytest.raises(ObservationValidationError, match="Unknown observation keys"):
            validate_observation_batch(batch, manifest, strict_observation_keys=True)

    def test_task_not_string_fails(self, manifest):
        batch = {
            "observation.images.wrist": "/tmp/wrist.png",
            "observation.state": [0.0] * 7,
            "task": 12345,  # not a string
        }
        with pytest.raises(ObservationValidationError, match="must be a string"):
            validate_observation_batch(batch, manifest)


class TestActionValidation:
    def test_valid_action_passes(self, manifest):
        action = [[0.01] * 7 for _ in range(16)]  # [16, 7]
        validate_action_output(action, manifest)

    def test_none_action_fails(self, manifest):
        with pytest.raises(ActionValidationError, match="None"):
            validate_action_output(None, manifest)

    def test_wrong_shape_fails(self, manifest):
        action = [[0.0] * 7]  # [1, 7] instead of [16, 7]
        with pytest.raises(ActionValidationError, match="shape mismatch"):
            validate_action_output(action, manifest)

    def test_nan_action_fails(self, manifest):
        action = [[0.0] * 7 for _ in range(16)]
        action[3][2] = float("nan")
        with pytest.raises(ActionValidationError, match="NaN"):
            validate_action_output(action, manifest)

    def test_inf_action_fails(self, manifest):
        action = [[0.0] * 7 for _ in range(16)]
        action[5][1] = float("inf")
        with pytest.raises(ActionValidationError, match="Inf"):
            validate_action_output(action, manifest)


class TestRobotTypeValidation:
    def test_matching_robot_type_passes(self, manifest):
        validate_robot_type("so100", manifest)

    def test_mismatched_robot_type_fails(self, manifest):
        with pytest.raises(ObservationValidationError, match="Robot type mismatch"):
            validate_robot_type("so101", manifest)

    def test_none_robot_type_passes(self, manifest):
        validate_robot_type(None, manifest)  # skip check
