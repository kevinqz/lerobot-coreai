# test_serialization.py — tests for make_json_safe_observation (no real torch/numpy needed).

import pytest
from pathlib import Path

from lerobot_coreai.serialization import make_json_safe_observation, is_json_serializable
from lerobot_coreai.errors import ObservationValidationError


class FakeTensor:
    """Simulates a torch.Tensor with detach/cpu/tolist."""
    def __init__(self, data):
        self._data = data
    def detach(self):
        return self
    def cpu(self):
        return self
    def tolist(self):
        return self._data


class FakeNumpy:
    """Simulates a numpy ndarray with tolist."""
    def __init__(self, data):
        self._data = data
    def tolist(self):
        return self._data


class TestIsJsonSerializable:
    def test_primitives(self):
        assert is_json_serializable(42)
        assert is_json_serializable("hello")
        assert is_json_serializable(3.14)
        assert is_json_serializable(True)
        assert is_json_serializable(None)

    def test_lists(self):
        assert is_json_serializable([1, 2, 3])
        assert is_json_serializable([[0.0] * 7])

    def test_non_serializable(self):
        assert not is_json_serializable(FakeTensor([1, 2, 3]))
        assert not is_json_serializable(object())


class TestMakeJsonSafe:
    def test_list_preserved(self):
        batch = {"observation.state": [0.0, 0.1, 0.2]}
        result = make_json_safe_observation(batch)
        assert result["observation.state"] == [0.0, 0.1, 0.2]

    def test_scalar_preserved(self):
        batch = {"task": "pick up the cube"}
        result = make_json_safe_observation(batch)
        assert result["task"] == "pick up the cube"

    def test_fake_tensor_converted(self):
        """FakeTensor with detach/cpu/tolist should be converted to list."""
        batch = {"observation.state": FakeTensor([0.0, 0.1, 0.2])}
        result = make_json_safe_observation(batch)
        assert result["observation.state"] == [0.0, 0.1, 0.2]

    def test_nested_fake_tensor(self):
        batch = {"observation.state": FakeTensor([[0.0] * 7] * 16)}
        result = make_json_safe_observation(batch)
        assert isinstance(result["observation.state"], list)
        assert len(result["observation.state"]) == 16

    def test_fake_numpy_converted(self):
        batch = {"observation.state": FakeNumpy([1.0, 2.0])}
        result = make_json_safe_observation(batch)
        assert result["observation.state"] == [1.0, 2.0]

    def test_image_path_string_preserved(self):
        batch = {"observation.images.wrist": "/tmp/wrist.png"}
        result = make_json_safe_observation(batch)
        assert result["observation.images.wrist"] == "/tmp/wrist.png"

    def test_non_serializable_non_image_raises(self):
        """Non-serializable, non-tensor, non-image object should raise."""
        batch = {"observation.state": object()}
        with pytest.raises(ObservationValidationError, match="non-serializable"):
            make_json_safe_observation(batch)

    def test_image_tensor_without_output_raises(self):
        """Image tensor without output_dir should raise clear error."""
        batch = {"observation.images.wrist": FakeTensor([[[0]*3]*224]*224)}
        with pytest.raises(ObservationValidationError, match="not JSON serializable"):
            make_json_safe_observation(batch)
