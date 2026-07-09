# test_observation_adapters.py — tests for observation adapter layer (v0.7.2).

import json
import pytest
from pathlib import Path

from lerobot_coreai.observation_adapters import (
    ObservationAdapterConfig,
    AdaptedObservation,
    adapt_observation,
)
from lerobot_coreai.errors import CoreAIPolicyError


class TestAdaptObservation:
    def test_passthrough_preserves_keys(self):
        raw = {"observation.images.wrist": "wrist.png", "observation.state": [0.0] * 7}
        config = ObservationAdapterConfig()
        result = adapt_observation(raw, config)
        assert "observation.images.wrist" in result.observation
        assert "observation.state" in result.observation

    def test_injects_task(self):
        raw = {"observation.state": [0.0]}
        config = ObservationAdapterConfig(task="pick up the cube")
        result = adapt_observation(raw, config)
        assert result.observation["task"] == "pick up the cube"

    def test_injects_state_vector(self):
        raw = {"observation.images.wrist": "wrist.png"}
        config = ObservationAdapterConfig(state_vector=[0.1, 0.2, 0.3])
        result = adapt_observation(raw, config)
        assert result.observation["observation.state"] == [0.1, 0.2, 0.3]

    def test_injects_state_json(self, tmp_path):
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps([1.0, 2.0, 3.0]))
        raw = {"observation.images.wrist": "wrist.png"}
        config = ObservationAdapterConfig(state_json=state_file)
        result = adapt_observation(raw, config)
        assert result.observation["observation.state"] == [1.0, 2.0, 3.0]

    def test_state_json_invalid_raises(self, tmp_path):
        state_file = tmp_path / "state.json"
        state_file.write_text("not json")
        raw = {}
        config = ObservationAdapterConfig(state_json=state_file)
        with pytest.raises(CoreAIPolicyError, match="Failed to load state JSON"):
            adapt_observation(raw, config)

    def test_state_vector_non_numeric_raises(self):
        raw = {}
        config = ObservationAdapterConfig(state_vector=[0.1, "bad", 0.3])
        with pytest.raises(CoreAIPolicyError, match="non-numeric"):
            adapt_observation(raw, config)

    def test_missing_required_key_raises(self):
        raw = {"observation.images.wrist": "wrist.png"}
        config = ObservationAdapterConfig(required_keys=["observation.state"])
        with pytest.raises(CoreAIPolicyError, match="Required observation keys missing"):
            adapt_observation(raw, config)

    def test_image_key_alias_mapping(self):
        raw = {"camera_front": "/path/front.png"}
        config = ObservationAdapterConfig(
            image_keys={"camera_front": "observation.images.front"}
        )
        result = adapt_observation(raw, config)
        assert "observation.images.front" in result.observation
        assert "camera_front" not in result.observation

    def test_drop_unknown_keys_with_manifest(self, valid_manifest_dict):
        from lerobot_coreai.manifest import LeRobotCoreAIManifest
        manifest = LeRobotCoreAIManifest.from_dict(valid_manifest_dict)
        raw = {
            "observation.images.wrist": "wrist.png",
            "observation.state": [0.0] * 7,
            "unknown_key": "should be dropped",
        }
        config = ObservationAdapterConfig(drop_unknown_keys=True)
        result = adapt_observation(raw, config, manifest=manifest)
        assert "unknown_key" not in result.observation
        assert any("unknown_key" in w for w in result.warnings)

    def test_require_task_passes_when_present(self):
        raw = {"task": "do something"}
        config = ObservationAdapterConfig(require_task=True)
        result = adapt_observation(raw, config)
        assert result.observation["task"] == "do something"

    def test_require_task_fails_when_absent(self):
        raw = {}
        config = ObservationAdapterConfig(require_task=True)
        with pytest.raises(CoreAIPolicyError, match="Required observation keys missing"):
            adapt_observation(raw, config)
