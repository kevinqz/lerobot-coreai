# test_dataset_adapter.py — tests for dataset_item_to_observation_batch.

import pytest
from lerobot_coreai.dataset import dataset_item_to_observation_batch
from lerobot_coreai.manifest import LeRobotCoreAIManifest
from lerobot_coreai.errors import ObservationValidationError


@pytest.fixture
def manifest(valid_manifest_dict):
    return LeRobotCoreAIManifest.from_dict(valid_manifest_dict)


class TestDatasetAdapter:
    def test_copies_manifest_observation_keys(self, manifest):
        item = {
            "observation.images.wrist": "image_data",
            "observation.state": [0.0] * 7,
            "task": "pick up the cube",
            "action": [0.0] * 7,  # should be ignored
        }
        batch = dataset_item_to_observation_batch(item, manifest)
        assert "observation.images.wrist" in batch
        assert "observation.state" in batch
        assert "task" in batch
        assert "action" not in batch  # ground truth ignored

    def test_includes_task_if_present(self, manifest):
        item = {
            "observation.images.wrist": "img",
            "observation.state": [0.0] * 7,
            "task": "do something",
        }
        batch = dataset_item_to_observation_batch(item, manifest)
        assert batch["task"] == "do something"

    def test_missing_required_observation_raises(self, manifest):
        item = {
            "observation.state": [0.0] * 7,
            # missing observation.images.wrist
        }
        with pytest.raises(ObservationValidationError, match="Required observation key"):
            dataset_item_to_observation_batch(item, manifest)

    def test_optional_task_missing_passes(self, manifest):
        item = {
            "observation.images.wrist": "img",
            "observation.state": [0.0] * 7,
            # task is optional
        }
        batch = dataset_item_to_observation_batch(item, manifest)
        assert "task" not in batch
