# test_fixtures.py — tests for observation fixture loading.

import json
import pytest
from pathlib import Path

from lerobot_coreai.fixtures import load_observation_fixture
from lerobot_coreai.errors import FixtureError


class TestFlatFixture:
    def test_flat_fixture_loads(self, tmp_path):
        f = tmp_path / "obs.json"
        f.write_text(json.dumps({
            "observation.images.wrist": "wrist.png",
            "observation.state": [0.0] * 7,
            "task": "pick up the cube",
        }))
        batch = load_observation_fixture(f)
        assert "observation.state" in batch
        assert batch["task"] == "pick up the cube"
        # Image path should be resolved relative to fixture dir
        assert batch["observation.images.wrist"].endswith("wrist.png")

    def test_flat_fixture_resolves_image_path(self, tmp_path):
        f = tmp_path / "obs.json"
        f.write_text(json.dumps({"observation.images.wrist": "images/wrist.png"}))
        batch = load_observation_fixture(f)
        assert batch["observation.images.wrist"] == str((tmp_path / "images/wrist.png").resolve())


class TestTypedFixture:
    def test_typed_fixture_loads(self, tmp_path):
        f = tmp_path / "obs.json"
        f.write_text(json.dumps({
            "observation": {
                "observation.images.wrist": {"kind": "image", "path": "wrist.png"},
                "observation.state": {"kind": "tensor", "value": [0.0] * 7},
                "task": {"kind": "text", "value": "pick up the cube"},
            }
        }))
        batch = load_observation_fixture(f)
        assert batch["observation.state"] == [0.0] * 7
        assert batch["task"] == "pick up the cube"
        assert batch["observation.images.wrist"].endswith("wrist.png")


class TestFixtureErrors:
    def test_missing_file_raises(self):
        with pytest.raises(FixtureError, match="not found"):
            load_observation_fixture("/nonexistent/obs.json")

    def test_invalid_json_raises(self, tmp_path):
        f = tmp_path / "obs.json"
        f.write_text("{invalid json")
        with pytest.raises(FixtureError, match="Invalid JSON"):
            load_observation_fixture(f)

    def test_non_json_extension_raises(self, tmp_path):
        f = tmp_path / "obs.txt"
        f.write_text("{}")
        with pytest.raises(FixtureError, match="must be a .json"):
            load_observation_fixture(f)
