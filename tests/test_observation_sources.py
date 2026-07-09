# test_observation_sources.py — tests for shadow-mode observation sources.

import json
import pytest
from pathlib import Path

from lerobot_coreai.observation_sources import (
    FixtureObservationSource,
    FixtureDirectoryObservationSource,
    FolderImageObservationSource,
    CameraObservationSource,
    build_observation_source,
)
from lerobot_coreai.errors import CoreAIPolicyError, FixtureError


def _write_fixture(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data))
    return path


class TestFixtureObservationSource:
    def test_reads_single_fixture_then_eof(self, tmp_path):
        fixture = _write_fixture(tmp_path / "obs.json", {
            "observation.state": [0.0] * 7,
            "observation.images.wrist": "wrist.png",
            "task": "pick up the cube",
        })
        src = FixtureObservationSource(fixture_path=fixture, repeat=False)
        src.open()
        obs1 = src.read()
        assert obs1 is not None
        assert "observation.state" in obs1
        obs2 = src.read()
        assert obs2 is None  # EOF after one read
        src.close()

    def test_repeat_true_yields_indefinitely(self, tmp_path):
        fixture = _write_fixture(tmp_path / "obs.json", {"observation.state": [0.0]})
        src = FixtureObservationSource(fixture_path=fixture, repeat=True)
        src.open()
        for _ in range(10):
            obs = src.read()
            assert obs is not None
            assert obs["observation.state"] == [0.0]
        src.close()

    def test_missing_fixture_raises(self, tmp_path):
        src = FixtureObservationSource(fixture_path=tmp_path / "nope.json")
        with pytest.raises(FixtureError, match="not found"):
            src.open()

    def test_close_makes_read_return_none(self, tmp_path):
        fixture = _write_fixture(tmp_path / "obs.json", {"observation.state": [0.0]})
        src = FixtureObservationSource(fixture_path=fixture, repeat=True)
        src.open()
        src.close()
        assert src.read() is None


class TestFixtureDirectoryObservationSource:
    def test_reads_ordered_sequence(self, tmp_path):
        _write_fixture(tmp_path / "000000.json", {"observation.state": [0.0]})
        _write_fixture(tmp_path / "000001.json", {"observation.state": [1.0]})
        _write_fixture(tmp_path / "000002.json", {"observation.state": [2.0]})
        src = FixtureDirectoryObservationSource(fixtures_dir=tmp_path)
        src.open()
        obs0 = src.read()
        obs1 = src.read()
        obs2 = src.read()
        obs3 = src.read()
        assert obs0["observation.state"] == [0.0]
        assert obs1["observation.state"] == [1.0]
        assert obs2["observation.state"] == [2.0]
        assert obs3 is None  # EOF
        src.close()

    def test_missing_directory_raises(self, tmp_path):
        src = FixtureDirectoryObservationSource(fixtures_dir=tmp_path / "nope")
        with pytest.raises(FixtureError, match="not found"):
            src.open()

    def test_empty_directory_raises(self, tmp_path):
        src = FixtureDirectoryObservationSource(fixtures_dir=tmp_path)
        with pytest.raises(FixtureError, match="No fixture JSON"):
            src.open()


class TestFolderImageObservationSource:
    def test_emits_image_path_observation(self, tmp_path):
        # Create dummy image files.
        for name in ["000000.png", "000001.png", "000002.png"]:
            (tmp_path / name).write_bytes(b"\x89PNG fake")
        src = FolderImageObservationSource(
            frames_dir=tmp_path,
            image_key="observation.images.wrist",
            task="pick up the cube",
        )
        src.open()
        obs0 = src.read()
        assert obs0 is not None
        assert obs0["observation.images.wrist"].endswith("000000.png")
        assert obs0["task"] == "pick up the cube"
        obs1 = src.read()
        assert obs1["observation.images.wrist"].endswith("000001.png")
        src.read()
        assert src.read() is None  # EOF
        src.close()

    def test_state_vector_included(self, tmp_path):
        (tmp_path / "frame.png").write_bytes(b"\x89PNG fake")
        src = FolderImageObservationSource(
            frames_dir=tmp_path,
            state_vector=[0.1, 0.2, 0.3],
        )
        src.open()
        obs = src.read()
        assert obs["observation.state"] == [0.1, 0.2, 0.3]
        src.close()

    def test_state_json_included(self, tmp_path):
        (tmp_path / "frame.png").write_bytes(b"\x89PNG fake")
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps([1.0, 2.0]))
        src = FolderImageObservationSource(
            frames_dir=tmp_path,
            state_json=state_file,
        )
        src.open()
        obs = src.read()
        assert obs["observation.state"] == [1.0, 2.0]
        src.close()

    def test_missing_directory_raises(self, tmp_path):
        src = FolderImageObservationSource(frames_dir=tmp_path / "nope")
        with pytest.raises(CoreAIPolicyError, match="not found"):
            src.open()

    def test_empty_directory_raises(self, tmp_path):
        src = FolderImageObservationSource(frames_dir=tmp_path)
        with pytest.raises(CoreAIPolicyError, match="No image frames"):
            src.open()


class TestCameraObservationSource:
    def test_open_without_cv2_raises_install_hint(self, monkeypatch):
        """Without cv2 installed, open() should raise with install hint."""
        # Simulate cv2 not being installed, regardless of the test environment.
        import sys
        monkeypatch.setitem(sys.modules, "cv2", None)
        src = CameraObservationSource()
        with pytest.raises(CoreAIPolicyError, match="lerobot-coreai\\[camera\\]"):
            src.open()

    def test_factory_camera(self):
        src = build_observation_source("camera", camera_index=0)
        assert isinstance(src, CameraObservationSource)


class TestBuildObservationSource:
    def test_factory_fixture(self, tmp_path):
        fixture = _write_fixture(tmp_path / "obs.json", {"observation.state": [0.0]})
        src = build_observation_source("fixture", fixture=fixture)
        assert isinstance(src, FixtureObservationSource)

    def test_factory_fixtures_dir(self, tmp_path):
        _write_fixture(tmp_path / "000000.json", {"observation.state": [0.0]})
        src = build_observation_source("fixtures", fixtures_dir=tmp_path)
        assert isinstance(src, FixtureDirectoryObservationSource)

    def test_factory_folder(self, tmp_path):
        (tmp_path / "frame.png").write_bytes(b"\x89PNG")
        src = build_observation_source("folder", frames_dir=tmp_path)
        assert isinstance(src, FolderImageObservationSource)

    def test_factory_camera(self):
        src = build_observation_source("camera")
        assert isinstance(src, CameraObservationSource)

    def test_factory_fixture_missing_arg(self):
        with pytest.raises(CoreAIPolicyError, match="requires --fixture"):
            build_observation_source("fixture")

    def test_factory_folder_missing_arg(self):
        with pytest.raises(CoreAIPolicyError, match="requires --frames-dir"):
            build_observation_source("folder")

    def test_factory_unknown_type(self):
        with pytest.raises(CoreAIPolicyError, match="Unknown observation source"):
            build_observation_source("magic")
