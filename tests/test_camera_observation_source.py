# test_camera_observation_source.py — tests for CameraObservationSource with mocked cv2.

import sys
import pytest
from pathlib import Path

from lerobot_coreai.observation_sources import CameraObservationSource, build_observation_source
from lerobot_coreai.errors import CoreAIPolicyError


# MARK: - Fake cv2 machinery

class FakeFrame:
    """A fake numpy-like frame (cv2 returns numpy arrays)."""
    pass


class FakeCapture:
    """Fake cv2.VideoCapture."""
    def __init__(self, index):
        self.index = index
        self.opened = True
        self.released = False
        self.props_set = {}

    def isOpened(self):
        return self.opened

    def read(self):
        return True, FakeFrame()

    def set(self, prop, value):
        self.props_set[prop] = value
        return True

    def release(self):
        self.released = True


class FakeCV2:
    """Minimal fake cv2 module."""
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_FPS = 5

    def __init__(self):
        self.captures = []
        self.frames_written = []

    def VideoCapture(self, index):
        cap = FakeCapture(index)
        self.captures.append(cap)
        return cap

    def imwrite(self, path, frame):
        # Create a dummy file so the path exists.
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"\x89PNG fake camera frame")
        self.frames_written.append(path)
        return True


@pytest.fixture
def fake_cv2(monkeypatch):
    """Inject a fake cv2 module into sys.modules."""
    cv2 = FakeCV2()
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    return cv2


# MARK: - Tests

class TestCameraSourceMissingCV2:
    def test_open_without_cv2_raises(self, monkeypatch):
        # Simulate cv2 not being installed, regardless of the test environment
        # (LeRobot may pull in opencv as a transitive dependency).
        monkeypatch.setitem(sys.modules, "cv2", None)
        src = CameraObservationSource()
        with pytest.raises(CoreAIPolicyError, match="lerobot-coreai\\[camera\\]"):
            src.open()

    def test_error_message_has_install_hint(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "cv2", None)
        src = CameraObservationSource()
        try:
            src.open()
        except CoreAIPolicyError as e:
            assert "pip install" in str(e)
            assert "camera" in str(e)


class TestCameraSourceOpen:
    def test_open_success(self, tmp_path, fake_cv2):
        src = CameraObservationSource(camera_index=0, output_dir=tmp_path)
        src.open()
        assert len(fake_cv2.captures) == 1
        assert fake_cv2.captures[0].index == 0
        assert src._cap is not None

    def test_open_failure_raises(self, tmp_path, monkeypatch):
        """If VideoCapture can't open the device, raise CoreAIPolicyError."""
        class BrokenCapture(FakeCapture):
            def isOpened(self):
                return False

        fake = FakeCV2()
        fake.VideoCapture = lambda idx: BrokenCapture(idx)
        monkeypatch.setitem(sys.modules, "cv2", fake)

        src = CameraObservationSource(camera_index=99, output_dir=tmp_path)
        with pytest.raises(CoreAIPolicyError, match="Could not open camera"):
            src.open()

    def test_open_applies_width_height_fps(self, tmp_path, fake_cv2):
        src = CameraObservationSource(
            camera_index=0,
            output_dir=tmp_path,
            width=1280,
            height=720,
            camera_fps=30.0,
        )
        src.open()
        cap = fake_cv2.captures[0]
        assert cap.props_set[FakeCV2.CAP_PROP_FRAME_WIDTH] == 1280
        assert cap.props_set[FakeCV2.CAP_PROP_FRAME_HEIGHT] == 720
        assert cap.props_set[FakeCV2.CAP_PROP_FPS] == 30.0

    def test_open_creates_frames_dir(self, tmp_path, fake_cv2):
        src = CameraObservationSource(camera_index=0, output_dir=tmp_path)
        src.open()
        assert (tmp_path / "frames").is_dir()


class TestCameraSourceRead:
    def test_read_returns_observation_with_image_path(self, tmp_path, fake_cv2):
        src = CameraObservationSource(
            camera_index=0,
            output_dir=tmp_path,
            image_key="observation.images.wrist",
        )
        src.open()
        obs = src.read()
        assert obs is not None
        assert "observation.images.wrist" in obs
        assert obs["observation.images.wrist"].endswith("step_000000.png")

    def test_read_includes_task(self, tmp_path, fake_cv2):
        src = CameraObservationSource(
            camera_index=0, output_dir=tmp_path, task="pick up the cube",
        )
        src.open()
        obs = src.read()
        assert obs["task"] == "pick up the cube"

    def test_read_includes_state_vector(self, tmp_path, fake_cv2):
        src = CameraObservationSource(
            camera_index=0, output_dir=tmp_path, state_vector=[0.0, 0.1, 0.2],
        )
        src.open()
        obs = src.read()
        assert obs["observation.state"] == [0.0, 0.1, 0.2]

    def test_read_increments_frame_index(self, tmp_path, fake_cv2):
        src = CameraObservationSource(camera_index=0, output_dir=tmp_path)
        src.open()
        obs1 = src.read()
        obs2 = src.read()
        assert obs1["observation.images.wrist"].endswith("step_000000.png")
        assert obs2["observation.images.wrist"].endswith("step_000001.png")

    def test_read_saves_frame_to_disk(self, tmp_path, fake_cv2):
        src = CameraObservationSource(camera_index=0, output_dir=tmp_path)
        src.open()
        src.read()
        assert len(fake_cv2.frames_written) == 1
        assert Path(fake_cv2.frames_written[0]).exists()

    def test_read_failure_raises(self, tmp_path, monkeypatch):
        """If cap.read() fails, raise CoreAIPolicyError."""
        class FailingCapture(FakeCapture):
            def read(self):
                return False, None

        fake = FakeCV2()
        fake.VideoCapture = lambda idx: FailingCapture(idx)
        monkeypatch.setitem(sys.modules, "cv2", fake)

        src = CameraObservationSource(camera_index=0, output_dir=tmp_path)
        src.open()
        with pytest.raises(CoreAIPolicyError, match="Failed to read frame"):
            src.read()

    def test_read_after_close_returns_none(self, tmp_path, fake_cv2):
        src = CameraObservationSource(camera_index=0, output_dir=tmp_path)
        src.open()
        src.close()
        assert src.read() is None


class TestCameraSourceAlwaysSavesFrames:
    """Camera frames are always saved — persistence is mandatory for auditability."""

    def test_frames_always_saved(self, tmp_path, fake_cv2):
        """Even without any save flag, frames must be written to disk."""
        src = CameraObservationSource(camera_index=0, output_dir=tmp_path)
        src.open()
        src.read()
        src.read()
        assert len(fake_cv2.frames_written) == 2
        for path in fake_cv2.frames_written:
            assert Path(path).exists()

    def test_save_without_output_dir_raises(self, fake_cv2):
        """Without output_dir, saving cannot proceed — frames are mandatory."""
        src = CameraObservationSource(camera_index=0, output_dir=None)
        src.open()
        with pytest.raises(CoreAIPolicyError, match="output_dir"):
            src.read()


class TestCameraSourceClose:
    def test_close_releases_capture(self, tmp_path, fake_cv2):
        src = CameraObservationSource(camera_index=0, output_dir=tmp_path)
        src.open()
        cap = fake_cv2.captures[0]
        assert not cap.released
        src.close()
        assert cap.released

    def test_close_idempotent(self, tmp_path, fake_cv2):
        src = CameraObservationSource(camera_index=0, output_dir=tmp_path)
        src.open()
        src.close()
        src.close()  # should not raise


class TestCameraSourceNoActuation:
    def test_no_actuation_fields_in_source(self, tmp_path, fake_cv2):
        """Camera source must not expose any actuation-related fields."""
        src = CameraObservationSource(camera_index=0, output_dir=tmp_path)
        src.open()
        obs = src.read()
        # Observation should only have image_key, optionally state/task.
        for key in obs:
            assert "action" not in key.lower()
            assert "motor" not in key.lower()
            assert "command" not in key.lower()


class TestCameraSourceFactory:
    def test_factory_passes_camera_params(self, tmp_path):
        src = build_observation_source(
            "camera",
            camera_index=2,
            camera_width=640,
            camera_height=480,
            camera_fps=15.0,
            output_dir=tmp_path,
            task="test task",
            state_vector=[1.0, 2.0],
        )
        assert isinstance(src, CameraObservationSource)
        assert src.camera_index == 2
        assert src.width == 640
        assert src.height == 480
        assert src.camera_fps == 15.0
        assert src.task == "test task"
        assert src.state_vector == [1.0, 2.0]
