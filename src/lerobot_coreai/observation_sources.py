# observation_sources.py — observation inputs for shadow mode (v0.7).
#
# Sources read observations (fixture / fixture directory / image folder) and feed
# the shadow loop. Camera is a stub until v0.7.1. No source may send commands to a
# robot, motor, or actuator — sources are observation inputs only.

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .errors import CoreAIPolicyError, FixtureError
from .fixtures import load_observation_fixture


@runtime_checkable
class ObservationSource(Protocol):
    """A stream of observations for the shadow loop.

    Lifecycle: open() → read() until None (EOF) → close().
    read() returns None when the source is exhausted.
    """

    def open(self) -> None: ...

    def read(self) -> dict[str, Any] | None: ...

    def close(self) -> None: ...


@dataclass
class ObservationFrame:
    """A single observation record with provenance."""

    index: int
    timestamp: str
    observation: dict[str, Any]
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


# MARK: - Fixture source (single fixture, optionally repeated)

@dataclass
class FixtureObservationSource:
    """Reads a single observation fixture.

    With repeat=False (default), yields one observation then EOF.
    With repeat=True, yields the same observation indefinitely (until the caller
    stops via max_steps / duration).
    """

    fixture_path: Path
    repeat: bool = False
    _observation: dict[str, Any] | None = None
    _read_once: bool = False
    _closed: bool = False

    def open(self) -> None:
        if not self.fixture_path.is_file():
            raise FixtureError(f"Observation fixture not found: {self.fixture_path}")
        self._observation = load_observation_fixture(self.fixture_path)
        self._read_once = False

    def read(self) -> dict[str, Any] | None:
        if self._closed or self._observation is None:
            return None
        if self._read_once and not self.repeat:
            return None
        self._read_once = True
        # Return a shallow copy so callers can mutate without affecting repeats.
        return dict(self._observation)

    def close(self) -> None:
        self._closed = True
        self._observation = None

    @property
    def source_type(self) -> str:
        return "fixture"


# MARK: - Fixture directory source (ordered sequence)

@dataclass
class FixtureDirectoryObservationSource:
    """Reads an ordered sequence of fixture JSON files from a directory.

    Files are matched as NNNNNN.json (zero-padded) if present, else all *.json in
    lexicographic order. EOF when the sequence is exhausted.
    """

    fixtures_dir: Path
    _files: list[Path] = field(default_factory=list)
    _index: int = 0
    _closed: bool = False

    def open(self) -> None:
        if not self.fixtures_dir.is_dir():
            raise FixtureError(f"Fixtures directory not found: {self.fixtures_dir}")
        all_json = sorted(self.fixtures_dir.glob("*.json"))
        if not all_json:
            raise FixtureError(f"No fixture JSON files found in: {self.fixtures_dir}")
        self._files = all_json
        self._index = 0

    def read(self) -> dict[str, Any] | None:
        if self._closed or self._index >= len(self._files):
            return None
        obs = load_observation_fixture(self._files[self._index])
        self._index += 1
        return obs

    def close(self) -> None:
        self._closed = True
        self._files = []

    @property
    def source_type(self) -> str:
        return "fixtures"


# MARK: - Folder image source (images on disk → observation)

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass
class FolderImageObservationSource:
    """Reads images from a folder and builds an observation batch per image.

    The observation is:
        {<image_key>: <abs image path>, ["observation.state": [...]], ["task": "..."]}

    State comes from state_json (a JSON array file) or state_vector (list of floats).
    """

    frames_dir: Path
    image_key: str = "observation.images.wrist"
    state_json: Path | None = None
    state_vector: list[float] | None = None
    task: str | None = None
    _images: list[Path] = field(default_factory=list)
    _index: int = 0
    _closed: bool = False

    def open(self) -> None:
        if not self.frames_dir.is_dir():
            raise CoreAIPolicyError(f"Frames directory not found: {self.frames_dir}")
        images = sorted(
            p for p in self.frames_dir.iterdir()
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
        )
        if not images:
            raise CoreAIPolicyError(f"No image frames found in: {self.frames_dir}")
        self._images = images
        self._index = 0

    def read(self) -> dict[str, Any] | None:
        if self._closed or self._index >= len(self._images):
            return None
        img_path = str(self._images[self._index].resolve())
        self._index += 1
        obs: dict[str, Any] = {self.image_key: img_path}
        state = self._load_state()
        if state is not None:
            obs["observation.state"] = state
        if self.task is not None:
            obs["task"] = self.task
        return obs

    def close(self) -> None:
        self._closed = True
        self._images = []

    @property
    def source_type(self) -> str:
        return "folder"

    def _load_state(self) -> list[float] | None:
        if self.state_vector is not None:
            return list(self.state_vector)
        if self.state_json is not None:
            try:
                data = json.loads(Path(self.state_json).read_text())
            except (OSError, json.JSONDecodeError) as e:
                raise CoreAIPolicyError(
                    f"Failed to load state JSON from {self.state_json}: {e}"
                ) from e
            if not isinstance(data, list):
                raise CoreAIPolicyError(
                    f"State JSON must be an array of numbers, got {type(data).__name__}"
                )
            return [float(v) for v in data]
        return None


# MARK: - Camera source (experimental — v0.7.1)

def _require_cv2():
    """Import cv2 lazily; raise a clear error if not installed."""
    try:
        import cv2
    except ImportError as e:
        raise CoreAIPolicyError(
            "Camera source requires OpenCV. Install with "
            '`pip install "lerobot-coreai[camera]"`.'
        ) from e
    return cv2


@dataclass
class CameraObservationSource:
    """Local RGB camera observation source for shadow mode (experimental).

    Opens a cv2.VideoCapture, captures frames, saves them to disk, and returns
    an observation with the image key pointing to the saved frame path.

    Frames are always saved to ``output_dir/frames/step_NNNNNN.png``. The runner
    receives frame paths (not raw arrays), and saved frames are part of the
    shadow-mode audit trail. Disabling frame persistence is not supported.

    This is observation-only. It does not connect to a robot or actuator.
    cv2 is imported lazily at open() time, so the core package works without it.
    """

    camera_index: int = 0
    image_key: str = "observation.images.wrist"
    output_dir: Path | None = None
    width: int | None = None
    height: int | None = None
    camera_fps: float | None = None
    task: str | None = None
    state_vector: list[float] | None = None
    _cv2: Any = None
    _cap: Any = None
    _frame_index: int = 0
    _closed: bool = False

    def open(self) -> None:
        cv2 = _require_cv2()
        self._cv2 = cv2
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap or not self._cap.isOpened():
            raise CoreAIPolicyError(
                f"Could not open camera index {self.camera_index}. "
                f"Check that the device is connected and not in use."
            )
        if self.width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.camera_fps:
            self._cap.set(cv2.CAP_PROP_FPS, self.camera_fps)
        if self.output_dir is not None:
            (Path(self.output_dir) / "frames").mkdir(parents=True, exist_ok=True)
        self._frame_index = 0

    def read(self) -> dict[str, Any] | None:
        if self._cap is None or self._closed:
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise CoreAIPolicyError(
                f"Failed to read frame from camera index {self.camera_index}."
            )
        frame_path = self._save_frame(frame)
        obs: dict[str, Any] = {self.image_key: str(frame_path)}
        if self.state_vector is not None:
            obs["observation.state"] = list(self.state_vector)
        if self.task is not None:
            obs["task"] = self.task
        self._frame_index += 1
        return obs

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._closed = True

    @property
    def source_type(self) -> str:
        return "camera"

    def _save_frame(self, frame: Any) -> Path:
        """Save a frame to disk. Returns the path.

        Frames are always saved — the runner observation uses the file path, and
        saved frames are part of the shadow-mode audit trail.
        """
        if self.output_dir is None:
            raise CoreAIPolicyError(
                "CameraObservationSource requires output_dir to save frames. "
                "Camera frame persistence is mandatory in shadow mode."
            )
        path = Path(self.output_dir) / "frames" / f"step_{self._frame_index:06d}.png"

        assert self._cv2 is not None  # open() guarantees cv2 is loaded
        success = self._cv2.imwrite(str(path), frame)
        if not success:
            raise CoreAIPolicyError(
                f"Failed to save camera frame to {path}."
            )
        return path


# MARK: - Factory

def build_observation_source(
    source_type: str,
    *,
    fixture: Path | None = None,
    fixtures_dir: Path | None = None,
    frames_dir: Path | None = None,
    image_key: str = "observation.images.wrist",
    state_json: Path | None = None,
    state_vector: list[float] | None = None,
    task: str | None = None,
    camera_index: int | None = None,
    camera_width: int | None = None,
    camera_height: int | None = None,
    camera_fps: float | None = None,
    output_dir: Path | None = None,
    repeat_fixture: bool = True,
) -> ObservationSource:
    """Dispatch to the right observation source by type name.

    Args:
        source_type: One of fixture, fixtures, folder, camera.
        repeat_fixture: For fixture sources, whether to repeat the single observation.

    Raises:
        CoreAIPolicyError: If required args are missing or source_type is unknown.
    """
    if source_type in ("fixture", "fixtures"):
        if source_type == "fixture":
            if fixture is None:
                raise CoreAIPolicyError(
                    "--observation-source fixture requires --fixture <path>."
                )
            return FixtureObservationSource(fixture_path=fixture, repeat=repeat_fixture)
        # fixtures (directory)
        if fixtures_dir is None:
            raise CoreAIPolicyError(
                "--observation-source fixtures requires --fixtures-dir <path>."
            )
        return FixtureDirectoryObservationSource(fixtures_dir=fixtures_dir)

    if source_type == "folder":
        if frames_dir is None:
            raise CoreAIPolicyError(
                "--observation-source folder requires --frames-dir <path>."
            )
        return FolderImageObservationSource(
            frames_dir=frames_dir,
            image_key=image_key,
            state_json=state_json,
            state_vector=state_vector,
            task=task,
        )

    if source_type == "camera":
        return CameraObservationSource(
            camera_index=camera_index if camera_index is not None else 0,
            image_key=image_key,
            output_dir=output_dir,
            width=camera_width,
            height=camera_height,
            camera_fps=camera_fps,
            task=task,
            state_vector=state_vector,
        )

    raise CoreAIPolicyError(
        f"Unknown observation source: {source_type!r}. "
        f"Choose from: fixture, fixtures, folder, camera."
    )
