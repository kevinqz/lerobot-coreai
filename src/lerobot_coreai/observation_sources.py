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


# MARK: - Camera source (stub — coming in v0.7.1)

@dataclass
class CameraObservationSource:
    """Camera capture source — experimental, coming in v0.7.1.

    open() raises immediately so that camera mode fails loudly and early rather
    than silently producing nothing.
    """

    camera_index: int | None = None
    camera_width: int | None = None
    camera_height: int | None = None
    camera_fps: float | None = None

    def open(self) -> None:
        raise CoreAIPolicyError(
            "Camera observation source is experimental and will be available in "
            "lerobot-coreai v0.7.1. Use --observation-source folder or fixtures for now."
        )

    def read(self) -> dict[str, Any] | None:
        raise CoreAIPolicyError(
            "Camera observation source is experimental and will be available in "
            "lerobot-coreai v0.7.1."
        )

    def close(self) -> None:
        pass

    @property
    def source_type(self) -> str:
        return "camera"


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
        return CameraObservationSource(camera_index=camera_index)

    raise CoreAIPolicyError(
        f"Unknown observation source: {source_type!r}. "
        f"Choose from: fixture, fixtures, folder, camera."
    )
