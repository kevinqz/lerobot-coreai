# dataset.py — optional LeRobotDataset integration (v0.4).
#
# All LeRobot imports are isolated here, guarded by require_lerobot().
# The core package never imports this module at the top level.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import CoreAIPolicyError


def require_lerobot() -> None:
    """Ensure LeRobot is installed and importable.

    Raises:
        CoreAIPolicyError: If LeRobot is not installed or Python < 3.12.
    """
    import sys
    try:
        import lerobot  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        raise CoreAIPolicyError(
            "LeRobotDataset eval requires `pip install \"lerobot-coreai[lerobot]\"` "
            "and Python 3.12+.\n"
            "No robot commands were sent."
        ) from None

    if sys.version_info < (3, 12):
        raise CoreAIPolicyError(
            "LeRobotDataset eval requires Python 3.12+ "
            "(LeRobot 0.6.x requirement).\n"
            "No robot commands were sent."
        )


@dataclass
class LeRobotDatasetEvalConfig:
    """Configuration for loading a LeRobotDataset for eval."""
    dataset_repo_id: str
    root: Path | None = None
    revision: str | None = None
    episodes: list[int] | None = None
    max_frames: int | None = None
    start_index: int = 0
    stride: int = 1
    download_videos: bool = True
    video_backend: str | None = None


def load_lerobot_dataset(config: LeRobotDatasetEvalConfig) -> Any:
    """Load a LeRobotDataset using the public constructor.

    Uses ``LeRobotDataset(repo_id, root, episodes, revision, download_videos, video_backend)``
    — the canonical public API in LeRobot 0.6.x.
    """
    require_lerobot()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore[import-not-found]

    return LeRobotDataset(
        repo_id=config.dataset_repo_id,
        root=config.root,
        episodes=config.episodes,
        revision=config.revision,
        download_videos=config.download_videos,
        video_backend=config.video_backend,
    )


def dataset_item_to_observation_batch(
    item: dict[str, Any],
    manifest: Any,  # LeRobotCoreAIManifest — typed loosely to avoid circular import
) -> dict[str, Any]:
    """Convert a LeRobotDataset item to a CoreAI observation batch.

    Uses exactly the keys from ``manifest.observation_features``:
    - If the key exists in the dataset item, copy it.
    - If the key is required and missing, raise ObservationValidationError.
    - Ignore action ground truth and other dataset keys.
    """
    from .errors import ObservationValidationError

    batch: dict[str, Any] = {}
    for key, spec in manifest.observation_features.items():
        if key in item:
            batch[key] = item[key]
        elif key == "task" and "task" in item:
            batch["task"] = item["task"]
        elif spec.required:
            raise ObservationValidationError(
                f"Required observation key '{key}' not found in dataset item.\n"
                f"Available keys: {list(item.keys())}"
            )
    return batch
