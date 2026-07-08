# serialization.py — convert LeRobotDataset items to JSON-safe observation batches (v0.4).
#
# LeRobotDataset items may contain torch.Tensor, numpy arrays, PIL images, etc.
# The runner expects JSON-serializable values (strings for image paths, lists for tensors).
# This module adapts without importing torch/numpy/PIL at the module level — they come
# via the [lerobot] extra and are duck-typed at runtime.

from __future__ import annotations

from pathlib import Path
from typing import Any

from .errors import ObservationValidationError


def is_json_serializable(value: Any) -> bool:
    """Check if a value can be serialized to JSON without conversion."""
    if value is None:
        return True
    if isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, (list, tuple)):
        return all(is_json_serializable(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and is_json_serializable(v) for k, v in value.items())
    return False


def _tensor_to_list(value: Any) -> Any:
    """Convert a tensor-like or array-like object to a nested list via duck typing.

    Handles torch.Tensor (has .detach().cpu().tolist()) and numpy ndarray (has .tolist()).
    """
    # torch.Tensor path: detach from graph, move to CPU, convert to list.
    if hasattr(value, "detach"):
        try:
            value = value.detach()
        except Exception:
            pass
    if hasattr(value, "cpu"):
        try:
            value = value.cpu()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if hasattr(value, "numpy"):
        try:
            np_val = value.numpy()
            if hasattr(np_val, "tolist"):
                return np_val.tolist()
        except Exception:
            pass
    return value


def _save_image_to_png(value: Any, path: Path) -> str:
    """Save an image-like object (tensor/array/PIL) to a PNG file. Returns the path string."""
    # Convert tensor → numpy first.
    if hasattr(value, "detach") or hasattr(value, "cpu"):
        value = _tensor_to_list(value)

    # Try numpy array → PIL Image.
    try:
        import numpy as np  # type: ignore[import-not-found]
        from PIL import Image  # type: ignore[import-not-found]

        if not isinstance(value, np.ndarray):
            value = np.array(value)

        # Handle [C, H, W] → [H, W, C] for PIL.
        if value.ndim == 3 and value.shape[0] in (1, 3, 4):
            value = np.transpose(value, (1, 2, 0))
        # Clamp to uint8 if float.
        if value.dtype != np.uint8:
            value = np.clip(value * 255, 0, 255).astype(np.uint8)

        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(value).save(str(path))
        return str(path)
    except ImportError:
        raise ObservationValidationError(
            f"Cannot save image to {path}: PIL and numpy are required. "
            f"Install with: pip install \"lerobot-coreai[lerobot]\"."
        )
    except Exception as e:
        raise ObservationValidationError(
            f"Failed to save image observation to {path}: {e}"
        ) from e


def make_json_safe_observation(
    batch: dict[str, Any],
    *,
    output_dir: Path | None = None,
    frame_index: int | None = None,
) -> dict[str, Any]:
    """Convert an observation batch to JSON-safe values.

    Rules:
    - str, int, float, bool, None: preserved
    - list/tuple: recursively converted
    - Objects with .detach()/.cpu()/.tolist() (torch.Tensor): converted to nested list
    - Objects with .tolist() (numpy): converted to nested list
    - observation.images.* keys:
        - If already a string (path): preserved
        - If tensor/array/PIL: saved as PNG to output_dir/frames/
        - If can't save: raises ObservationValidationError

    Args:
        batch: The raw observation batch from dataset_item_to_observation_batch.
        output_dir: Directory for saving image frames (required for image serialization).
        frame_index: Frame index for naming saved images.

    Returns:
        A JSON-safe observation batch.
    """
    result: dict[str, Any] = {}

    for key, value in batch.items():
        # Already JSON-safe (strings, lists, scalars).
        if is_json_serializable(value):
            result[key] = value
            continue

        # Image observation keys.
        if key.startswith("observation.images."):
            if isinstance(value, str):
                result[key] = value  # already a path
                continue

            # Need to save as PNG.
            if output_dir is None or frame_index is None:
                raise ObservationValidationError(
                    f"Image observation key '{key}' is not JSON serializable "
                    f"(got {type(value).__name__}). Provide image path or enable image serialization."
                )

            safe_key = key.replace(".", "_")
            img_path = output_dir / "frames" / f"frame_{frame_index:06d}_{safe_key}.png"
            result[key] = _save_image_to_png(value, img_path)
            continue

        # Non-image: try tensor → list conversion.
        converted = _tensor_to_list(value)
        if is_json_serializable(converted):
            result[key] = converted
            continue

        # Last resort: not serializable.
        raise ObservationValidationError(
            f"Observation key '{key}' has non-serializable value of type {type(value).__name__}. "
            f"Unable to convert to JSON-safe format."
        )

    return result
