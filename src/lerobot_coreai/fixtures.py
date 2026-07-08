# fixtures.py — observation fixture loading for dry-run rollout (v0.3).

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import FixtureError


def load_observation_fixture(path: str | Path) -> dict[str, Any]:
    """Load an observation fixture from a JSON file.

    Supports two formats:

    **Flat fixture** (simple, common)::

        {
            "observation.images.wrist": "assets/wrist.png",
            "observation.state": [0.0, 0.1, ...],
            "task": "pick up the cube"
        }

    **Typed fixture** (explicit kinds)::

        {
            "observation": {
                "observation.images.wrist": {"kind": "image", "path": "assets/wrist.png"},
                "observation.state": {"kind": "tensor", "value": [0.0, ...]},
                "task": {"kind": "text", "value": "pick up the cube"}
            }
        }

    Image paths are resolved relative to the fixture file's directory.

    Args:
        path: Path to the fixture JSON file.

    Returns:
        A flat observation batch dict.

    Raises:
        FixtureError: If the file is missing, not JSON, or malformed.
    """
    p = Path(path)

    if not p.is_file():
        raise FixtureError(f"Observation fixture not found: {p}")

    if p.suffix != ".json":
        raise FixtureError(f"Observation fixture must be a .json file, got: {p}")

    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise FixtureError(f"Invalid JSON in fixture {p}: {e}") from e

    if not isinstance(data, dict):
        raise FixtureError(f"Fixture must be a JSON object, got {type(data).__name__}")

    fixture_dir = p.parent

    # Typed fixture: unwrap the "observation" key.
    if "observation" in data and isinstance(data["observation"], dict):
        return _resolve_typed_fixture(data["observation"], fixture_dir)

    # Flat fixture: resolve image paths relative to the fixture directory.
    return _resolve_flat_fixture(data, fixture_dir)


def _resolve_flat_fixture(data: dict[str, Any], fixture_dir: Path) -> dict[str, Any]:
    """Resolve image paths in a flat fixture relative to the fixture directory."""
    result = {}
    for key, value in data.items():
        if key.startswith("observation.images.") and isinstance(value, str):
            # Resolve relative image paths.
            result[key] = str((fixture_dir / value).resolve())
        else:
            result[key] = value
    return result


def _resolve_typed_fixture(obs: dict[str, Any], fixture_dir: Path) -> dict[str, Any]:
    """Unwrap typed fixture entries and resolve image paths."""
    result = {}
    for key, entry in obs.items():
        if not isinstance(entry, dict):
            result[key] = entry
            continue
        kind = entry.get("kind", "")
        if kind == "image":
            path_val = entry.get("path", entry.get("value", ""))
            result[key] = str((fixture_dir / path_val).resolve())
        elif kind == "tensor":
            result[key] = entry.get("value")
        elif kind == "text":
            result[key] = entry.get("value")
        else:
            # Pass through value if present
            result[key] = entry.get("value", entry)
    return result
